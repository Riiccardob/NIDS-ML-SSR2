"""
================================================================================
NIDS-ML - Hyperparameter Tuning
================================================================================

Ricerca parametri ottimali per i modelli usando Random Search o Bayesian (Optuna) con F2-Score (Recall pesata doppio).

METRICA: F2-Score
Beta=2 enfatizza Recall (critico per sicurezza) rispetto a Precision.
Formula: F2 = (1 + 2²) * (precision * recall) / (2² * precision + recall)

USAGE:
------
    python src/hyperparameter_tuning.py --model <model> --method <method> [options]

PARAMETRI:
----------
    --model STR           Modello da ottimizzare: random_forest, xgboost, lightgbm
    --method STR          Metodo ricerca: random, bayesian (default: random)
    --n-iter INT          Iterazioni Random Search (default: 50)
    --n-trials INT        Trial Bayesian Optuna (default: 100)
    --cv INT              Fold cross-validation (default: 5)
    --task STR            binary o multiclass (default: binary)
    --timeout INT         Timeout in secondi (0 = no limit, default: 0)
    --n-jobs INT          CPU cores (default: auto)

ESEMPI:
-------
# Random Search
python src/hyperparameter_tuning.py --model random_forest --method random --n-iter 50

# Bayesian Optimization
python src/hyperparameter_tuning.py --model xgboost --method bayesian --n-trials 100

# Con timeout 2h
python src/hyperparameter_tuning.py --model lightgbm --method bayesian --timeout 7200

OUTPUT:
-------
Salva risultati in: tuning_results/<model>_best.json

================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import json
import time

def _get_arg(name, default=None):
    for i, arg in enumerate(sys.argv):
        if arg == f'--{name}' and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return sys.argv[i + 1]
    return default

_n_jobs_arg = _get_arg('n-jobs')
_n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

os.environ['OMP_NUM_THREADS'] = str(_n_cores)
os.environ['MKL_NUM_THREADS'] = str(_n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(_n_cores)
os.environ['LOKY_MAX_CPU_COUNT'] = str(_n_cores)

import psutil
try:
    p = psutil.Process()
    p.cpu_affinity(list(range(_n_cores)))
    p.nice(10)
except Exception:
    pass

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils import (
    get_logger, get_project_root, RANDOM_STATE,
    apply_cpu_limits, suppress_warnings
)
from src.preprocessing import load_processed_data
from src.feature_engineering import (
    load_artifacts, get_feature_columns, prepare_xy,
    transform_data, apply_feature_selection
)

suppress_warnings()


PARAM_DISTRIBUTIONS = {
    'random_forest': {
        'n_estimators': [100, 150, 200],
        'max_depth': [15, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    },
    'xgboost': {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    },
    'lightgbm': {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 10, 15, 20, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70, 100],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [10, 20, 30, 50],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0, 0.01, 0.1]
    }
}


def f2_scorer(y_true, y_pred):
    """F2-Score: Recall pesa doppio rispetto a Precision."""
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)


def tune_random_search(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    cv: int,
    n_jobs: int,
    task: str,
    logger
) -> Dict[str, Any]:
    """Hyperparameter tuning con Random Search e F2-Score."""
    
    logger.info(f"Random Search: {n_iter} iterations, cv={cv}")
    logger.info("Metrica: F2-Score (beta=2, Recall pesata doppio)")
    
    param_dist = PARAM_DISTRIBUTIONS[model_type]
    scoring = make_scorer(f2_scorer) if task == 'binary' else 'f1_weighted'
    
    if model_type == 'random_forest':
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs)
    elif model_type == 'xgboost':
        objective = 'binary:logistic' if task == 'binary' else 'multi:softmax'
        base_model = XGBClassifier(
            objective=objective,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            tree_method='hist'
        )
        if task == 'multiclass':
            base_model.set_params(num_class=len(y_train.unique()))
    elif model_type == 'lightgbm':
        objective = 'binary' if task == 'binary' else 'multiclass'
        base_model = LGBMClassifier(
            objective=objective,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=-1,
            force_col_wise=True
        )
        if task == 'multiclass':
            base_model.set_params(num_class=len(y_train.unique()))
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        return_train_score=False
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    results = {
        'method': 'random_search',
        'scoring_metric': 'f2_score',
        'best_params': search.best_params_,
        'best_score': float(search.best_score_),
        'n_iterations': n_iter,
        'cv_folds': cv,
        'search_time_seconds': elapsed,
        'all_results': []
    }
    
    cv_results = search.cv_results_
    for i in range(len(cv_results['params'])):
        results['all_results'].append({
            'rank': int(cv_results['rank_test_score'][i]),
            'params': cv_results['params'][i],
            'mean_score': float(cv_results['mean_test_score'][i]),
            'std_score': float(cv_results['std_test_score'][i])
        })
    
    logger.info(f"Best F2-score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    return results


def tune_bayesian_optuna(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv: int,
    n_jobs: int,
    task: str,
    timeout: int,
    logger
) -> Dict[str, Any]:
    """Hyperparameter tuning con Optuna (Bayesian Optimization) e F2-Score."""
    
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError(
            "Optuna non installato. Eseguire: pip install optuna\n"
            "Oppure usare --method random"
        )
    
    logger.info(f"Bayesian Optimization (Optuna): {n_trials} trials, cv={cv}")
    logger.info("Metrica: F2-Score (beta=2, Recall pesata doppio)")
    if timeout > 0:
        logger.info(f"Timeout: {timeout}s ({timeout/3600:.1f}h)")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    scoring = make_scorer(f2_scorer) if task == 'binary' else 'f1_weighted'
    
    def objective(trial):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'max_depth': trial.suggest_categorical('max_depth', [15, 20, 30, None]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                'random_state': RANDOM_STATE,
                'n_jobs': n_jobs
            }
            model = RandomForestClassifier(**params)
        
        elif model_type == 'xgboost':
            objective_fn = 'binary:logistic' if task == 'binary' else 'multi:softmax'
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'gamma': trial.suggest_float('gamma', 0, 0.2),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
                'objective': objective_fn,
                'random_state': RANDOM_STATE,
                'n_jobs': n_jobs,
                'tree_method': 'hist'
            }
            if task == 'multiclass':
                params['num_class'] = len(y_train.unique())
            model = XGBClassifier(**params)
        
        elif model_type == 'lightgbm':
            objective_fn = 'binary' if task == 'binary' else 'multiclass'
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15, 20, -1]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 100),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1),
                'objective': objective_fn,
                'random_state': RANDOM_STATE,
                'n_jobs': n_jobs,
                'verbose': -1,
                'force_col_wise': True
            }
            if task == 'multiclass':
                params['num_class'] = len(y_train.unique())
            model = LGBMClassifier(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        return scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout if timeout > 0 else None,
        show_progress_bar=True
    )
    elapsed = time.time() - start_time
    
    results = {
        'method': 'bayesian_optuna',
        'scoring_metric': 'f2_score',
        'best_params': study.best_params,
        'best_score': float(study.best_value),
        'n_trials': len(study.trials),
        'cv_folds': cv,
        'search_time_seconds': elapsed,
        'all_results': []
    }
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results['all_results'].append({
                'trial_number': trial.number,
                'params': trial.params,
                'score': float(trial.value)
            })
    
    logger.info(f"Best F2-score: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Completed trials: {len(study.trials)}")
    
    return results


def save_tuning_results(
    model_type: str,
    results: Dict[str, Any],
    task: str,
    output_dir: Path = None
) -> Path:
    """Salva risultati tuning in JSON."""
    
    if output_dir is None:
        output_dir = get_project_root() / "tuning_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'model_type': model_type,
        'task': task,
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_method': results['method'],
        'scoring_metric': results['scoring_metric'],
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'search_config': {
            'n_iterations': results.get('n_iterations', results.get('n_trials')),
            'cv_folds': results['cv_folds'],
            'search_time_seconds': results['search_time_seconds']
        },
        'all_results': results['all_results']
    }
    
    output_file = output_dir / f"{model_type}_best.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    return output_file


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning per NIDS-ML',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['random_forest', 'xgboost', 'lightgbm'],
                        help='Modello da ottimizzare')
    parser.add_argument('--method', type=str, default='random',
                        choices=['random', 'bayesian'],
                        help='Metodo ricerca (default: random)')
    parser.add_argument('--n-iter', type=int, default=50,
                        help='Iterazioni Random Search (default: 50)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Trial Bayesian Optuna (default: 100)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Fold cross-validation (default: 5)')
    parser.add_argument('--task', type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Tipo task (default: binary)')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Timeout secondi (0 = no limit, default: 0)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='CPU cores (default: auto)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    n_jobs = args.n_jobs if args.n_jobs else max(1, (os.cpu_count() or 4) - 2)
    apply_cpu_limits(n_jobs, set_low_priority=True)
    
    logger = get_logger(__name__)
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"\nModello:  {args.model}")
    print(f"Metodo:   {args.method}")
    print(f"Metrica:  F2-Score (beta=2)")
    print(f"Task:     {args.task}")
    print(f"CV:       {args.cv}")
    print(f"CPU:      {n_jobs}/{os.cpu_count()}")
    
    if args.method == 'random':
        print(f"N iter:   {args.n_iter}")
    else:
        print(f"N trials: {args.n_trials}")
        if args.timeout > 0:
            print(f"Timeout:  {args.timeout}s ({args.timeout/3600:.1f}h)")
    print()
    
    try:
        print("1. Caricamento dati...")
        train, val, test, _ = load_processed_data()
        
        print("2. Preparazione feature...")
        scaler, selected_features, _, _ = load_artifacts()
        
        label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
        feature_cols = get_feature_columns(train)
        
        X_train, y_train = prepare_xy(train, label_col, feature_cols)
        X_train_scaled = transform_data(X_train, scaler)
        X_train_final = apply_feature_selection(X_train_scaled, selected_features)
        
        print(f"   Shape: {X_train_final.shape}")
        
        print(f"\n3. Tuning ({args.method})...")
        
        if args.method == 'random':
            results = tune_random_search(
                model_type=args.model,
                X_train=X_train_final,
                y_train=y_train,
                n_iter=args.n_iter,
                cv=args.cv,
                n_jobs=n_jobs,
                task=args.task,
                logger=logger
            )
        else:
            results = tune_bayesian_optuna(
                model_type=args.model,
                X_train=X_train_final,
                y_train=y_train,
                n_trials=args.n_trials,
                cv=args.cv,
                n_jobs=n_jobs,
                task=args.task,
                timeout=args.timeout,
                logger=logger
            )
        
        print("\n4. Salvataggio risultati...")
        output_file = save_tuning_results(args.model, results, args.task)
        
        print("\n" + "=" * 70)
        print("TUNING COMPLETATO")
        print("=" * 70)
        print(f"\nBest F2-score: {results['best_score']:.4f}")
        print(f"Best params:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v}")
        print(f"\nRisultati salvati: {output_file}")
        print(f"\nProssimo step:")
        print(f"  python src/training/{args.model}.py --use-tuned-params")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima preprocessing e feature_engineering")
        sys.exit(1)
    except ImportError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()