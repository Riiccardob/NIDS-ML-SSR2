"""
================================================================================
NIDS-ML - Hyperparameter Tuning
================================================================================

Ricerca parametri ottimali usando Composite Score: 70% F2-Score + 30% Latency.

METRICA COMPOSITA (allineata con compare_models.py):
- F2-Score (70%): Enfatizza Recall (beta=2)
- Latency (30%): Tempo inferenza per sample

USAGE:
------
    python src/hyperparameter_tuning.py --model <model> [options]

PARAMETRI:
----------
    --model STR           Modello da ottimizzare: random_forest, xgboost, lightgbm
    --n-trials INT        Trial Bayesian Optuna (mutuamente esclusivo con --timeout)
    --timeout INT         Timeout in secondi (mutuamente esclusivo con --n-trials)
                          CONSIGLIATO: usa --timeout e lascia che Optuna gestisca i trials
    --cv INT              Fold cross-validation (default: 5)
    --task STR            binary o multiclass (default: binary)
    --max-latency-ms FLOAT Constraint latency ms/sample (default: 1.0)
    --n-jobs INT          CPU cores (default: auto)

ESEMPI:
-------
# CONSIGLIATO: Limita per tempo, Optuna decide i trials
python src/hyperparameter_tuning.py --model xgboost --timeout 3600  # 1 ora

# Alternativa: Limita per numero trials
python src/hyperparameter_tuning.py --model lightgbm --n-trials 50

# Entrambi i limiti (si ferma al primo raggiunto)
python src/hyperparameter_tuning.py --model random_forest --n-trials 100 --timeout 7200

OUTPUT:
-------
Salva risultati in: tuning_results/<model>/<descrittivo>.json

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
from sklearn.model_selection import cross_val_score
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

# CONFIGURAZIONE UNICA PER BAYESIAN OPTIMIZATION
HYPERPARAM_CONFIG = {
    'random_forest': {
        'n_estimators':      ('int', 50, 500),
        'max_depth':         ('cat', [10, 20, 30, 40, 50, None]),
        'min_samples_split': ('int', 2, 20),
        'min_samples_leaf':  ('int', 1, 10),
        'max_features':      ('cat', ['sqrt', 'log2', None]),
        'bootstrap':         ('cat', [True, False]),
        'class_weight':      ('cat', ['balanced', 'balanced_subsample', None])
    },
    'xgboost': {
        'n_estimators':      ('int', 100, 5000),
        'max_depth':         ('int', 3, 20),
        'learning_rate':     ('float', 0.001, 0.3, True),
        'subsample':         ('float', 0.5, 1.0, False),
        'colsample_bytree':  ('float', 0.5, 1.0, False),
        'min_child_weight':  ('int', 1, 10),
        'gamma':             ('float', 0.0, 5.0, False),
        'reg_alpha':         ('float', 1e-8, 10.0, True),
        'reg_lambda':        ('float', 1e-8, 10.0, True),
        'scale_pos_weight':  ('float', 1.0, 10.0, False)
    },
    'lightgbm': {
        'n_estimators':      ('int', 100, 5000),
        'max_depth':         ('int', -1, 30),
        'learning_rate':     ('float', 0.001, 0.3, True),
        'num_leaves':        ('int', 20, 300),
        'subsample':         ('float', 0.5, 1.0, False),
        'colsample_bytree':  ('float', 0.5, 1.0, False),
        'min_child_samples': ('int', 10, 100),
        'reg_alpha':         ('float', 1e-8, 10.0, True),
        'reg_lambda':        ('float', 1e-8, 10.0, True),
        'class_weight':      ('cat', ['balanced', None])
    }
}


# ==============================================================================
# LATENCY-AWARE SCORER (70% F2 + 30% Latency)
# ==============================================================================

class LatencyAwareScorer:
    """
    Custom scorer: 70% F2-Score + 30% Latency Score.
    ALLINEATO con compare_models.py per coerenza pipeline.
    """
    
    def __init__(self, max_latency_ms=1.0, f2_weight=0.7, latency_weight=0.3):
        self.max_latency_ms = max_latency_ms
        self.f2_weight = f2_weight
        self.latency_weight = latency_weight
    
    def __call__(self, estimator, X, y):
        # 1. F2-Score
        y_pred = estimator.predict(X)
        f2 = fbeta_score(y, y_pred, beta=2, zero_division=0)
        
        # 2. Misura latency
        n_samples = len(X)
        _ = estimator.predict(X[:min(100, n_samples)])  # Warmup
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = estimator.predict(X)
            times.append((time.perf_counter() - start) * 1000)
        
        latency_per_sample = np.mean(times) / n_samples
        
        # 3. Latency score normalizzato
        latency_score = max(0, 1 - (latency_per_sample / self.max_latency_ms))
        
        # 4. Composite score (70/30)
        composite = (self.f2_weight * f2) + (self.latency_weight * latency_score)
        
        return composite


def tune_bayesian_optuna(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv: int,
    n_jobs: int,
    task: str,
    timeout: int,
    max_latency_ms: float,
    logger
) -> Dict[str, Any]:
    """Hyperparameter tuning con Optuna e Composite Score."""
    
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError(
            "Optuna non installato. Eseguire: pip install optuna\n"
        )
    
    logger.info(f"Bayesian Optimization (Optuna): cv={cv}")
    logger.info("Metrica: 70% F2-Score + 30% Latency")
    logger.info(f"Max latency constraint: {max_latency_ms}ms/sample")
    
    # Gestione limiti
    if n_trials is not None and timeout > 0:
        logger.info(f"Limiti: {n_trials} trials OPPURE {timeout}s ({timeout/3600:.1f}h) - si ferma al primo")
    elif n_trials is not None:
        logger.info(f"Limite: {n_trials} trials")
    elif timeout > 0:
        logger.info(f"Limite: {timeout}s ({timeout/3600:.1f}h) - Optuna gestisce i trials")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    scorer = LatencyAwareScorer(
        max_latency_ms=max_latency_ms,
        f2_weight=0.7,
        latency_weight=0.3
    )
    
    def objective(trial):
        config = HYPERPARAM_CONFIG[model_type]
        params = {}
        
        for p_name, specs in config.items():
            p_type = specs[0]
            
            if p_type == 'int':
                params[p_name] = trial.suggest_int(p_name, specs[1], specs[2])
            elif p_type == 'float':
                log_scale = specs[3]
                params[p_name] = trial.suggest_float(p_name, specs[1], specs[2], log=log_scale)
            elif p_type == 'cat':
                params[p_name] = trial.suggest_categorical(p_name, specs[1])
        
        params['random_state'] = RANDOM_STATE
        params['n_jobs'] = n_jobs
        
        if model_type == 'xgboost':
            params['objective'] = 'binary:logistic' if task == 'binary' else 'multi:softmax'
            params['tree_method'] = 'hist'
            if task == 'multiclass':
                params['num_class'] = len(y_train.unique())
                if 'scale_pos_weight' in params: 
                    del params['scale_pos_weight']
                
        elif model_type == 'lightgbm':
            params['objective'] = 'binary' if task == 'binary' else 'multiclass'
            params['verbose'] = -1
            params['force_col_wise'] = True
            if task == 'multiclass':
                params['num_class'] = len(y_train.unique())
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_type == 'xgboost':
            model = XGBClassifier(**params)
        elif model_type == 'lightgbm':
            model = LGBMClassifier(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=1)
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
    
    # Estrai metriche separate del best trial
    best_params = study.best_params
    
    # Train finale per calcolare F2 e latency separati
    if model_type == 'random_forest':
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = n_jobs
        best_model = RandomForestClassifier(**best_params)
    elif model_type == 'xgboost':
        best_params['objective'] = 'binary:logistic' if task == 'binary' else 'multi:softmax'
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = n_jobs
        best_params['tree_method'] = 'hist'
        if task == 'multiclass':
            best_params['num_class'] = len(y_train.unique())
        best_model = XGBClassifier(**best_params)
    elif model_type == 'lightgbm':
        best_params['objective'] = 'binary' if task == 'binary' else 'multiclass'
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = n_jobs
        best_params['verbose'] = -1
        best_params['force_col_wise'] = True
        if task == 'multiclass':
            best_params['num_class'] = len(y_train.unique())
        best_model = LGBMClassifier(**best_params)
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_train)
    best_f2 = fbeta_score(y_train, y_pred, beta=2, zero_division=0)
    
    # Misura latency
    _ = best_model.predict(X_train[:100])
    latency_times = []
    for _ in range(5):
        start = time.perf_counter()
        _ = best_model.predict(X_train)
        latency_times.append((time.perf_counter() - start) * 1000)
    best_latency = np.mean(latency_times) / len(X_train)
    
    results = {
        'method': 'bayesian_optuna',
        'scoring_metric': 'f2_latency_composite',
        'best_params': study.best_params,
        'best_score': float(study.best_value),
        'best_f2_score': float(best_f2),
        'best_latency_ms': float(best_latency),
        'n_trials': len(study.trials),
        'cv_folds': cv,
        'search_time_seconds': elapsed,
        'max_latency_constraint': max_latency_ms,
        'f2_weight': 0.7,
        'latency_weight': 0.3,
        'all_results': []
    }
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results['all_results'].append({
                'trial_number': trial.number,
                'params': trial.params,
                'score': float(trial.value)
            })
    
    logger.info(f"Best composite score: {study.best_value:.4f}")
    logger.info(f"  - F2-Score: {best_f2:.4f}")
    logger.info(f"  - Latency: {best_latency:.4f}ms/sample")
    logger.info(f"Completed trials: {len(study.trials)}")
    logger.info(f"Best params: {study.best_params}")
    
    return results


def save_tuning_results(
    model_type: str,
    results: Dict[str, Any],
    task: str,
    output_dir: Path = None
) -> Path:
    """Salva risultati tuning in JSON organizzati per sottocartelle e con nomi descrittivi."""
    
    if output_dir is None:
        output_dir = get_project_root() / "tuning_results"
    
    model_dir = output_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'model_type': model_type,
        'task': task,
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_method': results['method'],
        'scoring_metric': results['scoring_metric'],
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'best_f2_score': results.get('best_f2_score'),
        'best_latency_ms': results.get('best_latency_ms'),
        'search_config': {
            'n_trials': results.get('n_trials'),
            'cv_folds': results['cv_folds'],
            'search_time_seconds': results['search_time_seconds'],
            'max_latency_constraint': results.get('max_latency_constraint'),
            'f2_weight': results.get('f2_weight', 0.7),
            'latency_weight': results.get('latency_weight', 0.3)
        },
        'all_results': results['all_results']
    }
    
    timestamp_short = datetime.now().strftime("%Y-%m-%d_%H.%M")
    cv = results['cv_folds']
    n_trials = results['n_trials']
    
    filename = f"bayesian_trials{n_trials}_cv{cv}_{timestamp_short}.json"
    output_file = model_dir / filename
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    return output_file


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning per NIDS-ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANTE: Specificare almeno uno tra --n-trials o --timeout

CONSIGLIATO: Usa --timeout e lascia che Optuna gestisca automaticamente 
il numero di trials in base al tempo disponibile.

Esempi:
  # CONSIGLIATO: Limite di tempo
  python src/hyperparameter_tuning.py --model xgboost --timeout 3600
  
  # Alternativa: Limite di trials
  python src/hyperparameter_tuning.py --model lightgbm --n-trials 50
  
  # Entrambi (si ferma al primo raggiunto)
  python src/hyperparameter_tuning.py --model random_forest --n-trials 100 --timeout 7200
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['random_forest', 'xgboost', 'lightgbm'],
                        help='Modello da ottimizzare')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Trial Bayesian Optuna (mutuamente esclusivo con --timeout)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Fold cross-validation (default: 5)')
    parser.add_argument('--task', type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Tipo task (default: binary)')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Timeout secondi (0 = no limit). CONSIGLIATO rispetto a --n-trials')
    parser.add_argument('--max-latency-ms', type=float, default=1.0,
                        help='Constraint latency ms/sample (default: 1.0)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='CPU cores (default: auto)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validazione: almeno uno tra n_trials e timeout deve essere specificato
    if args.n_trials is None and args.timeout == 0:
        print("\nERRORE: Specificare almeno uno tra --n-trials o --timeout")
        print("\nCONSIGLIATO: Usa --timeout per lasciare che Optuna gestisca i trials")
        print("Esempio: python src/hyperparameter_tuning.py --model xgboost --timeout 3600")
        sys.exit(1)
    
    n_jobs = args.n_jobs if args.n_jobs else max(1, (os.cpu_count() or 4) - 2)
    apply_cpu_limits(n_jobs, set_low_priority=True)
    
    logger = get_logger(__name__)
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"\nModello:      {args.model}")
    print(f"Metodo:       Bayesian Optimization (Optuna)")
    print(f"Metrica:      70% F2-Score + 30% Latency (composite)")
    print(f"Task:         {args.task}")
    print(f"CV:           {args.cv}")
    print(f"Max Latency:  {args.max_latency_ms}ms/sample")
    print(f"CPU:          {n_jobs}/{os.cpu_count()}")
    
    if args.n_trials is not None and args.timeout > 0:
        print(f"Limiti:       {args.n_trials} trials OPPURE {args.timeout}s ({args.timeout/3600:.1f}h)")
        print("              (si ferma al primo raggiunto)")
    elif args.n_trials is not None:
        print(f"N trials:     {args.n_trials}")
    elif args.timeout > 0:
        print(f"Timeout:      {args.timeout}s ({args.timeout/3600:.1f}h)")
        print("              Optuna gestisce il numero di trials")
    print()
    
    try:
        print("1. Caricamento dati...")
        train, val, test, _ = load_processed_data()
        
        print("2. Preparazione feature...")
        scaler, selected_features, _, _ = load_artifacts()
        
        label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'

        # --- FIX INIZIO ---
        # Recupera le feature che lo scaler si aspetta (necessario se usato statistical preprocessing)
        if hasattr(scaler, 'feature_names_in_'):
            # Usa le feature memorizzate nello scaler (più sicuro)
            feature_cols = list(scaler.feature_names_in_)
            print(f"   Scaler feature alignment: {len(feature_cols)} feature richieste")
        else:
            # Fallback
            feature_cols = get_feature_columns(train)
            
        # Verifica che tutte le feature richieste siano presenti nel train
        missing_cols = set(feature_cols) - set(train.columns)
        if missing_cols:
            raise ValueError(f"Mancano {len(missing_cols)} feature richieste dallo scaler nel dataset (es. {list(missing_cols)[:3]})")
            
        X_train, y_train = prepare_xy(train, label_col, feature_cols)
        # --- FIX FINE ---
        
        X_train_scaled = transform_data(X_train, scaler)
        X_train_final = apply_feature_selection(X_train_scaled, selected_features)
        
        print(f"   Shape: {X_train_final.shape}")
        
        print("\n3. Tuning (Bayesian Optuna)...")
        print("   NOTA: Misura latency durante CV, rallenta il processo")
        
        results = tune_bayesian_optuna(
            model_type=args.model,
            X_train=X_train_final,
            y_train=y_train,
            n_trials=args.n_trials,
            cv=args.cv,
            n_jobs=n_jobs,
            task=args.task,
            timeout=args.timeout,
            max_latency_ms=args.max_latency_ms,
            logger=logger
        )
        
        print("\n4. Salvataggio risultati...")
        output_file = save_tuning_results(args.model, results, args.task)
        
        print("\n" + "=" * 70)
        print("TUNING COMPLETATO")
        print("=" * 70)
        print(f"\nBest composite score: {results['best_score']:.4f}")
        print(f"  - F2-Score: {results.get('best_f2_score', 'N/A'):.4f}")
        print(f"  - Latency:  {results.get('best_latency_ms', 'N/A'):.4f}ms/sample")
        print(f"\nTrials completati: {results['n_trials']}")
        print(f"Tempo totale: {results['search_time_seconds']:.1f}s ({results['search_time_seconds']/3600:.2f}h)")
        print(f"\nBest params:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v}")
        print(f"\nRisultati salvati: {output_file}")
        print(f"\nProssimo step:")
        print(f"  python src/training/{args.model}.py")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima preprocessing e feature_engineering")
        sys.exit(1)
    except ImportError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()