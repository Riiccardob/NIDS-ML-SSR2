"""
================================================================================
NIDS-ML - Training XGBoost
================================================================================

GUIDA PARAMETRI:
----------------
    python src/training/xgboost_model.py [opzioni]

Opzioni:
    --task STR              'binary' o 'multiclass' (default: binary)
    --use-tuned-params      Usa parametri da hyperparameter_tuning.py
    --tuning-config FILE    Config specifica (default: più recente)
    --tuning-timestamp TS   Timestamp config (es: 2026-01-24_20.02)
    --list-configs          Mostra config disponibili ed esci
    --n-iter INT            Iterazioni random search (default: 20)
    --cv INT                Fold CV (default: 3)
    --early-stopping        Abilita early stopping (default)
    --no-early-stopping     Disabilita early stopping
    --n-jobs INT            Core CPU (default: auto)
    --gpu                   Usa GPU se disponibile (default: auto-detect)
    --no-gpu                Forza uso CPU

ESEMPI:
-------
# Parametri tuned (più recente)
python src/training/xgboost_model.py --use-tuned-params

# Config specifica
python src/training/xgboost_model.py --use-tuned-params --tuning-config random_iter50_cv5_2026-01-24_20.02.json

# Lista config
python src/training/xgboost_model.py --list-configs

# Random search
python src/training/xgboost_model.py --n-iter 5 --cv 2

# Con GPU
python src/training/xgboost_model.py --use-tuned-params --gpu

================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

def _get_arg(name, default=None):
    for i, arg in enumerate(sys.argv):
        if arg == f'--{name}' and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return sys.argv[i + 1]
        elif arg == f'--{name}':
            return True
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

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import gc

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import get_logger, get_project_root, RANDOM_STATE, ResourceLimiter, suppress_warnings
from src.preprocessing import load_processed_data
from src.feature_engineering import (
    load_artifacts, get_feature_columns, prepare_xy,
    transform_data, apply_feature_selection, run_feature_engineering
)
from src.timing import TimingLogger

suppress_warnings()
logger = get_logger(__name__)


def detect_gpu() -> bool:
    """Rileva se GPU CUDA è disponibile per XGBoost."""
    try:
        test_model = XGBClassifier(device='cuda', n_estimators=1)
        test_model.fit([[0, 1], [1, 0]], [0, 1])
        del test_model
        return True
    except Exception:
        pass
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return False
    except Exception:
        pass
    
    return False


PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

DEFAULT_N_ITER = 20
DEFAULT_CV_FOLDS = 3


def load_tuned_params(
    model_type: str = 'xgboost',
    task: str = 'binary',
    config_file: str = None,
    timestamp: str = None
) -> Tuple[Optional[Dict], Optional[Path]]:
    """Carica parametri da hyperparameter tuning."""
    tuning_dir = get_project_root() / "tuning_results" / model_type
    
    if not tuning_dir.exists():
        return None, None
    
    configs = []
    for json_file in tuning_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            configs.append({
                'filepath': json_file,
                'filename': json_file.name,
                'timestamp': data.get('tuning_timestamp'),
            })
        except Exception:
            continue
    
    if not configs:
        return None, None
    
    configs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    if config_file:
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = tuning_dir / config_path.name
        
        if not config_path.exists():
            logger.warning(f"Config file non trovato: {config_path}")
            return None, None
        
        tuning_filepath = config_path
    
    elif timestamp:
        found = None
        for cfg in configs:
            if timestamp in cfg['filename']:
                found = cfg['filepath']
                break
        
        if not found:
            logger.warning(f"Nessuna config trovata con timestamp '{timestamp}'")
            return None, None
        
        tuning_filepath = found
    
    else:
        tuning_filepath = configs[0]['filepath']
    
    with open(tuning_filepath) as f:
        data = json.load(f)
    
    if data.get('task') != task:
        logger.warning(
            f"Task mismatch: config per {data.get('task')}, richiesto {task}"
        )
        return None, None
    
    logger.info(f"Parametri caricati da: {tuning_filepath.name}")
    logger.info(f"Metodo tuning: {data.get('tuning_method')}")
    logger.info(f"Best score tuning: {data.get('best_score'):.4f}")
    
    return data['best_params'], tuning_filepath


def print_available_configs(model_type: str):
    """Stampa lista configurazioni disponibili."""
    tuning_dir = get_project_root() / "tuning_results" / model_type
    
    if not tuning_dir.exists():
        print(f"Nessuna configurazione tuning trovata per {model_type}")
        return
    
    configs = []
    for json_file in tuning_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            configs.append({
                'filename': json_file.name,
                'timestamp': data.get('tuning_timestamp'),
                'method': data.get('tuning_method'),
                'score': data.get('best_score')
            })
        except Exception:
            continue
    
    if not configs:
        print(f"Nessuna configurazione tuning trovata per {model_type}")
        return
    
    configs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"CONFIGURAZIONI TUNING DISPONIBILI - {model_type.upper()}")
    print(f"{'='*70}")
    print(f"\n{'#':<3} {'Filename':<45} {'Score':>8} {'Method':<10}")
    print("-"*70)
    
    for i, cfg in enumerate(configs, 1):
        print(f"{i:<3} {cfg['filename']:<45} {cfg['score']:>8.4f} {cfg['method']:<10}")
    
    print(f"\nTotale: {len(configs)} configurazioni")


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: pd.DataFrame,
                  y_val: pd.Series,
                  task: str = 'binary',
                  n_iter: int = DEFAULT_N_ITER,
                  cv: int = DEFAULT_CV_FOLDS,
                  use_early_stopping: bool = True,
                  n_jobs: int = None,
                  use_gpu: bool = False,
                  random_state: int = RANDOM_STATE,
                  use_tuned_params: bool = False,
                  tuned_params: Dict = None
                  ) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Training XGBoost con RandomizedSearchCV e supporto GPU."""
    
    if n_jobs is None:
        n_jobs = int(os.environ.get('OMP_NUM_THREADS', _n_cores))
    
    logger.info("=" * 50)
    logger.info(f"TRAINING XGBOOST ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    
    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    objective = 'binary:logistic' if task == 'binary' else 'multi:softmax'
    
    scale_pos_weight = None
    if task == 'binary':
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if use_tuned_params and tuned_params:
        logger.info("Modalita: TRAINING CON PARAMETRI TUNED")
        logger.info(f"Parametri: {tuned_params}")
        
        final_params = tuned_params.copy()
        final_params['objective'] = objective
        final_params['random_state'] = random_state
        final_params['use_label_encoder'] = False
        final_params['tree_method'] = 'hist'
        
        if use_gpu:
            final_params['device'] = 'cuda'
            final_params['n_jobs'] = 1
        else:
            final_params['n_jobs'] = n_jobs
        
        if scale_pos_weight:
            final_params['scale_pos_weight'] = scale_pos_weight
        
        if task == 'multiclass' and 'num_class' not in final_params:
            final_params['num_class'] = len(y_train.unique())
        
        best_model = XGBClassifier(**final_params)
        
        start_time = datetime.now()
        
        if use_early_stopping:
            best_model.set_params(n_estimators=500, early_stopping_rounds=20)
            best_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            best_iteration = best_model.best_iteration
        else:
            best_model.fit(X_train, y_train)
            best_iteration = None
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        best_params = tuned_params
        best_cv_score = None
    
    else:
        logger.info(f"Config: n_iter={n_iter}, cv={cv}, n_jobs={n_jobs}, GPU={use_gpu}")
        
        base_params = {
            'objective': objective,
            'random_state': random_state,
            'verbosity': 1,
            'use_label_encoder': False,
            'tree_method': 'hist',
        }
        
        if use_gpu:
            base_params['device'] = 'cuda'
            base_params['n_jobs'] = 1
            cv_n_jobs = 1
            print("   GPU: ABILITATA (CUDA)")
        else:
            base_params['n_jobs'] = n_jobs
            cv_n_jobs = n_jobs
            print(f"   CPU: {n_jobs} cores")
        
        if scale_pos_weight:
            base_params['scale_pos_weight'] = scale_pos_weight
        
        total_fits = n_iter * cv
        print(f"\n   RandomizedSearchCV: {n_iter} x {cv} = {total_fits} fit totali")
        print(f"   Attendere...\n")
        
        base_xgb = XGBClassifier(**base_params)
        
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=cv_n_jobs,
            verbose=2,
            return_train_score=False
        )
        
        start_time = datetime.now()
        search.fit(X_train, y_train)
        
        print(f"\n   Search completato")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        best_iteration = None
        if use_early_stopping:
            print("   Retraining con early stopping...")
            final_params = {**base_params, **search.best_params_}
            final_params['n_estimators'] = 500
            final_params['early_stopping_rounds'] = 20
            
            best_model = XGBClassifier(**final_params)
            best_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            best_iteration = best_model.best_iteration
            print(f"   Early stopping a iterazione: {best_iteration}")
        else:
            best_model = search.best_estimator_
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
        
        del search
        gc.collect()
    
    y_val_pred = best_model.predict(X_val)
    
    if task == 'binary':
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
        
        metrics = {
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision': float(precision_score(y_val, y_val_pred, zero_division=0)),
            'recall': float(recall_score(y_val, y_val_pred, zero_division=0)),
            'f1': float(f1_score(y_val, y_val_pred, zero_division=0)),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        }
    else:
        metrics = {
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision_weighted': float(precision_score(y_val, y_val_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_val, y_val_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_val, y_val_pred, average='weighted', zero_division=0))
        }
    
    logger.info("\nMetriche Validation:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    results = {
        'model_name': 'XGBoost',
        'task': task,
        'training_mode': 'tuned_params' if use_tuned_params else 'random_search',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'validation_metrics': metrics,
        'train_time_seconds': train_time,
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'early_stopping_used': use_early_stopping,
        'best_iteration': best_iteration,
        'n_iter': n_iter if not use_tuned_params else None,
        'cv_folds': cv if not use_tuned_params else None,
        'n_jobs': n_jobs,
        'gpu_used': use_gpu
    }
    
    return best_model, results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training XGBoost per NIDS')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Usa parametri da hyperparameter_tuning.py')
    parser.add_argument('--tuning-config', type=str, default=None,
                        help='File config specifico (default: più recente)')
    parser.add_argument('--tuning-timestamp', type=str, default=None,
                        help='Timestamp config da usare (es: 2026-01-24_20.02)')
    parser.add_argument('--list-configs', action='store_true',
                        help='Mostra configurazioni tuning disponibili ed esci')
    parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER)
    parser.add_argument('--cv', type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true', default=True)
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=None,
                        help='Forza uso GPU')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                        help='Forza uso CPU')
    parser.add_argument('--max-ram', type=int, default=85)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if args.list_configs:
        print_available_configs('xgboost')
        return
    
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    if args.use_gpu is None:
        use_gpu = detect_gpu()
    else:
        use_gpu = args.use_gpu
    
    print("\n" + "=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:           {args.task}")
    
    tuned_params = None
    tuning_filepath = None
    
    if args.use_tuned_params:
        print(f"  Mode:           Tuned parameters")
        tuned_params, tuning_filepath = load_tuned_params(
            'xgboost',
            args.task,
            config_file=args.tuning_config,
            timestamp=args.tuning_timestamp
        )
        
        if not tuned_params:
            print("\n⚠️  Parametri tuned non trovati, uso random search")
            args.use_tuned_params = False
        else:
            print(f"  Config:         {tuning_filepath.name}")
    else:
        print(f"  Mode:           Random search")
        print(f"  N iter:         {args.n_iter}")
        print(f"  CV folds:       {args.cv}")
    
    print(f"  Early stopping: {args.early_stopping}")
    print(f"  CPU cores:      {n_jobs}/{os.cpu_count()}")
    print(f"  GPU:            {'ABILITATA' if use_gpu else 'Disabilitata'}")
    print()
    
    try:
        print("1. Caricamento dati...")
        train, val, test, _ = load_processed_data()
        print(f"   Train: {len(train):,} | Val: {len(val):,}")
        
        try:
            print("\n2. Caricamento artifacts...")
            scaler, selected_features, _, _ = load_artifacts()
            
            feature_cols = get_feature_columns(train)
            X_train, y_train = prepare_xy(train, label_col, feature_cols)
            X_val, y_val = prepare_xy(val, label_col, feature_cols)
            
            X_train_scaled = transform_data(X_train, scaler)
            X_val_scaled = transform_data(X_val, scaler)
            
            X_train_final = apply_feature_selection(X_train_scaled, selected_features)
            X_val_final = apply_feature_selection(X_val_scaled, selected_features)
            
            del X_train, X_val, X_train_scaled, X_val_scaled, train, val
            gc.collect()
            
        except FileNotFoundError:
            print("\n2. Feature engineering...")
            X_train_final, X_val_final, _, y_train, y_val, _ = run_feature_engineering(
                train, val, test, label_col=label_col
            )
        
        print(f"   Shape: {X_train_final.shape}")
        limiter.log_status(logger)
        
        print("\n3. Training XGBoost...")
        model, results = train_xgboost(
            X_train_final, y_train, X_val_final, y_val,
            task=args.task, n_iter=args.n_iter, cv=args.cv,
            use_early_stopping=args.early_stopping,
            n_jobs=n_jobs, use_gpu=use_gpu, random_state=args.random_state,
            use_tuned_params=args.use_tuned_params,
            tuned_params=tuned_params
        )
        
        print("\n4. Salvataggio...")
        if args.use_tuned_params and tuning_filepath:
            from src.model_versioning import save_tuned_model
            
            version_dir, version_id = save_tuned_model(
                model=model,
                results=results,
                selected_features=selected_features,
                model_type='xgboost',
                tuning_filepath=tuning_filepath
            )
            
            model_path = version_dir / f"model_{args.task}.pkl"
            print(f"   Versione: {version_id}")
        else:
            output_dir = get_project_root() / "models" / "xgboost"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / f"model_{args.task}.pkl"
            results_path = output_dir / f"results_{args.task}.json"
            
            joblib.dump(model, model_path)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if selected_features is not None:
                features_path = output_dir / f"features_{args.task}.json"
                with open(features_path, 'w') as f:
                    json.dump(selected_features, f, indent=2)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO")
        print("=" * 60)
        print(f"\nMetriche validazione:")
        for name, value in results['validation_metrics'].items():
            print(f"  {name.capitalize():12}: {value:.4f}")
        print(f"\nTempo: {results['train_time_seconds']:.1f}s")
        print(f"GPU usata: {results.get('gpu_used', False)}")
        print(f"Modello: {model_path}")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()