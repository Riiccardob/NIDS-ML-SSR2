"""
================================================================================
NIDS-ML - Training LightGBM
================================================================================

Training con parametri da hyperparameter tuning (OBBLIGATORIO).

PREREQUISITO:
-------------
Prima eseguire: python src/hyperparameter_tuning.py --model lightgbm

GUIDA PARAMETRI:
----------------
    python src/training/lightgbm_model.py [opzioni]

Opzioni:
    --task STR              'binary' o 'multiclass' (default: binary)
    --tuning-config FILE    Config specifica (default: più recente)
    --tuning-timestamp TS   Timestamp config (es: 2026-01-24_20.02)
    --list-configs          Mostra config disponibili ed esci
    --early-stopping        Abilita early stopping (default)
    --no-early-stopping     Disabilita early stopping
    --n-jobs INT            Core CPU (default: auto)

ESEMPI:
-------
# Training con parametri tuned (più recente)
python src/training/lightgbm_model.py

# Config specifica
python src/training/lightgbm_model.py --tuning-config bayesian_trials50_cv5_2026-01-24_20.02.json

# Lista config
python src/training/lightgbm_model.py --list-configs

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

from lightgbm import LGBMClassifier, early_stopping
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


def load_tuned_params(
    model_type: str = 'lightgbm',
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


def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   task: str = 'binary',
                   use_early_stopping: bool = True,
                   n_jobs: int = None,
                   random_state: int = RANDOM_STATE,
                   tuned_params: Dict = None
                   ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    """Training LightGBM con parametri tuned."""
    
    if n_jobs is None:
        n_jobs = int(os.environ.get('OMP_NUM_THREADS', _n_cores))
    
    if tuned_params is None:
        raise ValueError("Parametri tuned richiesti. Eseguire prima hyperparameter_tuning.py")
    
    logger.info("=" * 50)
    logger.info(f"TRAINING LIGHTGBM ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    logger.info(f"Parametri tuned: {tuned_params}")
    
    objective = 'binary' if task == 'binary' else 'multiclass'
    
    final_params = tuned_params.copy()
    final_params['objective'] = objective
    final_params['random_state'] = random_state
    final_params['n_jobs'] = n_jobs
    final_params['verbose'] = -1
    final_params['class_weight'] = 'balanced'
    final_params['force_col_wise'] = True
    
    if task == 'multiclass' and 'num_class' not in final_params:
        final_params['num_class'] = len(y_train.unique())
    
    best_model = LGBMClassifier(**final_params)
    
    start_time = datetime.now()
    
    if use_early_stopping:
        best_model.set_params(n_estimators=500)
        eval_metric = 'binary_logloss' if task == 'binary' else 'multi_logloss'
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
        )
        best_iteration = best_model.best_iteration_
    else:
        best_model.fit(X_train, y_train)
        best_iteration = None
    
    train_time = (datetime.now() - start_time).total_seconds()
    
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
        'model_name': 'LightGBM',
        'task': task,
        'training_mode': 'tuned_params',
        'best_params': tuned_params,
        'best_cv_score': None,
        'validation_metrics': metrics,
        'train_time_seconds': train_time,
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'early_stopping_used': use_early_stopping,
        'best_iteration': best_iteration,
        'n_jobs': n_jobs
    }
    
    return best_model, results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Training LightGBM per NIDS (con parametri tuned)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PREREQUISITO: Eseguire prima hyperparameter tuning
  python src/hyperparameter_tuning.py --model lightgbm --timeout 3600

Esempi:
  # Parametri tuned (più recente)
  python src/training/lightgbm_model.py
  
  # Config specifica
  python src/training/lightgbm_model.py --tuning-config bayesian_trials50_cv5_2026-01-24_20.02.json
  
  # Lista config
  python src/training/lightgbm_model.py --list-configs
        """
    )
    
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--tuning-config', type=str, default=None,
                        help='File config specifico (default: più recente)')
    parser.add_argument('--tuning-timestamp', type=str, default=None,
                        help='Timestamp config da usare (es: 2026-01-24_20.02)')
    parser.add_argument('--list-configs', action='store_true',
                        help='Mostra configurazioni tuning disponibili ed esci')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true', default=True)
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--max-ram', type=int, default=85)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if args.list_configs:
        print_available_configs('lightgbm')
        return
    
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    print("\n" + "=" * 60)
    print("LIGHTGBM TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:           {args.task}")
    print(f"  Mode:           Tuned parameters (OBBLIGATORIO)")
    
    # Carica parametri tuned
    tuned_params, tuning_filepath = load_tuned_params(
        'lightgbm',
        args.task,
        config_file=args.tuning_config,
        timestamp=args.tuning_timestamp
    )
    
    if not tuned_params:
        print("\n❌ ERRORE: Nessuna configurazione tuning trovata!")
        print("\nPREREQUISITO: Eseguire prima hyperparameter tuning:")
        print("  python src/hyperparameter_tuning.py --model lightgbm --timeout 3600")
        print("\nOppure usa --list-configs per vedere le configurazioni disponibili")
        sys.exit(1)
    
    print(f"  Config:         {tuning_filepath.name}")
    print(f"  Early stopping: {args.early_stopping}")
    print(f"  CPU cores:      {n_jobs}/{os.cpu_count()}")
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
        
        print("\n3. Training LightGBM...")
        model, results = train_lightgbm(
            X_train_final, y_train, X_val_final, y_val,
            task=args.task,
            use_early_stopping=args.early_stopping,
            n_jobs=n_jobs, random_state=args.random_state,
            tuned_params=tuned_params
        )
        
        print("\n4. Salvataggio...")
        from src.model_versioning import save_tuned_model
        
        version_dir, version_id = save_tuned_model(
            model=model,
            results=results,
            selected_features=selected_features,
            model_type='lightgbm',
            tuning_filepath=tuning_filepath
        )
        
        model_path = version_dir / f"model_{args.task}.pkl"
        print(f"   Versione: {version_id}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO")
        print("=" * 60)
        print(f"\nMetriche validazione:")
        for name, value in results['validation_metrics'].items():
            print(f"  {name.capitalize():12}: {value:.4f}")
        print(f"\nTempo: {results['train_time_seconds']:.1f}s")
        print(f"Modello: {model_path}")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()