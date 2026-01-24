"""
================================================================================
NIDS-ML - Training Random Forest
================================================================================

GUIDA PARAMETRI:
----------------
    python src/training/random_forest.py [opzioni]

Opzioni disponibili:
    --task STR            'binary' o 'multiclass' (default: binary)
    --use-tuned-params    Usa parametri da hyperparameter_tuning.py
    --tuning-config FILE  Config specifica (default: più recente)
    --tuning-timestamp TS Timestamp config (es: 2026-01-24_20.02)
    --list-configs        Mostra config disponibili ed esci
    --n-iter INT          Iterazioni random search (default: 20)
    --cv INT              Fold cross-validation (default: 3)
    --n-jobs INT          Core CPU (default: auto, totale - 2)
    --max-ram INT         Limite RAM %
    --random-state INT    Seed random

ESEMPI:
-------
# Training con parametri tuned (più recente)
python src/training/random_forest.py --use-tuned-params

# Training con config specifica
python src/training/random_forest.py --use-tuned-params --tuning-config random_iter50_cv5_2026-01-24_20.02.json

# Training con timestamp
python src/training/random_forest.py --use-tuned-params --tuning-timestamp 2026-01-24_20.02

# Mostra config disponibili
python src/training/random_forest.py --list-configs

# Training standard (random search)
python src/training/random_forest.py

# Test veloce
python src/training/random_forest.py --n-iter 5 --cv 2

================================================================================
"""

# ==============================================================================
# SETUP LIMITI CPU - DEVE ESSERE PRIMA DI ALTRI IMPORT
# ==============================================================================
import sys
import os
import argparse
from pathlib import Path

def _get_arg(name, default=None):
    """Estrae argomento da sys.argv."""
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
from typing import Tuple, Dict, Any, List, Optional
import joblib
import json
from datetime import datetime
import gc
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    ResourceLimiter,
    suppress_warnings
)
from src.preprocessing import load_processed_data
from src.feature_engineering import (
    load_artifacts,
    get_feature_columns,
    prepare_xy,
    transform_data,
    apply_feature_selection,
    run_feature_engineering
)
from src.timing import TimingLogger

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

DEFAULT_N_ITER = 20
DEFAULT_CV_FOLDS = 3
DEFAULT_MAX_RAM = 85

# ==============================================================================
# IMPORT PARAM DA TUNING
# ==============================================================================

def load_tuned_params(
    model_type: str = 'random_forest',
    task: str = 'binary',
    config_file: str = None,
    timestamp: str = None
) -> Tuple[Optional[Dict], Optional[Path]]:
    """
    Carica parametri da hyperparameter tuning.
    
    Returns:
        Tuple (params, tuning_filepath)
    """
    tuning_dir = get_project_root() / "tuning_results" / model_type
    
    if not tuning_dir.exists():
        return None, None
    
    # Lista config disponibili
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
    
    # Trova il file tuning
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


# ==============================================================================
# TRAINING CON PROGRESS BAR
# ==============================================================================

def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        task: str = 'binary',
                        n_iter: int = DEFAULT_N_ITER,
                        cv: int = DEFAULT_CV_FOLDS,
                        n_jobs: int = None,
                        random_state: int = RANDOM_STATE,
                        use_tuned_params: bool = False,
                        tuned_params: Dict = None
                        ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Training Random Forest."""
    
    if n_jobs is None:
        n_jobs = int(os.environ.get('OMP_NUM_THREADS', _n_cores))
    
    logger.info("=" * 50)
    logger.info(f"TRAINING RANDOM FOREST ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    
    if use_tuned_params and tuned_params:
        logger.info("Modalita: TRAINING CON PARAMETRI TUNED")
        logger.info(f"Parametri: {tuned_params}")
        
        final_params = tuned_params.copy()
        final_params['random_state'] = random_state
        final_params['n_jobs'] = n_jobs
        
        best_model = RandomForestClassifier(**final_params)
        
        start_time = datetime.now()
        best_model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        
        best_params = tuned_params
        best_cv_score = None
    
    else:
        logger.info(f"Config: n_iter={n_iter}, cv={cv}, n_jobs={n_jobs}")
        
        scoring = 'f1' if task == 'binary' else 'f1_weighted'
        
        base_rf = RandomForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=2,
            return_train_score=False
        )
        
        start_time = datetime.now()
        search.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
        
        logger.info(f"Best CV score ({scoring}): {best_cv_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
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
        'model_name': 'RandomForest',
        'task': task,
        'training_mode': 'tuned_params' if use_tuned_params else 'random_search',
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'validation_metrics': metrics,
        'train_time_seconds': train_time,
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'n_iter': n_iter if not use_tuned_params else None,
        'cv_folds': cv if not use_tuned_params else None,
        'n_jobs': n_jobs
    }
    
    return best_model, results


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    """Parse argomenti CLI."""
    parser = argparse.ArgumentParser(
        description='Training Random Forest per NIDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Parametri tuned (più recente)
  python src/training/random_forest.py --use-tuned-params
  
  # Config specifica
  python src/training/random_forest.py --use-tuned-params --tuning-config random_iter50_cv5_2026-01-24_20.02.json
  
  # Per timestamp
  python src/training/random_forest.py --use-tuned-params --tuning-timestamp 2026-01-24_20.02
  
  # Lista config
  python src/training/random_forest.py --list-configs
  
  # Random search
  python src/training/random_forest.py --n-iter 20 --cv 3
        """
    )
    
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                        default='binary', help='Tipo classificazione')
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Usa parametri da hyperparameter_tuning.py')
    parser.add_argument('--tuning-config', type=str, default=None,
                        help='File config specifico (default: più recente)')
    parser.add_argument('--tuning-timestamp', type=str, default=None,
                        help='Timestamp config da usare (es: 2026-01-24_20.02)')
    parser.add_argument('--list-configs', action='store_true',
                        help='Mostra configurazioni tuning disponibili ed esci')
    parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER,
                        help=f'Iterazioni random search (default: {DEFAULT_N_ITER})')
    parser.add_argument('--cv', type=int, default=DEFAULT_CV_FOLDS,
                        help=f'Fold CV (default: {DEFAULT_CV_FOLDS})')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Core CPU (default: auto)')
    parser.add_argument('--max-ram', type=int, default=DEFAULT_MAX_RAM,
                        help='Limite RAM %%')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                        help='Seed random')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funzione principale."""
    args = parse_arguments()
    
    if args.list_configs:
        print_available_configs('random_forest')
        return
    
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    timer = TimingLogger("training_random_forest", parameters={
        'task': args.task,
        'use_tuned_params': args.use_tuned_params,
        'n_iter': args.n_iter,
        'cv': args.cv,
        'n_jobs': n_jobs,
        'max_ram': args.max_ram,
        'random_state': args.random_state
    })
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:         {args.task}")
    
    tuned_params = None
    tuning_filepath = None
    
    if args.use_tuned_params:
        print(f"  Mode:         Tuned parameters")
        tuned_params, tuning_filepath = load_tuned_params(
            'random_forest',
            args.task,
            config_file=args.tuning_config,
            timestamp=args.tuning_timestamp
        )
        
        if not tuned_params:
            print("\n⚠️  Parametri tuned non trovati, uso random search")
            args.use_tuned_params = False
        else:
            print(f"  Config:       {tuning_filepath.name}")
    else:
        print(f"  Mode:         Random search")
        print(f"  N iter:       {args.n_iter}")
        print(f"  CV folds:     {args.cv}")
    
    print(f"  CPU cores:    {n_jobs}/{os.cpu_count()}")
    print(f"  Max RAM:      {args.max_ram}%")
    print()
    
    try:
        print("1. Caricamento dati...")
        with timer.time_operation("caricamento_dati"):
            train, val, test, mappings = load_processed_data()
        print(f"   Train: {len(train):,} | Val: {len(val):,}")
        
        try:
            print("\n2. Caricamento artifacts...")
            with timer.time_operation("preparazione_features"):
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
            print("\n2. Artifacts non trovati, eseguo feature engineering...")
            with timer.time_operation("feature_engineering"):
                X_train_final, X_val_final, _, y_train, y_val, _ = run_feature_engineering(
                    train, val, test, label_col=label_col
                )
        
        print(f"   Shape: {X_train_final.shape}")
        limiter.log_status(logger)
        
        print("\n3. Training Random Forest...")
        with timer.time_operation("training"):
            model, results = train_random_forest(
                X_train_final, y_train,
                X_val_final, y_val,
                task=args.task,
                n_iter=args.n_iter,
                cv=args.cv,
                n_jobs=n_jobs,
                random_state=args.random_state,
                use_tuned_params=args.use_tuned_params,
                tuned_params=tuned_params
            )
        
        print("\n4. Salvataggio modello...")
        with timer.time_operation("salvataggio"):
            if args.use_tuned_params and tuning_filepath:
                from src.model_versioning import save_tuned_model
                
                version_dir, version_id = save_tuned_model(
                    model=model,
                    results=results,
                    selected_features=selected_features,
                    model_type='random_forest',
                    tuning_filepath=tuning_filepath
                )
                
                model_path = version_dir / f"model_{args.task}.pkl"
                print(f"   Versione: {version_id}")
            else:
                output_dir = get_project_root() / "models" / "random_forest"
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
        
        timer.add_metric("train_samples", len(X_train_final))
        timer.add_metric("accuracy", results['validation_metrics'].get('accuracy', 0))
        timer.add_metric("f1", results['validation_metrics'].get('f1', 0))
        timing_path = timer.save()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO")
        print("=" * 60)
        print(f"\nMetriche validazione:")
        for name, value in results['validation_metrics'].items():
            print(f"  {name.capitalize():12}: {value:.4f}")
        print(f"\nTempo: {results['train_time_seconds']:.1f}s")
        print(f"Modello: {model_path}")
        print(f"Timing: {timing_path}")
        
        timer.print_summary()
        limiter.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima preprocessing e feature_engineering")
        sys.exit(1)


if __name__ == "__main__":
    main()