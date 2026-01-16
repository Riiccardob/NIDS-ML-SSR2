"""
================================================================================
NIDS-ML - Training Random Forest
================================================================================

GUIDA PARAMETRI:
----------------
    python src/training/random_forest.py [opzioni]

Opzioni disponibili:
    --task STR            'binary' o 'multiclass' (default: binary)
    --n-iter INT          Iterazioni random search (default: 20)
    --cv INT              Fold cross-validation (default: 3)
    --n-jobs INT          Core CPU (default: auto, totale - 2)
    --max-ram INT         Limite RAM %
    --random-state INT    Seed random

ESEMPI:
-------
# Training standard
python src/training/random_forest.py

# Test veloce (poche iterazioni)
python src/training/random_forest.py --n-iter 5 --cv 2

# Limita a 4 core
python src/training/random_forest.py --n-jobs 4

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

# Configura limiti PRIMA di importare sklearn
_n_jobs_arg = _get_arg('n-jobs')
_n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

os.environ['OMP_NUM_THREADS'] = str(_n_cores)
os.environ['MKL_NUM_THREADS'] = str(_n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(_n_cores)
os.environ['LOKY_MAX_CPU_COUNT'] = str(_n_cores)

# Applica affinity CPU
import psutil
try:
    p = psutil.Process()
    p.cpu_affinity(list(range(_n_cores)))
    p.nice(10)
except Exception:
    pass

# Ora importa il resto
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
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
                        random_state: int = RANDOM_STATE
                        ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Training Random Forest con RandomizedSearchCV e progress bar.
    """
    if n_jobs is None:
        n_jobs = int(os.environ.get('OMP_NUM_THREADS', _n_cores))
    
    logger.info("=" * 50)
    logger.info(f"TRAINING RANDOM FOREST ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    logger.info(f"Config: n_iter={n_iter}, cv={cv}, n_jobs={n_jobs}")
    
    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    
    # Calcola totale fit per progress
    total_fits = n_iter * cv
    print(f"\n   RandomizedSearchCV: {n_iter} combinazioni x {cv} fold = {total_fits} fit totali")
    print(f"   Questo richiede tempo, attendere...\n")
    
    base_rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )
    
    # RandomizedSearchCV con verbose per vedere progresso
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2,  # Mostra progresso dettagliato
        return_train_score=False
    )
    
    start_time = datetime.now()
    search.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n   Search completato in {train_time:.1f}s")
    logger.info(f"Best CV score ({scoring}): {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    best_model = search.best_estimator_
    
    # Valutazione
    print("   Valutazione su validation set...")
    y_val_pred = best_model.predict(X_val)
    
    if task == 'binary':
        metrics = {
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision': float(precision_score(y_val, y_val_pred, zero_division=0)),
            'recall': float(recall_score(y_val, y_val_pred, zero_division=0)),
            'f1': float(f1_score(y_val, y_val_pred, zero_division=0))
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
        'best_params': search.best_params_,
        'best_cv_score': float(search.best_score_),
        'validation_metrics': metrics,
        'train_time_seconds': train_time,
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'n_iter': n_iter,
        'cv_folds': cv,
        'n_jobs': n_jobs
    }
    
    del search
    gc.collect()
    
    return best_model, results


def save_model(model: RandomForestClassifier,
               results: Dict[str, Any],
               output_dir: Path = None) -> Path:
    """Salva modello e risultati in models/random_forest/."""
    if output_dir is None:
        output_dir = get_project_root() / "models" / "random_forest"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = results['task']
    model_path = output_dir / f"model_{task}.pkl"
    results_path = output_dir / f"results_{task}.json"
    
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return model_path


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
  python src/training/random_forest.py
  python src/training/random_forest.py --n-iter 5 --cv 2  # Test veloce
  python src/training/random_forest.py --n-jobs 4
        """
    )
    
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                        default='binary', help='Tipo classificazione')
    parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER,
                        help=f'Iterazioni random search (default: {DEFAULT_N_ITER})')
    parser.add_argument('--cv', type=int, default=DEFAULT_CV_FOLDS,
                        help=f'Fold CV (default: {DEFAULT_CV_FOLDS})')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Core CPU (default: auto)')
    parser.add_argument('--max-ram', type=int, default=DEFAULT_MAX_RAM,
                        help='Limite RAM %')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                        help='Seed random')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funzione principale."""
    args = parse_arguments()
    
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:         {args.task}")
    print(f"  N iter:       {args.n_iter}")
    print(f"  CV folds:     {args.cv}")
    print(f"  CPU cores:    {n_jobs}/{os.cpu_count()}")
    print(f"  Max RAM:      {args.max_ram}%")
    print()
    
    try:
        print("1. Caricamento dati...")
        train, val, test, mappings = load_processed_data()
        print(f"   Train: {len(train):,} | Val: {len(val):,}")
        
        try:
            print("\n2. Caricamento artifacts...")
            scaler, selected_features, _ = load_artifacts()
            
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
            X_train_final, X_val_final, _, y_train, y_val, _ = run_feature_engineering(
                train, val, test, label_col=label_col
            )
        
        print(f"   Shape: {X_train_final.shape}")
        limiter.log_status(logger)
        
        print("\n3. Training Random Forest...")
        model, results = train_random_forest(
            X_train_final, y_train,
            X_val_final, y_val,
            task=args.task,
            n_iter=args.n_iter,
            cv=args.cv,
            n_jobs=n_jobs,
            random_state=args.random_state
        )
        
        print("\n4. Salvataggio modello...")
        model_path = save_model(model, results)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO")
        print("=" * 60)
        print(f"\nMetriche validazione:")
        for name, value in results['validation_metrics'].items():
            print(f"  {name.capitalize():12}: {value:.4f}")
        print(f"\nTempo: {results['train_time_seconds']:.1f}s")
        print(f"Modello: {model_path}")
        
        limiter.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima preprocessing e feature_engineering")
        sys.exit(1)


if __name__ == "__main__":
    main()