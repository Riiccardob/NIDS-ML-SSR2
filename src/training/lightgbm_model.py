"""
================================================================================
NIDS-ML - Training LightGBM
================================================================================

Training di classificatore LightGBM per Network Intrusion Detection.

LIGHTGBM (Light Gradient Boosting Machine):
-------------------------------------------
Algoritmo di gradient boosting sviluppato da Microsoft che:
- Usa histogram-based learning (discretizza valori continui)
- Crescita leaf-wise invece di level-wise (piu efficiente)
- Supporta parallel e GPU training
- Ottimizzato per grandi dataset

Vantaggi:
- Piu veloce di XGBoost su dataset grandi (>100k righe)
- Minor consumo memoria grazie a histogram binning
- Gestisce bene feature categoriche
- Early stopping integrato

Svantaggi:
- Puo overfittare su dataset piccoli
- Sensibile a num_leaves (parametro critico)

GUIDA PARAMETRI:
----------------
    python src/training/lightgbm_model.py [opzioni]

Opzioni disponibili:
    --task STR              'binary' o 'multiclass' (default: binary)
    --n-iter INT            Iterazioni random search (default: 20)
    --cv INT                Fold cross-validation (default: 3)
    --early-stopping        Abilita early stopping (default)
    --no-early-stopping     Disabilita early stopping
    --n-jobs INT            Core CPU (-1 = auto)
    --max-ram INT           Limite RAM %
    --random-state INT      Seed random

ESEMPI:
-------
# Training standard
python src/training/lightgbm_model.py

# Training multiclasse
python src/training/lightgbm_model.py --task multiclass

# Piu iterazioni, senza early stopping
python src/training/lightgbm_model.py --n-iter 50 --no-early-stopping

================================================================================
"""

import sys
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import joblib
import json
from datetime import datetime
import gc

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    ResourceMonitor,
    limit_cpu_cores,
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
# CONFIGURAZIONE DEFAULT
# ==============================================================================

# Spazio iperparametri per LightGBM
PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300],           # Numero boosting rounds
    'max_depth': [5, 10, 15, 20, -1],          # -1 = nessun limite
    'learning_rate': [0.01, 0.05, 0.1],        # Shrinkage rate
    'num_leaves': [31, 50, 70, 100],           # Max foglie per albero (critico!)
    'subsample': [0.7, 0.8, 0.9, 1.0],         # Frazione campioni
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # Frazione feature
    'min_child_samples': [10, 20, 30, 50],     # Min campioni per foglia
    'reg_alpha': [0, 0.01, 0.1],               # L1 regularization
    'reg_lambda': [0, 0.01, 0.1]               # L2 regularization
}

DEFAULT_N_ITER = 20
DEFAULT_CV_FOLDS = 3
DEFAULT_MAX_RAM = 85
DEFAULT_EARLY_STOPPING = True


# ==============================================================================
# FUNZIONI TRAINING
# ==============================================================================

def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   task: str = 'binary',
                   n_iter: int = DEFAULT_N_ITER,
                   cv: int = DEFAULT_CV_FOLDS,
                   use_early_stopping: bool = DEFAULT_EARLY_STOPPING,
                   n_jobs: int = -1,
                   random_state: int = RANDOM_STATE
                   ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    """
    Addestra LightGBM con RandomizedSearchCV e early stopping opzionale.
    
    Args:
        X_train: Feature training
        y_train: Target training
        X_val: Feature validation
        y_val: Target validation
        task: 'binary' o 'multiclass'
        n_iter: Combinazioni da testare
        cv: Fold cross-validation
        use_early_stopping: Abilita early stopping
        n_jobs: Core CPU
        random_state: Seed
    
    Returns:
        Tuple (modello, risultati)
    """
    logger.info("=" * 50)
    logger.info(f"TRAINING LIGHTGBM ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} campioni, {X_train.shape[1]} feature")
    logger.info(f"Early stopping: {use_early_stopping}")
    
    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    objective = 'binary' if task == 'binary' else 'multiclass'
    
    # Parametri base
    base_params = {
        'objective': objective,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbose': -1,
        'class_weight': 'balanced'  # Gestisce sbilanciamento
    }
    
    if task == 'multiclass':
        n_classes = len(y_train.unique())
        base_params['num_class'] = n_classes
    
    base_lgbm = LGBMClassifier(**base_params)
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_lgbm,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False
    )
    
    logger.info("Avvio RandomizedSearchCV...")
    start_time = datetime.now()
    search.fit(X_train, y_train)
    
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    # Early stopping
    best_iteration = None
    if use_early_stopping:
        logger.info("Retraining con early stopping...")
        final_params = {**base_params, **search.best_params_}
        final_params['n_estimators'] = 500
        
        best_model = LGBMClassifier(**final_params)
        
        eval_metric = 'binary_logloss' if task == 'binary' else 'multi_logloss'
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[
                # Early stopping callback
                __import__('lightgbm').early_stopping(stopping_rounds=20, verbose=False)
            ]
        )
        best_iteration = best_model.best_iteration_
        logger.info(f"Early stopping a iterazione: {best_iteration}")
    else:
        best_model = search.best_estimator_
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Valutazione
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
            'precision_weighted': float(precision_score(y_val, y_val_pred,
                                                        average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_val, y_val_pred,
                                                   average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_val, y_val_pred,
                                          average='weighted', zero_division=0))
        }
    
    logger.info("\nMetriche Validation:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    results = {
        'model_name': 'LightGBM',
        'task': task,
        'best_params': search.best_params_,
        'best_cv_score': float(search.best_score_),
        'validation_metrics': metrics,
        'train_time_seconds': train_time,
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'early_stopping_used': use_early_stopping,
        'best_iteration': best_iteration
    }
    
    del search
    gc.collect()
    
    return best_model, results


def save_model(model: LGBMClassifier,
               results: Dict[str, Any],
               output_dir: Path = None) -> Path:
    """Salva modello e risultati."""
    if output_dir is None:
        output_dir = get_project_root() / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = results['task']
    model_path = output_dir / f"lightgbm_{task}.pkl"
    results_path = output_dir / f"lightgbm_{task}_results.json"
    
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Risultati salvati: {results_path}")
    
    return model_path


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    """Parse argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description='Training LightGBM per NIDS',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                        default='binary', help='Tipo classificazione')
    parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER,
                        help=f'Iterazioni random search (default: {DEFAULT_N_ITER})')
    parser.add_argument('--cv', type=int, default=DEFAULT_CV_FOLDS,
                        help=f'Fold CV (default: {DEFAULT_CV_FOLDS})')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true',
                        default=True, help='Abilita early stopping (default)')
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false',
                        help='Disabilita early stopping')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Core CPU')
    parser.add_argument('--max-ram', type=int, default=DEFAULT_MAX_RAM, help='Limite RAM %')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE, help='Seed')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funzione principale."""
    args = parse_arguments()
    
    n_jobs = limit_cpu_cores() if args.n_jobs == -1 else args.n_jobs
    monitor = ResourceMonitor(max_ram=args.max_ram)
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    print("\n" + "=" * 60)
    print("LIGHTGBM TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:           {args.task}")
    print(f"  N iter:         {args.n_iter}")
    print(f"  CV folds:       {args.cv}")
    print(f"  Early stopping: {args.early_stopping}")
    print(f"  CPU cores:      {n_jobs}")
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
            
            del X_train, X_val, X_train_scaled, X_val_scaled
            gc.collect()
            
        except FileNotFoundError:
            print("\n2. Artifacts non trovati, eseguo feature engineering...")
            X_train_final, X_val_final, _, y_train, y_val, _ = run_feature_engineering(
                train, val, test, label_col=label_col
            )
        
        print(f"   Shape: {X_train_final.shape}")
        
        if not monitor.check_resources():
            monitor.wait_for_resources(timeout_seconds=120)
        monitor.log_status(logger)
        
        print("\n3. Training LightGBM...")
        model, results = train_lightgbm(
            X_train_final, y_train,
            X_val_final, y_val,
            task=args.task,
            n_iter=args.n_iter,
            cv=args.cv,
            use_early_stopping=args.early_stopping,
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
        print(f"\nTempo training: {results['train_time_seconds']:.1f}s")
        print(f"Modello: {model_path}")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima preprocessing e feature_engineering")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()