# """
# ================================================================================
# NIDS-ML - Training LightGBM
# ================================================================================

# GUIDA PARAMETRI:
# ----------------
#     python src/training/lightgbm_model.py [opzioni]

# Opzioni:
#     --task STR              'binary' o 'multiclass' (default: binary)
#     --n-iter INT            Iterazioni random search (default: 20)
#     --cv INT                Fold CV (default: 3)
#     --early-stopping        Abilita early stopping (default)
#     --no-early-stopping     Disabilita early stopping
#     --n-jobs INT            Core CPU (default: auto)

# ESEMPI:
# -------
# python src/training/lightgbm_model.py
# python src/training/lightgbm_model.py --n-iter 5 --cv 2  # Test veloce
# python src/training/lightgbm_model.py --n-jobs 4

# ================================================================================
# """

# # ==============================================================================
# # SETUP LIMITI CPU PRIMA DI ALTRI IMPORT
# # ==============================================================================
# import sys
# import os
# import argparse
# from pathlib import Path

# def _get_arg(name, default=None):
#     for i, arg in enumerate(sys.argv):
#         if arg == f'--{name}' and i + 1 < len(sys.argv):
#             try:
#                 return int(sys.argv[i + 1])
#             except ValueError:
#                 return sys.argv[i + 1]
#     return default

# _n_jobs_arg = _get_arg('n-jobs')
# _n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

# os.environ['OMP_NUM_THREADS'] = str(_n_cores)
# os.environ['MKL_NUM_THREADS'] = str(_n_cores)
# os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)
# os.environ['NUMEXPR_NUM_THREADS'] = str(_n_cores)
# os.environ['LOKY_MAX_CPU_COUNT'] = str(_n_cores)

# # Applica affinity CPU
# import psutil
# try:
#     p = psutil.Process()
#     p.cpu_affinity(list(range(_n_cores)))
#     p.nice(10)
# except Exception:
#     pass

# ROOT_DIR = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(ROOT_DIR))

# import pandas as pd
# import numpy as np
# from typing import Tuple, Dict, Any
# import joblib
# import json
# from datetime import datetime
# import gc

# from lightgbm import LGBMClassifier, early_stopping
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from src.utils import get_logger, get_project_root, RANDOM_STATE, ResourceLimiter, suppress_warnings
# from src.preprocessing import load_processed_data
# from src.feature_engineering import (
#     load_artifacts, get_feature_columns, prepare_xy,
#     transform_data, apply_feature_selection, run_feature_engineering
# )

# suppress_warnings()
# logger = get_logger(__name__)

# # ==============================================================================
# # CONFIGURAZIONE
# # ==============================================================================

# PARAM_DISTRIBUTIONS = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, 20, -1],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'num_leaves': [31, 50, 70, 100],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#     'min_child_samples': [10, 20, 30, 50],
#     'reg_alpha': [0, 0.01, 0.1],
#     'reg_lambda': [0, 0.01, 0.1]
# }

# DEFAULT_N_ITER = 20
# DEFAULT_CV_FOLDS = 3


# # ==============================================================================
# # TRAINING
# # ==============================================================================

# def train_lightgbm(X_train: pd.DataFrame,
#                    y_train: pd.Series,
#                    X_val: pd.DataFrame,
#                    y_val: pd.Series,
#                    task: str = 'binary',
#                    n_iter: int = DEFAULT_N_ITER,
#                    cv: int = DEFAULT_CV_FOLDS,
#                    use_early_stopping: bool = True,
#                    n_jobs: int = None,
#                    random_state: int = RANDOM_STATE
#                    ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
#     """Training LightGBM con RandomizedSearchCV."""
    
#     if n_jobs is None:
#         n_jobs = int(os.environ.get('OMP_NUM_THREADS', _n_cores))
    
#     logger.info("=" * 50)
#     logger.info(f"TRAINING LIGHTGBM ({task})")
#     logger.info("=" * 50)
#     logger.info(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
#     logger.info(f"Config: n_iter={n_iter}, cv={cv}, n_jobs={n_jobs}")
    
#     scoring = 'f1' if task == 'binary' else 'f1_weighted'
#     objective = 'binary' if task == 'binary' else 'multiclass'
    
#     base_params = {
#         'objective': objective,
#         'random_state': random_state,
#         'n_jobs': n_jobs,
#         'verbose': -1,
#         'class_weight': 'balanced'
#     }
    
#     if task == 'multiclass':
#         base_params['num_class'] = len(y_train.unique())
    
#     total_fits = n_iter * cv
#     print(f"\n   RandomizedSearchCV: {n_iter} x {cv} = {total_fits} fit totali")
#     print(f"   Attendere...\n")
    
#     base_lgbm = LGBMClassifier(**base_params)
    
#     search = RandomizedSearchCV(
#         estimator=base_lgbm,
#         param_distributions=PARAM_DISTRIBUTIONS,
#         n_iter=n_iter,
#         cv=cv,
#         scoring=scoring,
#         random_state=random_state,
#         n_jobs=n_jobs,
#         verbose=2,
#         return_train_score=False
#     )
    
#     start_time = datetime.now()
#     search.fit(X_train, y_train)
    
#     print(f"\n   Search completato")
#     logger.info(f"Best CV score: {search.best_score_:.4f}")
    
#     best_iteration = None
#     if use_early_stopping:
#         print("   Retraining con early stopping...")
#         final_params = {**base_params, **search.best_params_}
#         final_params['n_estimators'] = 500
        
#         best_model = LGBMClassifier(**final_params)
        
#         eval_metric = 'binary_logloss' if task == 'binary' else 'multi_logloss'
#         best_model.fit(
#             X_train, y_train,
#             eval_set=[(X_val, y_val)],
#             eval_metric=eval_metric,
#             callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
#         )
#         best_iteration = best_model.best_iteration_
#         print(f"   Early stopping a iterazione: {best_iteration}")
#     else:
#         best_model = search.best_estimator_
    
#     train_time = (datetime.now() - start_time).total_seconds()
    
#     y_val_pred = best_model.predict(X_val)
    
#     if task == 'binary':
#         metrics = {
#             'accuracy': float(accuracy_score(y_val, y_val_pred)),
#             'precision': float(precision_score(y_val, y_val_pred, zero_division=0)),
#             'recall': float(recall_score(y_val, y_val_pred, zero_division=0)),
#             'f1': float(f1_score(y_val, y_val_pred, zero_division=0))
#         }
#     else:
#         metrics = {
#             'accuracy': float(accuracy_score(y_val, y_val_pred)),
#             'precision_weighted': float(precision_score(y_val, y_val_pred, average='weighted', zero_division=0)),
#             'recall_weighted': float(recall_score(y_val, y_val_pred, average='weighted', zero_division=0)),
#             'f1_weighted': float(f1_score(y_val, y_val_pred, average='weighted', zero_division=0))
#         }
    
#     logger.info("\nMetriche Validation:")
#     for name, value in metrics.items():
#         logger.info(f"  {name}: {value:.4f}")
    
#     results = {
#         'model_name': 'LightGBM',
#         'task': task,
#         'best_params': search.best_params_,
#         'best_cv_score': float(search.best_score_),
#         'validation_metrics': metrics,
#         'train_time_seconds': train_time,
#         'train_samples': len(X_train),
#         'n_features': X_train.shape[1],
#         'early_stopping_used': use_early_stopping,
#         'best_iteration': best_iteration,
#         'n_jobs': n_jobs
#     }
    
#     del search
#     gc.collect()
    
#     return best_model, results


# def save_model(model, results, output_dir=None):
#     """Salva modello e risultati in models/lightgbm/."""
#     if output_dir is None:
#         output_dir = get_project_root() / "models" / "lightgbm"
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     task = results['task']
#     model_path = output_dir / f"model_{task}.pkl"
#     results_path = output_dir / f"results_{task}.json"
    
#     joblib.dump(model, model_path)
#     with open(results_path, 'w') as f:
#         json.dump(results, f, indent=2, default=str)
    
#     logger.info(f"Modello salvato: {model_path}")
#     return model_path


# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Training LightGBM per NIDS')
#     parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary')
#     parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER)
#     parser.add_argument('--cv', type=int, default=DEFAULT_CV_FOLDS)
#     parser.add_argument('--early-stopping', dest='early_stopping', action='store_true', default=True)
#     parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
#     parser.add_argument('--n-jobs', type=int, default=None)
#     parser.add_argument('--max-ram', type=int, default=85)
#     parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
#     return parser.parse_args()


# def main():
#     args = parse_arguments()
#     n_jobs = args.n_jobs if args.n_jobs else _n_cores
#     limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
#     label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
#     print("\n" + "=" * 60)
#     print("LIGHTGBM TRAINING")
#     print("=" * 60)
#     print(f"\nParametri:")
#     print(f"  Task:           {args.task}")
#     print(f"  N iter:         {args.n_iter}")
#     print(f"  CV folds:       {args.cv}")
#     print(f"  Early stopping: {args.early_stopping}")
#     print(f"  CPU cores:      {n_jobs}/{os.cpu_count()}")
#     print()
    
#     try:
#         print("1. Caricamento dati...")
#         train, val, test, _ = load_processed_data()
#         print(f"   Train: {len(train):,} | Val: {len(val):,}")
        
#         try:
#             print("\n2. Caricamento artifacts...")
#             scaler, selected_features, _ = load_artifacts()
            
#             feature_cols = get_feature_columns(train)
#             X_train, y_train = prepare_xy(train, label_col, feature_cols)
#             X_val, y_val = prepare_xy(val, label_col, feature_cols)
            
#             X_train_scaled = transform_data(X_train, scaler)
#             X_val_scaled = transform_data(X_val, scaler)
            
#             X_train_final = apply_feature_selection(X_train_scaled, selected_features)
#             X_val_final = apply_feature_selection(X_val_scaled, selected_features)
            
#             del X_train, X_val, X_train_scaled, X_val_scaled, train, val
#             gc.collect()
            
#         except FileNotFoundError:
#             print("\n2. Feature engineering...")
#             X_train_final, X_val_final, _, y_train, y_val, _ = run_feature_engineering(
#                 train, val, test, label_col=label_col
#             )
        
#         print(f"   Shape: {X_train_final.shape}")
#         limiter.log_status(logger)
        
#         print("\n3. Training LightGBM...")
#         model, results = train_lightgbm(
#             X_train_final, y_train, X_val_final, y_val,
#             task=args.task, n_iter=args.n_iter, cv=args.cv,
#             use_early_stopping=args.early_stopping,
#             n_jobs=n_jobs, random_state=args.random_state
#         )
        
#         print("\n4. Salvataggio...")
#         model_path = save_model(model, results)
        
#         print("\n" + "=" * 60)
#         print("TRAINING COMPLETATO")
#         print("=" * 60)
#         print(f"\nMetriche validazione:")
#         for name, value in results['validation_metrics'].items():
#             print(f"  {name.capitalize():12}: {value:.4f}")
#         print(f"\nTempo: {results['train_time_seconds']:.1f}s")
#         print(f"Modello: {model_path}")
        
#     except FileNotFoundError as e:
#         print(f"\nERRORE: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

"""
================================================================================
NIDS-ML - Training LightGBM
================================================================================

GUIDA PARAMETRI:
----------------
    python src/training/lightgbm_model.py [opzioni]

Opzioni:
    --task STR              'binary' o 'multiclass' (default: binary)
    --n-iter INT            Iterazioni random search (default: 20)
    --cv INT                Fold CV (default: 3)
    --early-stopping        Abilita early stopping (default)
    --no-early-stopping     Disabilita early stopping
    --n-jobs INT            Core CPU (default: auto)

ESEMPI:
-------
python src/training/lightgbm_model.py
python src/training/lightgbm_model.py --n-iter 5 --cv 2  # Test veloce
python src/training/lightgbm_model.py --n-jobs 4

================================================================================
"""

# ==============================================================================
# SETUP LIMITI CPU PRIMA DI ALTRI IMPORT
# ==============================================================================
import sys
import os
import argparse
from pathlib import Path

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

# Applica affinity CPU
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
from typing import Tuple, Dict, Any
import joblib
import json
from datetime import datetime
import gc

from lightgbm import LGBMClassifier, early_stopping
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

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70, 100],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [10, 20, 30, 50],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1]
}

DEFAULT_N_ITER = 20
DEFAULT_CV_FOLDS = 3


# ==============================================================================
# TRAINING
# ==============================================================================

def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   task: str = 'binary',
                   n_iter: int = DEFAULT_N_ITER,
                   cv: int = DEFAULT_CV_FOLDS,
                   use_early_stopping: bool = True,
                   n_jobs: int = None,
                   random_state: int = RANDOM_STATE
                   ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    """Training LightGBM con RandomizedSearchCV."""
    
    if n_jobs is None:
        n_jobs = int(os.environ.get('OMP_NUM_THREADS', _n_cores))
    
    logger.info("=" * 50)
    logger.info(f"TRAINING LIGHTGBM ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    logger.info(f"Config: n_iter={n_iter}, cv={cv}, n_jobs={n_jobs}")
    
    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    objective = 'binary' if task == 'binary' else 'multiclass'
    
    base_params = {
        'objective': objective,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbose': -1,
        'class_weight': 'balanced'
    }
    
    if task == 'multiclass':
        base_params['num_class'] = len(y_train.unique())
    
    total_fits = n_iter * cv
    print(f"\n   RandomizedSearchCV: {n_iter} x {cv} = {total_fits} fit totali")
    print(f"   Attendere...\n")
    
    base_lgbm = LGBMClassifier(**base_params)
    
    search = RandomizedSearchCV(
        estimator=base_lgbm,
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
    
    print(f"\n   Search completato")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    best_iteration = None
    if use_early_stopping:
        print("   Retraining con early stopping...")
        final_params = {**base_params, **search.best_params_}
        final_params['n_estimators'] = 500
        
        best_model = LGBMClassifier(**final_params)
        
        eval_metric = 'binary_logloss' if task == 'binary' else 'multi_logloss'
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
        )
        best_iteration = best_model.best_iteration_
        print(f"   Early stopping a iterazione: {best_iteration}")
    else:
        best_model = search.best_estimator_
    
    train_time = (datetime.now() - start_time).total_seconds()
    
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
        'model_name': 'LightGBM',
        'task': task,
        'best_params': search.best_params_,
        'best_cv_score': float(search.best_score_),
        'validation_metrics': metrics,
        'train_time_seconds': train_time,
        'train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'early_stopping_used': use_early_stopping,
        'best_iteration': best_iteration,
        'n_jobs': n_jobs
    }
    
    del search
    gc.collect()
    
    return best_model, results


def save_model(model, results, selected_features=None, output_dir=None):
    """Salva modello, risultati e feature list in models/lightgbm/."""
    if output_dir is None:
        output_dir = get_project_root() / "models" / "lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = results['task']
    model_path = output_dir / f"model_{task}.pkl"
    results_path = output_dir / f"results_{task}.json"
    
    joblib.dump(model, model_path)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Salva le feature usate per questo modello
    if selected_features is not None:
        features_path = output_dir / f"features_{task}.json"
        with open(features_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
        logger.info(f"Feature salvate: {features_path}")
    
    logger.info(f"Modello salvato: {model_path}")
    return model_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training LightGBM per NIDS')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER)
    parser.add_argument('--cv', type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true', default=True)
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--max-ram', type=int, default=85)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main():
    args = parse_arguments()
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    print("\n" + "=" * 60)
    print("LIGHTGBM TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:           {args.task}")
    print(f"  N iter:         {args.n_iter}")
    print(f"  CV folds:       {args.cv}")
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
            task=args.task, n_iter=args.n_iter, cv=args.cv,
            use_early_stopping=args.early_stopping,
            n_jobs=n_jobs, random_state=args.random_state
        )
        
        print("\n4. Salvataggio...")
        model_path = save_model(model, results, selected_features=selected_features)
        
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