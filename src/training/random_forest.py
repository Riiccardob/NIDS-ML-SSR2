"""
================================================================================
NIDS-ML - Training Random Forest
================================================================================

Training di classificatore Random Forest per Network Intrusion Detection.

RANDOM FOREST:
--------------
Ensemble di alberi decisionali che:
- Addestra N alberi su sottoinsiemi random dei dati (bagging)
- Ogni albero vota per la classe predetta
- La predizione finale e il voto di maggioranza

Vantaggi:
- Robusto a outlier e rumore
- Non richiede scaling (ma lo usiamo per uniformita)
- Fornisce feature importance
- Parallelizzabile

Svantaggi:
- Piu lento in inference rispetto a gradient boosting
- Modelli piu grandi in memoria

GUIDA PARAMETRI:
----------------
    python src/training/random_forest.py [opzioni]

Opzioni disponibili:
    --task STR            'binary' o 'multiclass' (default: binary)
    --n-iter INT          Iterazioni random search (default: 20)
    --cv INT              Fold cross-validation (default: 3)
    --n-jobs INT          Core CPU (-1 = auto) (default: -1)
    --max-ram INT         Limite RAM % (default: 85)
    --random-state INT    Seed random (default: 42)

ESEMPI:
-------
# Training standard binario
python src/training/random_forest.py

# Training multiclasse
python src/training/random_forest.py --task multiclass

# Piu iterazioni per search migliore (piu lento)
python src/training/random_forest.py --n-iter 50 --cv 5

# Limita risorse
python src/training/random_forest.py --n-jobs 4 --max-ram 70

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)

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

# Spazio iperparametri per RandomizedSearchCV
# Valori scelti basandosi su best practices per classificazione tabular
PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300],           # Numero alberi
    'max_depth': [15, 20, 30, None],           # Profondita max (None = illimitata)
    'min_samples_split': [2, 5, 10],           # Min campioni per split
    'min_samples_leaf': [1, 2, 4],             # Min campioni per foglia
    'max_features': ['sqrt', 'log2', None],    # Feature per split
    'class_weight': ['balanced', 'balanced_subsample']  # Gestione sbilanciamento
}

DEFAULT_N_ITER = 20    # Combinazioni da testare
DEFAULT_CV_FOLDS = 3   # Fold cross-validation
DEFAULT_MAX_RAM = 85   # Limite RAM %


# ==============================================================================
# FUNZIONI TRAINING
# ==============================================================================

def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        task: str = 'binary',
                        n_iter: int = DEFAULT_N_ITER,
                        cv: int = DEFAULT_CV_FOLDS,
                        n_jobs: int = -1,
                        random_state: int = RANDOM_STATE
                        ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Addestra Random Forest con RandomizedSearchCV per ottimizzazione iperparametri.
    
    Processo:
    1. Definisce spazio iperparametri
    2. RandomizedSearchCV testa n_iter combinazioni random
    3. Cross-validation per ogni combinazione
    4. Seleziona modello con migliore score CV
    5. Valuta su validation set
    
    Args:
        X_train: Feature training (scalate e selezionate)
        y_train: Target training
        X_val: Feature validation
        y_val: Target validation
        task: 'binary' per 2 classi, 'multiclass' per N classi
        n_iter: Numero combinazioni iperparametri da testare
        cv: Numero fold per cross-validation
        n_jobs: Core CPU da usare (-1 = tutti - 1)
        random_state: Seed per reproducibilita
    
    Returns:
        Tuple (modello_migliore, dizionario_risultati)
    """
    logger.info("=" * 50)
    logger.info(f"TRAINING RANDOM FOREST ({task})")
    logger.info("=" * 50)
    logger.info(f"Train: {X_train.shape[0]:,} campioni, {X_train.shape[1]} feature")
    logger.info(f"Search: {n_iter} iter, {cv}-fold CV, {n_jobs} jobs")
    
    # Metrica per ottimizzazione
    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    
    # Modello base
    base_rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False  # Risparmia memoria
    )
    
    # Training
    logger.info("Avvio RandomizedSearchCV...")
    start_time = datetime.now()
    search.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Training completato in {train_time:.1f}s")
    logger.info(f"Best CV score ({scoring}): {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    # Modello migliore
    best_model = search.best_estimator_
    
    # Valutazione su validation
    y_val_pred = best_model.predict(X_val)
    
    # Calcola metriche
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
    
    # Risultati completi
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
        'cv_folds': cv
    }
    
    # Libera memoria
    del search
    gc.collect()
    
    return best_model, results


def save_model(model: RandomForestClassifier,
               results: Dict[str, Any],
               output_dir: Path = None) -> Path:
    """
    Salva modello addestrato e risultati.
    
    File salvati:
    - random_forest_{task}.pkl: Modello serializzato
    - random_forest_{task}_results.json: Metriche e parametri
    
    Args:
        model: Modello addestrato
        results: Dizionario risultati
        output_dir: Directory output (default: models/)
    
    Returns:
        Path al file modello salvato
    """
    if output_dir is None:
        output_dir = get_project_root() / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = results['task']
    model_path = output_dir / f"random_forest_{task}.pkl"
    results_path = output_dir / f"random_forest_{task}_results.json"
    
    # Salva modello
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")
    
    # Salva risultati
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
        description='Training Random Forest per NIDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python src/training/random_forest.py
  python src/training/random_forest.py --task multiclass
  python src/training/random_forest.py --n-iter 50 --cv 5
        """
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['binary', 'multiclass'],
        default='binary',
        help='Tipo classificazione (default: binary)'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=DEFAULT_N_ITER,
        help=f'Iterazioni random search (default: {DEFAULT_N_ITER})'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=DEFAULT_CV_FOLDS,
        help=f'Fold cross-validation (default: {DEFAULT_CV_FOLDS})'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Core CPU (-1 = auto)'
    )
    parser.add_argument(
        '--max-ram',
        type=int,
        default=DEFAULT_MAX_RAM,
        help=f'Limite RAM % (default: {DEFAULT_MAX_RAM})'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=RANDOM_STATE,
        help=f'Seed random (default: {RANDOM_STATE})'
    )
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funzione principale."""
    args = parse_arguments()
    
    # Configura risorse
    n_jobs = limit_cpu_cores() if args.n_jobs == -1 else args.n_jobs
    monitor = ResourceMonitor(max_ram=args.max_ram)
    
    # Label column basata su task
    label_col = 'Label_Binary' if args.task == 'binary' else 'Label_Multiclass'
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Task:         {args.task}")
    print(f"  N iter:       {args.n_iter}")
    print(f"  CV folds:     {args.cv}")
    print(f"  CPU cores:    {n_jobs}")
    print(f"  Max RAM:      {args.max_ram}%")
    print(f"  Random state: {args.random_state}")
    print()
    
    try:
        # 1. Carica dati
        print("1. Caricamento dati...")
        train, val, test, mappings = load_processed_data()
        print(f"   Train: {len(train):,} | Val: {len(val):,}")
        
        # 2. Prepara feature
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
            
            # Libera memoria
            del X_train, X_val, X_train_scaled, X_val_scaled
            gc.collect()
            
        except FileNotFoundError:
            print("\n2. Artifacts non trovati, eseguo feature engineering...")
            X_train_final, X_val_final, _, y_train, y_val, _ = run_feature_engineering(
                train, val, test, label_col=label_col
            )
        
        print(f"   Shape: {X_train_final.shape}")
        
        # Verifica risorse prima del training
        if not monitor.check_resources():
            logger.warning("Risorse alte, attendo...")
            monitor.wait_for_resources(timeout_seconds=120)
        monitor.log_status(logger)
        
        # 3. Training
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
        
        # 4. Salvataggio
        print("\n4. Salvataggio modello...")
        model_path = save_model(model, results)
        
        # Report finale
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO")
        print("=" * 60)
        print(f"\nMetriche validazione:")
        for name, value in results['validation_metrics'].items():
            print(f"  {name.capitalize():12}: {value:.4f}")
        print(f"\nTempo training: {results['train_time_seconds']:.1f}s")
        print(f"Modello: {model_path}")
        
        monitor.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima:")
        print("  python src/preprocessing.py")
        print("  python src/feature_engineering.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()