"""
Training modelli per NIDS NetFlow-based con supporto per grandi dataset.

Supporta XGBoost, LightGBM e Random Forest con chunk-based data loading.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR,
    DEFAULT_MODEL, XGBOOST_PARAMS, RF_PARAMS, LIGHTGBM_PARAMS, 
    RANDOM_STATE, CHUNK_SIZE
)
from utils import setup_logger, compute_metrics, print_metrics, get_memory_usage
from feature_engineering import load_artifacts


logger = setup_logger(__name__, 'training.log')


class ParquetDataLoader:
    """
    Data loader per Parquet files con chunk-based loading.
    Utile per validazione e test su grandi dataset.
    """
    
    def __init__(self, parquet_path: Path, feature_cols: list, chunk_size: int = CHUNK_SIZE):
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.chunk_size = chunk_size
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.total_rows = self.parquet_file.metadata.num_rows
    
    def iter_batches(self):
        """Itera sui batch del Parquet file."""
        for batch in self.parquet_file.iter_batches(batch_size=self.chunk_size):
            df_batch = batch.to_pandas()
            X = df_batch[self.feature_cols].values
            y = df_batch['Label_Binary'].values
            yield X, y
    
    def get_total_rows(self):
        """Ritorna numero totale di righe."""
        return self.total_rows


def load_train_data_in_memory(train_path: Path, feature_cols: list) -> tuple:
    """
    Carica train set in memoria per training.
    
    Per dataset molto grandi (>10M samples), considera di usare:
    - XGBoost con external memory
    - LightGBM con sample (già ottimizzato per grandi dataset)
    - Subsampling strategico
    """
    
    logger.info(f"\nCaricamento train set in memoria...")
    logger.info(f"  Source: {train_path.name}")
    
    # Verifica dimensione
    parquet_file = pq.ParquetFile(train_path)
    total_rows = parquet_file.metadata.num_rows
    
    logger.info(f"  Total rows: {total_rows:,}")
    
    # Stima memoria richiesta (circa 8 bytes per float64 * n_features * n_rows)
    estimated_mb = (len(feature_cols) * total_rows * 8) / (1024**2)
    logger.info(f"  Memoria stimata: ~{estimated_mb:.0f} MB")
    
    current_mem = get_memory_usage()
    logger.info(f"  RAM corrente: {current_mem:.1f}%")
    
    # Se dataset troppo grande, avvisa
    if estimated_mb > 4000:  # >4GB
        logger.warning(f"\n    ATTENZIONE: Dataset grande ({estimated_mb:.0f} MB)")
        logger.warning(f"  Potrebbe causare problemi di memoria durante il training")
        logger.warning(f"  Considera di usare:")
        logger.warning(f"    - XGBoost con tree_method='hist' e max_bin ridotto")
        logger.warning(f"    - LightGBM (già ottimizzato)")
        logger.warning(f"    - Subsampling del train set\n")
    
    # Carica in chunk e concatena
    chunks = []
    
    logger.info(f"  Caricamento chunk-based...")
    
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
        df_batch = batch.to_pandas()
        chunks.append(df_batch)
        
        if (batch_idx + 1) % 20 == 0:
            progress = ((batch_idx + 1) * CHUNK_SIZE / total_rows) * 100
            logger.info(f"    Chunk {batch_idx + 1} - {min(progress, 100):.1f}%")
    
    # Concatena
    logger.info(f"  Concatenazione chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    # Libera memoria
    del chunks
    
    logger.info(f"   Train set caricato: {len(df):,} righe")
    logger.info(f"  RAM dopo caricamento: {get_memory_usage():.1f}%")
    
    # Separa X, y
    X = df[feature_cols].values
    y = df['Label_Binary'].values
    
    del df
    
    return X, y


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  val_loader: ParquetDataLoader) -> XGBClassifier:
    """
    Training XGBoost con early stopping su validation set.
    
    XGBoost con tree_method='hist' è ottimizzato per grandi dataset.
    """
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBOOST")
    logger.info("="*70)
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  tree_method: {XGBOOST_PARAMS.get('tree_method', 'auto')}")
    logger.info(f"  max_bin: {XGBOOST_PARAMS.get('max_bin', 256)}")
    
    # Carica validation set per early stopping
    logger.info(f"\n  Caricamento validation set per early stopping...")
    X_val_chunks = []
    y_val_chunks = []
    
    for X_batch, y_batch in val_loader.iter_batches():
        X_val_chunks.append(X_batch)
        y_val_chunks.append(y_batch)
    
    X_val = np.vstack(X_val_chunks)
    y_val = np.concatenate(y_val_chunks)
    
    del X_val_chunks, y_val_chunks
    
    logger.info(f"  Validation samples: {len(X_val):,}")
    
    # Setup modello
    model = XGBClassifier(**XGBOOST_PARAMS)
    
    # Train con early stopping
    logger.info(f"\n  Training in corso...")
    start_time = datetime.now()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else None
    logger.info(f"\n   Training completato in {training_time:.1f}s")
    logger.info(f"  Best iteration: {best_iteration}")
    
    del X_val, y_val
    
    return model


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   val_loader: ParquetDataLoader) -> LGBMClassifier:
    """
    Training LightGBM con early stopping.
    
    LightGBM è già altamente ottimizzato per grandi dataset.
    """
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING LIGHTGBM")
    logger.info("="*70)
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  max_bin: {LIGHTGBM_PARAMS.get('max_bin', 255)}")
    
    # Carica validation set
    logger.info(f"\n  Caricamento validation set per early stopping...")
    X_val_chunks = []
    y_val_chunks = []
    
    for X_batch, y_batch in val_loader.iter_batches():
        X_val_chunks.append(X_batch)
        y_val_chunks.append(y_batch)
    
    X_val = np.vstack(X_val_chunks)
    y_val = np.concatenate(y_val_chunks)
    
    del X_val_chunks, y_val_chunks
    
    logger.info(f"  Validation samples: {len(X_val):,}")
    
    # Setup modello
    model = LGBMClassifier(**LIGHTGBM_PARAMS)
    
    # Train
    logger.info(f"\n  Training in corso...")
    start_time = datetime.now()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[]
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else None
    logger.info(f"\n   Training completato in {training_time:.1f}s")
    logger.info(f"  Best iteration: {best_iteration}")
    
    del X_val, y_val
    
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Training Random Forest.
    
    NOTA: RF può essere lento su dataset molto grandi (>10M samples).
    """
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("="*70)
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  n_estimators: {RF_PARAMS.get('n_estimators', 100)}")
    logger.info(f"  max_depth: {RF_PARAMS.get('max_depth', 10)}")
    
    if len(X_train) > 10_000_000:
        logger.warning(f"\n    ATTENZIONE: Random Forest può essere lento su >10M samples")
        logger.warning(f"  Considera di usare XGBoost o LightGBM per performance migliori\n")
    
    # Setup modello
    model = RandomForestClassifier(**RF_PARAMS)
    
    # Train
    logger.info(f"\n  Training in corso (può richiedere tempo)...")
    start_time = datetime.now()
    
    model.fit(X_train, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\n   Training completato in {training_time:.1f}s")
    
    return model


def evaluate_model_on_loader(model, data_loader: ParquetDataLoader, dataset_name: str) -> dict:
    """
    Valuta modello usando data loader chunk-based.
    """
    
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATION: {dataset_name}")
    logger.info(f"{'='*70}")
    
    total_samples = data_loader.get_total_rows()
    logger.info(f"  Total samples: {total_samples:,}")
    
    # Predict chunk-by-chunk
    y_true_all = []
    y_pred_all = []
    
    samples_processed = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(data_loader.iter_batches()):
        
        # Predict
        y_pred_batch = model.predict(X_batch)
        
        y_true_all.append(y_batch)
        y_pred_all.append(y_pred_batch)
        
        samples_processed += len(y_batch)
        
        # Log progresso
        if (batch_idx + 1) % 20 == 0:
            progress = (samples_processed / total_samples) * 100
            logger.info(f"  Batch {batch_idx + 1} - {progress:.1f}%")
    
    # Concatena predictions
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Print
    print_metrics(metrics, f"{dataset_name} Metrics")
    
    return metrics


def save_model(model, model_type: str, metrics: dict, training_time: float) -> Path:
    """Salva modello e metadata."""
    
    # Create model directory
    model_dir = MODELS_DIR / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"\n   Modello salvato: {model_path}")
    
    # Save metrics
    import json
    metrics_path = model_dir / "metrics.json"
    
    metadata = {
        'model_type': model_type,
        'trained_at': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'metrics': metrics,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   Metrics salvate: {metrics_path}")
    
    return model_path


def main(model_type: str = None):
    """Pipeline training completa con chunk-based data loading."""
    
    if model_type is None:
        model_type = DEFAULT_MODEL
    
    logger.info("="*70)
    logger.info(f"TRAINING - {model_type.upper()} (CHUNK-BASED)")
    logger.info("="*70)
    logger.info(f"RAM iniziale: {get_memory_usage():.1f}%")
    logger.info("="*70)
    
    # ========================================================================
    # STEP 1: Load artifacts
    # ========================================================================
    
    logger.info("\nSTEP 1: Caricamento artifacts")
    
    _, feature_cols = load_artifacts()
    
    logger.info(f"   Feature caricate: {len(feature_cols)}")
    
    # ========================================================================
    # STEP 2: Setup data loaders
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 2: SETUP DATA LOADERS")
    logger.info("="*70)
    
    train_path = PROCESSED_DATA_DIR / "train_scaled.parquet"
    val_path = PROCESSED_DATA_DIR / "val_scaled.parquet"
    test_path = PROCESSED_DATA_DIR / "test_scaled.parquet"
    
    # Verifica esistenza file
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
    
    # Setup loaders
    val_loader = ParquetDataLoader(val_path, feature_cols)
    test_loader = ParquetDataLoader(test_path, feature_cols)
    
    logger.info(f"   Validation loader: {val_loader.get_total_rows():,} samples")
    logger.info(f"   Test loader: {test_loader.get_total_rows():,} samples")
    
    # ========================================================================
    # STEP 3: Load training data
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 3: CARICAMENTO TRAIN DATA")
    logger.info("="*70)
    
    X_train, y_train = load_train_data_in_memory(train_path, feature_cols)
    
    logger.info(f"\n   Train data in memoria")
    logger.info(f"  Shape: {X_train.shape}")
    logger.info(f"  RAM dopo load: {get_memory_usage():.1f}%")
    
    # ========================================================================
    # STEP 4: Train model
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 4: TRAINING")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    if model_type == 'xgboost':
        model = train_xgboost(X_train, y_train, val_loader)
    elif model_type == 'lightgbm':
        model = train_lightgbm(X_train, y_train, val_loader)
    elif model_type == 'random_forest':
        model = train_random_forest(X_train, y_train)
    else:
        raise ValueError(f"Modello non supportato: {model_type}")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Libera memoria train data
    del X_train, y_train
    
    logger.info(f"\n   Training completato: {training_time:.1f}s")
    logger.info(f"  RAM dopo training: {get_memory_usage():.1f}%")
    
    # ========================================================================
    # STEP 5: Evaluate
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 5: EVALUATION")
    logger.info("="*70)
    
    # Validation
    val_metrics = evaluate_model_on_loader(model, val_loader, "VALIDATION")
    
    # Test
    test_metrics = evaluate_model_on_loader(model, test_loader, "TEST")
    
    # ========================================================================
    # STEP 6: Save model
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 6: SALVATAGGIO MODELLO")
    logger.info("="*70)
    
    all_metrics = {
        'validation': val_metrics,
        'test': test_metrics,
    }
    
    model_path = save_model(model, model_type, all_metrics, training_time)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info(" TRAINING COMPLETATO")
    logger.info("="*70)
    logger.info(f"Modello: {model_type}")
    logger.info(f"Salvato in: {model_path}")
    logger.info(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    logger.info(f"\nPERFORMANCE TEST SET:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"  FPR:       {test_metrics['fpr']:.4f}")
    logger.info(f"  FNR:       {test_metrics['fnr']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Training modelli NIDS con chunk-based processing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['xgboost', 'random_forest', 'lightgbm'],
        default=DEFAULT_MODEL,
        help=f'Tipo modello da trainare (default: {DEFAULT_MODEL})'
    )
    
    args = parser.parse_args()
    
    main(args.model)
