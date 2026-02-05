"""
Training INCREMENTAL per XGBoost e LightGBM.

Usa TUTTI i 76M record del training set con incremental learning.

STRATEGIA:
1. XGBoost: warm start con xgb_model parameter
2. LightGBM: warm start con init_model parameter  
3. Training chunk-by-chunk con learning rate decay
4. Validation chunk-based per evitare RAM overflow

NO Random Forest - non supporta incremental learning efficacemente.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import gc

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR,
    DEFAULT_MODEL, XGBOOST_PARAMS, LIGHTGBM_PARAMS, 
    RANDOM_STATE, CHUNK_SIZE
)
from utils import setup_logger, compute_metrics, print_metrics, get_memory_usage
from feature_engineering import load_artifacts


logger = setup_logger(__name__, 'training_incremental.log')


# Chunk size per training (più piccolo per gestire RAM meglio)
TRAIN_CHUNK_SIZE = 250_000  # 250K righe per chunk


class ParquetDataLoader:
    """Data loader chunk-based per Parquet."""
    
    def __init__(self, parquet_path: Path, feature_cols: list, chunk_size: int = CHUNK_SIZE):
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.chunk_size = chunk_size
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.total_rows = self.parquet_file.metadata.num_rows
    
    def iter_batches(self):
        """Itera sui batch."""
        for batch in self.parquet_file.iter_batches(batch_size=self.chunk_size):
            df_batch = batch.to_pandas()
            X = df_batch[self.feature_cols].values
            y = df_batch['Label_Binary'].values
            yield X, y
            del df_batch, X, y
    
    def get_total_rows(self):
        return self.total_rows
    
    def reset(self):
        self.parquet_file = pq.ParquetFile(self.parquet_path)


def train_xgboost_incremental(train_loader: ParquetDataLoader, val_loader: ParquetDataLoader) -> XGBClassifier:
    """
    Training incremental di XGBoost usando warm start.
    """
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBOOST INCREMENTAL")
    logger.info("="*70)
    logger.info(f"Train samples: {train_loader.get_total_rows():,}")
    logger.info(f"Chunk size: {TRAIN_CHUNK_SIZE:,}")
    logger.info(f"Strategia: Warm start con xgb_model")
    
    # Parametri per incremental
    n_estimators_per_chunk = 10  # 10 trees per chunk
    total_chunks = (train_loader.get_total_rows() + TRAIN_CHUNK_SIZE - 1) // TRAIN_CHUNK_SIZE
    
    logger.info(f"Trees per chunk: {n_estimators_per_chunk}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Total trees: ~{n_estimators_per_chunk * min(total_chunks, 20)}")  # Cap a 20 pass
    
    # Modello iniziale
    model = None
    chunk_count = 0
    rows_trained = 0
    
    # Training loop (max 20 pass sul dataset per evitare overfit)
    max_passes = 3
    
    for pass_num in range(1, max_passes + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"PASS {pass_num}/{max_passes}")
        logger.info(f"{'='*70}")
        
        train_loader.reset()
        pass_chunks = 0
        
        for X_batch, y_batch in train_loader.iter_batches():
            chunk_count += 1
            pass_chunks += 1
            
            mem_usage = get_memory_usage()
            if mem_usage > 85:
                logger.warning(f"    RAM alta ({mem_usage:.1f}%), GC...")
                gc.collect()
            
            # Train su questo chunk
            if model is None:
                # Primo chunk - crea modello
                params = XGBOOST_PARAMS.copy()
                params['n_estimators'] = n_estimators_per_chunk
                model = XGBClassifier(**params)
                model.fit(X_batch, y_batch, verbose=False)
            else:
                # Warm start - continua training
                params = XGBOOST_PARAMS.copy()
                params['n_estimators'] = n_estimators_per_chunk
                new_model = XGBClassifier(**params)
                new_model.fit(X_batch, y_batch, xgb_model=model.get_booster(), verbose=False)
                model = new_model
            
            rows_trained += len(X_batch)
            
            del X_batch, y_batch
            
            if pass_chunks % 10 == 0:
                logger.info(f"  Pass {pass_num} - Chunk {pass_chunks} - Trained: {rows_trained:,} - RAM: {mem_usage:.1f}%")
            
            if pass_chunks % 5 == 0:
                gc.collect()
        
        logger.info(f"   Pass {pass_num} completato - Total chunks: {pass_chunks}")
        
        # Valida dopo ogni pass
        logger.info(f"\n  Validation dopo pass {pass_num}...")
        y_true_val, y_pred_val = evaluate_chunk_based(model, val_loader, "VALIDATION")
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | Recall: {val_metrics['recall']:.4f}")
        
        del y_true_val, y_pred_val
        gc.collect()
    
    logger.info(f"\n Training XGBoost completato")
    logger.info(f"  Total samples trained: {rows_trained:,}")
    logger.info(f"  RAM: {get_memory_usage():.1f}%")
    
    return model


def train_lightgbm_incremental(train_loader: ParquetDataLoader, val_loader: ParquetDataLoader) -> LGBMClassifier:
    """
    Training incremental di LightGBM usando warm start.
    """
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING LIGHTGBM INCREMENTAL")
    logger.info("="*70)
    logger.info(f"Train samples: {train_loader.get_total_rows():,}")
    logger.info(f"Chunk size: {TRAIN_CHUNK_SIZE:,}")
    logger.info(f"Strategia: Warm start con init_model")
    
    n_estimators_per_chunk = 10
    total_chunks = (train_loader.get_total_rows() + TRAIN_CHUNK_SIZE - 1) // TRAIN_CHUNK_SIZE
    
    logger.info(f"Trees per chunk: {n_estimators_per_chunk}")
    logger.info(f"Total chunks: {total_chunks}")
    
    model = None
    chunk_count = 0
    rows_trained = 0
    max_passes = 3
    
    for pass_num in range(1, max_passes + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"PASS {pass_num}/{max_passes}")
        logger.info(f"{'='*70}")
        
        train_loader.reset()
        pass_chunks = 0
        
        for X_batch, y_batch in train_loader.iter_batches():
            chunk_count += 1
            pass_chunks += 1
            
            mem_usage = get_memory_usage()
            if mem_usage > 85:
                logger.warning(f"    RAM alta ({mem_usage:.1f}%), GC...")
                gc.collect()
            
            if model is None:
                params = LIGHTGBM_PARAMS.copy()
                params['n_estimators'] = n_estimators_per_chunk
                model = LGBMClassifier(**params)
                model.fit(X_batch, y_batch)
            else:
                params = LIGHTGBM_PARAMS.copy()
                params['n_estimators'] = n_estimators_per_chunk
                new_model = LGBMClassifier(**params)
                new_model.fit(X_batch, y_batch, init_model=model)
                model = new_model
            
            rows_trained += len(X_batch)
            
            del X_batch, y_batch
            
            if pass_chunks % 10 == 0:
                logger.info(f"  Pass {pass_num} - Chunk {pass_chunks} - Trained: {rows_trained:,} - RAM: {mem_usage:.1f}%")
            
            if pass_chunks % 5 == 0:
                gc.collect()
        
        logger.info(f"   Pass {pass_num} completato")
        
        # Validation
        logger.info(f"\n  Validation dopo pass {pass_num}...")
        y_true_val, y_pred_val = evaluate_chunk_based(model, val_loader, "VALIDATION")
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | Recall: {val_metrics['recall']:.4f}")
        
        del y_true_val, y_pred_val
        gc.collect()
    
    logger.info(f"\n Training LightGBM completato")
    logger.info(f"  Total samples trained: {rows_trained:,}")
    logger.info(f"  RAM: {get_memory_usage():.1f}%")
    
    return model


def evaluate_chunk_based(model, data_loader: ParquetDataLoader, dataset_name: str) -> tuple:
    """Valuta modello chunk-based."""
    
    y_true_list = []
    y_pred_list = []
    
    samples_processed = 0
    batch_count = 0
    
    for X_batch, y_batch in data_loader.iter_batches():
        batch_count += 1
        
        mem_usage = get_memory_usage()
        if mem_usage > 85:
            gc.collect()
        
        y_pred_batch = model.predict(X_batch)
        
        y_true_list.append(y_batch)
        y_pred_list.append(y_pred_batch)
        
        samples_processed += len(X_batch)
        
        del X_batch, y_batch, y_pred_batch
        
        if batch_count % 10 == 0:
            gc.collect()
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    del y_true_list, y_pred_list
    gc.collect()
    
    return y_true, y_pred


def save_model(model, model_type: str, metrics: dict) -> Path:
    """Salva modello e metrics."""
    
    model_dir = MODELS_DIR / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"\n   Modello: {model_path}")
    
    import json
    metrics_path = model_dir / "metrics.json"
    
    metadata = {
        'model_type': model_type,
        'training_method': 'incremental',
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   Metrics: {metrics_path}")
    
    return model_path


def main(model_type: str = None):
    """Pipeline training incremental."""
    
    if model_type is None:
        model_type = DEFAULT_MODEL
    
    if model_type not in ['xgboost', 'lightgbm']:
        raise ValueError(f"Model {model_type} non supportato per incremental learning. Usa 'xgboost' o 'lightgbm'")
    
    logger.info("="*70)
    logger.info(f"TRAINING INCREMENTAL - {model_type.upper()}")
    logger.info("="*70)
    logger.info(f"RAM iniziale: {get_memory_usage():.1f}%")
    logger.info(f"Training su TUTTO il dataset (chunk-based)")
    logger.info("="*70)
    
    mem_usage = get_memory_usage()
    if mem_usage > 60:
        logger.warning(f"\n  RAM già alta ({mem_usage:.1f}%)")
        logger.warning(f"Raccomandato: chiudere altre applicazioni")
        input("\nPremi ENTER per continuare...")
    
    # ========================================================================
    # STEP 1: Load artifacts
    # ========================================================================
    
    logger.info("\nSTEP 1: Caricamento artifacts")
    
    _, feature_cols = load_artifacts()
    
    logger.info(f"   {len(feature_cols)} feature")
    
    # ========================================================================
    # STEP 2: Setup loaders
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 2: SETUP DATA LOADERS")
    logger.info("="*70)
    
    train_path = PROCESSED_DATA_DIR / "train_scaled.parquet"
    val_path = PROCESSED_DATA_DIR / "val_scaled.parquet"
    test_path = PROCESSED_DATA_DIR / "test_scaled.parquet"
    
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}\nEsegui prima feature_engineering_aggregate.py")
    
    train_loader = ParquetDataLoader(train_path, feature_cols, chunk_size=TRAIN_CHUNK_SIZE)
    val_loader = ParquetDataLoader(val_path, feature_cols)
    test_loader = ParquetDataLoader(test_path, feature_cols)
    
    logger.info(f"   Train: {train_loader.get_total_rows():,} samples")
    logger.info(f"   Val: {val_loader.get_total_rows():,} samples")
    logger.info(f"   Test: {test_loader.get_total_rows():,} samples")
    
    # ========================================================================
    # STEP 3: Training incremental
    # ========================================================================
    
    start_time = datetime.now()
    
    if model_type == 'xgboost':
        model = train_xgboost_incremental(train_loader, val_loader)
    elif model_type == 'lightgbm':
        model = train_lightgbm_incremental(train_loader, val_loader)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\n  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    
    gc.collect()
    
    # ========================================================================
    # STEP 4: Final evaluation
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 4: FINAL EVALUATION")
    logger.info("="*70)
    
    # Validation
    logger.info("\nValidation set...")
    val_loader.reset()
    y_true_val, y_pred_val = evaluate_chunk_based(model, val_loader, "VALIDATION")
    val_metrics = compute_metrics(y_true_val, y_pred_val)
    print_metrics(val_metrics, "VALIDATION")
    
    del y_true_val, y_pred_val
    gc.collect()
    
    # Test
    logger.info("\nTest set...")
    test_loader.reset()
    y_true_test, y_pred_test = evaluate_chunk_based(model, test_loader, "TEST")
    test_metrics = compute_metrics(y_true_test, y_pred_test)
    print_metrics(test_metrics, "TEST")
    
    del y_true_test, y_pred_test
    gc.collect()
    
    # ========================================================================
    # STEP 5: Save
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 5: SALVATAGGIO")
    logger.info("="*70)
    
    all_metrics = {
        'validation': val_metrics,
        'test': test_metrics,
        'training_time_seconds': training_time,
    }
    
    model_path = save_model(model, model_type, all_metrics)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info(" TRAINING INCREMENTAL COMPLETATO")
    logger.info("="*70)
    logger.info(f"Modello: {model_type}")
    logger.info(f"Training method: Incremental (tutti i {train_loader.get_total_rows():,} samples)")
    logger.info(f"Training time: {training_time/60:.1f} min")
    logger.info(f"Salvato: {model_path}")
    logger.info(f"\nPERFORMANCE TEST SET:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"  FPR:       {test_metrics['fpr']:.4f}")
    logger.info(f"  FNR:       {test_metrics['fnr']:.4f}")
    logger.info(f"\nRAM finale: {get_memory_usage():.1f}%")
    logger.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Training incremental NIDS')
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['xgboost', 'lightgbm'],
        default='xgboost',
        help='Modello da trainare (solo xgboost e lightgbm supportati)'
    )
    
    args = parser.parse_args()
    
    main(args.model)