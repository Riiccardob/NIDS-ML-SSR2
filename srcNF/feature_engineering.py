"""
Feature Engineering con AGGREGATE SCALER per tutti i dati.

STRATEGIA CORRETTA per LIVE SNIFFER:
1. Calcola statistiche (Q1, Q3, median) chunk-by-chunk su TUTTO il dataset
2. Aggrega statistiche → Scaler PERFETTO che vede TUTTI gli outlier
3. Apply scaling chunk-by-chunk
4. Se scaling genera inf → Sostituisci con valori grandi ma finiti (±1e308)

IMPORTANTE: Lo scaler DEVE vedere tutti gli outlier per funzionare in live!
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
import pyarrow.parquet as pq
import pyarrow as pa
import gc
from typing import Tuple, List

from config import (
    PROCESSED_DATA_DIR, ARTIFACTS_DIR, 
    FEATURES_TO_DROP, SCALER_TYPE, LABEL_COLUMN,
    CORRELATION_THRESHOLD, CHUNK_SIZE,
    PARQUET_COMPRESSION
)
from utils import setup_logger, get_memory_usage


logger = setup_logger(__name__, 'feature_engineering.log')


def select_all_numeric_features(df: pd.DataFrame) -> list:
    """Seleziona tutte le feature numeriche escludendo label."""
    
    logger.info("\nSelezione feature dal dataset...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"  Feature numeriche totali: {len(numeric_cols)}")
    
    # Rimuovi label
    label_cols = ['Label', 'Label_Binary', 'Label_Original']
    numeric_cols = [col for col in numeric_cols if col not in label_cols]
    logger.info(f"  Dopo rimozione label: {len(numeric_cols)}")
    
    # Rimuovi feature in FEATURES_TO_DROP
    features_to_drop_upper = [f.upper() for f in FEATURES_TO_DROP]
    final_features = [col for col in numeric_cols if col.upper() not in features_to_drop_upper]
    
    dropped = len(numeric_cols) - len(final_features)
    if dropped > 0:
        logger.info(f"  Feature rimosse da config: {dropped}")
    
    logger.info(f"   Feature selezionate: {len(final_features)}")
    return final_features


def remove_zero_variance_features(parquet_path: Path, feature_cols: list) -> list:
    """
    Rimuove feature con varianza zero analizzando chunk-by-chunk.
    """
    
    logger.info("\nAnalisi varianza feature (chunk-based)...")
    logger.info(f"  Feature da analizzare: {len(feature_cols)}")
    
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Accumula min/max per ogni feature
    min_vals = None
    max_vals = None
    chunk_count = 0
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        chunk_count += 1
        df_batch = batch.to_pandas()
        
        X_batch = df_batch[feature_cols].values
        
        if min_vals is None:
            min_vals = X_batch.min(axis=0)
            max_vals = X_batch.max(axis=0)
        else:
            min_vals = np.minimum(min_vals, X_batch.min(axis=0))
            max_vals = np.maximum(max_vals, X_batch.max(axis=0))
        
        del df_batch, X_batch
        
        if chunk_count % 50 == 0:
            logger.info(f"    Analizzati {chunk_count} chunk - RAM: {get_memory_usage():.1f}%")
            gc.collect()
    
    # Feature con min == max hanno varianza zero
    zero_var_mask = (min_vals == max_vals)
    zero_var_features = [feature_cols[i] for i, is_zero in enumerate(zero_var_mask) if is_zero]
    
    if zero_var_features:
        logger.info(f"  Trovate {len(zero_var_features)} feature a varianza zero:")
        for feat in zero_var_features[:5]:
            logger.info(f"    - {feat}")
        if len(zero_var_features) > 5:
            logger.info(f"    ... e altre {len(zero_var_features)-5}")
        
        feature_cols = [f for f in feature_cols if f not in zero_var_features]
        logger.info(f"   Rimanenti: {len(feature_cols)} feature")
    else:
        logger.info("   Nessuna feature a varianza zero")
    
    return feature_cols


def remove_highly_correlated_features_sample(parquet_path: Path, feature_cols: list, sample_size: int = 100_000) -> list:
    """
    Rimuove feature correlate usando un sample piccolo.
    """
    
    logger.info(f"\nRimozione feature correlate (su sample {sample_size:,})...")
    
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Carica sample
    samples = []
    sampled = 0
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        df_batch = batch.to_pandas()
        n_sample = min(sample_size - sampled, len(df_batch))
        
        if n_sample > 0:
            samples.append(df_batch[feature_cols].sample(n=n_sample, random_state=42))
            sampled += n_sample
        
        del df_batch
        
        if sampled >= sample_size:
            break
    
    df_sample = pd.concat(samples, ignore_index=True)
    del samples
    gc.collect()
    
    logger.info(f"  Sample caricato: {len(df_sample):,} righe")
    
    # Calcola correlazione
    corr_matrix = df_sample[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
    
    if to_drop:
        logger.info(f"  Trovate {len(to_drop)} feature correlate (>{CORRELATION_THRESHOLD})")
        feature_cols = [f for f in feature_cols if f not in to_drop]
        logger.info(f"   Rimanenti: {len(feature_cols)} feature")
    else:
        logger.info("   Nessuna feature altamente correlata")
    
    del df_sample, corr_matrix
    gc.collect()
    
    return feature_cols


def compute_aggregate_scaler_stats(parquet_path: Path, feature_cols: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola statistiche aggregate per RobustScaler su TUTTO il dataset.
    
    IMPORTANTE: Mantiene TUTTI gli outlier - essenziale per live sniffer!
    
    Returns:
        (q1, median, q3) per ogni feature
    """
    
    logger.info("\nCalcolo statistiche aggregate per scaler...")
    logger.info(f"  Strategia: Calcola Q1, median, Q3 per chunk, poi aggrega")
    logger.info(f"  Feature: {len(feature_cols)}")
    logger.info(f"  IMPORTANTE: Mantiene TUTTI gli outlier!")
    
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    
    logger.info(f"  Total rows: {total_rows:,}")
    
    # Liste per accumulare statistiche per chunk
    q1_list = []
    median_list = []
    q3_list = []
    
    chunk_count = 0
    rows_processed = 0
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        chunk_count += 1
        
        # CHECK RAM
        mem_usage = get_memory_usage()
        if mem_usage > 85:
            logger.warning(f"      RAM alta ({mem_usage:.1f}%), GC...")
            gc.collect()
        
        df_batch = batch.to_pandas()
        X_batch = df_batch[feature_cols].values
        
        # Calcola percentili per questo chunk
        q1 = np.percentile(X_batch, 25, axis=0)
        median = np.percentile(X_batch, 50, axis=0)
        q3 = np.percentile(X_batch, 75, axis=0)
        
        q1_list.append(q1)
        median_list.append(median)
        q3_list.append(q3)
        
        rows_processed += len(df_batch)
        
        del df_batch, X_batch
        
        if chunk_count % 20 == 0:
            pct = (rows_processed / total_rows) * 100
            logger.info(f"    Chunk {chunk_count} - Processed: {rows_processed:,}/{total_rows:,} ({pct:.1f}%) - RAM: {mem_usage:.1f}%")
        
        if chunk_count % 10 == 0:
            gc.collect()
    
    logger.info(f"\n   Statistiche calcolate per {chunk_count} chunk")
    logger.info(f"  Aggregazione finale...")
    
    # Aggrega: usa MEDIAN delle statistiche dei chunk
    # (median è più robusto agli outlier rispetto a mean)
    final_q1 = np.median(q1_list, axis=0)
    final_median = np.median(median_list, axis=0)
    final_q3 = np.median(q3_list, axis=0)
    
    logger.info(f"   Statistiche aggregate calcolate")
    logger.info(f"  RAM: {get_memory_usage():.1f}%")
    
    return final_q1, final_median, final_q3


def create_robust_scaler_from_stats(q1: np.ndarray, median: np.ndarray, q3: np.ndarray, feature_cols: list) -> RobustScaler:
    """
    Crea RobustScaler da statistiche pre-calcolate.
    """
    
    logger.info("\nCreazione RobustScaler da statistiche aggregate...")
    
    scaler = RobustScaler()
    
    # Imposta parametri manualmente
    scaler.center_ = median
    scaler.scale_ = q3 - q1
    
    # Evita divisione per zero
    scaler.scale_[scaler.scale_ == 0] = 1.0
    
    scaler.n_features_in_ = len(feature_cols)
    scaler.feature_names_in_ = np.array(feature_cols)
    
    logger.info(f"   RobustScaler creato")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  IQR medio: {np.mean(scaler.scale_):.4f}")
    logger.info(f"  IQR min: {np.min(scaler.scale_):.4f}")
    logger.info(f"  IQR max: {np.max(scaler.scale_):.4f}")
    
    return scaler


def handle_inf_after_scaling(X_scaled: np.ndarray) -> np.ndarray:
    """
    Gestisce inf/nan DOPO lo scaling.
    
    Se lo scaling genera inf (outlier estremi), sostituisce con valori grandi ma finiti.
    Questo permette al modello di imparare che "valore molto grande = potenziale anomalia".
    """
    
    # Sostituisci inf con valori grandi ma finiti
    X_scaled[np.isposinf(X_scaled)] = 1e10   # +inf → valore grande positivo
    X_scaled[np.isneginf(X_scaled)] = -1e10  # -inf → valore grande negativo
    
    # Sostituisci nan con 0 (valore scaled neutro)
    X_scaled[np.isnan(X_scaled)] = 0.0
    
    return X_scaled


def scale_parquet_file(input_path: Path, output_path: Path, scaler: RobustScaler, feature_cols: list):
    """
    Scala Parquet file chunk-by-chunk con gestione robusta di inf.
    """
    
    logger.info(f"\nScaling: {input_path.name} → {output_path.name}")
    logger.info(f"  Chunk size: {CHUNK_SIZE:,} righe")
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    logger.info(f"  Total rows: {total_rows:,}")
    
    writer = None
    rows_processed = 0
    batch_idx = 0
    inf_count_total = 0
    
    try:
        for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
            batch_idx += 1
            
            mem_usage = get_memory_usage()
            if mem_usage > 85:
                logger.warning(f"      RAM alta ({mem_usage:.1f}%), GC...")
                gc.collect()
            
            df_batch = batch.to_pandas()
            
            # Scala
            X_batch = df_batch[feature_cols].values
            X_scaled = scaler.transform(X_batch)
            
            # Conta inf PRIMA di gestirli
            n_inf = np.isinf(X_scaled).sum()
            if n_inf > 0:
                inf_count_total += n_inf
                if batch_idx == 1:  # Log solo primo batch
                    logger.warning(f"    Chunk {batch_idx}: {n_inf} valori inf generati dallo scaling (NORMALE per outlier estremi)")
            
            # Gestisci inf/nan
            X_scaled = handle_inf_after_scaling(X_scaled)
            
            # Update DataFrame
            df_batch[feature_cols] = X_scaled
            
            del X_batch, X_scaled
            
            # Write
            table = pa.Table.from_pandas(df_batch)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=PARQUET_COMPRESSION)
            
            writer.write_table(table)
            rows_processed += len(df_batch)
            
            del df_batch, table
            
            if batch_idx % 20 == 0:
                pct = (rows_processed / total_rows) * 100
                logger.info(f"    Processed: {rows_processed:,}/{total_rows:,} ({pct:.1f}%) - RAM: {mem_usage:.1f}%")
            
            if batch_idx % 20 == 0:
                gc.collect()
        
        if writer:
            writer.close()
        
        logger.info(f"   Scaling completato: {rows_processed:,} righe")
        
        if inf_count_total > 0:
            pct_inf = (inf_count_total / (rows_processed * len(feature_cols))) * 100
            logger.info(f"  ℹ  Valori inf gestiti: {inf_count_total:,} ({pct_inf:.4f}%) - sostituiti con ±1e10")
            logger.info(f"  Questo è NORMALE per outlier estremi in dataset NetFlow")
        
        gc.collect()
        
    except Exception as e:
        if writer:
            writer.close()
        raise e


def save_artifacts(scaler: RobustScaler, feature_cols: list, scaler_stats: dict) -> None:
    """Salva scaler e metadata."""
    
    logger.info(f"\nSalvataggio artifacts in {ARTIFACTS_DIR}")
    
    # Scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"   {scaler_path.name}")
    
    # Feature list + metadata
    import json
    features_path = ARTIFACTS_DIR / "features.json"
    with open(features_path, 'w') as f:
        json.dump({
            'features': feature_cols,
            'n_features': len(feature_cols),
            'scaler_type': 'robust_aggregate',
            'scaler_method': 'chunk_aggregate_statistics',
            'correlation_threshold': CORRELATION_THRESHOLD,
            'outlier_handling': 'kept_all_outliers_for_live_sniffer',
            'scaler_stats': scaler_stats,
        }, f, indent=2)
    logger.info(f"   {features_path.name}")
    
    # Lista leggibile
    features_txt_path = ARTIFACTS_DIR / "features.txt"
    with open(features_txt_path, 'w') as f:
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"Scaler type: RobustScaler (aggregate)\n")
        f.write(f"Scaler computed on: ALL {scaler_stats['total_rows']:,} rows\n")
        f.write(f"Method: Chunk-based aggregate statistics\n")
        f.write(f"Outlier handling: KEPT ALL (essential for live sniffer)\n")
        f.write(f"Correlation threshold: {CORRELATION_THRESHOLD}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("FEATURE LIST:\n")
        f.write("="*60 + "\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {feat}\n")
    logger.info(f"   {features_txt_path.name}")


def load_artifacts() -> Tuple[RobustScaler, list]:
    """Carica scaler e feature list."""
    
    logger.info(f"Caricamento artifacts da {ARTIFACTS_DIR}")
    
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    
    import json
    with open(ARTIFACTS_DIR / "features.json", 'r') as f:
        features_info = json.load(f)
    
    feature_cols = features_info['features']
    
    logger.info(f"   Scaler caricato (aggregate)")
    logger.info(f"   {len(feature_cols)} feature")
    
    return scaler, feature_cols


def main():
    """Pipeline feature engineering con aggregate scaler."""
    
    logger.info("="*70)
    logger.info("FEATURE ENGINEERING - AGGREGATE SCALER (LIVE SNIFFER READY)")
    logger.info("="*70)
    logger.info(f"RAM: {get_memory_usage():.1f}% usata")
    logger.info(f"Strategia: Aggregate statistics su TUTTO il dataset")
    logger.info(f"IMPORTANTE: Mantiene TUTTI gli outlier per live detection")
    logger.info("="*70)
    
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    
    if not train_path.exists():
        raise FileNotFoundError(f"File non trovato: {train_path}\nEsegui prima preprocessing.py")
    
    # ========================================================================
    # STEP 1: Load sample per feature selection
    # ========================================================================
    
    logger.info("\nSTEP 1: Feature selection (su sample)")
    logger.info(f"Caricamento sample...")
    
    # Small sample solo per feature selection
    parquet_file = pq.ParquetFile(train_path)
    batch = next(parquet_file.iter_batches(batch_size=CHUNK_SIZE))
    df_sample = batch.to_pandas()
    
    logger.info(f"  Sample: {len(df_sample):,} righe")
    
    feature_cols = select_all_numeric_features(df_sample)
    
    del df_sample
    gc.collect()
    
    # ========================================================================
    # STEP 2: Rimuovi feature problematiche
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 2: ANALISI FEATURE SU TUTTO IL DATASET")
    logger.info("="*70)
    
    # Varianza zero
    feature_cols = remove_zero_variance_features(train_path, feature_cols)
    
    # Correlazione alta (su sample piccolo è OK)
    feature_cols = remove_highly_correlated_features_sample(train_path, feature_cols)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE FINALI: {len(feature_cols)}")
    logger.info(f"{'='*70}")
    
    # ========================================================================
    # STEP 3: Calcola aggregate scaler su TUTTO il dataset
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 3: AGGREGATE SCALER SU TUTTO IL DATASET")
    logger.info("="*70)
    
    q1, median, q3 = compute_aggregate_scaler_stats(train_path, feature_cols)
    scaler = create_robust_scaler_from_stats(q1, median, q3, feature_cols)
    
    # Metadata scaler
    parquet_file = pq.ParquetFile(train_path)
    scaler_stats = {
        'total_rows': parquet_file.metadata.num_rows,
        'method': 'chunk_aggregate',
    }
    
    gc.collect()
    
    # ========================================================================
    # STEP 4: Scale tutti i dataset
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 4: SCALING DATASETS")
    logger.info("="*70)
    
    datasets = [
        ('train', PROCESSED_DATA_DIR / "train.parquet", PROCESSED_DATA_DIR / "train_scaled.parquet"),
        ('val', PROCESSED_DATA_DIR / "val.parquet", PROCESSED_DATA_DIR / "val_scaled.parquet"),
        ('test', PROCESSED_DATA_DIR / "test.parquet", PROCESSED_DATA_DIR / "test_scaled.parquet")
    ]
    
    for name, input_path, output_path in datasets:
        logger.info(f"\nScaling {name} set...")
        scale_parquet_file(input_path, output_path, scaler, feature_cols)
        
        output_size = output_path.stat().st_size / (1024**2)
        logger.info(f"   {output_path.name} ({output_size:.1f} MB)")
        
        gc.collect()
    
    # ========================================================================
    # STEP 5: Salva artifacts
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 5: SALVATAGGIO ARTIFACTS")
    logger.info("="*70)
    
    save_artifacts(scaler, feature_cols, scaler_stats)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info(" FEATURE ENGINEERING COMPLETATO")
    logger.info("="*70)
    logger.info(f"Feature: {len(feature_cols)}")
    logger.info(f"Scaler: RobustScaler (aggregate su {scaler_stats['total_rows']:,} righe)")
    logger.info(f"Outlier: TUTTI MANTENUTI (essenziale per live sniffer)")
    logger.info(f"Inf handling: Post-scaling replacement con ±1e10")
    logger.info(f"\nPROSSIMI PASSI:")
    logger.info("  python srcNF/training_incremental.py --model xgboost")
    logger.info("  python srcNF/training_incremental.py --model lightgbm")
    logger.info("="*70)


if __name__ == "__main__":
    main()