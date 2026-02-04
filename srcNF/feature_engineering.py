"""
Feature Engineering per NIDS NetFlow-based con chunk-based processing.

CRITICAL - STRATEGIA SCALER:
1. Fit scaler su SAMPLE RAPPRESENTATIVO del train set (1M righe)
2. Sample preso SENZA rimozione outlier (dati "sporchi")
3. Apply scaling a tutti i Parquet chunk-by-chunk
4. Feature selection minimale (solo varianza zero e alta correlazione)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
import pyarrow.parquet as pq
import pyarrow as pa

from config import (
    PROCESSED_DATA_DIR, ARTIFACTS_DIR, 
    FEATURES_TO_DROP, SCALER_TYPE, LABEL_COLUMN,
    CORRELATION_THRESHOLD, CHUNK_SIZE, SCALER_SAMPLE_SIZE,
    PARQUET_COMPRESSION
)
from utils import setup_logger, get_memory_usage


logger = setup_logger(__name__, 'feature_engineering.log')


def load_sample_for_scaler(parquet_path: Path, sample_size: int) -> pd.DataFrame:
    """
    Carica un sample rappresentativo per fitting dello scaler.
    
    CRITICAL: Sample preso SENZA rimozione outlier!
    Lo scaler deve vedere anche i picchi di traffico.
    
    Strategia:
    - Leggi righe uniformemente distribuite nel file
    - Mantieni stratificazione delle classi
    """
    
    logger.info(f"\nCaricamento sample per scaler...")
    logger.info(f"  Target sample size: {sample_size:,} righe")
    logger.info(f"  Source: {parquet_path.name}")
    
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    
    logger.info(f"  Total rows nel file: {total_rows:,}")
    
    if sample_size >= total_rows:
        logger.info(f"  Sample size >= total rows, uso tutto il file")
        sample_size = total_rows
    
    # Calcola step per sampling uniforme
    sample_ratio = sample_size / total_rows
    logger.info(f"  Sample ratio: {sample_ratio*100:.2f}%")
    
    # Leggi sample stratificato
    samples = []
    sampled_rows = 0
    
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        df_batch = batch.to_pandas()
        
        # Sample stratificato da questo batch
        batch_sample_size = int(len(df_batch) * sample_ratio)
        
        if batch_sample_size > 0:
            # Stratified sampling per classe
            benign = df_batch[df_batch['Label_Binary'] == 0]
            attack = df_batch[df_batch['Label_Binary'] == 1]
            
            benign_sample_size = int(len(benign) * sample_ratio)
            attack_sample_size = int(len(attack) * sample_ratio)
            
            sampled_benign = benign.sample(n=min(benign_sample_size, len(benign)), random_state=42)
            sampled_attack = attack.sample(n=min(attack_sample_size, len(attack)), random_state=42)
            
            batch_sample = pd.concat([sampled_benign, sampled_attack])
            samples.append(batch_sample)
            
            sampled_rows += len(batch_sample)
        
        if sampled_rows >= sample_size:
            break
    
    # Combina sample
    df_sample = pd.concat(samples, ignore_index=True)
    
    # Shuffle
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Limita alla dimensione target
    if len(df_sample) > sample_size:
        df_sample = df_sample.iloc[:sample_size]
    
    logger.info(f"   Sample caricato: {len(df_sample):,} righe")
    
    # Verifica distribuzione
    benign_count = (df_sample['Label_Binary'] == 0).sum()
    attack_count = (df_sample['Label_Binary'] == 1).sum()
    logger.info(f"  Distribuzione: Benign={benign_count:,} ({benign_count/len(df_sample)*100:.2f}%), "
                f"Attack={attack_count:,} ({attack_count/len(df_sample)*100:.2f}%)")
    
    return df_sample


def select_all_numeric_features(df: pd.DataFrame) -> list:
    """
    Seleziona TUTTE le feature numeriche dal dataset, 
    escludendo solo quelle in FEATURES_TO_DROP e le colonne label.
    
    Returns:
        Lista di tutte le feature numeriche disponibili
    """
    
    logger.info("\nSelezione feature dal dataset...")
    
    # Identifica tutte le colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"  Feature numeriche totali nel dataset: {len(numeric_cols)}")
    
    # Rimuovi colonne label
    label_cols = ['Label', 'Label_Binary', 'Label_Original']
    numeric_cols = [col for col in numeric_cols if col not in label_cols]
    
    logger.info(f"  Dopo rimozione label: {len(numeric_cols)}")
    
    # Rimuovi feature in FEATURES_TO_DROP
    features_to_drop_upper = [f.upper() for f in FEATURES_TO_DROP]
    
    final_features = []
    dropped_features = []
    
    for col in numeric_cols:
        if col.upper() in features_to_drop_upper:
            dropped_features.append(col)
        else:
            final_features.append(col)
    
    logger.info(f"  Feature escluse (FEATURES_TO_DROP): {len(dropped_features)}")
    for feat in dropped_features:
        logger.info(f"    - {feat}")
    
    logger.info(f"\n   Feature selezionate: {len(final_features)}")
    
    return final_features


def remove_zero_variance_features(X_sample: pd.DataFrame, feature_cols: list) -> list:
    """
    Rimuove feature con varianza zero (costanti).
    
    Returns:
        Lista feature con varianza > 0
    """
    
    logger.info("\nRimozione feature a varianza zero...")
    
    variances = X_sample[feature_cols].var()
    zero_var_features = variances[variances == 0].index.tolist()
    
    if zero_var_features:
        logger.info(f"  Trovate {len(zero_var_features)} feature costanti:")
        for feat in zero_var_features:
            logger.info(f"    - {feat}")
        
        feature_cols = [f for f in feature_cols if f not in zero_var_features]
        logger.info(f"   Feature rimanenti: {len(feature_cols)}")
    else:
        logger.info("   Nessuna feature a varianza zero")
    
    return feature_cols


def remove_highly_correlated_features(X_sample: pd.DataFrame, feature_cols: list, threshold: float = None) -> list:
    """
    Rimuove feature altamente correlate (|corr| > threshold).
    
    Returns:
        Lista feature dopo rimozione correlazioni
    """
    
    if threshold is None:
        threshold = CORRELATION_THRESHOLD
    
    logger.info(f"\nRimozione feature correlate (threshold={threshold})...")
    
    # Calcola matrice correlazione
    corr_matrix = X_sample[feature_cols].corr().abs()
    
    # Trova coppie altamente correlate
    to_drop = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col_to_drop = corr_matrix.columns[j]
                col_to_keep = corr_matrix.columns[i]
                
                if col_to_drop not in to_drop:
                    to_drop.append(col_to_drop)
                    logger.info(f"  Rimozione {col_to_drop} (corr={corr_matrix.iloc[i, j]:.3f} con {col_to_keep})")
    
    if to_drop:
        feature_cols = [f for f in feature_cols if f not in to_drop]
        logger.info(f"   Rimosse {len(to_drop)} feature correlate")
        logger.info(f"   Feature rimanenti: {len(feature_cols)}")
    else:
        logger.info("   Nessuna feature altamente correlata")
    
    return feature_cols


def fit_scaler(X_sample: pd.DataFrame, feature_cols: list) -> object:
    """
    Fit scaler su sample rappresentativo.
    
    CRITICAL: Fittato su dati COMPLETI (outlier inclusi)
    per gestire correttamente picchi di traffico in produzione.
    
    Il sample è già stato prelevato SENZA rimozione outlier.
    """
    
    logger.info("\n" + "="*70)
    logger.info("FITTING SCALER")
    logger.info("="*70)
    logger.info(f"  Tipo: {SCALER_TYPE}")
    logger.info(f"  Sample size: {len(X_sample):,} righe")
    logger.info(f"  Feature: {len(feature_cols)}")
    logger.info(f"  CRITICAL: Sample contiene OUTLIER (dati reali)")
    
    if SCALER_TYPE == 'robust':
        scaler = RobustScaler()
        logger.info("   RobustScaler (usa mediana e IQR, resistente outlier)")
    else:
        scaler = StandardScaler()
        logger.info("   StandardScaler (usa media e std)")
    
    # Fit
    X_for_fitting = X_sample[feature_cols]
    scaler.fit(X_for_fitting)
    
    logger.info("   Scaler fitted su sample rappresentativo")
    logger.info("="*70)
    
    return scaler


def scale_parquet_file(
    input_path: Path,
    output_path: Path,
    scaler: object,
    feature_cols: list
) -> None:
    """
    Scala file Parquet chunk-by-chunk.
    """
    
    logger.info(f"\nScaling {input_path.name}...")
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    writer = None
    rows_processed = 0
    
    try:
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
            
            df_batch = batch.to_pandas()
            rows_processed += len(df_batch)
            
            # Log progresso
            if (batch_idx + 1) % 20 == 0:
                progress = (rows_processed / total_rows) * 100
                mem_usage = get_memory_usage()
                logger.info(f"  Chunk {batch_idx + 1} - {progress:.1f}% - RAM: {mem_usage:.1f}%")
            
            # Scala feature
            X = df_batch[feature_cols].copy()
            X_scaled = scaler.transform(X)
            
            # Ricrea DataFrame con feature scalate
            df_scaled = pd.DataFrame(
                X_scaled,
                columns=feature_cols,
                index=df_batch.index
            )
            
            # Aggiungi label
            df_scaled['Label_Binary'] = df_batch['Label_Binary'].values
            
            # Converti a PyArrow Table
            table = pa.Table.from_pandas(df_scaled)
            
            # Inizializza writer
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression=PARQUET_COMPRESSION
                )
            
            # Scrivi
            writer.write_table(table)
            
            # Libera memoria
            del df_batch, X, X_scaled, df_scaled, table
        
        # Chiudi writer
        if writer:
            writer.close()
        
        logger.info(f"   Scaling completato: {rows_processed:,} righe")
        
    except Exception as e:
        if writer:
            writer.close()
        raise e


def save_artifacts(scaler: object, feature_cols: list) -> None:
    """Salva scaler e feature list."""
    
    logger.info(f"\nSalvataggio artifacts in {ARTIFACTS_DIR}")
    
    # Scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"   {scaler_path.name}")
    
    # Feature list
    import json
    features_path = ARTIFACTS_DIR / "features.json"
    with open(features_path, 'w') as f:
        json.dump({
            'features': feature_cols,
            'n_features': len(feature_cols),
            'scaler_type': SCALER_TYPE,
            'correlation_threshold': CORRELATION_THRESHOLD,
            'scaler_sample_size': SCALER_SAMPLE_SIZE,
        }, f, indent=2)
    logger.info(f"   {features_path.name}")
    
    # Salva anche lista leggibile
    features_txt_path = ARTIFACTS_DIR / "features.txt"
    with open(features_txt_path, 'w') as f:
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"Scaler type: {SCALER_TYPE}\n")
        f.write(f"Scaler fitted on: {SCALER_SAMPLE_SIZE:,} samples (WITH outliers)\n")
        f.write(f"Correlation threshold: {CORRELATION_THRESHOLD}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("FEATURE LIST:\n")
        f.write("="*60 + "\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {feat}\n")
    logger.info(f"   {features_txt_path.name}")


def load_artifacts() -> tuple:
    """Carica scaler e feature list."""
    
    logger.info(f"Caricamento artifacts da {ARTIFACTS_DIR}")
    
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    
    import json
    with open(ARTIFACTS_DIR / "features.json", 'r') as f:
        features_info = json.load(f)
    
    feature_cols = features_info['features']
    
    logger.info(f"   Scaler caricato ({features_info['scaler_type']})")
    logger.info(f"   {len(feature_cols)} feature")
    
    return scaler, feature_cols


def main():
    """Pipeline feature engineering completa con chunk-based processing."""
    
    logger.info("="*70)
    logger.info("FEATURE ENGINEERING - NetFlow NIDS (CHUNK-BASED)")
    logger.info("="*70)
    logger.info(f"RAM disponibile: {get_memory_usage():.1f}% usata")
    logger.info("="*70)
    
    # ========================================================================
    # STEP 1: Carica sample rappresentativo per feature selection
    # ========================================================================
    
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    
    logger.info("\nSTEP 1: Caricamento sample per feature selection")
    
    # Carica sample (SENZA rimozione outlier!)
    sample = load_sample_for_scaler(train_path, SCALER_SAMPLE_SIZE)
    
    # ========================================================================
    # STEP 2: Feature selection
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 2: FEATURE SELECTION")
    logger.info("="*70)
    
    # Seleziona tutte le feature numeriche
    feature_cols = select_all_numeric_features(sample)
    
    if not feature_cols:
        raise ValueError(
            "Nessuna feature numerica trovata nel dataset!\n"
            "Verifica il formato del dataset"
        )
    
    # Prepare X, y dal sample
    X_sample = sample[feature_cols].copy()
    y_sample = sample['Label_Binary'].copy()
    
    logger.info(f"\nFeature matrix sample preparata: {X_sample.shape}")
    
    # Rimuovi feature a varianza zero
    feature_cols = remove_zero_variance_features(X_sample, feature_cols)
    
    # Rimuovi feature altamente correlate
    feature_cols = remove_highly_correlated_features(X_sample, feature_cols)
    
    # Aggiorna X_sample dopo selezione
    X_sample = X_sample[feature_cols]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE FINALI: {len(feature_cols)}")
    logger.info(f"{'='*70}")
    
    # ========================================================================
    # STEP 3: Fit scaler su sample (CON OUTLIER!)
    # ========================================================================
    
    logger.info("\nSTEP 3: FITTING SCALER")
    
    scaler = fit_scaler(X_sample, feature_cols)
    
    # Libera memoria del sample
    del sample, X_sample, y_sample
    
    # ========================================================================
    # STEP 4: Scale tutti i Parquet files chunk-by-chunk
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
        
        # Verifica file generato
        output_size = output_path.stat().st_size / (1024**2)
        logger.info(f"   File generato: {output_path.name} ({output_size:.1f} MB)")
    
    # ========================================================================
    # STEP 5: Salva artifacts
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 5: SALVATAGGIO ARTIFACTS")
    logger.info("="*70)
    
    save_artifacts(scaler, feature_cols)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info(" FEATURE ENGINEERING COMPLETATO")
    logger.info("="*70)
    logger.info(f"Feature utilizzate: {len(feature_cols)}")
    logger.info(f"Scaler: {SCALER_TYPE} (fitted su {SCALER_SAMPLE_SIZE:,} samples CON outlier)")
    logger.info(f"Output: {PROCESSED_DATA_DIR}")
    logger.info(f"\nFile scaled generati:")
    for _, _, output_path in datasets:
        size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"  {output_path.name} ({size_mb:.1f} MB)")
    logger.info(f"\nPROSSIMI PASSI:")
    logger.info("  python srcNF/training.py --model xgboost")
    logger.info("="*70)


if __name__ == "__main__":
    main()
