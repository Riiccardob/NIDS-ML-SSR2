"""
Preprocessing del dataset NF-UQ-NIDS-v2 con chunk-based processing PARALLELO.

STRATEGIA per dataset grandi (76M records):
1. Leggi CSV a chunk (500k righe alla volta)
2. Pulisci chunk in PARALLELO con multiprocessing
3. Salva in Parquet compresso
4. Stratified split su Parquet senza caricare tutto in RAM

PARALLELIZZAZIONE:
- Ogni chunk viene processato da un worker separato
- Numero worker configurabile in config.py (% di CPU cores)
- Scrittura Parquet gestita in modo thread-safe
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool, Manager, Lock
from functools import partial
import time

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LABEL_COLUMN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE,
    CHUNK_SIZE, PARQUET_COMPRESSION, PARQUET_ENGINE,
    ENABLE_PARALLEL_PROCESSING, CPU_USAGE_PERCENT, MIN_WORKERS, MAX_WORKERS
)
from utils import setup_logger, get_memory_usage, get_worker_count


logger = setup_logger(__name__, 'preprocessing.log')


def get_csv_files() -> list:
    """Trova file CSV nel raw data directory."""
    
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"Nessun file CSV trovato in {RAW_DATA_DIR}\n"
            "Assicurati di aver scaricato NF-UQ-NIDS-v2"
        )
    
    return csv_files


def estimate_total_rows(csv_path: Path) -> int:
    """
    Stima il numero totale di righe nel CSV.
    Legge un piccolo sample e stima dal file size.
    """
    
    logger.info(f"Stima numero righe in {csv_path.name}...")
    
    # Leggi prime 1000 righe per stimare
    sample = pd.read_csv(csv_path, nrows=1000)
    
    # Calcola byte per riga
    file_size = csv_path.stat().st_size
    bytes_per_row = file_size / 1000  # Approssimazione
    
    # Stima totale (con margine di sicurezza)
    estimated_rows = int(file_size / bytes_per_row * 0.95)
    
    logger.info(f"  Stima: ~{estimated_rows:,} righe")
    
    return estimated_rows


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Pulizia base di un chunk.
    
    CRITICAL: NON rimuove outlier!
    Lo scaler deve vedere anche i picchi di traffico.
    """
    
    initial_rows = len(chunk)
    
    # Strip column names (solo nel primo chunk, ma non fa male ripeterlo)
    chunk.columns = chunk.columns.str.strip()
    
    # Rimuovi righe con NaN
    chunk = chunk.dropna()
    
    # Rimuovi infiniti (solo nelle colonne numeriche)
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        inf_mask = np.isinf(chunk[col])
        if inf_mask.any():
            chunk = chunk[~inf_mask]
    
    # IMPORTANTE: NON rimuoviamo outlier
    # Il RobustScaler deve vedere anche i picchi di traffico
    
    return chunk.reset_index(drop=True)


def encode_labels(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara label binarie.
    
    Il dataset NF-UQ-NIDS-v2 ha già Label binaria (0/1).
    """
    
    if LABEL_COLUMN not in chunk.columns:
        raise ValueError(f"Colonna {LABEL_COLUMN} non trovata")
    
    # Backup label originale
    chunk['Label_Original'] = chunk[LABEL_COLUMN].copy()
    
    # Label binaria (già presente nel dataset)
    chunk['Label_Binary'] = chunk[LABEL_COLUMN].copy()
    
    return chunk


def process_chunk_worker(chunk_data: Tuple[int, pd.DataFrame]) -> Tuple[int, pd.DataFrame, dict]:
    """
    Worker function per processare un singolo chunk.
    
    Args:
        chunk_data: Tupla (chunk_idx, chunk_df)
    
    Returns:
        Tupla (chunk_idx, cleaned_chunk, stats)
    """
    
    chunk_idx, chunk = chunk_data
    
    # Statistiche pre-pulizia
    initial_rows = len(chunk)
    
    # Pulisci chunk
    chunk = clean_chunk(chunk)
    
    # Encode labels
    chunk = encode_labels(chunk)
    
    # Statistiche
    stats = {
        'chunk_idx': chunk_idx,
        'rows_processed': initial_rows,
        'rows_cleaned': len(chunk),
        'benign': (chunk['Label_Binary'] == 0).sum(),
        'attack': (chunk['Label_Binary'] == 1).sum()
    }
    
    return chunk_idx, chunk, stats


def process_csv_to_parquet_parallel(csv_path: Path, output_path: Path, n_workers: int) -> dict:
    """
    Processa CSV a chunk in PARALLELO e salva in Parquet.
    
    Args:
        csv_path: Path del CSV da processare
        output_path: Path del Parquet di output
        n_workers: Numero di worker paralleli
    
    Returns:
        Dict con statistiche del processing
    """
    
    logger.info(f"\nProcessing {csv_path.name} (PARALLEL MODE)")
    logger.info(f"  Chunk size: {CHUNK_SIZE:,} righe")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  Output: {output_path}")
    
    # Stima totale righe
    estimated_rows = estimate_total_rows(csv_path)
    estimated_chunks = estimated_rows // CHUNK_SIZE + 1
    
    # Statistiche globali
    global_stats = {
        'total_rows_processed': 0,
        'total_rows_cleaned': 0,
        'chunks_processed': 0,
        'label_distribution': {'benign': 0, 'attack': 0}
    }
    
    # Writer Parquet
    writer = None
    schema = None
    
    start_time = time.time()
    
    try:
        # Leggi CSV a chunk
        chunk_iterator = pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
        
        # Buffer per accumula chunk da processare
        chunk_buffer = []
        buffer_size = n_workers * 2  # Buffer = 2x numero worker
        
        # Pool di worker
        with Pool(processes=n_workers) as pool:
            
            chunk_idx = 0
            
            for chunk in chunk_iterator:
                
                # Aggiungi al buffer
                chunk_buffer.append((chunk_idx, chunk))
                chunk_idx += 1
                
                # Quando buffer è pieno (o ultimo chunk), processa in parallelo
                if len(chunk_buffer) >= buffer_size or chunk_idx >= estimated_chunks:
                    
                    # Processa batch di chunk in parallelo
                    results = pool.map(process_chunk_worker, chunk_buffer)
                    
                    # Ordina risultati per chunk_idx (mantiene ordine)
                    results.sort(key=lambda x: x[0])
                    
                    # Scrivi risultati in ordine
                    for _, cleaned_chunk, stats in results:
                        
                        # Update global stats
                        global_stats['total_rows_processed'] += stats['rows_processed']
                        global_stats['total_rows_cleaned'] += stats['rows_cleaned']
                        global_stats['chunks_processed'] += 1
                        global_stats['label_distribution']['benign'] += stats['benign']
                        global_stats['label_distribution']['attack'] += stats['attack']
                        
                        # Converti a PyArrow Table
                        table = pa.Table.from_pandas(cleaned_chunk)
                        
                        # Inizializza writer con schema dal primo chunk
                        if writer is None:
                            schema = table.schema
                            writer = pq.ParquetWriter(
                                output_path,
                                schema,
                                compression=PARQUET_COMPRESSION
                            )
                        
                        # Scrivi chunk
                        writer.write_table(table)
                    
                    # Log progresso
                    progress = (global_stats['chunks_processed'] / estimated_chunks) * 100
                    elapsed = time.time() - start_time
                    chunks_per_sec = global_stats['chunks_processed'] / elapsed if elapsed > 0 else 0
                    eta_seconds = (estimated_chunks - global_stats['chunks_processed']) / chunks_per_sec if chunks_per_sec > 0 else 0
                    
                    logger.info(
                        f"  Chunk {global_stats['chunks_processed']}/{estimated_chunks} "
                        f"({progress:.1f}%) - "
                        f"{chunks_per_sec:.1f} chunks/s - "
                        f"ETA: {eta_seconds/60:.1f}min - "
                        f"RAM: {get_memory_usage():.1f}%"
                    )
                    
                    # Clear buffer
                    chunk_buffer = []
        
        # Chiudi writer
        if writer:
            writer.close()
        
        elapsed_total = time.time() - start_time
        
        logger.info(f"\n   Processing completato in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
        logger.info(f"  Righe processate: {global_stats['total_rows_processed']:,}")
        logger.info(f"  Righe pulite: {global_stats['total_rows_cleaned']:,}")
        logger.info(f"  Chunk processati: {global_stats['chunks_processed']}")
        logger.info(f"  Velocità media: {global_stats['chunks_processed']/elapsed_total:.2f} chunks/s")
        
        return global_stats
        
    except Exception as e:
        if writer:
            writer.close()
        raise e


def process_csv_to_parquet_sequential(csv_path: Path, output_path: Path) -> dict:
    """
    Processa CSV a chunk SEQUENZIALMENTE (fallback se parallel disabilitato).
    
    Versione single-threaded per compatibilità.
    """
    
    logger.info(f"\nProcessing {csv_path.name} (SEQUENTIAL MODE)")
    logger.info(f"  Chunk size: {CHUNK_SIZE:,} righe")
    logger.info(f"  Output: {output_path}")
    
    # Stima totale righe
    estimated_rows = estimate_total_rows(csv_path)
    estimated_chunks = estimated_rows // CHUNK_SIZE + 1
    
    # Statistiche
    stats = {
        'total_rows_processed': 0,
        'total_rows_cleaned': 0,
        'chunks_processed': 0,
        'label_distribution': {'benign': 0, 'attack': 0}
    }
    
    # Processore Parquet writer
    writer = None
    schema = None
    
    start_time = time.time()
    
    try:
        # Leggi CSV a chunk
        chunk_iterator = pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
        
        for chunk_idx, chunk in enumerate(chunk_iterator, 1):
            
            stats['total_rows_processed'] += len(chunk)
            
            # Pulisci chunk
            chunk = clean_chunk(chunk)
            
            # Encode labels
            chunk = encode_labels(chunk)
            
            # Update stats
            stats['total_rows_cleaned'] += len(chunk)
            stats['label_distribution']['benign'] += (chunk['Label_Binary'] == 0).sum()
            stats['label_distribution']['attack'] += (chunk['Label_Binary'] == 1).sum()
            stats['chunks_processed'] = chunk_idx
            
            # Converti a PyArrow Table
            table = pa.Table.from_pandas(chunk)
            
            # Inizializza writer con schema dal primo chunk
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression=PARQUET_COMPRESSION
                )
            
            # Scrivi chunk
            writer.write_table(table)
            
            # Log progresso
            if chunk_idx % 10 == 0:
                progress = (chunk_idx / estimated_chunks) * 100
                elapsed = time.time() - start_time
                chunks_per_sec = chunk_idx / elapsed if elapsed > 0 else 0
                eta_seconds = (estimated_chunks - chunk_idx) / chunks_per_sec if chunks_per_sec > 0 else 0
                
                logger.info(
                    f"  Chunk {chunk_idx}/{estimated_chunks} "
                    f"({progress:.1f}%) - "
                    f"{chunks_per_sec:.1f} chunks/s - "
                    f"ETA: {eta_seconds/60:.1f}min - "
                    f"RAM: {get_memory_usage():.1f}%"
                )
            
            # Libera memoria
            del chunk, table
        
        # Chiudi writer
        if writer:
            writer.close()
        
        elapsed_total = time.time() - start_time
        
        logger.info(f"\n   Processing completato in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
        logger.info(f"  Righe processate: {stats['total_rows_processed']:,}")
        logger.info(f"  Righe pulite: {stats['total_rows_cleaned']:,}")
        logger.info(f"  Chunk processati: {stats['chunks_processed']}")
        logger.info(f"  Velocità media: {stats['chunks_processed']/elapsed_total:.2f} chunks/s")
        
        return stats
        
    except Exception as e:
        if writer:
            writer.close()
        raise e


def analyze_parquet_file(parquet_path: Path) -> dict:
    """
    Analizza file Parquet senza caricarlo tutto in memoria.
    """
    
    logger.info(f"\nAnalisi {parquet_path.name}...")
    
    # Leggi metadata
    parquet_file = pq.ParquetFile(parquet_path)
    
    info = {
        'num_rows': parquet_file.metadata.num_rows,
        'num_columns': parquet_file.metadata.num_columns,
        'num_row_groups': parquet_file.metadata.num_row_groups,
        'file_size_mb': parquet_path.stat().st_size / (1024**2)
    }
    
    logger.info(f"  Righe: {info['num_rows']:,}")
    logger.info(f"  Colonne: {info['num_columns']}")
    logger.info(f"  Row groups: {info['num_row_groups']}")
    logger.info(f"  Size: {info['file_size_mb']:.1f} MB")
    
    return info


def stratified_split_parquet(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path
) -> dict:
    """
    Split stratificato su Parquet senza caricare tutto in RAM.
    
    Strategia:
    1. Prima passata: conta sample per classe
    2. Calcola indici target per split
    3. Seconda passata: assegna sample ai set appropriati
    """
    
    logger.info("\n" + "="*70)
    logger.info("STRATIFIED SPLIT")
    logger.info("="*70)
    
    # ========================================================================
    # FASE 1: Conta sample per classe
    # ========================================================================
    
    logger.info("\nFase 1: Conteggio sample per classe...")
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    class_counts = {'benign': 0, 'attack': 0}
    
    # Leggi solo colonna Label_Binary
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=['Label_Binary']):
        df_batch = batch.to_pandas()
        class_counts['benign'] += (df_batch['Label_Binary'] == 0).sum()
        class_counts['attack'] += (df_batch['Label_Binary'] == 1).sum()
    
    logger.info(f"  Benign: {class_counts['benign']:,} ({class_counts['benign']/total_rows*100:.2f}%)")
    logger.info(f"  Attack: {class_counts['attack']:,} ({class_counts['attack']/total_rows*100:.2f}%)")
    
    # ========================================================================
    # FASE 2: Calcola target counts per split
    # ========================================================================
    
    logger.info("\nFase 2: Calcolo split stratificato...")
    
    # Calcola target per ogni classe in ogni split
    targets = {}
    for class_name, class_count in class_counts.items():
        targets[class_name] = {
            'train': int(class_count * TRAIN_RATIO),
            'val': int(class_count * VAL_RATIO),
            'test': int(class_count * TEST_RATIO)
        }
    
    logger.info(f"\nTarget counts:")
    logger.info(f"  Train: Benign={targets['benign']['train']:,}, Attack={targets['attack']['train']:,}")
    logger.info(f"  Val:   Benign={targets['benign']['val']:,}, Attack={targets['attack']['val']:,}")
    logger.info(f"  Test:  Benign={targets['benign']['test']:,}, Attack={targets['attack']['test']:,}")
    
    # ========================================================================
    # FASE 3: Assegnamento probabilistico ai set
    # ========================================================================
    
    logger.info("\nFase 3: Assegnamento sample ai set...")
    
    # Inizializza writers
    writers = {}
    schemas = {}
    
    # Contatori attuali
    current_counts = {
        'benign': {'train': 0, 'val': 0, 'test': 0},
        'attack': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # Probabilità per split (con random seed)
    np.random.seed(RANDOM_STATE)
    
    try:
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
            
            if (batch_idx + 1) % 10 == 0:
                progress = ((batch_idx + 1) * CHUNK_SIZE / total_rows) * 100
                logger.info(f"  Batch {batch_idx + 1} - {min(progress, 100):.1f}%")
            
            df_batch = batch.to_pandas()
            
            # Per ogni sample, assegna a train/val/test in modo stratificato
            split_assignment = []
            
            for _, row in df_batch.iterrows():
                label = 'benign' if row['Label_Binary'] == 0 else 'attack'
                
                # Calcola probabilità rimanenti
                remaining = {
                    'train': max(0, targets[label]['train'] - current_counts[label]['train']),
                    'val': max(0, targets[label]['val'] - current_counts[label]['val']),
                    'test': max(0, targets[label]['test'] - current_counts[label]['test'])
                }
                
                total_remaining = sum(remaining.values())
                
                if total_remaining == 0:
                    # Tutti i target raggiunti, usa proportional split
                    probs = [TRAIN_RATIO, VAL_RATIO, TEST_RATIO]
                else:
                    # Usa probabilità basate su quanto manca
                    probs = [
                        remaining['train'] / total_remaining,
                        remaining['val'] / total_remaining,
                        remaining['test'] / total_remaining
                    ]
                
                # Assegna a un set
                split = np.random.choice(['train', 'val', 'test'], p=probs)
                split_assignment.append(split)
                current_counts[label][split] += 1
            
            # Split del batch
            df_batch['split'] = split_assignment
            
            for split in ['train', 'val', 'test']:
                df_split = df_batch[df_batch['split'] == split].drop('split', axis=1)
                
                if len(df_split) > 0:
                    table = pa.Table.from_pandas(df_split)
                    
                    # Inizializza writer se necessario
                    if split not in writers:
                        schema = table.schema
                        schemas[split] = schema
                        
                        split_path = {
                            'train': train_path,
                            'val': val_path,
                            'test': test_path
                        }[split]
                        
                        writers[split] = pq.ParquetWriter(
                            split_path,
                            schema,
                            compression=PARQUET_COMPRESSION
                        )
                    
                    writers[split].write_table(table)
            
            # Libera memoria
            del df_batch
        
        # Chiudi tutti i writers
        for writer in writers.values():
            writer.close()
        
        # ====================================================================
        # FASE 4: Verifica split
        # ====================================================================
        
        logger.info("\n" + "="*70)
        logger.info("VERIFICA SPLIT")
        logger.info("="*70)
        
        split_stats = {}
        
        for split_name, split_path in [('Train', train_path), ('Val', val_path), ('Test', test_path)]:
            
            parquet_split = pq.ParquetFile(split_path)
            
            benign_count = 0
            attack_count = 0
            
            for batch in parquet_split.iter_batches(batch_size=CHUNK_SIZE, columns=['Label_Binary']):
                df_batch = batch.to_pandas()
                benign_count += (df_batch['Label_Binary'] == 0).sum()
                attack_count += (df_batch['Label_Binary'] == 1).sum()
            
            total = benign_count + attack_count
            
            logger.info(
                f"{split_name:5s}: "
                f"Benign={benign_count:>10,} ({benign_count/total*100:>5.2f}%)  "
                f"Attack={attack_count:>10,} ({attack_count/total*100:>5.2f}%)  "
                f"Total={total:>10,}"
            )
            
            split_stats[split_name.lower()] = {
                'benign': benign_count,
                'attack': attack_count,
                'total': total
            }
        
        logger.info("="*70)
        
        return split_stats
        
    except Exception as e:
        # Cleanup in caso di errore
        for writer in writers.values():
            writer.close()
        raise e


def main():
    """Pipeline preprocessing completa con chunk-based processing parallelo."""
    
    logger.info("="*70)
    logger.info("PREPROCESSING NF-UQ-NIDS-v2 (CHUNK-BASED PARALLEL)")
    logger.info("="*70)
    
    # Determina modalità processing
    if ENABLE_PARALLEL_PROCESSING:
        n_workers = get_worker_count(CPU_USAGE_PERCENT, MIN_WORKERS, MAX_WORKERS)
        logger.info(f"Modalità: PARALLEL")
        logger.info(f"Workers: {n_workers} (CPU usage: {CPU_USAGE_PERCENT*100:.0f}%)")
    else:
        n_workers = 1
        logger.info(f"Modalità: SEQUENTIAL")
    
    logger.info(f"Chunk size: {CHUNK_SIZE:,} righe")
    logger.info(f"RAM disponibile: {get_memory_usage():.1f}% usata")
    logger.info("="*70)
    
    # ========================================================================
    # STEP 1: CSV → Parquet completo
    # ========================================================================
    
    csv_files = get_csv_files()
    logger.info(f"\nTrovati {len(csv_files)} file CSV")
    
    # Path output temporaneo
    temp_parquet = PROCESSED_DATA_DIR / "full_dataset_temp.parquet"
    
    # Processa CSV → Parquet
    if len(csv_files) == 1:
        # Single file processing
        if ENABLE_PARALLEL_PROCESSING:
            stats = process_csv_to_parquet_parallel(csv_files[0], temp_parquet, n_workers)
        else:
            stats = process_csv_to_parquet_sequential(csv_files[0], temp_parquet)
    else:
        # Multiple CSV files - processa uno alla volta
        logger.info("\nMultipli CSV rilevati - processing sequenziale per file...")
        
        combined_stats = {
            'total_rows_processed': 0,
            'total_rows_cleaned': 0,
            'chunks_processed': 0,
            'label_distribution': {'benign': 0, 'attack': 0}
        }
        
        temp_parquets = []
        
        for idx, csv_file in enumerate(csv_files, 1):
            temp_output = PROCESSED_DATA_DIR / f"temp_{idx}.parquet"
            
            if ENABLE_PARALLEL_PROCESSING:
                file_stats = process_csv_to_parquet_parallel(csv_file, temp_output, n_workers)
            else:
                file_stats = process_csv_to_parquet_sequential(csv_file, temp_output)
            
            temp_parquets.append(temp_output)
            
            # Accumula stats
            for key in ['total_rows_processed', 'total_rows_cleaned', 'chunks_processed']:
                combined_stats[key] += file_stats[key]
            for label in ['benign', 'attack']:
                combined_stats['label_distribution'][label] += file_stats['label_distribution'][label]
        
        # Combina Parquet files
        logger.info("\nCombinazione file Parquet...")
        
        writer = None
        for temp_pq in temp_parquets:
            pq_file = pq.ParquetFile(temp_pq)
            
            for batch in pq_file.iter_batches(batch_size=CHUNK_SIZE):
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_parquet,
                        batch.schema,
                        compression=PARQUET_COMPRESSION
                    )
                writer.write_table(pa.Table.from_batches([batch]))
            
            # Rimuovi temp file
            temp_pq.unlink()
        
        if writer:
            writer.close()
        
        stats = combined_stats
    
    # ========================================================================
    # STEP 2: Analisi dataset completo
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("ANALISI DATASET COMPLETO")
    logger.info("="*70)
    
    parquet_info = analyze_parquet_file(temp_parquet)
    
    total = stats['total_rows_cleaned']
    benign = stats['label_distribution']['benign']
    attack = stats['label_distribution']['attack']
    
    logger.info(f"\nDistribuzione label:")
    logger.info(f"  Benign (0): {benign:>12,} ({benign/total*100:>5.2f}%)")
    logger.info(f"  Attack (1): {attack:>12,} ({attack/total*100:>5.2f}%)")
    logger.info(f"  Totale:     {total:>12,}")
    
    # Verifica presenza entrambe le classi
    if benign == 0:
        raise ValueError("Dataset invalido: manca classe 0 (benign)")
    if attack == 0:
        raise ValueError("Dataset invalido: manca classe 1 (attack)")
    
    # ========================================================================
    # STEP 3: Stratified Split
    # ========================================================================
    
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"
    
    split_stats = stratified_split_parquet(
        temp_parquet,
        train_path,
        val_path,
        test_path
    )
    
    # Rimuovi file temporaneo
    temp_parquet.unlink()
    logger.info(f"\n File temporaneo rimosso")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info(" PREPROCESSING COMPLETATO CON SUCCESSO")
    logger.info("="*70)
    logger.info(f"\nRIEPILOGO:")
    logger.info(f"  Modalità: {'PARALLEL' if ENABLE_PARALLEL_PROCESSING else 'SEQUENTIAL'}")
    if ENABLE_PARALLEL_PROCESSING:
        logger.info(f"  Workers utilizzati: {n_workers}")
    logger.info(f"  Dataset totale:     {total:>12,} righe")
    logger.info(f"  Train set:          {split_stats['train']['total']:>12,} righe")
    logger.info(f"  Validation set:     {split_stats['val']['total']:>12,} righe")
    logger.info(f"  Test set:           {split_stats['test']['total']:>12,} righe")
    logger.info(f"  Output directory:   {PROCESSED_DATA_DIR}")
    logger.info(f"\nFile Parquet generati:")
    logger.info(f"  {train_path.name} ({train_path.stat().st_size / (1024**2):.1f} MB)")
    logger.info(f"  {val_path.name} ({val_path.stat().st_size / (1024**2):.1f} MB)")
    logger.info(f"  {test_path.name} ({test_path.stat().st_size / (1024**2):.1f} MB)")
    logger.info(f"\nPROSSIMI PASSI:")
    logger.info("  1. python srcNF/feature_engineering.py")
    logger.info("  2. python srcNF/training.py --model xgboost")
    logger.info("="*70)


if __name__ == "__main__":
    main()
