"""
Preprocessing del dataset NF-UQ-NIDS-v2 con chunk-based processing.

STRATEGIA RAM-SAFE per 16GB:
1. Controlla RAM disponibile PRIMA di iniziare
2. Se RAM bassa (<4GB free) → SEQUENTIAL MODE (sicuro)
3. Se RAM OK → PARALLEL MODE con worker ridotti  
4. Monitor RAM durante processing

SEQUENTIAL MODE è la scelta SICURA - usa questa se hai dubbi!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool
import time
import gc
import os
import psutil

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LABEL_COLUMN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE,
    CHUNK_SIZE, PARQUET_COMPRESSION,
    ENABLE_PARALLEL_PROCESSING, CPU_USAGE_PERCENT, MIN_WORKERS, MAX_WORKERS
)
from utils import setup_logger, get_memory_usage, get_worker_count

logger = setup_logger(__name__, 'preprocessing.log')

# SOGLIE RAM (con 16GB totale)
MIN_FREE_RAM_GB_FOR_PARALLEL = 6.0  # Serve 6GB liberi per parallel (CONSERVATIVO)
MIN_FREE_RAM_GB_SAFE = 2.0

def check_ram_available():
    """Verifica RAM e decide modalità."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    logger.info(f"\nRAM Check:")
    logger.info(f"  Totale: {mem.total / (1024**3):.1f} GB")
    logger.info(f"  Disponibile: {available_gb:.1f} GB")
    logger.info(f"  In uso: {mem.percent:.1f}%")
    
    if available_gb < MIN_FREE_RAM_GB_SAFE:
        raise RuntimeError(f"RAM troppo bassa ({available_gb:.1f}GB)! Chiudi altre app.")
    
    # CONSERVATIVO: se <6GB liberi, usa SEQUENTIAL
    if not ENABLE_PARALLEL_PROCESSING or available_gb < MIN_FREE_RAM_GB_FOR_PARALLEL:
        logger.warning(f"\n    RAM disponibile: {available_gb:.1f}GB")
        logger.warning(f"  Usando SEQUENTIAL MODE (sicuro)")
        return False, 1, available_gb
    
    # Parallel con max 2 worker per sicurezza
    n_workers = min(2, get_worker_count(CPU_USAGE_PERCENT, MIN_WORKERS, MAX_WORKERS))
    logger.info(f"   RAM OK - usando {n_workers} workers")
    
    return True, n_workers, available_gb

def get_csv_files():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV in {RAW_DATA_DIR}")
    return csv_files

def estimate_total_rows(csv_path):
    # TODO check se calcola giusto con questa modifica
    file_size = csv_path.stat().st_size
    with open(csv_path, 'rb') as f:
        chunk = f.read(1024 * 1024)
    lines_in_chunk = chunk.count(b'\n')

    estimated_rows = int((file_size / len(chunk)) * lines_in_chunk)
    
    gc.collect()
    return estimated_rows # 75987976 righe nel dataset

def clean_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk.dropna()
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(chunk[col]).any():
            chunk = chunk[~np.isinf(chunk[col])]
    return chunk.reset_index(drop=True)

def encode_labels(chunk):
    if LABEL_COLUMN not in chunk.columns:
        raise ValueError(f"Colonna {LABEL_COLUMN} non trovata")
    chunk['Label_Original'] = chunk[LABEL_COLUMN].copy()
    chunk['Label_Binary'] = chunk[LABEL_COLUMN].copy()
    return chunk

def process_chunk_worker(chunk_data):
    chunk_idx, chunk = chunk_data
    pid = os.getpid()
    initial_rows = len(chunk)
    
    chunk = clean_chunk(chunk)
    chunk = encode_labels(chunk)
    
    stats = {
        'chunk_idx': chunk_idx,
        'pid': pid,
        'rows_processed': initial_rows,
        'rows_cleaned': len(chunk),
        'benign': (chunk['Label_Binary'] == 0).sum(),
        'attack': (chunk['Label_Binary'] == 1).sum()
    }
    return chunk_idx, chunk, stats

def process_csv_sequential(csv_path, output_path):
    """MODALITÀ SICURA - usa sempre questa se hai dubbi!"""
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING: {csv_path.name} (SEQUENTIAL - RAM SAFE)")
    logger.info(f"{'='*70}")
    
    estimated_rows = estimate_total_rows(csv_path)
    estimated_chunks = estimated_rows // CHUNK_SIZE + 1
    
    stats = {
        'total_rows_processed': 0,
        'total_rows_cleaned': 0,
        'chunks_processed': 0,
        'label_distribution': {'benign': 0, 'attack': 0}
    }
    
    writer = None
    start_time = time.time()
    
    try:
        chunk_iterator = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)
        
        for chunk_idx, chunk in enumerate(chunk_iterator, 1):
            
            # Check RAM ogni 10 chunk
            if chunk_idx % 10 == 0:
                mem_pct = get_memory_usage()
                if mem_pct > 90:
                    raise MemoryError(f"RAM critica: {mem_pct:.1f}%")
            
            stats['total_rows_processed'] += len(chunk)
            
            chunk = clean_chunk(chunk)
            chunk = encode_labels(chunk)
            
            stats['total_rows_cleaned'] += len(chunk)
            stats['label_distribution']['benign'] += (chunk['Label_Binary'] == 0).sum()
            stats['label_distribution']['attack'] += (chunk['Label_Binary'] == 1).sum()
            stats['chunks_processed'] = chunk_idx
            
            table = pa.Table.from_pandas(chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=PARQUET_COMPRESSION)
            
            writer.write_table(table)
            
            if chunk_idx % 10 == 0:
                progress = (chunk_idx / estimated_chunks) * 100
                elapsed = time.time() - start_time
                rate = chunk_idx / elapsed if elapsed > 0 else 0
                eta = (estimated_chunks - chunk_idx) / rate if rate > 0 else 0
                
                logger.info(
                    f"  Chunk {chunk_idx}/{estimated_chunks} "
                    f"({progress:.1f}%) - {rate:.1f}/s - "
                    f"ETA: {eta/60:.1f}min - RAM: {get_memory_usage():.1f}%"
                )
            
            del chunk, table
            if chunk_idx % 20 == 0:
                gc.collect()
        
        if writer:
            writer.close()
        
        logger.info(f"\n   Completato in {(time.time()-start_time)/60:.1f} min")
        gc.collect()
        return stats
        
    except Exception as e:
        if writer:
            writer.close()
        raise e

def process_csv_parallel(csv_path, output_path, n_workers):
    """PARALLELO - usa solo se hai RAM sufficiente."""
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING: {csv_path.name} (PARALLEL - {n_workers} workers)")
    logger.info(f"{'='*70}")
    
    estimated_rows = estimate_total_rows(csv_path)
    estimated_chunks = estimated_rows // CHUNK_SIZE + 1
    
    global_stats = {
        'total_rows_processed': 0,
        'total_rows_cleaned': 0,
        'chunks_processed': 0,
        'label_distribution': {'benign': 0, 'attack': 0}
    }
    
    writer = None
    start_time = time.time()
    worker_pids = set()
    
    try:
        chunk_iterator = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)
        chunk_buffer = []
        
        with Pool(processes=n_workers) as pool:
            chunk_idx = 0
            
            for chunk in chunk_iterator:
                
                # Check RAM
                if get_memory_usage() > 85:
                    logger.warning(f"\n    RAM alta - switching to SEQUENTIAL")
                    pool.terminate()
                    pool.join()
                    
                    # Salva chunk corrente e continua sequential
                    if chunk is not None:
                        chunk = clean_chunk(chunk)
                        chunk = encode_labels(chunk)
                        table = pa.Table.from_pandas(chunk)
                        if writer is None:
                            writer = pq.ParquetWriter(output_path, table.schema, compression=PARQUET_COMPRESSION)
                        writer.write_table(table)
                    
                    # Continua resto in sequential
                    for c in chunk_iterator:
                        c = clean_chunk(c)
                        c = encode_labels(c)
                        writer.write_table(pa.Table.from_pandas(c))
                        global_stats['chunks_processed'] += 1
                        del c
                        if global_stats['chunks_processed'] % 20 == 0:
                            gc.collect()
                    break
                
                chunk_buffer.append((chunk_idx, chunk))
                chunk_idx += 1
                
                if len(chunk_buffer) >= n_workers:
                    results = pool.map(process_chunk_worker, chunk_buffer)
                    results.sort(key=lambda x: x[0])
                    
                    for _, cleaned, stats in results:
                        worker_pids.add(stats['pid'])
                        global_stats['total_rows_processed'] += stats['rows_processed']
                        global_stats['total_rows_cleaned'] += stats['rows_cleaned']
                        global_stats['chunks_processed'] += 1
                        global_stats['label_distribution']['benign'] += stats['benign']
                        global_stats['label_distribution']['attack'] += stats['attack']
                        
                        table = pa.Table.from_pandas(cleaned)
                        if writer is None:
                            writer = pq.ParquetWriter(output_path, table.schema, compression=PARQUET_COMPRESSION)
                        writer.write_table(table)
                    
                    progress = (global_stats['chunks_processed'] / estimated_chunks) * 100
                    elapsed = time.time() - start_time
                    rate = global_stats['chunks_processed'] / elapsed if elapsed > 0 else 0
                    eta = (estimated_chunks - global_stats['chunks_processed']) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"  Chunk {global_stats['chunks_processed']}/{estimated_chunks} "
                        f"({progress:.1f}%) - {rate:.1f}/s - "
                        f"ETA: {eta/60:.1f}min - PIDs: {len(worker_pids)} - RAM: {get_memory_usage():.1f}%"
                    )
                    
                    chunk_buffer = []
                    gc.collect()
            
            # Processa rimanenti
            if chunk_buffer:
                results = pool.map(process_chunk_worker, chunk_buffer)
                for _, cleaned, stats in results:
                    worker_pids.add(stats['pid'])
                    global_stats['total_rows_processed'] += stats['rows_processed']
                    global_stats['total_rows_cleaned'] += stats['rows_cleaned']
                    global_stats['chunks_processed'] += 1
                    global_stats['label_distribution']['benign'] += stats['benign']
                    global_stats['label_distribution']['attack'] += stats['attack']
                    
                    table = pa.Table.from_pandas(cleaned)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression=PARQUET_COMPRESSION)
                    writer.write_table(table)
        
        if writer:
            writer.close()
        
        logger.info(f"\n   Completato - Worker PIDs: {sorted(worker_pids)}")
        gc.collect()
        return global_stats
        
    except Exception as e:
        if writer:
            writer.close()
        raise e

def stratified_split_parquet(input_path, train_path, val_path, test_path):
    """Split VETTORIZZATO."""
    logger.info("\n" + "="*70)
    logger.info("SPLIT STRATIFICATO")
    logger.info("="*70)
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    # Conta classi
    class_counts = {'benign': 0, 'attack': 0}
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=['Label_Binary']):
        df = batch.to_pandas()
        class_counts['benign'] += (df['Label_Binary'] == 0).sum()
        class_counts['attack'] += (df['Label_Binary'] == 1).sum()
        del df
    
    logger.info(f"  Benign: {class_counts['benign']:,}")
    logger.info(f"  Attack: {class_counts['attack']:,}")
    
    # Target counts
    targets = {}
    for cls, count in class_counts.items():
        targets[cls] = {
            'train': int(count * TRAIN_RATIO),
            'val': int(count * VAL_RATIO),
            'test': int(count * TEST_RATIO)
        }
    
    writers = {}
    current = {
        'benign': {'train': 0, 'val': 0, 'test': 0},
        'attack': {'train': 0, 'val': 0, 'test': 0}
    }
    rng = np.random.RandomState(RANDOM_STATE)
    
    batch_idx = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
            batch_idx += 1
            
            if batch_idx % 10 == 0:
                progress = (batch_idx * CHUNK_SIZE / total_rows) * 100
                logger.info(f"  Batch {batch_idx} - {min(progress, 100):.1f}% - RAM: {get_memory_usage():.1f}%")
                if get_memory_usage() > 90:
                    raise MemoryError("RAM critica durante split")
            
            df = batch.to_pandas()
            n = len(df)
            split_assignment = np.empty(n, dtype='U5')
            
            benign_idx = np.where(df['Label_Binary'] == 0)[0]
            attack_idx = np.where(df['Label_Binary'] == 1)[0]
            
            # Assegna benign
            for idx in benign_idx:
                rem = {
                    'train': max(0, targets['benign']['train'] - current['benign']['train']),
                    'val': max(0, targets['benign']['val'] - current['benign']['val']),
                    'test': max(0, targets['benign']['test'] - current['benign']['test'])
                }
                tot = sum(rem.values())
                probs = [rem['train']/tot, rem['val']/tot, rem['test']/tot] if tot > 0 else [TRAIN_RATIO, VAL_RATIO, TEST_RATIO]
                split = rng.choice(['train', 'val', 'test'], p=probs)
                split_assignment[idx] = split
                current['benign'][split] += 1
            
            # Assegna attack
            for idx in attack_idx:
                rem = {
                    'train': max(0, targets['attack']['train'] - current['attack']['train']),
                    'val': max(0, targets['attack']['val'] - current['attack']['val']),
                    'test': max(0, targets['attack']['test'] - current['attack']['test'])
                }
                tot = sum(rem.values())
                probs = [rem['train']/tot, rem['val']/tot, rem['test']/tot] if tot > 0 else [TRAIN_RATIO, VAL_RATIO, TEST_RATIO]
                split = rng.choice(['train', 'val', 'test'], p=probs)
                split_assignment[idx] = split
                current['attack'][split] += 1
            
            df['split'] = split_assignment
            
            for split in ['train', 'val', 'test']:
                df_split = df[df['split'] == split].drop('split', axis=1)
                if len(df_split) > 0:
                    table = pa.Table.from_pandas(df_split)
                    if split not in writers:
                        sp = {'train': train_path, 'val': val_path, 'test': test_path}[split]
                        writers[split] = pq.ParquetWriter(sp, table.schema, compression=PARQUET_COMPRESSION)
                    writers[split].write_table(table)
            
            del df, split_assignment
            if batch_idx % 20 == 0:
                gc.collect()
        
        for w in writers.values():
            w.close()
        
        gc.collect()
        
        # Verifica
        logger.info("\nVerifica split:")
        split_stats = {}
        for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
            pf = pq.ParquetFile(path)
            b = a = 0
            for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=['Label_Binary']):
                df = batch.to_pandas()
                b += (df['Label_Binary'] == 0).sum()
                a += (df['Label_Binary'] == 1).sum()
                del df
            tot = b + a
            logger.info(f"  {name:5s}: B={b:,} ({b/tot*100:.2f}%)  A={a:,} ({a/tot*100:.2f}%)  Tot={tot:,}")
            split_stats[name] = {'benign': b, 'attack': a, 'total': tot}
        
        return split_stats
        
    except Exception as e:
        for w in writers.values():
            w.close()
        raise e

def main():
    logger.info("="*70)
    logger.info("PREPROCESSING (RAM-SAFE)")
    logger.info("="*70)
    
    # CHECK RAM
    use_parallel, n_workers, avail = check_ram_available()
    
    logger.info(f"\nModalità: {'PARALLEL' if use_parallel else 'SEQUENTIAL (SICURO)'}")
    if use_parallel:
        logger.info(f"Workers: {n_workers}")
    logger.info("="*70)
    
    csv_files = get_csv_files()
    temp_parquet = PROCESSED_DATA_DIR / "full_dataset_temp.parquet"
    
    # Process
    if use_parallel:
        stats = process_csv_parallel(csv_files[0], temp_parquet, n_workers)
    else:
        stats = process_csv_sequential(csv_files[0], temp_parquet)
    
    # Analisi
    logger.info("\n" + "="*70)
    logger.info("DATASET COMPLETO")
    logger.info("="*70)
    
    total = stats['total_rows_cleaned']
    benign = stats['label_distribution']['benign']
    attack = stats['label_distribution']['attack']
    
    logger.info(f"Benign: {benign:,} ({benign/total*100:.2f}%)")
    logger.info(f"Attack: {attack:,} ({attack/total*100:.2f}%)")
    logger.info(f"Totale: {total:,}")
    
    # Split
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"
    
    split_stats = stratified_split_parquet(temp_parquet, train_path, val_path, test_path)
    
    temp_parquet.unlink()
    gc.collect()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info(" COMPLETATO")
    logger.info("="*70)
    logger.info(f"Modalità: {'PARALLEL' if use_parallel else 'SEQUENTIAL'}")
    logger.info(f"Train: {split_stats['train']['total']:,}")
    logger.info(f"Val:   {split_stats['val']['total']:,}")
    logger.info(f"Test:  {split_stats['test']['total']:,}")
    logger.info("\nProssimo: python srcNF/feature_engineering.py")
    logger.info("="*70)

if __name__ == "__main__":
    main()