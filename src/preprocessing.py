"""
================================================================================
NIDS-ML - Modulo Preprocessing
================================================================================

Preprocessing del dataset CIC-IDS2017.

GUIDA PARAMETRI:
----------------
    python src/preprocessing.py [opzioni]

Opzioni:
    --input-dir PATH      Directory CSV raw (default: data/raw)
    --output-dir PATH     Directory output (default: data/processed)
    --balance-ratio FLOAT Rapporto majority:minority (default: 2.0)
    --no-balance          Disabilita bilanciamento
    --chunk-size INT      Righe per chunk (default: no chunking)
    --test-size FLOAT     Proporzione test (default: 0.15)
    --val-size FLOAT      Proporzione validation (default: 0.15)
    --n-jobs INT          Core CPU (default: auto)
    --random-state INT    Seed (default: 42)

ESEMPI:
-------
python src/preprocessing.py
python src/preprocessing.py --no-balance
python src/preprocessing.py --chunk-size 100000 --n-jobs 4
python src/preprocessing.py --balance-ratio 3.0

================================================================================
"""

# ==============================================================================
# SETUP LIMITI CPU - PRIMA DI TUTTO
# ==============================================================================
import sys
import os
import argparse
from pathlib import Path

def _get_arg(name, default=None, arg_type=str):
    for i, arg in enumerate(sys.argv):
        if arg == f'--{name}' and i + 1 < len(sys.argv):
            try:
                return arg_type(sys.argv[i + 1])
            except ValueError:
                return default
    return default

# Applica limiti CPU PRIMA di importare pandas/sklearn
_n_jobs_arg = _get_arg('n-jobs', None, int)
_n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

# Variabili d'ambiente
os.environ['OMP_NUM_THREADS'] = str(_n_cores)
os.environ['MKL_NUM_THREADS'] = str(_n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(_n_cores)
os.environ['LOKY_MAX_CPU_COUNT'] = str(_n_cores)

# Ora importa il resto
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Generator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import gc
import warnings

# Import utils DOPO aver impostato le variabili d'ambiente
from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    COLUMNS_TO_DROP,
    sizeof_fmt,
    apply_cpu_limits,
    ResourceMonitor,
    suppress_warnings
)
from src.timing import TimingLogger

# Applica limiti CPU (affinity + priority)
apply_cpu_limits(_n_cores, set_low_priority=True)

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CARICAMENTO DATI
# ==============================================================================

def load_csv_chunked(filepath: Path, 
                     chunk_size: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """Carica CSV in chunk per RAM limitata."""
    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    chunks = pd.read_csv(filepath, low_memory=False, encoding='utf-8', chunksize=chunk_size)
    
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip()
        yield chunk


def load_single_csv(filepath: Path) -> pd.DataFrame:
    """Carica un singolo CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
    df.columns = df.columns.str.strip()
    return df


def load_all_csv(raw_dir: Path, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """Carica e concatena tutti i CSV."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV trovato in {raw_dir}")
    
    logger.info(f"Trovati {len(csv_files)} file CSV")
    
    dataframes = []
    
    if chunk_size:
        logger.info(f"Caricamento chunked (chunk_size={chunk_size:,})")
        for csv_path in tqdm(csv_files, desc="Caricamento CSV"):
            file_chunks = []
            for chunk in load_csv_chunked(csv_path, chunk_size):
                file_chunks.append(chunk)
            if file_chunks:
                dataframes.append(pd.concat(file_chunks, ignore_index=True))
                del file_chunks
                gc.collect()
    else:
        for csv_path in tqdm(csv_files, desc="Caricamento CSV"):
            df = load_single_csv(csv_path)
            dataframes.append(df)
            logger.debug(f"{csv_path.name}: {len(df):,} righe")
    
    logger.info(f"Concatenazione {len(dataframes)} DataFrame...")
    combined = pd.concat(dataframes, ignore_index=True)
    
    del dataframes
    gc.collect()
    
    logger.info(f"Dataset combinato: {len(combined):,} righe, {len(combined.columns)} colonne")
    logger.info(f"Memoria: {sizeof_fmt(combined.memory_usage(deep=True).sum())}")
    
    return combined


# ==============================================================================
# PULIZIA DATI
# ==============================================================================

def clean_data(df: pd.DataFrame,
               columns_to_drop: List[str] = None,
               fill_nan_value: float = 0.0,
               remove_duplicates: bool = True) -> pd.DataFrame:
    """Pulizia completa del dataset."""
    logger.info("Inizio pulizia dati...")
    initial_rows = len(df)
    
    if columns_to_drop is None:
        columns_to_drop = COLUMNS_TO_DROP
    
    # Rimuovi colonne identificative
    cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Rimosse {len(cols_to_drop)} colonne identificative")
    
    # Trova colonna Label
    label_col = None
    for possible_label in ['Label', ' Label', 'label', 'LABEL']:
        if possible_label in df.columns:
            label_col = possible_label
            break
    
    if label_col is None:
        raise KeyError("Colonna 'Label' non trovata nel dataset")
    
    if label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
    
    # Colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Rimuovi righe con infiniti
    inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
    inf_count = inf_mask.sum()
    if inf_count > 0:
        df = df[~inf_mask]
        logger.info(f"Rimosse {inf_count:,} righe con valori infiniti")
    
    # Sostituisci NaN
    nan_before = df[numeric_cols].isna().sum().sum()
    if nan_before > 0:
        df[numeric_cols] = df[numeric_cols].fillna(fill_nan_value)
        logger.info(f"Sostituiti {nan_before:,} valori NaN con {fill_nan_value}")
    
    # Rimuovi duplicati
    if remove_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"Rimosse {duplicates:,} righe duplicate")
    
    # Assicura tipi numerici
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_nan_value)
    
    final_rows = len(df)
    logger.info(f"Pulizia completata: {initial_rows:,} -> {final_rows:,} righe")
    
    return df.reset_index(drop=True)


# ==============================================================================
# ENCODING LABEL
# ==============================================================================

def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Encoding label per classificazione binaria e multiclasse."""
    logger.info("Encoding label...")
    
    df['Label_Original'] = df['Label'].copy()
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    unique_labels = sorted(df['Label'].unique())
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    
    df['Label_Multiclass'] = df['Label'].map(label_to_int)
    
    n_benign = (df['Label_Binary'] == 0).sum()
    n_attack = (df['Label_Binary'] == 1).sum()
    
    logger.info(f"Classi trovate: {len(unique_labels)}")
    logger.info(f"Distribuzione binaria: Benign={n_benign:,}, Attack={n_attack:,}")
    
    mappings = {
        'binary': {'BENIGN': 0, 'ATTACK': 1},
        'multiclass': label_to_int,
        'multiclass_inverse': int_to_label
    }
    
    return df, mappings


# ==============================================================================
# BILANCIAMENTO
# ==============================================================================

def balance_dataset(df: pd.DataFrame,
                    ratio: float = 2.0,
                    label_col: str = 'Label_Binary',
                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Bilancia dataset tramite undersampling."""
    logger.info(f"Bilanciamento dataset (ratio {ratio}:1)...")
    
    class_counts = df[label_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]
    target_majority = int(minority_count * ratio)
    
    minority_df = df[df[label_col] == minority_class]
    majority_df = df[df[label_col] == majority_class]
    
    if majority_count > target_majority:
        majority_df = majority_df.sample(n=target_majority, random_state=random_state)
        logger.info(f"Undersampling classe {majority_class}: {majority_count:,} -> {target_majority:,}")
    
    balanced = pd.concat([minority_df, majority_df], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Dataset bilanciato: {len(balanced):,} righe")
    
    return balanced


# ==============================================================================
# SPLIT
# ==============================================================================

def split_dataset(df: pd.DataFrame,
                  test_size: float = TEST_SIZE,
                  val_size: float = VAL_SIZE,
                  label_col: str = 'Label_Binary',
                  random_state: int = RANDOM_STATE
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split stratificato train/val/test."""
    logger.info(f"Split dataset ({int((1-test_size-val_size)*100)}/{int(val_size*100)}/{int(test_size*100)})...")
    
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[label_col], random_state=random_state
    )
    
    logger.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ==============================================================================
# SALVATAGGIO/CARICAMENTO
# ==============================================================================

def save_processed_data(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                        mappings: dict, output_dir: Path = None) -> None:
    """Salva dataset processati."""
    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    
    with open(output_dir / "label_mappings.json", 'w') as f:
        json.dump(mappings, f, indent=2)
    
    logger.info(f"Dataset salvati in {output_dir}")


def load_processed_data(processed_dir: Path = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Carica dataset processati."""
    if processed_dir is None:
        processed_dir = get_project_root() / "data" / "processed"
    
    required = ['train.parquet', 'val.parquet', 'test.parquet', 'label_mappings.json']
    for f in required:
        if not (processed_dir / f).exists():
            raise FileNotFoundError(f"File non trovato: {processed_dir / f}")
    
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    
    with open(processed_dir / "label_mappings.json", 'r') as f:
        mappings = json.load(f)
    
    logger.info(f"Caricati: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    
    return train, val, test, mappings


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Preprocessing CIC-IDS2017',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--balance-ratio', type=float, default=2.0)
    parser.add_argument('--no-balance', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--test-size', type=float, default=TEST_SIZE)
    parser.add_argument('--val-size', type=float, default=VAL_SIZE)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    input_dir = args.input_dir or get_project_root() / "data" / "raw"
    output_dir = args.output_dir or get_project_root() / "data" / "processed"
    
    monitor = ResourceMonitor()
    
    # Inizializza timing logger
    timer = TimingLogger("preprocessing", parameters={
        'balance_ratio': args.balance_ratio,
        'no_balance': args.no_balance,
        'chunk_size': args.chunk_size,
        'n_jobs': _n_cores
    })
    
    print("\n" + "=" * 60)
    print("PREPROCESSING CIC-IDS2017")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Input:         {input_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Balance:       {'No' if args.no_balance else f'Si (ratio {args.balance_ratio}:1)'}")
    print(f"  Chunk size:    {args.chunk_size or 'Disabilitato'}")
    print(f"  Split:         {int((1-args.test_size-args.val_size)*100)}/{int(args.val_size*100)}/{int(args.test_size*100)}")
    print(f"  CPU cores:     {_n_cores}/{os.cpu_count()}")
    print()
    
    try:
        print(f"1. Caricamento CSV da {input_dir}...")
        with timer.time_operation("caricamento_csv"):
            df = load_all_csv(input_dir, chunk_size=args.chunk_size)
        
        print("\n2. Pulizia dati...")
        with timer.time_operation("pulizia_dati"):
            df = clean_data(df)
        
        print("\n3. Encoding label...")
        with timer.time_operation("encoding_label"):
            df, mappings = encode_labels(df)
        
        print("\n   Distribuzione classi:")
        class_dist = df['Label_Original'].value_counts()
        for label, count in class_dist.items():
            pct = count / len(df) * 100
            print(f"   - {label}: {count:,} ({pct:.2f}%)")
        
        if not args.no_balance:
            print(f"\n4. Bilanciamento (ratio {args.balance_ratio}:1)...")
            with timer.time_operation("bilanciamento"):
                df = balance_dataset(df, ratio=args.balance_ratio, random_state=args.random_state)
        else:
            print("\n4. Bilanciamento saltato")
        
        print("\n5. Split train/val/test...")
        with timer.time_operation("split_dataset"):
            train, val, test = split_dataset(
                df, args.test_size, args.val_size, random_state=args.random_state
            )
        
        print("\n6. Salvataggio...")
        with timer.time_operation("salvataggio"):
            save_processed_data(train, val, test, mappings, output_dir)
        
        # Salva timing
        timer.add_metric("train_rows", len(train))
        timer.add_metric("val_rows", len(val))
        timer.add_metric("test_rows", len(test))
        timing_path = timer.save()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETATO")
        print("=" * 60)
        print(f"\nOutput: {output_dir}")
        print(f"Train:  {len(train):,} righe")
        print(f"Val:    {len(val):,} righe")
        print(f"Test:   {len(test):,} righe")
        print(f"Timing: {timing_path}")
        print(f"\nProssimo step: python src/feature_engineering.py")
        
        timer.print_summary()
        monitor.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Inserire i CSV in data/raw/")
        sys.exit(1)


if __name__ == "__main__":
    main()