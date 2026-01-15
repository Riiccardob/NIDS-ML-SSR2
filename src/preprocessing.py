"""
================================================================================
NIDS-ML - Modulo Preprocessing
================================================================================

Preprocessing del dataset CIC-IDS2017: caricamento, pulizia, encoding, split.

GUIDA PARAMETRI:
----------------
Dalla linea di comando, tutti i parametri sono configurabili:

    python src/preprocessing.py [opzioni]

Opzioni disponibili:
    --input-dir PATH      Directory con CSV raw (default: data/raw)
    --output-dir PATH     Directory output processati (default: data/processed)
    --balance-ratio FLOAT Rapporto majority:minority (default: 2.0)
    --no-balance          Disabilita bilanciamento
    --chunk-size INT      Righe per chunk durante caricamento (default: None = no chunking)
    --test-size FLOAT     Proporzione test set (default: 0.15)
    --val-size FLOAT      Proporzione validation set (default: 0.15)
    --max-ram INT         Limite RAM in percentuale (default: 85)
    --random-state INT    Seed per reproducibilita (default: 42)

ESEMPI:
-------
# Esecuzione standard
python src/preprocessing.py

# Con chunking per RAM limitata (carica 100k righe alla volta)
python src/preprocessing.py --chunk-size 100000

# Senza bilanciamento (usa tutto il dataset)
python src/preprocessing.py --no-balance

# Rapporto bilanciamento diverso (3:1 invece di 2:1)
python src/preprocessing.py --balance-ratio 3.0

# Split diverso (80/10/10)
python src/preprocessing.py --test-size 0.10 --val-size 0.10

================================================================================
"""

import sys
from pathlib import Path
import argparse

# Setup path per import locali
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Generator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import gc

from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    COLUMNS_TO_DROP,
    sizeof_fmt,
    ResourceMonitor,
    suppress_warnings
)

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CARICAMENTO DATI
# ==============================================================================

def load_csv_chunked(filepath: Path, 
                     chunk_size: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """
    Carica un CSV in chunk per gestire file grandi con RAM limitata.
    
    Questo generatore permette di processare file CSV molto grandi
    senza caricarli interamente in memoria. Ogni chunk viene restituito
    come DataFrame indipendente.
    
    Args:
        filepath: Path al file CSV
        chunk_size: Numero di righe per chunk (default 100000)
    
    Yields:
        DataFrame con chunk_size righe (ultimo chunk puo essere minore)
    
    Raises:
        FileNotFoundError: Se il file non esiste
    
    Example:
        for chunk in load_csv_chunked("data.csv", chunk_size=50000):
            process(chunk)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    # Legge in chunk usando iteratore pandas
    chunks = pd.read_csv(
        filepath, 
        low_memory=False, 
        encoding='utf-8',
        chunksize=chunk_size
    )
    
    for chunk in chunks:
        # Normalizza nomi colonne (rimuove spazi)
        chunk.columns = chunk.columns.str.strip()
        yield chunk


def load_single_csv(filepath: Path) -> pd.DataFrame:
    """
    Carica un singolo file CSV interamente in memoria.
    
    Metodo standard per file che entrano in RAM. Normalizza
    automaticamente i nomi delle colonne rimuovendo spazi.
    
    Args:
        filepath: Path al file CSV
    
    Returns:
        DataFrame con tutti i dati
    
    Raises:
        FileNotFoundError: Se il file non esiste
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
    df.columns = df.columns.str.strip()
    return df


def load_all_csv(raw_dir: Path,
                 chunk_size: Optional[int] = None,
                 max_ram_percent: int = 85) -> pd.DataFrame:
    """
    Carica e concatena tutti i CSV da una directory.
    
    Supporta due modalita:
    1. Standard: carica ogni file interamente, poi concatena
    2. Chunked: carica in blocchi per RAM limitata
    
    Il monitor risorse verifica che la RAM non superi il limite
    configurato durante il caricamento.
    
    Args:
        raw_dir: Directory contenente i file CSV
        chunk_size: Se specificato, usa caricamento chunked
        max_ram_percent: Limite massimo RAM (default 85%)
    
    Returns:
        DataFrame concatenato con tutti i dati
    
    Raises:
        FileNotFoundError: Se directory vuota o non esiste
        MemoryError: Se RAM supera limite durante caricamento
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV trovato in {raw_dir}")
    
    logger.info(f"Trovati {len(csv_files)} file CSV")
    
    monitor = ResourceMonitor(max_ram=max_ram_percent)
    dataframes = []
    
    if chunk_size:
        # Modalita chunked - per RAM limitata
        logger.info(f"Caricamento chunked (chunk_size={chunk_size:,})")
        
        for csv_path in tqdm(csv_files, desc="Caricamento CSV"):
            file_chunks = []
            for chunk in load_csv_chunked(csv_path, chunk_size):
                file_chunks.append(chunk)
                
                # Verifica RAM dopo ogni chunk
                if not monitor.check_resources():
                    logger.warning("RAM alta, eseguo garbage collection...")
                    gc.collect()
                    if not monitor.wait_for_resources(timeout_seconds=60):
                        raise MemoryError("RAM oltre limite, impossibile continuare")
            
            # Concatena chunk del singolo file
            if file_chunks:
                dataframes.append(pd.concat(file_chunks, ignore_index=True))
                del file_chunks
                gc.collect()
    else:
        # Modalita standard - carica tutto in memoria
        for csv_path in tqdm(csv_files, desc="Caricamento CSV"):
            try:
                df = load_single_csv(csv_path)
                dataframes.append(df)
                logger.debug(f"{csv_path.name}: {len(df):,} righe")
            except Exception as e:
                logger.error(f"Errore caricamento {csv_path.name}: {e}")
                raise
    
    logger.info(f"Concatenazione {len(dataframes)} DataFrame...")
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Libera memoria dai DataFrame intermedi
    del dataframes
    gc.collect()
    
    logger.info(f"Dataset combinato: {len(combined):,} righe, {len(combined.columns)} colonne")
    logger.info(f"Memoria: {sizeof_fmt(combined.memory_usage(deep=True).sum())}")
    monitor.log_status(logger)
    
    return combined


# ==============================================================================
# PULIZIA DATI
# ==============================================================================

def clean_data(df: pd.DataFrame,
               columns_to_drop: List[str] = None,
               fill_nan_value: float = 0.0,
               remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Pulizia completa del dataset.
    
    Esegue le seguenti operazioni nell'ordine:
    1. Rimozione colonne identificative (non predittive)
    2. Normalizzazione nome colonna Label
    3. Rimozione righe con valori infiniti
    4. Sostituzione NaN con valore specificato
    5. Rimozione duplicati (opzionale)
    6. Conversione colonne numeriche
    
    Args:
        df: DataFrame grezzo da pulire
        columns_to_drop: Lista colonne da rimuovere (default: COLUMNS_TO_DROP)
        fill_nan_value: Valore per sostituire NaN (default: 0.0)
        remove_duplicates: Se True rimuove righe duplicate (default: True)
    
    Returns:
        DataFrame pulito
    
    Raises:
        KeyError: Se colonna Label non trovata
    """
    logger.info("Inizio pulizia dati...")
    initial_rows = len(df)
    
    if columns_to_drop is None:
        columns_to_drop = COLUMNS_TO_DROP
    
    # 1. Rimuovi colonne identificative
    cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Rimosse {len(cols_to_drop)} colonne identificative: {cols_to_drop}")
    
    # 2. Trova e normalizza colonna Label
    label_col = None
    for possible_label in ['Label', ' Label', 'label', 'LABEL']:
        if possible_label in df.columns:
            label_col = possible_label
            break
    
    if label_col is None:
        raise KeyError("Colonna 'Label' non trovata nel dataset. "
                       f"Colonne disponibili: {list(df.columns)}")
    
    if label_col != 'Label':
        df = df.rename(columns={label_col: 'Label'})
        logger.debug(f"Rinominata colonna '{label_col}' -> 'Label'")
    
    # 3. Identifica colonne numeriche (esclusa Label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 4. Rimuovi righe con valori infiniti
    inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
    inf_count = inf_mask.sum()
    if inf_count > 0:
        df = df[~inf_mask]
        logger.info(f"Rimosse {inf_count:,} righe con valori infiniti")
    
    # 5. Sostituisci NaN
    nan_before = df[numeric_cols].isna().sum().sum()
    if nan_before > 0:
        df[numeric_cols] = df[numeric_cols].fillna(fill_nan_value)
        logger.info(f"Sostituiti {nan_before:,} valori NaN con {fill_nan_value}")
    
    # 6. Rimuovi duplicati
    if remove_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"Rimosse {duplicates:,} righe duplicate")
    
    # 7. Assicura tipi numerici corretti
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_nan_value)
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    logger.info(f"Pulizia completata: {initial_rows:,} -> {final_rows:,} righe "
                f"(-{removed:,}, {removed/initial_rows*100:.1f}%)")
    
    return df.reset_index(drop=True)


# ==============================================================================
# ENCODING LABEL
# ==============================================================================

def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encoding delle label per classificazione binaria e multiclasse.
    
    Crea tre colonne di label:
    - Label_Original: label testuale originale (backup)
    - Label_Binary: 0=BENIGN, 1=ATTACK (qualsiasi attacco)
    - Label_Multiclass: encoding numerico per ogni tipo di attacco
    
    Args:
        df: DataFrame con colonna 'Label'
    
    Returns:
        Tuple contenente:
        - DataFrame con colonne label aggiunte
        - Dizionario con mapping label (binary, multiclass, inverse)
    
    Example:
        df, mappings = encode_labels(df)
        # mappings['multiclass'] = {'BENIGN': 0, 'DDoS': 1, ...}
        # mappings['multiclass_inverse'] = {0: 'BENIGN', 1: 'DDoS', ...}
    """
    logger.info("Encoding label...")
    
    # Backup label originale
    df['Label_Original'] = df['Label'].copy()
    
    # Binary: BENIGN vs tutto il resto
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Multiclass: ogni tipo di attacco ha un ID
    unique_labels = sorted(df['Label'].unique())
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    
    df['Label_Multiclass'] = df['Label'].map(label_to_int)
    
    # Statistiche
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
# BILANCIAMENTO DATASET
# ==============================================================================

def balance_dataset(df: pd.DataFrame,
                    ratio: float = 2.0,
                    label_col: str = 'Label_Binary',
                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Bilancia il dataset tramite undersampling della classe maggioritaria.
    
    L'undersampling riduce la classe maggioritaria (tipicamente BENIGN)
    per ottenere un rapporto controllato con la classe minoritaria (ATTACK).
    Questo previene che il modello impari semplicemente a predire sempre
    la classe maggioritaria.
    
    Args:
        df: DataFrame con label encoded
        ratio: Rapporto desiderato majority:minority (default 2.0 = 2:1)
               - 1.0 = classi bilanciate
               - 2.0 = majority e il doppio di minority
               - 3.0 = majority e il triplo di minority
        label_col: Nome colonna label per bilanciamento
        random_state: Seed per reproducibilita sampling
    
    Returns:
        DataFrame bilanciato con righe shufflate
    
    Example:
        # Da 1.9M benign + 336k attack a 672k benign + 336k attack (ratio 2:1)
        df_balanced = balance_dataset(df, ratio=2.0)
    """
    logger.info(f"Bilanciamento dataset (ratio {ratio}:1)...")
    
    class_counts = df[label_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]
    target_majority = int(minority_count * ratio)
    
    # Separa classi
    minority_df = df[df[label_col] == minority_class]
    majority_df = df[df[label_col] == majority_class]
    
    # Undersample solo se necessario
    if majority_count > target_majority:
        majority_df = majority_df.sample(n=target_majority, random_state=random_state)
        logger.info(f"Undersampling classe {majority_class}: "
                    f"{majority_count:,} -> {target_majority:,}")
    else:
        logger.info(f"Classe {majority_class} gia sotto target, nessun undersampling")
    
    # Ricombina e shuffle
    balanced = pd.concat([minority_df, majority_df], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Dataset bilanciato: {len(balanced):,} righe "
                f"(minority={len(minority_df):,}, majority={len(majority_df):,})")
    
    return balanced


# ==============================================================================
# SPLIT DATASET
# ==============================================================================

def split_dataset(df: pd.DataFrame,
                  test_size: float = TEST_SIZE,
                  val_size: float = VAL_SIZE,
                  label_col: str = 'Label_Binary',
                  random_state: int = RANDOM_STATE
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split stratificato in train/validation/test.
    
    Lo split stratificato mantiene le proporzioni delle classi
    in tutti i set. Questo e importante per dataset sbilanciati
    per assicurare che ogni set abbia esempi di tutte le classi.
    
    Con default (test=0.15, val=0.15):
    - Train: 70%
    - Validation: 15% 
    - Test: 15%
    
    Args:
        df: DataFrame da splittare
        test_size: Proporzione test set (default 0.15)
        val_size: Proporzione validation set (default 0.15)
        label_col: Colonna per stratificazione
        random_state: Seed per reproducibilita
    
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    logger.info(f"Split dataset ({int((1-test_size-val_size)*100)}/"
                f"{int(val_size*100)}/{int(test_size*100)})...")
    
    # Primo split: separa test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )
    
    # Secondo split: separa validation da train
    # Ricalcola proporzione perche val_size e relativa al totale
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val[label_col],
        random_state=random_state
    )
    
    logger.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    
    # Verifica proporzioni classi
    for name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
        class_ratio = split_df[label_col].value_counts(normalize=True)
        logger.debug(f"{name} - Classe 0: {class_ratio.get(0, 0)*100:.1f}%, "
                     f"Classe 1: {class_ratio.get(1, 0)*100:.1f}%")
    
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True)
    )


# ==============================================================================
# SALVATAGGIO E CARICAMENTO
# ==============================================================================

def save_processed_data(train: pd.DataFrame,
                        val: pd.DataFrame,
                        test: pd.DataFrame,
                        mappings: dict,
                        output_dir: Path = None) -> None:
    """
    Salva dataset processati in formato Parquet e mapping in JSON.
    
    Parquet e preferito a CSV perche:
    - Compressione efficiente (file piu piccoli)
    - Preserva tipi dati
    - Caricamento piu veloce
    
    Args:
        train: Training DataFrame
        val: Validation DataFrame
        test: Test DataFrame
        mappings: Dizionario mapping label
        output_dir: Directory output (default: data/processed)
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva DataFrame in Parquet
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    
    # Salva mapping in JSON
    with open(output_dir / "label_mappings.json", 'w') as f:
        json.dump(mappings, f, indent=2)
    
    logger.info(f"Dataset salvati in {output_dir}")
    
    # Log dimensioni file
    for name in ['train', 'val', 'test']:
        file_path = output_dir / f"{name}.parquet"
        size = file_path.stat().st_size
        logger.debug(f"  {name}.parquet: {sizeof_fmt(size)}")


def load_processed_data(processed_dir: Path = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Carica dataset processati da Parquet.
    
    Args:
        processed_dir: Directory con i file processati
    
    Returns:
        Tuple (train, val, test, mappings)
    
    Raises:
        FileNotFoundError: Se file non trovati
    """
    if processed_dir is None:
        processed_dir = get_project_root() / "data" / "processed"
    
    # Verifica esistenza file
    required_files = ['train.parquet', 'val.parquet', 'test.parquet', 'label_mappings.json']
    for f in required_files:
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
    """
    Parse argomenti da linea di comando.
    
    Returns:
        Namespace con tutti i parametri configurati
    """
    parser = argparse.ArgumentParser(
        description='Preprocessing dataset CIC-IDS2017',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python src/preprocessing.py
  python src/preprocessing.py --chunk-size 100000
  python src/preprocessing.py --no-balance
  python src/preprocessing.py --balance-ratio 3.0 --test-size 0.10
        """
    )
    
    parser.add_argument(
        '--input-dir', 
        type=Path, 
        default=None,
        help='Directory con CSV raw (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir', 
        type=Path, 
        default=None,
        help='Directory output (default: data/processed)'
    )
    parser.add_argument(
        '--balance-ratio', 
        type=float, 
        default=2.0,
        help='Rapporto majority:minority (default: 2.0)'
    )
    parser.add_argument(
        '--no-balance', 
        action='store_true',
        help='Disabilita bilanciamento dataset'
    )
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=None,
        help='Righe per chunk (default: None = no chunking)'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=TEST_SIZE,
        help=f'Proporzione test set (default: {TEST_SIZE})'
    )
    parser.add_argument(
        '--val-size', 
        type=float, 
        default=VAL_SIZE,
        help=f'Proporzione validation set (default: {VAL_SIZE})'
    )
    parser.add_argument(
        '--max-ram', 
        type=int, 
        default=85,
        help='Limite RAM in percentuale (default: 85)'
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
    """Funzione principale per esecuzione da linea di comando."""
    args = parse_arguments()
    
    # Setup directory
    input_dir = args.input_dir or get_project_root() / "data" / "raw"
    output_dir = args.output_dir or get_project_root() / "data" / "processed"
    
    print("\n" + "=" * 60)
    print("PREPROCESSING CIC-IDS2017")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Input:         {input_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Balance:       {'No' if args.no_balance else f'Si (ratio {args.balance_ratio}:1)'}")
    print(f"  Chunk size:    {args.chunk_size or 'Disabilitato'}")
    print(f"  Split:         {int((1-args.test_size-args.val_size)*100)}/"
          f"{int(args.val_size*100)}/{int(args.test_size*100)}")
    print(f"  Max RAM:       {args.max_ram}%")
    print(f"  Random state:  {args.random_state}")
    print()
    
    try:
        # 1. Caricamento
        print(f"1. Caricamento CSV da {input_dir}...")
        df = load_all_csv(
            input_dir, 
            chunk_size=args.chunk_size,
            max_ram_percent=args.max_ram
        )
        
        # 2. Pulizia
        print("\n2. Pulizia dati...")
        df = clean_data(df)
        
        # 3. Encoding
        print("\n3. Encoding label...")
        df, mappings = encode_labels(df)
        
        # 4. Stampa distribuzione
        print("\n   Distribuzione classi:")
        class_dist = df['Label_Original'].value_counts()
        for label, count in class_dist.items():
            pct = count / len(df) * 100
            print(f"   - {label}: {count:,} ({pct:.2f}%)")
        
        # 5. Bilanciamento (opzionale)
        if not args.no_balance:
            print(f"\n4. Bilanciamento (ratio {args.balance_ratio}:1)...")
            df = balance_dataset(
                df, 
                ratio=args.balance_ratio, 
                label_col='Label_Binary',
                random_state=args.random_state
            )
        else:
            print("\n4. Bilanciamento saltato")
        
        # 6. Split
        print("\n5. Split train/val/test...")
        train, val, test = split_dataset(
            df, 
            test_size=args.test_size,
            val_size=args.val_size,
            label_col='Label_Binary',
            random_state=args.random_state
        )
        
        # 7. Salvataggio
        print("\n6. Salvataggio...")
        save_processed_data(train, val, test, mappings, output_dir)
        
        # Report finale
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETATO")
        print("=" * 60)
        print(f"\nOutput: {output_dir}")
        print(f"Train:  {len(train):,} righe")
        print(f"Val:    {len(val):,} righe")
        print(f"Test:   {len(test):,} righe")
        print(f"\nProssimo step: python src/feature_engineering.py")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Inserire i CSV del dataset CIC-IDS2017 in data/raw/")
        sys.exit(1)
    except MemoryError as e:
        print(f"\nERRORE MEMORIA: {e}")
        print("Provare con --chunk-size 100000 per ridurre uso RAM")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()