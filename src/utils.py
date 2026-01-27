"""
================================================================================
NIDS-ML - Modulo Utility
================================================================================

Funzioni comuni per logging, gestione risorse, costanti globali.

Questo modulo fornisce:
- Logger configurabile con output su console e file
- Monitoraggio risorse (CPU, RAM) con limiti configurabili
- Costanti globali per reproducibilita e configurazione
- Funzioni helper per path e formattazione

================================================================================
"""

import logging
import sys
import os
import psutil
from pathlib import Path
from datetime import datetime
import warnings
import json
from typing import Optional, Dict, List

# ==============================================================================
# COSTANTI GLOBALI
# ==============================================================================

# Seed per reproducibilita - usato in tutti gli split e training
RANDOM_STATE = 42

# Proporzioni split dataset
TEST_SIZE = 0.15   # 15% per test
VAL_SIZE = 0.15    # 15% per validation (del rimanente dopo test)

# Colonne da rimuovere durante preprocessing (identificatori, non feature)
# Queste colonne non hanno valore predittivo e causerebbero overfitting
COLUMNS_TO_DROP = [
    'Flow ID',           # Identificatore univoco flusso
    'Source IP',         # IP sorgente - causerebbe overfitting su IP specifici
    'Src IP',            # Alias
    'Destination IP',    # IP destinazione
    'Dst IP',            # Alias
    'Source Port',       # Porta sorgente - troppo variabile
    'Src Port',          # Alias
    'Destination Port',  # Porta destinazione
    'Dst Port',          # Alias
    'Timestamp',         # Timestamp - non generalizzabile
    'Protocol'           # Categorica - gestita separatamente se necessario
]

# Colonne label (escluse dalle feature)
LABEL_COLUMNS = ['Label', 'Label_Original', 'Label_Binary', 'Label_Multiclass']

# Limiti risorse di default
DEFAULT_MAX_CPU_PERCENT = 85
DEFAULT_MAX_RAM_PERCENT = 85


# ==============================================================================
# GESTIONE RISORSE
# ==============================================================================

def set_cpu_affinity(n_cores: int) -> bool:
    """
    Limita il processo corrente a usare solo N core specifici.
    
    Questo e il metodo PIU EFFICACE per limitare la CPU perche agisce
    a livello di sistema operativo, non a livello di libreria.
    
    Args:
        n_cores: Numero di core da usare
    
    Returns:
        True se applicato con successo, False altrimenti
    """
    try:
        import psutil
        p = psutil.Process()
        total = psutil.cpu_count()
        
        # Seleziona i primi N core
        cores_to_use = list(range(min(n_cores, total)))
        p.cpu_affinity(cores_to_use)
        
        return True
    except Exception:
        return False


def set_process_priority_low() -> bool:
    """
    Imposta priorita bassa per il processo corrente.
    
    Questo permette ad altri processi (UI, sistema) di avere precedenza.
    
    Returns:
        True se applicato con successo
    """
    try:
        import psutil
        p = psutil.Process()
        # BELOW_NORMAL su Windows, nice 10 su Linux
        if sys.platform == 'win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)  # Valore piu alto = priorita piu bassa
        return True
    except Exception:
        return False


def apply_cpu_limits(n_cores: int = None, set_low_priority: bool = True) -> int:
    """
    Applica TUTTI i limiti CPU possibili.
    
    Questa funzione deve essere chiamata ALL'INIZIO dello script,
    PRIMA di importare sklearn, xgboost, lightgbm.
    
    Applica:
    1. Affinita CPU (limita a N core fisici)
    2. Variabili d'ambiente per thread pools
    3. Priorita bassa (opzionale)
    
    Args:
        n_cores: Numero core da usare (None = totale - 2)
        set_low_priority: Se True, imposta priorita bassa
    
    Returns:
        Numero di core configurato
    """
    total_cores = os.cpu_count() or 4
    
    if n_cores is None:
        n_cores = max(1, total_cores - 2)
    else:
        n_cores = max(1, min(n_cores, total_cores))
    
    n_str = str(n_cores)
    
    # 1. Variabili d'ambiente (devono essere impostate PRIMA degli import)
    os.environ['OMP_NUM_THREADS'] = n_str
    os.environ['MKL_NUM_THREADS'] = n_str
    os.environ['OPENBLAS_NUM_THREADS'] = n_str
    os.environ['NUMEXPR_NUM_THREADS'] = n_str
    os.environ['LOKY_MAX_CPU_COUNT'] = n_str
    os.environ['VECLIB_MAXIMUM_THREADS'] = n_str
    os.environ['NUMBA_NUM_THREADS'] = n_str
    
    # 2. Affinita CPU (il metodo piu efficace)
    set_cpu_affinity(n_cores)
    
    # 3. Priorita bassa
    if set_low_priority:
        set_process_priority_low()
    
    return n_cores


class ResourceMonitor:
    """
    Monitor per utilizzo risorse con limiti configurabili.
    """
    
    def __init__(self, max_cpu: int = 85, max_ram: int = 85):
        """
        Args:
            max_cpu: Limite CPU % per warning
            max_ram: Limite RAM % per warning
        """
        self.max_cpu_percent = max_cpu
        self.max_ram_percent = max_ram
    
    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=0.1)
    
    def get_ram_usage(self) -> float:
        return psutil.virtual_memory().percent
    
    def get_available_ram_gb(self) -> float:
        return psutil.virtual_memory().available / (1024 ** 3)
    
    def check_resources(self) -> bool:
        cpu = self.get_cpu_usage()
        ram = self.get_ram_usage()
        return cpu < self.max_cpu_percent and ram < self.max_ram_percent
    
    def log_status(self, logger: logging.Logger) -> None:
        cpu = self.get_cpu_usage()
        ram = self.get_ram_usage()
        available = self.get_available_ram_gb()
        
        # Mostra anche affinita corrente
        try:
            p = psutil.Process()
            affinity = p.cpu_affinity()
            cores_used = len(affinity)
            total = psutil.cpu_count()
            logger.info(f"CPU: {cpu:.1f}% | RAM: {ram:.1f}% | "
                        f"Disponibile: {available:.1f}GB | "
                        f"Core attivi: {cores_used}/{total}")
        except Exception:
            logger.info(f"CPU: {cpu:.1f}% | RAM: {ram:.1f}% | "
                        f"Disponibile: {available:.1f}GB")


class ResourceLimiter(ResourceMonitor):
    """
    Limiter risorse con applicazione affinita CPU.
    """
    
    def __init__(self, n_cores: int = None, max_ram_percent: int = 85):
        super().__init__(max_cpu=85, max_ram=max_ram_percent)
        
        self.total_cores = os.cpu_count() or 4
        
        if n_cores is None:
            self.n_cores = max(1, self.total_cores - 2)
        else:
            self.n_cores = max(1, min(n_cores, self.total_cores))
        
        self.n_jobs = self.n_cores
        self._applied = False
    
    def apply(self) -> None:
        """Applica limiti CPU."""
        apply_cpu_limits(self.n_cores, set_low_priority=True)
        self._applied = True


def limit_cpu_cores(n_cores: Optional[int] = None) -> int:
    """Legacy function per retrocompatibilita."""
    total_cores = os.cpu_count() or 4
    if n_cores is None:
        n_cores = max(1, total_cores - 2)
    return min(n_cores, total_cores)


# ==============================================================================
# LOGGING
# ==============================================================================

def get_logger(name: str, 
               log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Configura e restituisce un logger.
    
    Crea un logger con output su console (sempre) e opzionalmente su file.
    Il formato include timestamp, livello e messaggio.
    
    Args:
        name: Nome del logger (tipicamente __name__ del modulo chiamante)
        log_file: Path opzionale per file di log
        level: Livello di logging (default INFO)
    
    Returns:
        Logger configurato
    
    Example:
        logger = get_logger(__name__, "logs/training.log")
        logger.info("Messaggio informativo")
        logger.error("Messaggio di errore")
    """
    logger = logging.getLogger(name)
    
    # Evita duplicazione handler se logger gia configurato
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Formato: timestamp | livello | messaggio
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler console - sempre attivo
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler file - opzionale
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File cattura tutto
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ==============================================================================
# PATH E UTILITY
# ==============================================================================

def get_project_root() -> Path:
    """
    Restituisce la directory root del progetto.
    
    Calcola il path risalendo dalla posizione di questo file.
    
    Returns:
        Path assoluto alla root del progetto
    """
    return Path(__file__).parent.parent


def get_timestamp() -> str:
    """
    Restituisce timestamp formattato per nomi file.
    
    Returns:
        Stringa nel formato YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    """
    Formatta dimensione in bytes in formato leggibile.
    
    Args:
        num: Numero di bytes
        suffix: Suffisso unita (default 'B' per bytes)
    
    Returns:
        Stringa formattata (es. '1.5 GB', '256.0 KB')
    
    Example:
        sizeof_fmt(1536)      # '1.5 KB'
        sizeof_fmt(1073741824) # '1.0 GB'
    """
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} P{suffix}"


def suppress_warnings() -> None:
    """Silenzia warning non critici per output piu pulito."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # Silenzia warning pandas su dtype
    import pandas as pd
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ==============================================================================
# VALIDAZIONE
# ==============================================================================

def validate_dataframe(df, required_columns: list = None, 
                       min_rows: int = 1) -> bool:
    """
    Valida un DataFrame verificando colonne e dimensioni.
    
    Args:
        df: DataFrame da validare
        required_columns: Lista colonne richieste (opzionale)
        min_rows: Numero minimo di righe richieste
    
    Returns:
        True se valido
    
    Raises:
        ValueError: Se DataFrame non valido
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input non e un DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame ha {len(df)} righe, minimo richiesto: {min_rows}")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Colonne mancanti: {missing}")
    
    return True


# ==============================================================================
# BENCHMARK LATENZA MODELLO
# ==============================================================================

def benchmark_model_latency(model, n_features: int, n_samples: int = 1000, 
                            n_iterations: int = 10) -> dict:
    """
    Benchmark latenza modello con procedura standardizzata.
    
    Questa funzione centralizza il benchmark di latenza per garantire
    risultati consistenti tra evaluation.py e compare_models.py.
    
    Args:
        model: Modello sklearn-like con metodo predict()
        n_features: Numero feature attese dal modello
        n_samples: Numero sample per ogni iterazione
        n_iterations: Numero iterazioni benchmark
    
    Returns:
        Dict con metriche latenza:
        - latency_total_ms: Tempo totale medio in ms
        - latency_per_sample_ms: Tempo per sample in ms
        - latency_std_ms: Deviazione standard
        - samples_per_second: Throughput
    """
    import numpy as np
    import time
    
    np.random.seed(42)
    X_dummy = np.random.randn(n_samples, n_features)
    
    # Warmup (3 iterazioni)
    for _ in range(3):
        _ = model.predict(X_dummy[:100])
    
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_dummy)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    # Rimuovi outlier (min e max) se abbiamo abbastanza iterazioni
    latencies_sorted = sorted(latencies)
    if len(latencies_sorted) > 4:
        latencies_trimmed = latencies_sorted[1:-1]
    else:
        latencies_trimmed = latencies_sorted
    
    total_latency = np.mean(latencies_trimmed)
    
    return {
        'latency_total_ms': float(total_latency),
        'latency_per_sample_ms': float(total_latency / n_samples),
        'latency_std_ms': float(np.std(latencies_trimmed)),
        'samples_per_second': float(n_samples / (total_latency / 1000))
    }

# ==============================================================================
# Funzioni per gestire selezione parametri da hyperparameter tuning.
# ==============================================================================

def list_tuning_configs(model_type: str) -> List[Dict]:
    """
    Lista tutte le configurazioni tuning disponibili per un modello.
    
    Args:
        model_type: 'random_forest', 'xgboost', 'lightgbm'
    
    Returns:
        Lista di dict con info su ogni config, ordinata per data (più recente prima)
    """
    tuning_dir = get_project_root() / "tuning_results" / model_type
    
    if not tuning_dir.exists():
        return []
    
    configs = []
    for json_file in tuning_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            configs.append({
                'filepath': json_file,
                'filename': json_file.name,
                'timestamp': data.get('tuning_timestamp'),
                'method': data.get('tuning_method'),
                'best_score': data.get('best_score'),
                'n_iterations': data.get('search_config', {}).get('n_iterations'),
                'cv_folds': data.get('search_config', {}).get('cv_folds'),
            })
        except Exception:
            continue
    
    # Ordina per timestamp (più recente prima)
    configs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return configs


def select_tuning_config(
    model_type: str,
    config_file: Optional[str] = None,
    timestamp: Optional[str] = None,
    task: str = 'binary'
) -> Optional[Dict]:
    """
    Seleziona una configurazione tuning con logica flessibile.
    
    Priorità:
    1. Se `config_file` specificato → usa quello
    2. Se `timestamp` specificato → cerca per timestamp
    3. Altrimenti → usa il più recente
    
    Args:
        model_type: Tipo modello
        config_file: Path o nome file specifico (es. "random_iter50_cv5_2026-01-24_20.02.json")
        timestamp: Timestamp parziale da cercare (es. "2026-01-24_20.02")
        task: Task (per validazione)
    
    Returns:
        Dict con best_params oppure None
    """
    configs = list_tuning_configs(model_type)
    
    if not configs:
        return None
    
    # Caso 1: File specifico
    if config_file:
        config_path = Path(config_file)
        
        # Se è path assoluto
        if config_path.is_absolute() and config_path.exists():
            target = config_path
        # Se è solo nome file
        else:
            tuning_dir = get_project_root() / "tuning_results" / model_type
            target = tuning_dir / config_path.name
        
        if not target.exists():
            print(f"Config file non trovato: {target}")
            return None
        
        with open(target) as f:
            data = json.load(f)
        
        if data.get('task') != task:
            print(f"Task mismatch: config per {data.get('task')}, richiesto {task}")
            return None
        
        print(f"✓ Config selezionata: {target.name}")
        return data['best_params']
    
    # Caso 2: Timestamp parziale
    if timestamp:
        for cfg in configs:
            if timestamp in cfg['filename']:
                with open(cfg['filepath']) as f:
                    data = json.load(f)
                
                if data.get('task') != task:
                    print(f"Task mismatch: config per {data.get('task')}, richiesto {task}")
                    continue
                
                print(f"✓ Config selezionata (timestamp): {cfg['filename']}")
                return data['best_params']
        
        print(f"Nessuna config trovata con timestamp '{timestamp}'")
        return None
    
    # Caso 3: Più recente (default)
    most_recent = configs[0]
    with open(most_recent['filepath']) as f:
        data = json.load(f)
    
    if data.get('task') != task:
        print(f"Config più recente è per task {data.get('task')}, richiesto {task}")
        # Cerca il più recente con task corretto
        for cfg in configs:
            with open(cfg['filepath']) as f:
                d = json.load(f)
            if d.get('task') == task:
                print(f"✓ Config selezionata (più recente, task={task}): {cfg['filename']}")
                return d['best_params']
        return None
    
    print(f"✓ Config selezionata (più recente): {most_recent['filename']}")
    return data['best_params']


def print_available_configs(model_type: str):
    """Stampa lista configurazioni disponibili."""
    configs = list_tuning_configs(model_type)
    
    if not configs:
        print(f"Nessuna configurazione tuning trovata per {model_type}")
        return
    
    print(f"\n{'='*70}")
    print(f"CONFIGURAZIONI TUNING DISPONIBILI - {model_type.upper()}")
    print(f"{'='*70}")
    print(f"\n{'#':<3} {'Filename':<45} {'Score':>8} {'Method':<10}")
    print("-"*70)
    
    for i, cfg in enumerate(configs, 1):
        print(f"{i:<3} {cfg['filename']:<45} {cfg['best_score']:>8.4f} {cfg['method']:<10}")
    
    print(f"\nTotale: {len(configs)} configurazioni")


# ==============================================================================
# FIX #2: Validazione Ordine Colonne e Checksum
# ==============================================================================

def compute_column_checksum(columns: List[str]) -> str:
    """
    Calcola checksum deterministico da lista colonne.
    
    Usa per verificare che scaler e dataset abbiano le stesse colonne
    nello stesso ordine.
    
    Args:
        columns: Lista nomi colonne (ordinata)
    
    Returns:
        Stringa checksum (hash SHA256 primi 16 caratteri)
    
    Examples:
        >>> compute_column_checksum(['a', 'b', 'c'])
        'a9e6d7f8b2c1e3d4'
    """
    import hashlib
    
    # Crea stringa deterministica
    columns_str = '|'.join(sorted(columns))
    
    # Calcola hash
    hash_obj = hashlib.sha256(columns_str.encode('utf-8'))
    checksum = hash_obj.hexdigest()[:16]
    
    return checksum


def validate_column_consistency(
    expected_columns: List[str],
    actual_columns: List[str],
    context: str = "dataset"
) -> None:
    """
    Valida che le colonne siano consistenti con quelle attese.
    
    Verifica:
    1. Stesse colonne presenti
    2. Stesso ordine
    3. Nessuna colonna mancante o extra
    
    Args:
        expected_columns: Colonne attese (da scaler/artifacts)
        actual_columns: Colonne effettive (da dataset)
        context: Descrizione contesto per errori più chiari
    
    Raises:
        ValueError: Se c'è inconsistenza
    
    Examples:
        >>> validate_column_consistency(['a', 'b'], ['b', 'a'], "test")
        ValueError: Ordine colonne diverso in test
    """
    expected_set = set(expected_columns)
    actual_set = set(actual_columns)
    
    # Controlla colonne mancanti
    missing = expected_set - actual_set
    if missing:
        raise ValueError(
            f"Colonne mancanti in {context}: {sorted(missing)}\n"
            f"Attese {len(expected_columns)} colonne, trovate {len(actual_columns)}"
        )
    
    # Controlla colonne extra
    extra = actual_set - expected_set
    if extra:
        raise ValueError(
            f"Colonne extra in {context}: {sorted(extra)}\n"
            f"Attese {len(expected_columns)} colonne, trovate {len(actual_columns)}"
        )
    
    # Controlla ordine
    if list(expected_columns) != list(actual_columns):
        raise ValueError(
            f"Ordine colonne diverso in {context}!\n"
            f"Atteso: {expected_columns[:5]}...\n"
            f"Trovato: {actual_columns[:5]}...\n"
            f"Le colonne devono essere nello stesso ordine per compatibilità con scaler."
        )
    
    # Tutto OK
    pass


def ensure_column_order(df, expected_columns: List[str]):
    """
    Riordina DataFrame secondo l'ordine atteso delle colonne.
    
    Usa quando hai un DataFrame con le colonne giuste ma in ordine sbagliato.
    
    Args:
        df: DataFrame pandas
        expected_columns: Lista colonne nell'ordine corretto
    
    Returns:
        DataFrame con colonne riordinate
    
    Raises:
        ValueError: Se mancano colonne
    """
    import pandas as pd
    
    # Verifica che tutte le colonne siano presenti
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel DataFrame: {sorted(missing)}")
    
    # Riordina
    return df[expected_columns]