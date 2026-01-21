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
from typing import Optional
import warnings

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