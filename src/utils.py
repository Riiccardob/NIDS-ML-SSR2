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

class ResourceLimiter:
    """
    Limita effettivamente l'uso di CPU impostando affinita e variabili d'ambiente.
    
    Questo limiter funziona PRIMA che i job vengano lanciati, configurando:
    1. Variabili d'ambiente per thread pool (OMP, MKL, OpenBLAS, etc.)
    2. Numero di job per scikit-learn e modelli
    
    IMPORTANTE: Deve essere chiamato PRIMA di importare sklearn/xgboost/lightgbm
    o all'inizio dello script.
    
    Example:
        limiter = ResourceLimiter(n_cores=4)
        limiter.apply()  # Applica limiti a livello di sistema
        n_jobs = limiter.n_jobs  # Usa questo per i modelli
    """
    
    def __init__(self, n_cores: int = None, max_ram_percent: int = 85):
        """
        Inizializza il limiter.
        
        Args:
            n_cores: Numero core da usare. None = totale - 2 (min 1)
            max_ram_percent: Limite RAM per warning (non blocca, solo avvisa)
        """
        self.total_cores = os.cpu_count() or 4
        
        if n_cores is None:
            # Default: lascia 2 core liberi per sistema e UI
            self.n_cores = max(1, self.total_cores - 2)
        else:
            self.n_cores = max(1, min(n_cores, self.total_cores))
        
        self.n_jobs = self.n_cores
        self.max_ram_percent = max_ram_percent
        self._applied = False
    
    def apply(self) -> None:
        """
        Applica i limiti CPU a livello di sistema.
        
        Imposta variabili d'ambiente che controllano il parallelismo di:
        - OpenMP (usato da sklearn, xgboost, lightgbm)
        - MKL (Intel Math Kernel Library)
        - OpenBLAS
        - NumExpr
        - Joblib
        """
        n = str(self.n_cores)
        
        # OpenMP - usato da XGBoost, LightGBM, sklearn
        os.environ['OMP_NUM_THREADS'] = n
        
        # Intel MKL
        os.environ['MKL_NUM_THREADS'] = n
        
        # OpenBLAS
        os.environ['OPENBLAS_NUM_THREADS'] = n
        
        # NumExpr
        os.environ['NUMEXPR_NUM_THREADS'] = n
        
        # Joblib (sklearn parallel backend)
        os.environ['LOKY_MAX_CPU_COUNT'] = n
        
        # Anche questo per sicurezza
        os.environ['VECLIB_MAXIMUM_THREADS'] = n
        
        self._applied = True
    
    def get_status(self) -> dict:
        """Restituisce stato corrente delle risorse."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.5),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'cores_configured': self.n_cores,
            'cores_total': self.total_cores,
            'limits_applied': self._applied
        }
    
    def log_status(self, logger: logging.Logger) -> None:
        """Logga stato corrente."""
        status = self.get_status()
        logger.info(f"CPU: {status['cpu_percent']:.1f}% | "
                    f"RAM: {status['ram_percent']:.1f}% | "
                    f"Disponibile: {status['ram_available_gb']:.1f}GB | "
                    f"Core: {status['cores_configured']}/{status['cores_total']}")
    
    def check_ram(self) -> bool:
        """Verifica se RAM e sotto il limite."""
        return psutil.virtual_memory().percent < self.max_ram_percent
    
    def warn_if_high_ram(self, logger: logging.Logger) -> None:
        """Logga warning se RAM alta."""
        ram = psutil.virtual_memory().percent
        if ram > self.max_ram_percent:
            logger.warning(f"RAM al {ram:.1f}% (limite: {self.max_ram_percent}%)")


# Alias per retrocompatibilita
class ResourceMonitor(ResourceLimiter):
    """Alias per ResourceLimiter (retrocompatibilita)."""
    pass


def setup_resource_limits(n_cores: int = None, max_ram: int = 85) -> ResourceLimiter:
    """
    Funzione helper per setup rapido dei limiti risorse.
    
    CHIAMARE ALL'INIZIO DELLO SCRIPT, prima di altri import.
    
    Args:
        n_cores: Core da usare (None = auto)
        max_ram: Limite RAM percentuale
    
    Returns:
        ResourceLimiter configurato e applicato
    
    Example:
        # All'inizio dello script
        from src.utils import setup_resource_limits
        limiter = setup_resource_limits(n_cores=4)
        
        # Poi usa limiter.n_jobs per i modelli
        model = RandomForestClassifier(n_jobs=limiter.n_jobs)
    """
    limiter = ResourceLimiter(n_cores=n_cores, max_ram_percent=max_ram)
    limiter.apply()
    return limiter


def limit_cpu_cores(n_cores: Optional[int] = None) -> int:
    """
    Limita il numero di core CPU utilizzabili (legacy function).
    
    NOTA: Questa funzione restituisce solo il numero, non applica limiti.
    Per limiti effettivi usare setup_resource_limits() o ResourceLimiter.
    
    Args:
        n_cores: Numero di core da usare. Se None, usa tutti meno 2.
    
    Returns:
        Numero di core configurato
    """
    total_cores = os.cpu_count() or 4
    
    if n_cores is None:
        n_cores = max(1, total_cores - 2)
    else:
        n_cores = max(1, min(n_cores, total_cores))
    
    return n_cores


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