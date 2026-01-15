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

class ResourceMonitor:
    """
    Monitor per utilizzo CPU e RAM con limiti configurabili.
    
    Permette di verificare se il sistema ha risorse disponibili prima
    di operazioni intensive, evitando crash o rallentamenti eccessivi.
    
    Attributes:
        max_cpu_percent: Limite massimo CPU (default 85%)
        max_ram_percent: Limite massimo RAM (default 85%)
    
    Example:
        monitor = ResourceMonitor(max_cpu=80, max_ram=80)
        if monitor.check_resources():
            # Procedi con operazione intensiva
            pass
    """
    
    def __init__(self, 
                 max_cpu: int = DEFAULT_MAX_CPU_PERCENT,
                 max_ram: int = DEFAULT_MAX_RAM_PERCENT):
        """
        Inizializza il monitor risorse.
        
        Args:
            max_cpu: Percentuale massima CPU consentita (1-100)
            max_ram: Percentuale massima RAM consentita (1-100)
        """
        self.max_cpu_percent = max_cpu
        self.max_ram_percent = max_ram
    
    def get_cpu_usage(self) -> float:
        """Restituisce utilizzo CPU corrente in percentuale."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_ram_usage(self) -> float:
        """Restituisce utilizzo RAM corrente in percentuale."""
        return psutil.virtual_memory().percent
    
    def get_available_ram_gb(self) -> float:
        """Restituisce RAM disponibile in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)
    
    def check_resources(self) -> bool:
        """
        Verifica se le risorse sono sotto i limiti configurati.
        
        Returns:
            True se CPU e RAM sono sotto i limiti, False altrimenti
        """
        cpu = self.get_cpu_usage()
        ram = self.get_ram_usage()
        return cpu < self.max_cpu_percent and ram < self.max_ram_percent
    
    def wait_for_resources(self, 
                           timeout_seconds: int = 300,
                           check_interval: int = 5) -> bool:
        """
        Attende che le risorse tornino sotto i limiti.
        
        Args:
            timeout_seconds: Timeout massimo in secondi
            check_interval: Intervallo tra controlli in secondi
        
        Returns:
            True se risorse disponibili entro timeout, False altrimenti
        """
        import time
        elapsed = 0
        while elapsed < timeout_seconds:
            if self.check_resources():
                return True
            time.sleep(check_interval)
            elapsed += check_interval
        return False
    
    def log_status(self, logger: logging.Logger) -> None:
        """Logga stato corrente delle risorse."""
        cpu = self.get_cpu_usage()
        ram = self.get_ram_usage()
        available = self.get_available_ram_gb()
        logger.info(f"Risorse: CPU={cpu:.1f}%, RAM={ram:.1f}%, "
                    f"Disponibile={available:.1f}GB")


def limit_cpu_cores(n_cores: Optional[int] = None) -> int:
    """
    Limita il numero di core CPU utilizzabili.
    
    Args:
        n_cores: Numero di core da usare. Se None, usa tutti meno 1.
    
    Returns:
        Numero di core configurato
    """
    total_cores = os.cpu_count() or 4
    
    if n_cores is None:
        # Lascia almeno 1 core libero per il sistema
        n_cores = max(1, total_cores - 1)
    else:
        n_cores = min(n_cores, total_cores)
    
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