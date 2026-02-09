"""
Configurazione centralizzata per Live NIDS Sniffer.

AGGIORNATO: Feature list ridotta a quelle compatibili con nfstream.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

# Fix: Aggiungi parent directory al path per import relativi
SNIFFER_DIR = Path(__file__).parent
PROJECT_ROOT = SNIFFER_DIR.parent.parent  # srcNF/live_sniffer -> srcNF -> NIDS-ML-SSR2

# Aggiungi al path per permettere import dei moduli srcNF
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class OperationMode(Enum):
    """Modalita operative dello sniffer."""
    ALERT = "alert"
    BLOCK = "block"


class LogFormat(Enum):
    """Formato di output per i log."""
    CSV = "csv"
    JSON = "json"
    BOTH = "both"


# ============================================================================
# PATHS - Relativo alla root del progetto NIDS-ML-SSR2
# ============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "sniffer"

# Crea directory se non esistono
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File essenziali
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"


# ============================================================================
# NETWORK CAPTURE
# ============================================================================

NETWORK_INTERFACE: str | None = None
SNAPLEN: int = 65535
FLOW_IDLE_TIMEOUT: int = 60
FLOW_ACTIVE_TIMEOUT: int = 300
FLOW_EXPIRATION_CHECK_INTERVAL: int = 10


# ============================================================================
# FEATURE EXTRACTION - NFSTREAM COMPATIBLE
# ============================================================================

# Feature DISPONIBILI in nfstream (28 su 43 originali)
REQUIRED_FEATURES: List[str] = [
    # Base features (SEMPRE disponibili)
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    
    # Bytes & Packets (SEMPRE disponibili)
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    
    # Duration (SEMPRE disponibili)
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    
    # Packet size stats (disponibili)
    "MIN_TTL",                      # Proxy: min packet size
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    
    # Throughput (CALCOLATI)
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    
    # Packet distribution (STIMATI)
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    
    # Feature DROPPATE (non disponibili in nfstream):
    # - TCP_FLAGS, SERVER_TCP_FLAGS (solo SYN count)
    # - RETRANSMITTED_* (3 feature - non tracciabili)
    # - TCP_WIN_MAX_* (2 feature - non disponibili)
    # - DNS_*, FTP_*, ICMP_* (6 feature - no DPI)
    # - SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES (bug overflow)
]

# Numero feature dopo drop
N_FEATURES: int = len(REQUIRED_FEATURES)  # ~21 invece di 35


# ============================================================================
# PREDICTION
# ============================================================================

MODEL_TYPE: str = "xgboost"
MODEL_PATH: Path = MODELS_DIR / MODEL_TYPE / "model.pkl"
ATTACK_THRESHOLD: float = 0.5
INFERENCE_BATCH_SIZE: int = 100
INFERENCE_BATCH_TIMEOUT: float = 5.0


# ============================================================================
# OPERATION MODE
# ============================================================================

OPERATION_MODE: OperationMode = OperationMode.ALERT
BLOCK_DURATION_SECONDS: int = 3600
WHITELIST_IPS: List[str] = [
    "127.0.0.1",
    "::1",
]


# ============================================================================
# LOGGING
# ============================================================================

LOG_FORMAT_TYPE: LogFormat = LogFormat.BOTH
LOG_MAX_SIZE_MB: int = 100
LOG_BACKUP_COUNT: int = 10
LOG_LEVEL: str = "INFO"
LOG_PREFIX: str = "nids_sniffer"


# ============================================================================
# FIREWALL INTEGRATION
# ============================================================================

FIREWALL_TYPE: str = "iptables"
IPTABLES_CHAIN: str = "NIDS_BLOCK"
IPTABLES_JUMP_RULE: bool = True


# ============================================================================
# PERFORMANCE
# ============================================================================

MAX_FLOWS_IN_MEMORY: int = 100000
STATS_LOG_INTERVAL: int = 60
INFERENCE_WORKERS: int = 2


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> None:
    """Valida la configurazione all'avvio."""
    
    errors: List[str] = []
    
    if not SCALER_PATH.exists():
        errors.append(f"Scaler not found: {SCALER_PATH}")
    
    if not FEATURES_PATH.exists():
        errors.append(f"Features not found: {FEATURES_PATH}")
    
    if not MODEL_PATH.exists():
        errors.append(f"Model not found: {MODEL_PATH}")
    
    if not 0.0 <= ATTACK_THRESHOLD <= 1.0:
        errors.append(f"Invalid ATTACK_THRESHOLD: {ATTACK_THRESHOLD} (must be 0.0-1.0)")
    
    if INFERENCE_BATCH_SIZE < 1:
        errors.append(f"Invalid INFERENCE_BATCH_SIZE: {INFERENCE_BATCH_SIZE}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


def get_config_summary() -> Dict[str, Any]:
    """Restituisce summary della configurazione."""
    
    return {
        "operation_mode": OPERATION_MODE.value,
        "model_type": MODEL_TYPE,
        "network_interface": NETWORK_INTERFACE or "auto-detect",
        "attack_threshold": ATTACK_THRESHOLD,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "log_format": LOG_FORMAT_TYPE.value,
        "firewall_type": FIREWALL_TYPE if OPERATION_MODE == OperationMode.BLOCK else "N/A",
        "n_features": N_FEATURES,
        "features_dropped": 43 - N_FEATURES,  # Feature droppate
    }
