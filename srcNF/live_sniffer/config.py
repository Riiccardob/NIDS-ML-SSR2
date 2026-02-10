"""
Configurazione centralizzata per Live NIDS Sniffer.

SETUP A - CONSERVATIVE: Coerenza PERFETTA con training.
Feature: 21 (rimosse 3 in conflitto)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

SNIFFER_DIR = Path(__file__).parent
PROJECT_ROOT = SNIFFER_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class OperationMode(Enum):
    ALERT = "alert"
    BLOCK = "block"


class LogFormat(Enum):
    CSV = "csv"
    JSON = "json"
    BOTH = "both"


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "sniffer"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"

NETWORK_INTERFACE: str | None = None
SNAPLEN: int = 65535
FLOW_IDLE_TIMEOUT: int = 60
FLOW_ACTIVE_TIMEOUT: int = 300
FLOW_EXPIRATION_CHECK_INTERVAL: int = 10

# ============================================================================
# FEATURE EXTRACTION - 21 FEATURE (CONSERVATIVE)
# ============================================================================

REQUIRED_FEATURES: List[str] = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "CLIENT_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "ICMP_IPV4_TYPE",
]

N_FEATURES: int = 24

MODEL_TYPE: str = "xgboost"
MODEL_PATH: Path = MODELS_DIR / MODEL_TYPE / "model.pkl"
ATTACK_THRESHOLD: float = 0.5
INFERENCE_BATCH_SIZE: int = 100
INFERENCE_BATCH_TIMEOUT: float = 5.0

OPERATION_MODE: OperationMode = OperationMode.ALERT
BLOCK_DURATION_SECONDS: int = 3600
WHITELIST_IPS: List[str] = ["127.0.0.1", "::1"]

LOG_FORMAT_TYPE: LogFormat = LogFormat.BOTH
LOG_MAX_SIZE_MB: int = 100
LOG_BACKUP_COUNT: int = 10
LOG_LEVEL: str = "INFO"
LOG_PREFIX: str = "nids_sniffer"

FIREWALL_TYPE: str = "iptables"
IPTABLES_CHAIN: str = "NIDS_BLOCK"
IPTABLES_JUMP_RULE: bool = True

MAX_FLOWS_IN_MEMORY: int = 100000
STATS_LOG_INTERVAL: int = 60
INFERENCE_WORKERS: int = 2


def validate_config() -> None:
    errors: List[str] = []
    
    if not SCALER_PATH.exists():
        errors.append(f"Scaler not found: {SCALER_PATH}")
    
    if not FEATURES_PATH.exists():
        errors.append(f"Features not found: {FEATURES_PATH}")
    
    if not MODEL_PATH.exists():
        errors.append(f"Model not found: {MODEL_PATH}")
    
    if not 0.0 <= ATTACK_THRESHOLD <= 1.0:
        errors.append(f"Invalid ATTACK_THRESHOLD: {ATTACK_THRESHOLD}")
    
    if INFERENCE_BATCH_SIZE < 1:
        errors.append(f"Invalid INFERENCE_BATCH_SIZE: {INFERENCE_BATCH_SIZE}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


def get_config_summary() -> Dict[str, Any]:
    return {
        "operation_mode": OPERATION_MODE.value,
        "model_type": MODEL_TYPE,
        "network_interface": NETWORK_INTERFACE or "auto-detect",
        "attack_threshold": ATTACK_THRESHOLD,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "log_format": LOG_FORMAT_TYPE.value,
        "firewall_type": FIREWALL_TYPE if OPERATION_MODE == OperationMode.BLOCK else "N/A",
        "n_features": N_FEATURES,
        "features_dropped": 43 - N_FEATURES,
        "setup": "CONSERVATIVE (training-aligned)",
    }
