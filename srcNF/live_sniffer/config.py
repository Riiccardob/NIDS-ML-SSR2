"""
Configurazione centralizzata per Live NIDS Sniffer.

Le feature attese dal modello vengono caricate dinamicamente da
artifacts/features.json per garantire coerenza con il training.
Non modificare REQUIRED_FEATURES manualmente: il file JSON e' la
single source of truth.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

SNIFFER_DIR = Path(__file__).parent
PROJECT_ROOT = SNIFFER_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SNIFFER_DIR) not in sys.path:
    sys.path.insert(0, str(SNIFFER_DIR))


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

# ============================================================================
# FEATURE CONFIGURATION - CARICATA DINAMICAMENTE DA features.json
# ============================================================================

def _load_features_from_artifacts() -> tuple[List[str], int]:
    """
    Carica la lista feature e il conteggio direttamente da features.json.

    In questo modo config.py non contiene mai una lista hardcoded che
    potrebbe divergere da cio' che il modello si aspetta realmente.

    Returns:
        (required_features, n_features)

    Raises:
        FileNotFoundError: se features.json non esiste.
        KeyError: se le chiavi attese non sono presenti nel JSON.
        ValueError: se la lista feature e' vuota.
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"features.json non trovato: {FEATURES_PATH}\n"
            "Esegui prima srcNF/feature_engineering.py per generare gli artifacts."
        )

    with open(FEATURES_PATH, "r") as fh:
        data = json.load(fh)

    features: List[str] = data.get("features", [])
    n_features: int = data.get("n_features", len(features))

    if not features:
        raise ValueError(
            f"features.json contiene una lista feature vuota: {FEATURES_PATH}"
        )

    if len(features) != n_features:
        raise ValueError(
            f"Inconsistenza in features.json: "
            f"len(features)={len(features)} ma n_features={n_features}. "
            "Rigenera gli artifacts con feature_engineering.py."
        )

    return features, n_features


REQUIRED_FEATURES: List[str]
N_FEATURES: int

REQUIRED_FEATURES, N_FEATURES = _load_features_from_artifacts()

# ============================================================================
# NETWORK CAPTURE
# ============================================================================

NETWORK_INTERFACE: str | None = None
SNAPLEN: int = 65535
FLOW_IDLE_TIMEOUT: int = 60
FLOW_ACTIVE_TIMEOUT: int = 300
FLOW_EXPIRATION_CHECK_INTERVAL: int = 10

# ============================================================================
# MODEL
# ============================================================================

MODEL_TYPE: str = "xgboost"
MODEL_PATH: Path = MODELS_DIR / MODEL_TYPE / "model.pkl"
ATTACK_THRESHOLD: float = 0.5
INFERENCE_BATCH_SIZE: int = 100
INFERENCE_BATCH_TIMEOUT: float = 5.0

# ============================================================================
# OPERATION
# ============================================================================

OPERATION_MODE: OperationMode = OperationMode.ALERT
BLOCK_DURATION_SECONDS: int = 3600
WHITELIST_IPS: List[str] = ["127.0.0.1", "::1"]

# ============================================================================
# LOGGING
# ============================================================================

LOG_FORMAT_TYPE: LogFormat = LogFormat.BOTH
LOG_MAX_SIZE_MB: int = 100
LOG_BACKUP_COUNT: int = 10
LOG_LEVEL: str = "INFO"
LOG_PREFIX: str = "nids_sniffer"

# ============================================================================
# FIREWALL
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


def validate_config() -> None:
    """
    Valida la configurazione e verifica la coerenza tra artifacts.

    Controlla:
    - Esistenza scaler, features.json, model
    - Coerenza tra REQUIRED_FEATURES (da features.json) e scaler
    - Range dei parametri numerici

    Raises:
        ValueError: se sono presenti errori di configurazione.
        FileNotFoundError: se gli artifacts necessari mancano.
    """
    errors: List[str] = []

    if not SCALER_PATH.exists():
        errors.append(f"Scaler non trovato: {SCALER_PATH}")

    if not FEATURES_PATH.exists():
        errors.append(f"features.json non trovato: {FEATURES_PATH}")

    if not MODEL_PATH.exists():
        errors.append(f"Model non trovato: {MODEL_PATH}")

    if not 0.0 <= ATTACK_THRESHOLD <= 1.0:
        errors.append(f"ATTACK_THRESHOLD fuori range [0,1]: {ATTACK_THRESHOLD}")

    if INFERENCE_BATCH_SIZE < 1:
        errors.append(f"INFERENCE_BATCH_SIZE deve essere >= 1: {INFERENCE_BATCH_SIZE}")

    # Verifica coerenza scaler <-> features.json
    if SCALER_PATH.exists() and FEATURES_PATH.exists():
        try:
            import joblib
            scaler = joblib.load(SCALER_PATH)
            if hasattr(scaler, "n_features_in_"):
                if scaler.n_features_in_ != N_FEATURES:
                    errors.append(
                        f"Mismatch scaler/features.json: "
                        f"scaler.n_features_in_={scaler.n_features_in_} "
                        f"ma features.json.n_features={N_FEATURES}. "
                        "Rigenera gli artifacts con la stessa esecuzione di pipeline."
                    )
        except Exception as exc:
            errors.append(f"Impossibile verificare coerenza scaler: {exc}")

    if errors:
        raise ValueError(
            "Errori di configurazione:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def get_config_summary() -> Dict[str, Any]:
    """Restituisce un riepilogo della configurazione attiva."""
    return {
        "operation_mode": OPERATION_MODE.value,
        "model_type": MODEL_TYPE,
        "network_interface": NETWORK_INTERFACE or "auto-detect",
        "attack_threshold": ATTACK_THRESHOLD,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "log_format": LOG_FORMAT_TYPE.value,
        "firewall_type": FIREWALL_TYPE if OPERATION_MODE == OperationMode.BLOCK else "N/A",
        "n_features": N_FEATURES,
        "features_source": str(FEATURES_PATH),
    }
