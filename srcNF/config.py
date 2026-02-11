"""
Configuration centrale per NIDS NetFlow-based.

CHANGELOG v2:
    Aggiunte a FEATURES_TO_DROP le feature non riproducibili fedelmente
    da nfstream in produzione:

    - MIN_TTL: nfstream non espone il TTL per-flow. Passare 0.0 introduce
      un bias costante che altera le predizioni su ogni singolo flow.

    - NUM_PKTS_UP_TO_128_BYTES ... NUM_PKTS_1024_TO_1514_BYTES (5 feature):
      nfstream non tiene contatori per bucket di dimensione pacchetto.
      La stima gaussiana usata in precedenza non preserva la "firma" degli
      attacchi (es. DDoS con tutti pacchetti identici).

    Rimuovere queste feature dal training garantisce che il modello non
    sviluppi dipendenze da valori che il sniffer non puo' fornire.
    Il modello risultante avra' 18 feature invece di 24.
"""

from pathlib import Path
from typing import List

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                 MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_NAME = "NF-UQ-NIDS-v2"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ============================================================================
# CHUNK-BASED PROCESSING CONFIGURATION
# ============================================================================

CHUNK_SIZE = 500_000
SCALER_SAMPLE_SIZE = 2_000_000

# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

ENABLE_PARALLEL_PROCESSING = True
CPU_USAGE_PERCENT = 0.75
MIN_WORKERS = 2
MAX_WORKERS = 16

# ============================================================================
# FEATURE CONFIGURATION - NFSTREAM COMPATIBLE
# ============================================================================

LABEL_COLUMN = 'Label'

FEATURES_TO_DROP: List[str] = [
    # IP addresses (non numeriche)
    'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR',

    # Feature NON disponibili in nfstream (TCP analysis avanzato)
    'TCP_FLAGS',
    'SERVER_TCP_FLAGS',
    'RETRANSMITTED_IN_BYTES',
    'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES',

    # Feature NON disponibili (TCP window tracking)
    'TCP_WIN_MAX_IN',
    'TCP_WIN_MAX_OUT',

    # Feature NON disponibili (DPI profondo protocolli applicativi)
    'DNS_QUERY_ID',
    'DNS_QUERY_TYPE',
    'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE',
    'ICMP_TYPE',

    # Feature CORROTTE nel training set (bug overflow)
    'SRC_TO_DST_SECOND_BYTES',
    'DST_TO_SRC_SECOND_BYTES',

    # -------------------------------------------------------------------------
    # AGGIUNTO v2: Feature non riproducibili fedelmente da nfstream
    # -------------------------------------------------------------------------

    # nfstream non espone il TTL per-flow.
    # Passare sempre 0.0 introduce un bias costante su tutte le predizioni:
    # il RobustScaler produce (0 - median_TTL) / IQR_TTL per ogni flow,
    # un valore negativo fisso che il modello non ha mai visto in training.
    'MIN_TTL',

    # nfstream non tiene contatori per bucket di dimensione pacchetto.
    # La stima con distribuzione gaussiana sull'avg_packet_size non preserva
    # la firma degli attacchi che usano pacchetti uniformi (es. SYN flood
    # con pacchetti identici da 40 byte concentra tutto nel bucket UP_TO_128,
    # mentre la gaussiana lo sparge su tutti i bucket).
    'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
    
    # -------------------------------------------------------------------------
    # AGGIUNTO v3: Feature non disponibili in nfstream
    # -------------------------------------------------------------------------
    
    # nfstream (versione corrente) non espone contatori di ritrasmissione.
    # Gli attributi dst2src_retrans_packets, src2dst_retrans_packets,
    # bidirectional_retrans_packets non esistono.
    # Passare sempre 0 introduce un bias (il modello ha imparato che
    # ritrasmissioni > 0 sono indicative di problemi di rete/attacchi).
    'RETRANSMITTED_OUT_PKTS',
]

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# ============================================================================

CORRELATION_THRESHOLD = 0.95

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

SUPPORTED_MODELS = ['xgboost', 'random_forest', 'lightgbm']
DEFAULT_MODEL = 'xgboost'

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_bin': 256,
    'random_state': RANDOM_STATE,
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'max_bin': 255,
    'random_state': RANDOM_STATE,
    'verbose': -1,
}

# ============================================================================
# SCALING CONFIGURATION
# ============================================================================

SCALER_TYPE = 'robust'

# ============================================================================
# PARQUET CONFIGURATION
# ============================================================================

PARQUET_COMPRESSION = 'snappy'
PARQUET_ENGINE = 'pyarrow'

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

MAX_RAM_USAGE_PERCENT = 50
RAM_WARNING_THRESHOLD = 70
RAM_ERROR_THRESHOLD = 85

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'