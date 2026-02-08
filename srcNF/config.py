"""
Configuration centrale per NIDS NetFlow-based.

ULTRA RAM-SAFE per sistemi 16GB:
- Sample size limitati a 2M (HARD LIMIT)
- Monitoring RAM aggressivo
- GC frequente
- Strategia conservativa come preprocessing.py
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

# Crea directory se non esistono
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_NAME = "NF-UQ-NIDS-v2"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random state per riproducibilità
RANDOM_STATE = 42

# ============================================================================
# CHUNK-BASED PROCESSING CONFIGURATION
# ============================================================================

# Dimensione chunk per lettura CSV (righe per chunk)
CHUNK_SIZE = 500_000

# ============================================================================
# SCALER SAMPLE SIZE - ULTRA RAM-SAFE
# ============================================================================

# CRITICAL: Con 16GB RAM totale, usiamo sample PICCOLO per sicurezza
# 
# 2M righe × 40 features × 8 bytes = 640 MB base
# + DataFrame overhead (30%) = ~830 MB
# + Processing overhead = ~1.2 GB peak
# 
# Questo è SAFE anche se sistema usa già 8-10GB per OS/apps
# 
# NOTA: Il codice impone HARD LIMIT a 2M indipendentemente da questo valore
#       per evitare crash anche se utente lo aumenta
SCALER_SAMPLE_SIZE = 2_000_000  # 2M - ULTRA SAFE per 16GB RAM

# ALTERNATIVE (solo se hai più RAM disponibile):
# - 1_000_000   = 1M  - Ultra conservativo (~600 MB)
# - 2_000_000   = 2M  - Safe (default) (~1.2 GB)
# - 5_000_000   = 5M  - Aggressivo, richiede >8GB RAM libera
# - 10_000_000  = 10M - Molto aggressivo, richiede >12GB RAM libera

# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

ENABLE_PARALLEL_PROCESSING = True
CPU_USAGE_PERCENT = 0.75
MIN_WORKERS = 2
MAX_WORKERS = 16

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURES_TO_DROP: List[str] = [
    'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR',
]

LABEL_COLUMN = 'Label'

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# ============================================================================

CORRELATION_THRESHOLD = 0.95

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

SUPPORTED_MODELS = ['xgboost', 'random_forest', 'lightgbm']
DEFAULT_MODEL = 'xgboost'

# XGBoost params - ottimizzati per RAM limitata
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',        # Più efficiente per RAM
    'max_depth': 6,               # Limitato per RAM
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_bin': 256,               # Ridotto per RAM
    'random_state': RANDOM_STATE,
}

# Random Forest params - ridotti per RAM
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,              # Limitato per RAM
    'min_samples_split': 5,
    'max_features': 'sqrt',       # Riduce RAM
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# LightGBM params - già ottimizzato per RAM
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

# RobustScaler è più robusto agli outlier
# Importante per NIDS dove ci sono picchi di traffico
SCALER_TYPE = 'robust'  # 'robust' o 'standard'

# ============================================================================
# PARQUET CONFIGURATION
# ============================================================================

PARQUET_COMPRESSION = 'snappy'
PARQUET_ENGINE = 'pyarrow'

# ============================================================================
# MEMORY MANAGEMENT - ULTRA SAFE
# ============================================================================

# Percentuale massima RAM da utilizzare (molto conservativo)
MAX_RAM_USAGE_PERCENT = 50  # Max 50% RAM disponibile

# Threshold per warning RAM
RAM_WARNING_THRESHOLD = 70  # Warning se >70%

# Threshold per errore RAM
RAM_ERROR_THRESHOLD = 85    # Error se >85%

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'