"""
Configuration centrale per NIDS NetFlow-based.

Supporto per processing di dataset molto grandi (>70M records) tramite chunk-based approach.
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

# Nome dataset NF-UQ-NIDS-v2 (NetFlow V3)
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
# Con 16GB RAM, 500k righe sono sicure (~500MB-1GB per chunk a seconda delle feature)
CHUNK_SIZE = 500_000

# Dimensione sample per fitting dello scaler
# CRITICAL: Deve essere rappresentativo ma gestibile in RAM
# 1M righe = ~1-2GB RAM (dipende dalle feature)
SCALER_SAMPLE_SIZE = 1_000_000

# Numero massimo di chunk da processare per volta prima di salvare
# Utile per gestire memoria durante il processing
MAX_CHUNKS_IN_MEMORY = 3

# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

# Abilita processing parallelo
ENABLE_PARALLEL_PROCESSING = True

# Percentuale di CPU cores da utilizzare (0.0 - 1.0)
# 0.5 = usa 50% dei core disponibili
# 0.75 = usa 75% dei core disponibili
# 1.0 = usa tutti i core disponibili
CPU_USAGE_PERCENT = 0.75

# Numero minimo di worker (anche se CPU_USAGE_PERCENT porta a <2)
MIN_WORKERS = 2

# Numero massimo di worker (safety limit)
MAX_WORKERS = 16

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Feature da escludere (solo identificatori non predittivi)
FEATURES_TO_DROP: List[str] = [
    'IPV4_SRC_ADDR',        # IP sorgente (identificatore)
    'IPV4_DST_ADDR',        # IP destinazione (identificatore)
]

# Nome colonna label
LABEL_COLUMN = 'Label'

# NOTA: NON definiamo una lista fissa di feature.
# Tutte le feature numeriche del dataset (eccetto quelle in FEATURES_TO_DROP)
# verranno utilizzate automaticamente dopo i controlli di:
# - Varianza zero
# - Alta correlazione

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# ============================================================================

# Threshold per rimozione feature correlate
CORRELATION_THRESHOLD = 0.95

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Modelli supportati
SUPPORTED_MODELS = ['xgboost', 'random_forest', 'lightgbm']

# Default model
DEFAULT_MODEL = 'xgboost'

# XGBoost params base
# NOTA: max_bin ridotto per gestire meglio grandi dataset
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_bin': 256,  # Ridotto da default per RAM
    'random_state': RANDOM_STATE,
}

# Random Forest params base  
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'max_features': 'sqrt',  # Riduce memoria
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# LightGBM params base
# NOTA: LightGBM è già ottimizzato per grandi dataset
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'max_bin': 255,  # Default LightGBM
    'random_state': RANDOM_STATE,
    'verbose': -1,
}

# ============================================================================
# SCALING CONFIGURATION  
# ============================================================================

# CRITICAL: Usa RobustScaler fittato su dati "sporchi" (con outlier)
# per gestire picchi di traffico in produzione
# 
# Lo scaler viene fittato su un SAMPLE RAPPRESENTATIVO del train set
# PRIMA di qualsiasi rimozione di outlier
SCALER_TYPE = 'robust'  # 'robust' o 'standard'

# ============================================================================
# PARQUET CONFIGURATION
# ============================================================================

# Compressione Parquet (snappy è veloce e ben supportato)
PARQUET_COMPRESSION = 'snappy'

# Engine Parquet (pyarrow è il più veloce)
PARQUET_ENGINE = 'pyarrow'

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

# Percentuale massima RAM da utilizzare (safety buffer)
MAX_RAM_USAGE_PERCENT = 50  # Usa max 50% della RAM disponibile

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
