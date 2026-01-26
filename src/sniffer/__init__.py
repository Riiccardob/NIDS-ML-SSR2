"""
================================================================================
NIDS-ML - Sniffer Package
================================================================================

Network Intrusion Detection System con Machine Learning.

Modules:
---------
config      : Centralized configuration management
flow        : Flow class for packet aggregation and FlowManager
engine      : SnifferEngine for live/PCAP capture and ML detection
features    : CIC-IDS2017 compatible feature extraction (77 features)
validator   : Feature validation against ground truth datasets
evaluation  : Model evaluation on CSV/PCAP with metrics calculation
calibration : Feature calibration and diagnostic tools
main        : CLI entry point

Quick Start (CLI):
------------------
    # Show configuration and available models
    python -m src.sniffer.main config --show --list-models
    
    # Validate features against CIC-IDS2017 Tuesday
    python -m src.sniffer.main validate --day tuesday
    
    # Evaluate XGBoost model on all days
    python -m src.sniffer.main evaluate --all-days --model-type xgboost
    
    # Analyze PCAP with LightGBM
    python -m src.sniffer.main pcap --day wednesday --model-type lightgbm
    
    # Live capture
    sudo python -m src.sniffer.main live --interface eth0 --duration 300

Quick Start (Python API):
-------------------------
    from src.sniffer import Config, SnifferEngine, SnifferEvaluator
    
    # Configure
    config = Config()
    config.model.model_type = 'xgboost'
    config.model.model_version = 'best'
    
    # Analyze PCAP
    engine = SnifferEngine(
        model_dir=str(config.model.get_model_dir(config.paths)),
        artifacts_dir=str(config.paths.artifacts_dir)
    )
    results = engine.analyze_pcap('capture.pcap')
    
    # Evaluate on CSV
    evaluator = SnifferEvaluator(model_dir='models/best_model')
    metrics = evaluator.evaluate_csv('data/raw/Tuesday-WorkingHours.pcap_ISCX.csv')
    metrics.print_summary()

================================================================================
"""

# Configuration (import first as other modules may depend on it)
from .config import (
    Config,
    PathConfig,
    ModelConfig,
    SnifferConfig,
    ValidationConfig,
    DatasetConfig,
    get_config,
    set_config,
    load_config,
    list_available_models,
    get_cicids_csv_path,
    get_cicids_pcap_path,
    print_config_summary
)

# Core components
from .flow import Flow, FlowManager

# Feature extraction
from .features import (
    FeatureExtractor,
    FEATURE_NAMES,
    CRITICAL_FEATURES,
    get_feature_columns_ordered,
    validate_feature_dict
)

# Main engine
from .engine import (
    SnifferEngine,
    PacketProcessor,
    PacketInfo,
    PredictionResult,
    SessionStats,
    SnifferLogger,
    FirewallManager,
    quick_analyze_pcap
)

# Validation
from .validator import (
    FeatureValidator,
    ValidationReport,
    ValidationMode,
    quick_validate_csv
)

# Evaluation
from .evaluation import (
    SnifferEvaluator,
    EvaluationMetrics,
    LatencyBenchmarker,
    quick_evaluate,
    run_benchmark
)

# Calibration
from .calibration import (
    FeatureCalibrator,
    CalibrationReport,
    FeatureStats,
    quick_feature_check,
    generate_calibration_report
)

__all__ = [
    # Config
    'Config',
    'PathConfig',
    'ModelConfig',
    'SnifferConfig',
    'ValidationConfig',
    'DatasetConfig',
    'get_config',
    'set_config',
    'load_config',
    'list_available_models',
    'get_cicids_csv_path',
    'get_cicids_pcap_path',
    'print_config_summary',
    
    # Flow
    'Flow',
    'FlowManager',
    
    # Features
    'FeatureExtractor',
    'FEATURE_NAMES',
    'CRITICAL_FEATURES',
    'get_feature_columns_ordered',
    'validate_feature_dict',
    
    # Engine
    'SnifferEngine',
    'PacketProcessor',
    'PacketInfo',
    'PredictionResult',
    'SessionStats',
    'SnifferLogger',
    'FirewallManager',
    'quick_analyze_pcap',
    
    # Validation
    'FeatureValidator',
    'ValidationReport',
    'ValidationMode',
    'quick_validate_csv',
    
    # Evaluation
    'SnifferEvaluator',
    'EvaluationMetrics',
    'LatencyBenchmarker',
    'quick_evaluate',
    'run_benchmark',
    
    # Calibration
    'FeatureCalibrator',
    'CalibrationReport',
    'FeatureStats',
    'quick_feature_check',
    'generate_calibration_report',
]

__version__ = '2.0.0'
