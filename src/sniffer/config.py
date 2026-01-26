"""
================================================================================
NIDS-ML - Configuration Module
================================================================================

Centralized configuration for the entire NIDS system.
All paths, parameters, and settings should be configured here.

Usage:
    from src.sniffer.config import Config, get_config
    
    # Get default config
    config = get_config()
    
    # Or create custom config
    config = Config(
        model_name='xgboost',
        model_version='bayesian_trials50_cv5_20240115'
    )

================================================================================
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


# ==============================================================================
# PATH DETECTION
# ==============================================================================

def find_project_root() -> Path:
    """
    Find project root by looking for marker files.
    Searches upward from current file location.
    """
    current = Path(__file__).resolve().parent
    
    # Markers that indicate project root
    markers = ['config.yaml', 'requirements.txt', 'setup.py', 'pyproject.toml', '.git']
    
    for _ in range(10):  # Max 10 levels up
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent
    
    # Fallback: assume we're in src/sniffer, go up 2 levels
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = find_project_root()


# ==============================================================================
# DEFAULT PATHS
# ==============================================================================

@dataclass
class PathConfig:
    """Path configuration with sensible defaults."""
    
    # Project root (auto-detected)
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    
    # Data directories
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'data')
    raw_csv_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'data' / 'raw')
    pcap_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'data' / 'pcap')
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'data' / 'processed')
    external_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'data' / 'external')
    
    # Model directories
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'models')
    best_model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'models' / 'best_model')
    
    # Artifacts
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'artifacts')
    
    # Logs and reports
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'logs')
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'reports')
    
    def __post_init__(self):
        """Convert strings to Path objects if needed."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))
    
    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for field_name in ['logs_dir', 'reports_dir', 'processed_dir', 'external_dir']:
            path = getattr(self, field_name)
            path.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

@dataclass
class ModelConfig:
    """Model selection and configuration."""
    
    # Model type: 'xgboost', 'lightgbm', 'random_forest'
    model_type: str = 'xgboost'
    
    # Model version (subfolder name in models/{model_type}/)
    # Use 'best' to auto-select based on metrics, or specify exact version
    model_version: str = 'best'
    
    # Task type: 'binary' or 'multiclass'
    task_type: str = 'binary'
    
    # Confidence threshold for attack classification
    confidence_threshold: float = 0.7
    
    # Use feature selector if available
    use_feature_selector: bool = True
    
    def get_model_dir(self, paths: PathConfig) -> Path:
        """Get the model directory based on configuration."""
        if self.model_version == 'best':
            return paths.best_model_dir
        return paths.models_dir / self.model_type / self.model_version
    
    def get_model_filename(self) -> str:
        """Get model filename based on task type."""
        return f'model_{self.task_type}.pkl'


# ==============================================================================
# SNIFFER CONFIGURATION
# ==============================================================================

@dataclass
class SnifferConfig:
    """Sniffer runtime configuration."""
    
    # Network interface for live capture
    interface: str = 'eth0'
    
    # BPF filter
    bpf_filter: str = 'ip'
    
    # Promiscuous mode
    promiscuous: bool = True
    
    # Flow timeout (seconds)
    flow_timeout: float = 60.0
    
    # Max packets per flow before forced analysis
    max_packets_per_flow: int = 500
    
    # Firewall settings
    firewall_enabled: bool = False
    firewall_dry_run: bool = True
    firewall_chain: str = 'INPUT'
    
    # Logging
    log_all_flows: bool = True
    log_attacks_only: bool = False


# ==============================================================================
# VALIDATION CONFIGURATION
# ==============================================================================

@dataclass
class ValidationConfig:
    """Validation and calibration settings."""
    
    # Tolerance thresholds by feature category
    temporal_tolerance: float = 0.30   # IAT, Duration
    count_tolerance: float = 0.05      # Packet counts, flags
    size_tolerance: float = 0.15       # Bytes, lengths
    rate_tolerance: float = 0.50       # Rates (depends on duration)
    default_tolerance: float = 0.20
    
    # Minimum pass rate for validation
    min_pass_rate: float = 0.80
    
    # Sample size for statistical validation
    default_sample_size: int = 10000


# ==============================================================================
# CIC-IDS2017 DATASET CONFIGURATION
# ==============================================================================

@dataclass 
class DatasetConfig:
    """Dataset-specific configuration."""
    
    # CIC-IDS2017 CSV files mapping
    CICIDS2017_FILES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'monday': {
            'csv': 'Monday-WorkingHours.pcap_ISCX.csv',
            'pcap': 'Monday-WorkingHours.pcap',
            'attacks': [],  # Benign only
            'description': 'Normal traffic baseline'
        },
        'tuesday': {
            'csv': 'Tuesday-WorkingHours.pcap_ISCX.csv',
            'pcap': 'Tuesday-WorkingHours.pcap',
            'attacks': ['FTP-Patator', 'SSH-Patator'],
            'description': 'Brute Force attacks'
        },
        'wednesday': {
            'csv': 'Wednesday-workingHours.pcap_ISCX.csv',
            'pcap': 'Wednesday-workingHours.pcap',
            'attacks': ['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed'],
            'description': 'DoS and Heartbleed attacks'
        },
        'thursday_morning': {
            'csv': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'pcap': 'Thursday-WorkingHours.pcap',
            'attacks': ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection'],
            'description': 'Web attacks'
        },
        'thursday_afternoon': {
            'csv': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'pcap': 'Thursday-WorkingHours.pcap',
            'attacks': ['Infiltration'],
            'description': 'Infiltration attack'
        },
        'friday_morning': {
            'csv': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'pcap': 'Friday-WorkingHours.pcap',
            'attacks': ['Bot'],
            'description': 'Botnet traffic'
        },
        'friday_afternoon_portscan': {
            'csv': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'pcap': 'Friday-WorkingHours.pcap',
            'attacks': ['PortScan'],
            'description': 'Port scanning'
        },
        'friday_afternoon_ddos': {
            'csv': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'pcap': 'Friday-WorkingHours.pcap',
            'attacks': ['DDoS'],
            'description': 'DDoS attack'
        }
    })
    
    # Label column name variations
    label_columns: List[str] = field(default_factory=lambda: [
        'Label', ' Label', 'label', 'LABEL', 'class', 'Class', 'CLASS'
    ])
    
    # Columns to drop (identifiers, not features)
    drop_columns: List[str] = field(default_factory=lambda: [
        'Flow ID', 'Source IP', 'Src IP', 'Source Port', 'Src Port',
        'Destination IP', 'Dst IP', 'Destination Port', 'Dst Port',
        'Timestamp', 'Protocol'
    ])


# ==============================================================================
# MAIN CONFIG CLASS
# ==============================================================================

@dataclass
class Config:
    """
    Main configuration class that combines all settings.
    
    Usage:
        config = Config()  # Default config
        
        config = Config(
            model=ModelConfig(model_type='lightgbm', model_version='v2'),
            sniffer=SnifferConfig(interface='wlan0')
        )
    """
    
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sniffer: SnifferConfig = field(default_factory=SnifferConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    def __post_init__(self):
        """Ensure paths exist."""
        self.paths.ensure_dirs()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            paths=PathConfig(**data.get('paths', {})),
            model=ModelConfig(**data.get('model', {})),
            sniffer=SnifferConfig(**data.get('sniffer', {})),
            validation=ValidationConfig(**data.get('validation', {}))
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            paths=PathConfig(**data.get('paths', {})),
            model=ModelConfig(**data.get('model', {})),
            sniffer=SnifferConfig(**data.get('sniffer', {})),
            validation=ValidationConfig(**data.get('validation', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()},
            'model': self.model.__dict__,
            'sniffer': self.sniffer.__dict__,
            'validation': self.validation.__dict__
        }
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ==============================================================================
# GLOBAL CONFIG INSTANCE
# ==============================================================================

_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def load_config(path: str) -> Config:
    """Load configuration from file and set as global."""
    if path.endswith('.yaml') or path.endswith('.yml'):
        config = Config.from_yaml(path)
    elif path.endswith('.json'):
        config = Config.from_json(path)
    else:
        raise ValueError(f"Unknown config format: {path}")
    
    set_config(config)
    return config


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def list_available_models(config: Optional[Config] = None) -> Dict[str, List[str]]:
    """
    List all available trained models.
    
    Returns:
        Dictionary mapping model type to list of versions
    """
    config = config or get_config()
    models_dir = config.paths.models_dir
    
    available = {}
    
    for model_type in ['xgboost', 'lightgbm', 'random_forest']:
        type_dir = models_dir / model_type
        if type_dir.exists():
            versions = [d.name for d in type_dir.iterdir() if d.is_dir()]
            if versions:
                available[model_type] = sorted(versions)
    
    # Check best_model
    best_dir = config.paths.best_model_dir
    if best_dir.exists():
        available['best_model'] = ['default']
    
    return available


def get_cicids_csv_path(day: str, config: Optional[Config] = None) -> Path:
    """Get path to a CIC-IDS2017 CSV file by day name."""
    config = config or get_config()
    
    if day not in config.dataset.CICIDS2017_FILES:
        available = list(config.dataset.CICIDS2017_FILES.keys())
        raise ValueError(f"Unknown day: {day}. Available: {available}")
    
    filename = config.dataset.CICIDS2017_FILES[day]['csv']
    return config.paths.raw_csv_dir / filename


def get_cicids_pcap_path(day: str, config: Optional[Config] = None) -> Path:
    """Get path to a CIC-IDS2017 PCAP file by day name."""
    config = config or get_config()
    
    if day not in config.dataset.CICIDS2017_FILES:
        available = list(config.dataset.CICIDS2017_FILES.keys())
        raise ValueError(f"Unknown day: {day}. Available: {available}")
    
    filename = config.dataset.CICIDS2017_FILES[day]['pcap']
    return config.paths.pcap_dir / filename


def print_config_summary(config: Optional[Config] = None):
    """Print a summary of the current configuration."""
    config = config or get_config()
    
    print("\n" + "=" * 60)
    print("NIDS-ML Configuration Summary")
    print("=" * 60)
    
    print(f"\n📁 Paths:")
    print(f"   Project root: {config.paths.project_root}")
    print(f"   Raw CSV:      {config.paths.raw_csv_dir}")
    print(f"   PCAP files:   {config.paths.pcap_dir}")
    print(f"   Models:       {config.paths.models_dir}")
    print(f"   Artifacts:    {config.paths.artifacts_dir}")
    
    print(f"\n🤖 Model:")
    print(f"   Type:         {config.model.model_type}")
    print(f"   Version:      {config.model.model_version}")
    print(f"   Task:         {config.model.task_type}")
    print(f"   Threshold:    {config.model.confidence_threshold}")
    
    print(f"\n📡 Sniffer:")
    print(f"   Interface:    {config.sniffer.interface}")
    print(f"   Filter:       {config.sniffer.bpf_filter}")
    print(f"   Flow timeout: {config.sniffer.flow_timeout}s")
    print(f"   Firewall:     {'Enabled' if config.sniffer.firewall_enabled else 'Disabled'}")
    
    # List available models
    available = list_available_models(config)
    if available:
        print(f"\n📦 Available Models:")
        for model_type, versions in available.items():
            print(f"   {model_type}: {', '.join(versions[:3])}{'...' if len(versions) > 3 else ''}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Test configuration
    config = Config()
    print_config_summary(config)
    
    # List available models
    print("\nAvailable models:")
    for model_type, versions in list_available_models().items():
        print(f"  {model_type}: {versions}")
