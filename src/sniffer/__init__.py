"""
NIDS-ML Sniffer Package (Corrected)

Uso:
    from src.sniffer import SnifferEngine, SnifferEvaluator
    
    # Valutazione CSV (TUTTO il dataset per default)
    evaluator = SnifferEvaluator('models/best_model')
    result = evaluator.evaluate_csv('data/test.csv')
    result.print_summary()
    
    # Analisi PCAP (TUTTI i pacchetti per default)
    engine = SnifferEngine('models/best_model')
    attacks = engine.analyze_pcap('capture.pcap')
    
    # Live capture
    engine.start_live('eth0', duration=60)

CORREZIONI:
- Default sample_size=None (processa tutto)
- EvaluationResult con attributi accessibili
- Gestione robusta errori e versioni sklearn
"""

from .preprocessing import (
    load_pipeline_artifacts,
    InferencePipeline,
    PipelineArtifacts,
    create_inference_pipeline,
    validate_artifacts_consistency
)
from .flow import Flow, FlowManager, PacketInfo, DEFAULT_FLOW_TIMEOUT
from .features import FeatureExtractor, FEATURE_NAMES
from .engine import (
    SnifferEngine,
    PredictionResult,
    SessionStats,
    PacketProcessor
)
from .evaluation import (
    SnifferEvaluator,
    EvaluationResult,
    LatencyBenchmarker,
    quick_evaluate
)

__all__ = [
    'load_pipeline_artifacts',
    'InferencePipeline',
    'PipelineArtifacts',
    'create_inference_pipeline',
    'validate_artifacts_consistency',
    'Flow',
    'FlowManager',
    'PacketInfo',
    'DEFAULT_FLOW_TIMEOUT',
    'FeatureExtractor',
    'FEATURE_NAMES',
    'SnifferEngine',
    'SnifferEvaluator',
    'EvaluationResult',
    'PredictionResult',
    'SessionStats',
    'PacketProcessor',
    'LatencyBenchmarker',
    'quick_evaluate',
]

__version__ = '4.0.0'