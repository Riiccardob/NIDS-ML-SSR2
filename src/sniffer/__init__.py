"""
NIDS-ML Sniffer Package

Uso:
    from src.sniffer import SnifferEngine, SnifferEvaluator
    
    # Valutazione CSV
    evaluator = SnifferEvaluator('models/best_model')
    metrics = evaluator.evaluate_csv('data/test.csv')
    
    # Analisi PCAP
    engine = SnifferEngine('models/best_model')
    engine.analyze_pcap('capture.pcap')
    
    # Live capture
    engine.start_live('eth0', duration=60)
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
from .engine import SnifferEngine, SnifferEvaluator, PredictionResult, SessionStats, PacketProcessor

__all__ = [
    'load_pipeline_artifacts', 'InferencePipeline', 'PipelineArtifacts', 
    'create_inference_pipeline', 'validate_artifacts_consistency',
    'Flow', 'FlowManager', 'PacketInfo', 'DEFAULT_FLOW_TIMEOUT',
    'FeatureExtractor', 'FEATURE_NAMES',
    'SnifferEngine', 'SnifferEvaluator', 'PredictionResult', 'SessionStats', 'PacketProcessor',
]

__version__ = '3.0.0'