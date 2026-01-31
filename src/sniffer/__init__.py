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

NOTA: Esistono due SnifferEvaluator:
- src.sniffer.engine.SnifferEvaluator: versione base (restituisce dict)
- src.sniffer.evaluation.SnifferEvaluator: versione estesa (restituisce EvaluationResult)

Per compatibilita, questo package esporta la versione da engine.py che restituisce dict.
Se serve EvaluationResult, importare esplicitamente da evaluation.py.
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
    SnifferEvaluator,  # Versione che restituisce dict
    PredictionResult, 
    SessionStats, 
    PacketProcessor
)

# Per chi vuole EvaluationResult
try:
    from .evaluation import (
        SnifferEvaluator as SnifferEvaluatorExtended,
        EvaluationResult,
        quick_evaluate
    )
except ImportError:
    SnifferEvaluatorExtended = None
    EvaluationResult = None
    quick_evaluate = None

__all__ = [
    # Preprocessing
    'load_pipeline_artifacts', 
    'InferencePipeline', 
    'PipelineArtifacts', 
    'create_inference_pipeline', 
    'validate_artifacts_consistency',
    # Flow
    'Flow', 
    'FlowManager', 
    'PacketInfo', 
    'DEFAULT_FLOW_TIMEOUT',
    # Features
    'FeatureExtractor', 
    'FEATURE_NAMES',
    # Engine (principale)
    'SnifferEngine', 
    'SnifferEvaluator',  # Default: restituisce dict
    'PredictionResult', 
    'SessionStats', 
    'PacketProcessor',
    # Evaluation (opzionale)
    'SnifferEvaluatorExtended',
    'EvaluationResult',
    'quick_evaluate',
]

__version__ = '3.1.0'