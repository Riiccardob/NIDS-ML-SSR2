"""
NIDS-ML Sniffer - Evaluation Module (Corrected)

Valuta il modello su CSV CIC-IDS2017.

Pipeline:
1. Carica CSV con ~77 feature
2. Filtra alle scaler_columns (44 feature dopo statistical preprocessing)
3. Scala con RobustScaler
4. Seleziona le 30 feature finali (indici in ordine di importanza)
5. Predici

CORREZIONI:
- evaluate_csv processa TUTTO il dataset per default (sample_size=None)
- Gestione robusta colonna Label con varianti multiple
- Metriche complete incluso per-class breakdown
- Compatibilita sklearn versions
- Output dettagliato per debugging
"""

import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_column(columns: List[str], target: str) -> Optional[str]:
    """Trova colonna con matching case-insensitive e varianti."""
    target_lower = target.strip().lower()
    target_normalized = target_lower.replace(' ', '_').replace('-', '_')
    
    for col in columns:
        col_stripped = col.strip()
        col_lower = col_stripped.lower()
        col_normalized = col_lower.replace(' ', '_').replace('-', '_')
        
        if col_lower == target_lower:
            return col_stripped
        if col_normalized == target_normalized:
            return col_stripped
        if col_lower.replace('_', ' ') == target_lower.replace('_', ' '):
            return col_stripped
    
    return None


def find_label_column(columns: List[str]) -> Optional[str]:
    """Trova la colonna Label con supporto per varianti CIC-IDS2017."""
    label_variants = [
        'Label', ' Label', 'label', 'LABEL',
        'class', 'Class', 'CLASS',
        'attack', 'Attack', 'ATTACK',
        'target', 'Target', 'TARGET'
    ]
    
    for variant in label_variants:
        for col in columns:
            if col.strip() == variant.strip():
                return col
    
    for col in columns:
        if 'label' in col.lower():
            return col
    
    return None


@dataclass
class EvaluationResult:
    """Risultato valutazione completo con attributi accessibili direttamente."""
    total_samples: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    specificity: float = 0.0
    
    class_distribution: Dict[str, int] = field(default_factory=dict)
    predictions_distribution: Dict[str, int] = field(default_factory=dict)
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    processing_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    csv_path: str = ""
    model_info: str = ""
    
    @property
    def f1(self) -> float:
        """Alias per compatibilita."""
        return self.f1_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'specificity': self.specificity,
            'class_distribution': self.class_distribution,
            'predictions_distribution': self.predictions_distribution,
            'per_class_metrics': self.per_class_metrics,
            'processing_time_seconds': self.processing_time_seconds,
            'samples_per_second': self.samples_per_second,
            'csv_path': self.csv_path,
            'model_info': self.model_info
        }
    
    def print_summary(self):
        """Stampa riepilogo formattato."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"CSV:        {self.csv_path}")
        print(f"Model:      {self.model_info}")
        print(f"Samples:    {self.total_samples:,}")
        print("-" * 60)
        print(f"Accuracy:   {self.accuracy:.4f}")
        print(f"Precision:  {self.precision:.4f}")
        print(f"Recall:     {self.recall:.4f}")
        print(f"F1 Score:   {self.f1_score:.4f}")
        print("-" * 60)
        print(f"TP: {self.true_positives:,}  TN: {self.true_negatives:,}")
        print(f"FP: {self.false_positives:,}  FN: {self.false_negatives:,}")
        print(f"FPR: {self.false_positive_rate:.4f}  FNR: {self.false_negative_rate:.4f}")
        print("-" * 60)
        print(f"Processing: {self.processing_time_seconds:.2f}s ({self.samples_per_second:.0f} samples/s)")
        
        if self.class_distribution:
            print("\nClass Distribution (Ground Truth):")
            for cls, count in sorted(self.class_distribution.items()):
                pct = count / self.total_samples * 100 if self.total_samples > 0 else 0
                print(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        if self.per_class_metrics:
            print("\nPer-Class Metrics:")
            for cls, metrics in sorted(self.per_class_metrics.items()):
                print(f"  {cls}:")
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall:    {metrics.get('recall', 0):.4f}")
                print(f"    F1:        {metrics.get('f1', 0):.4f}")
                print(f"    Support:   {metrics.get('support', 0):,}")
        
        print("=" * 60)


class SnifferEvaluator:
    """
    Valuta modello su CSV con pipeline corretta.
    
    Processa TUTTO il dataset per default (nessun sampling).
    """
    
    LABEL_VARIANTS = [
        'Label', ' Label', 'label', 'LABEL',
        'class', 'Class', 'CLASS'
    ]
    
    def __init__(
        self, 
        model_dir: str = 'models/best_model', 
        artifacts_dir: str = 'artifacts',
        label_column: str = 'Label'
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.label_column = label_column
        self.logger = logging.getLogger('sniffer.evaluator')
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Carica modello e artifacts."""
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato in {self.model_dir}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*version.*')
            self.model = joblib.load(model_path)
        self.logger.info(f"Modello caricato: {type(self.model).__name__}")
        
        scaler_path = self.artifacts_dir / 'scaler.pkl'
        if scaler_path.exists():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Scaler caricato: {type(self.scaler).__name__}")
        else:
            self.scaler = None
            self.logger.warning("Scaler non trovato")
        
        selector_path = self.artifacts_dir / 'feature_selector.pkl'
        if selector_path.exists():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.selector = joblib.load(selector_path)
            self.logger.info("Feature selector caricato")
        else:
            self.selector = None
        
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if scaler_cols_path.exists():
            with open(scaler_cols_path, 'r') as f:
                self.scaler_columns = json.load(f)
            self.logger.info(f"Scaler columns: {len(self.scaler_columns)} features")
        else:
            self.scaler_columns = None
        
        features_path = self.artifacts_dir / 'selected_features.json'
        if not features_path.exists():
            features_path = self.model_dir / 'features_binary.json'
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            self.logger.info(f"Selected features: {len(self.selected_features)} features")
        else:
            self.selected_features = None
        
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Configura pipeline di preprocessing."""
        self._selected_indices = None
        self.use_selector = False
        
        if self.selector is not None and self.scaler_columns is not None:
            self.features_to_load = self.scaler_columns
            self.use_selector = True
            self.logger.info(f"Pipeline: {len(self.scaler_columns)} -> scale -> selector -> predict")
            
        elif self.selector is None and self.scaler_columns is not None and self.selected_features is not None:
            scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(self.scaler_columns)}
            selected_indices = []
            missing_features = []
            
            for feat in self.selected_features:
                feat_lower = feat.strip().lower()
                if feat_lower in scaler_cols_lower:
                    selected_indices.append(scaler_cols_lower[feat_lower])
                else:
                    missing_features.append(feat)
            
            if missing_features:
                self.logger.warning(f"Feature non trovate in scaler_columns: {missing_features}")
            
            if len(selected_indices) == len(self.selected_features):
                self._selected_indices = selected_indices
                self.features_to_load = self.scaler_columns
                self.use_selector = True
                self.logger.info(
                    f"Pipeline: {len(self.scaler_columns)} -> scale -> "
                    f"select[{len(self._selected_indices)}] -> predict"
                )
            else:
                self.features_to_load = self.selected_features
                self.use_selector = False
                self.logger.warning("Fallback: uso selected_features direttamente")
                
        elif self.selected_features is not None:
            self.features_to_load = self.selected_features
            self.use_selector = False
            self.logger.info(f"Pipeline: {len(self.selected_features)} -> scale -> predict")
        else:
            raise ValueError("Nessun artifact feature trovato")
    
    def _find_label_column(self, columns: List[str]) -> str:
        """Trova colonna label nel DataFrame."""
        found = find_label_column(columns)
        if found:
            return found
        raise ValueError(
            f"Colonna label non trovata. Colonne disponibili: {columns[:20]}..."
        )
    
    def _prepare_csv_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
        """
        Prepara dati CSV per predizione.
        
        Returns:
            (features_df, labels_binary, class_distribution)
        """
        label_col = self._find_label_column(df.columns.tolist())
        
        original_labels = df[label_col].astype(str).str.strip()
        class_distribution = original_labels.value_counts().to_dict()
        
        labels = original_labels.str.upper().apply(
            lambda x: 0 if x == 'BENIGN' else 1
        ).values
        
        feature_cols = self.features_to_load
        col_mapping = {}
        
        for target_col in feature_cols:
            found = find_column(df.columns.tolist(), target_col)
            col_mapping[target_col] = found
        
        features_data = {}
        missing_count = 0
        missing_features = []
        
        for target_col, source_col in col_mapping.items():
            if source_col and source_col in df.columns:
                features_data[target_col] = df[source_col].values
            else:
                features_data[target_col] = np.zeros(len(df))
                missing_count += 1
                missing_features.append(target_col)
        
        if missing_count > 0:
            self.logger.warning(
                f"{missing_count} feature non trovate (impostate a 0): "
                f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
            )
        
        features_df = pd.DataFrame(features_data)
        features_df = features_df[feature_cols]
        
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df, labels, class_distribution
    
    def _predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Esegue predizione su batch."""
        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                features_scaled = self.scaler.transform(features_df.values)
        else:
            features_scaled = features_df.values
        
        if self.use_selector:
            if self.selector is not None:
                features_selected = self.selector.transform(features_scaled)
            elif self._selected_indices is not None:
                features_selected = features_scaled[:, self._selected_indices]
            else:
                features_selected = features_scaled
        else:
            features_selected = features_scaled
        
        return self.model.predict(features_selected)
    
    def evaluate_csv(
        self, 
        csv_path: str, 
        sample_size: Optional[int] = None,
        batch_size: int = 50000,
        verbose: bool = True,
        random_state: int = 42
    ) -> EvaluationResult:
        """
        Valuta modello su file CSV.
        
        Args:
            csv_path: Path al file CSV
            sample_size: Numero di righe da campionare (None = TUTTO il dataset)
            batch_size: Dimensione batch per processing
            verbose: Output verboso
            random_state: Seed per riproducibilita sampling
        
        Returns:
            EvaluationResult con metriche complete
        """
        start_time = time.time()
        csv_path = Path(csv_path)
        
        if verbose:
            print(f"Caricamento CSV: {csv_path}")
        
        df = pd.read_csv(csv_path, low_memory=False)
        original_size = len(df)
        
        if verbose:
            print(f"Righe totali: {original_size:,}")
        
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state)
            if verbose:
                print(f"Campionate: {len(df):,} righe")
        
        features_df, y_true, class_distribution = self._prepare_csv_data(df)
        
        if verbose:
            print(f"Features caricate: {len(self.features_to_load)}")
            print("Distribuzione classi:")
            for cls, count in sorted(class_distribution.items()):
                print(f"  {cls}: {count:,}")
        
        all_predictions = []
        n_batches = (len(features_df) + batch_size - 1) // batch_size
        
        iterator = range(0, len(features_df), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Predizione", total=n_batches)
        
        for i in iterator:
            batch_df = features_df.iloc[i:i+batch_size]
            batch_preds = self._predict_batch(batch_df)
            all_predictions.extend(batch_preds)
        
        y_pred = np.array(all_predictions)
        
        result = self._compute_metrics(y_true, y_pred, class_distribution)
        
        result.csv_path = str(csv_path)
        result.model_info = f"{self.model_dir.name} ({type(self.model).__name__})"
        result.processing_time_seconds = time.time() - start_time
        result.samples_per_second = len(y_true) / result.processing_time_seconds
        
        return result
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        class_distribution: Dict[str, int]
    ) -> EvaluationResult:
        """Calcola tutte le metriche."""
        from sklearn.metrics import (
            confusion_matrix, precision_score, recall_score, 
            f1_score, accuracy_score
        )
        
        result = EvaluationResult()
        result.total_samples = len(y_true)
        result.class_distribution = class_distribution
        
        result.accuracy = float(accuracy_score(y_true, y_pred))
        result.correct_predictions = int((y_true == y_pred).sum())
        
        if len(np.unique(y_true)) == 1:
            unique_class = y_true[0]
            if unique_class == 0:
                cm = np.array([[len(y_true) - (y_pred == 1).sum(), (y_pred == 1).sum()],
                               [0, 0]])
            else:
                cm = np.array([[0, 0],
                               [(y_pred == 0).sum(), len(y_true) - (y_pred == 0).sum()]])
        else:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            self.logger.warning(f"Confusion matrix shape inattesa: {cm.shape}")
        
        result.true_positives = int(tp)
        result.true_negatives = int(tn)
        result.false_positives = int(fp)
        result.false_negatives = int(fn)
        
        result.precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        result.recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        result.f1_score = float(
            2 * result.precision * result.recall / (result.precision + result.recall)
        ) if (result.precision + result.recall) > 0 else 0.0
        
        result.false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        result.false_negative_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        result.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        pred_counts = {
            'BENIGN': int((y_pred == 0).sum()),
            'ATTACK': int((y_pred == 1).sum())
        }
        result.predictions_distribution = pred_counts
        
        result.per_class_metrics = {
            'BENIGN': {
                'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                'recall': result.specificity,
                'f1': 0.0,
                'support': int((y_true == 0).sum())
            },
            'ATTACK': {
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1_score,
                'support': int((y_true == 1).sum())
            }
        }
        
        benign_metrics = result.per_class_metrics['BENIGN']
        if (benign_metrics['precision'] + benign_metrics['recall']) > 0:
            benign_metrics['f1'] = float(
                2 * benign_metrics['precision'] * benign_metrics['recall'] / 
                (benign_metrics['precision'] + benign_metrics['recall'])
            )
        
        return result


class LatencyBenchmarker:
    """Benchmarker per latenza inferenza."""
    
    def __init__(
        self, 
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts'
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logging.getLogger('sniffer.benchmark')
        
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            self.model = joblib.load(model_path)
        
        from .preprocessing import load_pipeline_artifacts, InferencePipeline
        artifacts = load_pipeline_artifacts(str(self.artifacts_dir), str(self.model_dir))
        self.pipeline = InferencePipeline(artifacts)
    
    def benchmark(
        self, 
        n_samples: int = 1000, 
        n_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict[str, Any]:
        """Esegue benchmark latenza."""
        
        with open(self.artifacts_dir / 'selected_features.json') as f:
            feature_names = json.load(f)
        
        np.random.seed(42)
        test_data = {name: np.random.randn() * 1000 for name in feature_names}
        
        for _ in range(warmup_iterations):
            for _ in range(min(100, n_samples)):
                X = self.pipeline.transform(test_data)
                _ = self.model.predict(X)
        
        latencies = []
        for iteration in range(n_iterations):
            start = time.perf_counter()
            for _ in range(n_samples):
                X = self.pipeline.transform(test_data)
                _ = self.model.predict(X)
            end = time.perf_counter()
            
            iteration_time = (end - start) / n_samples * 1000
            latencies.append(iteration_time)
        
        latencies = np.array(latencies)
        
        return {
            'n_samples_per_iteration': n_samples,
            'n_iterations': n_iterations,
            'warmup_iterations': warmup_iterations,
            'latency_mean_ms': float(latencies.mean()),
            'latency_std_ms': float(latencies.std()),
            'latency_min_ms': float(latencies.min()),
            'latency_max_ms': float(latencies.max()),
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'throughput_samples_per_sec': float(1000 / latencies.mean()) if latencies.mean() > 0 else 0
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Stampa risultati benchmark."""
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Samples per iteration: {results['n_samples_per_iteration']}")
        print(f"Iterations: {results['n_iterations']}")
        print("-" * 60)
        print(f"Mean latency:   {results['latency_mean_ms']:.3f} ms")
        print(f"Std deviation:  {results['latency_std_ms']:.3f} ms")
        print(f"Min latency:    {results['latency_min_ms']:.3f} ms")
        print(f"Max latency:    {results['latency_max_ms']:.3f} ms")
        print(f"P50 latency:    {results['latency_p50_ms']:.3f} ms")
        print(f"P95 latency:    {results['latency_p95_ms']:.3f} ms")
        print(f"P99 latency:    {results['latency_p99_ms']:.3f} ms")
        print("-" * 60)
        print(f"Throughput:     {results['throughput_samples_per_sec']:.0f} samples/sec")
        print("=" * 60)


def quick_evaluate(
    csv_path: str,
    model_dir: str = 'models/best_model',
    artifacts_dir: str = 'artifacts',
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> EvaluationResult:
    """
    Funzione helper per valutazione rapida.
    
    Args:
        csv_path: Path al CSV
        model_dir: Directory modello
        artifacts_dir: Directory artifacts
        sample_size: Campione (None = tutto)
        verbose: Output verboso
    
    Returns:
        EvaluationResult
    """
    evaluator = SnifferEvaluator(
        model_dir=model_dir,
        artifacts_dir=artifacts_dir
    )
    return evaluator.evaluate_csv(
        csv_path,
        sample_size=sample_size,
        verbose=verbose
    )