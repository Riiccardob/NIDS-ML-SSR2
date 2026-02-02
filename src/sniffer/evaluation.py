"""
NIDS-ML Sniffer - Evaluation Module v4 (WORKING VERSION)

Pipeline CORRETTA (identica al training):
1. Raw CSV → 44 features (scaler_columns.json)
2. RobustScaler
3. Clip(-10, 10)
4. Model predict (44 features)

NO feature selection post-scaling!
"""

import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

logger = logging.getLogger(__name__)

CLIP_VALUE = 10.0


def normalize_name(name: str) -> str:
    """Normalizza nome colonna."""
    return name.strip().lower()


def find_label_column(columns: List[str]) -> Optional[str]:
    """Trova colonna label."""
    for col in columns:
        if col.strip().lower() == 'label':
            return col
    return None


@dataclass
class EvaluationResult:
    """Risultato valutazione."""
    total_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    false_positive_rate: float = 0.0
    
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    processing_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    csv_path: str = ""
    model_info: str = ""
    features_matched: int = 0
    
    attack_prob_mean: float = 0.0
    benign_prob_mean: float = 0.0
    
    @property
    def f1(self) -> float:
        return self.f1_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'false_positive_rate': self.false_positive_rate,
            'class_distribution': self.class_distribution,
            'processing_time_seconds': self.processing_time_seconds,
            'csv_path': self.csv_path,
            'model_info': self.model_info,
            'features_matched': self.features_matched,
            'attack_prob_mean': self.attack_prob_mean,
            'benign_prob_mean': self.benign_prob_mean
        }
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"CSV:       {self.csv_path}")
        print(f"Model:     {self.model_info}")
        print(f"Samples:   {self.total_samples:,}")
        print(f"Features:  {self.features_matched}")
        print("-" * 60)
        print(f"F1 Score:  {self.f1_score:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"FPR:       {self.false_positive_rate:.4f}")
        print("-" * 60)
        print(f"TP: {self.true_positives:,}  TN: {self.true_negatives:,}")
        print(f"FP: {self.false_positives:,}  FN: {self.false_negatives:,}")
        print("-" * 60)
        print(f"Attack prob mean: {self.attack_prob_mean:.4f}")
        print(f"Benign prob mean: {self.benign_prob_mean:.4f}")
        print("=" * 60)


class SnifferEvaluator:
    """
    Valuta modello su CSV CIC-IDS2017.
    
    Pipeline:
        Raw CSV → 44 features → RobustScaler → Clip → Model (44 features)
    
    NO feature selection post-scaling!
    """
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts',
        clip_value: float = CLIP_VALUE
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.clip_value = clip_value
        self.logger = logging.getLogger('sniffer.evaluator')
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Carica artifacts."""
        # Scaler
        scaler_path = self.artifacts_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.scaler = joblib.load(scaler_path)
        self.logger.info(f"Scaler caricato: {type(self.scaler).__name__}")
        
        # Scaler columns (44 features)
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if not scaler_cols_path.exists():
            raise FileNotFoundError(f"scaler_columns.json not found")
        
        with open(scaler_cols_path, 'r') as f:
            self.scaler_columns = json.load(f)
        self.logger.info(f"Scaler columns: {len(self.scaler_columns)} features")
        
        # Model
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found in {self.model_dir}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.model = joblib.load(model_path)
        self.logger.info(f"Modello caricato: {type(self.model).__name__}")
        
        # Check if LightGBM Booster
        self.is_booster = hasattr(self.model, 'predict') and not hasattr(self.model, 'predict_proba')
        self.logger.info(f"Is LightGBM Booster: {self.is_booster}")
        
        self.logger.info(f"Pipeline: {len(self.scaler_columns)} -> scale -> clip -> predict")
    
    def _extract_features(self, df: pd.DataFrame) -> tuple:
        """Estrae feature nell'ordine di scaler_columns."""
        csv_cols_norm = {normalize_name(c): c for c in df.columns}
        
        X = np.zeros((len(df), len(self.scaler_columns)), dtype=np.float64)
        matched = 0
        
        for i, col in enumerate(self.scaler_columns):
            col_norm = normalize_name(col)
            
            csv_col = None
            if col_norm in csv_cols_norm:
                csv_col = csv_cols_norm[col_norm]
            else:
                for var in [col_norm.replace(' ', '_'), col_norm.replace('_', ' ')]:
                    if var in csv_cols_norm:
                        csv_col = csv_cols_norm[var]
                        break
            
            if csv_col and csv_col in df.columns:
                X[:, i] = df[csv_col].values
                matched += 1
        
        return X, matched
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocessa: clean → scale → clip."""
        # Handle inf/nan
        X = np.where(np.isinf(X), 0, X)
        X = np.where(np.isnan(X), 0, X)
        
        # Scale
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_scaled = self.scaler.transform(X)
        
        # Clip
        X_clipped = np.clip(X_scaled, -self.clip_value, self.clip_value)
        
        return X_clipped
    
    def _predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple:
        """Predice."""
        if self.is_booster:
            y_prob = self.model.predict(X)
            y_pred = (y_prob > threshold).astype(int)
        else:
            y_prob = self.model.predict_proba(X)[:, 1]
            y_pred = self.model.predict(X)
        
        return y_pred, y_prob
    
    def evaluate_csv(
        self,
        csv_path: str,
        sample_size: Optional[int] = None,
        batch_size: int = 50000,
        verbose: bool = True
    ) -> EvaluationResult:
        """Valuta su CSV."""
        start_time = time.time()
        csv_path = Path(csv_path)
        
        if verbose:
            print(f"Caricamento CSV: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Strip column names (CIC-IDS2017 has leading spaces)
        df.columns = df.columns.str.strip()
        
        if verbose:
            print(f"Righe totali: {len(df):,}")
        
        # Sample if needed
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            if verbose:
                print(f"Campionate: {len(df):,}")
        
        # Find label column
        label_col = find_label_column(df.columns.tolist())
        if not label_col:
            raise ValueError("Label column not found")
        
        # Extract labels
        labels = df[label_col].astype(str).str.strip().str.upper()
        y_true = (labels != 'BENIGN').astype(int).values
        class_distribution = df[label_col].value_counts().to_dict()
        
        if verbose:
            print(f"Distribuzione classi:")
            for cls, count in sorted(class_distribution.items()):
                print(f"  {cls}: {count:,}")
        
        # Extract features
        X_raw, matched = self._extract_features(df)
        
        if verbose:
            print(f"Features caricate: {matched}")
        
        # Preprocess
        X_processed = self._preprocess(X_raw)
        
        # Predict in batches
        all_preds = []
        all_probs = []
        
        n_batches = (len(X_processed) + batch_size - 1) // batch_size
        iterator = range(0, len(X_processed), batch_size)
        
        if verbose:
            iterator = tqdm(iterator, desc="Predizione", total=n_batches)
        
        for i in iterator:
            batch = X_processed[i:i+batch_size]
            y_pred, y_prob = self._predict(batch)
            all_preds.extend(y_pred)
            all_probs.extend(y_prob)
        
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Compute metrics
        result = self._compute_metrics(y_true, y_pred, y_prob, class_distribution)
        
        # Add metadata
        result.csv_path = str(csv_path)
        result.model_info = f"{self.model_dir.name} ({type(self.model).__name__})"
        result.processing_time_seconds = time.time() - start_time
        result.samples_per_second = len(y_true) / result.processing_time_seconds
        result.features_matched = matched
        
        return result
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_distribution: Dict[str, int]
    ) -> EvaluationResult:
        """Calcola metriche."""
        result = EvaluationResult()
        result.total_samples = len(y_true)
        result.class_distribution = class_distribution
        
        # Confusion matrix
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        
        result.true_positives = tp
        result.true_negatives = tn
        result.false_positives = fp
        result.false_negatives = fn
        
        # Metrics
        result.accuracy = (tp + tn) / result.total_samples
        result.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        result.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result.f1_score = (
            2 * result.precision * result.recall / (result.precision + result.recall)
            if (result.precision + result.recall) > 0 else 0.0
        )
        result.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Prob stats
        result.attack_prob_mean = float(y_prob[y_true == 1].mean()) if (y_true == 1).any() else 0.0
        result.benign_prob_mean = float(y_prob[y_true == 0].mean()) if (y_true == 0).any() else 0.0
        
        return result


class LatencyBenchmarker:
    """Benchmark latenza."""
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts'
    ):
        self.evaluator = SnifferEvaluator(
            model_dir=model_dir,
            artifacts_dir=artifacts_dir
        )
    
    def benchmark(
        self,
        n_samples: int = 1000,
        n_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict[str, Any]:
        """Esegue benchmark."""
        np.random.seed(42)
        X_test = np.random.randn(n_samples, len(self.evaluator.scaler_columns))
        X_test = np.clip(X_test, -10, 10)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = self.evaluator._predict(X_test[:100])
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = self.evaluator._predict(X_test)
            end = time.perf_counter()
            latencies.append((end - start) / n_samples * 1000)
        
        latencies = np.array(latencies)
        
        return {
            'n_samples': n_samples,
            'n_iterations': n_iterations,
            'latency_mean_ms': float(latencies.mean()),
            'latency_std_ms': float(latencies.std()),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'throughput_samples_per_sec': float(1000 / latencies.mean()) if latencies.mean() > 0 else 0
        }
    
    def print_results(self, results: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Mean latency:  {results['latency_mean_ms']:.3f} ms")
        print(f"P95 latency:   {results['latency_p95_ms']:.3f} ms")
        print(f"Throughput:    {results['throughput_samples_per_sec']:.0f} samples/sec")
        print("=" * 60)


def quick_evaluate(
    csv_path: str,
    model_dir: str = 'models/best_model',
    artifacts_dir: str = 'artifacts',
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> EvaluationResult:
    """Helper per valutazione rapida."""
    evaluator = SnifferEvaluator(
        model_dir=model_dir,
        artifacts_dir=artifacts_dir
    )
    return evaluator.evaluate_csv(
        csv_path,
        sample_size=sample_size,
        verbose=verbose
    )