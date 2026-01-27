"""
NIDS Sniffer Evaluation Module
==============================

This module provides tools to evaluate the sniffer's performance:
1. Compare sniffer predictions against ground truth labels
2. Calculate metrics (F1, precision, recall, FPR)
3. Measure latency and throughput
4. Generate evaluation reports

Usage:
    # Test on CSV dataset
    evaluator = SnifferEvaluator(model_dir='models/best_model')
    results = evaluator.evaluate_csv('data/test.csv')
    
    # Test on PCAP with labels
    results = evaluator.evaluate_pcap('capture.pcap', 'labels.csv')
"""

import os
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

# Suppress sklearn feature name warnings (we use numpy arrays for speed)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
from tqdm import tqdm

from .features import FEATURE_NAMES, get_feature_columns_ordered
from .validator import FeatureValidator, find_column


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Evaluation metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    samples_total: int
    samples_benign: int
    samples_attack: int
    predictions_benign: int
    predictions_attack: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: np.ndarray
    classification_report: str
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p95_ms: float = 0.0
    throughput_flows_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'false_positive_rate': self.false_positive_rate,
            'samples': {
                'total': self.samples_total,
                'benign': self.samples_benign,
                'attack': self.samples_attack
            },
            'predictions': {
                'benign': self.predictions_benign,
                'attack': self.predictions_attack
            },
            'confusion': {
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            },
            'latency': {
                'mean_ms': self.latency_mean_ms,
                'std_ms': self.latency_std_ms,
                'p95_ms': self.latency_p95_ms
            },
            'throughput_flows_per_sec': self.throughput_flows_per_sec
        }
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n📊 METRICS:")
        print(f"  Accuracy:           {self.accuracy:.4f}")
        print(f"  Precision:          {self.precision:.4f}")
        print(f"  Recall:             {self.recall:.4f}")
        print(f"  F1 Score:           {self.f1_score:.4f}")
        print(f"  False Positive Rate: {self.false_positive_rate:.4f}")
        print(f"\n📈 SAMPLES:")
        print(f"  Total: {self.samples_total}")
        print(f"  Benign: {self.samples_benign}")
        print(f"  Attack: {self.samples_attack}")
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Latency (mean): {self.latency_mean_ms:.2f} ms")
        print(f"  Latency (p95):  {self.latency_p95_ms:.2f} ms")
        print(f"  Throughput:     {self.throughput_flows_per_sec:.1f} flows/sec")
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"  TP: {self.true_positives} | FP: {self.false_positives}")
        print(f"  FN: {self.false_negatives} | TN: {self.true_negatives}")
        print("\n" + "=" * 60)


# =============================================================================
# SNIFFER EVALUATOR
# =============================================================================

class SnifferEvaluator:
    """
    Evaluates sniffer/model performance on datasets.
    
    This class loads the model artifacts and runs inference on
    CSV datasets or PCAP files to calculate metrics.
    """
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts',
        label_column: str = 'Label'
    ):
        """
        Initialize evaluator with model artifacts.
        
        Args:
            model_dir: Directory containing trained model
            artifacts_dir: Directory containing scaler, features, etc.
            label_column: Column name for labels in CSV
        """
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.label_column = label_column
        
        # Setup logging
        self.logger = logging.getLogger('sniffer.evaluator')
        
        # Load artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessing artifacts."""
        self.logger.info("Loading artifacts...")
        
        # Load model
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found in {self.model_dir}")
        
        self.model = joblib.load(model_path)
        self.logger.info(f"Loaded model: {type(self.model).__name__}")
        
        # Load scaler
        scaler_path = self.artifacts_dir / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if self.scaler:
            self.logger.info(f"Loaded scaler")
        
        # Load feature selector (if exists)
        selector_path = self.artifacts_dir / 'feature_selector.pkl'
        self.selector = joblib.load(selector_path) if selector_path.exists() else None
        if self.selector:
            self.logger.info(f"Loaded feature selector")
        
        # Load scaler columns (features BEFORE selection - typically 77)
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if scaler_cols_path.exists():
            with open(scaler_cols_path, 'r') as f:
                self.scaler_columns = json.load(f)
            self.logger.info(f"Loaded scaler_columns: {len(self.scaler_columns)} features")
        else:
            self.scaler_columns = None
        
        # Load selected features (features AFTER selection - typically 30)
        features_path = self.artifacts_dir / 'selected_features.json'
        if not features_path.exists():
            features_path = self.model_dir / 'features_binary.json'
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            self.logger.info(f"Loaded selected_features: {len(self.selected_features)} features")
        else:
            self.selected_features = None
        
        # Determine pipeline strategy
        # CRITICAL FIX: Handle case where feature_selector.pkl doesn't exist
        # but we have scaler_columns (77) and selected_features (30)
        
        if self.selector is not None and self.scaler_columns is not None:
            # Best case: we have a selector pickle
            self.features_to_load = self.scaler_columns
            self.use_selector = True
            self.logger.info(f"Pipeline: {len(self.scaler_columns)} features -> scale -> select -> predict")
            
        elif self.selector is None and self.scaler_columns is not None and self.selected_features is not None:
            # No selector pickle, but we have both scaler_columns and selected_features
            # We need to create a simple selector that picks the right columns
            self.logger.warning("No feature_selector.pkl found - creating index-based selector from selected_features.json")
            
            # Find indices of selected features in scaler_columns
            scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(self.scaler_columns)}
            selected_indices = []
            for feat in self.selected_features:
                feat_lower = feat.strip().lower()
                if feat_lower in scaler_cols_lower:
                    selected_indices.append(scaler_cols_lower[feat_lower])
                else:
                    self.logger.warning(f"Selected feature '{feat}' not found in scaler_columns")
            
            if len(selected_indices) == len(self.selected_features):
                # Create a simple index-based selector function
                # CRITICAL: Do NOT sort! The order in selected_features.json is the order
                # the model expects to receive features
                self._selected_indices = selected_indices  # Keep original order!
                self.features_to_load = self.scaler_columns
                self.use_selector = True  # Will use custom selection
                self.logger.info(f"Pipeline: {len(self.scaler_columns)} features -> scale -> custom select ({len(self._selected_indices)} indices) -> predict")
            else:
                # Fallback: try to use selected features directly
                self.logger.warning("Could not match all selected features, trying direct load")
                self.features_to_load = self.selected_features
                self.use_selector = False
                self._selected_indices = None
                
        elif self.selected_features is not None:
            # No selector, no scaler_columns, but we have selected_features
            # Assume scaler was fitted on selected features only
            self.features_to_load = self.selected_features
            self.use_selector = False
            self._selected_indices = None
            self.logger.info(f"Pipeline: {len(self.selected_features)} selected features -> scale -> predict")
        else:
            # Fallback to all features (will likely fail if model expects fewer)
            self.features_to_load = get_feature_columns_ordered()
            self.use_selector = False
            self._selected_indices = None
            self.logger.warning("Using default feature list - this may cause issues!")
        
        self.logger.info("Artifacts loaded")
    
    def _prepare_csv_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare CSV data for prediction.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        # Find label column
        label_col = find_column(df.columns.tolist(), self.label_column)
        if not label_col:
            raise ValueError(f"Label column '{self.label_column}' not found in CSV")
        
        # Convert labels to binary
        labels = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        labels = labels.values
        
        # Use the pre-determined feature list
        feature_cols = self.features_to_load
        
        # Map column names (handle spacing differences)
        col_mapping = {}
        for target_col in feature_cols:
            found = find_column(df.columns.tolist(), target_col)
            if found:
                col_mapping[target_col] = found
            else:
                col_mapping[target_col] = None
        
        # Create features DataFrame
        features_data = {}
        missing_count = 0
        for target_col, source_col in col_mapping.items():
            if source_col and source_col in df.columns:
                features_data[target_col] = df[source_col].values
            else:
                features_data[target_col] = np.zeros(len(df))
                missing_count += 1
        
        if missing_count > 0:
            self.logger.warning(f"{missing_count} features not found in CSV, filled with zeros")
        
        features_df = pd.DataFrame(features_data)
        
        # Ensure column order matches expected
        features_df = features_df[feature_cols]
        
        # Handle NaN/Inf
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df, labels
    
    def _predict_batch(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, List[float]]:
        """
        Make predictions on a batch of features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Tuple of (predictions, latencies_per_sample)
        """
        latencies = []
        
        # Scale features
        if self.scaler is not None:
            start = time.perf_counter()
            features_scaled = self.scaler.transform(features_df)
            scale_time = time.perf_counter() - start
        else:
            features_scaled = features_df.values
            scale_time = 0
        
        # Select features
        if self.use_selector:
            start = time.perf_counter()
            if self.selector is not None:
                # Use sklearn selector
                features_selected = self.selector.transform(features_scaled)
            elif hasattr(self, '_selected_indices') and self._selected_indices is not None:
                # Use custom index-based selection
                features_selected = features_scaled[:, self._selected_indices]
            else:
                features_selected = features_scaled
            select_time = time.perf_counter() - start
        else:
            features_selected = features_scaled
            select_time = 0
        
        # Predict (batch)
        start = time.perf_counter()
        predictions = self.model.predict(features_selected)
        predict_time = time.perf_counter() - start
        
        # Calculate per-sample latency
        n_samples = len(features_df)
        per_sample_latency = (scale_time + select_time + predict_time) / n_samples
        latencies = [per_sample_latency * 1000] * n_samples  # Convert to ms
        
        return predictions, latencies
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        latencies: List[float]
    ) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            latencies: List of latencies in ms
            
        Returns:
            EvaluationMetrics object
        """
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Handle edge cases
        if len(cm.ravel()) != 4:
            # Only one class present
            if 1 not in y_true and 1 not in y_pred:
                tn = len(y_true)
                tp = fp = fn = 0
            elif 0 not in y_true and 0 not in y_pred:
                tp = len(y_true)
                tn = fp = fn = 0
        
        # Calculate FPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=['BENIGN', 'ATTACK'],
            zero_division=0
        )
        
        # Latency stats
        latencies_arr = np.array(latencies)
        
        return EvaluationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            false_positive_rate=fpr,
            samples_total=len(y_true),
            samples_benign=int(np.sum(y_true == 0)),
            samples_attack=int(np.sum(y_true == 1)),
            predictions_benign=int(np.sum(y_pred == 0)),
            predictions_attack=int(np.sum(y_pred == 1)),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            confusion_matrix=cm,
            classification_report=report,
            latency_mean_ms=float(np.mean(latencies_arr)) if len(latencies_arr) > 0 else 0,
            latency_std_ms=float(np.std(latencies_arr)) if len(latencies_arr) > 0 else 0,
            latency_p95_ms=float(np.percentile(latencies_arr, 95)) if len(latencies_arr) > 0 else 0,
            throughput_flows_per_sec=1000 / np.mean(latencies_arr) if np.mean(latencies_arr) > 0 else 0
        )
    
    def evaluate_csv(
        self,
        csv_path: str,
        sample_size: Optional[int] = None,
        batch_size: int = 10000,
        stratify: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate model on a CSV dataset.
        
        Args:
            csv_path: Path to CSV file
            sample_size: Optional sample size (None for full dataset)
            batch_size: Batch size for processing
            stratify: Whether to stratify sampling
            
        Returns:
            EvaluationMetrics object
        """
        self.logger.info(f"Evaluating on CSV: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path, low_memory=False)
        self.logger.info(f"Loaded {len(df)} rows")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            if stratify:
                # Find label column
                label_col = find_column(df.columns.tolist(), self.label_column)
                if label_col:
                    # Stratified sampling
                    from sklearn.model_selection import train_test_split
                    _, df = train_test_split(
                        df, test_size=sample_size,
                        stratify=df[label_col],
                        random_state=42
                    )
            else:
                df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Sampled to {len(df)} rows")
        
        # Prepare data
        features_df, y_true = self._prepare_csv_data(df)
        
        # Predict in batches
        all_predictions = []
        all_latencies = []
        
        for i in tqdm(range(0, len(features_df), batch_size), desc="Predicting"):
            batch = features_df.iloc[i:i+batch_size]
            predictions, latencies = self._predict_batch(batch)
            all_predictions.extend(predictions)
            all_latencies.extend(latencies)
        
        y_pred = np.array(all_predictions)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, all_latencies)
        
        return metrics
    
    def evaluate_multiple_csv(
        self,
        csv_paths: List[str],
        sample_per_file: Optional[int] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate on multiple CSV files.
        
        Args:
            csv_paths: List of CSV file paths
            sample_per_file: Optional sample size per file
            
        Returns:
            Dictionary mapping filename to metrics
        """
        results = {}
        
        for csv_path in csv_paths:
            filename = Path(csv_path).name
            self.logger.info(f"\nEvaluating: {filename}")
            
            try:
                metrics = self.evaluate_csv(csv_path, sample_size=sample_per_file)
                results[filename] = metrics
            except Exception as e:
                self.logger.error(f"Error evaluating {filename}: {e}")
                results[filename] = None
        
        return results
    
    def save_results(self, metrics: EvaluationMetrics, output_path: str):
        """Save evaluation results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        self.logger.info(f"Results saved to {output_path}")


# =============================================================================
# LATENCY BENCHMARKER
# =============================================================================

class LatencyBenchmarker:
    """
    Benchmarks inference latency with detailed breakdown.
    
    Measures time for each pipeline stage:
    - Feature preparation
    - Scaling
    - Feature selection
    - Model prediction
    """
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts'
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and artifacts."""
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        
        self.model = joblib.load(model_path)
        
        scaler_path = self.artifacts_dir / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        selector_path = self.artifacts_dir / 'feature_selector.pkl'
        self.selector = joblib.load(selector_path) if selector_path.exists() else None
        
        # Load scaler columns (features BEFORE selection)
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if scaler_cols_path.exists():
            with open(scaler_cols_path, 'r') as f:
                self.scaler_columns = json.load(f)
        else:
            self.scaler_columns = None
        
        # Load selected features (features AFTER selection)
        features_path = self.artifacts_dir / 'selected_features.json'
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
        else:
            self.selected_features = None
        
        # Determine feature configuration
        self._selected_indices = None
        
        if self.selector is not None and self.scaler_columns is not None:
            # Full pipeline: load scaler_columns -> scale -> select -> predict
            self.feature_columns = self.scaler_columns
            self.use_selector = True
            
        elif self.selector is None and self.scaler_columns is not None and self.selected_features is not None:
            # No selector pickle, but we have both lists
            # Create index-based selector
            scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(self.scaler_columns)}
            selected_indices = []
            for feat in self.selected_features:
                feat_lower = feat.strip().lower()
                if feat_lower in scaler_cols_lower:
                    selected_indices.append(scaler_cols_lower[feat_lower])
            
            if len(selected_indices) == len(self.selected_features):
                # CRITICAL: Do NOT sort! Keep original order from selected_features.json
                self._selected_indices = selected_indices
                self.feature_columns = self.scaler_columns
                self.use_selector = True
            else:
                # Fallback
                self.feature_columns = self.selected_features
                self.use_selector = False
                
        elif self.selected_features is not None:
            # No selector: load selected_features directly -> scale -> predict
            self.feature_columns = self.selected_features
            self.use_selector = False
        else:
            # Fallback
            self.feature_columns = get_feature_columns_ordered()
            self.use_selector = False
    
    def benchmark(
        self,
        n_samples: int = 1000,
        n_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run latency benchmark.
        
        Args:
            n_samples: Number of samples per iteration
            n_iterations: Number of benchmark iterations
            warmup_iterations: Warmup iterations (not measured)
            
        Returns:
            Dictionary with timing breakdown
        """
        # Create synthetic test data
        n_features = len(self.feature_columns)
        test_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=self.feature_columns
        )
        
        # Timing storage
        scale_times = []
        select_times = []
        predict_times = []
        total_times = []
        
        # Warmup
        for _ in range(warmup_iterations):
            if self.scaler:
                scaled = self.scaler.transform(test_data)
                if self.use_selector:
                    if self.selector is not None:
                        selected = self.selector.transform(scaled)
                    elif self._selected_indices is not None:
                        selected = scaled[:, self._selected_indices]
                    else:
                        selected = scaled
                    self.model.predict(selected)
                else:
                    self.model.predict(scaled)
            else:
                self.model.predict(test_data.values)
        
        # Benchmark
        for _ in tqdm(range(n_iterations), desc="Benchmarking"):
            total_start = time.perf_counter()
            
            # Scale
            if self.scaler:
                start = time.perf_counter()
                scaled = self.scaler.transform(test_data)
                scale_times.append(time.perf_counter() - start)
            else:
                scaled = test_data.values
                scale_times.append(0)
            
            # Select
            if self.use_selector:
                start = time.perf_counter()
                if self.selector is not None:
                    selected = self.selector.transform(scaled)
                elif self._selected_indices is not None:
                    selected = scaled[:, self._selected_indices]
                else:
                    selected = scaled
                select_times.append(time.perf_counter() - start)
            else:
                selected = scaled
                select_times.append(0)
            
            # Predict
            start = time.perf_counter()
            _ = self.model.predict(selected)
            predict_times.append(time.perf_counter() - start)
            
            total_times.append(time.perf_counter() - total_start)
        
        # Calculate statistics
        def stats(times):
            arr = np.array(times)
            return {
                'mean_ms': float(np.mean(arr) * 1000),
                'std_ms': float(np.std(arr) * 1000),
                'min_ms': float(np.min(arr) * 1000),
                'max_ms': float(np.max(arr) * 1000),
                'p95_ms': float(np.percentile(arr, 95) * 1000)
            }
        
        results = {
            'config': {
                'n_samples': n_samples,
                'n_iterations': n_iterations,
                'n_features': n_features,
                'model_type': type(self.model).__name__
            },
            'scaling': stats(scale_times),
            'feature_selection': stats(select_times),
            'prediction': stats(predict_times),
            'total': stats(total_times),
            'per_sample_ms': float(np.mean(total_times) * 1000 / n_samples),
            'throughput_samples_per_sec': float(n_samples / np.mean(total_times))
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Samples: {results['config']['n_samples']}")
        print(f"  Iterations: {results['config']['n_iterations']}")
        print(f"  Features: {results['config']['n_features']}")
        print(f"  Model: {results['config']['model_type']}")
        print(f"\nTimings (batch of {results['config']['n_samples']} samples):")
        print(f"  Scaling:          {results['scaling']['mean_ms']:.3f} ms")
        print(f"  Feature Selection: {results['feature_selection']['mean_ms']:.3f} ms")
        print(f"  Prediction:       {results['prediction']['mean_ms']:.3f} ms")
        print(f"  Total:            {results['total']['mean_ms']:.3f} ms")
        print(f"\nPer-sample latency: {results['per_sample_ms']:.4f} ms")
        print(f"Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_evaluate(
    csv_path: str,
    model_dir: str = 'models/best_model',
    artifacts_dir: str = 'artifacts',
    sample_size: Optional[int] = 10000
) -> EvaluationMetrics:
    """
    Quick evaluation on a CSV file.
    
    Args:
        csv_path: Path to CSV
        model_dir: Model directory
        artifacts_dir: Artifacts directory
        sample_size: Optional sample size
        
    Returns:
        EvaluationMetrics
    """
    evaluator = SnifferEvaluator(
        model_dir=model_dir,
        artifacts_dir=artifacts_dir
    )
    return evaluator.evaluate_csv(csv_path, sample_size=sample_size)


def run_benchmark(
    model_dir: str = 'models/best_model',
    artifacts_dir: str = 'artifacts',
    n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Run latency benchmark.
    
    Args:
        model_dir: Model directory
        artifacts_dir: Artifacts directory
        n_samples: Samples per iteration
        
    Returns:
        Benchmark results
    """
    benchmarker = LatencyBenchmarker(
        model_dir=model_dir,
        artifacts_dir=artifacts_dir
    )
    results = benchmarker.benchmark(n_samples=n_samples)
    benchmarker.print_results(results)
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sniffer Evaluation')
    parser.add_argument('--csv', help='CSV file to evaluate')
    parser.add_argument('--sample', type=int, help='Sample size')
    parser.add_argument('--model-dir', default='models/best_model')
    parser.add_argument('--artifacts-dir', default='artifacts')
    parser.add_argument('--benchmark', action='store_true', help='Run latency benchmark')
    parser.add_argument('--output', help='Output JSON path')
    
    args = parser.parse_args()
    
    if args.benchmark:
        results = run_benchmark(
            model_dir=args.model_dir,
            artifacts_dir=args.artifacts_dir
        )
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    
    elif args.csv:
        metrics = quick_evaluate(
            args.csv,
            model_dir=args.model_dir,
            artifacts_dir=args.artifacts_dir,
            sample_size=args.sample
        )
        metrics.print_summary()
        
        if args.output:
            evaluator = SnifferEvaluator(args.model_dir, args.artifacts_dir)
            evaluator.save_results(metrics, args.output)
    
    else:
        parser.print_help()