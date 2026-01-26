"""
NIDS Feature Calibration Module
================================

This module provides tools to diagnose and compare feature extraction:
1. Compare sniffer-extracted features vs CSV ground truth
2. Identify systematic biases and offsets
3. Generate calibration reports
4. Suggest corrections for feature alignment

The goal is to ensure that features extracted by the Python sniffer
closely match those in the CIC-IDS2017 dataset (extracted by CICFlowMeter).

Usage:
    calibrator = FeatureCalibrator(artifacts_dir='artifacts')
    
    # Run full calibration
    report = calibrator.calibrate_from_csv('data/test.csv')
    report.print_summary()
    
    # Quick feature check
    issues = quick_feature_check('data/test.csv')
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

from .features import (
    FEATURE_NAMES, CRITICAL_FEATURES, get_feature_columns_ordered,
    FeatureExtractor
)
from .flow import Flow
from .validator import find_column


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    name: str
    csv_mean: float
    csv_std: float
    csv_min: float
    csv_max: float
    csv_median: float
    csv_zeros_pct: float
    sniffer_mean: Optional[float] = None
    sniffer_std: Optional[float] = None
    sniffer_min: Optional[float] = None
    sniffer_max: Optional[float] = None
    sniffer_median: Optional[float] = None
    sniffer_zeros_pct: Optional[float] = None
    mean_diff_pct: Optional[float] = None
    std_diff_pct: Optional[float] = None
    correlation: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'csv': {
                'mean': self.csv_mean,
                'std': self.csv_std,
                'min': self.csv_min,
                'max': self.csv_max,
                'median': self.csv_median,
                'zeros_pct': self.csv_zeros_pct
            },
            'sniffer': {
                'mean': self.sniffer_mean,
                'std': self.sniffer_std,
                'min': self.sniffer_min,
                'max': self.sniffer_max,
                'median': self.sniffer_median,
                'zeros_pct': self.sniffer_zeros_pct
            } if self.sniffer_mean is not None else None,
            'comparison': {
                'mean_diff_pct': self.mean_diff_pct,
                'std_diff_pct': self.std_diff_pct,
                'correlation': self.correlation,
                'ks_statistic': self.ks_statistic,
                'ks_pvalue': self.ks_pvalue
            } if self.mean_diff_pct is not None else None
        }


@dataclass
class CalibrationReport:
    """Complete calibration report."""
    timestamp: str
    csv_source: str
    n_samples: int
    feature_stats: List[FeatureStats]
    missing_features: List[str]
    zero_variance_features: List[str]
    high_discrepancy_features: List[str]
    critical_issues: List[str]
    recommendations: List[str]
    overall_score: float  # 0-100, higher is better
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'csv_source': self.csv_source,
            'n_samples': self.n_samples,
            'overall_score': self.overall_score,
            'summary': {
                'total_features': len(self.feature_stats),
                'missing_features': len(self.missing_features),
                'zero_variance_features': len(self.zero_variance_features),
                'high_discrepancy_features': len(self.high_discrepancy_features),
                'critical_issues': len(self.critical_issues)
            },
            'missing_features': self.missing_features,
            'zero_variance_features': self.zero_variance_features,
            'high_discrepancy_features': self.high_discrepancy_features,
            'critical_issues': self.critical_issues,
            'recommendations': self.recommendations,
            'feature_stats': [f.to_dict() for f in self.feature_stats]
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("FEATURE CALIBRATION REPORT")
        print("=" * 70)
        print(f"\nSource: {self.csv_source}")
        print(f"Samples: {self.n_samples}")
        print(f"Overall Score: {self.overall_score:.1f}/100")
        
        print(f"\n📊 FEATURE ANALYSIS:")
        print(f"  Total features: {len(self.feature_stats)}")
        print(f"  Missing in CSV: {len(self.missing_features)}")
        print(f"  Zero variance: {len(self.zero_variance_features)}")
        print(f"  High discrepancy: {len(self.high_discrepancy_features)}")
        
        if self.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.critical_issues)}):")
            for issue in self.critical_issues[:5]:  # Top 5
                print(f"  ❌ {issue}")
            if len(self.critical_issues) > 5:
                print(f"  ... and {len(self.critical_issues) - 5} more")
        
        if self.high_discrepancy_features:
            print(f"\n⚠️  HIGH DISCREPANCY FEATURES:")
            for feat in self.high_discrepancy_features[:10]:
                print(f"  - {feat}")
            if len(self.high_discrepancy_features) > 10:
                print(f"  ... and {len(self.high_discrepancy_features) - 10} more")
        
        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.recommendations:
                print(f"  • {rec}")
        
        print("\n" + "=" * 70)
    
    def save(self, path: str):
        """Save report to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# CALIBRATOR
# =============================================================================

class FeatureCalibrator:
    """
    Analyzes and calibrates feature extraction.
    
    Compares features from:
    1. CSV ground truth (CIC-IDS2017 dataset)
    2. Python sniffer extraction (if PCAP available)
    
    Identifies discrepancies and suggests corrections.
    """
    
    # Tolerance thresholds by feature category
    TOLERANCES = {
        'temporal': 0.30,   # IAT, Duration - 30% tolerance
        'count': 0.05,      # Packet counts, flags - 5% tolerance
        'size': 0.15,       # Bytes, lengths - 15% tolerance
        'rate': 0.50,       # Rates (depends on duration) - 50% tolerance
        'default': 0.20     # Default - 20% tolerance
    }
    
    # Feature category mapping
    FEATURE_CATEGORIES = {
        'temporal': [
            'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
            'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
            'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
            'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Active Mean',
            'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
            'Idle Max', 'Idle Min'
        ],
        'count': [
            'Total Fwd Packets', 'Total Backward Packets', 'Total Fwd Packets',
            'Total Backward Packets', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count',
            'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count', 'Subflow Fwd Packets',
            'Subflow Bwd Packets', 'act_data_pkt_fwd'
        ],
        'size': [
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'Max Packet Length',
            'Min Packet Length', 'Fwd Header Length', 'Bwd Header Length',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
            'Fwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Subflow Fwd Bytes', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'min_seg_size_forward'
        ],
        'rate': [
            'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s',
            'Down/Up Ratio', 'Fwd Avg Bulk Rate', 'Bwd Avg Bulk Rate'
        ]
    }
    
    def __init__(self, artifacts_dir: str = 'artifacts'):
        """
        Initialize calibrator.
        
        Args:
            artifacts_dir: Path to artifacts directory
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logging.getLogger('sniffer.calibrator')
        self.feature_extractor = FeatureExtractor()
        
        # Load scaler columns if available
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if scaler_cols_path.exists():
            with open(scaler_cols_path, 'r') as f:
                self.expected_features = json.load(f)
        else:
            self.expected_features = get_feature_columns_ordered()
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Get category for a feature."""
        for category, features in self.FEATURE_CATEGORIES.items():
            if feature_name in features:
                return category
        return 'default'
    
    def _get_tolerance(self, feature_name: str) -> float:
        """Get tolerance threshold for a feature."""
        category = self._get_feature_category(feature_name)
        return self.TOLERANCES.get(category, self.TOLERANCES['default'])
    
    def _calculate_stats(self, values: np.ndarray, name: str) -> FeatureStats:
        """Calculate statistics for a feature array."""
        # Handle edge cases
        values = values[~np.isnan(values)]
        values = values[~np.isinf(values)]
        
        if len(values) == 0:
            return FeatureStats(
                name=name,
                csv_mean=0.0, csv_std=0.0, csv_min=0.0,
                csv_max=0.0, csv_median=0.0, csv_zeros_pct=100.0
            )
        
        zeros_pct = (np.sum(values == 0) / len(values)) * 100
        
        return FeatureStats(
            name=name,
            csv_mean=float(np.mean(values)),
            csv_std=float(np.std(values)),
            csv_min=float(np.min(values)),
            csv_max=float(np.max(values)),
            csv_median=float(np.median(values)),
            csv_zeros_pct=float(zeros_pct)
        )
    
    def _compare_distributions(
        self,
        csv_values: np.ndarray,
        sniffer_values: np.ndarray,
        feature_stats: FeatureStats
    ) -> FeatureStats:
        """Compare two distributions and update stats."""
        # Filter invalid values
        csv_clean = csv_values[~np.isnan(csv_values) & ~np.isinf(csv_values)]
        sniff_clean = sniffer_values[~np.isnan(sniffer_values) & ~np.isinf(sniffer_values)]
        
        if len(sniff_clean) == 0:
            return feature_stats
        
        # Sniffer stats
        feature_stats.sniffer_mean = float(np.mean(sniff_clean))
        feature_stats.sniffer_std = float(np.std(sniff_clean))
        feature_stats.sniffer_min = float(np.min(sniff_clean))
        feature_stats.sniffer_max = float(np.max(sniff_clean))
        feature_stats.sniffer_median = float(np.median(sniff_clean))
        feature_stats.sniffer_zeros_pct = float((np.sum(sniff_clean == 0) / len(sniff_clean)) * 100)
        
        # Comparison metrics
        if feature_stats.csv_mean != 0:
            feature_stats.mean_diff_pct = abs(
                (feature_stats.sniffer_mean - feature_stats.csv_mean) / feature_stats.csv_mean
            ) * 100
        else:
            feature_stats.mean_diff_pct = 0 if feature_stats.sniffer_mean == 0 else 100
        
        if feature_stats.csv_std != 0:
            feature_stats.std_diff_pct = abs(
                (feature_stats.sniffer_std - feature_stats.csv_std) / feature_stats.csv_std
            ) * 100
        else:
            feature_stats.std_diff_pct = 0 if feature_stats.sniffer_std == 0 else 100
        
        # Correlation (if enough samples)
        if len(csv_clean) == len(sniff_clean) and len(csv_clean) > 10:
            try:
                corr = np.corrcoef(csv_clean, sniff_clean)[0, 1]
                feature_stats.correlation = float(corr) if not np.isnan(corr) else None
            except:
                feature_stats.correlation = None
        
        # KS test (sample if too large)
        sample_size = min(1000, len(csv_clean), len(sniff_clean))
        if sample_size > 10:
            try:
                csv_sample = np.random.choice(csv_clean, sample_size, replace=False)
                sniff_sample = np.random.choice(sniff_clean, sample_size, replace=False)
                ks_stat, ks_pval = scipy_stats.ks_2samp(csv_sample, sniff_sample)
                feature_stats.ks_statistic = float(ks_stat)
                feature_stats.ks_pvalue = float(ks_pval)
            except:
                pass
        
        return feature_stats
    
    def calibrate_from_csv(
        self,
        csv_path: str,
        sample_size: Optional[int] = 10000
    ) -> CalibrationReport:
        """
        Analyze CSV dataset for feature calibration.
        
        This analyzes the CSV to understand feature distributions
        and identify potential issues.
        
        Args:
            csv_path: Path to CSV file
            sample_size: Optional sample size for analysis
            
        Returns:
            CalibrationReport
        """
        from datetime import datetime
        
        self.logger.info(f"Analyzing CSV: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path, low_memory=False)
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        n_samples = len(df)
        self.logger.info(f"Analyzing {n_samples} samples")
        
        # Analyze each feature
        feature_stats_list = []
        missing_features = []
        zero_variance_features = []
        high_discrepancy_features = []
        critical_issues = []
        
        for feature_name in tqdm(self.expected_features, desc="Analyzing features"):
            # Find column in CSV
            col_name = find_column(df.columns.tolist(), feature_name)
            
            if not col_name:
                missing_features.append(feature_name)
                if feature_name in CRITICAL_FEATURES:
                    critical_issues.append(f"CRITICAL feature missing in CSV: {feature_name}")
                continue
            
            # Get values
            values = df[col_name].values.astype(float)
            
            # Calculate stats
            stats = self._calculate_stats(values, feature_name)
            
            # Check for zero variance
            if stats.csv_std == 0:
                zero_variance_features.append(feature_name)
            
            feature_stats_list.append(stats)
        
        # Check for issues
        recommendations = []
        
        if missing_features:
            recommendations.append(
                f"Add missing features to CSV or adjust feature extraction: "
                f"{', '.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}"
            )
        
        if zero_variance_features:
            recommendations.append(
                f"Features with zero variance may not be useful: "
                f"{', '.join(zero_variance_features[:3])}{'...' if len(zero_variance_features) > 3 else ''}"
            )
        
        # Calculate overall score
        n_total = len(self.expected_features)
        n_present = n_total - len(missing_features)
        n_useful = n_present - len(zero_variance_features)
        
        coverage_score = (n_present / n_total) * 50  # 50 points for coverage
        usefulness_score = (n_useful / n_total) * 50  # 50 points for usefulness
        overall_score = coverage_score + usefulness_score
        
        # Build report
        report = CalibrationReport(
            timestamp=datetime.now().isoformat(),
            csv_source=str(csv_path),
            n_samples=n_samples,
            feature_stats=feature_stats_list,
            missing_features=missing_features,
            zero_variance_features=zero_variance_features,
            high_discrepancy_features=high_discrepancy_features,
            critical_issues=critical_issues,
            recommendations=recommendations,
            overall_score=overall_score
        )
        
        return report
    
    def analyze_feature_importance(
        self,
        csv_path: str,
        label_column: str = 'Label',
        sample_size: int = 10000
    ) -> Dict[str, float]:
        """
        Quick analysis of feature importance using variance and correlation with label.
        
        Args:
            csv_path: Path to CSV
            label_column: Label column name
            sample_size: Sample size for analysis
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        df = pd.read_csv(csv_path, low_memory=False)
        
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Find label column
        label_col = find_column(df.columns.tolist(), label_column)
        if not label_col:
            raise ValueError(f"Label column '{label_column}' not found")
        
        # Convert labels to binary
        labels = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        
        importance = {}
        
        for feature_name in self.expected_features:
            col_name = find_column(df.columns.tolist(), feature_name)
            if not col_name:
                continue
            
            values = df[col_name].values.astype(float)
            values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
            
            # Calculate importance based on correlation with label
            try:
                corr = abs(np.corrcoef(values, labels)[0, 1])
                if np.isnan(corr):
                    corr = 0
            except:
                corr = 0
            
            # Also consider variance (normalized)
            variance = np.var(values)
            max_val = np.max(np.abs(values))
            norm_variance = variance / (max_val ** 2) if max_val > 0 else 0
            
            # Combined score
            importance[feature_name] = 0.7 * corr + 0.3 * min(norm_variance, 1.0)
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_feature_check(csv_path: str, artifacts_dir: str = 'artifacts') -> Dict[str, Any]:
    """
    Quick check of feature compatibility.
    
    Args:
        csv_path: Path to CSV file
        artifacts_dir: Artifacts directory
        
    Returns:
        Dictionary with check results
    """
    calibrator = FeatureCalibrator(artifacts_dir=artifacts_dir)
    report = calibrator.calibrate_from_csv(csv_path, sample_size=1000)
    
    return {
        'compatible': len(report.critical_issues) == 0,
        'score': report.overall_score,
        'missing_features': report.missing_features,
        'critical_issues': report.critical_issues,
        'recommendations': report.recommendations
    }


def generate_calibration_report(
    csv_path: str,
    output_path: str,
    artifacts_dir: str = 'artifacts'
):
    """
    Generate and save a full calibration report.
    
    Args:
        csv_path: Path to CSV file
        output_path: Path for output JSON report
        artifacts_dir: Artifacts directory
    """
    calibrator = FeatureCalibrator(artifacts_dir=artifacts_dir)
    report = calibrator.calibrate_from_csv(csv_path)
    report.print_summary()
    report.save(output_path)
    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Calibration')
    parser.add_argument('csv', help='CSV file to analyze')
    parser.add_argument('--artifacts-dir', default='artifacts')
    parser.add_argument('--output', help='Output JSON path')
    parser.add_argument('--sample', type=int, default=10000, help='Sample size')
    parser.add_argument('--importance', action='store_true',
                       help='Show feature importance analysis')
    
    args = parser.parse_args()
    
    calibrator = FeatureCalibrator(artifacts_dir=args.artifacts_dir)
    
    if args.importance:
        importance = calibrator.analyze_feature_importance(args.csv, sample_size=args.sample)
        print("\nFeature Importance (top 20):")
        for i, (feat, score) in enumerate(list(importance.items())[:20]):
            print(f"  {i+1:2d}. {feat}: {score:.4f}")
    else:
        report = calibrator.calibrate_from_csv(args.csv, sample_size=args.sample)
        report.print_summary()
        
        if args.output:
            report.save(args.output)
