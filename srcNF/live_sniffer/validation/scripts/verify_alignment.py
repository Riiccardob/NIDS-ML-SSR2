#!/usr/bin/env python3
"""
Feature Alignment Verification Tool.

Verifica che le feature estratte dallo sniffer siano allineate
con quelle del training set (stessa unità di misura, scala, range).

USAGE:
    python3 verify_alignment.py --pcap test.pcap --training-sample train_sample.csv
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from nfstream import NFStreamer

# Import sniffer components
PROJECT_ROOT = Path(__file__).resolve().parents[2] / "srcNF" / "live_sniffer"
sys.path.insert(0, str(PROJECT_ROOT))
from core.feature_mapper import FeatureMapper
from config import REQUIRED_FEATURES


class FeatureAlignmentChecker:
    """Verifica allineamento feature sniffer vs training."""
    
    def __init__(self, training_sample_path: Path):
        """
        Inizializza checker.
        
        Args:
            training_sample_path: Path al CSV con sample del training set
        """
        self.training_sample = pd.read_csv(training_sample_path)
        self.feature_mapper = FeatureMapper()
        
        # Verifica che tutte le feature richieste esistano nel training
        missing = set(REQUIRED_FEATURES) - set(self.training_sample.columns)
        if missing:
            print(f"WARNING: Features missing in training sample: {missing}")
    
    def extract_features_from_pcap(self, pcap_path: Path) -> pd.DataFrame:
        """
        Estrae feature da PCAP usando lo sniffer.
        
        Args:
            pcap_path: Path al file PCAP
        
        Returns:
            DataFrame con feature estratte
        """
        
        print(f"\nExtracting features from {pcap_path.name}...")
        
        streamer = NFStreamer(source=str(pcap_path))
        
        features_list = []
        flow_count = 0
        
        for flow in streamer:
            flow_count += 1
            
            # Estrai feature vector
            features = self.feature_mapper.extract_features(flow)
            
            # Converti in dict
            feature_dict = {
                name: float(features[i])
                for i, name in enumerate(REQUIRED_FEATURES)
            }
            
            features_list.append(feature_dict)
            
            if flow_count % 100 == 0:
                print(f"  Processed {flow_count} flows...")
        
        print(f"  Total flows extracted: {flow_count}")
        
        return pd.DataFrame(features_list)
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcola statistiche per ogni feature.
        
        Args:
            df: DataFrame con feature
        
        Returns:
            Dict {feature_name: {mean, std, min, max, median}}
        """
        
        stats = {}
        
        for col in REQUIRED_FEATURES:
            if col not in df.columns:
                continue
            
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
            }
        
        return stats
    
    def check_alignment(
        self,
        sniffer_stats: Dict[str, Dict[str, float]],
        training_stats: Dict[str, Dict[str, float]],
        tolerance: float = 10.0
    ) -> Dict[str, Any]:
        """
        Verifica allineamento tra sniffer e training.
        
        Args:
            sniffer_stats: Statistiche da sniffer
            training_stats: Statistiche da training
            tolerance: Tolleranza percentuale per differenze
        
        Returns:
            Dict con risultati verifica
        """
        
        results = {
            'aligned': [],
            'misaligned': [],
            'missing': [],
            'warnings': [],
        }
        
        for feature in REQUIRED_FEATURES:
            if feature not in sniffer_stats:
                results['missing'].append({
                    'feature': feature,
                    'reason': 'Not extracted by sniffer'
                })
                continue
            
            if feature not in training_stats:
                results['missing'].append({
                    'feature': feature,
                    'reason': 'Not in training sample'
                })
                continue
            
            s_stats = sniffer_stats[feature]
            t_stats = training_stats[feature]
            
            # Check scale (ordine di grandezza)
            s_mean = s_stats['mean']
            t_mean = t_stats['mean']
            
            if s_mean == 0 and t_mean == 0:
                # Entrambi zero - OK
                results['aligned'].append({
                    'feature': feature,
                    'reason': 'Both zero (feature not present in traffic)'
                })
                continue
            
            # Calcola differenza percentuale
            if t_mean != 0:
                diff_pct = abs((s_mean - t_mean) / t_mean) * 100
            else:
                diff_pct = 100.0 if s_mean != 0 else 0.0
            
            # Check se scala diversa (es. secondi vs millisecondi)
            scale_factor = s_mean / t_mean if t_mean != 0 else 0
            
            if diff_pct > tolerance:
                # Possibile misalignment
                
                # Check se fattore 1000 (secondi vs millisecondi)
                if abs(scale_factor - 1000) < 10 or abs(scale_factor - 0.001) < 0.00001:
                    results['misaligned'].append({
                        'feature': feature,
                        'issue': 'SCALE_MISMATCH',
                        'sniffer_mean': s_mean,
                        'training_mean': t_mean,
                        'scale_factor': scale_factor,
                        'likely_cause': 'Seconds vs Milliseconds conversion',
                        'severity': 'CRITICAL'
                    })
                else:
                    results['misaligned'].append({
                        'feature': feature,
                        'issue': 'VALUE_MISMATCH',
                        'sniffer_mean': s_mean,
                        'training_mean': t_mean,
                        'diff_pct': diff_pct,
                        'severity': 'HIGH' if diff_pct > 50 else 'MEDIUM'
                    })
            else:
                # Allineato
                results['aligned'].append({
                    'feature': feature,
                    'sniffer_mean': s_mean,
                    'training_mean': t_mean,
                    'diff_pct': diff_pct
                })
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Stampa report di allineamento."""
        
        print("\n" + "="*70)
        print("FEATURE ALIGNMENT VERIFICATION REPORT")
        print("="*70)
        
        total = len(REQUIRED_FEATURES)
        aligned = len(results['aligned'])
        misaligned = len(results['misaligned'])
        missing = len(results['missing'])
        
        print(f"\nSUMMARY:")
        print(f"  Total features: {total}")
        print(f"  Aligned:        {aligned} ({aligned/total*100:.1f}%)")
        print(f"  Misaligned:     {misaligned} ({misaligned/total*100:.1f}%)")
        print(f"  Missing:        {missing} ({missing/total*100:.1f}%)")
        
        # Misaligned features (CRITICAL)
        if results['misaligned']:
            print("\n" + "!"*70)
            print("MISALIGNED FEATURES (ACTION REQUIRED)")
            print("!"*70)
            
            for item in results['misaligned']:
                print(f"\n  Feature: {item['feature']}")
                print(f"  Issue: {item['issue']}")
                print(f"  Severity: {item['severity']}")
                
                if item['issue'] == 'SCALE_MISMATCH':
                    print(f"  Sniffer mean:  {item['sniffer_mean']:.2e}")
                    print(f"  Training mean: {item['training_mean']:.2e}")
                    print(f"  Scale factor:  {item['scale_factor']:.2f}x")
                    print(f"  Likely cause:  {item['likely_cause']}")
                    print(f"  FIX: Check unit conversion in feature_mapper.py")
                else:
                    print(f"  Sniffer mean:  {item['sniffer_mean']:.4f}")
                    print(f"  Training mean: {item['training_mean']:.4f}")
                    print(f"  Difference:    {item['diff_pct']:.1f}%")
        
        # Missing features
        if results['missing']:
            print("\n" + "!"*70)
            print("MISSING FEATURES")
            print("!"*70)
            
            for item in results['missing']:
                print(f"  {item['feature']}: {item['reason']}")
        
        # Aligned features (OK)
        if results['aligned']:
            print("\n" + "="*70)
            print("ALIGNED FEATURES (OK)")
            print("="*70)
            
            print(f"\n{len(results['aligned'])} features correctly aligned:")
            for item in results['aligned'][:10]:  # Mostra prime 10
                if 'diff_pct' in item:
                    print(f"  {item['feature']:30s} diff: {item['diff_pct']:5.2f}%")
                else:
                    print(f"  {item['feature']:30s} {item['reason']}")
            
            if len(results['aligned']) > 10:
                print(f"  ... and {len(results['aligned'])-10} more")
        
        # Final verdict
        print("\n" + "="*70)
        if misaligned == 0 and missing == 0:
            print("STATUS: PASS - All features aligned correctly")
            print("Sniffer is ready for production")
        elif misaligned > 0:
            print("STATUS: FAIL - Critical misalignment detected")
            print("Fix required before production deployment")
        else:
            print("STATUS: WARNING - Some features missing")
            print("Review and validate before deployment")
        print("="*70)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Verify feature alignment between sniffer and training set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

1. Verify alignment with training sample:
   python3 verify_alignment.py \\
       --pcap test_traffic.pcap \\
       --training-sample train_sample.csv

2. Custom tolerance:
   python3 verify_alignment.py \\
       --pcap test.pcap \\
       --training-sample train.csv \\
       --tolerance 5.0

NOTES:
- training-sample should be a CSV with ~10k-100k rows from training set
- PCAP should contain similar traffic types (attacks + benign)
- Tolerance is in percentage (default: 10%)
        """
    )
    
    parser.add_argument(
        '--pcap',
        type=Path,
        required=True,
        help='PCAP file to extract features from'
    )
    
    parser.add_argument(
        '--training-sample',
        type=Path,
        required=True,
        help='CSV file with training set sample'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=10.0,
        help='Tolerance for differences (percent, default: 10.0)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed report to file'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pcap.exists():
        print(f"ERROR: PCAP file not found: {args.pcap}")
        return 1
    
    if not args.training_sample.exists():
        print(f"ERROR: Training sample not found: {args.training_sample}")
        return 1
    
    # Run verification
    print("="*70)
    print("FEATURE ALIGNMENT VERIFICATION")
    print("="*70)
    print(f"PCAP:            {args.pcap}")
    print(f"Training sample: {args.training_sample}")
    print(f"Tolerance:       {args.tolerance}%")
    
    try:
        checker = FeatureAlignmentChecker(args.training_sample)
        
        # Extract features from PCAP
        sniffer_df = checker.extract_features_from_pcap(args.pcap)
        
        # Compute statistics
        print("\nComputing statistics...")
        sniffer_stats = checker.compute_statistics(sniffer_df)
        training_stats = checker.compute_statistics(checker.training_sample)
        
        # Check alignment
        print("Checking alignment...")
        results = checker.check_alignment(sniffer_stats, training_stats, args.tolerance)
        
        # Print report
        checker.print_report(results)
        
        # Save if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {args.output}")
        
        # Exit code
        if results['misaligned']:
            return 1  # Fail
        else:
            return 0  # Pass
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
