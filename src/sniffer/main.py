#!/usr/bin/env python3
"""
================================================================================
NIDS-ML Sniffer - Main Entry Point
================================================================================

Unified CLI for all NIDS operations: validation, evaluation, calibration,
PCAP analysis, and live capture.

Usage:
    python -m src.sniffer.main <command> [options]
    
    # Or directly
    python src/sniffer/main.py <command> [options]

Commands:
    config      Show/manage configuration
    validate    Validate feature extraction against CSV
    calibrate   Run feature calibration diagnostics  
    evaluate    Evaluate model performance on CSV
    benchmark   Run latency benchmark
    pcap        Analyze PCAP file
    live        Start live packet capture
    
Examples:
    # Show current configuration
    python -m src.sniffer.main config --show
    
    # Validate with specific model
    python -m src.sniffer.main validate --csv data/raw/Tuesday-WorkingHours.pcap_ISCX.csv \
        --model-type xgboost --model-version bayesian_trials50_cv5
    
    # Evaluate all CIC-IDS2017 days
    python -m src.sniffer.main evaluate --all-days --sample 10000
    
    # Analyze PCAP with LightGBM model
    python -m src.sniffer.main pcap --file data/pcap/Tuesday-WorkingHours.pcap \
        --model-type lightgbm
    
    # Live capture
    sudo python -m src.sniffer.main live --interface eth0 --duration 300

================================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sniffer.config import (
    Config, PathConfig, ModelConfig, SnifferConfig, ValidationConfig,
    get_config, set_config, load_config, print_config_summary,
    list_available_models, get_cicids_csv_path, get_cicids_pcap_path
)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def create_config_from_args(args) -> Config:
    """Create Config object from command line arguments."""
    config = get_config()
    
    # Update model config
    if hasattr(args, 'model_type') and args.model_type:
        config.model.model_type = args.model_type
    if hasattr(args, 'model_version') and args.model_version:
        config.model.model_version = args.model_version
    if hasattr(args, 'task_type') and args.task_type:
        config.model.task_type = args.task_type
    if hasattr(args, 'confidence') and args.confidence:
        config.model.confidence_threshold = args.confidence
    
    # Update paths
    if hasattr(args, 'model_dir') and args.model_dir:
        config.paths.models_dir = Path(args.model_dir)
    if hasattr(args, 'artifacts_dir') and args.artifacts_dir:
        config.paths.artifacts_dir = Path(args.artifacts_dir)
    if hasattr(args, 'log_dir') and args.log_dir:
        config.paths.logs_dir = Path(args.log_dir)
    
    # Update sniffer config
    if hasattr(args, 'interface') and args.interface:
        config.sniffer.interface = args.interface
    if hasattr(args, 'filter') and args.filter:
        config.sniffer.bpf_filter = args.filter
    if hasattr(args, 'firewall') and args.firewall:
        config.sniffer.firewall_enabled = args.firewall
    if hasattr(args, 'firewall_execute') and args.firewall_execute:
        config.sniffer.firewall_dry_run = not args.firewall_execute
    
    set_config(config)
    return config


# ==============================================================================
# COMMAND: CONFIG
# ==============================================================================

def cmd_config(args):
    """Show or manage configuration."""
    config = create_config_from_args(args)
    
    if args.show:
        print_config_summary(config)
    
    if args.list_models:
        print("\n📦 Available Models:")
        available = list_available_models(config)
        if not available:
            print("   No models found!")
        else:
            for model_type, versions in available.items():
                print(f"\n   {model_type}:")
                for v in versions:
                    print(f"      - {v}")
    
    if args.list_data:
        print("\n📁 CIC-IDS2017 Dataset Files:")
        print("\n   CSV Files (data/raw/):")
        for day, info in config.dataset.CICIDS2017_FILES.items():
            csv_path = config.paths.raw_csv_dir / info['csv']
            exists = "✓" if csv_path.exists() else "✗"
            attacks = ", ".join(info['attacks']) if info['attacks'] else "Benign only"
            print(f"      {exists} {day}: {info['csv']}")
            print(f"         Attacks: {attacks}")
        
        print("\n   PCAP Files (data/pcap/):")
        seen_pcaps = set()
        for day, info in config.dataset.CICIDS2017_FILES.items():
            if info['pcap'] not in seen_pcaps:
                pcap_path = config.paths.pcap_dir / info['pcap']
                exists = "✓" if pcap_path.exists() else "✗"
                print(f"      {exists} {info['pcap']}")
                seen_pcaps.add(info['pcap'])
    
    if args.save:
        config.save_yaml(args.save)
        print(f"\nConfiguration saved to: {args.save}")


# ==============================================================================
# COMMAND: VALIDATE
# ==============================================================================

def cmd_validate(args):
    """Validate feature extraction against CSV."""
    from src.sniffer.validator import FeatureValidator, ValidationMode
    
    config = create_config_from_args(args)
    
    # Determine CSV path
    if args.csv:
        csv_path = Path(args.csv)
    elif args.day:
        csv_path = get_cicids_csv_path(args.day, config)
    else:
        print("Error: Specify --csv or --day")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("NIDS-ML Feature Validation")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print("=" * 60)
    
    validator = FeatureValidator(
        artifacts_dir=str(config.paths.artifacts_dir),
        tolerances={
            'temporal': config.validation.temporal_tolerance,
            'count': config.validation.count_tolerance,
            'size': config.validation.size_tolerance,
            'rate': config.validation.rate_tolerance
        }
    )
    
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=args.sample or 1000, low_memory=False)
    
    report = validator.validate_csv_coverage(df)
    report.print_summary()
    
    if args.output:
        report.save_json(args.output)
        print(f"\nReport saved to: {args.output}")


# ==============================================================================
# COMMAND: CALIBRATE
# ==============================================================================

def cmd_calibrate(args):
    """Run feature calibration diagnostics."""
    from src.sniffer.calibration import FeatureCalibrator
    
    config = create_config_from_args(args)
    
    # Determine CSV path
    if args.csv:
        csv_path = Path(args.csv)
    elif args.day:
        csv_path = get_cicids_csv_path(args.day, config)
    else:
        print("Error: Specify --csv or --day")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("NIDS-ML Feature Calibration")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print("=" * 60)
    
    calibrator = FeatureCalibrator(artifacts_dir=str(config.paths.artifacts_dir))
    
    if args.importance:
        importance = calibrator.analyze_feature_importance(
            str(csv_path),
            sample_size=args.sample or 10000
        )
        print("\nFeature Importance (top 30):")
        for i, (feat, score) in enumerate(list(importance.items())[:30]):
            print(f"  {i+1:2d}. {feat}: {score:.4f}")
    else:
        report = calibrator.calibrate_from_csv(
            str(csv_path),
            sample_size=args.sample or 10000
        )
        report.print_summary()
        
        if args.output:
            report.save(args.output)
            print(f"\nReport saved to: {args.output}")


# ==============================================================================
# COMMAND: EVALUATE
# ==============================================================================

def cmd_evaluate(args):
    """Evaluate model performance on CSV."""
    from src.sniffer.evaluation import SnifferEvaluator
    
    config = create_config_from_args(args)
    
    print("=" * 60)
    print("NIDS-ML Model Evaluation")
    print("=" * 60)
    print(f"Model: {config.model.model_type} / {config.model.model_version}")
    print(f"Task:  {config.model.task_type}")
    print("=" * 60)
    
    # Get model directory
    model_dir = config.model.get_model_dir(config.paths)
    
    evaluator = SnifferEvaluator(
        model_dir=str(model_dir),
        artifacts_dir=str(config.paths.artifacts_dir)
    )
    
    results = {}
    
    if args.all_days:
        # Evaluate on all CIC-IDS2017 days
        print("\nEvaluating all CIC-IDS2017 days...")
        for day, info in config.dataset.CICIDS2017_FILES.items():
            csv_path = config.paths.raw_csv_dir / info['csv']
            if csv_path.exists():
                print(f"\n--- {day} ---")
                try:
                    metrics = evaluator.evaluate_csv(
                        str(csv_path),
                        sample_size=args.sample,
                        batch_size=args.batch_size or 10000
                    )
                    results[day] = metrics.to_dict()
                    print(f"F1: {metrics.f1_score:.4f} | FPR: {metrics.false_positive_rate:.4f}")
                except Exception as e:
                    print(f"Error: {e}")
                    results[day] = {'error': str(e)}
            else:
                print(f"\n--- {day} --- SKIPPED (file not found)")
    else:
        # Single CSV evaluation
        if args.csv:
            csv_path = Path(args.csv)
        elif args.day:
            csv_path = get_cicids_csv_path(args.day, config)
        else:
            print("Error: Specify --csv, --day, or --all-days")
            sys.exit(1)
        
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
        
        print(f"CSV: {csv_path}")
        print(f"Sample size: {args.sample or 'full dataset'}")
        
        metrics = evaluator.evaluate_csv(
            str(csv_path),
            sample_size=args.sample,
            batch_size=args.batch_size or 10000
        )
        metrics.print_summary()
        results = metrics.to_dict()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


# ==============================================================================
# COMMAND: BENCHMARK
# ==============================================================================

def cmd_benchmark(args):
    """Run latency benchmark."""
    from src.sniffer.evaluation import LatencyBenchmarker
    
    config = create_config_from_args(args)
    
    print("=" * 60)
    print("NIDS-ML Latency Benchmark")
    print("=" * 60)
    print(f"Model: {config.model.model_type} / {config.model.model_version}")
    print("=" * 60)
    
    model_dir = config.model.get_model_dir(config.paths)
    
    benchmarker = LatencyBenchmarker(
        model_dir=str(model_dir),
        artifacts_dir=str(config.paths.artifacts_dir)
    )
    
    results = benchmarker.benchmark(
        n_samples=args.samples or 1000,
        n_iterations=args.iterations or 10,
        warmup_iterations=args.warmup or 3
    )
    
    benchmarker.print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


# ==============================================================================
# COMMAND: PCAP
# ==============================================================================

def cmd_pcap(args):
    """Analyze PCAP file."""
    from src.sniffer.engine import SnifferEngine
    
    config = create_config_from_args(args)
    
    # Determine PCAP path
    if args.file:
        pcap_path = Path(args.file)
    elif args.day:
        pcap_path = get_cicids_pcap_path(args.day, config)
    else:
        print("Error: Specify --file or --day")
        sys.exit(1)
    
    if not pcap_path.exists():
        print(f"Error: PCAP file not found: {pcap_path}")
        sys.exit(1)
    
    # Get verbose flag (default False)
    verbose = getattr(args, 'verbose', False)
    
    print("=" * 60)
    print("NIDS-ML PCAP Analysis")
    print("=" * 60)
    print(f"File:    {pcap_path}")
    print(f"Model:   {config.model.model_type} / {config.model.model_version}")
    print(f"Verbose: {verbose}")
    if args.max_packets:
        print(f"Limit:   {args.max_packets:,} packets")
    print("=" * 60)
    
    model_dir = config.model.get_model_dir(config.paths)
    
    engine = SnifferEngine(
        model_dir=str(model_dir),
        artifacts_dir=str(config.paths.artifacts_dir),
        log_dir=str(config.paths.logs_dir),
        firewall_enabled=False,
        confidence_threshold=config.model.confidence_threshold
    )
    
    try:
        results = engine.analyze_pcap(
            str(pcap_path), 
            max_packets=args.max_packets,
            verbose=verbose
        )
        
        attacks = [r for r in results if r.label != 'BENIGN']
        print(f"\nResults: {len(results)} flows analyzed")
        print(f"Attacks detected: {len(attacks)}")
        
        if attacks and verbose:
            print("\nAttack details (top 20):")
            for attack in attacks[:20]:
                print(f"  - {attack.label} | confidence: {attack.confidence:.2%}")
            if len(attacks) > 20:
                print(f"  ... and {len(attacks) - 20} more")
        
        if args.output:
            output_data = {
                'pcap_file': str(pcap_path),
                'model': f"{config.model.model_type}/{config.model.model_version}",
                'stats': engine.get_stats(),
                'attacks_count': len(attacks),
                'attacks': [r.to_dict() for r in attacks[:100]]  # Limit to first 100
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    finally:
        engine.close()


# ==============================================================================
# COMMAND: LIVE
# ==============================================================================

def cmd_live(args):
    """Start live packet capture."""
    from src.sniffer.engine import SnifferEngine
    
    config = create_config_from_args(args)
    
    print("=" * 60)
    print("NIDS-ML Live Capture")
    print("=" * 60)
    print(f"Interface: {config.sniffer.interface}")
    print(f"Duration:  {args.duration or 'indefinite'} seconds")
    print(f"Model:     {config.model.model_type} / {config.model.model_version}")
    print(f"Firewall:  {'ENABLED' if config.sniffer.firewall_enabled else 'disabled'}")
    print("=" * 60)
    
    model_dir = config.model.get_model_dir(config.paths)
    
    engine = SnifferEngine(
        model_dir=str(model_dir),
        artifacts_dir=str(config.paths.artifacts_dir),
        log_dir=str(config.paths.logs_dir),
        firewall_enabled=config.sniffer.firewall_enabled,
        firewall_dry_run=config.sniffer.firewall_dry_run,
        flow_timeout=config.sniffer.flow_timeout,
        confidence_threshold=config.model.confidence_threshold
    )
    
    try:
        engine.start_live(
            interface=config.sniffer.interface,
            duration=args.duration,
            filter_str=config.sniffer.bpf_filter,
            promisc=config.sniffer.promiscuous
        )
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    finally:
        if args.output:
            stats = engine.get_stats()
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to: {args.output}")
        engine.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NIDS-ML Sniffer - Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config --show --list-models
  %(prog)s validate --day tuesday --sample 5000
  %(prog)s evaluate --all-days --model-type xgboost --sample 10000
  %(prog)s pcap --day wednesday --model-type lightgbm
  %(prog)s live --interface eth0 --duration 300
        """
    )
    
    # Global arguments
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('-c', '--config-file', 
                       help='Configuration file (YAML or JSON)')
    
    # Model selection (global)
    model_group = parser.add_argument_group('Model Selection')
    model_group.add_argument('--model-type', 
                            choices=['xgboost', 'lightgbm', 'random_forest'],
                            help='Model type')
    model_group.add_argument('--model-version', 
                            help='Model version (folder name, or "best")')
    model_group.add_argument('--task-type',
                            choices=['binary', 'multiclass'],
                            help='Task type')
    model_group.add_argument('--confidence', type=float,
                            help='Confidence threshold (0-1)')
    
    # Path overrides
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument('--model-dir', help='Models directory')
    path_group.add_argument('--artifacts-dir', help='Artifacts directory')
    path_group.add_argument('--log-dir', help='Log directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # CONFIG command
    config_parser = subparsers.add_parser('config', help='Show/manage configuration')
    config_parser.add_argument('--show', action='store_true', 
                              help='Show current configuration')
    config_parser.add_argument('--list-models', action='store_true',
                              help='List available models')
    config_parser.add_argument('--list-data', action='store_true',
                              help='List available CIC-IDS2017 data files')
    config_parser.add_argument('--save', help='Save configuration to file')
    
    # VALIDATE command
    valid_parser = subparsers.add_parser('validate', help='Validate features against CSV')
    valid_parser.add_argument('--csv', help='CSV file path')
    valid_parser.add_argument('--day', help='CIC-IDS2017 day name')
    valid_parser.add_argument('--sample', type=int, help='Sample size')
    valid_parser.add_argument('-o', '--output', help='Output JSON file path')
    
    # CALIBRATE command
    calib_parser = subparsers.add_parser('calibrate', help='Feature calibration')
    calib_parser.add_argument('--csv', help='CSV file path')
    calib_parser.add_argument('--day', help='CIC-IDS2017 day name')
    calib_parser.add_argument('--sample', type=int, default=10000, help='Sample size')
    calib_parser.add_argument('--importance', action='store_true',
                             help='Show feature importance analysis')
    calib_parser.add_argument('-o', '--output', help='Output JSON file path')
    
    # EVALUATE command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on CSV')
    eval_parser.add_argument('--csv', help='CSV file path')
    eval_parser.add_argument('--day', help='CIC-IDS2017 day name')
    eval_parser.add_argument('--all-days', action='store_true',
                            help='Evaluate on all CIC-IDS2017 days')
    eval_parser.add_argument('--sample', type=int, help='Sample size')
    eval_parser.add_argument('--batch-size', type=int, default=10000,
                            help='Batch size for processing')
    eval_parser.add_argument('-o', '--output', help='Output JSON file path')
    
    # BENCHMARK command
    bench_parser = subparsers.add_parser('benchmark', help='Latency benchmark')
    bench_parser.add_argument('--samples', type=int, default=1000,
                             help='Samples per iteration')
    bench_parser.add_argument('--iterations', type=int, default=10,
                             help='Number of iterations')
    bench_parser.add_argument('--warmup', type=int, default=3,
                             help='Warmup iterations')
    bench_parser.add_argument('-o', '--output', help='Output JSON file path')
    
    # PCAP command
    pcap_parser = subparsers.add_parser('pcap', help='Analyze PCAP file')
    pcap_parser.add_argument('--file', help='PCAP file path')
    pcap_parser.add_argument('--day', help='CIC-IDS2017 day name')
    pcap_parser.add_argument('--max-packets', type=int,
                            help='Maximum packets to process')
    pcap_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Verbose output')
    pcap_parser.add_argument('-o', '--output', help='Output JSON file path')
    
    # LIVE command
    live_parser = subparsers.add_parser('live', help='Live packet capture')
    live_parser.add_argument('-i', '--interface', help='Network interface')
    live_parser.add_argument('-d', '--duration', type=int,
                            help='Capture duration (seconds)')
    live_parser.add_argument('-f', '--filter', help='BPF filter')
    live_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Verbose output')
    live_parser.add_argument('--firewall', action='store_true',
                            help='Enable firewall blocking')
    live_parser.add_argument('--firewall-execute', action='store_true',
                            help='Actually execute firewall rules')
    live_parser.add_argument('--no-promisc', action='store_true',
                            help='Disable promiscuous mode')
    live_parser.add_argument('-o', '--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load config file if specified
    if args.config_file:
        load_config(args.config_file)
    
    setup_logging(args.verbose)
    
    # Dispatch command
    commands = {
        'config': cmd_config,
        'validate': cmd_validate,
        'calibrate': cmd_calibrate,
        'evaluate': cmd_evaluate,
        'benchmark': cmd_benchmark,
        'pcap': cmd_pcap,
        'live': cmd_live
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(0)
    except Exception as e:
        if args.verbose:
            raise
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()