#!/usr/bin/env python3
"""
NIDS-ML Sniffer - Main Entry Point (Corrected v2)

CLI unificato per tutte le operazioni NIDS.

CORREZIONI v2:
- Flag -v/--verbose propagato correttamente a tutti i subcommand
- Gestione corretta nomi colonne CSV CIC-IDS2017
- Default sample_size=None (processa TUTTO)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configura logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )


CICIDS2017_FILES = {
    'monday': {
        'csv': 'Monday-WorkingHours.pcap_ISCX.csv',
        'pcap': 'Monday-WorkingHours.pcap',
        'attacks': []
    },
    'tuesday': {
        'csv': 'Tuesday-WorkingHours.pcap_ISCX.csv',
        'pcap': 'Tuesday-WorkingHours.pcap',
        'attacks': ['FTP-Patator', 'SSH-Patator']
    },
    'wednesday': {
        'csv': 'Wednesday-workingHours.pcap_ISCX.csv',
        'pcap': 'Wednesday-workingHours.pcap',
        'attacks': ['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed']
    },
    'thursday_morning': {
        'csv': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'pcap': 'Thursday-WorkingHours.pcap',
        'attacks': ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection']
    },
    'thursday_afternoon': {
        'csv': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'pcap': 'Thursday-WorkingHours.pcap',
        'attacks': ['Infiltration']
    },
    'friday_morning': {
        'csv': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'pcap': 'Friday-WorkingHours.pcap',
        'attacks': ['Bot']
    },
    'friday_portscan': {
        'csv': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'pcap': 'Friday-WorkingHours.pcap',
        'attacks': ['PortScan']
    },
    'friday_ddos': {
        'csv': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'pcap': 'Friday-WorkingHours.pcap',
        'attacks': ['DDoS']
    }
}


def get_csv_path(day: str, base_dir: str = 'data/raw') -> Path:
    """Ottiene path CSV per giorno CIC-IDS2017."""
    day_lower = day.lower().replace('-', '_').replace(' ', '_')
    if day_lower not in CICIDS2017_FILES:
        raise ValueError(f"Giorno non valido: {day}. Validi: {list(CICIDS2017_FILES.keys())}")
    return Path(base_dir) / CICIDS2017_FILES[day_lower]['csv']


def get_pcap_path(day: str, base_dir: str = 'data/pcap') -> Path:
    """Ottiene path PCAP per giorno CIC-IDS2017."""
    day_lower = day.lower().replace('-', '_').replace(' ', '_')
    if day_lower not in CICIDS2017_FILES:
        raise ValueError(f"Giorno non valido: {day}. Validi: {list(CICIDS2017_FILES.keys())}")
    return Path(base_dir) / CICIDS2017_FILES[day_lower]['pcap']


def cmd_config(args):
    """Mostra configurazione corrente."""
    print("=" * 60)
    print("NIDS-ML Configuration")
    print("=" * 60)
    
    model_dir = Path(args.model_dir)
    artifacts_dir = Path(args.artifacts_dir)
    
    print(f"\nModel directory:     {model_dir}")
    print(f"Artifacts directory: {artifacts_dir}")
    
    model_path = model_dir / 'model_binary.pkl'
    if not model_path.exists():
        model_path = model_dir / 'model.pkl'
    print(f"Model file:          {'EXISTS' if model_path.exists() else 'NOT FOUND'}")
    
    artifacts_files = [
        'scaler.pkl', 'scaler_columns.json', 
        'selected_features.json', 'label_encoder.pkl'
    ]
    print("\nArtifacts:")
    for f in artifacts_files:
        path = artifacts_dir / f
        status = 'OK' if path.exists() else 'MISSING'
        print(f"  {f}: {status}")
    
    if args.list_models:
        models_base = Path('models')
        print("\nAvailable models:")
        if models_base.exists():
            for model_type in ['xgboost', 'lightgbm', 'random_forest']:
                type_dir = models_base / model_type
                if type_dir.exists():
                    versions = [d.name for d in type_dir.iterdir() if d.is_dir()]
                    if versions:
                        print(f"  {model_type}:")
                        for v in versions:
                            print(f"    - {v}")
    
    if args.list_data:
        print("\nCIC-IDS2017 Dataset Files:")
        csv_dir = Path('data/raw')
        pcap_dir = Path('data/pcap')
        
        print("\n  CSV Files:")
        for day, info in CICIDS2017_FILES.items():
            csv_path = csv_dir / info['csv']
            status = 'OK' if csv_path.exists() else 'MISSING'
            attacks = ', '.join(info['attacks']) if info['attacks'] else 'Benign only'
            print(f"    [{status}] {day}: {info['csv']}")
            print(f"          Attacks: {attacks}")
        
        print("\n  PCAP Files:")
        seen_pcaps = set()
        for info in CICIDS2017_FILES.values():
            if info['pcap'] not in seen_pcaps:
                pcap_path = pcap_dir / info['pcap']
                status = 'OK' if pcap_path.exists() else 'MISSING'
                print(f"    [{status}] {info['pcap']}")
                seen_pcaps.add(info['pcap'])


def cmd_evaluate(args):
    """Valuta modello su CSV."""
    from src.sniffer.evaluation import SnifferEvaluator
    
    if args.csv:
        csv_path = Path(args.csv)
    elif args.day:
        csv_path = get_csv_path(args.day)
    else:
        print("Errore: specificare --csv o --day")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"Errore: CSV non trovato: {csv_path}")
        sys.exit(1)
    
    verbose = getattr(args, 'verbose', False)
    
    print("=" * 60)
    print("NIDS-ML Model Evaluation")
    print("=" * 60)
    print(f"CSV:         {csv_path}")
    print(f"Model:       {args.model_dir}")
    print(f"Sample size: {args.sample if args.sample else 'FULL DATASET'}")
    print("=" * 60)
    
    evaluator = SnifferEvaluator(
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir
    )
    
    result = evaluator.evaluate_csv(
        str(csv_path),
        sample_size=args.sample,
        batch_size=args.batch_size,
        verbose=True
    )
    
    result.print_summary()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return result


def cmd_evaluate_all(args):
    """Valuta modello su tutti i CSV CIC-IDS2017."""
    from src.sniffer.evaluation import SnifferEvaluator
    
    print("=" * 60)
    print("NIDS-ML Full Dataset Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Sample size per file: {args.sample if args.sample else 'FULL'}")
    print("=" * 60)
    
    evaluator = SnifferEvaluator(
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir
    )
    
    all_results = {}
    csv_dir = Path('data/raw')
    
    for day, info in CICIDS2017_FILES.items():
        csv_path = csv_dir / info['csv']
        
        if not csv_path.exists():
            print(f"\n[SKIP] {day}: file non trovato")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {day}")
        print(f"Attacks: {', '.join(info['attacks']) if info['attacks'] else 'Benign only'}")
        print(f"{'=' * 60}")
        
        try:
            result = evaluator.evaluate_csv(
                str(csv_path),
                sample_size=args.sample,
                batch_size=args.batch_size,
                verbose=True
            )
            
            all_results[day] = result.to_dict()
            
            # print(f"\nSummary: F1={result.f1_score:.4f} | FPR={result.false_positive_rate:.4f} | "
            #       f"Recall={result.recall:.4f} | Features: {result.features_matched}/{result.features_matched + result.features_missing}")

            print(f"\nSummary: F1={result.f1_score:.4f} | FPR={result.false_positive_rate:.4f} | "
                f"Recall={result.recall:.4f} | Features: {result.features_matched}")
            
        except Exception as e:
            print(f"Errore: {e}")
            all_results[day] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Day':<25} {'F1':>8} {'FPR':>8} {'Recall':>8} {'Samples':>12}")
    print("-" * 60)
    
    for day, res in all_results.items():
        if 'error' in res:
            print(f"{day:<25} {'ERROR':>8}")
        else:
            print(f"{day:<25} {res['f1_score']:>8.4f} {res['false_positive_rate']:>8.4f} "
                  f"{res['recall']:>8.4f} {res['total_samples']:>12,}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def cmd_pcap(args):
    """Analizza file PCAP."""
    from src.sniffer.engine import SnifferEngine
    
    if args.file:
        pcap_path = Path(args.file)
    elif args.day:
        pcap_path = get_pcap_path(args.day)
    else:
        print("Errore: specificare --file o --day")
        sys.exit(1)
    
    if not pcap_path.exists():
        print(f"Errore: PCAP non trovato: {pcap_path}")
        sys.exit(1)
    
    verbose = getattr(args, 'verbose', False)
    
    print("=" * 60)
    print("NIDS-ML PCAP Analysis")
    print("=" * 60)
    print(f"File:        {pcap_path}")
    print(f"Model:       {args.model_dir}")
    print(f"Max packets: {args.max_packets if args.max_packets else 'ALL'}")
    print(f"Verbose:     {verbose}")
    print("=" * 60)
    
    engine = SnifferEngine(
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir,
        firewall_enabled=False,
        confidence_threshold=args.confidence
    )
    
    try:
        results = engine.analyze_pcap(
            str(pcap_path),
            max_packets=args.max_packets,
            verbose=verbose
        )
        
        print(f"\nAttacks detected: {len(results)}")
        
        if results and verbose:
            print("\nAttack details (top 20):")
            for r in results[:20]:
                print(f"  - {r.label} | {r.src_ip}:{r.src_port} -> {r.dst_ip}:{r.dst_port} | "
                      f"conf: {r.confidence:.1%} | pkts: {r.packets}")
            if len(results) > 20:
                print(f"  ... and {len(results) - 20} more")
        
        if args.output:
            output_data = {
                'pcap_file': str(pcap_path),
                'stats': engine.get_stats(),
                'attacks_count': len(results),
                'attacks': [r.to_dict() for r in results[:1000]]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    finally:
        engine.close()


def cmd_live(args):
    """Cattura live."""
    from src.sniffer.engine import SnifferEngine
    
    verbose = getattr(args, 'verbose', False)
    
    print("=" * 60)
    print("NIDS-ML Live Capture")
    print("=" * 60)
    print(f"Interface: {args.interface}")
    print(f"Duration:  {args.duration if args.duration else 'indefinite'} seconds")
    print(f"Model:     {args.model_dir}")
    print(f"Firewall:  {'ENABLED' if args.firewall else 'disabled'}")
    print("=" * 60)
    
    engine = SnifferEngine(
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir,
        firewall_enabled=args.firewall,
        firewall_dry_run=not args.firewall_execute,
        confidence_threshold=args.confidence
    )
    
    try:
        engine.start_live(
            interface=args.interface,
            duration=args.duration,
            filter_str=args.filter or 'ip',
            promisc=not args.no_promisc,
            verbose=verbose
        )
    except KeyboardInterrupt:
        print("\nCapture interrupted")
    finally:
        if args.output:
            stats = engine.get_stats()
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to: {args.output}")
        engine.close()


def cmd_benchmark(args):
    """Benchmark latenza."""
    from src.sniffer.evaluation import LatencyBenchmarker
    
    print("=" * 60)
    print("NIDS-ML Latency Benchmark")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print("=" * 60)
    
    benchmarker = LatencyBenchmarker(
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir
    )
    
    results = benchmarker.benchmark(
        n_samples=args.samples,
        n_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    benchmarker.print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='NIDS-ML Sniffer - Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on single CSV (full dataset)
  %(prog)s evaluate --csv data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
  
  # Evaluate on all CIC-IDS2017 days
  %(prog)s evaluate-all
  
  # Evaluate with sampling
  %(prog)s evaluate --day tuesday --sample 50000
  
  # Analyze PCAP (all packets) with verbose
  %(prog)s pcap --file data/pcap/Tuesday-WorkingHours.pcap --verbose
  
  # Live capture
  sudo %(prog)s live --interface eth0 --duration 300
  
  # Benchmark latency
  %(prog)s benchmark --samples 1000 --iterations 10

Available days: monday, tuesday, wednesday, thursday_morning, thursday_afternoon,
                friday_morning, friday_portscan, friday_ddos
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--model-dir', default='models/best_model', help='Model directory')
    parser.add_argument('--artifacts-dir', default='artifacts', help='Artifacts directory')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--list-models', action='store_true', help='List available models')
    config_parser.add_argument('--list-data', action='store_true', help='List dataset files')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on single CSV')
    eval_parser.add_argument('--csv', help='CSV file path')
    eval_parser.add_argument('--day', help='CIC-IDS2017 day name')
    eval_parser.add_argument('--sample', type=int, default=None,
                            help='Sample size (default: FULL dataset)')
    eval_parser.add_argument('--batch-size', type=int, default=50000, help='Batch size')
    eval_parser.add_argument('-o', '--output', help='Output JSON path')
    eval_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    eval_all_parser = subparsers.add_parser('evaluate-all', help='Evaluate on all CIC-IDS2017 days')
    eval_all_parser.add_argument('--sample', type=int, default=None,
                                help='Sample size per file (default: FULL)')
    eval_all_parser.add_argument('--batch-size', type=int, default=50000, help='Batch size')
    eval_all_parser.add_argument('-o', '--output', help='Output JSON path')
    eval_all_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    pcap_parser = subparsers.add_parser('pcap', help='Analyze PCAP file')
    pcap_parser.add_argument('--file', help='PCAP file path')
    pcap_parser.add_argument('--day', help='CIC-IDS2017 day name')
    pcap_parser.add_argument('--max-packets', type=int, default=None,
                            help='Max packets (default: ALL)')
    pcap_parser.add_argument('-o', '--output', help='Output JSON path')
    pcap_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    live_parser = subparsers.add_parser('live', help='Live capture')
    live_parser.add_argument('-i', '--interface', default='eth0', help='Network interface')
    live_parser.add_argument('-d', '--duration', type=int, help='Duration (seconds)')
    live_parser.add_argument('-f', '--filter', help='BPF filter')
    live_parser.add_argument('--firewall', action='store_true', help='Enable firewall blocking')
    live_parser.add_argument('--firewall-execute', action='store_true',
                            help='Actually execute firewall rules')
    live_parser.add_argument('--no-promisc', action='store_true', help='Disable promiscuous mode')
    live_parser.add_argument('-o', '--output', help='Output JSON path')
    live_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    bench_parser = subparsers.add_parser('benchmark', help='Latency benchmark')
    bench_parser.add_argument('--samples', type=int, default=1000, help='Samples per iteration')
    bench_parser.add_argument('--iterations', type=int, default=10, help='Iterations')
    bench_parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    bench_parser.add_argument('-o', '--output', help='Output JSON path')
    bench_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    commands = {
        'config': cmd_config,
        'evaluate': cmd_evaluate,
        'evaluate-all': cmd_evaluate_all,
        'pcap': cmd_pcap,
        'live': cmd_live,
        'benchmark': cmd_benchmark
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