"""
================================================================================
NIDS-ML - Sniff Evaluation & Comparison System
================================================================================

Sistema completo per:
1. Eseguire sniffate su PCAP/CSV con uno o tutti i modelli
2. Valutare efficacia confrontando con ground truth (CSV CIC-IDS2017)
3. Confrontare performance tra diversi modelli
4. Salvare risultati in modo strutturato

USO DA LINEA DI COMANDO:
------------------------
# Analizza PCAP con un modello specifico
python src/sniff_evaluation.py --pcap Friday.pcap --model-path models/xgboost/cv5_iter100/model_binary.pkl

# Analizza PCAP con TUTTI i modelli di un tipo
python src/sniff_evaluation.py --pcap Friday.pcap --model-type xgboost

# Analizza PCAP con TUTTI i modelli disponibili e confronta
python src/sniff_evaluation.py --pcap Friday.pcap --model-type all

# Analizza CSV (test diretto su dati etichettati)
python src/sniff_evaluation.py --csv Friday-WorkingHours.csv --model-type all

# Analizza CSV con sample ridotto (veloce)
python src/sniff_evaluation.py --csv Friday.csv --model-type all --sample 10000

# Mostra risultati salvati
python src/sniff_evaluation.py --list-results

STRUTTURA OUTPUT:
-----------------
    sniff_results/
    ├── pcap/
    │   ├── Friday-WorkingHours/
    │   │   ├── xgboost_cv5_iter100_gpu.json
    │   │   ├── lightgbm_cv5_iter50.json
    │   │   └── comparison.json
    │   └── Monday-WorkingHours/
    ├── csv/
    │   └── Friday-WorkingHours/
    │       ├── xgboost_cv5_iter100_gpu.json
    │       └── comparison.json
    └── reports/
        └── model_comparison_report.json

================================================================================
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import sys
import time

# Setup path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_project_root, get_logger
from src.model_versioning import list_model_versions

logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

SNIFF_RESULTS_DIR = "sniff_results"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_PACKETS = 2
DEFAULT_TIMEOUT = 120


# ==============================================================================
# DIRECTORY MANAGEMENT
# ==============================================================================

def get_results_dir(data_type: str = 'pcap', data_name: str = None) -> Path:
    """
    Ottiene directory per risultati.
    
    Args:
        data_type: 'pcap' o 'csv'
        data_name: Nome del file (senza estensione)
    """
    base_dir = get_project_root() / SNIFF_RESULTS_DIR / data_type
    if data_name:
        base_dir = base_dir / data_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_model_id(model_path: Path) -> str:
    """Estrae ID univoco del modello dal path."""
    parts = model_path.parts
    
    # Cerca pattern: models/xgboost/cv5_iter100_gpu/model_binary.pkl
    try:
        models_idx = None
        for i, p in enumerate(parts):
            if p == 'models':
                models_idx = i
                break
        
        if models_idx is not None:
            model_type = parts[models_idx + 1]
            
            if len(parts) > models_idx + 3:
                version_id = parts[models_idx + 2]
                return f"{model_type}_{version_id}"
            else:
                return f"{model_type}_default"
    except (ValueError, IndexError):
        pass
    
    return model_path.stem


# ==============================================================================
# ANALISI PCAP
# ==============================================================================

def analyze_pcap_with_model(pcap_path: Path, model_path: Path,
                            threshold: float = DEFAULT_THRESHOLD,
                            min_packets: int = DEFAULT_MIN_PACKETS,
                            timeout: int = DEFAULT_TIMEOUT,
                            verbose: bool = False) -> Dict:
    """
    Analizza PCAP con un modello specifico.
    
    Returns:
        Dict con risultati completi
    """
    from src.sniffer import analyze_pcap_file
    
    print(f"\n  Modello: {model_path.name}")
    print(f"  Threshold: {threshold}, Min packets: {min_packets}")
    
    start_time = time.time()
    
    result = analyze_pcap_file(
        pcap_path=str(pcap_path),
        model_path=str(model_path),
        threshold=threshold,
        timeout=timeout,
        min_packets=min_packets,
        verbose=verbose,
        progress_interval=100000,
        show_progress=True
    )
    
    elapsed = time.time() - start_time
    result['analysis_time_seconds'] = elapsed
    result['model_path'] = str(model_path)
    result['model_id'] = get_model_id(model_path)
    
    return result


# ==============================================================================
# ANALISI CSV (Test su dati etichettati)
# ==============================================================================

def analyze_csv_with_model(csv_path: Path, model_path: Path,
                           task: str = 'binary',
                           sample_size: int = None) -> Dict:
    """
    Analizza CSV etichettato con un modello.
    
    Questo permette di testare il modello direttamente sui dati
    del dataset CIC-IDS2017 e calcolare metriche reali.
    
    Args:
        csv_path: Path al CSV
        model_path: Path al modello
        task: 'binary' o 'multiclass'
        sample_size: Se specificato, usa solo N righe (per test veloci)
    
    Returns:
        Dict con metriche complete
    """
    import joblib
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix
    )
    from src.feature_engineering import load_artifacts
    
    print(f"\n  CSV: {csv_path.name}")
    print(f"  Modello: {model_path.name}")
    
    # Carica modello e artifacts
    model = joblib.load(model_path)
    scaler, selected_features, _, scaler_columns = load_artifacts()
    
    if scaler_columns is None:
        raise RuntimeError("scaler_columns.json mancante. Rieseguire feature_engineering.py")
    
    # Carica feature specifiche del modello se presenti
    model_features_path = model_path.parent / f"features_{task}.json"
    if model_features_path.exists():
        with open(model_features_path) as f:
            selected_features = json.load(f)
    
    # Carica CSV
    print(f"  Caricamento CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Sample: {sample_size:,} righe")
    
    # Trova colonna label
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Colonna label non trovata")
    
    # Prepara labels
    if task == 'binary':
        y_true = (df[label_col].str.upper() != 'BENIGN').astype(int)
    else:
        y_true = df[label_col]
    
    # Prepara features
    missing_cols = set(scaler_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    
    X = df[scaler_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scala e seleziona
    print(f"  Preprocessing...")
    X_scaled = pd.DataFrame(scaler.transform(X), columns=scaler_columns)
    X_selected = pd.DataFrame(
        X_scaled[selected_features].values,
        columns=list(selected_features)
    )
    
    # Predizione
    print(f"  Predizione su {len(X_selected):,} samples...")
    start_time = time.time()
    y_pred = model.predict(X_selected)
    pred_time = time.time() - start_time
    
    # Metriche
    if task == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    else:
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
    
    # Statistiche dataset
    total_attacks = int((y_true == 1).sum()) if task == 'binary' else 0
    total_benign = int((y_true == 0).sum()) if task == 'binary' else 0
    
    result = {
        'csv': csv_path.name,
        'model_path': str(model_path),
        'model_id': get_model_id(model_path),
        'task': task,
        'total_samples': len(df),
        'total_attacks_in_data': total_attacks,
        'total_benign_in_data': total_benign,
        'metrics': metrics,
        'prediction_time_seconds': pred_time,
        'samples_per_second': len(df) / pred_time if pred_time > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Stampa risultati
    print(f"\n  Risultati:")
    print(f"    Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"    Precision: {metrics.get('precision', metrics.get('precision_weighted', 0)):.4f}")
    print(f"    Recall:    {metrics.get('recall', metrics.get('recall_weighted', 0)):.4f}")
    print(f"    F1:        {metrics.get('f1', metrics.get('f1_weighted', 0)):.4f}")
    if 'false_positive_rate' in metrics:
        print(f"    FPR:       {metrics['false_positive_rate']:.4f}")
    
    return result


# ==============================================================================
# CONFRONTO MODELLI
# ==============================================================================

def compare_results(results: List[Dict], data_name: str) -> Dict:
    """Confronta risultati di diversi modelli sugli stessi dati."""
    
    comparison = {
        'data_name': data_name,
        'timestamp': datetime.now().isoformat(),
        'models_compared': len(results),
        'results': [],
        'ranking': []
    }
    
    for r in results:
        if 'error' in r:
            continue
            
        model_id = r.get('model_id', 'unknown')
        
        # Per risultati PCAP
        if 'attacks_detected' in r:
            comparison['results'].append({
                'model_id': model_id,
                'flows_analyzed': r.get('flows_analyzed', 0),
                'attacks_detected': r.get('attacks_detected', 0),
                'detection_rate': r.get('detection_rate', 0),
                'analysis_time': r.get('analysis_time_seconds', 0)
            })
        
        # Per risultati CSV
        elif 'metrics' in r:
            m = r['metrics']
            comparison['results'].append({
                'model_id': model_id,
                'f1': m.get('f1', m.get('f1_weighted', 0)),
                'recall': m.get('recall', m.get('recall_weighted', 0)),
                'precision': m.get('precision', m.get('precision_weighted', 0)),
                'fpr': m.get('false_positive_rate', 'N/A'),
                'prediction_time': r.get('prediction_time_seconds', 0)
            })
    
    # Ranking
    if comparison['results']:
        if 'f1' in comparison['results'][0]:
            # Ranking per CSV (by F1)
            sorted_results = sorted(
                comparison['results'],
                key=lambda x: x.get('f1', 0),
                reverse=True
            )
        else:
            # Ranking per PCAP (by detection rate)
            sorted_results = sorted(
                comparison['results'],
                key=lambda x: x.get('detection_rate', 0),
                reverse=True
            )
        
        comparison['ranking'] = [
            {'rank': i+1, 'model_id': r['model_id']}
            for i, r in enumerate(sorted_results)
        ]
    
    return comparison


def save_results(result: Dict, data_type: str, data_name: str, model_id: str) -> Path:
    """Salva risultato singolo."""
    output_dir = get_results_dir(data_type, data_name)
    output_file = output_dir / f"{model_id}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return output_file


def save_comparison(comparison: Dict, data_type: str, data_name: str) -> Path:
    """Salva confronto."""
    output_dir = get_results_dir(data_type, data_name)
    output_file = output_dir / "comparison.json"
    
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    return output_file


# ==============================================================================
# LISTA RISULTATI
# ==============================================================================

def list_saved_results():
    """Mostra tutti i risultati salvati."""
    base_dir = get_project_root() / SNIFF_RESULTS_DIR
    
    if not base_dir.exists():
        print("Nessun risultato salvato")
        return
    
    print("\n" + "=" * 60)
    print("RISULTATI SALVATI")
    print("=" * 60)
    
    for data_type in ['pcap', 'csv']:
        type_dir = base_dir / data_type
        if not type_dir.exists():
            continue
        
        print(f"\n{data_type.upper()}:")
        
        for data_dir in sorted(type_dir.iterdir()):
            if not data_dir.is_dir():
                continue
            
            results = list(data_dir.glob("*.json"))
            models = [r.stem for r in results if r.stem != 'comparison']
            
            print(f"  {data_dir.name}/")
            for m in sorted(models):
                print(f"    - {m}")
            
            if (data_dir / "comparison.json").exists():
                print(f"    [comparison.json]")


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def run_analysis(data_path: Path, data_type: str, model_paths: List[Path],
                 threshold: float = DEFAULT_THRESHOLD,
                 min_packets: int = DEFAULT_MIN_PACKETS,
                 timeout: int = DEFAULT_TIMEOUT,
                 task: str = 'binary',
                 sample_size: int = None,
                 verbose: bool = False) -> Dict:
    """
    Esegue analisi completa su PCAP o CSV con uno o piu' modelli.
    
    Returns:
        Dict con tutti i risultati e confronto
    """
    data_name = data_path.stem
    results = []
    
    print(f"\n{'=' * 60}")
    print(f"ANALISI: {data_path.name}")
    print(f"{'=' * 60}")
    print(f"Tipo:    {data_type.upper()}")
    print(f"Modelli: {len(model_paths)}")
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] {'-' * 40}")
        
        try:
            if data_type == 'pcap':
                result = analyze_pcap_with_model(
                    data_path, model_path,
                    threshold=threshold,
                    min_packets=min_packets,
                    timeout=timeout,
                    verbose=verbose
                )
            else:  # csv
                result = analyze_csv_with_model(
                    data_path, model_path,
                    task=task,
                    sample_size=sample_size
                )
            
            results.append(result)
            
            # Salva risultato singolo
            save_results(result, data_type, data_name, result['model_id'])
            
        except Exception as e:
            print(f"  ERRORE: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model_id': get_model_id(model_path),
                'model_path': str(model_path),
                'error': str(e)
            })
    
    # Confronto se piu' modelli
    if len([r for r in results if 'error' not in r]) > 1:
        comparison = compare_results(results, data_name)
        save_comparison(comparison, data_type, data_name)
        
        # Stampa ranking
        print(f"\n{'=' * 60}")
        print("RANKING MODELLI")
        print(f"{'=' * 60}")
        
        for item in comparison.get('ranking', []):
            print(f"  #{item['rank']}: {item['model_id']}")
    
    return {
        'data_path': str(data_path),
        'data_type': data_type,
        'results': results,
        'comparison': compare_results(results, data_name) if len(results) > 1 else None
    }


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='NIDS Sniff Evaluation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Analizza PCAP con un modello
  python src/sniff_evaluation.py --pcap Friday.pcap --model-path models/xgboost/model_binary.pkl
  
  # Analizza PCAP con TUTTI i modelli XGBoost
  python src/sniff_evaluation.py --pcap Friday.pcap --model-type xgboost
  
  # Analizza PCAP con TUTTI i modelli e confronta
  python src/sniff_evaluation.py --pcap Friday.pcap --model-type all
  
  # Analizza CSV (test diretto su dati etichettati)
  python src/sniff_evaluation.py --csv Friday.csv --model-type all
  
  # Analizza CSV con sample ridotto (veloce)
  python src/sniff_evaluation.py --csv Friday.csv --model-type all --sample 10000
  
  # Mostra risultati salvati
  python src/sniff_evaluation.py --list-results
        """
    )
    
    # Dati input
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--pcap', type=Path, help='File PCAP da analizzare')
    data_group.add_argument('--csv', type=Path, help='File CSV da analizzare')
    data_group.add_argument('--list-results', action='store_true', 
                            help='Mostra risultati salvati')
    
    # Modello
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model-path', type=Path, 
                             help='Path a singolo modello')
    model_group.add_argument('--model-type', type=str,
                             choices=['xgboost', 'lightgbm', 'random_forest', 'all'],
                             help='Usa tutti i modelli di questo tipo')
    
    # Opzioni analisi
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Soglia probabilita (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--min-packets', type=int, default=DEFAULT_MIN_PACKETS,
                        help=f'Min pacchetti per flusso (default: {DEFAULT_MIN_PACKETS})')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout flusso secondi (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                        default='binary', help='Tipo classificazione')
    parser.add_argument('--sample', type=int, default=None,
                        help='Numero righe da usare per CSV (per test veloci)')
    
    # Output
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Output dettagliato')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    # Lista risultati
    if args.list_results:
        list_saved_results()
        return
    
    # Verifica input
    if not args.pcap and not args.csv:
        print("Errore: specificare --pcap o --csv o --list-results")
        sys.exit(1)
    
    data_path = args.pcap or args.csv
    data_type = 'pcap' if args.pcap else 'csv'
    
    if not data_path.exists():
        print(f"Errore: file non trovato: {data_path}")
        sys.exit(1)
    
    # Determina modelli da usare
    model_paths = []
    
    if args.model_path:
        if not args.model_path.exists():
            print(f"Errore: modello non trovato: {args.model_path}")
            sys.exit(1)
        model_paths = [args.model_path]
    
    elif args.model_type:
        if args.model_type == 'all':
            types = ['xgboost', 'lightgbm', 'random_forest']
        else:
            types = [args.model_type]
        
        for mtype in types:
            versions = list_model_versions(model_type=mtype, task=args.task)
            for v in versions:
                model_paths.append(v['model_path'])
        
        if not model_paths:
            print(f"Nessun modello trovato per tipo: {args.model_type}")
            sys.exit(1)
    
    else:
        # Default: cerca best_model
        best_model = get_project_root() / "models" / "best_model" / f"model_{args.task}.pkl"
        if best_model.exists():
            model_paths = [best_model]
        else:
            print("Errore: specificare --model-path o --model-type")
            sys.exit(1)
    
    print(f"\nModelli da testare: {len(model_paths)}")
    for mp in model_paths:
        print(f"  - {get_model_id(mp)}")
    
    # Esegui analisi
    results = run_analysis(
        data_path=data_path,
        data_type=data_type,
        model_paths=model_paths,
        threshold=args.threshold,
        min_packets=args.min_packets,
        timeout=args.timeout,
        task=args.task,
        sample_size=args.sample,
        verbose=args.verbose
    )
    
    # Riepilogo finale
    print(f"\n{'=' * 60}")
    print("RIEPILOGO")
    print(f"{'=' * 60}")
    
    successful = [r for r in results['results'] if 'error' not in r]
    failed = [r for r in results['results'] if 'error' in r]
    
    print(f"Modelli testati: {len(successful)}/{len(results['results'])}")
    
    if data_type == 'pcap' and successful:
        print(f"\n{'Modello':<35} {'Flussi':>10} {'Attacchi':>10} {'Rate':>8}")
        print("-" * 70)
        for r in sorted(successful, key=lambda x: x.get('detection_rate', 0), reverse=True):
            print(f"{r['model_id']:<35} {r.get('flows_analyzed', 0):>10,} "
                  f"{r.get('attacks_detected', 0):>10,} {r.get('detection_rate', 0):>7.1f}%")
    
    elif data_type == 'csv' and successful:
        print(f"\n{'Modello':<35} {'F1':>8} {'Recall':>8} {'FPR':>8}")
        print("-" * 65)
        for r in sorted(successful, key=lambda x: x.get('metrics', {}).get('f1', 0), reverse=True):
            m = r.get('metrics', {})
            fpr = m.get('false_positive_rate', 'N/A')
            fpr_str = f"{fpr:.4f}" if isinstance(fpr, float) else fpr
            print(f"{r['model_id']:<35} {m.get('f1', 0):>8.4f} "
                  f"{m.get('recall', 0):>8.4f} {fpr_str:>8}")
    
    if failed:
        print(f"\nModelli con errori:")
        for r in failed:
            print(f"  - {r['model_id']}: {r['error']}")
    
    # Percorso risultati
    output_dir = get_results_dir(data_type, data_path.stem)
    print(f"\nRisultati salvati in: {output_dir}")


if __name__ == "__main__":
    main()