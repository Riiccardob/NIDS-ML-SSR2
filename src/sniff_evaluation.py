"""
================================================================================
NIDS-ML - Sniff Evaluation
================================================================================

Sistema per testare e confrontare modelli su CSV o PCAP.

QUANDO USARE COSA:
------------------
- --csv: Per VALIDARE che i modelli funzionino (hai le label reali)
- --pcap: Per TESTARE il pipeline completo (parsing + predizione)

I risultati CSV sono AFFIDABILI (metriche calcolate su label reali).
I risultati PCAP sono INDICATIVI (non hai ground truth).

USO:
----
# Test su CSV (RACCOMANDATO)
python src/sniff_evaluation.py --csv data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv --model-type all

# Test su singolo modello
python src/sniff_evaluation.py --csv Friday.csv --model-path models/best_model/model_binary.pkl

# Test veloce con sample
python src/sniff_evaluation.py --csv Friday.csv --model-type all --sample 50000

# Lista risultati salvati
python src/sniff_evaluation.py --list

================================================================================
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys
import time
import joblib

# Setup path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_project_root, get_logger
from src.model_versioning import list_model_versions
from src.feature_engineering import load_artifacts

logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

RESULTS_DIR = "sniff_results"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_PACKETS = 2


# ==============================================================================
# UTILITY
# ==============================================================================

def get_output_dir(data_type: str, data_name: str, model_type: str) -> Path:
    """Crea e restituisce directory output strutturata."""
    output_dir = get_project_root() / RESULTS_DIR / data_type / data_name / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_model_info(model_path: Path) -> tuple:
    """Estrae tipo e versione dal path del modello."""
    parts = model_path.parts
    model_type = 'unknown'
    version_id = 'default'
    
    for i, p in enumerate(parts):
        if p == 'models' and i + 1 < len(parts):
            model_type = parts[i + 1]
            if i + 2 < len(parts) and parts[i + 2] != model_path.name:
                version_id = parts[i + 2]
            break
    
    return model_type, version_id


# ==============================================================================
# TEST SU CSV
# ==============================================================================

def test_on_csv(csv_path: Path, model_path: Path,
                task: str = 'binary',
                sample_size: int = None,
                verbose: bool = False) -> Dict:
    """
    Testa modello su CSV etichettato.
    
    Questo e' il modo MIGLIORE per validare un modello perche'
    hai le label reali e puoi calcolare metriche precise.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix
    )
    
    model_type, version_id = get_model_info(model_path)
    print(f"\n  [{model_type}/{version_id}]")
    
    # Carica modello
    model = joblib.load(model_path)
    scaler, selected_features, _, scaler_columns = load_artifacts()
    
    if scaler_columns is None:
        raise RuntimeError("scaler_columns.json mancante. Esegui feature_engineering.py")
    
    # Feature specifiche del modello
    features_path = model_path.parent / f"features_{task}.json"
    if features_path.exists():
        with open(features_path) as f:
            selected_features = json.load(f)
    
    # Carica CSV
    if verbose:
        print(f"    Caricamento {csv_path.name}...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    original_size = len(df)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        if verbose:
            print(f"    Sample: {sample_size:,} / {original_size:,} righe")
    
    # Trova colonna label
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if not label_col:
        raise ValueError(f"Colonna label non trovata nel CSV. Colonne: {list(df.columns)[:10]}")
    
    # Prepara y
    if task == 'binary':
        y_true = (df[label_col].str.strip().str.upper() != 'BENIGN').astype(int)
    else:
        y_true = df[label_col]
    
    # Prepara X
    missing_cols = set(scaler_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    
    X = df[scaler_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Trasforma
    if verbose:
        print(f"    Preprocessing...")
    
    X_scaled = pd.DataFrame(scaler.transform(X), columns=scaler_columns)
    X_selected = pd.DataFrame(
        X_scaled[selected_features].values,
        columns=list(selected_features)
    )
    
    # Predici
    if verbose:
        print(f"    Predizione su {len(X_selected):,} samples...")
    
    start = time.time()
    y_pred = model.predict(X_selected)
    pred_time = time.time() - start
    
    # Metriche
    if task == 'binary':
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Solo una classe presente
            tn, fp, fn, tp = len(y_true), 0, 0, 0
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
        }
        confusion = {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
    else:
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        confusion = {}
    
    result = {
        'model_type': model_type,
        'version_id': version_id,
        'model_path': str(model_path),
        'csv': csv_path.name,
        'task': task,
        'total_samples': original_size,
        'samples_tested': len(df),
        'attacks_in_data': int((y_true == 1).sum()) if task == 'binary' else 0,
        'benign_in_data': int((y_true == 0).sum()) if task == 'binary' else 0,
        'metrics': metrics,
        'confusion': confusion,
        'prediction_time_sec': pred_time,
        'samples_per_sec': len(df) / pred_time if pred_time > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Output
    m = metrics
    if task == 'binary':
        print(f"    F1={m['f1']:.4f} | Recall={m['recall']:.4f} | FPR={m['fpr']:.4f} | Attacchi={result['attacks_in_data']:,}")
    else:
        print(f"    F1={m['f1_weighted']:.4f}")
    
    return result


# ==============================================================================
# TEST SU PCAP
# ==============================================================================

def test_on_pcap(pcap_path: Path, model_path: Path,
                 threshold: float = DEFAULT_THRESHOLD,
                 min_packets: int = DEFAULT_MIN_PACKETS,
                 timeout: int = 60,
                 verbose: bool = False) -> Dict:
    """
    Testa modello su file PCAP.
    
    NOTA: Senza ground truth non puoi sapere se i risultati sono corretti.
    Usa questo per testare che il pipeline funzioni.
    """
    # Import qui per evitare dipendenza Scapy se non necessaria
    from src.sniffer import SnifferEngine
    
    model_type, version_id = get_model_info(model_path)
    print(f"\n  [{model_type}/{version_id}]")
    
    start = time.time()
    
    engine = SnifferEngine(
        model_path=model_path,
        pcap_file=pcap_path,
        threshold=threshold,
        min_packets=min_packets,
        timeout=timeout,
        verbose=verbose,
        quiet=True
    )
    
    summary = engine.start_pcap()
    elapsed = time.time() - start
    
    result = {
        'model_type': model_type,
        'version_id': version_id,
        'model_path': str(model_path),
        'pcap': pcap_path.name,
        'threshold': threshold,
        'min_packets': min_packets,
        'packets_processed': summary['packets_processed'],
        'flows_analyzed': summary['flows_analyzed'],
        'attacks_detected': summary['attacks_detected'],
        'detection_rate': summary['detection_rate'],
        'analysis_time_sec': elapsed,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"    Flussi={result['flows_analyzed']:,} | Attacchi={result['attacks_detected']:,} | Rate={result['detection_rate']:.1f}%")
    
    return result


# ==============================================================================
# SALVATAGGIO E CONFRONTO
# ==============================================================================

def save_result(result: Dict, data_type: str, data_name: str) -> Path:
    """Salva risultato."""
    output_dir = get_output_dir(data_type, data_name, result['model_type'])
    output_file = output_dir / f"{result['version_id']}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return output_file


def create_comparison(results: List[Dict], data_type: str, data_name: str) -> Dict:
    """Crea e salva confronto tra risultati."""
    valid = [r for r in results if 'error' not in r]
    
    comparison = {
        'data_type': data_type,
        'data_name': data_name,
        'timestamp': datetime.now().isoformat(),
        'models_tested': len(results),
        'models_successful': len(valid),
        'ranking': []
    }
    
    if data_type == 'csv':
        valid.sort(key=lambda x: x.get('metrics', {}).get('f1', 0), reverse=True)
        for i, r in enumerate(valid, 1):
            comparison['ranking'].append({
                'rank': i,
                'model': f"{r['model_type']}/{r['version_id']}",
                'f1': r['metrics']['f1'],
                'recall': r['metrics']['recall'],
                'fpr': r['metrics']['fpr']
            })
    else:
        valid.sort(key=lambda x: x.get('detection_rate', 0), reverse=True)
        for i, r in enumerate(valid, 1):
            comparison['ranking'].append({
                'rank': i,
                'model': f"{r['model_type']}/{r['version_id']}",
                'flows': r.get('flows_analyzed', 0),
                'attacks': r.get('attacks_detected', 0),
                'rate': r.get('detection_rate', 0)
            })
    
    # Salva
    output_dir = get_project_root() / RESULTS_DIR / data_type / data_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def list_results():
    """Mostra risultati salvati."""
    base = get_project_root() / RESULTS_DIR
    
    if not base.exists():
        print("Nessun risultato salvato")
        return
    
    print(f"\n{'='*60}")
    print("RISULTATI SALVATI")
    print(f"{'='*60}")
    
    for dtype in ['csv', 'pcap']:
        dtype_dir = base / dtype
        if not dtype_dir.exists():
            continue
        
        print(f"\n{dtype.upper()}:")
        
        for data_dir in sorted(dtype_dir.iterdir()):
            if not data_dir.is_dir():
                continue
            
            print(f"\n  {data_dir.name}/")
            
            for model_dir in sorted(data_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                
                results = list(model_dir.glob("*.json"))
                print(f"    {model_dir.name}/ ({len(results)} risultati)")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NIDS Sniff Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Test su CSV (RACCOMANDATO)
  python src/sniff_evaluation.py --csv data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv --model-type all
  
  # Test singolo modello
  python src/sniff_evaluation.py --csv Friday.csv --model-path models/best_model/model_binary.pkl
  
  # Test veloce
  python src/sniff_evaluation.py --csv Friday.csv --model-type xgboost --sample 50000
  
  # Lista risultati
  python src/sniff_evaluation.py --list
        """
    )
    
    # Input
    input_grp = parser.add_mutually_exclusive_group()
    input_grp.add_argument('--csv', type=Path, help='CSV da testare')
    input_grp.add_argument('--pcap', type=Path, help='PCAP da testare')
    input_grp.add_argument('--list', action='store_true', help='Mostra risultati')
    
    # Modello
    model_grp = parser.add_mutually_exclusive_group()
    model_grp.add_argument('--model-path', type=Path, help='Singolo modello')
    model_grp.add_argument('--model-type',
                           choices=['xgboost', 'lightgbm', 'random_forest', 'all'],
                           help='Tutti i modelli di un tipo')
    
    # Opzioni
    parser.add_argument('--sample', type=int, help='Righe da usare (CSV)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-packets', type=int, default=2)
    parser.add_argument('--task', default='binary', choices=['binary', 'multiclass'])
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Lista
    if args.list:
        list_results()
        return
    
    # Verifica input
    if not args.csv and not args.pcap:
        print("Specificare --csv, --pcap, o --list")
        parser.print_help()
        sys.exit(1)
    
    data_path = args.csv or args.pcap
    data_type = 'csv' if args.csv else 'pcap'
    
    if not data_path.exists():
        print(f"File non trovato: {data_path}")
        sys.exit(1)
    
    # Raccogli modelli
    model_paths = []
    
    if args.model_path:
        if not args.model_path.exists():
            print(f"Modello non trovato: {args.model_path}")
            sys.exit(1)
        model_paths = [args.model_path]
    
    elif args.model_type:
        types = ['xgboost', 'lightgbm', 'random_forest'] if args.model_type == 'all' else [args.model_type]
        for t in types:
            versions = list_model_versions(model_type=t, task=args.task)
            model_paths.extend([v['model_path'] for v in versions])
    
    else:
        # Default: best_model
        best = get_project_root() / "models" / "best_model" / f"model_{args.task}.pkl"
        if best.exists():
            model_paths = [best]
        else:
            print("Specificare --model-path o --model-type")
            sys.exit(1)
    
    if not model_paths:
        print("Nessun modello trovato")
        sys.exit(1)
    
    # Header
    print(f"\n{'='*60}")
    print("SNIFF EVALUATION")
    print(f"{'='*60}")
    print(f"Input:   {data_path.name}")
    print(f"Tipo:    {data_type.upper()}")
    print(f"Modelli: {len(model_paths)}")
    
    # Esegui test
    results = []
    data_name = data_path.stem
    
    for mp in model_paths:
        try:
            if data_type == 'csv':
                r = test_on_csv(data_path, mp, args.task, args.sample, args.verbose)
            else:
                r = test_on_pcap(data_path, mp, args.threshold, args.min_packets, verbose=args.verbose)
            
            results.append(r)
            save_result(r, data_type, data_name)
        
        except Exception as e:
            model_type, version_id = get_model_info(mp)
            print(f"\n  ERRORE [{model_type}/{version_id}]: {e}")
            results.append({
                'model_type': model_type,
                'version_id': version_id,
                'error': str(e)
            })
    
    # Confronto
    valid = [r for r in results if 'error' not in r]
    
    if len(valid) > 1:
        comparison = create_comparison(results, data_type, data_name)
        
        print(f"\n{'='*60}")
        print("RANKING")
        print(f"{'='*60}")
        
        for item in comparison['ranking'][:10]:
            if data_type == 'csv':
                print(f"  #{item['rank']:2} {item['model']:<40} F1={item['f1']:.4f}")
            else:
                print(f"  #{item['rank']:2} {item['model']:<40} Rate={item['rate']:.1f}%")
    
    # Riepilogo
    errors = [r for r in results if 'error' in r]
    
    print(f"\n{'='*60}")
    print("RIEPILOGO")
    print(f"{'='*60}")
    print(f"Testati con successo: {len(valid)}/{len(results)}")
    
    if errors:
        print(f"\nErrori:")
        for e in errors:
            print(f"  - {e['model_type']}/{e['version_id']}: {e['error']}")
    
    output_dir = get_project_root() / RESULTS_DIR / data_type / data_name
    print(f"\nRisultati salvati: {output_dir}")


if __name__ == "__main__":
    main()