"""
================================================================================
NIDS-ML - Sniff Evaluation System
================================================================================

Sistema per testare e confrontare modelli su PCAP o CSV.

CONCETTI CHIAVE:
---------------
1. PCAP Analysis: Testa lo sniffer su traffico reale
   - Pro: Simula uso reale
   - Contro: Non sai se i risultati sono corretti (no ground truth)

2. CSV Analysis: Testa direttamente sui dati etichettati
   - Pro: Hai le label reali -> puoi calcolare F1, FPR, etc.
   - Contro: Non testa il parsing PCAP

RACCOMANDAZIONE:
---------------
Per validare che i modelli funzionino, usa CSV Analysis.
Per testare l'intero pipeline (parsing PCAP + predizione), usa PCAP Analysis.

USO:
----
# Test CSV - RACCOMANDATO per validare modelli
python src/sniff_evaluation.py --csv Friday.csv --model-type all

# Test PCAP
python src/sniff_evaluation.py --pcap Friday.pcap --model-type xgboost

# Mostra risultati salvati
python src/sniff_evaluation.py --list

STRUTTURA OUTPUT:
-----------------
sniff_results/
├── csv/
│   └── Friday-WorkingHours/
│       ├── xgboost/
│       │   ├── cv5_iter200_gpu.json
│       │   └── default.json
│       ├── lightgbm/
│       │   └── cv5_iter200.json
│       └── comparison.json
└── pcap/
    └── Friday-WorkingHours/
        ├── xgboost/
        │   └── cv5_iter200_gpu.json
        └── comparison.json

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

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_project_root, get_logger
from src.model_versioning import list_model_versions

logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

RESULTS_DIR = "sniff_results"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_PACKETS = 2


# ==============================================================================
# GESTIONE DIRECTORY
# ==============================================================================

def get_output_dir(data_type: str, data_name: str, model_type: str) -> Path:
    """
    Crea e restituisce directory output strutturata.
    
    Struttura: sniff_results/{csv|pcap}/{data_name}/{model_type}/
    """
    output_dir = get_project_root() / RESULTS_DIR / data_type / data_name / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_version_id(model_path: Path) -> str:
    """Estrae version_id dal path del modello."""
    parts = model_path.parts
    
    # Pattern: models/xgboost/cv5_iter200_gpu/model_binary.pkl
    try:
        for i, p in enumerate(parts):
            if p == 'models' and i + 2 < len(parts):
                # Se c'e' una sottocartella oltre al tipo
                if parts[i + 2] != model_path.name:
                    return parts[i + 2]
                else:
                    return 'default'
    except:
        pass
    
    return 'default'


def get_model_type(model_path: Path) -> str:
    """Estrae tipo modello dal path."""
    parts = model_path.parts
    
    for i, p in enumerate(parts):
        if p == 'models' and i + 1 < len(parts):
            return parts[i + 1]
    
    return 'unknown'


# ==============================================================================
# TEST SU CSV (RACCOMANDATO)
# ==============================================================================

def test_model_on_csv(csv_path: Path, model_path: Path, 
                      task: str = 'binary',
                      sample_size: int = None) -> Dict:
    """
    Testa modello su CSV etichettato.
    
    Questo e' il metodo MIGLIORE per validare un modello perche'
    hai le label reali e puoi calcolare metriche precise.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix
    )
    from src.feature_engineering import load_artifacts
    
    model_type = get_model_type(model_path)
    version_id = get_version_id(model_path)
    
    print(f"\n  [{model_type}/{version_id}]")
    
    # Carica modello
    model = joblib.load(model_path)
    scaler, selected_features, _, scaler_columns = load_artifacts()
    
    if scaler_columns is None:
        raise RuntimeError("scaler_columns.json mancante")
    
    # Feature specifiche del modello
    features_path = model_path.parent / f"features_{task}.json"
    if features_path.exists():
        with open(features_path) as f:
            selected_features = json.load(f)
    
    # Carica CSV
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Trova label
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if not label_col:
        raise ValueError("Colonna label non trovata")
    
    # Prepara y
    y_true = (df[label_col].str.upper() != 'BENIGN').astype(int)
    
    # Prepara X
    for col in scaler_columns:
        if col not in df.columns:
            df[col] = 0
    
    X = df[scaler_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Trasforma
    X_scaled = pd.DataFrame(scaler.transform(X), columns=scaler_columns)
    X_selected = pd.DataFrame(
        X_scaled[selected_features].values,
        columns=list(selected_features)
    )
    
    # Predici
    start = time.time()
    y_pred = model.predict(X_selected)
    pred_time = time.time() - start
    
    # Metriche
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    result = {
        'model_type': model_type,
        'version_id': version_id,
        'model_path': str(model_path),
        'csv': csv_path.name,
        'samples': len(df),
        'attacks_in_data': int(y_true.sum()),
        'benign_in_data': int((y_true == 0).sum()),
        'metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
        },
        'confusion': {
            'tp': int(tp), 'fp': int(fp), 
            'tn': int(tn), 'fn': int(fn)
        },
        'prediction_time_sec': pred_time,
        'samples_per_sec': len(df) / pred_time if pred_time > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    m = result['metrics']
    print(f"    F1={m['f1']:.4f} | Recall={m['recall']:.4f} | FPR={m['fpr']:.4f}")
    
    return result


# ==============================================================================
# TEST SU PCAP
# ==============================================================================

def test_model_on_pcap(pcap_path: Path, model_path: Path,
                       threshold: float = DEFAULT_THRESHOLD,
                       min_packets: int = DEFAULT_MIN_PACKETS,
                       verbose: bool = False) -> Dict:
    """
    Testa modello su file PCAP.
    
    NOTA: Senza ground truth non puoi sapere se i risultati sono corretti.
    Usa questo per testare che il pipeline funzioni, non per validare.
    """
    from src.sniffer import analyze_pcap_file
    
    model_type = get_model_type(model_path)
    version_id = get_version_id(model_path)
    
    print(f"\n  [{model_type}/{version_id}]")
    
    start = time.time()
    
    result = analyze_pcap_file(
        pcap_path=str(pcap_path),
        model_path=str(model_path),
        threshold=threshold,
        min_packets=min_packets,
        verbose=verbose,
        show_progress=True,
        progress_interval=200000
    )
    
    elapsed = time.time() - start
    
    # Aggiungi info modello
    result['model_type'] = model_type
    result['version_id'] = version_id
    result['model_path'] = str(model_path)
    result['analysis_time_sec'] = elapsed
    result['timestamp'] = datetime.now().isoformat()
    
    print(f"    Flussi={result.get('flows_analyzed', 0):,} | "
          f"Attacchi={result.get('attacks_detected', 0):,} | "
          f"Rate={result.get('detection_rate', 0):.1f}%")
    
    return result


# ==============================================================================
# SALVATAGGIO E CONFRONTO
# ==============================================================================

def save_result(result: Dict, data_type: str, data_name: str) -> Path:
    """Salva risultato singolo nella struttura corretta."""
    model_type = result['model_type']
    version_id = result['version_id']
    
    output_dir = get_output_dir(data_type, data_name, model_type)
    output_file = output_dir / f"{version_id}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return output_file


def create_comparison(results: List[Dict], data_type: str, data_name: str) -> Dict:
    """Crea e salva confronto tra risultati."""
    
    comparison = {
        'data_type': data_type,
        'data_name': data_name,
        'timestamp': datetime.now().isoformat(),
        'models_tested': len(results),
        'ranking': []
    }
    
    # Filtra errori
    valid = [r for r in results if 'error' not in r]
    
    if data_type == 'csv':
        # Ordina per F1
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
        # Ordina per detection rate
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


# ==============================================================================
# LISTA RISULTATI
# ==============================================================================

def list_results():
    """Mostra tutti i risultati salvati."""
    base = get_project_root() / RESULTS_DIR
    
    if not base.exists():
        print("Nessun risultato salvato")
        return
    
    print("\n" + "=" * 60)
    print("RISULTATI SALVATI")
    print("=" * 60)
    
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
                
                for r in sorted(results):
                    print(f"      - {r.stem}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test e confronto modelli NIDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Test CSV (RACCOMANDATO per validare)
  python src/sniff_evaluation.py --csv Friday.csv --model-type all
  
  # Test PCAP
  python src/sniff_evaluation.py --pcap Friday.pcap --model-type xgboost
  
  # Singolo modello
  python src/sniff_evaluation.py --csv Friday.csv --model-path models/xgboost/cv5_iter200_gpu/model_binary.pkl
  
  # Sample ridotto (veloce)
  python src/sniff_evaluation.py --csv Friday.csv --model-type all --sample 50000
  
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
    parser.add_argument('--threshold', type=float, default=0.5, help='Soglia (PCAP)')
    parser.add_argument('--task', default='binary', choices=['binary', 'multiclass'])
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Lista risultati
    if args.list:
        list_results()
        return
    
    # Verifica input
    if not args.csv and not args.pcap:
        print("Specificare --csv, --pcap, o --list")
        sys.exit(1)
    
    data_path = args.csv or args.pcap
    data_type = 'csv' if args.csv else 'pcap'
    data_name = data_path.stem
    
    if not data_path.exists():
        print(f"File non trovato: {data_path}")
        sys.exit(1)
    
    # Raccogli modelli
    model_paths = []
    
    if args.model_path:
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
    
    print(f"\n{'=' * 60}")
    print(f"SNIFF EVALUATION")
    print(f"{'=' * 60}")
    print(f"Input:   {data_path.name}")
    print(f"Tipo:    {data_type.upper()}")
    print(f"Modelli: {len(model_paths)}")
    
    # Esegui test
    results = []
    
    for mp in model_paths:
        try:
            if data_type == 'csv':
                r = test_model_on_csv(data_path, mp, args.task, args.sample)
            else:
                r = test_model_on_pcap(data_path, mp, args.threshold, verbose=args.verbose)
            
            results.append(r)
            save_result(r, data_type, data_name)
            
        except Exception as e:
            print(f"\n  ERRORE: {e}")
            results.append({
                'model_type': get_model_type(mp),
                'version_id': get_version_id(mp),
                'error': str(e)
            })
    
    # Confronto
    valid = [r for r in results if 'error' not in r]
    
    if len(valid) > 1:
        comparison = create_comparison(results, data_type, data_name)
        
        print(f"\n{'=' * 60}")
        print("RANKING")
        print(f"{'=' * 60}")
        
        for item in comparison['ranking'][:10]:
            if data_type == 'csv':
                print(f"  #{item['rank']:2} {item['model']:<35} F1={item['f1']:.4f}")
            else:
                print(f"  #{item['rank']:2} {item['model']:<35} Rate={item['rate']:.1f}%")
    
    # Riepilogo
    print(f"\n{'=' * 60}")
    print("RIEPILOGO")
    print(f"{'=' * 60}")
    print(f"Testati: {len(valid)}/{len(results)}")
    
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\nErrori:")
        for e in errors:
            print(f"  - {e['model_type']}/{e['version_id']}: {e['error']}")
    
    output_dir = get_project_root() / RESULTS_DIR / data_type / data_name
    print(f"\nRisultati: {output_dir}")


if __name__ == "__main__":
    main()