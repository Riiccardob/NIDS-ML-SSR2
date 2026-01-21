"""
================================================================================
NIDS-ML - Sniff Evaluation & Comparison System
================================================================================

Sistema per:
1. Salvare risultati sniffate in modo strutturato
2. Valutare efficacia confrontando con ground truth (CSV CIC-IDS2017)
3. Confrontare performance tra diversi modelli

STRUTTURA OUTPUT:
-----------------
    sniff_results/
    ├── xgboost/
    │   ├── cv5_iter100_gpu/
    │   │   ├── Friday-WorkingHours.json      # Risultato singolo PCAP
    │   │   ├── Monday-WorkingHours.json
    │   │   └── summary.json                   # Aggregato tutti PCAP
    │   └── cv3_iter20_gpu/
    │       └── ...
    ├── lightgbm/
    │   └── ...
    ├── comparison_report.json                 # Confronto tutti modelli
    └── comparison_report.html                 # Report visuale

================================================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from utils import get_project_root, get_logger

logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

SNIFF_RESULTS_DIR = "sniff_results"


# ==============================================================================
# SALVATAGGIO STRUTTURATO
# ==============================================================================

def get_sniff_results_dir(model_type: str = None, version_id: str = None) -> Path:
    """
    Ottiene la directory per salvare risultati sniffate.
    
    Args:
        model_type: Tipo modello (xgboost, lightgbm, random_forest)
        version_id: ID versione (cv5_iter100_gpu, default, etc)
    
    Returns:
        Path alla directory
    """
    base_dir = get_project_root() / SNIFF_RESULTS_DIR
    
    if model_type:
        base_dir = base_dir / model_type
        if version_id:
            base_dir = base_dir / version_id
    
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_sniff_result(result: Dict, model_type: str, version_id: str, 
                      pcap_name: str = None) -> Path:
    """
    Salva risultato singola sniffata in modo strutturato.
    
    Args:
        result: Dizionario risultato da analyze_pcap_file o start_pcap
        model_type: Tipo modello
        version_id: ID versione
        pcap_name: Nome file PCAP (opzionale, estratto da result se presente)
    
    Returns:
        Path al file salvato
    """
    output_dir = get_sniff_results_dir(model_type, version_id)
    
    if pcap_name is None:
        pcap_name = result.get('pcap', 'unknown')
    
    # Rimuovi estensione per nome file
    base_name = Path(pcap_name).stem
    output_file = output_dir / f"{base_name}.json"
    
    # Aggiungi metadata
    result_with_meta = {
        'metadata': {
            'model_type': model_type,
            'version_id': version_id,
            'pcap_name': pcap_name,
            'timestamp': datetime.now().isoformat(),
        },
        'results': result
    }
    
    with open(output_file, 'w') as f:
        json.dump(result_with_meta, f, indent=2, default=str)
    
    logger.info(f"Risultato salvato: {output_file}")
    return output_file


def load_sniff_results(model_type: str, version_id: str) -> List[Dict]:
    """
    Carica tutti i risultati sniffate per un modello/versione.
    
    Returns:
        Lista di risultati
    """
    results_dir = get_sniff_results_dir(model_type, version_id)
    results = []
    
    for json_file in results_dir.glob("*.json"):
        if json_file.name == 'summary.json':
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            logger.warning(f"Errore caricamento {json_file}: {e}")
    
    return results


# ==============================================================================
# GROUND TRUTH EXTRACTION
# ==============================================================================

def extract_ground_truth_from_csv(csv_path: Path, 
                                   pcap_time_range: Tuple[float, float] = None
                                   ) -> Dict[str, Dict]:
    """
    Estrae ground truth dal CSV CIC-IDS2017.
    
    Il CSV contiene le label per ogni flusso. Estraiamo:
    - Tutti i flussi con la loro label
    - Statistiche attacchi/benigni
    
    Args:
        csv_path: Path al CSV
        pcap_time_range: Opzionale (start, end) timestamp per filtrare
    
    Returns:
        Dict con:
        - flows: {flow_id: {'label': str, 'is_attack': bool}}
        - stats: statistiche aggregate
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Normalizza nomi colonne (CIC-IDS2017 ha nomi inconsistenti)
    df.columns = df.columns.str.strip()
    
    # Trova colonna label
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Colonna label non trovata in {csv_path}")
    
    # Costruisci flow_id compatibile con sniffer
    # Formato: src_ip:src_port->dst_ip:dst_port
    flow_id_cols = []
    for possible_names in [
        ('Source IP', 'Src IP', 'Src_IP'),
        ('Source Port', 'Src Port', 'Src_Port'),
        ('Destination IP', 'Dst IP', 'Dst_IP', 'Destination_IP'),
        ('Destination Port', 'Dst Port', 'Dst_Port', 'Destination_Port')
    ]:
        found = None
        for name in possible_names:
            if name in df.columns:
                found = name
                break
        if found:
            flow_id_cols.append(found)
    
    if len(flow_id_cols) < 4:
        # Fallback: usa Flow ID se presente
        if 'Flow ID' in df.columns:
            df['_flow_id'] = df['Flow ID']
        else:
            raise ValueError("Impossibile costruire flow_id dal CSV")
    else:
        src_ip, src_port, dst_ip, dst_port = flow_id_cols
        df['_flow_id'] = (
            df[src_ip].astype(str) + ':' + df[src_port].astype(str) + 
            '->' + df[dst_ip].astype(str) + ':' + df[dst_port].astype(str)
        )
    
    # Costruisci dizionario ground truth
    flows = {}
    attack_types = defaultdict(int)
    
    for _, row in df.iterrows():
        flow_id = row['_flow_id']
        label = str(row[label_col]).strip()
        is_attack = label.upper() not in ('BENIGN', 'NORMAL', '')
        
        flows[flow_id] = {
            'label': label,
            'is_attack': is_attack
        }
        
        if is_attack:
            attack_types[label] += 1
    
    total_attacks = sum(1 for f in flows.values() if f['is_attack'])
    total_benign = len(flows) - total_attacks
    
    return {
        'flows': flows,
        'stats': {
            'total_flows': len(flows),
            'total_attacks': total_attacks,
            'total_benign': total_benign,
            'attack_rate': total_attacks / len(flows) * 100 if flows else 0,
            'attack_types': dict(attack_types)
        },
        'csv_path': str(csv_path)
    }


def match_pcap_to_csv(pcap_name: str, csv_dir: Path) -> Optional[Path]:
    """
    Trova il CSV corrispondente a un PCAP.
    
    CIC-IDS2017 ha naming convention simile:
    - PCAP: Friday-WorkingHours.pcap
    - CSV: Friday-WorkingHours-Morning.csv, Friday-WorkingHours-Afternoon.csv
    
    Returns:
        Path al CSV (o None se non trovato)
    """
    pcap_base = Path(pcap_name).stem.lower()
    
    # Cerca CSV con nome simile
    for csv_file in csv_dir.glob("*.csv"):
        csv_base = csv_file.stem.lower()
        
        # Match esatto o parziale
        if pcap_base in csv_base or csv_base in pcap_base:
            return csv_file
        
        # Match per giorno della settimana
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            if day in pcap_base and day in csv_base:
                return csv_file
    
    return None


# ==============================================================================
# EVALUATION CON GROUND TRUTH
# ==============================================================================

def evaluate_sniff_against_ground_truth(sniff_result: Dict, 
                                         ground_truth: Dict) -> Dict:
    """
    Valuta risultato sniffata confrontando con ground truth.
    
    Calcola:
    - True Positives: flussi correttamente identificati come attacchi
    - False Positives: flussi benigni identificati come attacchi
    - False Negatives: attacchi non rilevati
    - True Negatives: flussi benigni correttamente ignorati
    
    Args:
        sniff_result: Risultato da analyze_pcap_file
        ground_truth: Ground truth da extract_ground_truth_from_csv
    
    Returns:
        Dict con metriche di evaluation
    """
    gt_flows = ground_truth['flows']
    
    # Flussi rilevati come attacco dallo sniffer
    detected_attacks = set()
    if 'attack_flows' in sniff_result:
        for att in sniff_result['attack_flows']:
            flow_id = att.get('flow_id', '')
            detected_attacks.add(flow_id)
    
    # Flussi analizzati totali (approssimazione)
    flows_analyzed = sniff_result.get('flows_analyzed', 0)
    attacks_detected = sniff_result.get('attacks_detected', 0)
    
    # Metriche
    tp = 0  # Attacco rilevato correttamente
    fp = 0  # Benigno rilevato come attacco
    fn = 0  # Attacco non rilevato
    tn = 0  # Benigno ignorato correttamente
    
    # Per ogni flusso rilevato come attacco
    for flow_id in detected_attacks:
        # Normalizza flow_id per matching (rimuovi protocollo se presente)
        flow_id_normalized = flow_id.split(':')[0] + ':' + flow_id.split(':')[1] if ':' in flow_id else flow_id
        
        # Cerca nel ground truth
        gt_info = gt_flows.get(flow_id) or gt_flows.get(flow_id_normalized)
        
        if gt_info:
            if gt_info['is_attack']:
                tp += 1
            else:
                fp += 1
        else:
            # Flusso non nel ground truth - assumiamo FP
            fp += 1
    
    # Per attacchi nel ground truth non rilevati
    for flow_id, info in gt_flows.items():
        if info['is_attack']:
            if flow_id not in detected_attacks:
                fn += 1
        else:
            if flow_id not in detected_attacks:
                tn += 1
    
    # Calcola metriche
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'confusion_matrix': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        },
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        },
        'summary': {
            'attacks_in_ground_truth': ground_truth['stats']['total_attacks'],
            'attacks_detected_by_sniffer': attacks_detected,
            'correctly_detected': tp,
            'missed_attacks': fn,
            'false_alarms': fp
        }
    }


# ==============================================================================
# COMPARISON FRAMEWORK
# ==============================================================================

def compare_sniff_results(results_list: List[Dict], 
                          labels: List[str] = None) -> Dict:
    """
    Confronta risultati sniffate di diversi modelli.
    
    Args:
        results_list: Lista di risultati da diversi modelli
        labels: Etichette per ogni modello (opzionale)
    
    Returns:
        Dict con confronto
    """
    if labels is None:
        labels = [f"Model_{i}" for i in range(len(results_list))]
    
    comparison = {
        'models': labels,
        'per_model': {},
        'rankings': {}
    }
    
    for label, result in zip(labels, results_list):
        r = result.get('results', result)
        
        comparison['per_model'][label] = {
            'packets_processed': r.get('packets_processed', 0),
            'flows_analyzed': r.get('flows_analyzed', 0),
            'attacks_detected': r.get('attacks_detected', 0),
            'detection_rate': r.get('detection_rate', 0),
            'threshold': r.get('threshold', 0.5),
            'stats': r.get('stats', {})
        }
    
    # Rankings
    by_attacks = sorted(
        comparison['per_model'].items(),
        key=lambda x: x[1]['attacks_detected'],
        reverse=True
    )
    comparison['rankings']['by_attacks_detected'] = [k for k, v in by_attacks]
    
    by_rate = sorted(
        comparison['per_model'].items(),
        key=lambda x: x[1]['detection_rate'],
        reverse=True
    )
    comparison['rankings']['by_detection_rate'] = [k for k, v in by_rate]
    
    return comparison


def generate_comparison_report(model_results: Dict[str, List[Dict]], 
                               output_dir: Path = None) -> Path:
    """
    Genera report completo di confronto tra modelli.
    
    Args:
        model_results: {model_id: [risultati per ogni PCAP]}
        output_dir: Directory output (default: sniff_results/)
    
    Returns:
        Path al report
    """
    if output_dir is None:
        output_dir = get_sniff_results_dir()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'models_compared': list(model_results.keys()),
        'per_model_summary': {},
        'per_pcap_comparison': {},
        'overall_ranking': []
    }
    
    # Aggregato per modello
    for model_id, results in model_results.items():
        total_packets = sum(r.get('results', r).get('packets_processed', 0) for r in results)
        total_flows = sum(r.get('results', r).get('flows_analyzed', 0) for r in results)
        total_attacks = sum(r.get('results', r).get('attacks_detected', 0) for r in results)
        
        report['per_model_summary'][model_id] = {
            'pcaps_analyzed': len(results),
            'total_packets': total_packets,
            'total_flows': total_flows,
            'total_attacks': total_attacks,
            'overall_detection_rate': total_attacks / total_flows * 100 if total_flows > 0 else 0
        }
    
    # Per ogni PCAP
    all_pcaps = set()
    for results in model_results.values():
        for r in results:
            pcap = r.get('results', r).get('pcap') or r.get('metadata', {}).get('pcap_name')
            if pcap:
                all_pcaps.add(pcap)
    
    for pcap in sorted(all_pcaps):
        pcap_comparison = {}
        for model_id, results in model_results.items():
            for r in results:
                r_pcap = r.get('results', r).get('pcap') or r.get('metadata', {}).get('pcap_name')
                if r_pcap == pcap:
                    data = r.get('results', r)
                    pcap_comparison[model_id] = {
                        'attacks': data.get('attacks_detected', 0),
                        'flows': data.get('flows_analyzed', 0),
                        'rate': data.get('detection_rate', 0)
                    }
        report['per_pcap_comparison'][pcap] = pcap_comparison
    
    # Ranking finale
    ranking = sorted(
        report['per_model_summary'].items(),
        key=lambda x: x[1]['overall_detection_rate'],
        reverse=True
    )
    report['overall_ranking'] = [
        {'rank': i+1, 'model': k, 'detection_rate': v['overall_detection_rate']}
        for i, (k, v) in enumerate(ranking)
    ]
    
    # Salva JSON
    report_path = output_dir / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report confronto salvato: {report_path}")
    
    return report_path


# ==============================================================================
# MAIN (per testing)
# ==============================================================================

def main():
    """Test del modulo."""
    print("Sniff Evaluation System")
    print("=" * 60)
    
    # Mostra struttura directory
    results_dir = get_sniff_results_dir()
    print(f"Directory risultati: {results_dir}")
    
    # Lista risultati esistenti
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in ['comparison_report.json']:
            print(f"\n{model_dir.name}/")
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir():
                    pcaps = list(version_dir.glob("*.json"))
                    print(f"  {version_dir.name}/ ({len(pcaps)} risultati)")


if __name__ == "__main__":
    main()