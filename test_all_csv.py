#!/usr/bin/env python3
"""Test sniffer su tutti i CSV CIC-IDS2017."""

import sys
import os
from pathlib import Path

# Aggiungi project root al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sniffer import SnifferEvaluator

# Configurazione
CSV_DIR = Path('data/raw')  # Cambia se necessario
MODEL_DIR = 'models/best_model'
ARTIFACTS_DIR = 'artifacts'

# Mapping giorni -> file
CSV_FILES = {
    'Monday': 'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday': 'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday': 'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-Morning': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-Afternoon': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-Morning': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-PortScan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-DDoS': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
}

# Soglie PASS/FAIL
MIN_F1 = 0.90
MAX_FPR = 0.05

def main():
    print("=" * 80)
    print("TEST SNIFFER SU TUTTI I CSV CIC-IDS2017")
    print("=" * 80)
    
    evaluator = SnifferEvaluator(model_dir=MODEL_DIR, artifacts_dir=ARTIFACTS_DIR)
    
    results = []
    
    for day, filename in CSV_FILES.items():
        csv_path = CSV_DIR / filename
        
        # Cerca file (può avere varianti nel nome)
        if not csv_path.exists():
            for f in CSV_DIR.glob('*.csv'):
                if day.lower().replace('-', '') in f.name.lower().replace('-', ''):
                    csv_path = f
                    break
        
        if not csv_path.exists():
            print(f"\n {day}: file non trovato, skip")
            continue
        
        print(f"\n{'='*80}")
        print(f"DAY: {day}")
        print(f"File: {csv_path.name}")
        print("=" * 80)
        
        try:
            metrics = evaluator.evaluate_csv(str(csv_path))
            
            # Monday è solo benigno → F1=0 atteso
            if 'monday' in day.lower():
                status = ' PASS' if metrics['fpr'] <= MAX_FPR else ' FAIL'
                note = '(solo benigno, F1=0 atteso)'
            else:
                f1_ok = metrics['f1'] >= MIN_F1
                fpr_ok = metrics['fpr'] <= MAX_FPR
                status = ' PASS' if (f1_ok and fpr_ok) else ' FAIL'
                note = ''
            
            results.append({
                'day': day,
                'status': status,
                'f1': metrics['f1'],
                'fpr': metrics['fpr'],
                'recall': metrics['recall'],
                'precision': metrics['precision'],
                'samples': metrics['samples'],
                'attacks': metrics['attack'],
                'note': note,
            })
            
            print(f"\nStatus: {status} {note}")
            
        except Exception as e:
            print(f"\n ERRORE: {e}")
            results.append({'day': day, 'status': ' ERROR', 'error': str(e)})
    
    # Riepilogo finale
    print("\n" + "=" * 100)
    print("RIEPILOGO FINALE")
    print("=" * 100)
    print(f"\n{'Day':<25} {'Status':<12} {'F1':>8} {'FPR':>8} {'Recall':>8} {'Samples':>12} {'Attacks':>10}")
    print("-" * 100)
    
    pass_count = 0
    for r in results:
        if 'error' in r:
            print(f"{r['day']:<25} {r['status']:<12} {'ERROR'}")
        else:
            print(f"{r['day']:<25} {r['status']:<12} {r['f1']:>8.4f} {r['fpr']:>8.4f} {r['recall']:>8.4f} {r['samples']:>12,} {r['attacks']:>10,}")
            if '' in r['status']:
                pass_count += 1
    
    print("-" * 100)
    print(f"\nTotale: {pass_count}/{len(results)} PASS")
    
    if pass_count == len(results):
        print("\n TUTTI I TEST SUPERATI!")
        return 0
    else:
        print(f"\n {len(results) - pass_count} test falliti")
        return 1

if __name__ == '__main__':
    sys.exit(main())
