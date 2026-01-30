#!/usr/bin/env python3
"""Test sniffer su PCAP CIC-IDS2017."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.sniffer import SnifferEngine

# Configurazione
PCAP_DIR = Path('data/pcap')  # Cambia se necessario
MODEL_DIR = 'models/best_model'
ARTIFACTS_DIR = 'artifacts'

# File PCAP disponibili (adatta ai tuoi file)
PCAP_FILES = [
    'Monday-WorkingHours.pcap',
    'Tuesday-WorkingHours.pcap',
    'Wednesday-WorkingHours.pcap',
    'Thursday-WorkingHours.pcap',
    'Friday-WorkingHours.pcap',
]

def main():
    print("=" * 80)
    print("TEST SNIFFER SU PCAP CIC-IDS2017")
    print("=" * 80)
    
    engine = SnifferEngine(model_dir=MODEL_DIR, artifacts_dir=ARTIFACTS_DIR)
    
    results = []
    
    for pcap_name in PCAP_FILES:
        pcap_path = PCAP_DIR / pcap_name
        
        # Cerca varianti
        if not pcap_path.exists():
            for f in PCAP_DIR.glob('*.pcap'):
                if pcap_name.split('-')[0].lower() in f.name.lower():
                    pcap_path = f
                    break
        
        if not pcap_path.exists():
            print(f"\n {pcap_name}: non trovato, skip")
            continue
        
        print(f"\n{'='*80}")
        print(f"PCAP: {pcap_path.name}")
        print("=" * 80)
        
        try:
            # Analizza TUTTI i pacchetti (nessun limite)
            engine.analyze_pcap(str(pcap_path), max_packets=None)
            
            stats = engine.get_stats()
            results.append({
                'pcap': pcap_path.name,
                'flows': stats['flows_analyzed'],
                'attacks': stats['attacks_detected'],
                'benign': stats['benign_flows'],
                'packets': stats['packets_processed'],
            })
            
        except Exception as e:
            print(f"\n ERRORE: {e}")
            results.append({'pcap': pcap_path.name, 'error': str(e)})
    
    # Riepilogo
    print("\n" + "=" * 100)
    print("RIEPILOGO PCAP")
    print("=" * 100)
    print(f"\n{'PCAP':<40} {'Packets':>12} {'Flows':>10} {'Attacks':>10} {'Benign':>10} {'Attack%':>10}")
    print("-" * 100)
    
    total_flows = total_attacks = 0
    for r in results:
        if 'error' in r:
            print(f"{r['pcap']:<40} ERROR: {r['error'][:50]}")
        else:
            rate = r['attacks'] / r['flows'] * 100 if r['flows'] > 0 else 0
            print(f"{r['pcap']:<40} {r['packets']:>12,} {r['flows']:>10,} {r['attacks']:>10,} {r['benign']:>10,} {rate:>9.2f}%")
            total_flows += r['flows']
            total_attacks += r['attacks']
    
    print("-" * 100)
    total_rate = total_attacks / total_flows * 100 if total_flows > 0 else 0
    print(f"{'TOTALE':<40} {'':<12} {total_flows:>10,} {total_attacks:>10,} {total_flows-total_attacks:>10,} {total_rate:>9.2f}%")
    
    print("\n NOTA: I risultati PCAP differiscono da CSV per via dell'aggregazione flussi diversa.")
    print("   Il test su CSV è il gold standard per le metriche.")

if __name__ == '__main__':
    main()
