#!/usr/bin/env python3
"""
Show Features from Training Artifacts

Mostra le VERE 24 feature selezionate dal training.
Usa queste per sincronizzare sniffer config e feature_mapper.

USAGE:
    cd NIDS-ML-SSR2
    python3 show_training_features.py
"""

import json
import sys
from pathlib import Path


def main():
    artifacts_json = Path("artifacts/features.json")
    
    if not artifacts_json.exists():
        print(f" ERROR: {artifacts_json} not found!")
        print(f"   Run training first: python srcNF/pipeline.py --model xgboost")
        return 1
    
    print("="*70)
    print("TRAINING ARTIFACTS - FEATURE LIST")
    print("="*70)
    print()
    
    with open(artifacts_json, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    n_features = data['n_features']
    
    print(f"Total features: {n_features}")
    print(f"Scaler type: {data.get('scaler_type', 'unknown')}")
    
    if 'scaler_stats' in data:
        stats = data['scaler_stats']
        print(f"Scaler fitted on: {stats.get('total_rows', 'unknown'):,} rows")
    
    print()
    print("="*70)
    print("FEATURE LIST (EXACT ORDER)")
    print("="*70)
    
    for i, feat in enumerate(features, 1):
        print(f"{i:2d}. {feat}")
    
    print()
    print("="*70)
    print("PYTHON FORMAT (for config.py)")
    print("="*70)
    print()
    print("REQUIRED_FEATURES: List[str] = [")
    for feat in features:
        print(f'    "{feat}",')
    print("]")
    print()
    print(f"N_FEATURES: int = {n_features}")
    print()
    
    # Identifica feature problematiche
    problematic = []
    
    for feat in features:
        if "RETRANSMITTED" in feat:
            problematic.append((feat, "nfstream non fornisce per direction"))
        elif feat == "MIN_TTL":
            problematic.append((feat, "nfstream non fornisce TTL"))
        elif feat == "MAX_IP_PKT_LEN" or feat == "MEAN_IP_PKT_LEN":
            problematic.append((feat, "Calcolabili ma non implementate"))
    
    if problematic:
        print("="*70)
        print("  FEATURE PROBLEMATICHE")
        print("="*70)
        for feat, reason in problematic:
            print(f"  {feat:30s} - {reason}")
        print()
        print("AZIONE RICHIESTA:")
        print("  1. Aggiungi gestione in feature_mapper.py, OPPURE")
        print("  2. Re-train senza queste feature")
    
    # Genera feature_mapper template
    print()
    print("="*70)
    print("FEATURE_MAPPER.PY TEMPLATE")
    print("="*70)
    print()
    
    for feat in features:
        if feat in ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
                    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
                    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
                    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN",
                    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
                    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
                    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
                    "NUM_PKTS_1024_TO_1514_BYTES", "ICMP_IPV4_TYPE", "CLIENT_TCP_FLAGS"]:
            status = " OK (implementata)"
        elif feat == "MIN_TTL":
            status = "  Ritorna 0.0 (non disponibile)"
        elif "RETRANSMITTED" in feat:
            status = " NON implementata (nfstream no support)"
        elif feat in ["MAX_IP_PKT_LEN", "MEAN_IP_PKT_LEN"]:
            status = "  Calcolabile (da implementare)"
        else:
            status = " Da verificare"
        
        print(f"  {feat:35s} {status}")
    
    print()
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
