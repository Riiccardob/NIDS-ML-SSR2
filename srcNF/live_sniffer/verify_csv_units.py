#!/usr/bin/env python3
"""
Verifica unità di misura feature nel dataset NF-UQ-NIDS-v2.

Confronta i valori raw del CSV con quelli estratti da nfstream su PCAP.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Leggi sample del CSV originale
csv_path = Path("../../data/raw/NF-UQ-NIDS-v2.csv")
print(f"Caricamento CSV: {csv_path}")
print("Leggendo prime 10,000 righe...\n")

df = pd.read_csv(csv_path, nrows=10000)

# Feature critiche da verificare
features = [
    "DURATION_IN",
    "DURATION_OUT", 
    "FLOW_DURATION_MILLISECONDS",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "RETRANSMITTED_OUT_PKTS",
]

print("=" * 80)
print("STATISTICHE FEATURE RAW (CSV Originale)")
print("=" * 80)

for feat in features:
    if feat in df.columns:
        vals = df[feat].dropna()
        print(f"\n{feat}:")
        print(f"  Min:    {vals.min():.6f}")
        print(f"  Max:    {vals.max():.6f}")
        print(f"  Mean:   {vals.mean():.6f}")
        print(f"  Median: {vals.median():.6f}")
        print(f"  P90:    {vals.quantile(0.9):.6f}")
        print(f"  Nonzero: {(vals != 0).sum() / len(vals) * 100:.1f}%")
        
        # Sample valori (primi 5 non-zero)
        samples = vals[vals != 0].head(5).values
        print(f"  Sample: {samples}")
    else:
        print(f"\n{feat}: COLONNA NON TROVATA")

print("\n" + "=" * 80)
print("VERIFICA UNITÀ DI MISURA")
print("=" * 80)

# DURATION_* dovrebbe essere in millisecondi se FLOW_DURATION_MILLISECONDS è il riferimento
if "FLOW_DURATION_MILLISECONDS" in df.columns and "DURATION_IN" in df.columns:
    flow_dur = df["FLOW_DURATION_MILLISECONDS"].dropna()
    dur_in = df["DURATION_IN"].dropna()
    
    print(f"\nFLOW_DURATION_MILLISECONDS:")
    print(f"  Range tipico: {flow_dur.quantile(0.1):.1f} - {flow_dur.quantile(0.9):.1f} ms")
    
    print(f"\nDURATION_IN:")
    print(f"  Range tipico: {dur_in.quantile(0.1):.1f} - {dur_in.quantile(0.9):.1f}")
    
    # Se DURATION_IN è in millisecondi, dovrebbe essere < FLOW_DURATION in media
    # Se è in secondi, sarà molto più piccolo
    ratio = dur_in.mean() / flow_dur.mean()
    print(f"\nRatio DURATION_IN / FLOW_DURATION: {ratio:.6f}")
    
    if ratio < 0.001:
        print("  → DURATION_IN probabilmente in SECONDI (ratio ~0.001)")
    elif 0.1 < ratio < 2:
        print("  → DURATION_IN probabilmente in MILLISECONDI (ratio ~1)")
    else:
        print(f"  → Unità non chiara (ratio anomalo: {ratio})")

# Throughput: bytes/s o KB/s?
if "DST_TO_SRC_AVG_THROUGHPUT" in df.columns and "OUT_BYTES" in df.columns:
    tput = df["DST_TO_SRC_AVG_THROUGHPUT"].dropna()
    out_bytes = df["OUT_BYTES"].dropna()
    
    print(f"\n\nDST_TO_SRC_AVG_THROUGHPUT:")
    print(f"  Range: {tput.min():.1f} - {tput.max():.1f}")
    print(f"  Mean:  {tput.mean():.1f}")
    
    print(f"\nOUT_BYTES:")
    print(f"  Range: {out_bytes.min():.1f} - {out_bytes.max():.1f}")
    print(f"  Mean:  {out_bytes.mean():.1f}")
    
    # Se throughput è in bytes/s, dovrebbe essere comparabile a OUT_BYTES / DURATION
    # Se è in KB/s o Mb/s, sarà molto più piccolo
    print(f"\nRatio OUT_BYTES / THROUGHPUT: {out_bytes.mean() / max(tput.mean(), 1):.1f}")
    
    if tput.mean() > 1e6:
        print("  → THROUGHPUT probabilmente in bytes/s (valori > 1MB/s)")
    elif tput.mean() > 1e3:
        print("  → THROUGHPUT probabilmente in KB/s (valori in migliaia)")
    else:
        print("  → THROUGHPUT probabilmente in MB/s o altro")

print("\n" + "=" * 80)
