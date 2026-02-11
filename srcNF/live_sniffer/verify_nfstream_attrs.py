#!/usr/bin/env python3
"""
Verifica attributi disponibili in nfstream su PCAP reale.

Stampa tutti gli attributi del flow per capire cosa è disponibile.
"""

import sys
from pathlib import Path
from nfstream import NFStreamer

pcap_path = Path("../../data/pcap/Tuesday-WorkingHours.pcap")
print(f"Analizzando PCAP: {pcap_path}\n")

streamer = NFStreamer(
    source=str(pcap_path),
    statistical_analysis=True,
)

print("=" * 80)
print("ATTRIBUTI NFSTREAM (primo flow)")
print("=" * 80)

for i, flow in enumerate(streamer):
    if i >= 1:
        break
    
    # Lista tutti gli attributi
    attrs = [a for a in dir(flow) if not a.startswith('_')]
    
    print(f"\nTotale attributi: {len(attrs)}\n")
    
    # Attributi critici
    critical = [
        "dst2src_duration_ms",
        "src2dst_duration_ms",
        "bidirectional_duration_ms",
        "dst2src_retrans_packets",
        "src2dst_retrans_packets",
        "bidirectional_retrans_packets",
        "dst2src_bytes",
        "src2dst_bytes",
        "dst2src_packets",
        "src2dst_packets",
    ]
    
    print("ATTRIBUTI CRITICI:")
    print("-" * 80)
    for attr in critical:
        if hasattr(flow, attr):
            val = getattr(flow, attr)
            print(f"  {attr:40s} = {val}")
        else:
            print(f"  {attr:40s} = NOT FOUND ")
    
    print("\n\nTUTTI GLI ATTRIBUTI:")
    print("-" * 80)
    for attr in sorted(attrs):
        try:
            val = getattr(flow, attr)
            # Mostra solo valori non callable
            if not callable(val):
                val_str = str(val)
                if len(val_str) > 50:
                    val_str = val_str[:50] + "..."
                print(f"  {attr:40s} = {val_str}")
        except:
            pass

print("\n" + "=" * 80)
print("CONFRONTO CON FEATURE MAPPER")
print("=" * 80)

print("\nCalcolo manuale THROUGHPUT:")
streamer2 = NFStreamer(source=str(pcap_path), statistical_analysis=True)
for i, flow in enumerate(streamer2):
    if i >= 3:
        break
    
    dst_bytes = getattr(flow, "dst2src_bytes", 0)
    dst_dur = getattr(flow, "dst2src_duration_ms", 0)
    
    if dst_dur > 0:
        # Formula attuale feature_mapper
        tput_current = dst_bytes * 1000.0 / dst_dur
        
        # Formula alternativa (senza * 1000)
        tput_alt = dst_bytes / dst_dur
        
        print(f"\nFlow {i+1}:")
        print(f"  dst2src_bytes:      {dst_bytes}")
        print(f"  dst2src_duration_ms: {dst_dur}")
        print(f"  Throughput (x1000):  {tput_current:.2f} bytes/s")
        print(f"  Throughput (no mul): {tput_alt:.6f}")

print("\n" + "=" * 80)
