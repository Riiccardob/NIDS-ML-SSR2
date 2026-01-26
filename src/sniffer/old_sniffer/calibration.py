"""
================================================================================
NIDS-ML - Feature Calibration Tool
================================================================================

Verifica che le feature estratte dallo sniffer (Python/Scapy) siano
allineate con quelle del dataset CIC-IDS2017 (CICFlowMeter/Java).

MODALITA:
---------
1. --csv: Analizza feature nel CSV e mostra statistiche
2. --csv + --pcap: Confronta statistiche feature (non richiede colonne IP)
3. --pcap + --check-coverage: Verifica copertura (richiede colonne IP)

================================================================================
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Setup path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_project_root, get_logger
from src.feature_engineering import load_artifacts

logger = get_logger(__name__)


# ==============================================================================
# FEATURE CRITICHE
# ==============================================================================

CRITICAL_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Fwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Bwd Packet Length Max',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Packet Length Mean',
    'Packet Length Std',
    'FIN Flag Count',
    'SYN Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
]


# ==============================================================================
# UTILITY
# ==============================================================================

def find_column(df: pd.DataFrame, *possible_names: str) -> Optional[str]:
    """Trova colonna cercando tra varianti (case insensitive, spazi)."""
    col_map = {col.strip().lower(): col for col in df.columns}
    for name in possible_names:
        name_lower = name.strip().lower()
        if name_lower in col_map:
            return col_map[name_lower]
    return None


# ==============================================================================
# ANALISI CSV
# ==============================================================================

def analyze_csv(csv_path: Path, sample_size: int = 10000) -> Dict:
    """Analizza le feature nel CSV."""
    print(f"\n{'='*60}")
    print(f"ANALISI CSV: {csv_path.name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    total_rows = len(df)
    
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"\nRighe totali: {total_rows:,}")
    print(f"Righe analizzate: {len(df_sample):,}")
    print(f"Colonne: {len(df.columns)}")
    
    # Label
    label_col = find_column(df, 'Label', 'label')
    if label_col:
        print(f"\n{'─'*40}")
        print(f"DISTRIBUZIONE LABEL")
        print(f"{'─'*40}")
        
        counts = df[label_col].value_counts()
        for label, count in counts.items():
            pct = count / len(df) * 100
            print(f"  {label:<30} {count:>10,} ({pct:>5.1f}%)")
        
        benign = counts.get('BENIGN', 0)
        attacks = len(df) - benign
        print(f"\n  TOTALE BENIGNI:  {benign:,}")
        print(f"  TOTALE ATTACCHI: {attacks:,}")
    
    # Verifica feature
    print(f"\n{'─'*40}")
    print(f"VERIFICA FEATURE")
    print(f"{'─'*40}")
    
    try:
        scaler, selected_features, _, scaler_columns = load_artifacts()
        print(f"\nFeature richieste dal modello: {len(selected_features)}")
        
        df_cols_lower = {c.lower() for c in df.columns}
        missing = [f for f in selected_features if f.lower() not in df_cols_lower]
        
        if missing:
            print(f"⚠️  Feature mancanti: {missing[:5]}")
        else:
            print("✓ Tutte le feature del modello presenti")
    except Exception as e:
        print(f"⚠️  Errore caricamento artifacts: {e}")
        selected_features = []
    
    # Statistiche
    print(f"\n{'─'*40}")
    print(f"STATISTICHE FEATURE")
    print(f"{'─'*40}")
    
    stats = {}
    print(f"\n{'Feature':<28} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    
    for feat in CRITICAL_FEATURES[:15]:
        col = find_column(df_sample, feat)
        if col:
            data = df_sample[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) > 0:
                stats[feat] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'median': float(data.median())
                }
                print(f"{feat:<28} {data.min():>12.1f} {data.max():>12.1f} {data.mean():>12.1f} {data.std():>12.1f}")
    
    return {'file': csv_path.name, 'rows': total_rows, 'stats': stats}


# ==============================================================================
# CONFRONTO STATISTICHE PCAP vs CSV
# ==============================================================================

def compare_stats_pcap_csv(pcap_path: Path, csv_path: Path, max_packets: int = 500000) -> Dict:
    """
    Confronta le STATISTICHE delle feature estratte da PCAP vs quelle del CSV.
    Non richiede colonne IP nel CSV.
    """
    print(f"\n{'='*60}")
    print(f"CONFRONTO STATISTICHE PCAP vs CSV")
    print(f"{'='*60}")
    
    try:
        from scapy.all import PcapReader, IP, TCP, UDP
    except ImportError:
        print("ERRORE: Scapy non installato")
        return {'error': 'Scapy not installed'}
    
    from src.sniffer import Flow
    
    # Carica statistiche CSV
    print(f"\nCaricamento CSV: {csv_path.name}")
    df_csv = pd.read_csv(csv_path, low_memory=False)
    df_csv.columns = df_csv.columns.str.strip()
    
    csv_stats = {}
    for feat in CRITICAL_FEATURES:
        col = find_column(df_csv, feat)
        if col:
            data = df_csv[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) > 0:
                csv_stats[feat] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'p25': float(data.quantile(0.25)),
                    'p50': float(data.median()),
                    'p75': float(data.quantile(0.75))
                }
    
    print(f"Righe CSV: {len(df_csv):,}")
    print(f"Feature con statistiche: {len(csv_stats)}")
    
    # Estrai flussi da PCAP
    print(f"\nEstrazione da PCAP: {pcap_path.name}")
    
    flows = {}
    packets = 0
    
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if packets >= max_packets:
                break
            
            if not pkt.haslayer(IP):
                continue
            
            ip = pkt[IP]
            src_ip, dst_ip = ip.src, ip.dst
            protocol = ip.proto
            pkt_len = len(pkt)
            timestamp = float(pkt.time)
            
            src_port = dst_port = 0
            tcp_flags = {}
            window = None
            
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                window = tcp.window
                flags = tcp.flags
                tcp_flags = {
                    'F': bool(flags & 0x01),
                    'S': bool(flags & 0x02),
                    'R': bool(flags & 0x04),
                    'P': bool(flags & 0x08),
                    'A': bool(flags & 0x10),
                    'U': bool(flags & 0x20)
                }
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            
            if (src_ip, src_port) < (dst_ip, dst_port):
                flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                is_forward = True
            else:
                flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
                is_forward = False
            
            if flow_key not in flows:
                if is_forward:
                    flows[flow_key] = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                else:
                    flows[flow_key] = Flow(dst_ip, src_ip, dst_port, src_port, protocol)
                flows[flow_key].start_time = timestamp
            
            flows[flow_key].add_packet(pkt_len, is_forward, timestamp, tcp_flags, window)
            packets += 1
            
            if packets % 100000 == 0:
                print(f"  Processati: {packets:,} pacchetti, {len(flows):,} flussi")
    
    print(f"\nPacchetti processati: {packets:,}")
    print(f"Flussi estratti: {len(flows):,}")
    
    # Estrai feature da tutti i flussi con >= 5 pacchetti
    valid_flows = [f for f in flows.values() if f.total_packets >= 5]
    print(f"Flussi validi (>=5 pkt): {len(valid_flows):,}")
    
    if len(valid_flows) < 100:
        print("⚠️  Troppi pochi flussi validi per statistiche affidabili")
    
    # Calcola statistiche PCAP
    pcap_features = {feat: [] for feat in CRITICAL_FEATURES}
    
    for flow in valid_flows:
        extracted = flow.extract_features()
        for feat in CRITICAL_FEATURES:
            if feat in extracted:
                pcap_features[feat].append(extracted[feat])
    
    pcap_stats = {}
    for feat, values in pcap_features.items():
        if values:
            arr = np.array(values)
            arr = arr[np.isfinite(arr)]  # Rimuovi inf/nan
            if len(arr) > 0:
                pcap_stats[feat] = {
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'p25': float(np.percentile(arr, 25)),
                    'p50': float(np.median(arr)),
                    'p75': float(np.percentile(arr, 75))
                }
    
    # Confronto
    print(f"\n{'='*60}")
    print(f"CONFRONTO STATISTICHE")
    print(f"{'='*60}")
    
    print(f"\nNOTA: Confrontiamo MEDIANE (p50) perché più robuste agli outlier")
    print(f"\n{'Feature':<28} {'CSV p50':>12} {'PCAP p50':>12} {'Ratio':>10} {'Status':>10}")
    print("-" * 75)
    
    comparisons = []
    
    for feat in CRITICAL_FEATURES[:15]:
        if feat in csv_stats and feat in pcap_stats:
            csv_median = csv_stats[feat]['p50']
            pcap_median = pcap_stats[feat]['p50']
            
            # Calcola ratio (evita divisione per zero)
            if csv_median != 0:
                ratio = pcap_median / csv_median
            elif pcap_median != 0:
                ratio = float('inf')
            else:
                ratio = 1.0
            
            # Status
            if 0.1 <= ratio <= 10:
                status = "✓ OK"
            elif 0.01 <= ratio <= 100:
                status = "⚠️ Diff"
            else:
                status = "✗ WRONG"
            
            print(f"{feat:<28} {csv_median:>12.1f} {pcap_median:>12.1f} {ratio:>10.2f}x {status:>10}")
            
            comparisons.append({
                'feature': feat,
                'csv_median': csv_median,
                'pcap_median': pcap_median,
                'ratio': ratio,
                'ok': 0.1 <= ratio <= 10
            })
    
    # Riepilogo
    ok_count = sum(1 for c in comparisons if c['ok'])
    total = len(comparisons)
    
    print(f"\n{'='*60}")
    print(f"RIEPILOGO")
    print(f"{'='*60}")
    print(f"\nFeature allineate (ratio 0.1x-10x): {ok_count}/{total}")
    
    if ok_count == total:
        print("\n✓ TUTTE le feature sono allineate!")
        print("  Lo sniffer dovrebbe funzionare correttamente.")
    elif ok_count >= total * 0.7:
        print("\n⚠️  La maggior parte delle feature sono allineate.")
        print("  Lo sniffer probabilmente funzionerà, ma con performance ridotte.")
    else:
        print("\n✗ MOLTE feature sono disallineate!")
        print("  Controlla Flow.extract_features() in sniffer.py")
        print("\n  Problemi comuni:")
        print("  - Flow Duration: deve essere in MICROSECONDI")
        print("  - IAT features: devono essere in MICROSECONDI")
    
    return {
        'csv_stats': csv_stats,
        'pcap_stats': pcap_stats,
        'comparisons': comparisons,
        'alignment_rate': ok_count / total if total > 0 else 0
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='NIDS Feature Calibration')
    
    parser.add_argument('--csv', type=Path, help='CSV CIC-IDS2017')
    parser.add_argument('--pcap', type=Path, help='PCAP per confronto statistiche')
    parser.add_argument('--check-coverage', nargs='+', type=Path,
                        help='(Non disponibile - CSV senza colonne IP)')
    parser.add_argument('--sample', type=int, default=10000)
    parser.add_argument('--max-packets', type=int, default=500000)
    
    args = parser.parse_args()
    
    if args.check_coverage:
        print("\n⚠️  --check-coverage non disponibile!")
        print("   I tuoi CSV non contengono le colonne IP (Source IP, Destination IP).")
        print("   Questa è una versione preprocessata del dataset.")
        print("\n   Usa invece: --csv <file> --pcap <file>")
        print("   per confrontare le statistiche delle feature.")
        return
    
    if args.csv:
        if not args.csv.exists():
            print(f"CSV non trovato: {args.csv}")
            sys.exit(1)
        
        analyze_csv(args.csv, args.sample)
        
        if args.pcap:
            if not args.pcap.exists():
                print(f"PCAP non trovato: {args.pcap}")
                sys.exit(1)
            
            compare_stats_pcap_csv(args.pcap, args.csv, args.max_packets)
    
    else:
        print("Specificare --csv")
        parser.print_help()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"PROSSIMI PASSI")
    print(f"{'='*60}")
    print("""
1. Se le feature sono ALLINEATE:
   python src/sniff_evaluation.py --csv Friday.csv --model-type all

2. Se le feature sono DISALLINEATE:
   - Controlla le unità in Flow.extract_features()
   - Flow Duration e IAT devono essere in MICROSECONDI
""")


if __name__ == "__main__":
    main()