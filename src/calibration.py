"""
================================================================================
NIDS-ML - Feature Calibration Tool
================================================================================

Verifica che le feature estratte dallo sniffer (Python/Scapy) siano
allineate con quelle del dataset CIC-IDS2017 (CICFlowMeter/Java).

PERCHE' E' IMPORTANTE:
---------------------
Il modello e' stato trainato su feature estratte da CICFlowMeter (Java).
Se il nostro estrattore (Python) produce feature diverse, il modello non funziona.

USO:
----
# Solo analisi CSV
python src/calibration.py --csv data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv

# Confronto PCAP vs CSV
python src/calibration.py --csv Friday.csv --pcap Friday.pcap

# Verifica copertura PCAP
python src/calibration.py --pcap Friday.pcap --check-coverage data/raw/Friday*.csv

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
# FEATURE CRITICHE PER IL MODELLO
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
    'Flow IAT Max',
    'Fwd IAT Mean',
    'Fwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Total',
    'Packet Length Mean',
    'Packet Length Std',
    'FIN Flag Count',
    'SYN Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
]


# ==============================================================================
# UTILITY PER COLONNE CSV
# ==============================================================================

def find_column(df: pd.DataFrame, *possible_names: str) -> Optional[str]:
    """
    Trova una colonna nel DataFrame cercando tra vari nomi possibili.
    Gestisce spazi iniziali/finali nei nomi colonne.
    """
    # Normalizza nomi colonne (rimuovi spazi)
    col_map = {col.strip().lower(): col for col in df.columns}
    
    for name in possible_names:
        name_lower = name.strip().lower()
        if name_lower in col_map:
            return col_map[name_lower]
    
    return None


def get_ip_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Trova le colonne Source IP e Destination IP nel DataFrame.
    Gestisce varianti di nomi e spazi.
    """
    src_col = find_column(df, 
        'Source IP', 'Src IP', 'source_ip', 'src_ip', 
        ' Source IP', 'SourceIP', 'SrcIP')
    
    dst_col = find_column(df,
        'Destination IP', 'Dst IP', 'destination_ip', 'dst_ip',
        ' Destination IP', 'DestinationIP', 'DstIP')
    
    return src_col, dst_col


# ==============================================================================
# ANALISI CSV
# ==============================================================================

def analyze_csv(csv_path: Path, sample_size: int = 5000) -> Dict:
    """
    Analizza le feature nel CSV CIC-IDS2017.
    
    Mostra:
    - Distribuzione label
    - Feature presenti vs richieste
    - Statistiche feature critiche
    """
    print(f"\n{'='*60}")
    print(f"ANALISI CSV: {csv_path.name}")
    print(f"{'='*60}")
    
    # Carica
    df = pd.read_csv(csv_path, low_memory=False)
    
    # IMPORTANTE: Strip spazi da TUTTE le colonne
    df.columns = df.columns.str.strip()
    
    total_rows = len(df)
    
    # Sample per statistiche
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"\nRighe totali: {total_rows:,}")
    print(f"Righe analizzate: {len(df_sample):,}")
    print(f"Colonne: {len(df.columns)}")
    
    # Label
    label_col = find_column(df, 'Label', 'label', ' Label')
    
    if label_col:
        print(f"\n{'─'*40}")
        print(f"DISTRIBUZIONE LABEL")
        print(f"{'─'*40}")
        
        counts = df[label_col].value_counts()
        total = len(df)
        
        for label, count in counts.items():
            pct = count / total * 100
            print(f"  {label:<30} {count:>10,} ({pct:>5.1f}%)")
        
        # Conta attacchi vs benigni
        benign = counts.get('BENIGN', 0)
        attacks = total - benign
        print(f"\n  TOTALE BENIGNI:  {benign:,}")
        print(f"  TOTALE ATTACCHI: {attacks:,}")
    
    # Verifica feature
    print(f"\n{'─'*40}")
    print(f"VERIFICA FEATURE")
    print(f"{'─'*40}")
    
    # Carica artifacts
    try:
        scaler, selected_features, _, scaler_columns = load_artifacts()
        print(f"\nFeature richieste dal modello: {len(selected_features)}")
    except Exception as e:
        print(f"\n⚠️  Impossibile caricare artifacts: {e}")
        selected_features = []
        scaler_columns = []
    
    print(f"Colonne nel CSV: {len(df.columns)}")
    
    # Feature critiche mancanti (normalizzato)
    df_cols_lower = [c.lower() for c in df.columns]
    missing_critical = [f for f in CRITICAL_FEATURES 
                        if f.lower() not in df_cols_lower]
    missing_model = [f for f in selected_features 
                     if f.lower() not in df_cols_lower]
    
    if missing_critical:
        print(f"\n⚠️  Feature CRITICHE mancanti nel CSV:")
        for f in missing_critical[:10]:
            print(f"   - {f}")
    else:
        print(f"\n✓ Tutte le feature critiche presenti")
    
    if missing_model:
        print(f"\n⚠️  Feature del MODELLO mancanti nel CSV:")
        for f in missing_model[:10]:
            print(f"   - {f}")
    else:
        print(f"\n✓ Tutte le feature del modello presenti")
    
    # Verifica colonne IP
    src_col, dst_col = get_ip_columns(df)
    print(f"\nColonne IP trovate:")
    print(f"  Source IP:      {src_col or 'NON TROVATA'}")
    print(f"  Destination IP: {dst_col or 'NON TROVATA'}")
    
    # Statistiche
    print(f"\n{'─'*40}")
    print(f"STATISTICHE FEATURE (campione)")
    print(f"{'─'*40}")
    
    stats = {}
    
    print(f"\n{'Feature':<30} {'Min':>12} {'Max':>12} {'Mean':>12} {'Median':>12}")
    print("-" * 80)
    
    for feat in CRITICAL_FEATURES[:15]:
        # Cerca la colonna (gestisce varianti)
        col_name = find_column(df_sample, feat)
        if col_name:
            col = df_sample[col_name].replace([np.inf, -np.inf], np.nan).dropna()
            if len(col) > 0:
                stats[feat] = {
                    'min': float(col.min()),
                    'max': float(col.max()),
                    'mean': float(col.mean()),
                    'median': float(col.median())
                }
                print(f"{feat:<30} {col.min():>12.1f} {col.max():>12.1f} {col.mean():>12.1f} {col.median():>12.1f}")
    
    return {
        'file': csv_path.name,
        'total_rows': total_rows,
        'columns': list(df.columns),
        'missing_critical': missing_critical,
        'missing_model': missing_model,
        'stats': stats,
        'src_col': src_col,
        'dst_col': dst_col
    }


# ==============================================================================
# CONFRONTO PCAP VS CSV
# ==============================================================================

def compare_pcap_csv(pcap_path: Path, csv_path: Path, num_flows: int = 30) -> Dict:
    """
    Estrae feature da PCAP e confronta con CSV.
    
    Questo e' il TEST DEFINITIVO per verificare se lo sniffer
    estrae feature correttamente.
    """
    print(f"\n{'='*60}")
    print(f"CONFRONTO PCAP vs CSV")
    print(f"{'='*60}")
    
    try:
        from scapy.all import PcapReader, IP, TCP, UDP
    except ImportError:
        print("ERRORE: Scapy non installato. Esegui: pip install scapy")
        return {'error': 'Scapy not installed'}
    
    from src.sniffer import Flow
    
    # Carica CSV
    print(f"\nCaricamento CSV: {csv_path.name}")
    df_csv = pd.read_csv(csv_path, low_memory=False)
    df_csv.columns = df_csv.columns.str.strip()  # IMPORTANTE!
    print(f"Righe CSV: {len(df_csv):,}")
    
    # Identifica colonne IP nel CSV
    src_col, dst_col = get_ip_columns(df_csv)
    print(f"Colonna Source IP: {src_col}")
    print(f"Colonna Dest IP: {dst_col}")
    
    if not src_col or not dst_col:
        print("\n⚠️  ERRORE: Colonne IP non trovate nel CSV!")
        print("    Colonne disponibili:")
        for col in sorted(df_csv.columns)[:20]:
            print(f"      - '{col}'")
        return {'error': 'IP columns not found'}
    
    # Estrai flussi da PCAP
    print(f"\nEstrazione da PCAP: {pcap_path.name}")
    
    flows = {}
    packets = 0
    max_packets = 500000
    
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
            
            # Chiave flusso bidirezionale
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
    
    # Filtra flussi validi
    valid_flows = [(k, f) for k, f in flows.items() if f.total_packets >= 10]
    print(f"Flussi con >=10 pacchetti: {len(valid_flows)}")
    
    if not valid_flows:
        print("\n⚠️  Nessun flusso valido trovato!")
        return {'error': 'No valid flows'}
    
    # Confronta feature
    print(f"\n{'─'*40}")
    print(f"CONFRONTO FEATURE")
    print(f"{'─'*40}")
    
    comparisons = []
    matches_found = 0
    
    # Features da confrontare
    compare_features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Fwd Packet Length Mean',
        'Bwd Packet Length Mean',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Flow IAT Mean',
    ]
    
    for flow_key, flow in valid_flows[:200]:  # Cerca nei primi 200
        if matches_found >= num_flows:
            break
        
        # Cerca corrispondenza nel CSV
        mask = (
            (df_csv[src_col] == flow.src_ip) &
            (df_csv[dst_col] == flow.dst_ip)
        )
        matching = df_csv[mask]
        
        if len(matching) == 0:
            # Prova direzione opposta
            mask = (
                (df_csv[src_col] == flow.dst_ip) &
                (df_csv[dst_col] == flow.src_ip)
            )
            matching = df_csv[mask]
        
        if len(matching) == 0:
            continue
        
        csv_row = matching.iloc[0]
        matches_found += 1
        
        print(f"\n--- Flusso {matches_found}: {flow.src_ip} -> {flow.dst_ip} ---")
        print(f"    Pacchetti PCAP: {flow.total_packets}")
        
        # Estrai feature dal flusso
        pcap_features = flow.extract_features()
        
        for feat in compare_features:
            col_name = find_column(df_csv, feat)
            if col_name and col_name in csv_row.index:
                csv_val = float(csv_row[col_name])
                pcap_val = float(pcap_features.get(feat, 0))
                
                if csv_val != 0:
                    diff = abs((pcap_val - csv_val) / csv_val) * 100
                else:
                    diff = 0 if pcap_val == 0 else 100
                
                status = "✓" if diff < 50 else "✗"
                print(f"    {status} {feat:<25} CSV={csv_val:>12.1f} PCAP={pcap_val:>12.1f} ({diff:>5.1f}%)")
                
                comparisons.append({
                    'feature': feat,
                    'csv': csv_val,
                    'pcap': pcap_val,
                    'diff_pct': diff
                })
    
    if matches_found == 0:
        print("\n⚠️  Nessuna corrispondenza trovata tra PCAP e CSV!")
        print("    Possibili cause:")
        print("    - Il PCAP non corrisponde al CSV")
        print("    - Formati IP diversi")
        return {'error': 'No matches found', 'comparisons': []}
    
    # Riepilogo
    if comparisons:
        df_comp = pd.DataFrame(comparisons)
        
        print(f"\n{'='*60}")
        print(f"RIEPILOGO CALIBRAZIONE")
        print(f"{'='*60}")
        
        good = len(df_comp[df_comp['diff_pct'] < 50])
        bad = len(df_comp[df_comp['diff_pct'] >= 50])
        total = len(df_comp)
        
        print(f"\nConfronto totali: {total}")
        print(f"Allineati (<50% diff): {good} ({good/total*100:.1f}%)")
        print(f"Disallineati (>=50%): {bad} ({bad/total*100:.1f}%)")
        
        # Media per feature
        print(f"\n{'Feature':<30} {'Diff Media %':>15}")
        print("-" * 50)
        
        by_feat = df_comp.groupby('feature')['diff_pct'].mean().sort_values(ascending=False)
        for feat, avg_diff in by_feat.items():
            status = "✓" if avg_diff < 50 else "✗"
            print(f"{status} {feat:<28} {avg_diff:>15.1f}%")
        
        if bad > total * 0.3:
            print(f"\n⚠️  ATTENZIONE: Molte feature disallineate!")
            print(f"    Il modello potrebbe non funzionare correttamente.")
            print(f"    Controlla Flow.extract_features() in sniffer.py")
        else:
            print(f"\n✓ La maggior parte delle feature sono allineate.")
            print(f"    Lo sniffer dovrebbe funzionare.")
    
    return {
        'matches_found': matches_found,
        'comparisons': comparisons
    }


# ==============================================================================
# VERIFICA COPERTURA PCAP vs CSV
# ==============================================================================

def check_pcap_csv_coverage(pcap_path: Path, csv_paths: List[Path]) -> Dict:
    """
    Verifica che il PCAP contenga tutti i dati dei CSV corrispondenti.
    """
    print(f"\n{'='*60}")
    print(f"VERIFICA COPERTURA PCAP vs CSV")
    print(f"{'='*60}")
    
    try:
        from scapy.all import PcapReader, IP
    except ImportError:
        print("ERRORE: Scapy non installato")
        return {'error': 'Scapy not installed'}
    
    # Conta flussi unici nel PCAP
    print(f"\nAnalisi PCAP: {pcap_path.name}")
    
    pcap_flows = set()
    packets = 0
    max_packets = 1000000
    
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if packets >= max_packets:
                print(f"  (limitato a {max_packets:,} pacchetti)")
                break
            
            if pkt.haslayer(IP):
                ip = pkt[IP]
                # Chiave semplificata (solo IP)
                key = tuple(sorted([ip.src, ip.dst]))
                pcap_flows.add(key)
            
            packets += 1
            
            if packets % 200000 == 0:
                print(f"  Processati: {packets:,} pacchetti")
    
    print(f"Pacchetti analizzati: {packets:,}")
    print(f"Coppie IP uniche nel PCAP: {len(pcap_flows):,}")
    
    # Conta flussi nei CSV
    print(f"\nAnalisi CSV:")
    
    total_csv_rows = 0
    csv_flows = set()
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"  ⚠️  Non trovato: {csv_path.name}")
            continue
        
        df = pd.read_csv(csv_path, low_memory=False, nrows=100000)
        df.columns = df.columns.str.strip()  # IMPORTANTE!
        
        # Trova colonne IP
        src_col, dst_col = get_ip_columns(df)
        
        if src_col and dst_col:
            for _, row in df.iterrows():
                try:
                    key = tuple(sorted([str(row[src_col]), str(row[dst_col])]))
                    csv_flows.add(key)
                except:
                    pass
        else:
            print(f"  ⚠️  Colonne IP non trovate in {csv_path.name}")
            print(f"      Colonne: {list(df.columns)[:10]}")
        
        total_csv_rows += len(df)
        print(f"  {csv_path.name}: {len(df):,} righe")
    
    print(f"\nTotale righe CSV: {total_csv_rows:,}")
    print(f"Coppie IP uniche nei CSV: {len(csv_flows):,}")
    
    # Confronto
    common = pcap_flows & csv_flows
    only_pcap = pcap_flows - csv_flows
    only_csv = csv_flows - pcap_flows
    
    print(f"\n{'─'*40}")
    print(f"COPERTURA")
    print(f"{'─'*40}")
    print(f"IP comuni (PCAP ∩ CSV): {len(common):,}")
    print(f"Solo in PCAP: {len(only_pcap):,}")
    print(f"Solo in CSV: {len(only_csv):,}")
    
    if len(csv_flows) > 0:
        coverage = len(common) / len(csv_flows) * 100
        print(f"\nCopertura CSV da PCAP: {coverage:.1f}%")
        
        if coverage > 80:
            print("✓ Il PCAP contiene la maggior parte dei flussi dei CSV")
        elif coverage > 50:
            print("⚠️  Copertura parziale - alcuni flussi CSV non sono nel PCAP")
        else:
            print("⚠️  Bassa copertura - il PCAP potrebbe non corrispondere ai CSV")
    else:
        print("\n⚠️  Nessuna coppia IP trovata nei CSV!")
        coverage = 0
    
    return {
        'pcap_flows': len(pcap_flows),
        'csv_flows': len(csv_flows),
        'common': len(common),
        'coverage_pct': coverage
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NIDS Feature Calibration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Analisi CSV
  python src/calibration.py --csv data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv

  # Confronto PCAP vs CSV  
  python src/calibration.py --csv Friday.csv --pcap Friday.pcap

  # Verifica copertura (PCAP contiene i dati di piu' CSV?)
  python src/calibration.py --pcap Friday.pcap --check-coverage data/raw/Friday*.csv
        """
    )
    
    parser.add_argument('--csv', type=Path, help='CSV CIC-IDS2017 da analizzare')
    parser.add_argument('--pcap', type=Path, help='PCAP per confronto')
    parser.add_argument('--check-coverage', nargs='+', type=Path,
                        help='CSV per verifica copertura PCAP')
    parser.add_argument('--sample', type=int, default=5000,
                        help='Righe da analizzare per statistiche')
    parser.add_argument('--flows', type=int, default=30,
                        help='Flussi da confrontare')
    
    args = parser.parse_args()
    
    # Verifica copertura
    if args.pcap and args.check_coverage:
        if not args.pcap.exists():
            print(f"PCAP non trovato: {args.pcap}")
            sys.exit(1)
        check_pcap_csv_coverage(args.pcap, args.check_coverage)
        return
    
    # Analisi CSV
    if args.csv:
        if not args.csv.exists():
            print(f"CSV non trovato: {args.csv}")
            sys.exit(1)
        
        analyze_csv(args.csv, args.sample)
        
        # Confronto PCAP se fornito
        if args.pcap:
            if not args.pcap.exists():
                print(f"PCAP non trovato: {args.pcap}")
                sys.exit(1)
            
            compare_pcap_csv(args.pcap, args.csv, args.flows)
    
    else:
        print("Specificare almeno --csv")
        parser.print_help()
        sys.exit(1)
    
    # Suggerimenti
    print(f"\n{'='*60}")
    print(f"PROSSIMI PASSI")
    print(f"{'='*60}")
    
    if args.pcap:
        print("""
Se le feature sono ALLINEATE (diff < 50%):
  1. Procedi con i test di evaluation su CSV
  2. python src/sniff_evaluation.py --csv Friday.csv --model-type all

Se le feature sono DISALLINEATE (diff > 50%):
  1. Controlla Flow.extract_features() in sniffer.py
  2. Verifica le unita' (es: microsecondi vs secondi per IAT)
  3. Confronta con la documentazione CICFlowMeter
""")
    else:
        print(f"""
Per un confronto completo con il PCAP:
  python src/calibration.py --csv {args.csv.name} --pcap <pcap_corrispondente>

Per verificare copertura PCAP:
  python src/calibration.py --pcap Friday.pcap --check-coverage data/raw/Friday*.csv
""")


if __name__ == "__main__":
    main()