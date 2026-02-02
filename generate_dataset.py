#!/usr/bin/env python3
"""
NIDS-ML Dataset Generator v2 - USA IL TUO VERO EXTRACTOR

IMPORTANTE: Questo script importa e usa il TUO vero feature extractor
da src/sniffer/features.py, NON una copia semplificata.

Questo garantisce che:
1. Le feature generate sono IDENTICHE a quelle dello sniffer live
2. Il modello retrainato sarà compatibile con lo sniffer

Pipeline:
1. Legge PCAP
2. Crea Flow objects (come fa il tuo sniffer)
3. Estrae feature con FeatureExtractor (il TUO vero extractor)
4. Applica labeling basato su IP/timestamp
5. Salva CSV pronto per retrain_aligned.py

Uso:
    python generate_dataset_v2.py --day tuesday --pcap data/pcap/Tuesday-WorkingHours.pcap
    python generate_dataset_v2.py --all --pcap-dir data/pcap/
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd

# CRITICO: Aggiungi il path del progetto per importare i moduli sniffer
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Verifica dipendenze
try:
    from scapy.all import PcapReader, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("ERROR: scapy not installed. Run: pip install scapy")
    sys.exit(1)

# CRITICO: Importa il TUO VERO feature extractor
try:
    from src.sniffer.features import FeatureExtractor, FEATURE_NAMES
    from src.sniffer.flow import Flow
    SNIFFER_AVAILABLE = True
    print("SUCCESS: Imported YOUR sniffer modules (src.sniffer.features, src.sniffer.flow)")
except ImportError as e:
    print(f"WARNING: Could not import sniffer modules: {e}")
    print("Will use fallback Flow class")
    SNIFFER_AVAILABLE = False


# ==============================================================================
# GROUND TRUTH DATA
# ==============================================================================

GROUND_TRUTH = {
    "monday": {
        "date": "2017-07-03",
        "attacks": []
    },
    "tuesday": {
        "date": "2017-07-04",
        "attacks": [
            {"name": "FTP-Patator", "start": "09:20", "end": "10:20", "attacker": "172.16.0.1", "victim_port": 21},
            {"name": "SSH-Patator", "start": "14:00", "end": "15:00", "attacker": "172.16.0.1", "victim_port": 22}
        ]
    },
    "wednesday": {
        "date": "2017-07-05",
        "attacks": [
            {"name": "DoS slowloris", "start": "09:47", "end": "10:10", "attacker": "172.16.0.1"},
            {"name": "DoS Slowhttptest", "start": "10:14", "end": "10:35", "attacker": "172.16.0.1"},
            {"name": "DoS Hulk", "start": "10:43", "end": "11:00", "attacker": "172.16.0.1"},
            {"name": "DoS GoldenEye", "start": "11:10", "end": "11:23", "attacker": "172.16.0.1"},
            {"name": "Heartbleed", "start": "15:12", "end": "15:32", "attacker": "172.16.0.1"}
        ]
    },
    "thursday_morning": {
        "date": "2017-07-06",
        "attacks": [
            {"name": "Web Attack - Brute Force", "start": "09:20", "end": "10:00", "attacker": "172.16.0.1"},
            {"name": "Web Attack - XSS", "start": "10:15", "end": "10:35", "attacker": "172.16.0.1"},
            {"name": "Web Attack - Sql Injection", "start": "10:40", "end": "10:42", "attacker": "172.16.0.1"}
        ]
    },
    "thursday_afternoon": {
        "date": "2017-07-06",
        "attacks": [
            {"name": "Infiltration", "start": "14:19", "end": "14:35", "attacker": "172.16.0.1"}
        ]
    },
    "friday_morning": {
        "date": "2017-07-07",
        "attacks": [
            {"name": "Bot", "start": "10:02", "end": "11:02", "attacker": "192.168.10.8"}
        ]
    },
    "friday_portscan": {
        "date": "2017-07-07",
        "attacks": [
            {"name": "PortScan", "start": "13:55", "end": "14:35", "attacker": "172.16.0.1"}
        ]
    },
    "friday_ddos": {
        "date": "2017-07-07",
        "attacks": [
            {"name": "DDoS", "start": "15:56", "end": "16:16", "attacker": "172.16.0.1"}
        ]
    }
}

ATTACKER_IPS = {"172.16.0.1", "192.168.10.8"}


# ==============================================================================
# LABELING FUNCTIONS
# ==============================================================================

def parse_time(time_str: str, date_str: str) -> datetime:
    """Parse time string to datetime."""
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")


def is_attack_flow(
    src_ip: str,
    dst_ip: str,
    timestamp: datetime,
    day_info: Dict
) -> Tuple[bool, str]:
    """
    Determina se un flusso è un attacco.
    
    Returns:
        (is_attack, attack_name)
    """
    involves_attacker = src_ip in ATTACKER_IPS or dst_ip in ATTACKER_IPS
    
    if not involves_attacker:
        return False, "BENIGN"
    
    for attack in day_info.get("attacks", []):
        try:
            start = parse_time(attack["start"], day_info["date"])
            end = parse_time(attack["end"], day_info["date"])
            
            # Margine di 5 minuti
            start -= timedelta(minutes=5)
            end += timedelta(minutes=5)
            
            if start <= timestamp <= end:
                if "attacker" in attack:
                    if src_ip == attack["attacker"] or dst_ip == attack["attacker"]:
                        return True, attack["name"]
        except Exception:
            continue
    
    return False, "BENIGN"


# ==============================================================================
# FLOW MANAGEMENT (uses YOUR Flow class if available)
# ==============================================================================

class FlowManager:
    """
    Gestisce i flussi di rete.
    Usa la classe Flow del TUO sniffer se disponibile.
    """
    
    def __init__(self, flow_timeout: float = 120.0):
        self.flows: Dict[str, 'Flow'] = {}
        self.flow_timeout = flow_timeout
        self.feature_extractor = FeatureExtractor() if SNIFFER_AVAILABLE else None
    
    def _get_flow_key(self, pkt) -> Optional[Tuple[str, str, int, int, int]]:
        """Genera chiave univoca per il flusso (bidirezionale)."""
        try:
            if not pkt.haslayer(IP):
                return None
            
            ip = pkt[IP]
            src_ip = ip.src
            dst_ip = ip.dst
            proto = ip.proto
            
            src_port = 0
            dst_port = 0
            
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            
            # Chiave bidirezionale (ordina per garantire consistenza)
            if (src_ip, src_port) < (dst_ip, dst_port):
                return (src_ip, dst_ip, src_port, dst_port, proto)
            else:
                return (dst_ip, src_ip, dst_port, src_port, proto)
                
        except Exception:
            return None
    
    def _flow_key_to_string(self, key: Tuple) -> str:
        """Converte chiave in stringa."""
        return f"{key[0]}:{key[2]}-{key[1]}:{key[3]}-{key[4]}"
    
    def process_packet(self, pkt, timestamp: float):
        """Processa un pacchetto e aggiorna/crea il flusso."""
        key = self._get_flow_key(pkt)
        if not key:
            return
        
        key_str = self._flow_key_to_string(key)
        
        if key_str not in self.flows:
            # Crea nuovo Flow usando la TUA classe
            if SNIFFER_AVAILABLE:
                flow = Flow(
                    src_ip=key[0],
                    dst_ip=key[1],
                    src_port=key[2],
                    dst_port=key[3],
                    protocol=key[4],
                    # timestamp=timestamp
                )
            else:
                # Fallback se sniffer non disponibile
                flow = FallbackFlow(
                    src_ip=key[0],
                    dst_ip=key[1],
                    src_port=key[2],
                    dst_port=key[3],
                    protocol=key[4],
                    # timestamp=timestamp
                )
            self.flows[key_str] = flow
        
        # Aggiungi pacchetto al flow
        flow = self.flows[key_str]
        try:
            flow.add_packet(pkt, timestamp)
        except Exception as e:
            # Se add_packet fallisce, prova metodo alternativo
            pass
    
    def get_expired_flows(self, current_time: float) -> List[str]:
        """Restituisce chiavi dei flussi scaduti."""
        expired = []
        for key, flow in self.flows.items():
            try:
                last_time = flow.last_timestamp if hasattr(flow, 'last_timestamp') else flow.end_time
                if current_time - last_time > self.flow_timeout:
                    expired.append(key)
            except Exception:
                expired.append(key)
        return expired
    
    def extract_features(self, flow_key: str) -> Optional[Dict[str, float]]:
        """Estrae feature da un flusso usando IL TUO FeatureExtractor."""
        if flow_key not in self.flows:
            return None
        
        flow = self.flows[flow_key]
        
        try:
            if self.feature_extractor:
                # USA IL TUO VERO EXTRACTOR
                features = self.feature_extractor.extract(flow)
            else:
                # Fallback
                features = flow.get_features()
            
            # Aggiungi metadati del flusso
            features['_src_ip'] = flow.src_ip
            features['_dst_ip'] = flow.dst_ip
            features['_src_port'] = flow.src_port
            features['_dst_port'] = flow.dst_port
            features['_timestamp'] = flow.start_time if hasattr(flow, 'start_time') else 0
            
            return features
            
        except Exception as e:
            return None
    
    def remove_flow(self, flow_key: str):
        """Rimuove un flusso."""
        if flow_key in self.flows:
            del self.flows[flow_key]
    
    def get_all_flow_keys(self) -> List[str]:
        """Restituisce tutte le chiavi dei flussi."""
        return list(self.flows.keys())


class FallbackFlow:
    """Flow class di fallback se src.sniffer.flow non è disponibile."""
    
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol, timestamp):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.start_time = timestamp
        self.end_time = timestamp
        self.last_timestamp = timestamp
        
        self.packets = []
        self.fwd_lengths = []
        self.bwd_lengths = []
        self.iats = []
        self.fwd_iats = []
        self.bwd_iats = []
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.total_bytes = 0
        self.total_packets = 0
        
        self.init_win_bytes_forward = 0
        self.init_win_bytes_backward = 0
        self.act_data_pkt_fwd = 0
        
        self.active_times = []
        self.idle_times = []
        
        self.flags = defaultdict(int)
        self.fwd_psh = 0
        self.bwd_psh = 0
        self.fwd_urg = 0
        self.bwd_urg = 0
        
        self._last_fwd_time = None
        self._last_bwd_time = None
        self._last_pkt_time = None
    
    def add_packet(self, pkt, timestamp):
        """Aggiunge pacchetto al flusso."""
        if not pkt.haslayer(IP):
            return
        
        ip = pkt[IP]
        size = len(pkt)
        
        self.packets.append((pkt, timestamp))
        self.total_bytes += size
        self.total_packets += 1
        
        # IAT
        if self._last_pkt_time is not None:
            iat = timestamp - self._last_pkt_time
            self.iats.append(iat)
        self._last_pkt_time = timestamp
        
        # Forward/Backward
        is_forward = ip.src == self.src_ip
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += size
            self.fwd_lengths.append(size)
            
            if self._last_fwd_time is not None:
                self.fwd_iats.append(timestamp - self._last_fwd_time)
            self._last_fwd_time = timestamp
            
            if pkt.haslayer(TCP) and self.init_win_bytes_forward == 0:
                self.init_win_bytes_forward = pkt[TCP].window
        else:
            self.bwd_packets += 1
            self.bwd_bytes += size
            self.bwd_lengths.append(size)
            
            if self._last_bwd_time is not None:
                self.bwd_iats.append(timestamp - self._last_bwd_time)
            self._last_bwd_time = timestamp
            
            if pkt.haslayer(TCP) and self.init_win_bytes_backward == 0:
                self.init_win_bytes_backward = pkt[TCP].window
        
        # TCP Flags
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            if tcp.flags.F:
                self.flags['FIN'] += 1
            if tcp.flags.S:
                self.flags['SYN'] += 1
            if tcp.flags.R:
                self.flags['RST'] += 1
            if tcp.flags.P:
                self.flags['PSH'] += 1
                if is_forward:
                    self.fwd_psh += 1
                else:
                    self.bwd_psh += 1
            if tcp.flags.A:
                self.flags['ACK'] += 1
            if tcp.flags.U:
                self.flags['URG'] += 1
                if is_forward:
                    self.fwd_urg += 1
                else:
                    self.bwd_urg += 1
        
        self.end_time = timestamp
        self.last_timestamp = timestamp
    
    @property
    def duration(self):
        return max(self.end_time - self.start_time, 0.000001)


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def process_pcap(
    pcap_path: str,
    day: str,
    output_csv: str,
    max_packets: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Processa PCAP e genera CSV con LE TUE feature.
    """
    day_info = GROUND_TRUTH.get(day)
    if not day_info:
        raise ValueError(f"Unknown day: {day}. Available: {list(GROUND_TRUTH.keys())}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {pcap_path}")
    print(f"Day: {day} ({day_info['date']})")
    print(f"Using SNIFFER extractor: {SNIFFER_AVAILABLE}")
    print(f"Expected attacks: {[a['name'] for a in day_info.get('attacks', [])]}")
    print(f"{'='*60}")
    
    flow_manager = FlowManager(flow_timeout=120.0)
    all_features = []
    
    packet_count = 0
    flow_count = 0
    attack_count = 0
    
    start_time = time.time()
    last_report_time = start_time
    
    # Reference date for timestamp conversion
    ref_date = datetime.strptime(day_info['date'], "%Y-%m-%d")
    
    print("\nProcessing packets...")
    
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            packet_count += 1
            
            if max_packets and packet_count >= max_packets:
                break
            
            # Get timestamp
            try:
                ts = float(pkt.time)
            except Exception:
                continue
            
            # Process packet
            flow_manager.process_packet(pkt, ts)
            
            # Check expired flows periodically
            if packet_count % 10000 == 0:
                expired = flow_manager.get_expired_flows(ts)
                
                for flow_key in expired:
                    features = flow_manager.extract_features(flow_key)
                    
                    if features:
                        # Convert timestamp to datetime for labeling
                        flow_ts = features.get('_timestamp', ts)
                        # The PCAP timestamps are Unix timestamps
                        try:
                            flow_datetime = datetime.fromtimestamp(flow_ts)
                            # Adjust to reference date (keep only time)
                            flow_datetime = ref_date.replace(
                                hour=flow_datetime.hour,
                                minute=flow_datetime.minute,
                                second=flow_datetime.second
                            )
                        except Exception:
                            flow_datetime = ref_date
                        
                        # Label the flow
                        is_attack, attack_name = is_attack_flow(
                            features['_src_ip'],
                            features['_dst_ip'],
                            flow_datetime,
                            day_info
                        )
                        
                        # Remove metadata before saving
                        for key in ['_src_ip', '_dst_ip', '_src_port', '_dst_port', '_timestamp']:
                            features.pop(key, None)
                        
                        features['Label'] = attack_name if is_attack else 'BENIGN'
                        all_features.append(features)
                        
                        if is_attack:
                            attack_count += 1
                        flow_count += 1
                    
                    flow_manager.remove_flow(flow_key)
                
                # Progress report
                if verbose and time.time() - last_report_time > 5:
                    elapsed = time.time() - start_time
                    pkt_rate = packet_count / elapsed
                    print(f"  Packets: {packet_count:,} ({pkt_rate:.0f}/s) | "
                          f"Flows: {flow_count:,} | Attacks: {attack_count:,}")
                    last_report_time = time.time()
    
    # Flush remaining flows
    print("\nFlushing remaining flows...")
    
    for flow_key in flow_manager.get_all_flow_keys():
        features = flow_manager.extract_features(flow_key)
        
        if features:
            flow_ts = features.get('_timestamp', 0)
            try:
                flow_datetime = datetime.fromtimestamp(flow_ts)
                flow_datetime = ref_date.replace(
                    hour=flow_datetime.hour,
                    minute=flow_datetime.minute,
                    second=flow_datetime.second
                )
            except Exception:
                flow_datetime = ref_date
            
            is_attack, attack_name = is_attack_flow(
                features['_src_ip'],
                features['_dst_ip'],
                flow_datetime,
                day_info
            )
            
            for key in ['_src_ip', '_dst_ip', '_src_port', '_dst_port', '_timestamp']:
                features.pop(key, None)
            
            features['Label'] = attack_name if is_attack else 'BENIGN'
            all_features.append(features)
            
            if is_attack:
                attack_count += 1
            flow_count += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"{'='*60}")
    print(f"Packets processed: {packet_count:,}")
    print(f"Flows extracted:   {flow_count:,}")
    print(f"Attack flows:      {attack_count:,} ({attack_count/max(flow_count,1)*100:.2f}%)")
    print(f"Benign flows:      {flow_count - attack_count:,}")
    print(f"Time:              {elapsed:.1f}s")
    print(f"Output:            {output_csv}")
    
    if not df.empty:
        print(f"\nLabel Distribution:")
        for label, count in df['Label'].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")
        
        print(f"\nFeature columns: {len(df.columns) - 1}")  # -1 for Label
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate training dataset from PCAP using YOUR feature extractor'
    )
    parser.add_argument('--day', type=str, 
                        choices=list(GROUND_TRUTH.keys()),
                        help='Day to process')
    parser.add_argument('--pcap', type=str, help='Path to PCAP file')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--max-packets', type=int, default=None, 
                        help='Max packets to process (None for all)')
    parser.add_argument('--all', action='store_true', help='Process all available days')
    parser.add_argument('--pcap-dir', type=str, default='data/pcap', 
                        help='Directory containing PCAP files')
    
    args = parser.parse_args()
    
    if not SNIFFER_AVAILABLE:
        print("\n" + "="*60)
        print("WARNING: Could not import src.sniffer modules!")
        print("Using fallback Flow class - features may differ slightly")
        print("Make sure you're running from the project root directory")
        print("="*60 + "\n")
    
    if args.all:
        pcap_dir = Path(args.pcap_dir)
        output_dir = Path('data/generated')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping day -> PCAP filename
        pcap_mapping = {
            'monday': 'Monday-WorkingHours.pcap',
            'tuesday': 'Tuesday-WorkingHours.pcap',
            'wednesday': 'Wednesday-workingHours.pcap',
            'thursday_morning': 'Thursday-WorkingHours.pcap',
            'thursday_afternoon': 'Thursday-WorkingHours.pcap',
            'friday_morning': 'Friday-WorkingHours.pcap',
            'friday_portscan': 'Friday-WorkingHours.pcap',
            'friday_ddos': 'Friday-WorkingHours.pcap',
        }
        
        results = {}
        
        for day, pcap_name in pcap_mapping.items():
            pcap_path = pcap_dir / pcap_name
            
            if not pcap_path.exists():
                print(f"\nSkipping {day}: PCAP not found ({pcap_path})")
                continue
            
            output_csv = output_dir / f"{day}_generated.csv"
            
            try:
                df = process_pcap(
                    str(pcap_path), 
                    day, 
                    str(output_csv), 
                    args.max_packets
                )
                results[day] = {
                    'flows': len(df),
                    'attacks': (df['Label'] != 'BENIGN').sum() if not df.empty else 0
                }
            except Exception as e:
                print(f"Error processing {day}: {e}")
                import traceback
                traceback.print_exc()
                results[day] = {'error': str(e)}
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for day, res in results.items():
            if 'error' in res:
                print(f"  {day}: ERROR - {res['error']}")
            else:
                print(f"  {day}: {res['flows']:,} flows, {res['attacks']:,} attacks")
        
    else:
        if not args.day or not args.pcap:
            parser.print_help()
            print("\nExamples:")
            print("  python generate_dataset_v2.py --day tuesday --pcap data/pcap/Tuesday-WorkingHours.pcap")
            print("  python generate_dataset_v2.py --all --pcap-dir data/pcap/")
            return
        
        if not Path(args.pcap).exists():
            print(f"ERROR: PCAP not found: {args.pcap}")
            return
        
        output = args.output or f"data/generated/{args.day}_generated.csv"
        
        process_pcap(args.pcap, args.day, output, args.max_packets)


if __name__ == "__main__":
    main()
