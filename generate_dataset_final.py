#!/usr/bin/env python3
"""
NIDS-ML Dataset Generator - VERSIONE FINALE

Correzioni applicate:
1. Offset temporale: -3 ore (UTC → ADT, confermato dalla diagnosi)
2. Margine temporale: 15 minuti prima/dopo (per catturare traffico ai bordi)
3. IP attaccanti: 172.16.0.1 (principale dopo NAT)
4. Usa IL TUO vero FeatureExtractor

Uso:
    python generate_dataset_final.py --day tuesday --pcap data/pcap/Tuesday-WorkingHours.pcap
    python generate_dataset_final.py --all --pcap-dir data/pcap/
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scapy.all import PcapReader, IP, TCP, UDP
except ImportError:
    print("ERROR: scapy not installed. Run: pip install scapy")
    sys.exit(1)

try:
    from src.sniffer.features import FeatureExtractor
    from src.sniffer.flow import Flow
    print("SUCCESS: Imported src.sniffer modules")
    SNIFFER_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import sniffer modules: {e}")
    print("Make sure you're running from the project root directory")
    SNIFFER_AVAILABLE = False
    sys.exit(1)


# ==============================================================================
# CONFIGURAZIONE CORRETTA (basata sulla diagnosi)
# ==============================================================================

# Offset temporale: I PCAP sono in UTC, il ground truth è in ADT (UTC-3)
# Quindi: timestamp_pcap (UTC) - 3 ore = ora locale ADT
TIMEZONE_OFFSET_HOURS = -3

# Margine temporale: 15 minuti prima e dopo per catturare traffico ai bordi
TIME_MARGIN_MINUTES = 15

# Flow timeout
FLOW_TIMEOUT = 120

# IP attaccante principale (dopo NAT nel PCAP interno)
ATTACKER_IP = "172.16.0.1"

# Ground truth CORRETTO basato sul sito ufficiale
# https://www.unb.ca/cic/datasets/ids-2017.html
GROUND_TRUTH = {
    "monday": {
        "date": "2017-07-03",
        "attacks": []
    },
    "tuesday": {
        "date": "2017-07-04",
        "attacks": [
            {"name": "FTP-Patator", "start": "09:20", "end": "10:20", "port": 21},
            {"name": "SSH-Patator", "start": "14:00", "end": "15:00", "port": 22}
        ]
    },
    "wednesday": {
        "date": "2017-07-05",
        "attacks": [
            {"name": "DoS slowloris", "start": "09:47", "end": "10:10"},
            {"name": "DoS Slowhttptest", "start": "10:14", "end": "10:35"},
            {"name": "DoS Hulk", "start": "10:43", "end": "11:00"},
            {"name": "DoS GoldenEye", "start": "11:10", "end": "11:23"},
            {"name": "Heartbleed", "start": "15:12", "end": "15:32"}
        ]
    },
    "thursday": {
        "date": "2017-07-06",
        "attacks": [
            {"name": "Web Attack - Brute Force", "start": "09:20", "end": "10:00"},
            {"name": "Web Attack - XSS", "start": "10:15", "end": "10:35"},
            {"name": "Web Attack - Sql Injection", "start": "10:40", "end": "10:42"},
            {"name": "Infiltration", "start": "14:19", "end": "15:45"}
        ]
    },
    "friday": {
        "date": "2017-07-07",
        "attacks": [
            {"name": "Bot", "start": "10:02", "end": "11:02"},
            {"name": "PortScan", "start": "13:55", "end": "15:29"},
            {"name": "DDoS", "start": "15:56", "end": "16:16"}
        ]
    }
}


# ==============================================================================
# LABELING
# ==============================================================================

def parse_time(time_str: str, date_str: str) -> datetime:
    """Parse HH:MM string con data."""
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")


def get_flow_label(
    src_ip: str,
    dst_ip: str,
    flow_timestamp: float,
    day_info: Dict,
    dst_port: int = 0
) -> str:
    """
    Determina la label di un flusso.
    
    Logica:
    1. Converti timestamp UTC → ora locale (ADT, -3h)
    2. Verifica se coinvolge l'IP attaccante (172.16.0.1)
    3. Verifica se rientra nella finestra temporale (con margine 15min)
    """
    attacks = day_info.get('attacks', [])
    if not attacks:
        return "BENIGN"
    
    day_date = day_info.get('date')
    if not day_date:
        return "BENIGN"
    
    # Verifica se coinvolge l'attaccante
    involves_attacker = (src_ip == ATTACKER_IP or dst_ip == ATTACKER_IP)
    if not involves_attacker:
        return "BENIGN"
    
    # Converti timestamp UTC → ora locale ADT
    try:
        # Il timestamp nel PCAP è in UTC
        flow_dt_utc = datetime.utcfromtimestamp(flow_timestamp)
        # Applica offset per ottenere ora locale ADT
        flow_dt_local = flow_dt_utc + timedelta(hours=TIMEZONE_OFFSET_HOURS)
        
        # Sincronizza la data con il ground truth
        ref_date = datetime.strptime(day_date, "%Y-%m-%d")
        flow_dt = ref_date.replace(
            hour=flow_dt_local.hour,
            minute=flow_dt_local.minute,
            second=flow_dt_local.second
        )
    except Exception:
        return "BENIGN"
    
    # Controlla ogni attacco
    for attack in attacks:
        start_str = attack.get('start')
        end_str = attack.get('end')
        label = attack.get('name')
        attack_port = attack.get('port')
        
        if not all([start_str, end_str, label]):
            continue
        
        # Finestra temporale con margine
        try:
            start = parse_time(start_str, day_date) - timedelta(minutes=TIME_MARGIN_MINUTES)
            end = parse_time(end_str, day_date) + timedelta(minutes=TIME_MARGIN_MINUTES)
            
            if start <= flow_dt <= end:
                # Se l'attacco ha una porta specifica, verifica anche quella
                if attack_port and dst_port > 0:
                    if dst_port == attack_port:
                        return label
                    # Se la porta non matcha, potrebbe essere un altro attacco
                    continue
                return label
        except Exception:
            continue
    
    return "BENIGN"


# ==============================================================================
# FLOW MANAGER
# ==============================================================================

class FlowManager:
    """Gestisce i flussi usando IL TUO Flow e FeatureExtractor."""
    
    def __init__(self, timeout: float = FLOW_TIMEOUT):
        self.flows = {}
        self.timeout = timeout
        self.feature_extractor = FeatureExtractor()
    
    def _get_flow_key(self, pkt):
        try:
            if not pkt.haslayer(IP):
                return None
            
            ip = pkt[IP]
            src_ip, dst_ip, proto = ip.src, ip.dst, ip.proto
            src_port, dst_port = 0, 0
            
            if pkt.haslayer(TCP):
                src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
            elif pkt.haslayer(UDP):
                src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
            
            # Chiave bidirezionale
            if (src_ip, src_port) < (dst_ip, dst_port):
                return (src_ip, dst_ip, src_port, dst_port, proto)
            else:
                return (dst_ip, src_ip, dst_port, src_port, proto)
        except Exception:
            return None
    
    def process_packet(self, pkt, timestamp: float):
        key = self._get_flow_key(pkt)
        if not key:
            return
        
        if key not in self.flows:
            self.flows[key] = Flow(key[0], key[1], key[2], key[3], key[4])
        
        try:
            self.flows[key].add_packet(pkt, timestamp)
        except Exception:
            pass
    
    def get_expired_flows(self, current_time: float) -> List[tuple]:
        expired = []
        for key, flow in self.flows.items():
            try:
                last_ts = getattr(flow, 'last_timestamp', 0) or getattr(flow, 'end_time', 0)
                if current_time - last_ts > self.timeout:
                    expired.append(key)
            except Exception:
                expired.append(key)
        return expired
    
    def extract_features(self, key) -> Optional[Dict]:
        if key not in self.flows:
            return None
        
        flow = self.flows[key]
        try:
            features = self.feature_extractor.extract(flow)
            features['_src_ip'] = flow.src_ip
            features['_dst_ip'] = flow.dst_ip
            features['_src_port'] = flow.src_port
            features['_dst_port'] = flow.dst_port
            features['_timestamp'] = getattr(flow, 'start_time', 0)
            return features
        except Exception:
            return None
    
    def remove_flow(self, key):
        if key in self.flows:
            del self.flows[key]
    
    def get_all_keys(self):
        return list(self.flows.keys())


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def find_pcap(pcap_dir: str, day: str) -> Optional[Path]:
    """Trova il file PCAP per un giorno."""
    patterns = {
        "monday": "Monday",
        "tuesday": "Tuesday",
        "wednesday": "Wednesday",
        "thursday": "Thursday",
        "friday": "Friday"
    }
    
    pattern = patterns.get(day, day.title())
    pcap_path = Path(pcap_dir)
    
    for f in pcap_path.rglob("*.pcap"):
        if pattern.lower() in f.name.lower():
            return f
    return None


def process_pcap(
    pcap_path: str,
    day: str,
    output_csv: str,
    max_packets: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Processa un PCAP e genera CSV con label."""
    
    day_info = GROUND_TRUTH.get(day)
    if not day_info:
        raise ValueError(f"Unknown day: {day}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {day.upper()}")
    print(f"{'='*60}")
    print(f"PCAP: {pcap_path}")
    print(f"Date: {day_info.get('date')}")
    print(f"Timezone offset: {TIMEZONE_OFFSET_HOURS}h (UTC → ADT)")
    print(f"Time margin: ±{TIME_MARGIN_MINUTES} min")
    print(f"Attacker IP: {ATTACKER_IP}")
    
    attacks = day_info.get('attacks', [])
    print(f"Expected attacks: {[a.get('name') for a in attacks]}")
    
    manager = FlowManager()
    all_features = []
    label_counts = Counter()
    
    packet_count = 0
    flow_count = 0
    attacker_packets = 0
    
    start_time = time.time()
    last_report = start_time
    
    print("\nProcessing packets...")
    
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            packet_count += 1
            
            if max_packets and packet_count >= max_packets:
                break
            
            try:
                ts = float(pkt.time)
            except Exception:
                continue
            
            # Conta pacchetti dell'attaccante
            if pkt.haslayer(IP):
                if pkt[IP].src == ATTACKER_IP or pkt[IP].dst == ATTACKER_IP:
                    attacker_packets += 1
            
            manager.process_packet(pkt, ts)
            
            # Garbage collection ogni 20k pacchetti
            if packet_count % 20000 == 0:
                expired = manager.get_expired_flows(ts)
                
                for key in expired:
                    features = manager.extract_features(key)
                    
                    if features:
                        src_ip = features.pop('_src_ip')
                        dst_ip = features.pop('_dst_ip')
                        src_port = features.pop('_src_port')
                        dst_port = features.pop('_dst_port')
                        flow_ts = features.pop('_timestamp')
                        
                        label = get_flow_label(src_ip, dst_ip, flow_ts, day_info, dst_port)
                        features['Label'] = label
                        all_features.append(features)
                        
                        label_counts[label] += 1
                        flow_count += 1
                    
                    manager.remove_flow(key)
                
                # Progress report
                if verbose and time.time() - last_report > 10:
                    elapsed = time.time() - start_time
                    attack_count = sum(v for k, v in label_counts.items() if k != 'BENIGN')
                    print(f"  {packet_count:,} pkts | {flow_count:,} flows | "
                          f"{attack_count:,} attacks | {attacker_packets:,} attacker pkts")
                    last_report = time.time()
    
    # Flush remaining flows
    print("\nFlushing remaining flows...")
    for key in manager.get_all_keys():
        features = manager.extract_features(key)
        if features:
            src_ip = features.pop('_src_ip')
            dst_ip = features.pop('_dst_ip')
            src_port = features.pop('_src_port')
            dst_port = features.pop('_dst_port')
            flow_ts = features.pop('_timestamp')
            
            label = get_flow_label(src_ip, dst_ip, flow_ts, day_info, dst_port)
            features['Label'] = label
            all_features.append(features)
            
            label_counts[label] += 1
            flow_count += 1
    
    # Crea DataFrame
    df = pd.DataFrame(all_features)
    
    # Salva
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    attack_count = sum(v for k, v in label_counts.items() if k != 'BENIGN')
    
    print(f"\n{'='*60}")
    print(f"COMPLETED: {day.upper()}")
    print(f"{'='*60}")
    print(f"Packets processed: {packet_count:,}")
    print(f"Attacker packets:  {attacker_packets:,}")
    print(f"Flows extracted:   {flow_count:,}")
    print(f"Attack flows:      {attack_count:,} ({attack_count/max(flow_count,1)*100:.2f}%)")
    print(f"Time:              {elapsed:.1f}s")
    print(f"Output:            {output_csv}")
    
    print(f"\nLabel Distribution:")
    for label, count in label_counts.most_common():
        pct = count / max(flow_count, 1) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")
    
    # Diagnostica
    if attack_count == 0 and attacks:
        print("\n" + "!"*60)
        print("WARNING: Zero attacks labeled!")
        print(f"Attacker packets in PCAP: {attacker_packets:,}")
        if attacker_packets < 100:
            print("DIAGNOSIS: Very few attacker packets - PCAP may be incomplete")
            print("           or traffic is on different IPs after NAT")
        print("!"*60)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset with correct timezone and margin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_dataset_final.py --day tuesday --pcap data/pcap/Tuesday-WorkingHours.pcap
    python generate_dataset_final.py --all --pcap-dir data/pcap/
        """
    )
    
    parser.add_argument('--day', type=str, 
                        choices=['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])
    parser.add_argument('--pcap', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--max-packets', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--pcap-dir', type=str, default='data/pcap')
    parser.add_argument('--output-dir', type=str, default='data/generated')
    
    args = parser.parse_args()
    
    if args.all:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            pcap_file = find_pcap(args.pcap_dir, day)
            if not pcap_file:
                print(f"\nSkipping {day}: PCAP not found")
                continue
            
            output_csv = output_dir / f"{day}_generated.csv"
            
            try:
                df = process_pcap(str(pcap_file), day, str(output_csv), args.max_packets)
                attack_count = (df['Label'] != 'BENIGN').sum() if not df.empty else 0
                results[day] = {'flows': len(df), 'attacks': attack_count}
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        total_f, total_a = 0, 0
        for day, res in results.items():
            print(f"  {day}: {res['flows']:,} flows, {res['attacks']:,} attacks")
            total_f += res['flows']
            total_a += res['attacks']
        print(f"\n  TOTAL: {total_f:,} flows, {total_a:,} attacks")
    
    else:
        if not args.day:
            parser.print_help()
            return
        
        pcap_path = args.pcap or find_pcap(args.pcap_dir, args.day)
        if not pcap_path or not Path(pcap_path).exists():
            print(f"ERROR: PCAP not found for {args.day}")
            return
        
        output = args.output or f"{args.output_dir}/{args.day}_generated.csv"
        process_pcap(str(pcap_path), args.day, output, args.max_packets)


if __name__ == "__main__":
    main()
