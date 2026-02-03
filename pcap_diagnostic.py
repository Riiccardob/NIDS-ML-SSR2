#!/usr/bin/env python3
"""
PCAP Diagnostic - Analizza cosa c'è REALMENTE nel PCAP

Questo script:
1. Legge i primi N pacchetti del PCAP
2. Mostra gli IP più frequenti
3. Mostra i timestamp (per capire il timezone)
4. Cerca gli IP attaccanti noti

Questo ci dirà SE gli IP attaccanti sono presenti e QUANDO.
"""

import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from scapy.all import PcapReader, IP, TCP, UDP
except ImportError:
    print("pip install scapy")
    sys.exit(1)

# IP attaccanti noti dal sito ufficiale CIC
KNOWN_ATTACKERS = {
    "205.174.165.73",   # Kali
    "172.16.0.1",       # Firewall/NAT
    "205.174.165.69",   # DDoS
    "205.174.165.70",   # DDoS
    "205.174.165.71",   # DDoS
    "192.168.10.8",     # Compromised
}

KNOWN_VICTIMS = {
    "192.168.10.50",    # Web server
    "192.168.10.51",    # Ubuntu server
}


def analyze_pcap(pcap_path: str, max_packets: int = 500000):
    print(f"Analyzing: {pcap_path}")
    print(f"Max packets: {max_packets:,}")
    print("="*60)
    
    ip_counter = Counter()
    src_ip_counter = Counter()
    dst_ip_counter = Counter()
    port_counter = Counter()
    
    attacker_flows = defaultdict(list)  # IP -> [(timestamp, dst_ip, dst_port)]
    
    timestamps = []
    packet_count = 0
    
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            packet_count += 1
            
            if packet_count > max_packets:
                break
            
            if not pkt.haslayer(IP):
                continue
            
            ip = pkt[IP]
            src = ip.src
            dst = ip.dst
            
            ip_counter[src] += 1
            ip_counter[dst] += 1
            src_ip_counter[src] += 1
            dst_ip_counter[dst] += 1
            
            # Get timestamp
            try:
                ts = float(pkt.time)
                if packet_count <= 10 or packet_count % 100000 == 0:
                    timestamps.append((packet_count, ts))
            except:
                pass
            
            # Track attacker flows
            if src in KNOWN_ATTACKERS:
                dst_port = 0
                if pkt.haslayer(TCP):
                    dst_port = pkt[TCP].dport
                elif pkt.haslayer(UDP):
                    dst_port = pkt[UDP].dport
                attacker_flows[src].append((ts, dst, dst_port))
            
            # Get ports
            if pkt.haslayer(TCP):
                port_counter[pkt[TCP].dport] += 1
            elif pkt.haslayer(UDP):
                port_counter[pkt[UDP].dport] += 1
            
            if packet_count % 100000 == 0:
                print(f"  Processed {packet_count:,} packets...")
    
    print(f"\nTotal packets analyzed: {packet_count:,}")
    
    # Timestamp analysis
    print("\n" + "="*60)
    print("TIMESTAMP ANALYSIS")
    print("="*60)
    
    if timestamps:
        print("\nSample timestamps:")
        for pkt_num, ts in timestamps[:5]:
            dt = datetime.fromtimestamp(ts)
            dt_utc = datetime.utcfromtimestamp(ts)
            print(f"  Packet {pkt_num}: {ts}")
            print(f"    Local: {dt}")
            print(f"    UTC:   {dt_utc}")
    
    # IP analysis
    print("\n" + "="*60)
    print("TOP 20 IPs (by packet count)")
    print("="*60)
    
    for ip, count in ip_counter.most_common(20):
        marker = ""
        if ip in KNOWN_ATTACKERS:
            marker = " <-- ATTACKER"
        elif ip in KNOWN_VICTIMS:
            marker = " <-- VICTIM"
        print(f"  {ip}: {count:,}{marker}")
    
    # Attacker presence
    print("\n" + "="*60)
    print("ATTACKER IP PRESENCE")
    print("="*60)
    
    for attacker in KNOWN_ATTACKERS:
        count = ip_counter.get(attacker, 0)
        src_count = src_ip_counter.get(attacker, 0)
        dst_count = dst_ip_counter.get(attacker, 0)
        
        if count > 0:
            print(f"  {attacker}: {count:,} packets (src: {src_count:,}, dst: {dst_count:,}) FOUND!")
        else:
            print(f"  {attacker}: NOT FOUND")
    
    # Attacker flow details
    print("\n" + "="*60)
    print("ATTACKER FLOW DETAILS")
    print("="*60)
    
    for attacker, flows in attacker_flows.items():
        if flows:
            print(f"\n{attacker} ({len(flows):,} packets as source):")
            
            # Show timestamp range
            ts_list = [f[0] for f in flows]
            min_ts = min(ts_list)
            max_ts = max(ts_list)
            
            print(f"  Time range: {datetime.fromtimestamp(min_ts)} - {datetime.fromtimestamp(max_ts)}")
            
            # Show target ports
            ports = Counter(f[2] for f in flows)
            print(f"  Top target ports: {ports.most_common(5)}")
            
            # Show target IPs
            targets = Counter(f[1] for f in flows)
            print(f"  Top targets: {targets.most_common(5)}")
    
    # Port analysis
    print("\n" + "="*60)
    print("TOP 20 DESTINATION PORTS")
    print("="*60)
    
    for port, count in port_counter.most_common(20):
        marker = ""
        if port == 21:
            marker = " <-- FTP"
        elif port == 22:
            marker = " <-- SSH"
        elif port == 80:
            marker = " <-- HTTP"
        elif port == 443:
            marker = " <-- HTTPS"
        print(f"  Port {port}: {count:,}{marker}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    attackers_found = [ip for ip in KNOWN_ATTACKERS if ip_counter.get(ip, 0) > 0]
    
    if attackers_found:
        print(f"Attacker IPs found: {attackers_found}")
        print("\nThe attacker IPs ARE in the PCAP.")
        print("If generate_dataset finds 0 attacks, the problem is:")
        print("  1. Timestamp/timezone mismatch")
        print("  2. Flow aggregation creating different flow keys")
        print("  3. Labeling logic issue")
    else:
        print("NO ATTACKER IPs FOUND IN PCAP!")
        print("\nThis means the PCAP has been NAT'd or the IPs are different.")
        print("Check if traffic uses internal IPs only (192.168.x.x)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pcap_diagnostic.py <pcap_file> [max_packets]")
        print("Example: python pcap_diagnostic.py data/pcap/Tuesday-WorkingHours.pcap 500000")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    max_pkts = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
    
    analyze_pcap(pcap_file, max_pkts)
