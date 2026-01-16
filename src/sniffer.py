"""
================================================================================
NIDS-ML - Network Sniffer
================================================================================

Cattura pacchetti di rete, estrae feature e classifica traffico in tempo reale.

REQUISITI:
----------
- Esecuzione come root (sudo) per cattura pacchetti
- Scapy installato
- Modello addestrato disponibile

GUIDA PARAMETRI:
----------------
    sudo python src/sniffer.py [opzioni]

Opzioni:
    --model-path PATH     Path al modello .pkl (obbligatorio)
    --interface STR       Interfaccia di rete (default: auto-detect)
    --timeout INT         Timeout flow in secondi (default: 60)
    --threshold FLOAT     Soglia probabilita per alert (default: 0.5)
    --log-file PATH       File log attacchi (default: logs/attacks.log)
    --packet-count INT    Numero pacchetti da catturare (default: 0 = infinito)
    --verbose             Mostra tutti i flussi (non solo attacchi)

ESEMPI:
-------
# Sniffer base con Random Forest
sudo python src/sniffer.py --model-path models/random_forest/model_binary.pkl

# Specifica interfaccia
sudo python src/sniffer.py --model-path models/xgboost/model_binary.pkl --interface eth0

# Cattura 1000 pacchetti poi esci
sudo python src/sniffer.py --model-path models/lightgbm/model_binary.pkl --packet-count 1000

# Modalita verbose (mostra anche traffico benigno)
sudo python src/sniffer.py --model-path models/random_forest/model_binary.pkl --verbose

================================================================================

ARCHITETTURA:
-------------
1. Scapy cattura pacchetti raw
2. Pacchetti vengono aggregati in flussi (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol)
3. Quando un flusso scade (timeout) o raggiunge soglia pacchetti, viene estratto feature vector
4. Feature vector viene scalato e passato al modello
5. Se classificato come attacco, genera alert

LIMITAZIONI:
------------
- Le feature estratte sono un sottoinsieme di quelle del dataset CIC-IDS2017
- Alcune feature temporali (IAT) richiedono piu pacchetti per essere accurate
- Performance dipende dal carico di rete

================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
import signal
import json

# Setup path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib

from src.utils import get_logger, get_project_root, suppress_warnings
from src.feature_engineering import load_artifacts

suppress_warnings()

# Check root privileges
if os.geteuid() != 0:
    print("ATTENZIONE: Lo sniffer richiede privilegi root per catturare pacchetti.")
    print("Eseguire con: sudo python src/sniffer.py ...")

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("ERRORE: Scapy non installato. Eseguire: pip install scapy")


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

DEFAULT_TIMEOUT = 60  # Secondi prima che un flusso scada
DEFAULT_THRESHOLD = 0.5  # Soglia probabilita per alert
MAX_PACKETS_PER_FLOW = 1000  # Max pacchetti prima di forzare analisi


# ==============================================================================
# FLOW MANAGER
# ==============================================================================

class Flow:
    """
    Rappresenta un flusso di rete (aggregazione di pacchetti).
    
    Un flusso e identificato dalla 5-tupla:
    (src_ip, dst_ip, src_port, dst_port, protocol)
    """
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        # Timestamp
        self.start_time = time.time()
        self.last_time = self.start_time
        
        # Contatori pacchetti
        self.fwd_packets = 0  # Pacchetti forward (src -> dst)
        self.bwd_packets = 0  # Pacchetti backward (dst -> src)
        
        # Contatori bytes
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        
        # Liste lunghezze pacchetti
        self.fwd_lengths: List[int] = []
        self.bwd_lengths: List[int] = []
        
        # Inter-arrival times
        self.fwd_iats: List[float] = []
        self.bwd_iats: List[float] = []
        
        # Ultimo timestamp per direzione
        self.last_fwd_time = None
        self.last_bwd_time = None
        
        # Flag TCP
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0
    
    @property
    def flow_id(self) -> str:
        """Identificatore univoco del flusso."""
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}:{self.protocol}"
    
    @property
    def duration(self) -> float:
        """Durata del flusso in secondi."""
        return self.last_time - self.start_time
    
    @property
    def total_packets(self) -> int:
        """Numero totale di pacchetti."""
        return self.fwd_packets + self.bwd_packets
    
    def add_packet(self, packet_len: int, is_forward: bool, timestamp: float, tcp_flags: dict = None):
        """
        Aggiunge un pacchetto al flusso.
        
        Args:
            packet_len: Lunghezza pacchetto in bytes
            is_forward: True se pacchetto va da src a dst
            timestamp: Timestamp del pacchetto
            tcp_flags: Dizionario flag TCP (opzionale)
        """
        self.last_time = timestamp
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
            self.fwd_lengths.append(packet_len)
            
            if self.last_fwd_time is not None:
                iat = timestamp - self.last_fwd_time
                self.fwd_iats.append(iat)
            self.last_fwd_time = timestamp
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len
            self.bwd_lengths.append(packet_len)
            
            if self.last_bwd_time is not None:
                iat = timestamp - self.last_bwd_time
                self.bwd_iats.append(iat)
            self.last_bwd_time = timestamp
        
        # Conta flag TCP
        if tcp_flags:
            if tcp_flags.get('F'): self.fin_count += 1
            if tcp_flags.get('S'): self.syn_count += 1
            if tcp_flags.get('R'): self.rst_count += 1
            if tcp_flags.get('P'): self.psh_count += 1
            if tcp_flags.get('A'): self.ack_count += 1
            if tcp_flags.get('U'): self.urg_count += 1
    
    def extract_features(self) -> Dict[str, float]:
        """
        Estrae feature vector dal flusso.
        
        Restituisce un dizionario con le feature compatibili con il modello.
        Alcune feature del dataset originale non sono estraibili in tempo reale.
        """
        features = {}
        
        # Durata
        features['Flow Duration'] = self.duration * 1e6  # Converti in microsecondi
        
        # Contatori pacchetti
        features['Total Fwd Packets'] = self.fwd_packets
        features['Total Backward Packets'] = self.bwd_packets
        
        # Bytes
        features['Total Length of Fwd Packets'] = self.fwd_bytes
        features['Total Length of Bwd Packets'] = self.bwd_bytes
        
        # Statistiche lunghezza pacchetti forward
        if self.fwd_lengths:
            features['Fwd Packet Length Max'] = max(self.fwd_lengths)
            features['Fwd Packet Length Min'] = min(self.fwd_lengths)
            features['Fwd Packet Length Mean'] = np.mean(self.fwd_lengths)
            features['Fwd Packet Length Std'] = np.std(self.fwd_lengths) if len(self.fwd_lengths) > 1 else 0
        else:
            features['Fwd Packet Length Max'] = 0
            features['Fwd Packet Length Min'] = 0
            features['Fwd Packet Length Mean'] = 0
            features['Fwd Packet Length Std'] = 0
        
        # Statistiche lunghezza pacchetti backward
        if self.bwd_lengths:
            features['Bwd Packet Length Max'] = max(self.bwd_lengths)
            features['Bwd Packet Length Min'] = min(self.bwd_lengths)
            features['Bwd Packet Length Mean'] = np.mean(self.bwd_lengths)
            features['Bwd Packet Length Std'] = np.std(self.bwd_lengths) if len(self.bwd_lengths) > 1 else 0
        else:
            features['Bwd Packet Length Max'] = 0
            features['Bwd Packet Length Min'] = 0
            features['Bwd Packet Length Mean'] = 0
            features['Bwd Packet Length Std'] = 0
        
        # Flow rates
        if self.duration > 0:
            features['Flow Bytes/s'] = (self.fwd_bytes + self.bwd_bytes) / self.duration
            features['Flow Packets/s'] = self.total_packets / self.duration
        else:
            features['Flow Bytes/s'] = 0
            features['Flow Packets/s'] = 0
        
        # IAT statistics
        all_iats = self.fwd_iats + self.bwd_iats
        if all_iats:
            features['Flow IAT Mean'] = np.mean(all_iats) * 1e6
            features['Flow IAT Std'] = np.std(all_iats) * 1e6 if len(all_iats) > 1 else 0
            features['Flow IAT Max'] = max(all_iats) * 1e6
            features['Flow IAT Min'] = min(all_iats) * 1e6
        else:
            features['Flow IAT Mean'] = 0
            features['Flow IAT Std'] = 0
            features['Flow IAT Max'] = 0
            features['Flow IAT Min'] = 0
        
        # Forward IAT
        if self.fwd_iats:
            features['Fwd IAT Total'] = sum(self.fwd_iats) * 1e6
            features['Fwd IAT Mean'] = np.mean(self.fwd_iats) * 1e6
            features['Fwd IAT Std'] = np.std(self.fwd_iats) * 1e6 if len(self.fwd_iats) > 1 else 0
            features['Fwd IAT Max'] = max(self.fwd_iats) * 1e6
            features['Fwd IAT Min'] = min(self.fwd_iats) * 1e6
        else:
            features['Fwd IAT Total'] = 0
            features['Fwd IAT Mean'] = 0
            features['Fwd IAT Std'] = 0
            features['Fwd IAT Max'] = 0
            features['Fwd IAT Min'] = 0
        
        # Backward IAT
        if self.bwd_iats:
            features['Bwd IAT Total'] = sum(self.bwd_iats) * 1e6
            features['Bwd IAT Mean'] = np.mean(self.bwd_iats) * 1e6
            features['Bwd IAT Std'] = np.std(self.bwd_iats) * 1e6 if len(self.bwd_iats) > 1 else 0
            features['Bwd IAT Max'] = max(self.bwd_iats) * 1e6
            features['Bwd IAT Min'] = min(self.bwd_iats) * 1e6
        else:
            features['Bwd IAT Total'] = 0
            features['Bwd IAT Mean'] = 0
            features['Bwd IAT Std'] = 0
            features['Bwd IAT Max'] = 0
            features['Bwd IAT Min'] = 0
        
        # TCP Flags
        features['FIN Flag Count'] = self.fin_count
        features['SYN Flag Count'] = self.syn_count
        features['RST Flag Count'] = self.rst_count
        features['PSH Flag Count'] = self.psh_count
        features['ACK Flag Count'] = self.ack_count
        features['URG Flag Count'] = self.urg_count
        
        # Packet length statistics (combined)
        all_lengths = self.fwd_lengths + self.bwd_lengths
        if all_lengths:
            features['Max Packet Length'] = max(all_lengths)
            features['Min Packet Length'] = min(all_lengths)
            features['Packet Length Mean'] = np.mean(all_lengths)
            features['Packet Length Std'] = np.std(all_lengths) if len(all_lengths) > 1 else 0
            features['Packet Length Variance'] = np.var(all_lengths) if len(all_lengths) > 1 else 0
        else:
            features['Max Packet Length'] = 0
            features['Min Packet Length'] = 0
            features['Packet Length Mean'] = 0
            features['Packet Length Std'] = 0
            features['Packet Length Variance'] = 0
        
        # Average segment size
        features['Avg Fwd Segment Size'] = features['Fwd Packet Length Mean']
        features['Avg Bwd Segment Size'] = features['Bwd Packet Length Mean']
        
        # Subflow
        features['Subflow Fwd Packets'] = self.fwd_packets
        features['Subflow Fwd Bytes'] = self.fwd_bytes
        features['Subflow Bwd Packets'] = self.bwd_packets
        features['Subflow Bwd Bytes'] = self.bwd_bytes
        
        # Average packet size
        if self.total_packets > 0:
            features['Average Packet Size'] = (self.fwd_bytes + self.bwd_bytes) / self.total_packets
        else:
            features['Average Packet Size'] = 0
        
        # Down/Up ratio
        if self.fwd_packets > 0:
            features['Down/Up Ratio'] = self.bwd_packets / self.fwd_packets
        else:
            features['Down/Up Ratio'] = 0
        
        return features


class FlowManager:
    """Gestisce tutti i flussi attivi."""
    
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.flows: Dict[str, Flow] = {}
        self.timeout = timeout
        self.lock = threading.Lock()
    
    def get_flow_key(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int) -> Tuple[str, bool]:
        """
        Genera chiave flusso e determina direzione.
        
        Returns:
            Tuple (flow_key, is_forward)
        """
        # Chiave normalizzata (ordine canonico)
        if (src_ip, src_port) < (dst_ip, dst_port):
            key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
            is_forward = True
        else:
            key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
            is_forward = False
        
        return key, is_forward
    
    def add_packet(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                   protocol: int, packet_len: int, timestamp: float, tcp_flags: dict = None) -> Optional[Flow]:
        """
        Aggiunge pacchetto al flusso appropriato.
        
        Returns:
            Flow se ha raggiunto max pacchetti (da analizzare), None altrimenti
        """
        key, is_forward = self.get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
        
        with self.lock:
            if key not in self.flows:
                # Crea nuovo flusso (usa ordine canonico)
                if is_forward:
                    self.flows[key] = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                else:
                    self.flows[key] = Flow(dst_ip, src_ip, dst_port, src_port, protocol)
            
            flow = self.flows[key]
            flow.add_packet(packet_len, is_forward, timestamp, tcp_flags)
            
            # Se raggiunto max pacchetti, restituisci per analisi
            if flow.total_packets >= MAX_PACKETS_PER_FLOW:
                del self.flows[key]
                return flow
        
        return None
    
    def get_expired_flows(self) -> List[Flow]:
        """Restituisce e rimuove flussi scaduti."""
        expired = []
        current_time = time.time()
        
        with self.lock:
            expired_keys = [
                key for key, flow in self.flows.items()
                if current_time - flow.last_time > self.timeout
            ]
            
            for key in expired_keys:
                expired.append(self.flows.pop(key))
        
        return expired
    
    def get_all_flows(self) -> List[Flow]:
        """Restituisce e svuota tutti i flussi (per shutdown)."""
        with self.lock:
            flows = list(self.flows.values())
            self.flows.clear()
        return flows


# ==============================================================================
# SNIFFER
# ==============================================================================

class NIDSSniffer:
    """
    Network Intrusion Detection Sniffer.
    
    Cattura pacchetti, estrae feature e classifica traffico.
    """
    
    def __init__(self,
                 model_path: Path,
                 interface: str = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 threshold: float = DEFAULT_THRESHOLD,
                 log_file: Path = None,
                 verbose: bool = False):
        """
        Inizializza sniffer.
        
        Args:
            model_path: Path al modello addestrato
            interface: Interfaccia di rete (None = auto)
            timeout: Timeout flussi in secondi
            threshold: Soglia probabilita per alert
            log_file: File per log attacchi
            verbose: Se True, mostra tutti i flussi
        """
        self.interface = interface
        self.timeout = timeout
        self.threshold = threshold
        self.verbose = verbose
        
        # Setup logging
        if log_file is None:
            log_file = get_project_root() / "logs" / "attacks.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file
        self.logger = get_logger("sniffer", str(log_file))
        
        # Carica modello e artifacts
        self.logger.info(f"Caricamento modello: {model_path}")
        self.model = joblib.load(model_path)
        
        self.logger.info("Caricamento artifacts...")
        self.scaler, self.selected_features, _ = load_artifacts()
        
        # Flow manager
        self.flow_manager = FlowManager(timeout=timeout)
        
        # Statistiche
        self.stats = {
            'packets_captured': 0,
            'flows_analyzed': 0,
            'attacks_detected': 0,
            'benign_flows': 0,
            'start_time': None
        }
        
        # Flag per shutdown
        self.running = False
    
    def packet_callback(self, packet):
        """Callback per ogni pacchetto catturato."""
        self.stats['packets_captured'] += 1
        
        if not packet.haslayer(IP):
            return
        
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol = ip_layer.proto
        packet_len = len(packet)
        timestamp = time.time()
        
        # Estrai porte e flag TCP
        src_port = 0
        dst_port = 0
        tcp_flags = None
        
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            tcp_flags = {
                'F': bool(tcp_layer.flags & 0x01),  # FIN
                'S': bool(tcp_layer.flags & 0x02),  # SYN
                'R': bool(tcp_layer.flags & 0x04),  # RST
                'P': bool(tcp_layer.flags & 0x08),  # PSH
                'A': bool(tcp_layer.flags & 0x10),  # ACK
                'U': bool(tcp_layer.flags & 0x20),  # URG
            }
        elif packet.haslayer(UDP):
            udp_layer = packet[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
        
        # Aggiungi pacchetto al flow manager
        completed_flow = self.flow_manager.add_packet(
            src_ip, dst_ip, src_port, dst_port,
            protocol, packet_len, timestamp, tcp_flags
        )
        
        # Analizza flusso completato
        if completed_flow:
            self.analyze_flow(completed_flow)
    
    def analyze_flow(self, flow: Flow):
        """Analizza un flusso e classifica."""
        self.stats['flows_analyzed'] += 1
        
        # Estrai feature
        features = flow.extract_features()
        
        # Crea DataFrame con le feature selezionate
        # Aggiungi feature mancanti con valore 0
        feature_dict = {}
        for feat in self.selected_features:
            feature_dict[feat] = features.get(feat, 0)
        
        df = pd.DataFrame([feature_dict])
        
        # Scala feature
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns
        )
        
        # Predizione
        prediction = self.model.predict(df_scaled)[0]
        
        # Probabilita (se disponibile)
        prob = None
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(df_scaled)[0][1]
        
        # Classifica
        is_attack = prediction == 1
        
        if is_attack:
            self.stats['attacks_detected'] += 1
            self.alert(flow, prob)
        else:
            self.stats['benign_flows'] += 1
            if self.verbose:
                print(f"[BENIGN] {flow.flow_id} | Packets: {flow.total_packets} | "
                      f"Prob: {prob:.3f if prob else 'N/A'}")
    
    def alert(self, flow: Flow, probability: float = None):
        """Genera alert per flusso malevolo."""
        alert_msg = (f"[ALERT] Possibile attacco rilevato!\n"
                     f"  Flow: {flow.flow_id}\n"
                     f"  Packets: {flow.total_packets}\n"
                     f"  Duration: {flow.duration:.2f}s\n"
                     f"  Bytes: {flow.fwd_bytes + flow.bwd_bytes}\n"
                     f"  Probability: {probability:.3f if probability else 'N/A'}")
        
        print("\n" + "!" * 60)
        print(alert_msg)
        print("!" * 60 + "\n")
        
        # Log su file
        self.logger.warning(alert_msg.replace('\n', ' | '))
    
    def check_expired_flows(self):
        """Thread per controllo flussi scaduti."""
        while self.running:
            time.sleep(5)  # Check ogni 5 secondi
            
            expired = self.flow_manager.get_expired_flows()
            for flow in expired:
                if flow.total_packets >= 3:  # Analizza solo flussi con almeno 3 pacchetti
                    self.analyze_flow(flow)
    
    def print_stats(self):
        """Stampa statistiche correnti."""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        print(f"\n--- Statistiche ---")
        print(f"Tempo: {elapsed:.1f}s")
        print(f"Pacchetti catturati: {self.stats['packets_captured']:,}")
        print(f"Flussi analizzati: {self.stats['flows_analyzed']:,}")
        print(f"Attacchi rilevati: {self.stats['attacks_detected']:,}")
        print(f"Flussi benigni: {self.stats['benign_flows']:,}")
        if elapsed > 0:
            print(f"Pacchetti/s: {self.stats['packets_captured']/elapsed:.1f}")
    
    def start(self, packet_count: int = 0):
        """
        Avvia lo sniffer.
        
        Args:
            packet_count: Numero pacchetti da catturare (0 = infinito)
        """
        if not SCAPY_AVAILABLE:
            print("ERRORE: Scapy non disponibile")
            return
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Determina interfaccia
        if self.interface is None:
            # Auto-detect: prendi prima interfaccia non-loopback
            interfaces = get_if_list()
            for iface in interfaces:
                if iface != 'lo' and not iface.startswith('docker'):
                    self.interface = iface
                    break
            
            if self.interface is None:
                self.interface = conf.iface
        
        print(f"\nAvvio sniffer su interfaccia: {self.interface}")
        print(f"Timeout flussi: {self.timeout}s")
        print(f"Soglia alert: {self.threshold}")
        print(f"Log file: {self.log_file}")
        print("\nPremi Ctrl+C per fermare...\n")
        
        # Avvia thread per flussi scaduti
        expire_thread = threading.Thread(target=self.check_expired_flows, daemon=True)
        expire_thread.start()
        
        # Setup signal handler per Ctrl+C
        def signal_handler(sig, frame):
            print("\n\nShutdown in corso...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Avvia cattura
            sniff(
                iface=self.interface,
                prn=self.packet_callback,
                store=False,
                count=packet_count if packet_count > 0 else 0,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            print(f"Errore durante cattura: {e}")
        finally:
            self.running = False
            
            # Analizza flussi rimanenti
            print("\nAnalisi flussi rimanenti...")
            remaining = self.flow_manager.get_all_flows()
            for flow in remaining:
                if flow.total_packets >= 3:
                    self.analyze_flow(flow)
            
            self.print_stats()


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='NIDS Network Sniffer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  sudo python src/sniffer.py --model-path models/random_forest/model_binary.pkl
  sudo python src/sniffer.py --model-path models/xgboost/model_binary.pkl --interface eth0
  sudo python src/sniffer.py --model-path models/lightgbm/model_binary.pkl --packet-count 1000
        """
    )
    
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path al modello .pkl')
    parser.add_argument('--interface', type=str, default=None,
                        help='Interfaccia di rete')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'Timeout flussi in secondi (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Soglia probabilita alert (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--log-file', type=Path, default=None,
                        help='File log attacchi')
    parser.add_argument('--packet-count', type=int, default=0,
                        help='Pacchetti da catturare (0 = infinito)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostra tutti i flussi')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    print("\n" + "=" * 60)
    print("NIDS NETWORK SNIFFER")
    print("=" * 60)
    
    if os.geteuid() != 0:
        print("\nERRORE: Richiesti privilegi root")
        print("Eseguire con: sudo python src/sniffer.py ...")
        sys.exit(1)
    
    if not SCAPY_AVAILABLE:
        print("\nERRORE: Scapy non installato")
        print("Eseguire: pip install scapy")
        sys.exit(1)
    
    try:
        sniffer = NIDSSniffer(
            model_path=args.model_path,
            interface=args.interface,
            timeout=args.timeout,
            threshold=args.threshold,
            log_file=args.log_file,
            verbose=args.verbose
        )
        
        sniffer.start(packet_count=args.packet_count)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()