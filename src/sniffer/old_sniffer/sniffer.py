"""
================================================================================
NIDS-ML - Network Sniffer
================================================================================

Sistema di Network Intrusion Detection con Machine Learning.

ARCHITETTURA:
-------------
1. Cattura pacchetti (live o PCAP)
2. Aggrega pacchetti in flussi bidirezionali
3. Estrae feature compatibili con CIC-IDS2017
4. Classifica con modello ML
5. Genera alert e (opzionalmente) blocca IP

USO:
----
# Analisi PCAP
python src/sniffer.py --pcap capture.pcap

# Cattura live
sudo python src/sniffer.py --interface eth0

# Prevention mode (blocca IP)
sudo python src/sniffer.py --interface eth0 --mode prevention

# Modello specifico
python src/sniffer.py --pcap file.pcap --model-path models/xgboost/cv5_iter200_gpu/model_binary.pkl

================================================================================
"""

import sys
import os
import argparse
import json
import time
import signal
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

# Setup path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_logger, get_project_root, suppress_warnings
from src.feature_engineering import load_artifacts

suppress_warnings()
logger = get_logger(__name__)

# Verifica Scapy
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list, conf, PcapReader
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("ATTENZIONE: Scapy non installato. Esegui: pip install scapy")


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

DEFAULT_TIMEOUT = 60          # Secondi prima che un flusso scada
DEFAULT_THRESHOLD = 0.5       # Soglia probabilita per classificare come attacco
DEFAULT_MIN_PACKETS = 2       # Minimo pacchetti per analizzare un flusso
MAX_PACKETS_PER_FLOW = 500    # Forza analisi dopo N pacchetti
EXPIRE_CHECK_INTERVAL = 5     # Secondi tra controlli flussi scaduti


# ==============================================================================
# FLOW CLASS
# ==============================================================================

class Flow:
    """
    Rappresenta un flusso di rete bidirezionale.
    
    Raccoglie statistiche sui pacchetti e calcola feature
    compatibili con il formato CIC-IDS2017.
    """
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int,
                 dst_port: int, protocol: int):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        # Timing
        self.start_time = time.time()
        self.last_time = self.start_time
        
        # Contatori
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        
        # Liste per statistiche
        self.fwd_lengths: List[int] = []
        self.bwd_lengths: List[int] = []
        self.fwd_iats: List[float] = []
        self.bwd_iats: List[float] = []
        
        # Timing per IAT
        self.last_fwd_time: Optional[float] = None
        self.last_bwd_time: Optional[float] = None
        
        # TCP Flags
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0
        
        # Window size
        self.init_win_fwd: Optional[int] = None
        self.init_win_bwd: Optional[int] = None
    
    @property
    def flow_id(self) -> str:
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}:{self.protocol}"
    
    @property
    def duration(self) -> float:
        return max(self.last_time - self.start_time, 0.000001)
    
    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets
    
    @property
    def total_bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes
    
    def add_packet(self, packet_len: int, is_forward: bool, timestamp: float,
                   tcp_flags: dict = None, window_size: int = None):
        """Aggiunge un pacchetto al flusso."""
        self.last_time = timestamp
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
            self.fwd_lengths.append(packet_len)
            
            if self.last_fwd_time is not None:
                self.fwd_iats.append(timestamp - self.last_fwd_time)
            self.last_fwd_time = timestamp
            
            if self.init_win_fwd is None and window_size is not None:
                self.init_win_fwd = window_size
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len
            self.bwd_lengths.append(packet_len)
            
            if self.last_bwd_time is not None:
                self.bwd_iats.append(timestamp - self.last_bwd_time)
            self.last_bwd_time = timestamp
            
            if self.init_win_bwd is None and window_size is not None:
                self.init_win_bwd = window_size
        
        if tcp_flags:
            if tcp_flags.get('F'): self.fin_count += 1
            if tcp_flags.get('S'): self.syn_count += 1
            if tcp_flags.get('R'): self.rst_count += 1
            if tcp_flags.get('P'): self.psh_count += 1
            if tcp_flags.get('A'): self.ack_count += 1
            if tcp_flags.get('U'): self.urg_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte flusso in dizionario."""
        return {
            'flow_id': self.flow_id,
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'duration': self.duration,
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes
        }
    
    def extract_features(self) -> Dict[str, float]:
        """
        Estrae feature compatibili con CIC-IDS2017.
        
        IMPORTANTE: I nomi e le unita devono corrispondere esattamente
        a quelli del dataset originale (generato da CICFlowMeter).
        """
        features = {}
        
        # Duration in MICROSECONDI (come CIC-IDS2017)
        features['Flow Duration'] = self.duration * 1e6
        
        # Packet counts
        features['Total Fwd Packets'] = self.fwd_packets
        features['Total Backward Packets'] = self.bwd_packets
        
        # Byte counts
        features['Total Length of Fwd Packets'] = self.fwd_bytes
        features['Total Length of Bwd Packets'] = self.bwd_bytes
        
        # Forward packet length stats
        if self.fwd_lengths:
            features['Fwd Packet Length Max'] = max(self.fwd_lengths)
            features['Fwd Packet Length Min'] = min(self.fwd_lengths)
            features['Fwd Packet Length Mean'] = np.mean(self.fwd_lengths)
            features['Fwd Packet Length Std'] = np.std(self.fwd_lengths, ddof=0)
        else:
            features['Fwd Packet Length Max'] = 0
            features['Fwd Packet Length Min'] = 0
            features['Fwd Packet Length Mean'] = 0
            features['Fwd Packet Length Std'] = 0
        
        # Backward packet length stats
        if self.bwd_lengths:
            features['Bwd Packet Length Max'] = max(self.bwd_lengths)
            features['Bwd Packet Length Min'] = min(self.bwd_lengths)
            features['Bwd Packet Length Mean'] = np.mean(self.bwd_lengths)
            features['Bwd Packet Length Std'] = np.std(self.bwd_lengths, ddof=0)
        else:
            features['Bwd Packet Length Max'] = 0
            features['Bwd Packet Length Min'] = 0
            features['Bwd Packet Length Mean'] = 0
            features['Bwd Packet Length Std'] = 0
        
        # Flow rates
        duration_sec = self.duration
        features['Flow Bytes/s'] = self.total_bytes / duration_sec
        features['Flow Packets/s'] = self.total_packets / duration_sec
        features['Fwd Packets/s'] = self.fwd_packets / duration_sec
        features['Bwd Packets/s'] = self.bwd_packets / duration_sec
        
        # IAT (Inter-Arrival Time) in MICROSECONDI
        all_iats = self.fwd_iats + self.bwd_iats
        
        if all_iats:
            features['Flow IAT Mean'] = np.mean(all_iats) * 1e6
            features['Flow IAT Std'] = np.std(all_iats, ddof=0) * 1e6
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
            features['Fwd IAT Std'] = np.std(self.fwd_iats, ddof=0) * 1e6
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
            features['Bwd IAT Std'] = np.std(self.bwd_iats, ddof=0) * 1e6
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
        
        # Combined packet length
        all_lengths = self.fwd_lengths + self.bwd_lengths
        if all_lengths:
            features['Max Packet Length'] = max(all_lengths)
            features['Min Packet Length'] = min(all_lengths)
            features['Packet Length Mean'] = np.mean(all_lengths)
            features['Packet Length Std'] = np.std(all_lengths, ddof=0)
            features['Packet Length Variance'] = np.var(all_lengths, ddof=0)
        else:
            features['Max Packet Length'] = 0
            features['Min Packet Length'] = 0
            features['Packet Length Mean'] = 0
            features['Packet Length Std'] = 0
            features['Packet Length Variance'] = 0
        
        # Segment size
        features['Avg Fwd Segment Size'] = features['Fwd Packet Length Mean']
        features['Avg Bwd Segment Size'] = features['Bwd Packet Length Mean']
        
        # Subflow
        features['Subflow Fwd Packets'] = self.fwd_packets
        features['Subflow Fwd Bytes'] = self.fwd_bytes
        features['Subflow Bwd Packets'] = self.bwd_packets
        features['Subflow Bwd Bytes'] = self.bwd_bytes
        
        # Header length (approssimazione: 20 bytes TCP/IP)
        features['Fwd Header Length'] = self.fwd_packets * 20
        features['Fwd Header Length.1'] = features['Fwd Header Length']
        features['Bwd Header Length'] = self.bwd_packets * 20
        
        # Average packet size
        features['Average Packet Size'] = self.total_bytes / max(self.total_packets, 1)
        
        # Down/Up ratio
        features['Down/Up Ratio'] = self.bwd_packets / max(self.fwd_packets, 1)
        
        # Init window
        features['Init_Win_bytes_forward'] = self.init_win_fwd if self.init_win_fwd else 65535
        features['Init_Win_bytes_backward'] = self.init_win_bwd if self.init_win_bwd else 65535
        
        # Active/Idle (placeholder - difficili da calcolare accuratamente)
        for stat in ['Mean', 'Std', 'Max', 'Min']:
            features[f'Active {stat}'] = 0
            features[f'Idle {stat}'] = 0
        
        # Altri
        features['act_data_pkt_fwd'] = self.fwd_packets
        features['min_seg_size_forward'] = min(self.fwd_lengths) if self.fwd_lengths else 0
        
        return features


# ==============================================================================
# FLOW MANAGER
# ==============================================================================

class FlowManager:
    """Gestisce l'aggregazione dei pacchetti in flussi."""
    
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.flows: Dict[str, Flow] = {}
        self.timeout = timeout
        self.lock = threading.Lock()
    
    def _get_flow_key(self, src_ip: str, dst_ip: str, src_port: int,
                      dst_port: int, protocol: int) -> Tuple[str, bool]:
        """Genera chiave normalizzata per flusso bidirezionale."""
        if (src_ip, src_port) < (dst_ip, dst_port):
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}", True
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}", False
    
    def add_packet(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                   protocol: int, packet_len: int, timestamp: float,
                   tcp_flags: dict = None, window_size: int = None) -> Optional[Flow]:
        """Aggiunge pacchetto. Restituisce Flow se completo."""
        key, is_forward = self._get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
        
        with self.lock:
            if key not in self.flows:
                if is_forward:
                    self.flows[key] = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                else:
                    self.flows[key] = Flow(dst_ip, src_ip, dst_port, src_port, protocol)
                self.flows[key].start_time = timestamp
            
            flow = self.flows[key]
            flow.add_packet(packet_len, is_forward, timestamp, tcp_flags, window_size)
            
            if flow.total_packets >= MAX_PACKETS_PER_FLOW:
                del self.flows[key]
                return flow
        
        return None
    
    def get_expired_flows(self, reference_time: float = None) -> List[Flow]:
        """Restituisce flussi scaduti."""
        expired = []
        current_time = reference_time if reference_time else time.time()
        
        with self.lock:
            expired_keys = [
                key for key, flow in self.flows.items()
                if current_time - flow.last_time > self.timeout
            ]
            for key in expired_keys:
                expired.append(self.flows.pop(key))
        
        return expired
    
    def get_all_flows(self) -> List[Flow]:
        """Restituisce tutti i flussi (per shutdown)."""
        with self.lock:
            flows = list(self.flows.values())
            self.flows.clear()
        return flows
    
    def get_flow_count(self) -> int:
        with self.lock:
            return len(self.flows)


# ==============================================================================
# SNIFFER LOGGER
# ==============================================================================

class SnifferLogger:
    """Sistema di logging per lo sniffer."""
    
    def __init__(self, log_dir: Path, session_id: str):
        self.log_dir = log_dir
        self.session_id = session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.general_log = log_dir / f"sniffer_{session_id}.log"
        self.attack_log = log_dir / f"attacks_{session_id}.log"
        self.flow_log = log_dir / f"flows_{session_id}.jsonl"
        
        self.stats = {
            'packets_captured': 0,
            'packets_processed': 0,
            'flows_analyzed': 0,
            'attacks_detected': 0,
            'start_time': datetime.now().isoformat(),
            'unique_ips': set()
        }
    
    def _write_log(self, file: Path, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(file, 'a') as f:
            f.write(f"{timestamp} | {message}\n")
    
    def log_info(self, message: str):
        self._write_log(self.general_log, f"INFO | {message}")
    
    def log_attack(self, flow_info: dict, probability: float):
        self.stats['attacks_detected'] += 1
        entry = {
            'timestamp': datetime.now().isoformat(),
            'flow': flow_info,
            'probability': probability
        }
        with open(self.attack_log, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_flow(self, flow_info: dict, prediction: int, probability: float):
        self.stats['flows_analyzed'] += 1
        entry = {
            'timestamp': datetime.now().isoformat(),
            'flow': flow_info,
            'prediction': prediction,
            'probability': probability
        }
        with open(self.flow_log, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def update_stats(self, captured: int = 0, processed: int = 0):
        self.stats['packets_captured'] += captured
        self.stats['packets_processed'] += processed
    
    def add_ip(self, ip: str):
        self.stats['unique_ips'].add(ip)
    
    def get_summary(self) -> dict:
        return {
            'session_id': self.session_id,
            'packets_processed': self.stats['packets_processed'],
            'flows_analyzed': self.stats['flows_analyzed'],
            'attacks_detected': self.stats['attacks_detected'],
            'unique_ips': len(self.stats['unique_ips']),
            'detection_rate': (self.stats['attacks_detected'] / max(self.stats['flows_analyzed'], 1)) * 100
        }


# ==============================================================================
# FIREWALL MANAGER
# ==============================================================================

class FirewallManager:
    """Gestisce blocco IP via iptables."""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.blocked_ips: set = set()
    
    def block_ip(self, ip: str) -> bool:
        if not self.enabled or ip in self.blocked_ips:
            return False
        try:
            result = subprocess.run(
                ['iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                self.blocked_ips.add(ip)
                return True
        except Exception:
            pass
        return False
    
    def cleanup(self):
        for ip in list(self.blocked_ips):
            try:
                subprocess.run(
                    ['iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP'],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass
        self.blocked_ips.clear()


# ==============================================================================
# SNIFFER ENGINE
# ==============================================================================

class SnifferEngine:
    """Engine principale dello sniffer NIDS."""
    
    def __init__(self,
                 model_path: Path,
                 interface: str = None,
                 pcap_file: Path = None,
                 mode: str = 'detection',
                 timeout: int = DEFAULT_TIMEOUT,
                 threshold: float = DEFAULT_THRESHOLD,
                 min_packets: int = DEFAULT_MIN_PACKETS,
                 log_dir: Path = None,
                 verbose: bool = False,
                 quiet: bool = False):
        
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy non installato")
        
        self.interface = interface
        self.pcap_file = Path(pcap_file) if pcap_file else None
        self.mode = mode
        self.timeout = timeout
        self.threshold = threshold
        self.min_packets = min_packets
        self.verbose = verbose
        self.quiet = quiet
        
        # Session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) if log_dir else get_project_root() / "logs"
        
        self.sniffer_logger = SnifferLogger(self.log_dir, self.session_id)
        
        # Carica modello
        self._load_model(model_path)
        
        # Inizializza componenti
        self.flow_manager = FlowManager(timeout=timeout)
        self.firewall = FirewallManager(enabled=(mode == 'prevention'))
        
        self.running = False
    
    def _load_model(self, model_path: Path):
        """Carica modello e artifacts."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        logger.info(f"Caricamento modello: {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info("Caricamento artifacts...")
        self.scaler, self.selected_features, _, self.scaler_columns = load_artifacts()
        
        if self.scaler_columns is None:
            raise RuntimeError("scaler_columns.json mancante")
        
        # Feature specifiche del modello
        features_path = model_path.parent / "features_binary.json"
        if features_path.exists():
            with open(features_path) as f:
                self.selected_features = json.load(f)
            logger.info(f"Caricate feature dal modello: {len(self.selected_features)}")
    
    def _extract_packet_info(self, packet) -> Optional[dict]:
        """Estrae informazioni da un pacchetto."""
        if not packet.haslayer(IP):
            return None
        
        ip = packet[IP]
        info = {
            'src_ip': ip.src,
            'dst_ip': ip.dst,
            'protocol': ip.proto,
            'length': len(packet),
            'timestamp': float(packet.time),
            'tcp_flags': {},
            'window_size': None
        }
        
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            info['src_port'] = tcp.sport
            info['dst_port'] = tcp.dport
            info['window_size'] = tcp.window
            flags = tcp.flags
            info['tcp_flags'] = {
                'F': bool(flags & 0x01),
                'S': bool(flags & 0x02),
                'R': bool(flags & 0x04),
                'P': bool(flags & 0x08),
                'A': bool(flags & 0x10),
                'U': bool(flags & 0x20)
            }
        elif packet.haslayer(UDP):
            info['src_port'] = packet[UDP].sport
            info['dst_port'] = packet[UDP].dport
        else:
            info['src_port'] = 0
            info['dst_port'] = 0
        
        return info
    
    def _analyze_flow(self, flow: Flow):
        """Analizza un flusso con il modello ML."""
        extracted = flow.extract_features()
        
        # Crea DataFrame con colonne scaler
        feature_dict = {col: extracted.get(col, 0) for col in self.scaler_columns}
        df_full = pd.DataFrame([feature_dict])
        
        # Scala
        scaled = self.scaler.transform(df_full)
        df_scaled = pd.DataFrame(scaled, columns=self.scaler_columns)
        
        # Seleziona feature (IMPORTANTE: nuovo DataFrame per XGBoost)
        df_selected = pd.DataFrame(
            df_scaled[self.selected_features].values,
            columns=list(self.selected_features)
        )
        
        # Predizione
        prediction = int(self.model.predict(df_selected)[0])
        
        prob = 0.5
        if hasattr(self.model, 'predict_proba'):
            prob = float(self.model.predict_proba(df_selected)[0][1])
        
        # Log
        self.sniffer_logger.log_flow(flow.to_dict(), prediction, prob)
        
        is_attack = prediction == 1 and prob >= self.threshold
        
        if is_attack:
            self._handle_attack(flow, prob)
        elif self.verbose and not self.quiet:
            print(f"[BENIGN] {flow.flow_id} | Pkts: {flow.total_packets} | Prob: {prob:.3f}")
    
    def _handle_attack(self, flow: Flow, probability: float):
        """Gestisce un flusso classificato come attacco."""
        self.sniffer_logger.log_attack(flow.to_dict(), probability)
        
        if not self.quiet:
            print(f"\n{'!'*60}")
            print(f"[ATTACK DETECTED]")
            print(f"  Flow: {flow.flow_id}")
            print(f"  Packets: {flow.total_packets}")
            print(f"  Probability: {probability:.3f}")
            print(f"{'!'*60}\n")
        
        if self.mode == 'prevention':
            self.firewall.block_ip(flow.src_ip)
    
    def _process_expired_flows(self, reference_time: float = None):
        """Processa flussi scaduti."""
        for flow in self.flow_manager.get_expired_flows(reference_time):
            if flow.total_packets >= self.min_packets:
                self._analyze_flow(flow)
    
    def _process_remaining_flows(self):
        """Processa flussi rimanenti."""
        for flow in self.flow_manager.get_all_flows():
            if flow.total_packets >= self.min_packets:
                self._analyze_flow(flow)
    
    def start_pcap(self, packet_count: int = 0) -> dict:
        """Analizza file PCAP."""
        if not self.pcap_file.exists():
            raise FileNotFoundError(f"PCAP non trovato: {self.pcap_file}")
        
        print(f"\nAnalisi PCAP: {self.pcap_file}")
        print(f"Session ID: {self.session_id}")
        print()
        
        self.running = True
        processed = 0
        last_time = None
        
        try:
            with PcapReader(str(self.pcap_file)) as reader:
                for packet in reader:
                    if not self.running:
                        break
                    if packet_count > 0 and processed >= packet_count:
                        break
                    
                    info = self._extract_packet_info(packet)
                    if info:
                        last_time = info['timestamp']
                        self.sniffer_logger.update_stats(captured=1, processed=1)
                        self.sniffer_logger.add_ip(info['src_ip'])
                        self.sniffer_logger.add_ip(info['dst_ip'])
                        
                        completed = self.flow_manager.add_packet(
                            info['src_ip'], info['dst_ip'],
                            info['src_port'], info['dst_port'],
                            info['protocol'], info['length'],
                            info['timestamp'], info['tcp_flags'],
                            info['window_size']
                        )
                        
                        if completed and completed.total_packets >= self.min_packets:
                            self._analyze_flow(completed)
                    
                    processed += 1
                    
                    if processed % 50000 == 0:
                        stats = self.sniffer_logger.stats
                        print(f"  Pacchetti: {processed:,} | Flussi: {stats['flows_analyzed']:,} | Attacchi: {stats['attacks_detected']:,}")
                        
                        if last_time:
                            self._process_expired_flows(reference_time=last_time)
        
        except Exception as e:
            print(f"ERRORE: {e}")
            raise
        
        # Processa rimanenti
        if last_time:
            self._process_expired_flows(reference_time=last_time)
        self._process_remaining_flows()
        
        return self._finalize()
    
    def start_live(self, packet_count: int = 0, duration: int = 0) -> dict:
        """Cattura live da interfaccia."""
        if not self.interface:
            raise ValueError("Interfaccia non specificata")
        
        available = get_if_list()
        if self.interface not in available:
            raise ValueError(f"Interfaccia {self.interface} non trovata. Disponibili: {available}")
        
        print(f"\nCattura live: {self.interface}")
        print(f"Session ID: {self.session_id}")
        print(f"Mode: {self.mode}")
        print("Ctrl+C per terminare\n")
        
        self.running = True
        
        def packet_callback(packet):
            if not self.running:
                return
            
            info = self._extract_packet_info(packet)
            if info:
                self.sniffer_logger.update_stats(captured=1, processed=1)
                self.sniffer_logger.add_ip(info['src_ip'])
                
                completed = self.flow_manager.add_packet(
                    info['src_ip'], info['dst_ip'],
                    info['src_port'], info['dst_port'],
                    info['protocol'], info['length'],
                    info['timestamp'], info['tcp_flags'],
                    info['window_size']
                )
                
                if completed and completed.total_packets >= self.min_packets:
                    self._analyze_flow(completed)
        
        def signal_handler(sig, frame):
            print("\nTerminazione...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Thread expire
        def expire_loop():
            while self.running:
                time.sleep(EXPIRE_CHECK_INTERVAL)
                self._process_expired_flows()
        
        expire_thread = threading.Thread(target=expire_loop, daemon=True)
        expire_thread.start()
        
        try:
            sniff(
                iface=self.interface,
                prn=packet_callback,
                store=False,
                count=packet_count if packet_count > 0 else 0,
                timeout=duration if duration > 0 else None
            )
        except Exception as e:
            print(f"ERRORE: {e}")
        finally:
            self.running = False
            self._process_remaining_flows()
        
        return self._finalize()
    
    def _finalize(self) -> dict:
        """Finalizza sessione."""
        self.firewall.cleanup()
        
        summary = self.sniffer_logger.get_summary()
        
        print(f"\n{'='*60}")
        print("RIEPILOGO")
        print(f"{'='*60}")
        print(f"Pacchetti processati: {summary['packets_processed']:,}")
        print(f"Flussi analizzati:    {summary['flows_analyzed']:,}")
        print(f"Attacchi rilevati:    {summary['attacks_detected']:,}")
        print(f"Detection rate:       {summary['detection_rate']:.1f}%")
        print(f"\nLog: {self.log_dir}")
        
        # Salva summary
        with open(self.log_dir / f"summary_{self.session_id}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NIDS Network Sniffer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Sorgente
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--interface', type=str, help='Interfaccia per cattura live')
    source.add_argument('--pcap', type=Path, help='File PCAP da analizzare')
    
    # Modello
    parser.add_argument('--model-path', type=Path, default=None)
    
    # Modalita
    parser.add_argument('--mode', choices=['detection', 'prevention'], default='detection')
    
    # Parametri
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--min-packets', type=int, default=DEFAULT_MIN_PACKETS)
    
    # Controllo
    parser.add_argument('--packet-count', type=int, default=0)
    parser.add_argument('--duration', type=int, default=0)
    
    # Output
    parser.add_argument('--log-dir', type=Path, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    
    # Model path default
    if args.model_path is None:
        args.model_path = get_project_root() / "models" / "best_model" / "model_binary.pkl"
    
    if not args.model_path.exists():
        print(f"ERRORE: Modello non trovato: {args.model_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("NIDS NETWORK SNIFFER")
    print(f"{'='*60}")
    
    try:
        engine = SnifferEngine(
            model_path=args.model_path,
            interface=args.interface,
            pcap_file=args.pcap,
            mode=args.mode,
            timeout=args.timeout,
            threshold=args.threshold,
            min_packets=args.min_packets,
            log_dir=args.log_dir,
            verbose=args.verbose,
            quiet=args.quiet
        )
        
        if args.pcap:
            engine.start_pcap(packet_count=args.packet_count)
        else:
            engine.start_live(packet_count=args.packet_count, duration=args.duration)
    
    except KeyboardInterrupt:
        print("\nInterrotto")
    except Exception as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()