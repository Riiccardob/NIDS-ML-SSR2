"""
================================================================================
NIDS-ML - Network Sniffer (v2.0)
================================================================================

Sistema di Network Intrusion Detection con supporto real-time e PCAP.

ARCHITETTURA:
-------------
Pattern Producer-Consumer per evitare packet loss:
1. Thread PRODUCER: cattura pacchetti e li mette in coda (veloce)
2. Thread CONSUMER: estrae dalla coda, aggrega in flussi, analizza (lento)
3. Thread EXPIRE: controlla flussi scaduti periodicamente

MODALITA:
---------
- DETECTION: Solo logging, nessuna modifica al sistema
- PREVENTION: Logging + blocco IP malevoli via iptables

GUIDA PARAMETRI:
----------------
    sudo python src/sniffer.py [opzioni]

Opzioni sorgente dati (mutualmente esclusive):
    --interface STR       Interfaccia di rete per cattura live
    --pcap PATH           File PCAP da analizzare (offline)

Opzioni modello:
    --model-path PATH     Path al modello (default: models/best_model/model_binary.pkl)

Opzioni modalita:
    --mode STR            'detection' o 'prevention' (default: detection)
                          detection = solo logging
                          prevention = logging + blocco firewall

Opzioni analisi:
    --timeout INT         Timeout flow in secondi (default: 60)
    --threshold FLOAT     Soglia probabilita alert (default: 0.5)
    --min-packets INT     Minimo pacchetti per analizzare flow (default: 3)

Opzioni output:
    --log-dir PATH        Directory log (default: logs/)
    --verbose             Mostra tutti i flussi (anche benigni)
    --quiet               Mostra solo alert critici

Opzioni controllo:
    --packet-count INT    Numero pacchetti da processare (0 = infinito)
    --duration INT        Durata cattura in secondi (0 = infinito)

ESEMPI:
-------
# Cattura live con best model (detection mode)
sudo python src/sniffer.py --interface eth0

# Analisi file PCAP
python src/sniffer.py --pcap capture.pcap

# Prevention mode (blocca IP malevoli)
sudo python src/sniffer.py --interface eth0 --mode prevention

# Usa modello specifico
sudo python src/sniffer.py --interface eth0 --model-path models/xgboost/model_binary.pkl

# Analisi PCAP con soglia alta (meno falsi positivi)
python src/sniffer.py --pcap test.pcap --threshold 0.8 --verbose

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
import queue
import time
import signal
import json
import subprocess
import hashlib

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib

from src.utils import get_logger, get_project_root, suppress_warnings
from src.feature_engineering import load_artifacts

suppress_warnings()

# Verifica Scapy
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

DEFAULT_TIMEOUT = 15  # Secondi prima che un flusso scada (basso per test, alzare in prod)
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_PACKETS = 2  # Minimo pacchetti per analisi flusso
MAX_PACKETS_PER_FLOW = 500  # Forza analisi dopo N pacchetti
QUEUE_MAX_SIZE = 10000  # Buffer per evitare memory overflow
EXPIRE_CHECK_INTERVAL = 3  # Secondi tra controlli flussi scaduti


# ==============================================================================
# LOGGING AVANZATO
# ==============================================================================

class SnifferLogger:
    """
    Sistema di logging avanzato per lo sniffer.
    
    Gestisce:
    - Log generale (tutte le operazioni)
    - Log attacchi (solo alert)
    - Log flussi (dettaglio ogni flusso analizzato)
    - Log firewall (azioni iptables)
    - Statistiche in tempo reale
    """
    
    def __init__(self, log_dir: Path, session_id: str):
        self.log_dir = log_dir
        self.session_id = session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # File di log separati
        self.general_log = log_dir / f"sniffer_{session_id}.log"
        self.attack_log = log_dir / f"attacks_{session_id}.log"
        self.flow_log = log_dir / f"flows_{session_id}.jsonl"
        self.firewall_log = log_dir / f"firewall_{session_id}.log"
        self.stats_file = log_dir / f"stats_{session_id}.json"
        
        # Logger principale
        self.logger = get_logger(f"sniffer_{session_id}", str(self.general_log))
        
        # Statistiche
        self.stats = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'packets_captured': 0,
            'packets_processed': 0,
            'packets_dropped': 0,
            'flows_analyzed': 0,
            'attacks_detected': 0,
            'benign_flows': 0,
            'ips_blocked': 0,
            'unique_src_ips': set(),
            'unique_dst_ips': set(),
            'attack_types': defaultdict(int)
        }
        
        self.lock = threading.Lock()
    
    def log_info(self, message: str) -> None:
        """Log messaggio informativo."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log errore."""
        self.logger.error(message)
    
    def log_attack(self, flow_data: Dict[str, Any], probability: float) -> None:
        """
        Log attacco rilevato.
        
        Args:
            flow_data: Dati del flusso
            probability: Probabilita predetta
        """
        timestamp = datetime.now().isoformat()
        
        attack_entry = {
            'timestamp': timestamp,
            'src_ip': flow_data.get('src_ip'),
            'dst_ip': flow_data.get('dst_ip'),
            'src_port': flow_data.get('src_port'),
            'dst_port': flow_data.get('dst_port'),
            'protocol': flow_data.get('protocol'),
            'probability': float(probability),
            'packets': flow_data.get('total_packets'),
            'bytes': flow_data.get('total_bytes'),
            'duration': flow_data.get('duration')
        }
        
        # Scrivi su file attacchi
        with open(self.attack_log, 'a') as f:
            f.write(f"[{timestamp}] ATTACK DETECTED | "
                    f"Src: {attack_entry['src_ip']}:{attack_entry['src_port']} -> "
                    f"Dst: {attack_entry['dst_ip']}:{attack_entry['dst_port']} | "
                    f"Prob: {probability:.3f} | "
                    f"Packets: {attack_entry['packets']}\n")
        
        with self.lock:
            self.stats['attacks_detected'] += 1
    
    def log_flow(self, flow_data: Dict[str, Any], prediction: int, probability: float) -> None:
        """
        Log flusso analizzato (formato JSONL per analisi successiva).
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': 'attack' if prediction == 1 else 'benign',
            'probability': float(probability),  # Converti numpy.float32 a float
            **flow_data
        }
        
        # Converti tutti i valori numpy a tipi Python nativi
        def convert_numpy(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        entry = convert_numpy(entry)
        
        with open(self.flow_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        with self.lock:
            self.stats['flows_analyzed'] += 1
            if prediction == 1:
                self.stats['attacks_detected'] += 1
            else:
                self.stats['benign_flows'] += 1
    
    def log_firewall_action(self, action: str, ip: str, success: bool) -> None:
        """Log azione firewall."""
        timestamp = datetime.now().isoformat()
        status = "SUCCESS" if success else "FAILED"
        
        with open(self.firewall_log, 'a') as f:
            f.write(f"[{timestamp}] {action} | IP: {ip} | Status: {status}\n")
        
        if success and action == "BLOCK":
            with self.lock:
                self.stats['ips_blocked'] += 1
    
    def update_packet_stats(self, captured: int = 0, processed: int = 0, dropped: int = 0) -> None:
        """Aggiorna statistiche pacchetti."""
        with self.lock:
            self.stats['packets_captured'] += captured
            self.stats['packets_processed'] += processed
            self.stats['packets_dropped'] += dropped
    
    def add_ip(self, src_ip: str, dst_ip: str) -> None:
        """Registra IP univoci."""
        with self.lock:
            self.stats['unique_src_ips'].add(src_ip)
            self.stats['unique_dst_ips'].add(dst_ip)
    
    def save_stats(self) -> None:
        """Salva statistiche finali su file."""
        with self.lock:
            self.stats['end_time'] = datetime.now().isoformat()
            
            # Converti set in liste per JSON
            stats_copy = self.stats.copy()
            stats_copy['unique_src_ips'] = list(self.stats['unique_src_ips'])
            stats_copy['unique_dst_ips'] = list(self.stats['unique_dst_ips'])
            stats_copy['attack_types'] = dict(self.stats['attack_types'])
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        self.logger.info(f"Statistiche salvate: {self.stats_file}")
    
    def get_stats_summary(self) -> str:
        """Restituisce riepilogo statistiche."""
        with self.lock:
            duration = 0
            if self.stats['start_time']:
                start = datetime.fromisoformat(self.stats['start_time'])
                duration = (datetime.now() - start).total_seconds()
            
            return (
                f"Durata: {duration:.1f}s | "
                f"Pacchetti: {self.stats['packets_captured']:,} | "
                f"Flussi: {self.stats['flows_analyzed']:,} | "
                f"Attacchi: {self.stats['attacks_detected']:,} | "
                f"IP bloccati: {self.stats['ips_blocked']}"
            )


# ==============================================================================
# FIREWALL MANAGER
# ==============================================================================

class FirewallManager:
    """
    Gestisce interazione con iptables per blocco IP.
    
    ATTENZIONE: Richiede privilegi root.
    """
    
    def __init__(self, enabled: bool = False, logger: SnifferLogger = None):
        self.enabled = enabled
        self.logger = logger
        self.blocked_ips = set()
        self.lock = threading.Lock()
        
        # Verifica se iptables e disponibile
        if enabled:
            self._check_iptables()
    
    def _check_iptables(self) -> bool:
        """Verifica disponibilita iptables."""
        try:
            result = subprocess.run(
                ['iptables', '-L', '-n'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"iptables non disponibile: {e}")
            return False
    
    def block_ip(self, ip: str) -> bool:
        """
        Blocca un IP con iptables.
        
        Args:
            ip: Indirizzo IP da bloccare
        
        Returns:
            True se blocco riuscito
        """
        if not self.enabled:
            return False
        
        with self.lock:
            if ip in self.blocked_ips:
                return True  # Gia bloccato
            
            try:
                # Aggiungi regola INPUT
                result = subprocess.run(
                    ['iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'],
                    capture_output=True,
                    timeout=5
                )
                
                success = result.returncode == 0
                
                if success:
                    self.blocked_ips.add(ip)
                    if self.logger:
                        self.logger.log_firewall_action("BLOCK", ip, True)
                        self.logger.log_info(f"IP bloccato: {ip}")
                else:
                    if self.logger:
                        self.logger.log_firewall_action("BLOCK", ip, False)
                        self.logger.log_error(f"Errore blocco IP {ip}: {result.stderr.decode()}")
                
                return success
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Eccezione blocco IP {ip}: {e}")
                return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Sblocca un IP."""
        if not self.enabled:
            return False
        
        with self.lock:
            try:
                result = subprocess.run(
                    ['iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP'],
                    capture_output=True,
                    timeout=5
                )
                
                success = result.returncode == 0
                
                if success:
                    self.blocked_ips.discard(ip)
                    if self.logger:
                        self.logger.log_firewall_action("UNBLOCK", ip, True)
                
                return success
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Eccezione sblocco IP {ip}: {e}")
                return False
    
    def cleanup(self) -> None:
        """Rimuove tutte le regole aggiunte durante la sessione."""
        if not self.enabled:
            return
        
        if self.logger:
            self.logger.log_info(f"Pulizia firewall: {len(self.blocked_ips)} IP da sbloccare")
        
        for ip in list(self.blocked_ips):
            self.unblock_ip(ip)


# ==============================================================================
# FLOW CLASS
# ==============================================================================

class Flow:
    """Rappresenta un flusso di rete aggregato."""
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        self.start_time = time.time()
        self.last_time = self.start_time
        
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        
        self.fwd_lengths: List[int] = []
        self.bwd_lengths: List[int] = []
        self.fwd_iats: List[float] = []
        self.bwd_iats: List[float] = []
        
        self.last_fwd_time = None
        self.last_bwd_time = None
        
        self.fin_count = 0
        self.syn_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0
    
    @property
    def flow_id(self) -> str:
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}:{self.protocol}"
    
    @property
    def duration(self) -> float:
        return max(self.last_time - self.start_time, 0.000001)  # Evita divisione per zero
    
    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets
    
    @property
    def total_bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes
    
    def add_packet(self, packet_len: int, is_forward: bool, timestamp: float, tcp_flags: dict = None):
        """Aggiunge pacchetto al flusso."""
        self.last_time = timestamp
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_len
            self.fwd_lengths.append(packet_len)
            if self.last_fwd_time is not None:
                self.fwd_iats.append(timestamp - self.last_fwd_time)
            self.last_fwd_time = timestamp
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_len
            self.bwd_lengths.append(packet_len)
            if self.last_bwd_time is not None:
                self.bwd_iats.append(timestamp - self.last_bwd_time)
            self.last_bwd_time = timestamp
        
        if tcp_flags:
            if tcp_flags.get('F'): self.fin_count += 1
            if tcp_flags.get('S'): self.syn_count += 1
            if tcp_flags.get('R'): self.rst_count += 1
            if tcp_flags.get('P'): self.psh_count += 1
            if tcp_flags.get('A'): self.ack_count += 1
            if tcp_flags.get('U'): self.urg_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte flusso in dizionario per logging."""
        return {
            'flow_id': self.flow_id,
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'duration': self.duration,
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes,
            'fwd_packets': self.fwd_packets,
            'bwd_packets': self.bwd_packets
        }
    
    def extract_features(self) -> Dict[str, float]:
        """Estrae feature vector compatibile con il modello."""
        features = {}
        
        # Durata in microsecondi (come CIC-IDS2017)
        features['Flow Duration'] = self.duration * 1e6
        
        # Contatori
        features['Total Fwd Packets'] = self.fwd_packets
        features['Total Backward Packets'] = self.bwd_packets
        features['Total Length of Fwd Packets'] = self.fwd_bytes
        features['Total Length of Bwd Packets'] = self.bwd_bytes
        
        # Statistiche lunghezza forward
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
        
        # Statistiche lunghezza backward
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
        features['Flow Bytes/s'] = self.total_bytes / self.duration
        features['Flow Packets/s'] = self.total_packets / self.duration
        
        # IAT statistics
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
        
        # Packet length combined
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
        
        # Average packet size
        features['Average Packet Size'] = self.total_bytes / max(self.total_packets, 1)
        
        # Ratio
        features['Down/Up Ratio'] = self.bwd_packets / max(self.fwd_packets, 1)
        
        return features


# ==============================================================================
# FLOW MANAGER
# ==============================================================================

class FlowManager:
    """Gestisce aggregazione pacchetti in flussi."""
    
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.flows: Dict[str, Flow] = {}
        self.timeout = timeout
        self.lock = threading.Lock()
    
    def _get_flow_key(self, src_ip: str, dst_ip: str, src_port: int, 
                      dst_port: int, protocol: int) -> Tuple[str, bool]:
        """Genera chiave normalizzata e determina direzione."""
        if (src_ip, src_port) < (dst_ip, dst_port):
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}", True
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}", False
    
    def add_packet(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                   protocol: int, packet_len: int, timestamp: float, 
                   tcp_flags: dict = None) -> Optional[Flow]:
        """Aggiunge pacchetto e restituisce flusso se completo."""
        key, is_forward = self._get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
        
        with self.lock:
            if key not in self.flows:
                if is_forward:
                    self.flows[key] = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                else:
                    self.flows[key] = Flow(dst_ip, src_ip, dst_port, src_port, protocol)
            
            flow = self.flows[key]
            flow.add_packet(packet_len, is_forward, timestamp, tcp_flags)
            
            if flow.total_packets >= MAX_PACKETS_PER_FLOW:
                del self.flows[key]
                return flow
        
        return None
    
    def get_expired_flows(self, reference_time: float = None) -> List[Flow]:
        """
        Restituisce flussi scaduti.
        
        Args:
            reference_time: Tempo di riferimento per il controllo timeout.
                           Se None, usa time.time() (per cattura live).
                           Per analisi PCAP, passare il timestamp dell'ultimo pacchetto.
        
        Returns:
            Lista di flussi scaduti
        """
        expired = []
        current_time = reference_time if reference_time is not None else time.time()
        
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


# ==============================================================================
# SNIFFER PRINCIPALE
# ==============================================================================

class NIDSSniffer:
    """Network Intrusion Detection System Sniffer."""
    
    def __init__(self,
                 model_path: Path = None,
                 interface: str = None,
                 pcap_file: Path = None,
                 mode: str = 'detection',
                 timeout: float = DEFAULT_TIMEOUT,
                 threshold: float = DEFAULT_THRESHOLD,
                 min_packets: int = DEFAULT_MIN_PACKETS,
                 log_dir: Path = None,
                 verbose: bool = False,
                 quiet: bool = False):
        """
        Inizializza sniffer.
        
        Args:
            model_path: Path modello (default: best_model)
            interface: Interfaccia rete per live capture
            pcap_file: File PCAP per analisi offline
            mode: 'detection' o 'prevention'
            timeout: Timeout flussi
            threshold: Soglia probabilita
            min_packets: Minimo pacchetti per analisi
            log_dir: Directory log
            verbose: Mostra tutti i flussi
            quiet: Mostra solo alert
        """
        self.interface = interface
        self.pcap_file = pcap_file
        self.mode = mode
        self.timeout = timeout
        self.threshold = threshold
        self.min_packets = min_packets
        self.verbose = verbose
        self.quiet = quiet
        
        # Genera session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        if log_dir is None:
            log_dir = get_project_root() / "logs"
        self.sniffer_logger = SnifferLogger(log_dir, self.session_id)
        
        # Determina path modello
        if model_path is None:
            model_path = get_project_root() / "models" / "best_model" / "model_binary.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        # Carica modello e artifacts
        self.sniffer_logger.log_info(f"Caricamento modello: {model_path}")
        self.model = joblib.load(model_path)
        
        self.sniffer_logger.log_info("Caricamento artifacts...")
        self.scaler, self.selected_features, _, self.scaler_columns = load_artifacts()
        
        # Cerca file features specifico del modello
        features_path = model_path.parent / "features_binary.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            self.sniffer_logger.log_info(f"Caricate feature dal modello: {len(self.selected_features)}")
        
        # Se scaler_columns non e disponibile (vecchi artifacts), ERRORE
        if self.scaler_columns is None:
            error_msg = (
                "scaler_columns.json non trovato negli artifacts.\n"
                "Questo file e essenziale per la corretta trasformazione delle feature.\n"
                "Soluzione: rieseguire python src/feature_engineering.py"
            )
            self.sniffer_logger.log_error(error_msg)
            raise RuntimeError(error_msg)
        
        # Flow manager
        self.flow_manager = FlowManager(timeout=timeout)
        
        # Firewall manager
        self.firewall = FirewallManager(
            enabled=(mode == 'prevention'),
            logger=self.sniffer_logger
        )
        
        # Coda per pattern producer-consumer
        self.packet_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        
        # Flag controllo
        self.running = False
        self.capture_complete = False
    
    def _extract_packet_info(self, packet) -> Optional[Dict[str, Any]]:
        """Estrae informazioni da un pacchetto."""
        if not packet.haslayer(IP):
            return None
        
        ip_layer = packet[IP]
        info = {
            'src_ip': ip_layer.src,
            'dst_ip': ip_layer.dst,
            'protocol': ip_layer.proto,
            'length': len(packet),
            'timestamp': float(packet.time) if hasattr(packet, 'time') else time.time(),
            'tcp_flags': None
        }
        
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            info['src_port'] = tcp_layer.sport
            info['dst_port'] = tcp_layer.dport
            info['tcp_flags'] = {
                'F': bool(tcp_layer.flags & 0x01),
                'S': bool(tcp_layer.flags & 0x02),
                'R': bool(tcp_layer.flags & 0x04),
                'P': bool(tcp_layer.flags & 0x08),
                'A': bool(tcp_layer.flags & 0x10),
                'U': bool(tcp_layer.flags & 0x20),
            }
        elif packet.haslayer(UDP):
            udp_layer = packet[UDP]
            info['src_port'] = udp_layer.sport
            info['dst_port'] = udp_layer.dport
        else:
            info['src_port'] = 0
            info['dst_port'] = 0
        
        return info
    
    def _packet_callback(self, packet):
        """Callback producer: mette pacchetto in coda."""
        info = self._extract_packet_info(packet)
        if info:
            try:
                self.packet_queue.put_nowait(info)
                self.sniffer_logger.update_packet_stats(captured=1)
            except queue.Full:
                self.sniffer_logger.update_packet_stats(dropped=1)
    
    def _consumer_thread(self):
        """Thread consumer: processa pacchetti dalla coda."""
        while self.running or not self.packet_queue.empty():
            try:
                info = self.packet_queue.get(timeout=1)
            except queue.Empty:
                # Controlla flussi scaduti
                self._process_expired_flows()
                continue
            
            self.sniffer_logger.update_packet_stats(processed=1)
            self.sniffer_logger.add_ip(info['src_ip'], info['dst_ip'])
            
            # Aggiungi a flow manager
            completed_flow = self.flow_manager.add_packet(
                info['src_ip'], info['dst_ip'],
                info['src_port'], info['dst_port'],
                info['protocol'], info['length'],
                info['timestamp'], info['tcp_flags']
            )
            
            if completed_flow and completed_flow.total_packets >= self.min_packets:
                self._analyze_flow(completed_flow)
        
        # Processa flussi rimanenti
        self._process_remaining_flows()
    
    def _expire_thread(self):
        """Thread per controllo periodico flussi scaduti."""
        while self.running:
            time.sleep(EXPIRE_CHECK_INTERVAL)
            self._process_expired_flows()
    
    def _process_expired_flows(self, reference_time: float = None):
        """
        Processa flussi scaduti.
        
        Args:
            reference_time: Tempo di riferimento per il timeout.
                           None = usa time.time() (live capture)
                           float = usa timestamp pacchetto (PCAP analysis)
        """
        expired = self.flow_manager.get_expired_flows(reference_time=reference_time)
        for flow in expired:
            if flow.total_packets >= self.min_packets:
                self._analyze_flow(flow)
    
    def _process_remaining_flows(self):
        """Processa flussi rimanenti al termine."""
        remaining = self.flow_manager.get_all_flows()
        for flow in remaining:
            if flow.total_packets >= self.min_packets:
                self._analyze_flow(flow)
    
    def _analyze_flow(self, flow: Flow):
        """Analizza un flusso e genera alert se necessario."""
        # Estrai feature dal flusso
        extracted_features = flow.extract_features()
        
        # Crea DataFrame con TUTTE le colonne usate per fittare lo scaler
        # Le feature non estratte vengono impostate a 0
        feature_dict = {}
        for col in self.scaler_columns:
            feature_dict[col] = extracted_features.get(col, 0)
        
        df_full = pd.DataFrame([feature_dict])
        
        # Scala usando tutte le colonne originali
        scaled_values = self.scaler.transform(df_full)
        df_scaled = pd.DataFrame(scaled_values, columns=self.scaler_columns)
        
        # Seleziona solo le feature usate dal modello (nell'ordine esatto)
        # IMPORTANTE: Crea un NUOVO DataFrame per garantire che XGBoost riceva i nomi
        df_selected = pd.DataFrame(
            df_scaled[self.selected_features].values,
            columns=list(self.selected_features)
        )
        
        # Predizione
        prediction = int(self.model.predict(df_selected)[0])
        
        # Probabilita (converti a float Python nativo)
        prob = 0.5
        if hasattr(self.model, 'predict_proba'):
            prob = float(self.model.predict_proba(df_selected)[0][1])
        
        # Log flusso
        self.sniffer_logger.log_flow(flow.to_dict(), prediction, prob)
        
        # Gestisci risultato
        is_attack = prediction == 1 and prob >= self.threshold
        
        # Stampa progresso ogni 10 flussi
        total_flows = self.sniffer_logger.stats.get('flows_analyzed', 0) + 1
        if total_flows % 10 == 0 and not self.quiet:
            attacks = self.sniffer_logger.stats.get('attacks_detected', 0)
            print(f"[PROGRESS] Flussi: {total_flows} | Attacchi: {attacks} | Ultimo: {'ATTACK' if is_attack else 'BENIGN'} ({prob:.2f})")
        
        if is_attack:
            self._handle_attack(flow, prob)
        elif self.verbose:
            if not self.quiet:
                print(f"[BENIGN] {flow.flow_id} | Pkts: {flow.total_packets} | Prob: {prob:.3f}")
    
    def _handle_attack(self, flow: Flow, probability: float):
        """Gestisce flusso classificato come attacco."""
        # Log attacco
        self.sniffer_logger.log_attack(flow.to_dict(), probability)
        
        # Output a video
        if not self.quiet:
            print("\n" + "!" * 60)
            print(f"[ALERT] ATTACCO RILEVATO!")
            print(f"  Flow:     {flow.flow_id}")
            print(f"  Packets:  {flow.total_packets}")
            print(f"  Bytes:    {flow.total_bytes}")
            print(f"  Duration: {flow.duration:.2f}s")
            print(f"  Prob:     {probability:.3f}")
            
            if self.mode == 'prevention':
                print(f"  Action:   BLOCKING {flow.src_ip}")
            else:
                print(f"  Mode:     Detection only (no block)")
            
            print("!" * 60 + "\n")
        
        # Blocca IP se in prevention mode
        if self.mode == 'prevention':
            self.firewall.block_ip(flow.src_ip)
    
    def start_live(self, packet_count: int = 0, duration: int = 0):
        """Avvia cattura live."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy non disponibile")
        
        if os.geteuid() != 0:
            raise PermissionError("Richiesti privilegi root per cattura live")
        
        # Determina interfaccia
        if self.interface is None:
            interfaces = get_if_list()
            for iface in interfaces:
                if iface != 'lo' and not iface.startswith('docker'):
                    self.interface = iface
                    break
            if self.interface is None:
                self.interface = conf.iface
        
        self.sniffer_logger.log_info(f"Avvio cattura live su {self.interface}")
        self.sniffer_logger.log_info(f"Mode: {self.mode}")
        self.sniffer_logger.log_info(f"Threshold: {self.threshold}")
        
        print(f"\nAvvio sniffer su: {self.interface}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Soglia: {self.threshold}")
        print(f"Session ID: {self.session_id}")
        print("\nPremi Ctrl+C per fermare...\n")
        
        self.running = True
        
        # Avvia thread consumer
        consumer = threading.Thread(target=self._consumer_thread, daemon=True)
        consumer.start()
        
        # Avvia thread expire
        expire = threading.Thread(target=self._expire_thread, daemon=True)
        expire.start()
        
        # Signal handler
        def signal_handler(sig, frame):
            print("\n\nShutdown in corso...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Timer per durata
        if duration > 0:
            def stop_timer():
                time.sleep(duration)
                self.running = False
            timer = threading.Thread(target=stop_timer, daemon=True)
            timer.start()
        
        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                store=False,
                count=packet_count if packet_count > 0 else 0,
                stop_filter=lambda x: not self.running
            )
        finally:
            self.running = False
            consumer.join(timeout=5)
            self._finalize()
    
    def start_pcap(self, packet_count: int = 0):
        """Analizza file PCAP."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy non disponibile")
        
        if not self.pcap_file.exists():
            raise FileNotFoundError(f"File PCAP non trovato: {self.pcap_file}")
        
        self.sniffer_logger.log_info(f"Analisi PCAP: {self.pcap_file}")
        
        print(f"\nAnalisi PCAP: {self.pcap_file}")
        print(f"Session ID: {self.session_id}")
        print(f"Modalita: STREAMING (memory-efficient)")
        print()
        
        self.running = True
        
        # Usa PcapReader per streaming (non carica tutto in RAM)
        from scapy.all import PcapReader
        
        print("Analisi in corso (streaming)...")
        processed = 0
        last_packet_time = None  # Traccia timestamp ultimo pacchetto per PCAP
        
        try:
            with PcapReader(str(self.pcap_file)) as reader:
                for packet in reader:
                    if not self.running:
                        break
                    
                    if packet_count > 0 and processed >= packet_count:
                        break
                    
                    info = self._extract_packet_info(packet)
                    if info:
                        last_packet_time = info['timestamp']  # Aggiorna timestamp
                        
                        self.sniffer_logger.update_packet_stats(captured=1, processed=1)
                        self.sniffer_logger.add_ip(info['src_ip'], info['dst_ip'])
                        
                        completed_flow = self.flow_manager.add_packet(
                            info['src_ip'], info['dst_ip'],
                            info['src_port'], info['dst_port'],
                            info['protocol'], info['length'],
                            info['timestamp'], info['tcp_flags']
                        )
                        
                        if completed_flow and completed_flow.total_packets >= self.min_packets:
                            self._analyze_flow(completed_flow)
                    
                    processed += 1
                    
                    # Progress e cleanup periodico
                    if processed % 10000 == 0:
                        attacks = self.sniffer_logger.stats.get('attacks_detected', 0)
                        flows = self.sniffer_logger.stats.get('flows_analyzed', 0)
                        print(f"  Pacchetti: {processed:,} | Flussi: {flows:,} | Attacchi: {attacks}")
                        
                        # Cleanup flussi scaduti ogni 50k pacchetti
                        # USA IL TIMESTAMP DEL PACCHETTO, non time.time()!
                        if processed % 50000 == 0 and last_packet_time is not None:
                            self._process_expired_flows(reference_time=last_packet_time)
        
        except Exception as e:
            print(f"ERRORE durante analisi PCAP: {e}")
            raise
        
        print(f"\nPacchetti processati: {processed:,}")
        
        # Processa flussi rimanenti usando timestamp PCAP
        if last_packet_time is not None:
            self._process_expired_flows(reference_time=last_packet_time)
        self._process_remaining_flows()
        
        self._finalize()
    
    def _finalize(self):
        """Finalizza sessione."""
        # Cleanup firewall
        self.firewall.cleanup()
        
        # Salva statistiche
        self.sniffer_logger.save_stats()
        
        # Report finale
        print("\n" + "=" * 60)
        print("SESSIONE TERMINATA")
        print("=" * 60)
        print(f"\n{self.sniffer_logger.get_stats_summary()}")
        print(f"\nLog directory: {self.sniffer_logger.log_dir}")
        print(f"Session ID: {self.session_id}")


# ==============================================================================
# FUNZIONE PER USO DA NOTEBOOK/SCRIPT
# ==============================================================================

def analyze_pcap_file(
    pcap_path: str,
    model_path: str = None,
    threshold: float = 0.5,
    timeout: int = 60,
    min_packets: int = 2,
    verbose: bool = False,
    progress_interval: int = 10000,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Analizza un file PCAP e restituisce i risultati.
    
    Questa funzione può essere importata e usata direttamente da notebook o script
    senza dover passare per la CLI.
    
    Args:
        pcap_path: Path al file PCAP
        model_path: Path al modello (default: models/best_model/model_binary.pkl)
        threshold: Soglia probabilità per classificare come attacco (default: 0.5)
        timeout: Timeout flusso in secondi (default: 60)
        min_packets: Minimo pacchetti per analizzare un flusso (default: 2)
        verbose: Mostra dettagli flussi benigni (default: False)
        progress_interval: Intervallo pacchetti per progress (default: 10000)
        show_progress: Mostra progress bar con stima tempo (default: True)
    
    Returns:
        Dict con risultati:
            - pcap: nome file
            - packets_processed: pacchetti totali processati
            - flows_analyzed: flussi analizzati
            - attacks_detected: attacchi rilevati
            - benign_detected: flussi benigni
            - attack_flows: lista top attacchi (max 100)
            - detection_rate: percentuale attacchi
            - stats: statistiche dettagliate
    
    Example:
        >>> from src.sniffer import analyze_pcap_file
        >>> results = analyze_pcap_file(
        ...     "Friday-WorkingHours.pcap",
        ...     threshold=0.3,
        ...     min_packets=1
        ... )
        >>> print(f"Attacchi: {results['attacks_detected']}")
    """
    from scapy.all import PcapReader, IP, TCP, UDP
    
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP non trovato: {pcap_path}")
    
    # Carica modello
    project_root = get_project_root()
    if model_path is None:
        model_path = project_root / "models" / "best_model" / "model_binary.pkl"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    print(f"Caricamento modello: {model_path.name}")
    model = joblib.load(model_path)
    
    # Carica artifacts
    scaler, selected_features, _, scaler_columns = load_artifacts()
    
    if scaler_columns is None:
        raise RuntimeError(
            "scaler_columns.json mancante. Rieseguire feature_engineering.py"
        )
    
    # Carica feature specifiche del modello se presenti
    model_features_path = model_path.parent / "features_binary.json"
    if model_features_path.exists():
        with open(model_features_path) as f:
            selected_features = json.load(f)
    
    print(f"Feature: {len(selected_features)}")
    print(f"Soglia: {threshold}")
    print(f"Min packets: {min_packets}")
    
    # Risultati
    results = {
        'pcap': pcap_path.name,
        'model_path': str(model_path),
        'threshold': threshold,
        'min_packets': min_packets,
        'timeout': timeout,
        'packets_processed': 0,
        'flows_analyzed': 0,
        'attacks_detected': 0,
        'benign_detected': 0,
        'attack_flows': [],
        'benign_sample': [],
        'probabilities': []
    }
    
    # Flow tracking usando la stessa classe Flow di start_pcap
    flows: Dict[str, Flow] = {}
    
    # Stima pacchetti
    file_size = pcap_path.stat().st_size
    estimated_packets = int(file_size / 200)
    
    print(f"\n{'='*60}")
    print(f"Analisi: {pcap_path.name}")
    print(f"Size: {file_size / (1024**2):.1f} MB")
    print(f"Pacchetti stimati: ~{estimated_packets:,}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    def predict_flow(flow: Flow):
        """Predice se un flusso e' un attacco usando la classe Flow."""
        features = flow.extract_features()
        
        # Crea DataFrame con colonne scaler
        feature_dict = {col: features.get(col, 0) for col in scaler_columns}
        df = pd.DataFrame([feature_dict])
        
        # Scala
        df_scaled = pd.DataFrame(scaler.transform(df), columns=scaler_columns)
        
        # Seleziona feature modello
        df_selected = df_scaled[selected_features]
        
        # Predici
        pred = int(model.predict(df_selected)[0])
        prob = float(model.predict_proba(df_selected)[0][1]) if hasattr(model, 'predict_proba') else 0.5
        
        return pred, prob
    
    def analyze_and_clear_flow(flow_key: str, flow: Flow):
        """Analizza un flusso e aggiorna risultati."""
        if flow.total_packets < min_packets:
            return
        
        pred, prob = predict_flow(flow)
        
        results['flows_analyzed'] += 1
        results['probabilities'].append(prob)
        
        is_attack = pred == 1 and prob >= threshold
        
        if is_attack:
            results['attacks_detected'] += 1
            if len(results['attack_flows']) < 100:
                results['attack_flows'].append({
                    'flow_id': flow.flow_id,
                    'probability': prob,
                    'packets': flow.total_packets,
                    'duration': flow.duration,
                    'bytes': flow.total_bytes
                })
            if verbose:
                print(f"  [ATTACK] {flow.flow_id} | prob={prob:.3f} | pkts={flow.total_packets}")
        else:
            results['benign_detected'] += 1
            if len(results['benign_sample']) < 10:
                results['benign_sample'].append({
                    'flow_id': flow.flow_id,
                    'probability': prob,
                    'packets': flow.total_packets
                })
    
    def get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol):
        """Genera chiave flusso bidirezionale (stessa logica di FlowManager)."""
        if (src_ip, src_port) < (dst_ip, dst_port):
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}", True
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}", False
    
    # Processa PCAP in streaming
    try:
        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                if not pkt.haslayer(IP):
                    continue
                
                ip = pkt[IP]
                src_ip, dst_ip = ip.src, ip.dst
                protocol = ip.proto
                pkt_len = len(pkt)
                timestamp = float(pkt.time)
                
                src_port = dst_port = 0
                tcp_flags = {}
                
                if pkt.haslayer(TCP):
                    tcp_layer = pkt[TCP]
                    src_port = tcp_layer.sport
                    dst_port = tcp_layer.dport
                    # Estrai flags come dizionario (stessa logica di _extract_packet_info)
                    flags = tcp_layer.flags
                    tcp_flags = {
                        'F': bool(flags & 0x01),  # FIN
                        'S': bool(flags & 0x02),  # SYN
                        'R': bool(flags & 0x04),  # RST
                        'P': bool(flags & 0x08),  # PSH
                        'A': bool(flags & 0x10),  # ACK
                        'U': bool(flags & 0x20)   # URG
                    }
                elif pkt.haslayer(UDP):
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport
                
                # Chiave flusso bidirezionale
                flow_key, is_forward = get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
                
                # Crea o aggiorna flusso usando la classe Flow
                if flow_key not in flows:
                    if is_forward:
                        flows[flow_key] = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                    else:
                        flows[flow_key] = Flow(dst_ip, src_ip, dst_port, src_port, protocol)
                    # Imposta start_time dal pacchetto PCAP, non da time.time()
                    flows[flow_key].start_time = timestamp
                    flows[flow_key].last_time = timestamp
                
                flow = flows[flow_key]
                flow.add_packet(pkt_len, is_forward, timestamp, tcp_flags)
                
                results['packets_processed'] += 1
                
                # Progress e cleanup periodico
                if results['packets_processed'] % progress_interval == 0:
                    elapsed = time.time() - start_time
                    pps = results['packets_processed'] / elapsed if elapsed > 0 else 0
                    
                    if show_progress and estimated_packets > 0:
                        progress_pct = min(100, results['packets_processed'] / estimated_packets * 100)
                        remaining_pkts = max(0, estimated_packets - results['packets_processed'])
                        eta_seconds = remaining_pkts / pps if pps > 0 else 0
                        eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s" if eta_seconds > 0 else "???"
                        
                        print(f"  [{progress_pct:5.1f}%] Pkts: {results['packets_processed']:,} | "
                              f"Flows: {len(flows):,} | Attacks: {results['attacks_detected']} | "
                              f"Speed: {pps/1000:.1f}k/s | ETA: {eta_str}")
                    else:
                        print(f"  Packets: {results['packets_processed']:,} | Flows: {len(flows):,} | Attacks: {results['attacks_detected']}")
                    
                    # Cleanup flussi scaduti (usando timestamp PCAP, non time.time())
                    expired_keys = [
                        k for k, f in flows.items() 
                        if (timestamp - f.last_time) > timeout
                    ]
                    
                    for key in expired_keys:
                        analyze_and_clear_flow(key, flows.pop(key))
        
        # Analizza flussi rimanenti
        print(f"\nAnalisi flussi rimanenti: {len(flows):,}")
        for key, flow in flows.items():
            analyze_and_clear_flow(key, flow)
        
    except Exception as e:
        print(f"ERRORE: {e}")
        results['error'] = str(e)
        raise
    
    # Calcola statistiche finali
    if results['flows_analyzed'] > 0:
        results['detection_rate'] = results['attacks_detected'] / results['flows_analyzed'] * 100
    else:
        results['detection_rate'] = 0
    
    # Statistiche probabilità
    if results['probabilities']:
        probs = np.array(results['probabilities'])
        results['stats'] = {
            'prob_mean': float(np.mean(probs)),
            'prob_std': float(np.std(probs)),
            'prob_min': float(np.min(probs)),
            'prob_max': float(np.max(probs)),
            'prob_median': float(np.median(probs)),
            'above_threshold': int(np.sum(probs >= threshold)),
            'below_threshold': int(np.sum(probs < threshold))
        }
    
    # Rimuovi lista probabilità (troppo grande)
    del results['probabilities']
    
    # Riepilogo
    print(f"\n{'='*60}")
    print(f"RISULTATI: {pcap_path.name}")
    print(f"{'='*60}")
    print(f"Pacchetti processati: {results['packets_processed']:,}")
    print(f"Flussi analizzati:    {results['flows_analyzed']:,}")
    print(f"Attacchi rilevati:    {results['attacks_detected']:,} ({results['detection_rate']:.1f}%)")
    print(f"Benigni:              {results['benign_detected']:,}")
    
    if 'stats' in results:
        print(f"\nDistribuzione probabilità:")
        print(f"  Media:   {results['stats']['prob_mean']:.3f}")
        print(f"  Mediana: {results['stats']['prob_median']:.3f}")
        print(f"  Min:     {results['stats']['prob_min']:.3f}")
        print(f"  Max:     {results['stats']['prob_max']:.3f}")
        print(f"  >= {threshold}: {results['stats']['above_threshold']:,}")
        print(f"  <  {threshold}: {results['stats']['below_threshold']:,}")
    
    if results['attack_flows']:
        print(f"\nTop 5 attacchi (per probabilità):")
        for att in sorted(results['attack_flows'], key=lambda x: x['probability'], reverse=True)[:5]:
            print(f"  - {att['flow_id']}: {att['probability']:.3f} ({att['packets']} pkts)")
    
    return results


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='NIDS Network Sniffer v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Live capture con best model
  sudo python src/sniffer.py --interface eth0
  
  # Analisi PCAP
  python src/sniffer.py --pcap capture.pcap
  
  # Prevention mode
  sudo python src/sniffer.py --interface eth0 --mode prevention
  
  # Modello specifico + verbose
  sudo python src/sniffer.py --interface eth0 --model-path models/xgboost/model_binary.pkl --verbose
        """
    )
    
    # Sorgente dati
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--interface', type=str, default=None,
                        help='Interfaccia rete per cattura live')
    source.add_argument('--pcap', type=Path, default=None,
                        help='File PCAP da analizzare')
    
    # Modello
    parser.add_argument('--model-path', type=Path, default=None,
                        help='Path modello (default: models/best_model/model_binary.pkl)')
    
    # Modalita
    parser.add_argument('--mode', type=str, choices=['detection', 'prevention'],
                        default='detection',
                        help='Modalita: detection (solo log) o prevention (log + firewall)')
    
    # Analisi
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'Timeout flussi in secondi (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Soglia probabilita alert (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--min-packets', type=int, default=DEFAULT_MIN_PACKETS,
                        help=f'Minimo pacchetti per analisi (default: {DEFAULT_MIN_PACKETS})')
    
    # Output
    parser.add_argument('--log-dir', type=Path, default=None,
                        help='Directory log')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostra tutti i flussi')
    parser.add_argument('--quiet', action='store_true',
                        help='Mostra solo alert critici')
    
    # Controllo
    parser.add_argument('--packet-count', type=int, default=0,
                        help='Pacchetti da processare (0 = infinito)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Durata cattura in secondi (0 = infinito)')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    print("\n" + "=" * 60)
    print("NIDS NETWORK SNIFFER v2.0")
    print("=" * 60)
    
    # Verifica Scapy
    if not SCAPY_AVAILABLE:
        print("\nERRORE: Scapy non installato")
        print("Eseguire: pip install scapy")
        sys.exit(1)
    
    # Verifica sorgente dati
    if args.interface is None and args.pcap is None:
        print("\nERRORE: Specificare --interface o --pcap")
        sys.exit(1)
    
    # Verifica privilegi per live capture
    if args.interface and os.geteuid() != 0:
        print("\nERRORE: Cattura live richiede privilegi root")
        print("Eseguire con: sudo python src/sniffer.py ...")
        sys.exit(1)
    
    try:
        sniffer = NIDSSniffer(
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
            sniffer.start_pcap(packet_count=args.packet_count)
        else:
            sniffer.start_live(packet_count=args.packet_count, duration=args.duration)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()