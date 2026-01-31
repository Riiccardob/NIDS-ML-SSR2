"""
NIDS-ML Sniffer - Main Engine (Corrected)

Cattura e analisi traffico di rete con ML.
Supporta: cattura live, analisi PCAP.

CORREZIONI:
- Memory leak prevention con garbage collection periodico
- Gestione versioni sklearn con warning appropriato
- Supporto completo per analisi PCAP senza limiti di default
- Progress tracking migliorato
- Statistiche dettagliate per ogni sessione
"""

import os
import gc
import time
import signal
import logging
import warnings
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set
from collections import defaultdict

warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import pandas as pd
import joblib

from .flow import Flow, FlowManager, PacketInfo, DEFAULT_FLOW_TIMEOUT
from .features import FeatureExtractor
from .preprocessing import load_pipeline_artifacts, InferencePipeline

logger = logging.getLogger(__name__)


GC_INTERVAL_SECONDS = 10.0
MAX_FLOW_AGE_SECONDS = 120.0
PROGRESS_INTERVAL = 50000


@dataclass
class PredictionResult:
    """Risultato predizione per singolo flusso."""
    flow_key: Tuple
    prediction: int
    label: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 0
    packets: int = 0
    bytes: int = 0
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'flow_key': str(self.flow_key),
            'prediction': self.prediction,
            'label': self.label,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'packets': self.packets,
            'bytes': self.bytes,
            'duration_ms': self.duration_ms
        }


@dataclass
class SessionStats:
    """Statistiche sessione di analisi."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    packets_processed: int = 0
    packets_skipped: int = 0
    flows_analyzed: int = 0
    attacks_detected: int = 0
    benign_flows: int = 0
    flows_expired: int = 0
    flows_completed: int = 0
    unique_src_ips: Set[str] = field(default_factory=set)
    unique_dst_ips: Set[str] = field(default_factory=set)
    predictions_by_label: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    attack_ips: Set[str] = field(default_factory=set)
    
    @property
    def duration(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def packets_per_second(self) -> float:
        if self.duration > 0:
            return self.packets_processed / self.duration
        return 0.0
    
    @property
    def flows_per_second(self) -> float:
        if self.duration > 0:
            return self.flows_analyzed / self.duration
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration,
            'packets_processed': self.packets_processed,
            'packets_skipped': self.packets_skipped,
            'packets_per_second': self.packets_per_second,
            'flows_analyzed': self.flows_analyzed,
            'flows_per_second': self.flows_per_second,
            'attacks_detected': self.attacks_detected,
            'benign_flows': self.benign_flows,
            'flows_expired': self.flows_expired,
            'flows_completed': self.flows_completed,
            'unique_src_ips': len(self.unique_src_ips),
            'unique_dst_ips': len(self.unique_dst_ips),
            'predictions_by_label': dict(self.predictions_by_label),
            'attack_ips': list(self.attack_ips)[:100]
        }


class PacketProcessor:
    """Estrae info da pacchetti Scapy."""
    
    TCP_FLAGS = {
        'FIN': 0x01, 'SYN': 0x02, 'RST': 0x04, 'PSH': 0x08,
        'ACK': 0x10, 'URG': 0x20, 'ECE': 0x40, 'CWR': 0x80
    }
    
    @classmethod
    def process(cls, packet) -> Optional[PacketInfo]:
        """Estrae PacketInfo da pacchetto Scapy."""
        try:
            from scapy.all import IP, TCP, UDP
        except ImportError:
            return None
        
        if not packet.haslayer(IP):
            return None
        
        ip = packet[IP]
        
        if packet.haslayer(TCP):
            t = packet[TCP]
            protocol = 6
            src_port = t.sport
            dst_port = t.dport
            window_size = t.window
            tcp_flags = {n: bool(int(t.flags) & b) for n, b in cls.TCP_FLAGS.items()}
            tcp_header_len = t.dataofs * 4 if hasattr(t, 'dataofs') and t.dataofs else 20
            header_len = (ip.ihl * 4) + tcp_header_len
        elif packet.haslayer(UDP):
            t = packet[UDP]
            protocol = 17
            src_port = t.sport
            dst_port = t.dport
            window_size = 0
            tcp_flags = {}
            header_len = (ip.ihl * 4) + 8
        else:
            return None
        
        payload_size = max(0, ip.len - header_len)
        
        return PacketInfo(
            timestamp=float(packet.time),
            src_ip=ip.src,
            dst_ip=ip.dst,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            payload_size=payload_size,
            header_length=header_len,
            tcp_flags=tcp_flags,
            window_size=window_size
        )


class SnifferEngine:
    """
    Motore principale NIDS per analisi traffico.
    
    Supporta:
    - Analisi PCAP (file completi senza limiti di default)
    - Cattura live con garbage collection
    - Predizione ML con pipeline completa
    - Statistiche dettagliate
    """
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts',
        log_dir: Optional[str] = None,
        flow_timeout: float = DEFAULT_FLOW_TIMEOUT,
        max_packets_per_flow: int = 500,
        confidence_threshold: float = 0.5,
        firewall_enabled: bool = False,
        firewall_dry_run: bool = True,
        gc_interval: float = GC_INTERVAL_SECONDS,
        max_flow_age: float = MAX_FLOW_AGE_SECONDS
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.log_dir = Path(log_dir) if log_dir else None
        self.confidence_threshold = confidence_threshold
        self.firewall_enabled = firewall_enabled
        self.firewall_dry_run = firewall_dry_run
        self.gc_interval = gc_interval
        self.max_flow_age = max_flow_age
        self.logger = logging.getLogger('sniffer.engine')
        
        self._load_artifacts()
        
        self.flow_manager = FlowManager(
            flow_timeout=flow_timeout,
            max_packets=max_packets_per_flow
        )
        self.feature_extractor = FeatureExtractor()
        self.stats = SessionStats()
        self._running = False
        self._gc_thread = None
        self._last_gc_time = time.time()
        
        self.logger.info(
            f"Engine inizializzato: {len(self.pipeline.scaler_columns)} -> "
            f"{len(self.pipeline.selected_features)} features"
        )
    
    def _load_artifacts(self):
        """Carica modello e pipeline."""
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato in {self.model_dir}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*version.*')
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                self.logger.warning(f"Warning durante caricamento modello: {e}")
                self.model = joblib.load(model_path)
        
        self.logger.info(f"Modello: {type(self.model).__name__}")
        
        artifacts = load_pipeline_artifacts(str(self.artifacts_dir), str(self.model_dir))
        self.pipeline = InferencePipeline(artifacts)
        
        labels_path = self.artifacts_dir / 'label_encoder.pkl'
        if labels_path.exists():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.label_encoder = joblib.load(labels_path)
        else:
            self.label_encoder = None
        
        self.label_map = {0: 'BENIGN', 1: 'ATTACK'}
    
    def _get_label(self, pred: int) -> str:
        """Converte predizione numerica in label testuale."""
        if self.label_encoder:
            try:
                return self.label_encoder.inverse_transform([pred])[0]
            except Exception:
                pass
        return self.label_map.get(pred, f'CLASS_{pred}')
    
    def analyze_flow(self, flow: Flow) -> Optional[PredictionResult]:
        """Analizza un flusso e restituisce la predizione."""
        try:
            features = self.feature_extractor.extract(flow)
            X = self.pipeline.transform(features)
            
            pred = self.model.predict(X)[0]
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                conf = float(proba[pred])
            else:
                conf = 1.0
            
            label = self._get_label(pred)
            
            self.stats.flows_analyzed += 1
            self.stats.predictions_by_label[label] += 1
            
            result = PredictionResult(
                flow_key=flow.flow_key,
                prediction=int(pred),
                label=label,
                confidence=conf,
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                src_port=flow.src_port,
                dst_port=flow.dst_port,
                protocol=flow.protocol,
                packets=flow.total_packets,
                bytes=flow.total_bytes,
                duration_ms=flow.duration * 1000
            )
            
            if label == 'BENIGN':
                self.stats.benign_flows += 1
            else:
                self.stats.attacks_detected += 1
                self.stats.attack_ips.add(flow.src_ip)
                
                if conf >= self.confidence_threshold:
                    self.logger.warning(
                        f"ATTACK DETECTED: {label} | {flow.src_ip}:{flow.src_port} -> "
                        f"{flow.dst_ip}:{flow.dst_port} | confidence: {conf:.1%} | "
                        f"packets: {flow.total_packets}"
                    )
                    if self.firewall_enabled:
                        self._block_ip(flow.src_ip)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Errore analisi flusso {flow.flow_key}: {e}")
            return None
    
    def _block_ip(self, ip: str):
        """Blocca IP tramite iptables."""
        if self.firewall_dry_run:
            self.logger.info(f"[DRY-RUN] Blocco IP: {ip}")
            return
        
        import subprocess
        try:
            cmd = ['iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP']
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"IP bloccato: {ip}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Errore blocco IP {ip}: {e}")
    
    def _process_packet(self, packet):
        """Processa un singolo pacchetto."""
        pkt = PacketProcessor.process(packet)
        if not pkt:
            self.stats.packets_skipped += 1
            return
        
        self.stats.packets_processed += 1
        self.stats.unique_src_ips.add(pkt.src_ip)
        self.stats.unique_dst_ips.add(pkt.dst_ip)
        
        flow = self.flow_manager.add_packet(pkt)
        if flow:
            self.stats.flows_completed += 1
            self.analyze_flow(flow)
        
        current_time = time.time()
        if current_time - self._last_gc_time > self.gc_interval:
            self._garbage_collect(pkt.timestamp)
            self._last_gc_time = current_time
    
    def _garbage_collect(self, current_time: float):
        """Garbage collection per flussi scaduti."""
        expired = self.flow_manager.expire_flows(current_time)
        for flow in expired:
            self.analyze_flow(flow)
            self.stats.flows_expired += 1
        
        if expired:
            self.logger.debug(f"GC: {len(expired)} flussi scaduti analizzati")
        
        gc.collect()
    
    def analyze_pcap(
        self,
        pcap_path: str,
        max_packets: Optional[int] = None,
        progress_interval: int = PROGRESS_INTERVAL,
        verbose: bool = True
    ) -> List[PredictionResult]:
        """
        Analizza file PCAP.
        
        Args:
            pcap_path: Path al file PCAP
            max_packets: Limite pacchetti (None = TUTTI i pacchetti)
            progress_interval: Intervallo stampa progresso
            verbose: Output verboso
        
        Returns:
            Lista di PredictionResult per tutti i flussi
        """
        from scapy.utils import PcapReader
        
        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP non trovato: {pcap_path}")
        
        file_size_mb = pcap_path.stat().st_size / (1024 ** 2)
        
        if verbose:
            print(f"\nAnalisi PCAP: {pcap_path}")
            print(f"Dimensione: {file_size_mb:.1f} MB")
            print(f"Max packets: {max_packets if max_packets else 'TUTTI'}")
            print("-" * 60)
        
        self.logger.info(f"PCAP: {pcap_path} ({file_size_mb:.1f} MB)")
        
        results = []
        self.stats = SessionStats()
        self.flow_manager = FlowManager(
            flow_timeout=self.flow_manager.flow_timeout,
            max_packets=self.flow_manager.max_packets
        )
        
        packet_count = 0
        last_progress_time = time.time()
        last_progress_count = 0
        last_packet_time = None
        
        try:
            with PcapReader(str(pcap_path)) as reader:
                for packet in reader:
                    if max_packets and packet_count >= max_packets:
                        break
                    
                    self._process_packet(packet)
                    packet_count += 1
                    
                    try:
                        last_packet_time = float(packet.time)
                    except Exception:
                        pass
                    
                    if packet_count % progress_interval == 0:
                        elapsed = time.time() - last_progress_time
                        rate = (packet_count - last_progress_count) / elapsed if elapsed > 0 else 0
                        
                        msg = (
                            f"Packets: {packet_count:,} | {rate:,.0f} pkt/s | "
                            f"Flows: {self.stats.flows_analyzed:,} | "
                            f"Attacks: {self.stats.attacks_detected:,} | "
                            f"Active: {self.flow_manager.get_flow_count():,}"
                        )
                        
                        if verbose:
                            print(msg)
                        self.logger.info(msg)
                        
                        last_progress_time = time.time()
                        last_progress_count = packet_count
                    
                    if last_packet_time and packet_count % 10000 == 0:
                        self._garbage_collect(last_packet_time)
        
        except Exception as e:
            self.logger.error(f"Errore lettura PCAP: {e}")
            raise
        
        remaining_flows = self.flow_manager.get_all_flows()
        for flow in remaining_flows:
            r = self.analyze_flow(flow)
            if r:
                results.append(r)
        
        self.stats.end_time = datetime.now()
        
        if verbose:
            self._print_summary()
        
        attack_results = [r for r in results if r.label != 'BENIGN']
        results = attack_results
        
        return results
    
    def start_live(
        self,
        interface: str = 'eth0',
        duration: Optional[int] = None,
        filter_str: str = 'ip',
        promisc: bool = True,
        verbose: bool = True
    ):
        """
        Cattura live da interfaccia di rete.
        
        Args:
            interface: Nome interfaccia (es. eth0, wlan0)
            duration: Durata in secondi (None = indefinita)
            filter_str: Filtro BPF
            promisc: Modalita promiscua
            verbose: Output verboso
        """
        from scapy.all import sniff
        
        if verbose:
            print(f"\nLive capture su: {interface}")
            print(f"Durata: {duration if duration else 'indefinita'} secondi")
            print(f"Filtro: {filter_str}")
            print(f"Promiscuo: {promisc}")
            print("-" * 60)
            print("Premi Ctrl+C per interrompere...")
            print()
        
        self.logger.info(
            f"Live capture: interface={interface}, duration={duration}, "
            f"filter={filter_str}, promisc={promisc}"
        )
        
        self.stats = SessionStats()
        self.flow_manager = FlowManager(
            flow_timeout=self.flow_manager.flow_timeout,
            max_packets=self.flow_manager.max_packets
        )
        self._running = True
        
        def signal_handler(sig, frame):
            self._running = False
            if verbose:
                print("\nInterruzione richiesta...")
            self.logger.info("Interruzione richiesta")
        
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        self._start_gc_thread()
        
        try:
            if duration:
                sniff(
                    iface=interface,
                    filter=filter_str,
                    prn=self._process_packet,
                    store=False,
                    timeout=duration,
                    promisc=promisc
                )
            else:
                sniff(
                    iface=interface,
                    filter=filter_str,
                    prn=self._process_packet,
                    store=False,
                    stop_filter=lambda x: not self._running,
                    promisc=promisc
                )
            
            for flow in self.flow_manager.get_all_flows():
                self.analyze_flow(flow)
        
        finally:
            self._running = False
            self._stop_gc_thread()
            self.stats.end_time = datetime.now()
            
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            
            if verbose:
                self._print_summary()
    
    def _start_gc_thread(self):
        """Avvia thread garbage collection per cattura live."""
        def gc_loop():
            while self._running:
                time.sleep(self.gc_interval)
                if self._running:
                    current_time = time.time()
                    expired = self.flow_manager.expire_flows(current_time)
                    for flow in expired:
                        self.analyze_flow(flow)
                        self.stats.flows_expired += 1
                    gc.collect()
        
        self._gc_thread = threading.Thread(target=gc_loop, daemon=True)
        self._gc_thread.start()
    
    def _stop_gc_thread(self):
        """Ferma thread garbage collection."""
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=2.0)
    
    def _print_summary(self):
        """Stampa riepilogo sessione."""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration:           {self.stats.duration:.1f} seconds")
        print(f"Packets processed:  {self.stats.packets_processed:,}")
        print(f"Packets skipped:    {self.stats.packets_skipped:,}")
        print(f"Packet rate:        {self.stats.packets_per_second:,.0f} pkt/s")
        print("-" * 60)
        print(f"Flows analyzed:     {self.stats.flows_analyzed:,}")
        print(f"  - Completed:      {self.stats.flows_completed:,}")
        print(f"  - Expired (GC):   {self.stats.flows_expired:,}")
        print(f"Flow rate:          {self.stats.flows_per_second:,.0f} flows/s")
        print("-" * 60)
        print(f"Attacks detected:   {self.stats.attacks_detected:,}")
        print(f"Benign flows:       {self.stats.benign_flows:,}")
        print(f"Unique source IPs:  {len(self.stats.unique_src_ips):,}")
        print(f"Unique dest IPs:    {len(self.stats.unique_dst_ips):,}")
        
        if self.stats.predictions_by_label:
            print("\nPredictions by label:")
            for label, count in sorted(self.stats.predictions_by_label.items()):
                pct = count / self.stats.flows_analyzed * 100 if self.stats.flows_analyzed > 0 else 0
                print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        if self.stats.attack_ips:
            print(f"\nAttack source IPs (top 10):")
            for ip in list(self.stats.attack_ips)[:10]:
                print(f"  - {ip}")
            if len(self.stats.attack_ips) > 10:
                print(f"  ... and {len(self.stats.attack_ips) - 10} more")
        
        print("=" * 60)
    
    def get_stats(self) -> Dict:
        """Restituisce statistiche sessione."""
        return self.stats.to_dict()
    
    def close(self):
        """Chiude engine e rilascia risorse."""
        self._running = False
        self._stop_gc_thread()
        self.flow_manager.get_all_flows()
        gc.collect()
        self.logger.info("Engine chiuso")


class SnifferEvaluator:
    """
    Wrapper per compatibilita con vecchio codice.
    Usa evaluation.py per implementazione completa.
    """
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts'
    ):
        from .evaluation import SnifferEvaluator as EvalSnifferEvaluator
        self._evaluator = EvalSnifferEvaluator(
            model_dir=model_dir,
            artifacts_dir=artifacts_dir
        )
    
    def evaluate_csv(
        self,
        csv_path: str,
        sample_size: Optional[int] = None,
        batch_size: int = 50000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Valuta su CSV, restituisce dict per compatibilita."""
        result = self._evaluator.evaluate_csv(
            csv_path=csv_path,
            sample_size=sample_size,
            batch_size=batch_size,
            verbose=verbose
        )
        return result.to_dict()