"""
NIDS-ML Sniffer - Main Engine

Cattura e analisi traffico di rete con ML.
Supporta: cattura live, analisi PCAP, valutazione CSV.

FIXES APPLICATI:
- Memory leak prevention con garbage collection periodico
- Gestione versioni sklearn con warning appropriato
- Parametro verbose in evaluate_csv
- Metodo close() corretto
"""

import os
import time
import signal
import logging
import warnings
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import pandas as pd
import joblib

from .flow import Flow, FlowManager, PacketInfo, DEFAULT_FLOW_TIMEOUT
from .features import FeatureExtractor
from .preprocessing import load_pipeline_artifacts, InferencePipeline

logger = logging.getLogger(__name__)


# Configurazione garbage collection
GC_INTERVAL_SECONDS = 10.0
MAX_FLOW_AGE_SECONDS = 120.0


@dataclass
class PredictionResult:
    flow_key: Tuple
    prediction: int
    label: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'flow_key': str(self.flow_key), 
            'prediction': self.prediction, 
            'label': self.label, 
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SessionStats:
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    packets_processed: int = 0
    flows_analyzed: int = 0
    attacks_detected: int = 0
    benign_flows: int = 0
    flows_expired: int = 0
    unique_src_ips: set = field(default_factory=set)
    unique_dst_ips: set = field(default_factory=set)
    predictions_by_label: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def duration(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict:
        return {
            'duration_seconds': self.duration,
            'packets_processed': self.packets_processed,
            'flows_analyzed': self.flows_analyzed,
            'attacks_detected': self.attacks_detected,
            'benign_flows': self.benign_flows,
            'flows_expired': self.flows_expired,
            'unique_src_ips': len(self.unique_src_ips),
            'unique_dst_ips': len(self.unique_dst_ips),
            'predictions_by_label': dict(self.predictions_by_label),
        }


class PacketProcessor:
    """Estrae info da pacchetti Scapy."""
    
    TCP_FLAGS = {
        'FIN': 0x01, 'SYN': 0x02, 'RST': 0x04, 'PSH': 0x08, 
        'ACK': 0x10, 'URG': 0x20, 'ECE': 0x40, 'CWR': 0x80
    }
    
    @classmethod
    def process(cls, packet) -> Optional[PacketInfo]:
        try:
            from scapy.all import IP, TCP, UDP
        except ImportError:
            return None
        
        if not packet.haslayer(IP):
            return None
        
        ip = packet[IP]
        
        if packet.haslayer(TCP):
            t = packet[TCP]
            protocol, src_port, dst_port = 6, t.sport, t.dport
            window_size = t.window
            tcp_flags = {n: bool(int(t.flags) & b) for n, b in cls.TCP_FLAGS.items()}
            header_len = (ip.ihl * 4) + (t.dataofs * 4 if hasattr(t, 'dataofs') else 20)
        elif packet.haslayer(UDP):
            t = packet[UDP]
            protocol, src_port, dst_port = 17, t.sport, t.dport
            window_size, tcp_flags = 0, {}
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
    Motore principale NIDS.
    
    FIX: Aggiunto garbage collection periodico per prevenire memory leak
    durante cattura live prolungata o con traffico anomalo (IP spoofing, slowloris).
    """
    
    def __init__(
        self, 
        model_dir: str = 'models/best_model', 
        artifacts_dir: str = 'artifacts',
        log_dir: str = None,
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
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato in {self.model_dir}")
        
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            if 'InconsistentVersionWarning' in str(type(e).__name__) or 'version' in str(e).lower():
                self.logger.warning(
                    f"Warning versione sklearn durante caricamento modello. "
                    f"Considera di ri-trainare il modello con la versione corrente di sklearn."
                )
            self.model = joblib.load(model_path)
        
        self.logger.info(f"Modello: {type(self.model).__name__}")
        
        artifacts = load_pipeline_artifacts(str(self.artifacts_dir), str(self.model_dir))
        self.pipeline = InferencePipeline(artifacts)
        
        labels_path = self.artifacts_dir / 'label_encoder.pkl'
        self.label_encoder = joblib.load(labels_path) if labels_path.exists() else None
        self.label_map = {0: 'BENIGN', 1: 'ATTACK'}
    
    def _get_label(self, pred: int) -> str:
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
            
            if label == 'BENIGN':
                self.stats.benign_flows += 1
            else:
                self.stats.attacks_detected += 1
                if conf >= self.confidence_threshold:
                    self.logger.warning(
                        f"ATTACK: {label} | {flow.src_ip}:{flow.src_port} -> "
                        f"{flow.dst_ip}:{flow.dst_port} | {conf:.1%}"
                    )
                    if self.firewall_enabled:
                        self._block_ip(flow.src_ip)
            
            return PredictionResult(flow.flow_key, int(pred), label, conf)
        
        except Exception as e:
            self.logger.error(f"Errore analisi flusso: {e}")
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
            return
        
        self.stats.packets_processed += 1
        self.stats.unique_src_ips.add(pkt.src_ip)
        self.stats.unique_dst_ips.add(pkt.dst_ip)
        
        flow = self.flow_manager.add_packet(pkt)
        if flow:
            self.analyze_flow(flow)
        
        current_time = time.time()
        if current_time - self._last_gc_time > self.gc_interval:
            self._garbage_collect(pkt.timestamp)
            self._last_gc_time = current_time
    
    def _garbage_collect(self, current_time: float):
        """
        Garbage collection per flussi scaduti.
        FIX: Previene memory leak durante cattura prolungata.
        """
        expired = self.flow_manager.expire_flows(current_time)
        for flow in expired:
            self.analyze_flow(flow)
            self.stats.flows_expired += 1
        
        if expired:
            self.logger.debug(f"GC: {len(expired)} flussi scaduti analizzati")
    
    def analyze_pcap(
        self, 
        pcap_path: str, 
        max_packets: Optional[int] = None, 
        progress_interval: int = 50000,
        verbose: bool = False
    ) -> List[PredictionResult]:
        """
        Analizza file PCAP.
        
        Args:
            pcap_path: Path al file PCAP
            max_packets: Limite pacchetti (None = tutti)
            progress_interval: Intervallo log progresso
            verbose: Output verboso
        
        Returns:
            Lista di PredictionResult
        """
        from scapy.utils import PcapReader
        
        self.logger.info(f"PCAP: {pcap_path}")
        file_size = os.path.getsize(pcap_path) / (1024**2)
        self.logger.info(f"Dimensione: {file_size:.1f} MB, Max packets: {max_packets or 'tutti'}")
        
        results = []
        self.stats = SessionStats()
        
        packet_count = 0
        last_progress = time.time()
        last_packet_time = None
        
        try:
            with PcapReader(pcap_path) as reader:
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
                        elapsed = time.time() - last_progress
                        rate = progress_interval / elapsed if elapsed > 0 else 0
                        msg = (
                            f"Processed {packet_count:,} | {rate:.0f} pkt/s | "
                            f"Flows: {self.stats.flows_analyzed} | "
                            f"Attacks: {self.stats.attacks_detected} | "
                            f"Active flows: {self.flow_manager.get_flow_count()}"
                        )
                        if verbose:
                            print(msg)
                        self.logger.info(msg)
                        last_progress = time.time()
                    
                    if last_packet_time and packet_count % 10000 == 0:
                        self._garbage_collect(last_packet_time)
        
        except Exception as e:
            self.logger.error(f"Errore lettura PCAP: {e}")
            raise
        
        for flow in self.flow_manager.get_all_flows():
            r = self.analyze_flow(flow)
            if r:
                results.append(r)
        
        self.stats.end_time = datetime.now()
        self._print_summary()
        
        return results
    
    def start_live(
        self, 
        interface: str = 'eth0', 
        duration: Optional[int] = None, 
        filter_str: str = 'ip',
        promisc: bool = True
    ):
        """
        Cattura live.
        
        Args:
            interface: Interfaccia di rete
            duration: Durata in secondi (None = indefinita)
            filter_str: Filtro BPF
            promisc: Modalita promiscua
        """
        from scapy.all import sniff
        
        self.logger.info(
            f"Live capture: {interface}, duration: {duration or 'indefinita'}, "
            f"filter: {filter_str}, promisc: {promisc}"
        )
        self.stats = SessionStats()
        self._running = True
        
        def handler(sig, frame):
            self._running = False
            self.logger.info("Interruzione richiesta...")
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        
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
        
        self._gc_thread = threading.Thread(target=gc_loop, daemon=True)
        self._gc_thread.start()
    
    def _stop_gc_thread(self):
        """Ferma thread garbage collection."""
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=2.0)
    
    def _print_summary(self):
        print("\n" + "="*60)
        print("RIEPILOGO")
        print("="*60)
        print(f"Durata: {self.stats.duration:.1f}s")
        print(f"Pacchetti: {self.stats.packets_processed:,}")
        print(f"Flussi analizzati: {self.stats.flows_analyzed:,}")
        print(f"Flussi scaduti (GC): {self.stats.flows_expired:,}")
        print(f"Attacchi: {self.stats.attacks_detected:,}")
        print(f"Benigni: {self.stats.benign_flows:,}")
        print(f"IP sorgente unici: {len(self.stats.unique_src_ips):,}")
        print(f"IP destinazione unici: {len(self.stats.unique_dst_ips):,}")
        if self.stats.predictions_by_label:
            print("Predizioni per label:", dict(self.stats.predictions_by_label))
        print("="*60)
    
    def get_stats(self) -> Dict:
        return self.stats.to_dict()
    
    def close(self):
        """Chiude engine e rilascia risorse."""
        self._running = False
        self._stop_gc_thread()
        self.flow_manager.get_all_flows()
        self.logger.info("Engine chiuso")


class SnifferEvaluator:
    """
    Valuta modello su CSV.
    
    FIX: 
    - Aggiunto parametro verbose a evaluate_csv
    - Restituisce dict invece di oggetto con attributo f1
    """
    
    def __init__(
        self, 
        model_dir: str = 'models/best_model', 
        artifacts_dir: str = 'artifacts'
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logging.getLogger('sniffer.evaluator')
        
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato in {self.model_dir}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*version.*')
            self.model = joblib.load(model_path)
        
        artifacts = load_pipeline_artifacts(str(self.artifacts_dir), str(self.model_dir))
        self.pipeline = InferencePipeline(artifacts)
        self.logger.info(
            f"Evaluator: {len(self.pipeline.scaler_columns)} -> "
            f"{len(self.pipeline.selected_features)} features"
        )
    
    def evaluate_csv(
        self, 
        csv_path: str, 
        sample_size: Optional[int] = None,
        verbose: bool = True,
        batch_size: int = 10000
    ) -> Dict[str, Any]:
        """
        Valuta su CSV. Restituisce metriche come dict.
        
        Args:
            csv_path: Path al file CSV
            sample_size: Numero campioni (None = tutti)
            verbose: Stampa risultati
            batch_size: Dimensione batch per predizione
        
        Returns:
            Dict con metriche (samples, accuracy, precision, recall, f1, fpr, etc.)
        """
        from sklearn.metrics import (
            confusion_matrix, f1_score, precision_score, 
            recall_score, accuracy_score
        )
        
        self.logger.info(f"Valutazione: {csv_path}")
        
        df = pd.read_csv(csv_path, low_memory=False)
        original = len(df)
        self.logger.info(f"Righe: {original:,}")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Campionate: {sample_size:,}")
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        label_col = None
        for col in df.columns:
            if col.strip().lower() == 'label':
                label_col = col
                break
        if not label_col:
            raise KeyError("Colonna 'Label' non trovata nel CSV")
        
        y_true = np.array([
            0 if str(l).strip().upper() == 'BENIGN' else 1 
            for l in df[label_col]
        ])
        
        start_time = time.perf_counter()
        X = self.pipeline.transform_dataframe(df)
        
        if len(X) <= batch_size:
            y_pred = self.model.predict(X)
        else:
            y_pred = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                y_pred.extend(self.model.predict(batch))
            y_pred = np.array(y_pred)
        
        elapsed = time.perf_counter() - start_time
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics = {
            'samples': len(y_true),
            'benign': int((y_true == 0).sum()),
            'attack': int((y_true == 1).sum()),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'tp': int(tp), 
            'tn': int(tn), 
            'fp': int(fp), 
            'fn': int(fn),
            'latency_total_ms': elapsed * 1000,
            'latency_per_sample_ms': (elapsed / len(y_true)) * 1000 if len(y_true) > 0 else 0
        }
        
        if verbose:
            print("\n" + "="*60)
            print("RISULTATI VALUTAZIONE")
            print("="*60)
            print(f"Campioni: {metrics['samples']:,} (Benign: {metrics['benign']:,}, Attack: {metrics['attack']:,})")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1:        {metrics['f1']:.4f}")
            print(f"FPR:       {metrics['fpr']:.4f}")
            print(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            print(f"Latency:   {metrics['latency_per_sample_ms']:.4f} ms/sample")
            print("="*60)
        
        return metrics