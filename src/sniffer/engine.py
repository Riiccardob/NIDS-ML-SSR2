"""
NIDS-ML Sniffer - Main Engine

Cattura e analisi traffico di rete con ML.
Supporta: cattura live, analisi PCAP, valutazione CSV.
"""

import os
import time
import signal
import logging
import warnings
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


@dataclass
class PredictionResult:
    flow_key: Tuple
    prediction: int
    label: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {'flow_key': str(self.flow_key), 'prediction': self.prediction, 
                'label': self.label, 'confidence': self.confidence}


@dataclass
class SessionStats:
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    packets_processed: int = 0
    flows_analyzed: int = 0
    attacks_detected: int = 0
    benign_flows: int = 0
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
            'predictions_by_label': dict(self.predictions_by_label),
        }


class PacketProcessor:
    """Estrae info da pacchetti Scapy."""
    TCP_FLAGS = {'FIN': 0x01, 'SYN': 0x02, 'RST': 0x04, 'PSH': 0x08, 
                 'ACK': 0x10, 'URG': 0x20, 'ECE': 0x40, 'CWR': 0x80}
    
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
            timestamp=float(packet.time), src_ip=ip.src, dst_ip=ip.dst,
            src_port=src_port, dst_port=dst_port, protocol=protocol,
            payload_size=payload_size, header_length=header_len,
            tcp_flags=tcp_flags, window_size=window_size
        )


class SnifferEngine:
    """Motore principale NIDS."""
    
    def __init__(self, model_dir: str = 'models/best_model', artifacts_dir: str = 'artifacts',
                 flow_timeout: float = DEFAULT_FLOW_TIMEOUT, max_packets_per_flow: int = 500,
                 confidence_threshold: float = 0.5):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger('sniffer.engine')
        
        self._load_artifacts()
        
        self.flow_manager = FlowManager(flow_timeout=flow_timeout, max_packets=max_packets_per_flow)
        self.feature_extractor = FeatureExtractor()
        self.stats = SessionStats()
        self._running = False
        
        self.logger.info(f"Engine inizializzato: {len(self.pipeline.scaler_columns)} → {len(self.pipeline.selected_features)} features")
    
    def _load_artifacts(self):
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato in {self.model_dir}")
        
        self.model = joblib.load(model_path)
        self.logger.info(f"Modello: {type(self.model).__name__}")
        
        artifacts = load_pipeline_artifacts(str(self.artifacts_dir), str(self.model_dir))
        self.pipeline = InferencePipeline(artifacts)
        
        labels_path = self.artifacts_dir / 'label_encoder.pkl'
        self.label_encoder = joblib.load(labels_path) if labels_path.exists() else None
        self.label_map = {0: 'BENIGN', 1: 'ATTACK'}
    
    def _get_label(self, pred: int) -> str:
        if self.label_encoder:
            return self.label_encoder.inverse_transform([pred])[0]
        return self.label_map.get(pred, f'CLASS_{pred}')
    
    def analyze_flow(self, flow: Flow) -> Optional[PredictionResult]:
        try:
            features = self.feature_extractor.extract(flow)
            X = self.pipeline.transform(features)
            
            pred = self.model.predict(X)[0]
            conf = self.model.predict_proba(X)[0][pred] if hasattr(self.model, 'predict_proba') else 1.0
            label = self._get_label(pred)
            
            self.stats.flows_analyzed += 1
            self.stats.predictions_by_label[label] += 1
            
            if label == 'BENIGN':
                self.stats.benign_flows += 1
            else:
                self.stats.attacks_detected += 1
                self.logger.warning(f" ATTACK: {label} | {flow.src_ip}:{flow.src_port} → {flow.dst_ip}:{flow.dst_port} | {conf:.1%}")
            
            return PredictionResult(flow.flow_key, int(pred), label, float(conf))
        except Exception as e:
            self.logger.error(f"Errore analisi: {e}")
            return None
    
    def _process_packet(self, packet):
        pkt = PacketProcessor.process(packet)
        if not pkt:
            return
        
        self.stats.packets_processed += 1
        self.stats.unique_src_ips.add(pkt.src_ip)
        self.stats.unique_dst_ips.add(pkt.dst_ip)
        
        flow = self.flow_manager.add_packet(pkt)
        if flow:
            self.analyze_flow(flow)
    
    def analyze_pcap(self, pcap_path: str, max_packets: Optional[int] = None, 
                     progress_interval: int = 50000) -> List[PredictionResult]:
        """Analizza file PCAP."""
        from scapy.utils import PcapReader
        
        self.logger.info(f"PCAP: {pcap_path}")
        file_size = os.path.getsize(pcap_path) / (1024**2)
        self.logger.info(f"Dimensione: {file_size:.1f} MB, Max packets: {max_packets or 'tutti'}")
        
        results = []
        self.stats = SessionStats()
        
        packet_count = 0
        last_progress = time.time()
        
        with PcapReader(pcap_path) as reader:
            for packet in reader:
                if max_packets and packet_count >= max_packets:
                    break
                
                self._process_packet(packet)
                packet_count += 1
                
                if packet_count % progress_interval == 0:
                    elapsed = time.time() - last_progress
                    rate = progress_interval / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Processed {packet_count:,} | {rate:.0f} pkt/s | Flows: {self.stats.flows_analyzed} | Attacks: {self.stats.attacks_detected}")
                    last_progress = time.time()
        
        # Flussi rimanenti
        for flow in self.flow_manager.get_all_flows():
            r = self.analyze_flow(flow)
            if r:
                results.append(r)
        
        self.stats.end_time = datetime.now()
        self._print_summary()
        return results
    
    def start_live(self, interface: str = 'eth0', duration: Optional[int] = None, filter_str: str = 'ip'):
        """Cattura live."""
        from scapy.all import sniff
        
        self.logger.info(f"Live capture: {interface}, duration: {duration or 'indefinita'}")
        self.stats = SessionStats()
        self._running = True
        
        def handler(sig, frame):
            self._running = False
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        
        try:
            if duration:
                sniff(iface=interface, filter=filter_str, prn=self._process_packet, store=False, timeout=duration)
            else:
                sniff(iface=interface, filter=filter_str, prn=self._process_packet, store=False, stop_filter=lambda x: not self._running)
            
            for flow in self.flow_manager.get_all_flows():
                self.analyze_flow(flow)
        finally:
            self.stats.end_time = datetime.now()
            self._print_summary()
    
    def _print_summary(self):
        print("\n" + "="*60)
        print("RIEPILOGO")
        print("="*60)
        print(f"Durata: {self.stats.duration:.1f}s")
        print(f"Pacchetti: {self.stats.packets_processed:,}")
        print(f"Flussi: {self.stats.flows_analyzed:,}")
        print(f"Attacchi: {self.stats.attacks_detected:,}")
        print(f"Benigni: {self.stats.benign_flows:,}")
        if self.stats.predictions_by_label:
            print("Per label:", dict(self.stats.predictions_by_label))
        print("="*60)
    
    def get_stats(self) -> Dict:
        return self.stats.to_dict()


class SnifferEvaluator:
    """Valuta modello su CSV."""
    
    def __init__(self, model_dir: str = 'models/best_model', artifacts_dir: str = 'artifacts'):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logging.getLogger('sniffer.evaluator')
        
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        self.model = joblib.load(model_path)
        
        artifacts = load_pipeline_artifacts(str(self.artifacts_dir), str(self.model_dir))
        self.pipeline = InferencePipeline(artifacts)
        self.logger.info(f"Evaluator: {len(self.pipeline.scaler_columns)} → {len(self.pipeline.selected_features)} features")
    
    def evaluate_csv(self, csv_path: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Valuta su CSV. Restituisce metriche."""
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
        from tqdm import tqdm
        
        self.logger.info(f"Valutazione: {csv_path}")
        
        df = pd.read_csv(csv_path, low_memory=False)
        original = len(df)
        self.logger.info(f"Righe: {original:,}")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Campionate: {sample_size:,}")
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Trova colonna label
        label_col = None
        for col in df.columns:
            if col.strip().lower() == 'label':
                label_col = col
                break
        if not label_col:
            raise KeyError("Colonna 'Label' non trovata")
        
        y_true = np.array([0 if str(l).upper() == 'BENIGN' else 1 for l in df[label_col]])
        
        X = self.pipeline.transform_dataframe(df)
        
        self.logger.info("Predizione...")
        y_pred = self.model.predict(X)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics = {
            'samples': len(y_true),
            'benign': int((y_true == 0).sum()),
            'attack': int((y_true == 1).sum()),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        }
        
        print("\n" + "="*60)
        print("RISULTATI")
        print("="*60)
        print(f"Campioni: {metrics['samples']:,} (Benign: {metrics['benign']:,}, Attack: {metrics['attack']:,})")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"FPR:       {metrics['fpr']:.4f}")
        print(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print("="*60)
        
        return metrics