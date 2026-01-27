"""
NIDS Sniffer Engine - Core detection system
============================================

This module implements the main sniffer engine that:
1. Captures packets (live or from PCAP files)
2. Aggregates them into flows using FlowManager
3. Extracts features using FeatureExtractor
4. Makes predictions using the trained ML model
5. Handles responses (logging, alerting, firewall blocking)

Usage:
    # Live capture
    engine = SnifferEngine(model_dir='models/best_model')
    engine.start_live(interface='eth0', duration=300)
    
    # PCAP analysis
    results = engine.analyze_pcap('capture.pcap')
"""

import os
import json
import time
import signal
import logging
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple
from collections import defaultdict

# Suppress sklearn feature name warnings (we use numpy arrays for speed)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, rdpcap, IP, TCP, UDP, Packet

from .flow import Flow, FlowManager, DEFAULT_FLOW_TIMEOUT
from .features import FeatureExtractor, FEATURE_NAMES, CRITICAL_FEATURES, get_feature_columns_ordered


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PacketInfo:
    """Extracted information from a single packet."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP
    payload_size: int
    header_length: int
    tcp_flags: Dict[str, bool] = field(default_factory=dict)
    window_size: int = 0
    
    @property
    def flow_key(self) -> Tuple[str, str, int, int, int]:
        """Generate bidirectional flow key (sorted to ensure same key both directions)."""
        if (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port):
            return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
        return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)
    
    @property
    def is_forward(self) -> bool:
        """Check if packet is in forward direction (src < dst)."""
        return (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port)


@dataclass
class PredictionResult:
    """Result of a single flow prediction."""
    flow_key: Tuple
    prediction: int
    label: str
    confidence: float
    features: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'flow_key': str(self.flow_key),
            'prediction': self.prediction,
            'label': self.label,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'features': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in self.features.items()}
        }


@dataclass
class SessionStats:
    """Statistics for a sniffer session."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    packets_processed: int = 0
    flows_analyzed: int = 0
    attacks_detected: int = 0
    benign_flows: int = 0
    unique_src_ips: set = field(default_factory=set)
    unique_dst_ips: set = field(default_factory=set)
    predictions_by_label: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        duration = (self.end_time or datetime.now()) - self.start_time
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration.total_seconds(),
            'packets_processed': self.packets_processed,
            'flows_analyzed': self.flows_analyzed,
            'attacks_detected': self.attacks_detected,
            'benign_flows': self.benign_flows,
            'unique_src_ips': len(self.unique_src_ips),
            'unique_dst_ips': len(self.unique_dst_ips),
            'predictions_by_label': dict(self.predictions_by_label),
            'errors_count': len(self.errors)
        }


# =============================================================================
# PACKET PROCESSOR
# =============================================================================

class PacketProcessor:
    """Extracts relevant information from Scapy packets."""
    
    # TCP flag bit positions
    TCP_FLAGS = {
        'FIN': 0x01,
        'SYN': 0x02,
        'RST': 0x04,
        'PSH': 0x08,
        'ACK': 0x10,
        'URG': 0x20,
        'ECE': 0x40,
        'CWR': 0x80
    }
    
    @classmethod
    def process(cls, packet: Packet) -> Optional[PacketInfo]:
        """
        Extract PacketInfo from a Scapy packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            PacketInfo if packet has IP layer, None otherwise
        """
        if not packet.haslayer(IP):
            return None
        
        ip_layer = packet[IP]
        
        # Determine protocol and ports
        if packet.haslayer(TCP):
            protocol = 6
            transport = packet[TCP]
            src_port = transport.sport
            dst_port = transport.dport
            window_size = transport.window
            tcp_flags = cls._extract_tcp_flags(transport.flags)
        elif packet.haslayer(UDP):
            protocol = 17
            transport = packet[UDP]
            src_port = transport.sport
            dst_port = transport.dport
            window_size = 0
            tcp_flags = {}
        else:
            return None  # Only handle TCP/UDP
        
        # Calculate header length (IP header + TCP/UDP header)
        ip_header_len = ip_layer.ihl * 4  # IHL is in 32-bit words
        if protocol == 6:
            tcp_header_len = transport.dataofs * 4 if hasattr(transport, 'dataofs') else 20
            header_length = ip_header_len + tcp_header_len
        else:
            header_length = ip_header_len + 8  # UDP header is always 8 bytes
        
        # Calculate payload size
        total_len = ip_layer.len
        payload_size = max(0, total_len - header_length)
        
        return PacketInfo(
            timestamp=float(packet.time),
            src_ip=ip_layer.src,
            dst_ip=ip_layer.dst,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            payload_size=payload_size,
            header_length=header_length,
            tcp_flags=tcp_flags,
            window_size=window_size
        )
    
    @classmethod
    def _extract_tcp_flags(cls, flags) -> Dict[str, bool]:
        """Extract TCP flags from Scapy flags field."""
        flag_int = int(flags)
        return {name: bool(flag_int & bit) for name, bit in cls.TCP_FLAGS.items()}


# =============================================================================
# LOGGING SYSTEM
# =============================================================================

class SnifferLogger:
    """Handles logging for sniffer sessions with structured output."""
    
    def __init__(self, log_dir: str = 'logs', session_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup loggers
        self._setup_general_logger()
        self._setup_attack_logger()
        self._setup_flow_logger()
    
    def _setup_general_logger(self):
        """Setup main application logger."""
        self.logger = logging.getLogger(f'sniffer.{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f'sniffer_{self.session_id}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        ))
        
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
    
    def _setup_attack_logger(self):
        """Setup attack detection logger (JSONL format)."""
        self.attack_log_path = self.log_dir / f'attacks_{self.session_id}.jsonl'
        self.attack_file = open(self.attack_log_path, 'a')
    
    def _setup_flow_logger(self):
        """Setup flow logger for all analyzed flows (JSONL format)."""
        self.flow_log_path = self.log_dir / f'flows_{self.session_id}.jsonl'
        self.flow_file = open(self.flow_log_path, 'a')
    
    def log_attack(self, result: PredictionResult, flow: Flow):
        """Log detected attack to attacks JSONL file."""
        entry = {
            'timestamp': result.timestamp.isoformat(),
            'flow_key': str(result.flow_key),
            'label': result.label,
            'confidence': result.confidence,
            'src_ip': flow.src_ip,
            'dst_ip': flow.dst_ip,
            'src_port': flow.src_port,
            'dst_port': flow.dst_port,
            'protocol': flow.protocol,
            'packets': flow.total_packets,
            'bytes': flow.total_bytes,
            'duration_ms': flow.duration * 1e6  # Convert to microseconds
        }
        self.attack_file.write(json.dumps(entry) + '\n')
        self.attack_file.flush()
    
    def log_flow(self, result: PredictionResult):
        """Log analyzed flow to flows JSONL file."""
        self.flow_file.write(json.dumps(result.to_dict()) + '\n')
        self.flow_file.flush()
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def close(self):
        """Close file handles."""
        if hasattr(self, 'attack_file'):
            self.attack_file.close()
        if hasattr(self, 'flow_file'):
            self.flow_file.close()


# =============================================================================
# FIREWALL MANAGER
# =============================================================================

class FirewallManager:
    """Manages iptables rules for blocking malicious IPs."""
    
    def __init__(self, enabled: bool = False, dry_run: bool = True, 
                 chain: str = 'INPUT', logger: Optional[SnifferLogger] = None):
        """
        Initialize firewall manager.
        
        Args:
            enabled: Whether firewall blocking is enabled
            dry_run: If True, only log what would be done without executing
            chain: iptables chain to use (INPUT, OUTPUT, FORWARD)
            logger: Optional logger instance
        """
        self.enabled = enabled
        self.dry_run = dry_run
        self.chain = chain
        self.logger = logger
        self.blocked_ips: Dict[str, datetime] = {}
    
    def block_ip(self, ip: str, reason: str = 'attack_detected') -> bool:
        """
        Block an IP address using iptables.
        
        Args:
            ip: IP address to block
            reason: Reason for blocking (for logging)
            
        Returns:
            True if successfully blocked (or would be blocked in dry_run)
        """
        if not self.enabled:
            return False
        
        if ip in self.blocked_ips:
            return True  # Already blocked
        
        # Validate IP format (basic check)
        parts = ip.split('.')
        if len(parts) != 4:
            if self.logger:
                self.logger.warning(f"Invalid IP format: {ip}")
            return False
        
        cmd = ['iptables', '-A', self.chain, '-s', ip, '-j', 'DROP']
        
        if self.dry_run:
            if self.logger:
                self.logger.info(f"[DRY-RUN] Would block IP: {ip} (reason: {reason})")
            self.blocked_ips[ip] = datetime.now()
            return True
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.blocked_ips[ip] = datetime.now()
            if self.logger:
                self.logger.info(f"Blocked IP: {ip} (reason: {reason})")
            return True
        except subprocess.CalledProcessError as e:
            if self.logger:
                self.logger.error(f"Failed to block IP {ip}: {e}")
            return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Remove block for an IP address."""
        if ip not in self.blocked_ips:
            return True
        
        cmd = ['iptables', '-D', self.chain, '-s', ip, '-j', 'DROP']
        
        if self.dry_run:
            if self.logger:
                self.logger.info(f"[DRY-RUN] Would unblock IP: {ip}")
            del self.blocked_ips[ip]
            return True
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            del self.blocked_ips[ip]
            if self.logger:
                self.logger.info(f"Unblocked IP: {ip}")
            return True
        except subprocess.CalledProcessError as e:
            if self.logger:
                self.logger.error(f"Failed to unblock IP {ip}: {e}")
            return False
    
    def get_blocked_ips(self) -> Dict[str, datetime]:
        """Return dictionary of blocked IPs and when they were blocked."""
        return self.blocked_ips.copy()


# =============================================================================
# MAIN SNIFFER ENGINE
# =============================================================================

class SnifferEngine:
    """
    Main sniffer engine that integrates all components.
    
    This is the primary interface for running the NIDS:
    - Loads trained model and artifacts
    - Captures and processes packets
    - Aggregates flows and extracts features
    - Makes predictions and handles responses
    """
    
    def __init__(
        self,
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts',
        log_dir: str = 'logs',
        firewall_enabled: bool = False,
        firewall_dry_run: bool = True,
        flow_timeout: float = DEFAULT_FLOW_TIMEOUT,
        confidence_threshold: float = 0.7,
        attack_callback: Optional[Callable[[PredictionResult, Flow], None]] = None
    ):
        """
        Initialize the sniffer engine.
        
        Args:
            model_dir: Directory containing the trained model
            artifacts_dir: Directory containing scaler, features, etc.
            log_dir: Directory for log files
            firewall_enabled: Whether to enable IP blocking
            firewall_dry_run: If True, don't actually block IPs
            flow_timeout: Seconds before a flow is considered complete
            confidence_threshold: Minimum confidence for attack classification
            attack_callback: Optional callback function when attack detected
        """
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.log_dir = Path(log_dir)
        self.flow_timeout = flow_timeout
        self.confidence_threshold = confidence_threshold
        self.attack_callback = attack_callback
        
        # Initialize components
        self.logger = SnifferLogger(log_dir=str(self.log_dir))
        self.firewall = FirewallManager(
            enabled=firewall_enabled,
            dry_run=firewall_dry_run,
            logger=self.logger
        )
        self.flow_manager = FlowManager(timeout=flow_timeout)
        self.feature_extractor = FeatureExtractor()
        
        # Load model and artifacts
        self._load_artifacts()
        
        # Session tracking
        self.stats = SessionStats()
        self._running = False
    
    def _load_artifacts(self):
        """Load model and preprocessing artifacts."""
        self.logger.info("Loading model and artifacts...")
        
        # Load model
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            # Try alternative path
            model_path = self.model_dir / 'model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found in {self.model_dir}")
        
        self.model = joblib.load(model_path)
        self.logger.info(f"Loaded model: {type(self.model).__name__}")
        
        # Load scaler
        scaler_path = self.artifacts_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Loaded scaler")
        else:
            self.scaler = None
            self.logger.warning("No scaler found - using raw features")
        
        # Load feature selector
        selector_path = self.artifacts_dir / 'feature_selector.pkl'
        if selector_path.exists():
            self.selector = joblib.load(selector_path)
            self.logger.info("Loaded feature selector")
        else:
            self.selector = None
            self.logger.warning("No feature selector found")
        
        # Load selected features list
        features_path = self.artifacts_dir / 'selected_features.json'
        if not features_path.exists():
            features_path = self.model_dir / 'features_binary.json'
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            self.logger.info(f"Loaded {len(self.selected_features)} selected features")
        else:
            self.selected_features = None
            self.logger.warning("No selected features list found")
        
        # Load scaler columns (for backward compatibility)
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if scaler_cols_path.exists():
            with open(scaler_cols_path, 'r') as f:
                self.scaler_columns = json.load(f)
        else:
            self.scaler_columns = self.selected_features
        
        # CRITICAL: Create index-based selector if no selector.pkl exists
        # but we have both scaler_columns (77 features) and selected_features (30 features)
        self._selected_indices = None
        self._use_index_selection = False
        
        if self.selector is None and self.scaler_columns is not None and self.selected_features is not None:
            if len(self.scaler_columns) != len(self.selected_features):
                # Need to create index-based selection
                self.logger.info("Creating index-based feature selector from selected_features.json")
                scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(self.scaler_columns)}
                selected_indices = []
                for feat in self.selected_features:
                    feat_lower = feat.strip().lower()
                    if feat_lower in scaler_cols_lower:
                        selected_indices.append(scaler_cols_lower[feat_lower])
                    else:
                        self.logger.warning(f"Feature '{feat}' not found in scaler_columns")
                
                if len(selected_indices) == len(self.selected_features):
                    # CRITICAL: Do NOT sort! Keep original order from selected_features.json
                    self._selected_indices = selected_indices
                    self._use_index_selection = True
                    self.logger.info(f"Index-based selection ready: {len(self._selected_indices)} features")
                else:
                    self.logger.error(f"Could not match all selected features!")
        
        # Load label mapping if available
        labels_path = self.artifacts_dir / 'label_encoder.pkl'
        if labels_path.exists():
            self.label_encoder = joblib.load(labels_path)
        else:
            # Default binary labels
            self.label_encoder = None
            self.label_map = {0: 'BENIGN', 1: 'ATTACK'}
        
        self.logger.info("All artifacts loaded successfully")
    
    def _prepare_features(self, flow: Flow) -> Optional[pd.DataFrame]:
        """
        Extract and prepare features from a flow for prediction.
        
        Args:
            flow: Flow object with packet data
            
        Returns:
            DataFrame with features ready for prediction, or None if error
        """
        try:
            # Extract raw features
            feature_dict = self.feature_extractor.extract(flow)
            
            # Determine column order
            if self.scaler_columns:
                columns = self.scaler_columns
            elif self.selected_features:
                columns = self.selected_features
            else:
                columns = get_feature_columns_ordered()
            
            # Create DataFrame with correct column order
            df = pd.DataFrame([feature_dict])
            
            # Ensure all required columns exist
            for col in columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Select only needed columns in order
            df = df[columns]
            
            # Handle NaN/Inf
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def _predict(self, features_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction on prepared features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Scale if scaler exists
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features_df)
            except Exception as e:
                self.logger.warning(f"Scaling error: {e}, using raw features")
                features_scaled = features_df.values
        else:
            features_scaled = features_df.values
        
        # Select features
        if self.selector is not None:
            # Use sklearn selector
            try:
                features_selected = self.selector.transform(features_scaled)
            except Exception:
                features_selected = features_scaled
        elif self._use_index_selection and self._selected_indices is not None:
            # Use index-based selection (when no selector.pkl exists)
            features_selected = features_scaled[:, self._selected_indices]
        else:
            features_selected = features_scaled
        
        # Make prediction
        prediction = self.model.predict(features_selected)[0]
        
        # Get confidence (probability if available)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_selected)[0]
            confidence = max(proba)  # Confidence in the prediction
        else:
            confidence = 1.0
        
        return int(prediction), float(confidence)
    
    def _get_label(self, prediction: int) -> str:
        """Convert numeric prediction to label string."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform([prediction])[0]
        return self.label_map.get(prediction, f'CLASS_{prediction}')
    
    def _handle_prediction(self, result: PredictionResult, flow: Flow):
        """Handle a prediction result (logging, alerting, blocking)."""
        # Log all flows
        self.logger.log_flow(result)
        
        # Update stats
        self.stats.flows_analyzed += 1
        self.stats.predictions_by_label[result.label] += 1
        
        if result.label == 'BENIGN':
            self.stats.benign_flows += 1
        else:
            # Attack detected
            self.stats.attacks_detected += 1
            
            # Log attack
            self.logger.log_attack(result, flow)
            self.logger.warning(
                f"🚨 ATTACK DETECTED: {result.label} | "
                f"Src: {flow.src_ip}:{flow.src_port} → "
                f"Dst: {flow.dst_ip}:{flow.dst_port} | "
                f"Confidence: {result.confidence:.2%}"
            )
            
            # Block IP if enabled and confidence is high enough
            if result.confidence >= self.confidence_threshold:
                self.firewall.block_ip(flow.src_ip, reason=result.label)
            
            # Call custom callback if provided
            if self.attack_callback:
                try:
                    self.attack_callback(result, flow)
                except Exception as e:
                    self.logger.error(f"Attack callback error: {e}")
    
    def analyze_flow(self, flow: Flow) -> Optional[PredictionResult]:
        """
        Analyze a single flow and return prediction.
        
        Args:
            flow: Flow object to analyze
            
        Returns:
            PredictionResult or None if error
        """
        # Prepare features
        features_df = self._prepare_features(flow)
        if features_df is None:
            return None
        
        # Make prediction
        prediction, confidence = self._predict(features_df)
        label = self._get_label(prediction)
        
        # Create result
        result = PredictionResult(
            flow_key=flow.flow_key,
            prediction=prediction,
            label=label,
            confidence=confidence,
            features=features_df.iloc[0].to_dict()
        )
        
        # Handle the prediction
        self._handle_prediction(result, flow)
        
        return result
    
    def _process_packet(self, packet: Packet):
        """Process a single packet from capture."""
        # Extract packet info
        pkt_info = PacketProcessor.process(packet)
        if pkt_info is None:
            return
        
        self.stats.packets_processed += 1
        self.stats.unique_src_ips.add(pkt_info.src_ip)
        self.stats.unique_dst_ips.add(pkt_info.dst_ip)
        
        # Add to flow manager
        flow = self.flow_manager.add_packet_from_info(pkt_info)
        
        # Analyze if flow is complete
        if flow is not None:
            self.analyze_flow(flow)
    
    def analyze_pcap(
        self, 
        pcap_path: str, 
        max_packets: Optional[int] = None,
        verbose: bool = False,
        progress_interval: int = 10000
    ) -> List[PredictionResult]:
        """
        Analyze a PCAP file and return predictions.
        
        Uses streaming to avoid loading entire PCAP into memory.
        
        Args:
            pcap_path: Path to PCAP file
            max_packets: Maximum packets to process (None for all)
            verbose: Show detailed progress
            progress_interval: How often to show progress (packets)
            
        Returns:
            List of PredictionResult objects
        """
        from scapy.utils import PcapReader
        import os
        
        self.logger.info(f"Analyzing PCAP: {pcap_path}")
        
        # Get file size for progress estimation
        file_size_mb = os.path.getsize(pcap_path) / (1024 * 1024)
        self.logger.info(f"File size: {file_size_mb:.1f} MB")
        
        if max_packets:
            self.logger.info(f"Processing up to {max_packets:,} packets")
        else:
            self.logger.info("Processing ALL packets (this may take a while for large files)")
        
        results = []
        
        # Reset stats
        self.stats = SessionStats()
        
        try:
            # Use streaming reader instead of loading all into memory
            packet_count = 0
            last_progress_time = time.time()
            
            with PcapReader(pcap_path) as pcap_reader:
                for packet in pcap_reader:
                    # Check max packets limit
                    if max_packets and packet_count >= max_packets:
                        break
                    
                    # Process packet
                    self._process_packet(packet)
                    packet_count += 1
                    
                    # Progress update
                    if packet_count % progress_interval == 0:
                        elapsed = time.time() - last_progress_time
                        rate = progress_interval / elapsed if elapsed > 0 else 0
                        
                        if verbose:
                            self.logger.info(
                                f"Processed {packet_count:,} packets | "
                                f"Rate: {rate:.0f} pkt/s | "
                                f"Flows: {self.flow_manager.get_flow_count()} active, "
                                f"{self.stats.flows_analyzed} analyzed | "
                                f"Attacks: {self.stats.attacks_detected}"
                            )
                        else:
                            self.logger.info(f"Processed {packet_count:,} packets...")
                        
                        last_progress_time = time.time()
            
            self.logger.info(f"Finished reading {packet_count:,} packets")
            
            # Analyze any remaining flows
            self.logger.info("Analyzing remaining flows...")
            remaining_flows = self.flow_manager.get_all_flows()
            remaining_count = len(remaining_flows)
            
            for i, flow in enumerate(remaining_flows):
                result = self.analyze_flow(flow)
                if result:
                    results.append(result)
                
                # Progress for remaining flows
                if (i + 1) % 1000 == 0:
                    self.logger.info(f"Analyzed {i+1}/{remaining_count} remaining flows")
            
            # Finalize stats
            self.stats.end_time = datetime.now()
            
            # Log summary
            self.logger.info("=" * 50)
            self.logger.info(f"PCAP ANALYSIS COMPLETE")
            self.logger.info(f"  Packets processed: {self.stats.packets_processed:,}")
            self.logger.info(f"  Flows analyzed: {self.stats.flows_analyzed:,}")
            self.logger.info(f"  Attacks detected: {self.stats.attacks_detected:,}")
            self.logger.info(f"  Benign flows: {self.stats.benign_flows:,}")
            self.logger.info(f"  Unique source IPs: {len(self.stats.unique_src_ips):,}")
            self.logger.info(f"  Unique dest IPs: {len(self.stats.unique_dst_ips):,}")
            self.logger.info("=" * 50)
            
            return results
            
        except Exception as e:
            self.logger.error(f"PCAP analysis error: {e}")
            self.stats.errors.append(str(e))
            raise
    
    def start_live(
        self,
        interface: str = 'eth0',
        duration: Optional[int] = None,
        filter_str: str = 'ip',
        promisc: bool = True
    ):
        """
        Start live packet capture and analysis.
        
        Args:
            interface: Network interface to capture on
            duration: Capture duration in seconds (None for indefinite)
            filter_str: BPF filter string
            promisc: Enable promiscuous mode
        """
        self.logger.info(f"Starting live capture on {interface}")
        self.logger.info(f"Filter: {filter_str}, Duration: {duration or 'indefinite'}")
        
        # Reset stats
        self.stats = SessionStats()
        self._running = True
        
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self.logger.info("Shutdown signal received...")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start packet capture
            if duration:
                # Use timeout for duration-based capture
                sniff(
                    iface=interface,
                    filter=filter_str,
                    prn=self._process_packet,
                    store=False,
                    timeout=duration,
                    promisc=promisc
                )
            else:
                # Indefinite capture with stop condition
                sniff(
                    iface=interface,
                    filter=filter_str,
                    prn=self._process_packet,
                    store=False,
                    stop_filter=lambda x: not self._running,
                    promisc=promisc
                )
            
            # Analyze remaining flows
            self.logger.info("Capture ended, analyzing remaining flows...")
            remaining_flows = self.flow_manager.get_all_flows()
            for flow in remaining_flows:
                self.analyze_flow(flow)
            
        except Exception as e:
            self.logger.error(f"Live capture error: {e}")
            self.stats.errors.append(str(e))
        
        finally:
            # Finalize
            self.stats.end_time = datetime.now()
            self._running = False
            
            # Print summary
            self._print_summary()
    
    def stop(self):
        """Stop the sniffer."""
        self._running = False
        self.logger.info("Sniffer stop requested")
    
    def _print_summary(self):
        """Print session summary."""
        stats = self.stats.to_dict()
        
        self.logger.info("=" * 60)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {stats['duration_seconds']:.1f} seconds")
        self.logger.info(f"Packets processed: {stats['packets_processed']}")
        self.logger.info(f"Flows analyzed: {stats['flows_analyzed']}")
        self.logger.info(f"Attacks detected: {stats['attacks_detected']}")
        self.logger.info(f"Benign flows: {stats['benign_flows']}")
        self.logger.info(f"Unique source IPs: {stats['unique_src_ips']}")
        self.logger.info(f"Unique destination IPs: {stats['unique_dst_ips']}")
        
        if stats['predictions_by_label']:
            self.logger.info("Predictions by label:")
            for label, count in stats['predictions_by_label'].items():
                self.logger.info(f"  {label}: {count}")
        
        self.logger.info("=" * 60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current session statistics."""
        return self.stats.to_dict()
    
    def close(self):
        """Clean up resources."""
        self.logger.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_analyze_pcap(
    pcap_path: str,
    model_dir: str = 'models/best_model',
    artifacts_dir: str = 'artifacts'
) -> Dict[str, Any]:
    """
    Quick analysis of a PCAP file.
    
    Args:
        pcap_path: Path to PCAP file
        model_dir: Model directory
        artifacts_dir: Artifacts directory
        
    Returns:
        Dictionary with results summary
    """
    engine = SnifferEngine(
        model_dir=model_dir,
        artifacts_dir=artifacts_dir,
        firewall_enabled=False
    )
    
    try:
        results = engine.analyze_pcap(pcap_path)
        stats = engine.get_stats()
        
        return {
            'success': True,
            'stats': stats,
            'results': [r.to_dict() for r in results],
            'attacks': [r.to_dict() for r in results if r.label != 'BENIGN']
        }
    finally:
        engine.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NIDS Sniffer Engine')
    parser.add_argument('--mode', choices=['live', 'pcap'], required=True,
                       help='Capture mode')
    parser.add_argument('--interface', default='eth0',
                       help='Network interface for live capture')
    parser.add_argument('--pcap', help='PCAP file path for analysis')
    parser.add_argument('--duration', type=int, help='Capture duration (seconds)')
    parser.add_argument('--model-dir', default='models/best_model',
                       help='Model directory')
    parser.add_argument('--artifacts-dir', default='artifacts',
                       help='Artifacts directory')
    parser.add_argument('--firewall', action='store_true',
                       help='Enable firewall blocking')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='Actually execute firewall rules')
    
    args = parser.parse_args()
    
    engine = SnifferEngine(
        model_dir=args.model_dir,
        artifacts_dir=args.artifacts_dir,
        firewall_enabled=args.firewall,
        firewall_dry_run=not args.no_dry_run
    )
    
    try:
        if args.mode == 'pcap':
            if not args.pcap:
                print("Error: --pcap required for pcap mode")
                exit(1)
            engine.analyze_pcap(args.pcap)
        else:
            engine.start_live(
                interface=args.interface,
                duration=args.duration
            )
    finally:
        engine.close()