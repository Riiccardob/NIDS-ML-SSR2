"""
Feature Mapper: nfstream → NF-UQ-NIDS-v2 format.

Converte i flow di nfstream nel formato feature richiesto dal modello.
FIXED per nfstream 6.5.4 API.
"""

import numpy as np
from typing import Dict, Any, List

from config import REQUIRED_FEATURES, N_FEATURES
from utils.logger import get_logger


logger = get_logger()


class FeatureMapper:
    """Mappa feature da nfstream a formato NF-UQ-NIDS."""
    
    def __init__(self):
        self.required_features = REQUIRED_FEATURES
        self.n_features = N_FEATURES
        
        # Mapping diretto nfstream -> NF-UQ-NIDS
        self.feature_mapping = self._build_feature_mapping()
        
        logger.info(f"FeatureMapper initialized with {self.n_features} features")
    
    def _build_feature_mapping(self) -> Dict[str, str]:
        """
        Costruisce mapping tra nomi feature nfstream e NF-UQ-NIDS.
        
        Returns:
            Dict mapping NF-UQ-NIDS feature name -> nfstream attribute
        """
        
        mapping = {
            # Port e Protocol
            "L4_SRC_PORT": "src_port",
            "L4_DST_PORT": "dst_port",
            "PROTOCOL": "protocol",
            "L7_PROTO": "application_name",
            
            # Bytes e Packets
            "IN_BYTES": "src2dst_bytes",
            "IN_PKTS": "src2dst_packets",
            "OUT_BYTES": "dst2src_bytes",
            
            # TCP Flags
            "TCP_FLAGS": "src2dst_syn_packets",  # Proxy per flags
            "SERVER_TCP_FLAGS": "dst2src_syn_packets",
            
            # Duration
            "FLOW_DURATION_MILLISECONDS": "bidirectional_duration_ms",
            "DURATION_IN": "src2dst_duration_ms",
            "DURATION_OUT": "dst2src_duration_ms",
            
            # TTL e Packet Length
            "MIN_TTL": "src2dst_min_ps",  # Proxy
            "LONGEST_FLOW_PKT": "bidirectional_max_ps",
            "SHORTEST_FLOW_PKT": "bidirectional_min_ps",
            "MIN_IP_PKT_LEN": "bidirectional_min_ps",
            
            # Throughput
            "SRC_TO_DST_SECOND_BYTES": "src2dst_bytes",  # Will calculate
            "DST_TO_SRC_SECOND_BYTES": "dst2src_bytes",  # Will calculate
            
            # Retransmission (estimated)
            "RETRANSMITTED_IN_BYTES": "src2dst_bytes",  # Proxy
            "RETRANSMITTED_IN_PKTS": "src2dst_packets",  # Proxy
            "RETRANSMITTED_OUT_BYTES": "dst2src_bytes",  # Proxy
            
            # Average Throughput
            "SRC_TO_DST_AVG_THROUGHPUT": "src2dst_bytes",  # Will calculate
            "DST_TO_SRC_AVG_THROUGHPUT": "dst2src_bytes",  # Will calculate
            
            # Packet Size Distribution
            "NUM_PKTS_UP_TO_128_BYTES": "bidirectional_packets",  # Estimated
            "NUM_PKTS_128_TO_256_BYTES": "bidirectional_packets",
            "NUM_PKTS_256_TO_512_BYTES": "bidirectional_packets",
            "NUM_PKTS_512_TO_1024_BYTES": "bidirectional_packets",
            "NUM_PKTS_1024_TO_1514_BYTES": "bidirectional_packets",
            
            # TCP Window
            "TCP_WIN_MAX_IN": "src2dst_max_ps",  # Proxy
            "TCP_WIN_MAX_OUT": "dst2src_max_ps",  # Proxy
            
            # ICMP (if available)
            "ICMP_TYPE": "protocol",  # Proxy
            
            # DNS (if available via DPI)
            "DNS_QUERY_ID": "application_name",  # Proxy
            "DNS_QUERY_TYPE": "application_name",
            "DNS_TTL_ANSWER": "application_name",
            
            # FTP
            "FTP_COMMAND_RET_CODE": "application_name",
        }
        
        return mapping
    
    def extract_features(self, flow: Any) -> np.ndarray:
        """
        Estrae feature vector da un nfstream flow.
        
        Args:
            flow: Flow object da nfstream (dict-like)
        
        Returns:
            numpy array (shape: (35,)) con feature nell'ordine corretto
        """
        
        features = []
        
        for feature_name in self.required_features:
            value = self._extract_single_feature(flow, feature_name)
            features.append(value)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_single_feature(self, flow: Any, feature_name: str) -> float:
        """
        Estrae una singola feature dal flow.
        
        Args:
            flow: Flow object (dict-like o object con attributes)
            feature_name: Nome feature NF-UQ-NIDS
        
        Returns:
            Valore feature (float)
        """
        
        # Ottieni nome attributo nfstream
        nfstream_attr = self.feature_mapping.get(feature_name)
        
        if nfstream_attr is None:
            logger.warning(f"Unknown feature: {feature_name}, using 0.0")
            return 0.0
        
        # Gestione speciale per alcuni attributi
        value = self._get_flow_attribute(flow, nfstream_attr, feature_name)
        
        # Converti in float, gestisci None/NaN
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        
        return float(value)
    
    def _get_flow_attribute(self, flow: Any, attr: str, feature_name: str) -> Any:
        """
        Ottiene attributo da flow con gestione custom.
        
        Args:
            flow: Flow object
            attr: Nome attributo nfstream
            feature_name: Nome feature originale
        
        Returns:
            Valore attributo
        """
        
        # Flow puo essere dict o object
        if isinstance(flow, dict):
            value = flow.get(attr, None)
        else:
            value = getattr(flow, attr, None)
        
        # Feature calcolate custom
        if feature_name == "L7_PROTO":
            app = self._get_value(flow, "application_name")
            if app:
                return self._map_application_to_code(str(app))
            return 0
        
        if feature_name == "SRC_TO_DST_AVG_THROUGHPUT":
            duration_ms = self._get_value(flow, "src2dst_duration_ms", 0)
            if duration_ms > 0:
                bytes_val = self._get_value(flow, "src2dst_bytes", 0)
                return (bytes_val * 1000.0) / duration_ms  # bytes/sec
            return 0.0
        
        if feature_name == "DST_TO_SRC_AVG_THROUGHPUT":
            duration_ms = self._get_value(flow, "dst2src_duration_ms", 0)
            if duration_ms > 0:
                bytes_val = self._get_value(flow, "dst2src_bytes", 0)
                return (bytes_val * 1000.0) / duration_ms
            return 0.0
        
        # Packet size distribution (estimated equally)
        if "NUM_PKTS" in feature_name:
            total_packets = self._get_value(flow, "bidirectional_packets", 0)
            return total_packets / 5.0  # Divide equally in 5 bins
        
        # DNS/FTP/ICMP potrebbero non essere presenti
        if feature_name.startswith(("DNS_", "FTP_", "ICMP_")):
            return 0
        
        return value if value is not None else 0.0
    
    def _get_value(self, flow: Any, attr: str, default: Any = None) -> Any:
        """Helper per ottenere valore da flow (dict o object)."""
        if isinstance(flow, dict):
            return flow.get(attr, default)
        return getattr(flow, attr, default)
    
    def _map_application_to_code(self, app_name: str) -> int:
        """Mappa nome applicazione L7 a codice numerico."""
        
        app_map = {
            "HTTP": 7,
            "HTTPS": 7,
            "DNS": 53,
            "SSH": 22,
            "FTP": 21,
            "SMTP": 25,
            "IMAP": 143,
            "POP3": 110,
            "TELNET": 23,
            "SMB": 445,
            "RDP": 3389,
            "MYSQL": 3306,
            "POSTGRESQL": 5432,
            "Unknown": 0,
        }
        
        return app_map.get(app_name.upper(), 0)
    
    def extract_flow_metadata(self, flow: Any) -> Dict[str, Any]:
        """
        Estrae metadata utili per logging.
        
        Args:
            flow: Flow object da nfstream
        
        Returns:
            Dict con metadata
        """
        
        return {
            "src_ip": self._get_value(flow, "src_ip", "0.0.0.0"),
            "dst_ip": self._get_value(flow, "dst_ip", "0.0.0.0"),
            "src_port": self._get_value(flow, "src_port", 0),
            "dst_port": self._get_value(flow, "dst_port", 0),
            "protocol": self._get_value(flow, "protocol", 0),
            "l7_proto": self._get_value(flow, "application_name", "Unknown"),
            "duration_ms": self._get_value(flow, "bidirectional_duration_ms", 0),
            "bytes_in": self._get_value(flow, "src2dst_bytes", 0),
            "bytes_out": self._get_value(flow, "dst2src_bytes", 0),
            "packets_in": self._get_value(flow, "src2dst_packets", 0),
            "packets_out": self._get_value(flow, "dst2src_packets", 0),
        }
    
    def validate_feature_vector(self, features: np.ndarray) -> bool:
        """Valida feature vector."""
        
        if features.shape != (self.n_features,):
            logger.error(f"Invalid feature shape: {features.shape}, expected ({self.n_features},)")
            return False
        
        if np.any(np.isnan(features)):
            logger.warning("NaN values detected in feature vector")
            return False
        
        if np.any(np.isinf(features)):
            logger.warning("Inf values detected in feature vector")
            return False
        
        return True
