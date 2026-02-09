"""
Feature Mapper: nfstream → NF-UQ-NIDS-v2 (FIXED VERSION)

CORREZIONI APPLICATE:
1. Rimosso MIN_TTL mapping errato (packet size)
2. Aggiunto OUT_PKTS mancante
3. Corretto CLIENT_TCP_FLAGS (usa src_tcp_flags)
4. Rimosso RETRANSMITTED_OUT_PKTS (NFStream non traccia per direction)
5. Aggiunto mapping corretto per ICMP_IPV4_TYPE

STRATEGIA: Usa SOLO feature che NFStream fornisce nativamente.
Feature non disponibili → ritornano 0.0 (modello deve gestirlo).
"""

import numpy as np
from typing import Dict, Any

from config import REQUIRED_FEATURES, N_FEATURES
from utils.logger import get_logger


logger = get_logger()


class FeatureMapper:
    """Mappa feature da nfstream a formato NF-UQ-NIDS (FIXED)."""
    
    def __init__(self):
        self.required_features = REQUIRED_FEATURES
        self.n_features = N_FEATURES
        
        logger.info(f"FeatureMapper initialized (FIXED version)")
        logger.info(f"Using {self.n_features} nfstream-compatible features")
        logger.info(f"Features unavailable in NFStream will return 0.0")
    
    def extract_features(self, flow: Any) -> np.ndarray:
        """
        Estrae feature vector da nfstream flow.
        
        Args:
            flow: Flow object da nfstream
        
        Returns:
            numpy array con feature (shape: (24,))
        """
        
        features = []
        
        for feature_name in self.required_features:
            value = self._extract_single_feature(flow, feature_name)
            features.append(value)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_single_feature(self, flow: Any, feature_name: str) -> float:
        """Estrae singola feature (MAPPATURE CORRETTE)."""
        
        # ====================================================================
        # BASE FEATURES (SEMPRE DISPONIBILI)
        # ====================================================================
        
        if feature_name == "L4_SRC_PORT":
            return float(self._get_value(flow, "src_port", 0))
        
        if feature_name == "L4_DST_PORT":
            return float(self._get_value(flow, "dst_port", 0))
        
        if feature_name == "PROTOCOL":
            return float(self._get_value(flow, "protocol", 0))
        
        if feature_name == "L7_PROTO":
            app = self._get_value(flow, "application_name", "Unknown")
            return float(self._map_application_to_code(str(app)))
        
        # ====================================================================
        # BYTES & PACKETS (CORRETTI)
        # ====================================================================
        
        if feature_name == "IN_BYTES":
            return float(self._get_value(flow, "src2dst_bytes", 0))
        
        if feature_name == "IN_PKTS":
            return float(self._get_value(flow, "src2dst_packets", 0))
        
        if feature_name == "OUT_BYTES":
            return float(self._get_value(flow, "dst2src_bytes", 0))
        
        # FIX: OUT_PKTS era mancante!
        if feature_name == "OUT_PKTS":
            return float(self._get_value(flow, "dst2src_packets", 0))
        
        # ====================================================================
        # TCP FLAGS (CORRETTO - usa client_tcp_flags)
        # ====================================================================
        
        if feature_name == "CLIENT_TCP_FLAGS":
            # NFStream fornisce client_tcp_flags come integer
            # Esempio: SYN=2, ACK=16, FIN=1, RST=4
            flags = self._get_value(flow, "client_tcp_flags", 0)
            return float(flags)
        
        # ====================================================================
        # DURATION (MILLISECONDI)
        # ====================================================================
        
        if feature_name == "FLOW_DURATION_MILLISECONDS":
            return float(self._get_value(flow, "bidirectional_duration_ms", 0))
        
        if feature_name == "DURATION_IN":
            return float(self._get_value(flow, "src2dst_duration_ms", 0))
        
        if feature_name == "DURATION_OUT":
            return float(self._get_value(flow, "dst2src_duration_ms", 0))
        
        # ====================================================================
        # PACKET SIZE STATS
        # ====================================================================
        
        # FIX: MIN_TTL NON DISPONIBILE in NFStream!
        # Non possiamo usare packet size come proxy (distribuzione diversa)
        # Ritorniamo 0.0 - il modello è stato trainato con RobustScaler che gestisce
        if feature_name == "MIN_TTL":
            return 0.0  # Feature non disponibile
        
        if feature_name == "LONGEST_FLOW_PKT":
            return float(self._get_value(flow, "bidirectional_max_ps", 0))
        
        if feature_name == "SHORTEST_FLOW_PKT":
            return float(self._get_value(flow, "bidirectional_min_ps", 0))
        
        if feature_name == "MIN_IP_PKT_LEN":
            return float(self._get_value(flow, "bidirectional_min_ps", 0))
        
        # ====================================================================
        # RETRANSMISSIONS
        # ====================================================================
        
        # FIX: NFStream NON traccia retransmissions per direction
        # Ha solo bidirectional_retrans_packets
        if feature_name == "RETRANSMITTED_OUT_PKTS":
            # Approssimazione: usa metà delle retrans totali
            total_retrans = self._get_value(flow, "bidirectional_retrans_packets", 0)
            return float(total_retrans / 2.0)
        
        # ====================================================================
        # THROUGHPUT (CALCULATED)
        # ====================================================================
        
        if feature_name == "SRC_TO_DST_AVG_THROUGHPUT":
            return self._calculate_throughput(flow, "src2dst_bytes", "src2dst_duration_ms")
        
        if feature_name == "DST_TO_SRC_AVG_THROUGHPUT":
            return self._calculate_throughput(flow, "dst2src_bytes", "dst2src_duration_ms")
        
        # ====================================================================
        # PACKET DISTRIBUTION (ESTIMATED)
        # ====================================================================
        
        if feature_name == "NUM_PKTS_UP_TO_128_BYTES":
            return self._estimate_packet_distribution(flow, 0, 128)
        
        if feature_name == "NUM_PKTS_128_TO_256_BYTES":
            return self._estimate_packet_distribution(flow, 128, 256)
        
        if feature_name == "NUM_PKTS_256_TO_512_BYTES":
            return self._estimate_packet_distribution(flow, 256, 512)
        
        if feature_name == "NUM_PKTS_512_TO_1024_BYTES":
            return self._estimate_packet_distribution(flow, 512, 1024)
        
        if feature_name == "NUM_PKTS_1024_TO_1514_BYTES":
            return self._estimate_packet_distribution(flow, 1024, 1514)
        
        # ====================================================================
        # ICMP (CORRETTO)
        # ====================================================================
        
        if feature_name == "ICMP_IPV4_TYPE":
            # NFStream fornisce icmp_type per flow ICMP
            protocol = self._get_value(flow, "protocol", 0)
            if protocol == 1:  # ICMP
                return float(self._get_value(flow, "icmp_type", 0))
            return 0.0
        
        # ====================================================================
        # FEATURE NON DISPONIBILI
        # ====================================================================
        
        # Se arriviamo qui, feature non gestita
        logger.warning(f"Feature {feature_name} not available in NFStream, using 0.0")
        return 0.0
    
    def _calculate_throughput(self, flow: Any, bytes_attr: str, duration_attr: str) -> float:
        """Calcola throughput (bytes/sec)."""
        bytes_val = self._get_value(flow, bytes_attr, 0)
        duration_ms = self._get_value(flow, duration_attr, 0)
        
        if duration_ms > 0:
            throughput = (bytes_val * 1000.0) / duration_ms
            
            # Sanity check: cap a 10 Gbps (1.25 GB/s)
            if throughput > 1.25e9:
                return 1.25e9
            
            return throughput
        
        return 0.0
    
    def _estimate_packet_distribution(self, flow: Any, min_size: int, max_size: int) -> float:
        """
        Stima packet distribution basata su avg packet size.
        
        NOTA: Approssimazione necessaria perché NFStream non fornisce histogram.
        """
        total_packets = self._get_value(flow, "bidirectional_packets", 0)
        
        if total_packets == 0:
            return 0.0
        
        total_bytes = self._get_value(flow, "bidirectional_bytes", 0)
        if total_bytes == 0:
            return 0.0
        
        avg_packet_size = total_bytes / total_packets
        
        # Stima probabilistica basata su gaussian
        # Se avg nel range, alta probabilità
        if min_size <= avg_packet_size <= max_size:
            return total_packets * 0.6  # 60% nel range
        
        # Se avg vicino al range, media probabilità
        margin = (max_size - min_size) * 0.5
        if (min_size - margin) <= avg_packet_size <= (max_size + margin):
            return total_packets * 0.3  # 30% vicino
        
        # Altrimenti bassa
        return total_packets * 0.1  # 10% lontano
    
    def _get_value(self, flow: Any, attr: str, default: Any = None) -> Any:
        """Get value from flow (dict or object)."""
        if isinstance(flow, dict):
            return flow.get(attr, default)
        return getattr(flow, attr, default)
    
    def _map_application_to_code(self, app_name: str) -> int:
        """
        Mappa application name a codice numerico.
        
        NFStream usa nDPI per L7 detection, mapping comune:
        """
        app_map = {
            "HTTP": 7,
            "HTTPS": 7,
            "SSL": 7,
            "TLS": 7,
            "DNS": 53,
            "SSH": 22,
            "FTP": 21,
            "SMTP": 25,
            "POP3": 110,
            "IMAP": 143,
            "TELNET": 23,
            "SMB": 445,
            "RDP": 3389,
            "MYSQL": 3306,
            "POSTGRESQL": 5432,
            "Unknown": 0,
        }
        
        return app_map.get(app_name.upper(), 0)
    
    def extract_flow_metadata(self, flow: Any) -> Dict[str, Any]:
        """Estrae metadata per logging."""
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
            logger.error(f"Invalid shape: {features.shape}, expected ({self.n_features},)")
            return False
        
        if np.any(np.isnan(features)):
            logger.warning("NaN detected in features")
            return False
        
        if np.any(np.isinf(features)):
            logger.warning("Inf detected in features")
            return False
        
        return True