"""
Feature Mapper: nfstream -> NF-UQ-NIDS-v2 (v3, 18 feature).

Versione aggiornata dopo il retraining con FEATURES_TO_DROP esteso (v2+v3).
Le feature non riproducibili fedelmente da nfstream sono state rimosse
dal training set e quindi non compaiono piu' in REQUIRED_FEATURES:

    Rimosse v2 (24 -> 19 feature):
        MIN_TTL                   -- nfstream non espone TTL per-flow
        NUM_PKTS_UP_TO_128_BYTES  -- nfstream non ha bucket per packet size
        NUM_PKTS_128_TO_256_BYTES
        NUM_PKTS_256_TO_512_BYTES
        NUM_PKTS_512_TO_1024_BYTES
        NUM_PKTS_1024_TO_1514_BYTES
    
    Rimosse v3 (19 -> 18 feature):
        RETRANSMITTED_OUT_PKTS    -- nfstream non espone retrans counters

    Rimaste (18):
        L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO,
        IN_BYTES, IN_PKTS, OUT_BYTES, CLIENT_TCP_FLAGS,
        FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT,
        LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN,
        SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT,
        ICMP_IPV4_TYPE, MAX_TTL

Correzioni unita' di misura v3:
    - DURATION_IN:  nfstream (ms) -> NF-UQ-NIDS-v2 (µs)  [x1000]
    - DURATION_OUT: nfstream (ms) -> NF-UQ-NIDS-v2 (µs)  [x1000]
"""

from typing import Any, Dict, List

import numpy as np

from config import N_FEATURES, REQUIRED_FEATURES
from utils.logger import get_logger

logger = get_logger()

_L7_PROTO_MAP: Dict[str, int] = {
    "HTTP": 7, "HTTPS": 443, "DNS": 53, "FTP": 21,
    "FTP_DATA": 20, "SSH": 22, "SMTP": 25, "IMAP": 143,
    "POP3": 110, "TELNET": 23, "NTP": 123, "SNMP": 161,
    "DHCP": 67, "MDNS": 5353, "QUIC": 443, "TLS": 443,
    "SSL": 443, "ICMP": 1, "ICMPv6": 58, "IGMP": 2,
    "ARP": 0, "NBNS": 137, "SMB": 445, "RDP": 3389,
    "VNC": 5900, "MYSQL": 3306, "POSTGRESQL": 5432,
    "REDIS": 6379, "MONGODB": 27017, "Unknown": 0,
}

# Feature che nella v1 erano nel modello ma sono state rimosse in v2/v3.
# Se compaiono in REQUIRED_FEATURES emette un warning esplicito.
_REMOVED_V2_V3 = frozenset({
    # Rimosse v2 (non riproducibili con precisione)
    "MIN_TTL",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    
    # Rimosse v3 (non disponibili in nfstream)
    "RETRANSMITTED_OUT_PKTS",
})


class FeatureMapper:
    """
    Mappa i flow nfstream nel formato NF-UQ-NIDS-v2 usato in training.

    Il mapper e' configurato dinamicamente da REQUIRED_FEATURES in
    live_sniffer/config.py, che a sua volta carica features.json.
    """

    def __init__(self) -> None:
        self.required_features: List[str] = REQUIRED_FEATURES
        self.n_features: int = N_FEATURES
        self._log_inventory()

    def _log_inventory(self) -> None:
        logger.info(f"FeatureMapper inizializzato con {self.n_features} feature")
        logger.info(f"Feature source: artifacts/features.json")

        stale = _REMOVED_V2_V3 & set(self.required_features)
        if stale:
            logger.warning(
                f"ATTENZIONE: le seguenti feature sono state rimosse in v2/v3 "
                f"ma sono ancora in REQUIRED_FEATURES. E' necessario "
                f"rigenerare gli artifacts e retrainare: {stale}"
            )

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def extract_features(self, flow: Any) -> np.ndarray:
        """
        Estrae il vettore feature dal flow nfstream.

        Args:
            flow: Oggetto flow nfstream oppure dict con gli stessi campi.

        Returns:
            np.ndarray di forma (N_FEATURES,), dtype float32.
        """
        return np.array(
            [self._extract_single(flow, name) for name in self.required_features],
            dtype=np.float32,
        )

    def validate_feature_vector(self, features: np.ndarray) -> bool:
        """True se il vettore ha la forma attesa e non contiene NaN/Inf."""
        if features.shape != (self.n_features,):
            logger.warning(
                f"Feature shape errata: {features.shape}, atteso ({self.n_features},)"
            )
            return False
        if not np.all(np.isfinite(features)):
            logger.warning("Feature vector contiene NaN o Inf")
            return False
        return True

    def extract_flow_metadata(self, flow: Any) -> Dict[str, Any]:
        """Metadati del flow usati per il logging (non per la predizione)."""
        return {
            "src_ip":      self._get(flow, "src_ip", "0.0.0.0"),
            "dst_ip":      self._get(flow, "dst_ip", "0.0.0.0"),
            "src_port":    int(self._get(flow, "src_port", 0)),
            "dst_port":    int(self._get(flow, "dst_port", 0)),
            "protocol":    int(self._get(flow, "protocol", 0)),
            "l7_proto":    str(self._get(flow, "application_name", "Unknown")),
            "duration_ms": int(self._get(flow, "bidirectional_duration_ms", 0)),
            "bytes_in":    int(self._get(flow, "src2dst_bytes", 0)),
            "bytes_out":   int(self._get(flow, "dst2src_bytes", 0)),
            "packets_in":  int(self._get(flow, "src2dst_packets", 0)),
            "packets_out": int(self._get(flow, "dst2src_packets", 0)),
        }

    # ------------------------------------------------------------------
    # Estrazione singola feature
    # ------------------------------------------------------------------

    def _extract_single(self, flow: Any, name: str) -> float:
        """Mappa una singola feature NF-UQ-NIDS-v2 dall'oggetto nfstream."""

        if name == "L4_SRC_PORT":
            return float(self._get(flow, "src_port", 0))
        if name == "L4_DST_PORT":
            return float(self._get(flow, "dst_port", 0))
        if name == "PROTOCOL":
            return float(self._get(flow, "protocol", 0))
        if name == "L7_PROTO":
            proto_name = str(self._get(flow, "application_name", "Unknown"))
            return float(_L7_PROTO_MAP.get(proto_name, 0))

        if name == "IN_BYTES":
            return float(self._get(flow, "src2dst_bytes", 0))
        if name == "IN_PKTS":
            return float(self._get(flow, "src2dst_packets", 0))
        if name == "OUT_BYTES":
            return float(self._get(flow, "dst2src_bytes", 0))
        if name == "OUT_PKTS":
            return float(self._get(flow, "dst2src_packets", 0))

        if name == "CLIENT_TCP_FLAGS":
            return float(self._get(flow, "client_tcp_flags", 0))

        if name == "FLOW_DURATION_MILLISECONDS":
            return float(self._get(flow, "bidirectional_duration_ms", 0))
        if name == "DURATION_IN":
            # nfstream: src2dst_duration_ms (millisecondi)
            # Dataset NF-UQ-NIDS-v2: DURATION_IN (microsecondi)
            # Conversione: ms * 1000 = µs
            ms = float(self._get(flow, "src2dst_duration_ms", 0))
            return ms * 1000.0
        if name == "DURATION_OUT":
            # nfstream: dst2src_duration_ms (millisecondi)
            # Dataset NF-UQ-NIDS-v2: DURATION_OUT (microsecondi)
            # Conversione: ms * 1000 = µs
            ms = float(self._get(flow, "dst2src_duration_ms", 0))
            return ms * 1000.0

        if name == "LONGEST_FLOW_PKT":
            return float(self._get(flow, "bidirectional_max_ps", 0))
        if name == "SHORTEST_FLOW_PKT":
            return float(self._get(flow, "bidirectional_min_ps", 0))
        if name == "MIN_IP_PKT_LEN":
            return float(self._get(flow, "bidirectional_min_ps", 0))

        # RETRANSMITTED_OUT_PKTS: rimossa in v3 (nfstream non espone retrans counters)
        # Se compare in REQUIRED_FEATURES, il warning viene emesso in _log_inventory()

        if name == "SRC_TO_DST_AVG_THROUGHPUT":
            return self._throughput(flow, "src2dst_bytes", "src2dst_duration_ms")
        if name == "DST_TO_SRC_AVG_THROUGHPUT":
            return self._throughput(flow, "dst2src_bytes", "dst2src_duration_ms")

        if name == "ICMP_IPV4_TYPE":
            if int(self._get(flow, "protocol", 0)) == 1:
                return float(self._get(flow, "icmp_type", 0))
            return 0.0

        # Feature rimosse in v2/v3: non devono mai arrivare qui se features.json
        # e' stato rigenerato correttamente. Log debug per debug.
        if name in _REMOVED_V2_V3:
            logger.debug(
                f"Feature '{name}' rimossa in v2/v3 ancora richiesta. "
                "Rigenerare gli artifacts."
            )
            return 0.0

        logger.debug(f"Feature '{name}' non gestita, valore 0.0")
        return 0.0

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get(flow: Any, attr: str, default: Any = 0) -> Any:
        if isinstance(flow, dict):
            return flow.get(attr, default)
        v = getattr(flow, attr, None)
        return v if v is not None else default

    def _throughput(self, flow: Any, bytes_attr: str, dur_attr: str) -> float:
        """Throughput in bytes/s, cap a 10 Gbps (1.25e9 B/s)."""
        b = float(self._get(flow, bytes_attr, 0))
        d = float(self._get(flow, dur_attr, 0))
        if d > 0:
            return min(b * 1000.0 / d, 1.25e9)
        return 0.0