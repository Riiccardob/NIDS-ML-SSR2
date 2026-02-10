"""
Feature Mapper: nfstream flow -> vettore NF-UQ-NIDS-v2.

Ogni feature e' mappata sull'attributo nfstream piu' fedele disponibile.
Le feature per cui nfstream non offre una corrispondenza diretta sono
documentate esplicitamente con la strategia adottata e il suo impatto.

Attributi nfstream rilevanti (nfstream >= 6.3, statistical_analysis=True):

  Identificatori
    src_ip, dst_ip, src_port, dst_port, protocol, application_name

  Bytes/Pacchetti direzionali
    src2dst_bytes, src2dst_packets
    dst2src_bytes, dst2src_packets
    bidirectional_bytes, bidirectional_packets

  Durata
    bidirectional_duration_ms
    src2dst_duration_ms, dst2src_duration_ms

  Statistiche dimensione pacchetti (statistical_analysis=True)
    bidirectional_min_ps, bidirectional_max_ps
    bidirectional_mean_ps, bidirectional_stddev_ps
    src2dst_min_ps, src2dst_max_ps, src2dst_mean_ps, src2dst_stddev_ps
    dst2src_min_ps, dst2src_max_ps, dst2src_mean_ps, dst2src_stddev_ps

  TCP flags
    bidirectional_syn_packets, bidirectional_fin_packets
    bidirectional_rst_packets, bidirectional_psh_packets
    bidirectional_ack_packets, bidirectional_urg_packets
    src2dst_syn_packets, src2dst_fin_packets ...  (stessa struttura)
    dst2src_syn_packets ...
    client_tcp_flags, server_tcp_flags (bitfield)

  Ritrasmissioni
    bidirectional_retrans_packets
    src2dst_retrans_packets, dst2src_retrans_packets

  ICMP
    icmp_type (disponibile solo se protocol==1)

Feature NON disponibili in nfstream:
  MIN_TTL             - nfstream non espone il TTL minimo osservato nel flow.
                        Viene impostato a 0.0. Impatto: il modello vede un valore
                        fuori dalla distribuzione di training (TTL reali: 32-255).
                        Questo introduce un bias sistematico su questa feature.
"""

import numpy as np
from typing import Any, Dict

from config import REQUIRED_FEATURES, N_FEATURES
from utils.logger import get_logger


logger = get_logger()

# Costante per il cap del throughput (10 Gbps in bytes/sec)
_THROUGHPUT_CAP_BYTES_PER_SEC: float = 1.25e9

# Mappa application_name -> codice nDPI approssimativo.
# NF-UQ-NIDS-v2 usa codici nDPI generati da CICFlowMeter/nDPI.
# Questa mappa copre i protocolli piu' frequenti nel dataset.
# Il traffico non riconosciuto ottiene 0 (Unknown).
_APP_TO_NDPI_CODE: Dict[str, int] = {
    "HTTP": 7,
    "HTTP_PROXY": 7,
    "HTTPS": 91,
    "TLS": 91,
    "SSL": 91,
    "QUIC": 188,
    "DNS": 5,
    "SSH": 22,
    "FTP": 1,
    "FTP_DATA": 2,
    "SMTP": 3,
    "SMTPS": 3,
    "POP3": 9,
    "POP3S": 9,
    "IMAP": 10,
    "IMAPS": 10,
    "TELNET": 23,
    "SMB": 45,
    "RDP": 122,
    "MYSQL": 97,
    "POSTGRESQL": 5432,
    "NTP": 35,
    "SNMP": 36,
    "ICMP": 81,
    "ICMPV6": 82,
    "Unknown": 0,
}


class FeatureMapper:
    """
    Mappa un flow nfstream nel vettore di feature atteso dal modello.

    La lista di feature attiva viene letta da config.REQUIRED_FEATURES,
    che a sua volta e' caricata dinamicamente da artifacts/features.json.
    In questo modo il mapper si adatta automaticamente al modello trainato.
    """

    def __init__(self) -> None:
        self.required_features = REQUIRED_FEATURES
        self.n_features = N_FEATURES

        logger.info(f"FeatureMapper inizializzato con {self.n_features} feature")
        logger.info(f"Feature source: artifacts/features.json")

        # Log feature non disponibili nativamente
        _unavailable = {"MIN_TTL"}
        _approximate = {"RETRANSMITTED_OUT_PKTS", "NUM_PKTS_UP_TO_128_BYTES",
                        "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
                        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"}
        active_unavailable = _unavailable & set(self.required_features)
        active_approximate = _approximate & set(self.required_features)
        if active_unavailable:
            logger.warning(
                f"Feature NON disponibili in nfstream (saranno 0.0): {active_unavailable}"
            )
        if active_approximate:
            logger.warning(
                f"Feature approssimate (distribuzione pacchetti stimata): {active_approximate}"
            )

    def extract_features(self, flow: Any) -> np.ndarray:
        """
        Estrae il vettore di feature da un flow nfstream.

        Args:
            flow: Oggetto flow nfstream oppure dict compatibile (per test).

        Returns:
            numpy array shape (N_FEATURES,) dtype float32.
        """
        features = [
            self._extract_single(flow, name)
            for name in self.required_features
        ]
        return np.array(features, dtype=np.float32)

    def _extract_single(self, flow: Any, feature_name: str) -> float:
        """Estrae e restituisce il valore float di una singola feature."""

        g = self._get

        # --------------------------------------------------------------------
        # Identificatori di porta e protocollo
        # --------------------------------------------------------------------
        if feature_name == "L4_SRC_PORT":
            return float(g(flow, "src_port", 0))

        if feature_name == "L4_DST_PORT":
            return float(g(flow, "dst_port", 0))

        if feature_name == "PROTOCOL":
            return float(g(flow, "protocol", 0))

        if feature_name == "L7_PROTO":
            app = str(g(flow, "application_name", "Unknown"))
            return float(self._app_to_code(app))

        # --------------------------------------------------------------------
        # Bytes e pacchetti
        # --------------------------------------------------------------------
        if feature_name == "IN_BYTES":
            return float(g(flow, "src2dst_bytes", 0))

        if feature_name == "IN_PKTS":
            return float(g(flow, "src2dst_packets", 0))

        if feature_name == "OUT_BYTES":
            return float(g(flow, "dst2src_bytes", 0))

        if feature_name == "OUT_PKTS":
            return float(g(flow, "dst2src_packets", 0))

        # --------------------------------------------------------------------
        # TCP flags
        # --------------------------------------------------------------------
        if feature_name == "CLIENT_TCP_FLAGS":
            # nfstream espone client_tcp_flags come bitfield cumulativo
            return float(g(flow, "client_tcp_flags", 0))

        if feature_name == "SERVER_TCP_FLAGS":
            return float(g(flow, "server_tcp_flags", 0))

        # --------------------------------------------------------------------
        # Durata
        # --------------------------------------------------------------------
        if feature_name == "FLOW_DURATION_MILLISECONDS":
            return float(g(flow, "bidirectional_duration_ms", 0))

        if feature_name == "DURATION_IN":
            return float(g(flow, "src2dst_duration_ms", 0))

        if feature_name == "DURATION_OUT":
            return float(g(flow, "dst2src_duration_ms", 0))

        # --------------------------------------------------------------------
        # TTL - non disponibile in nfstream
        # --------------------------------------------------------------------
        if feature_name == "MIN_TTL":
            # nfstream non espone il TTL minimo osservato per flow.
            # Valore fisso 0.0: introduce un bias rispetto alla distribuzione
            # di training (valori tipici: 32, 64, 128, 255).
            return 0.0

        if feature_name == "MAX_TTL":
            return 0.0

        # --------------------------------------------------------------------
        # Dimensione pacchetti - statistiche nfstream
        # --------------------------------------------------------------------
        if feature_name == "LONGEST_FLOW_PKT":
            return float(g(flow, "bidirectional_max_ps", 0))

        if feature_name == "SHORTEST_FLOW_PKT":
            return float(g(flow, "bidirectional_min_ps", 0))

        if feature_name == "MIN_IP_PKT_LEN":
            # src2dst_min_ps e' la dimensione minima dei pacchetti nel senso
            # client->server, proxy piu' fedele a MIN_IP_PKT_LEN del dataset.
            return float(g(flow, "src2dst_min_ps", 0))

        if feature_name == "MAX_IP_PKT_LEN":
            return float(g(flow, "src2dst_max_ps", 0))

        # --------------------------------------------------------------------
        # Ritrasmissioni
        # --------------------------------------------------------------------
        if feature_name == "RETRANSMITTED_IN_PKTS":
            # nfstream >= 6.3 espone i contatori direzionali.
            # src2dst_retrans_packets corrisponde al senso IN (client->server).
            val = g(flow, "src2dst_retrans_packets", None)
            if val is not None:
                return float(val)
            # Fallback per versioni piu' vecchie: meta delle ritrasmissioni bidirezionali.
            # E' un'approssimazione: usare nfstream aggiornato per il valore preciso.
            return float(g(flow, "bidirectional_retrans_packets", 0)) / 2.0

        if feature_name == "RETRANSMITTED_OUT_PKTS":
            val = g(flow, "dst2src_retrans_packets", None)
            if val is not None:
                return float(val)
            return float(g(flow, "bidirectional_retrans_packets", 0)) / 2.0

        # --------------------------------------------------------------------
        # Throughput (bytes/sec)
        # --------------------------------------------------------------------
        if feature_name == "SRC_TO_DST_AVG_THROUGHPUT":
            return self._throughput(
                flow, "src2dst_bytes", "src2dst_duration_ms"
            )

        if feature_name == "DST_TO_SRC_AVG_THROUGHPUT":
            return self._throughput(
                flow, "dst2src_bytes", "dst2src_duration_ms"
            )

        # --------------------------------------------------------------------
        # Distribuzione dimensioni pacchetti
        #
        # nfstream con statistical_analysis=True espone media e deviazione
        # standard dei packet size, ma NON i contatori per bucket dimensionale.
        # La strategia corretta e' stimare i contatori usando la distribuzione
        # empirica approssimata con i parametri statistici disponibili.
        #
        # Approccio adottato:
        #   1. Calcola mean_ps e stddev_ps bidirezionali.
        #   2. Approssima la distribuzione come gaussiana troncata (0, MTU).
        #   3. Stima la frazione di pacchetti in ogni bucket come integrale
        #      della gaussiana sull'intervallo [min_size, max_size).
        #   4. Moltiplica per bidirectional_packets.
        #
        # Limitazione: le distribuzioni reali di packet size non sono
        # gaussiane (spesso bimodali: ACK piccoli + dati grandi). Questo
        # introduce errore sistematico. L'alternativa corretta richiederebbe
        # l'accesso ai singoli pacchetti, non ai flow aggregati.
        # --------------------------------------------------------------------
        if feature_name == "NUM_PKTS_UP_TO_128_BYTES":
            return self._pkt_bucket(flow, 0, 128)

        if feature_name == "NUM_PKTS_128_TO_256_BYTES":
            return self._pkt_bucket(flow, 128, 256)

        if feature_name == "NUM_PKTS_256_TO_512_BYTES":
            return self._pkt_bucket(flow, 256, 512)

        if feature_name == "NUM_PKTS_512_TO_1024_BYTES":
            return self._pkt_bucket(flow, 512, 1024)

        if feature_name == "NUM_PKTS_1024_TO_1514_BYTES":
            return self._pkt_bucket(flow, 1024, 1514)

        # --------------------------------------------------------------------
        # ICMP
        # --------------------------------------------------------------------
        if feature_name == "ICMP_IPV4_TYPE":
            if int(g(flow, "protocol", 0)) == 1:
                return float(g(flow, "icmp_type", 0))
            return 0.0

        # --------------------------------------------------------------------
        # Fallback: feature presente in features.json ma non mappata
        # --------------------------------------------------------------------
        logger.warning(
            f"Feature '{feature_name}' non gestita nel mapper, valore 0.0. "
            "Verificare allineamento features.json con feature_mapper.py."
        )
        return 0.0

    # ------------------------------------------------------------------------
    # Metodi di supporto
    # ------------------------------------------------------------------------

    def _throughput(
        self,
        flow: Any,
        bytes_attr: str,
        duration_ms_attr: str,
    ) -> float:
        """
        Calcola il throughput medio in bytes/sec.

        Args:
            flow: Flow nfstream o dict.
            bytes_attr: Nome attributo bytes.
            duration_ms_attr: Nome attributo durata in millisecondi.

        Returns:
            Throughput cappato a _THROUGHPUT_CAP_BYTES_PER_SEC.
        """
        bytes_val = float(self._get(flow, bytes_attr, 0))
        duration_ms = float(self._get(flow, duration_ms_attr, 0))

        if duration_ms <= 0:
            return 0.0

        throughput = (bytes_val * 1000.0) / duration_ms
        return min(throughput, _THROUGHPUT_CAP_BYTES_PER_SEC)

    def _pkt_bucket(self, flow: Any, low: int, high: int) -> float:
        """
        Stima il numero di pacchetti con dimensione in [low, high) bytes.

        Usa la distribuzione gaussiana approssimata con mean_ps e stddev_ps
        bidirezionali. Vedi nota nella sezione NUM_PKTS_* sopra.

        Args:
            flow: Flow nfstream o dict.
            low:  Limite inferiore del bucket (incluso).
            high: Limite superiore del bucket (escluso).

        Returns:
            Stima del numero di pacchetti nel bucket (float).
        """
        total_pkts = float(self._get(flow, "bidirectional_packets", 0))
        if total_pkts <= 0:
            return 0.0

        mean_ps = float(self._get(flow, "bidirectional_mean_ps", 0))
        std_ps = float(self._get(flow, "bidirectional_stddev_ps", 0))

        if mean_ps <= 0:
            # Nessuna informazione statistica: distribuzione uniforme sui bucket.
            # 5 bucket standard: 0-128, 128-256, 256-512, 512-1024, 1024-1514
            return total_pkts / 5.0

        if std_ps <= 0:
            # Tutti i pacchetti hanno la stessa dimensione.
            return total_pkts if (low <= mean_ps < high) else 0.0

        # CDF gaussiana approssimata con formula di Abramowitz & Stegun.
        # Piu' precisa della divisione per 2 usata in precedenza.
        def _gaussian_cdf(x: float) -> float:
            import math
            if std_ps <= 0:
                return 1.0 if x >= mean_ps else 0.0
            z = (x - mean_ps) / (std_ps * (2.0 ** 0.5))
            return 0.5 * (1.0 + math.erf(z))

        fraction = _gaussian_cdf(high) - _gaussian_cdf(low)
        fraction = max(0.0, min(1.0, fraction))
        return total_pkts * fraction

    @staticmethod
    def _get(flow: Any, attr: str, default: Any = 0) -> Any:
        """
        Legge un attributo da un flow nfstream o da un dict.

        Args:
            flow:    Oggetto nfstream o dict.
            attr:    Nome dell'attributo.
            default: Valore di default se l'attributo non esiste o e' None.

        Returns:
            Valore dell'attributo o default.
        """
        if isinstance(flow, dict):
            val = flow.get(attr)
        else:
            val = getattr(flow, attr, None)

        return default if val is None else val

    @staticmethod
    def _app_to_code(app_name: str) -> int:
        """Mappa il nome applicazione nfstream al codice nDPI corrispondente."""
        key = app_name.upper().split(".")[0]
        return _APP_TO_NDPI_CODE.get(key, 0)

    def extract_flow_metadata(self, flow: Any) -> Dict[str, Any]:
        """
        Estrae i metadati di identificazione del flow per il logging.

        Returns:
            Dict con src_ip, dst_ip, porte, protocollo, bytes, pacchetti.
        """
        g = self._get
        return {
            "src_ip":       g(flow, "src_ip",                   "0.0.0.0"),
            "dst_ip":       g(flow, "dst_ip",                   "0.0.0.0"),
            "src_port":     g(flow, "src_port",                  0),
            "dst_port":     g(flow, "dst_port",                  0),
            "protocol":     g(flow, "protocol",                  0),
            "l7_proto":     g(flow, "application_name",         "Unknown"),
            "duration_ms":  g(flow, "bidirectional_duration_ms", 0),
            "bytes_in":     g(flow, "src2dst_bytes",             0),
            "bytes_out":    g(flow, "dst2src_bytes",             0),
            "packets_in":   g(flow, "src2dst_packets",           0),
            "packets_out":  g(flow, "dst2src_packets",           0),
        }

    def validate_feature_vector(self, features: np.ndarray) -> bool:
        """
        Valida il vettore feature estratto.

        Args:
            features: Array numpy restituito da extract_features.

        Returns:
            True se valido, False altrimenti (con log del motivo).
        """
        if features.shape != (self.n_features,):
            logger.error(
                f"Shape errata: {features.shape}, atteso ({self.n_features},)"
            )
            return False

        if np.any(np.isnan(features)):
            logger.warning("Vettore feature contiene NaN")
            return False

        if np.any(np.isinf(features)):
            logger.warning("Vettore feature contiene Inf")
            return False

        return True
