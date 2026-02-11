"""
Alert Manager per gestione e logging degli eventi di sicurezza.

Responsabilita':
- Coordinare logging strutturato di alert e flow benigni
- Applicare rate limiting per IP per evitare storm di alert
- Coordinare il blocco IP tramite FirewallController in BLOCK mode
- Esporre statistiche utili al calcolo del False Positive Rate

Rate limiting:
    Ogni IP che genera un alert viene messo in cooldown per
    alert_cooldown secondi (configurabile nel costruttore, default 30s).
    Alert successivi dallo stesso IP vengono contati ma non loggati
    nuovamente. Questo evita migliaia di subprocess iptables e loop di
    scrittura su disco durante un attacco sostenuto come un SYN flood.

    Per la demo live usare cooldown basso (es. 5s) tramite --cooldown.
    Per la produzione mantenere il default di 30s.

Log dei flow benigni:
    I flow classificati come benigni vengono loggati a campione
    (1 ogni BENIGN_SAMPLE_RATE flow). Questo permette di calcolare
    il False Positive Rate in produzione senza saturare il disco.
    Il campionamento e' deterministico (modulo sul contatore).
"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

from config import OPERATION_MODE, OperationMode
from security.firewall_controller import FirewallController
from utils.logger import get_logger

logger = get_logger()

# Campionamento flow benigni: 1 ogni N flow viene loggato.
# Valore 100 = 1% dei flow benigni, sufficiente per stimare FPR.
# Impostare a 1 per loggare tutto (sconsigliato in produzione).
BENIGN_SAMPLE_RATE: int = 100


class AlertManager:
    """
    Manager per la gestione degli alert e delle azioni di risposta.

    Thread-safety: non e' thread-safe. Il caller (processing loop) deve
    garantire che process_flow() venga chiamato da un unico thread.
    """

    def __init__(
        self,
        operation_mode: OperationMode = OPERATION_MODE,
        firewall: Optional[FirewallController] = None,
        alert_cooldown: float = 30.0,
    ) -> None:
        """
        Args:
            operation_mode:  ALERT (solo log) o BLOCK (log + firewall).
            firewall:        FirewallController, obbligatorio se mode=BLOCK.
            alert_cooldown:  Secondi di cooldown per IP dopo un alert.
                             Default 30s per produzione.
                             Usare valori bassi (es. 5s) per demo live.
        """
        self.operation_mode = operation_mode
        self.firewall = firewall
        self._alert_cooldown_sec: float = alert_cooldown

        if self.operation_mode == OperationMode.BLOCK and self.firewall is None:
            raise ValueError("FirewallController obbligatorio in BLOCK mode")

        # Statistiche cumulative
        self.total_alerts: int = 0
        self.total_blocks: int = 0
        self.total_benign_logged: int = 0
        self.total_benign_seen: int = 0
        self._suppressed_alerts: int = 0

        self.alerts_by_ip: Dict[str, int] = defaultdict(int)
        self.alerts_by_protocol: Dict[str, int] = defaultdict(int)

        # Rate limiting: ip -> timestamp dell'ultimo alert loggato
        self._last_alert_time: Dict[str, float] = {}

        # Contatore per il campionamento benigni
        self._benign_counter: int = 0

        logger.info(f"AlertManager inizializzato in modalita' {operation_mode.value.upper()}")
        logger.info(
            f"  Cooldown alert per IP: {self._alert_cooldown_sec}s | "
            f"Campionamento benigni: 1/{BENIGN_SAMPLE_RATE}"
        )

    # ------------------------------------------------------------------
    # Interfaccia principale
    # ------------------------------------------------------------------

    def process_flow(
        self,
        src_ip: str,
        dst_ip: str,
        prediction: int,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Processa un flow classificato (attack o benign).

        Riceve tutti i flow per consentire il logging campionato dei
        flow benigni, utile al calcolo del FPR in produzione.

        Args:
            src_ip:     IP sorgente.
            dst_ip:     IP destinazione.
            prediction: 1=attack, 0=benign.
            confidence: Probabilita' di attacco [0, 1].
            metadata:   Dict con porte, protocollo, byte, pacchetti.

        Returns:
            Stringa che descrive l'azione intrapresa:
            "alert_logged", "alert_suppressed", "blocked",
            "already_blocked", "whitelisted", "benign_sampled", "benign_skipped"
        """
        if prediction == 1:
            return self._process_attack(src_ip, dst_ip, confidence, metadata)
        else:
            return self._process_benign(src_ip, dst_ip, confidence, metadata)

    # ------------------------------------------------------------------
    # Logica interna
    # ------------------------------------------------------------------

    def _process_attack(
        self,
        src_ip: str,
        dst_ip: str,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> str:
        """Gestisce un flow classificato come attacco."""
        self.total_alerts += 1
        self.alerts_by_ip[src_ip] += 1
        protocol = str(metadata.get("protocol", "unknown"))
        self.alerts_by_protocol[protocol] += 1

        # Rate limiting: controlla se l'IP e' in cooldown
        now = time.monotonic()
        last_time = self._last_alert_time.get(src_ip, 0.0)
        in_cooldown = (now - last_time) < self._alert_cooldown_sec

        if in_cooldown:
            self._suppressed_alerts += 1
            return "alert_suppressed"

        # Aggiorna il timestamp di ultimo alert per questo IP
        self._last_alert_time[src_ip] = now

        # Determina azione
        action = "alert_logged"

        if self.operation_mode == OperationMode.BLOCK:
            if self.firewall is not None:
                if src_ip in self.firewall.whitelist:
                    action = "whitelisted"
                elif self.firewall.is_blocked(src_ip):
                    action = "already_blocked"
                else:
                    reason = f"confidence_{confidence:.3f}"
                    if self.firewall.block_ip(src_ip, reason=reason):
                        action = "blocked"
                        self.total_blocks += 1

        alert_data = self._build_record(
            src_ip, dst_ip, "attack", confidence, action, metadata
        )
        logger.log_alert(alert_data)

        return action

    def _process_benign(
        self,
        src_ip: str,
        dst_ip: str,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Gestisce un flow classificato come benigno.

        Logga 1 flow ogni BENIGN_SAMPLE_RATE per consentire il calcolo
        del False Positive Rate in produzione.
        """
        self.total_benign_seen += 1
        self._benign_counter += 1

        if self._benign_counter % BENIGN_SAMPLE_RATE != 0:
            return "benign_skipped"

        self.total_benign_logged += 1
        benign_data = self._build_record(
            src_ip, dst_ip, "benign", confidence, "logged", metadata
        )
        logger.log_benign(benign_data)

        return "benign_sampled"

    @staticmethod
    def _build_record(
        src_ip: str,
        dst_ip: str,
        prediction_label: str,
        confidence: float,
        action: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Costruisce il dizionario da passare al logger strutturato."""
        return {
            "timestamp":   datetime.now().isoformat(),
            "src_ip":      src_ip,
            "src_port":    metadata.get("src_port", 0),
            "dst_ip":      dst_ip,
            "dst_port":    metadata.get("dst_port", 0),
            "protocol":    metadata.get("protocol", 0),
            "l7_proto":    metadata.get("l7_proto", "unknown"),
            "prediction":  prediction_label,
            "confidence":  confidence,
            "action":      action,
            "duration_ms": metadata.get("duration_ms", 0),
            "bytes_in":    metadata.get("bytes_in", 0),
            "bytes_out":   metadata.get("bytes_out", 0),
            "packets_in":  metadata.get("packets_in", 0),
            "packets_out": metadata.get("packets_out", 0),
        }

    # ------------------------------------------------------------------
    # Statistiche
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """
        Restituisce le statistiche aggregate dell'alert manager.

        Include dati per il calcolo del False Positive Rate:
            fpr_estimate = total_alerts / (total_alerts + total_benign_seen)
            (stima approssimata, non il FPR reale che richiederebbe ground truth)
        """
        total_classified = self.total_alerts + self.total_benign_seen
        alert_rate = (
            self.total_alerts / total_classified if total_classified > 0 else 0.0
        )

        stats: Dict[str, Any] = {
            "operation_mode":      self.operation_mode.value,
            "alert_cooldown_sec":  self._alert_cooldown_sec,
            "total_alerts":        self.total_alerts,
            "total_blocks":        self.total_blocks,
            "total_benign_seen":   self.total_benign_seen,
            "total_benign_logged": self.total_benign_logged,
            "suppressed_alerts":   self._suppressed_alerts,
            "alert_rate_pct":      round(alert_rate * 100, 2),
            "top_alerting_ips":    self._get_top_n(self.alerts_by_ip, 10),
            "alerts_by_protocol":  dict(self.alerts_by_protocol),
        }

        if self.firewall is not None:
            stats["blocked_ips_count"] = self.firewall.get_block_count()
            stats["whitelist_count"] = len(self.firewall.whitelist)

        return stats

    def reset_statistics(self) -> None:
        """Azzera tutti i contatori (utile per test)."""
        self.total_alerts = 0
        self.total_blocks = 0
        self.total_benign_logged = 0
        self.total_benign_seen = 0
        self._suppressed_alerts = 0
        self.alerts_by_ip.clear()
        self.alerts_by_protocol.clear()
        self._last_alert_time.clear()
        self._benign_counter = 0
        logger.info("Statistiche alert azzerate")

    @staticmethod
    def _get_top_n(counter: Dict[str, int], n: int) -> Dict[str, int]:
        """Restituisce i primi N elementi ordinati per valore decrescente."""
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:n])