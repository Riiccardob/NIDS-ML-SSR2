"""
Live NIDS Sniffer -- Entry Point.

Network Intrusion Detection System basato su ML con nfstream.

Utilizzo:
    sudo $(which python) main.py --mode alert --interface eth0
    sudo $(which python) main.py --mode alert --interface br-xxxx --fast --verbose
    sudo $(which python) main.py --mode block --interface eth0 --idle-timeout 10
    sudo $(which python) main.py --mode alert --interface eth0 --batch-size 200
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Il modulo e' eseguito direttamente: aggiusta sys.path prima di qualsiasi
# import locale.
_SNIFFER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SNIFFER_DIR.parent.parent
for _p in (_PROJECT_ROOT, _SNIFFER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import (
    FLOW_ACTIVE_TIMEOUT,
    FLOW_EXPIRATION_CHECK_INTERVAL,
    FLOW_IDLE_TIMEOUT,
    INFERENCE_BATCH_SIZE,
    INFERENCE_BATCH_TIMEOUT,
    NETWORK_INTERFACE,
    OPERATION_MODE,
    STATS_LOG_INTERVAL,
    OperationMode,
    get_config_summary,
    validate_config,
)
from core.capture import FlowCaptureEngine
from core.feature_mapper import FeatureMapper
from core.predictor import ModelPredictor
from core.preprocessor import FeaturePreprocessor
from security.alert_manager import AlertManager
from security.firewall_controller import FirewallController
from utils.logger import get_logger

logger = get_logger()

# Codici ANSI per output console colorato.
# Usati esclusivamente in _print_live_event(): il logger strutturato
# (CSV/JSONL) non viene mai toccato da questi codici.
_ANSI_RED    = "\033[91m\033[1m"
_ANSI_GREEN  = "\033[92m"
_ANSI_GREY   = "\033[90m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET  = "\033[0m"


class LiveNIDSSniffer:
    """
    Orchestratore del Live NIDS Sniffer.

    Ciclo di vita:
        sniffer = LiveNIDSSniffer(...)
        sniffer.setup()
        sniffer.start()   # bloccante fino a Ctrl+C o SIGTERM
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        operation_mode: OperationMode = OPERATION_MODE,
        batch_size: int = INFERENCE_BATCH_SIZE,
        batch_timeout: float = INFERENCE_BATCH_TIMEOUT,
        idle_timeout: int = FLOW_IDLE_TIMEOUT,
        active_timeout: int = FLOW_ACTIVE_TIMEOUT,
        verbose: bool = False,
        alert_cooldown: float = 30.0,
    ) -> None:
        """
        Args:
            interface:       Interfaccia di rete (None = auto-detect).
            operation_mode:  ALERT o BLOCK.
            batch_size:      Flow per batch di inferenza.
            batch_timeout:   Secondi massimi di attesa prima di forzare il batch.
            idle_timeout:    Idle timeout nfstream in secondi. Con --fast = 1.
            active_timeout:  Active timeout nfstream in secondi. Con --fast = 10.
            verbose:         Se True, stampa a console anche i flow benigni.
            alert_cooldown:  Cooldown in secondi per rate limiting per IP.
        """
        self.interface = interface or NETWORK_INTERFACE
        self.operation_mode = operation_mode
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.verbose = verbose
        self.alert_cooldown = alert_cooldown

        # Componenti (inizializzati in setup())
        self.capture_engine: Optional[FlowCaptureEngine] = None
        self.feature_mapper: Optional[FeatureMapper] = None
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.predictor: Optional[ModelPredictor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.firewall: Optional[FirewallController] = None

        # Buffer batch
        self.feature_batch: List[np.ndarray] = []
        self.metadata_batch: List[dict] = []
        self.batch_start_time: float = time.monotonic()

        # Contatori
        self.total_flows_processed: int = 0
        self.total_predictions: int = 0

        # Timer
        self.last_stats_time: float = time.monotonic()
        self.last_cleanup_time: float = time.monotonic()

        self.running: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Inizializza e valida tutti i componenti.

        Raises:
            ValueError: Se la configurazione e' inconsistente.
            FileNotFoundError: Se mancano artifacts.
        """
        logger.info("=" * 70)
        logger.info("NIDS LIVE SNIFFER -- INIZIALIZZAZIONE")
        logger.info("=" * 70)

        validate_config()
        logger.info("Configurazione validata")

        cfg = get_config_summary()
        logger.info(f"  Modalita':       {cfg['operation_mode']}")
        logger.info(f"  Modello:         {cfg['model_type']} ({cfg['n_features']} feature)")
        logger.info(f"  Interfaccia:     {self.interface or 'auto-detect'}")
        logger.info(f"  Threshold:       {cfg['attack_threshold']}")
        logger.info(f"  Batch:           {self.batch_size} flow / {self.batch_timeout}s")
        logger.info(f"  Idle timeout:    {self.idle_timeout}s")
        logger.info(f"  Active timeout:  {self.active_timeout}s")
        logger.info(f"  Verbose:         {self.verbose}")
        logger.info(f"  Alert cooldown:  {self.alert_cooldown}s")

        if self.operation_mode == OperationMode.BLOCK:
            logger.info("\nInizializzazione FirewallController...")
            self.firewall = FirewallController()

        logger.info("\nInizializzazione componenti ML...")
        self.feature_mapper = FeatureMapper()
        self.preprocessor = FeaturePreprocessor()
        self.predictor = ModelPredictor()

        self.alert_manager = AlertManager(
            operation_mode=self.operation_mode,
            firewall=self.firewall,
            alert_cooldown=self.alert_cooldown,
        )

        logger.info("\nInizializzazione capture engine...")
        self.capture_engine = FlowCaptureEngine(
            interface=self.interface,
            idle_timeout=self.idle_timeout,
            active_timeout=self.active_timeout,
        )

        logger.info("\n" + "=" * 70)
        logger.info("SETUP COMPLETATO")
        logger.info("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Ciclo di vita
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Avvia lo sniffer.

        Bloccante: ritorna solo quando l'utente invia Ctrl+C / SIGTERM
        o si verifica un errore fatale.
        """
        logger.info("Avvio Live NIDS Sniffer...")
        logger.info(f"Modalita': {self.operation_mode.value.upper()}")
        logger.info(f"Interfaccia: {self.interface}")
        logger.info("Premi Ctrl+C per fermare\n")

        # Banner console separato dal logger strutturato
        if self.verbose:
            print(
                f"{_ANSI_GREEN}[*] Verbose mode attivo: "
                f"verranno mostrati anche i flow benigni{_ANSI_RESET}"
            )
        print(
            f"{_ANSI_YELLOW}[*] Idle timeout: {self.idle_timeout}s  "
            f"Active timeout: {self.active_timeout}s  "
            f"Cooldown alert: {self.alert_cooldown}s{_ANSI_RESET}\n"
        )

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.capture_engine.start()
        self.running = True

        try:
            self._processing_loop()
        except KeyboardInterrupt:
            logger.info("\nInterrotto dall'utente")
        except Exception as exc:
            logger.error(f"Errore fatale nel processing loop: {exc}")
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Ferma lo sniffer in modo pulito.

        Ordine di shutdown:
          1. Segnala stop al processing loop.
          2. Processa il batch rimanente.
          3. Ferma il capture engine.
          4. Log statistiche finali.
          5. Teardown firewall (se BLOCK mode).
          6. Shutdown logger (flush writer thread asincroni).
        """
        logger.info("\nArresto Live NIDS Sniffer in corso...")
        self.running = False

        if self.feature_batch:
            logger.info(f"Elaborazione batch residuo ({len(self.feature_batch)} flow)...")
            self._process_batch()

        if self.capture_engine:
            self.capture_engine.stop()

        self._log_statistics()

        if self.alert_manager:
            alert_stats = self.alert_manager.get_statistics()
            logger.info("\n" + "=" * 70)
            logger.info("STATISTICHE FINALI")
            logger.info("=" * 70)
            logger.info(f"Flow processati:   {self.total_flows_processed:,}")
            logger.info(f"Predizioni totali: {self.total_predictions:,}")
            logger.info(f"Alert totali:      {alert_stats['total_alerts']:,}")
            logger.info(f"Alert soppressi:   {alert_stats['suppressed_alerts']:,}")
            logger.info(f"Blocchi totali:    {alert_stats['total_blocks']:,}")
            logger.info(f"Benigni visti:     {alert_stats['total_benign_seen']:,}")
            logger.info(f"Benigni loggati:   {alert_stats['total_benign_logged']:,}")
            logger.info(f"Alert rate:        {alert_stats['alert_rate_pct']:.2f}%")

            top_ips = alert_stats.get("top_alerting_ips", {})
            if top_ips:
                logger.info("\nTop IP attaccanti:")
                for ip, count in list(top_ips.items())[:5]:
                    logger.info(f"  {ip}: {count} alert")

        if self.firewall:
            logger.info("\nCleanup firewall...")
            self.firewall.teardown()

        # Shutdown del logger DOPO tutto il resto per non perdere log.
        logger.info("\n" + "=" * 70)
        logger.info("SNIFFER FERMATO")
        logger.info("=" * 70)
        logger.shutdown()

    # ------------------------------------------------------------------
    # Loop di processing
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """Loop principale di lettura dalla queue e invio batch."""
        logger.info("Processing loop avviato\n")

        while self.running:
            flow = self.capture_engine.get_flow(timeout=0.1)

            if flow is not None:
                self._process_flow(flow)

            now = time.monotonic()

            batch_elapsed = now - self.batch_start_time
            if batch_elapsed >= self.batch_timeout and self.feature_batch:
                self._process_batch()

            if len(self.feature_batch) >= self.batch_size:
                self._process_batch()

            if (now - self.last_stats_time) >= STATS_LOG_INTERVAL:
                self._log_statistics()

            if (
                self.firewall is not None
                and (now - self.last_cleanup_time) >= FLOW_EXPIRATION_CHECK_INTERVAL
            ):
                self.firewall.cleanup_expired_blocks()
                self.last_cleanup_time = now

    def _process_flow(self, flow: object) -> None:
        """Estrae le feature da un singolo flow e lo aggiunge al batch."""
        try:
            features = self.feature_mapper.extract_features(flow)

            if not self.feature_mapper.validate_feature_vector(features):
                return

            metadata = self.feature_mapper.extract_flow_metadata(flow)
            self.feature_batch.append(features)
            self.metadata_batch.append(metadata)
            self.total_flows_processed += 1

        except Exception as exc:
            logger.error(f"Errore nell'elaborazione del flow: {exc}")

    def _process_batch(self) -> None:
        """
        Elabora il batch corrente: scaling + predizione + alert.

        Tutti i flow del batch vengono passati all'AlertManager (non solo
        gli attacchi) in modo che i flow benigni campionati possano essere
        loggati per il calcolo del FPR.

        Per ogni predizione viene anche chiamato _print_live_event() per
        l'output colorato a console (separato dal logger strutturato).
        """
        if not self.feature_batch:
            return

        try:
            feature_matrix = np.vstack(self.feature_batch)
            scaled_matrix = self.preprocessor.preprocess_batch(feature_matrix)
            results = self.predictor.predict_batch(scaled_matrix)

            for result, metadata in zip(results, self.metadata_batch):
                self.alert_manager.process_flow(
                    src_ip=metadata["src_ip"],
                    dst_ip=metadata["dst_ip"],
                    prediction=result.prediction,
                    confidence=result.confidence,
                    metadata=metadata,
                )
                self._print_live_event(result, metadata)
                self.total_predictions += 1

        except Exception as exc:
            logger.error(f"Errore nell'elaborazione del batch: {exc}")

        finally:
            self.feature_batch.clear()
            self.metadata_batch.clear()
            self.batch_start_time = time.monotonic()

    def _print_live_event(self, result: object, metadata: dict) -> None:
        """
        Stampa a console l'evento in tempo reale con colorazione ANSI.

        Gli attacchi vengono sempre mostrati (rosso grassetto).
        I flow benigni vengono mostrati solo se --verbose e' attivo (grigio).

        Nota: questo metodo scrive su stdout direttamente, separato dal
        logger strutturato che scrive su CSV/JSONL. Non ci sono codici ANSI
        nei file di log.
        """
        src   = f"{metadata.get('src_ip', '?')}:{metadata.get('src_port', '?')}"
        dst   = f"{metadata.get('dst_ip', '?')}:{metadata.get('dst_port', '?')}"
        conf  = float(getattr(result, "confidence", 0.0))
        pred  = int(getattr(result, "prediction", 0))
        proto = metadata.get("l7_proto", "?")

        if pred == 1:
            print(
                f"{_ANSI_RED}"
                f"[ATTACK] {src} -> {dst}"
                f"  proto={proto}  conf={conf:.4f}"
                f"{_ANSI_RESET}"
            )
        elif self.verbose:
            print(
                f"{_ANSI_GREY}"
                f"[SAFE]   {src} -> {dst}"
                f"  proto={proto}  conf={conf:.4f}"
                f"{_ANSI_RESET}"
            )

    # ------------------------------------------------------------------
    # Statistiche e segnali
    # ------------------------------------------------------------------

    def _log_statistics(self) -> None:
        """Registra le statistiche operative periodiche."""
        capture_stats = self.capture_engine.get_stats()

        alert_stats: dict = {}
        if self.alert_manager:
            alert_stats = self.alert_manager.get_statistics()

        stats = {
            "timestamp":          datetime.now().isoformat(),
            "total_flows":        self.total_flows_processed,
            "total_predictions":  self.total_predictions,
            "total_alerts":       alert_stats.get("total_alerts", 0),
            "total_blocks":       alert_stats.get("total_blocks", 0),
            "total_benign_seen":  alert_stats.get("total_benign_seen", 0),
            "suppressed_alerts":  alert_stats.get("suppressed_alerts", 0),
            "alert_rate_pct":     alert_stats.get("alert_rate_pct", 0.0),
            "flows_per_second":   capture_stats.flows_per_second,
            "dropped_flows":      capture_stats.dropped_flows,
            "memory_usage_mb":    capture_stats.memory_usage_mb,
            "cpu_usage_percent":  capture_stats.cpu_usage_percent,
            "batch_pending":      len(self.feature_batch),
        }

        logger.log_statistics(stats)
        self.last_stats_time = time.monotonic()

    def _signal_handler(self, signum: int, frame: object) -> None:
        logger.info(f"\nSegnale {signum} ricevuto, arresto in corso...")
        self.running = False


# --------------------------------------------------------------------------
# Entry point CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Network Intrusion Detection System (ML-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Demo veloce (timeout ridotti, output colorato):
  sudo $(which python) main.py --mode alert --interface br-xxxx --fast --verbose

  # Demo con cooldown basso per vedere alert ripetuti:
  sudo $(which python) main.py --mode alert --interface br-xxxx --fast --verbose --cooldown 5

  # Produzione (timeout default, nessun output verbose):
  sudo $(which python) main.py --mode alert --interface eth0

  # Modalita' BLOCK con timeout personalizzati:
  sudo $(which python) main.py --mode block --interface eth0 --idle-timeout 30 --active-timeout 300

  # Batch personalizzato:
  sudo $(which python) main.py --mode alert --batch-size 200 --batch-timeout 3.0
        """,
    )

    parser.add_argument(
        "--interface", "-i",
        type=str,
        default=None,
        help="Interfaccia di rete da monitorare (default: auto-detect)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["alert", "block"],
        default="alert",
        help="Modalita' operativa: alert (solo log) o block (log + firewall) [default: alert]",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=INFERENCE_BATCH_SIZE,
        metavar="N",
        help=f"Numero di flow per batch di inferenza [default: {INFERENCE_BATCH_SIZE}]",
    )
    parser.add_argument(
        "--batch-timeout", "-t",
        type=float,
        default=INFERENCE_BATCH_TIMEOUT,
        metavar="SEC",
        help=f"Timeout batch in secondi [default: {INFERENCE_BATCH_TIMEOUT}]",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            f"Idle timeout nfstream in secondi (default produzione: {FLOW_IDLE_TIMEOUT}s). "
            "Sovrascrive --fast se specificato esplicitamente."
        ),
    )
    parser.add_argument(
        "--active-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            f"Active timeout nfstream in secondi (default produzione: {FLOW_ACTIVE_TIMEOUT}s). "
            "Sovrascrive --fast se specificato esplicitamente."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Modalita' Demo: idle-timeout=1s, active-timeout=10s. "
            "Riduce la latenza da ~120s a ~1s dopo l'ultimo pacchetto del flow. "
            "I valori espliciti --idle-timeout / --active-timeout hanno precedenza."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostra a console (con colore grigio) anche i flow classificati come benigni",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=30.0,
        metavar="SEC",
        help=(
            "Cooldown rate limiting alert per IP in secondi [default: 30.0]. "
            "Usare valori bassi (es. 5) per demo in cui si vogliono vedere alert ripetuti."
        ),
    )

    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERRORE: Questo script richiede privilegi root (sudo)")
        sys.exit(1)

    # Risoluzione timeout.
    # Priorita': valore esplicito > --fast > default config.
    idle_to = (
        args.idle_timeout
        if args.idle_timeout is not None
        else (1 if args.fast else FLOW_IDLE_TIMEOUT)
    )
    active_to = (
        args.active_timeout
        if args.active_timeout is not None
        else (10 if args.fast else FLOW_ACTIVE_TIMEOUT)
    )

    if args.fast:
        print(
            f"{_ANSI_YELLOW}"
            f"[!] Modalita' FAST attiva: idle={idle_to}s  active={active_to}s"
            f"{_ANSI_RESET}"
        )

    operation_mode = (
        OperationMode.BLOCK if args.mode == "block" else OperationMode.ALERT
    )

    try:
        sniffer = LiveNIDSSniffer(
            interface=args.interface,
            operation_mode=operation_mode,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            idle_timeout=idle_to,
            active_timeout=active_to,
            verbose=args.verbose,
            alert_cooldown=args.cooldown,
        )
        sniffer.setup()
        sniffer.start()

    except KeyboardInterrupt:
        sys.exit(0)

    except Exception as exc:
        logger.error(f"Errore fatale: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()