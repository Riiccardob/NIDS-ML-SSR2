"""
Network Capture Engine con nfstream.

Gestisce la cattura pacchetti e la generazione dei flow NetFlow.

Nota sul problema di shutdown:
    nfstream espone i flow tramite un iteratore sincrono. Il thread di
    cattura rimane bloccato su `for flow in self.streamer` finche' non
    arriva un nuovo pacchetto dall'interfaccia. Su interfacce a basso
    traffico o a fine PCAP questo significa che `join(timeout=5)` scade
    sempre, lasciando il thread in vita.

    Soluzione adottata:
    1. `stop_event` viene impostato per segnalare l'intenzione di stop.
    2. Si tenta `streamer.terminate()` se disponibile (nfstream >= 6.4).
    3. Se il thread non termina entro `_SHUTDOWN_TIMEOUT_SEC` secondi,
       viene abbandonato (daemon=True garantisce che muoia con il processo
       principale). Non si usa Thread.kill() perche' non esiste in Python.
    4. Un sentinel None viene messo in coda per sbloccare eventualmente
       il consumer che attende su `flow_queue.get()`.

Nota su accounting_mode=3:
    Il dataset NF-UQ-NIDS-v2 e' stato generato con nProbe/nfdump in
    modalita' accounting_mode=3, che include gli header L2/L3/L4 nel
    conteggio dei byte. Usare accounting_mode=1 (solo payload) introduce
    una discrepanza sistematica su IN_BYTES, OUT_BYTES e sulle feature
    di throughput derivate (SRC_TO_DST_AVG_THROUGHPUT,
    DST_TO_SRC_AVG_THROUGHPUT). La modalita' 3 garantisce l'allineamento
    con i valori visti dallo scaler durante il training.
"""

import os
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Iterator, Optional

import psutil
from nfstream import NFStreamer

from config import (
    FLOW_ACTIVE_TIMEOUT,
    FLOW_IDLE_TIMEOUT,
    MAX_FLOWS_IN_MEMORY,
    NETWORK_INTERFACE,
)
from utils.logger import get_logger

logger = get_logger()

# Tempo massimo atteso per la terminazione del thread di cattura.
# Dopo questo timeout il thread viene abbandonato (daemon).
_SHUTDOWN_TIMEOUT_SEC: float = 8.0

# Sentinel inserito in coda per sbloccare il consumer durante lo shutdown.
_QUEUE_SENTINEL = None


@dataclass
class CaptureStats:
    """Statistiche di cattura aggiornate ogni secondo."""

    total_flows: int = 0
    total_packets: int = 0
    flows_per_second: float = 0.0
    packets_per_second: float = 0.0
    active_flows: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    dropped_flows: int = 0


class FlowCaptureEngine:
    """
    Engine per la cattura e la generazione dei flow NetFlow via nfstream.

    Il thread di cattura e' daemon: viene terminato automaticamente se
    il processo principale termina, anche in caso di join timeout.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        flow_callback: Optional[Callable[[Any], None]] = None,
        idle_timeout: int = FLOW_IDLE_TIMEOUT,
        active_timeout: int = FLOW_ACTIVE_TIMEOUT,
    ) -> None:
        """
        Args:
            interface:      Nome interfaccia di rete (None = auto-detect).
            flow_callback:  Funzione opzionale chiamata per ogni flow completato.
            idle_timeout:   Timeout idle flow in secondi.
                            Default: FLOW_IDLE_TIMEOUT da config (produzione).
                            Per la demo live usare 1 tramite --fast in main.py.
            active_timeout: Timeout active flow in secondi.
                            Default: FLOW_ACTIVE_TIMEOUT da config (produzione).
                            Per la demo live usare 10 tramite --fast in main.py.
        """
        self.interface = interface or self._auto_detect_interface()
        self.flow_callback = flow_callback
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout

        self.streamer: Optional[NFStreamer] = None
        self.flow_queue: Queue = Queue(maxsize=MAX_FLOWS_IN_MEMORY)

        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None

        self.stats = CaptureStats()
        self._stats_window_start = time.time()
        self._window_flows = 0
        self._window_packets = 0

        logger.info(f"FlowCaptureEngine inizializzato su interfaccia: {self.interface}")

    # ------------------------------------------------------------------
    # Ciclo di vita pubblico
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Avvia la cattura in un thread daemon in background."""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            logger.warning("Cattura gia' in esecuzione")
            return

        logger.info(f"Avvio cattura su {self.interface}")
        self.stop_event.clear()

        self.capture_thread = Thread(
            target=self._capture_loop,
            name="nids-capture",
            daemon=True,   # Garantisce terminazione con il processo principale
        )
        self.capture_thread.start()
        logger.info("Thread di cattura avviato")

    def stop(self) -> None:
        """
        Ferma la cattura in modo pulito.

        Strategia a tre livelli per evitare il deadlock sull'iteratore
        nfstream bloccante:
          1. Imposta stop_event per segnalare al loop di uscire.
          2. Tenta streamer.terminate() (sblocca l'iteratore internamente).
          3. Se il thread non termina entro _SHUTDOWN_TIMEOUT_SEC, lo
             abbandona (e' daemon, termina con il processo).
          4. Inserisce sentinel in coda per sbloccare eventuali consumer.
        """
        logger.info("Arresto cattura in corso...")
        self.stop_event.set()

        # Tentativo 1: usare l'API di terminazione di nfstream se disponibile.
        # NFStreamer.terminate() segnala al C layer di smettere di processare
        # pacchetti, causando l'uscita dall'iteratore senza attendere il
        # prossimo pacchetto.
        if self.streamer is not None:
            if hasattr(self.streamer, "terminate"):
                try:
                    self.streamer.terminate()
                    logger.info("NFStreamer.terminate() invocato")
                except Exception as exc:
                    logger.warning(f"NFStreamer.terminate() fallito: {exc}")

        # Tentativo 2: attendere la terminazione del thread.
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=_SHUTDOWN_TIMEOUT_SEC)
            if self.capture_thread.is_alive():
                # Il thread e' ancora bloccato sull'iteratore nfstream.
                # Non e' possibile ucciderlo forzatamente in Python senza
                # strumenti esterni. Poiche' e' daemon, terminera' quando
                # terminera' il processo principale.
                logger.warning(
                    f"Thread di cattura non terminato entro {_SHUTDOWN_TIMEOUT_SEC}s. "
                    "Il thread e' daemon e terminera' con il processo principale."
                )

        # Sentinel per sbloccare il consumer nella flow_queue.
        try:
            self.flow_queue.put_nowait(_QUEUE_SENTINEL)
        except Exception:
            pass

        logger.info("Cattura fermata")

    # ------------------------------------------------------------------
    # Interfaccia per il consumer (processing loop)
    # ------------------------------------------------------------------

    def get_flow(self, timeout: float = 1.0) -> Optional[Any]:
        """
        Restituisce il prossimo flow dalla coda.

        Returns:
            Flow nfstream, None se timeout o sentinel di shutdown.
        """
        try:
            flow = self.flow_queue.get(timeout=timeout)
            # Propaga il sentinel al caller per segnalare shutdown
            return flow
        except Empty:
            return None

    def iter_flows(self, timeout: float = 1.0) -> Iterator[Any]:
        """
        Iteratore sui flow: termina automaticamente allo stop.

        Restituisce None sul sentinel (fine stream / shutdown).
        """
        while not self.stop_event.is_set():
            flow = self.get_flow(timeout=timeout)
            if flow is _QUEUE_SENTINEL:
                break
            if flow is not None:
                yield flow

    def is_running(self) -> bool:
        """True se il thread di cattura e' attivo."""
        return self.capture_thread is not None and self.capture_thread.is_alive()

    def get_stats(self) -> CaptureStats:
        """Restituisce le statistiche di cattura correnti."""
        self.stats.active_flows = self.flow_queue.qsize()
        return self.stats

    # ------------------------------------------------------------------
    # Loop interno (eseguito nel thread daemon)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Loop principale di cattura.

        Il controllo `stop_event.is_set()` all'interno del for-loop e'
        l'unico modo affidabile per interrompere l'iterazione: nfstream
        non espone un'API di pausa. Se il C layer non risponde a
        terminate(), il loop esce solo alla prossima scadenza di un flow
        (idle_timeout o active_timeout).

        accounting_mode=3: conta i byte includendo gli header L2/L3/L4,
        allineato al metodo usato da nProbe/nfdump per generare il dataset
        NF-UQ-NIDS-v2.
        """
        try:
            self.streamer = NFStreamer(
                source=self.interface,
                idle_timeout=self.idle_timeout,
                active_timeout=self.active_timeout,
                accounting_mode=3,
                decode_tunnels=True,
                bpf_filter=None,
                promiscuous_mode=True,
                n_dissections=20,
                statistical_analysis=True,
                splt_analysis=0,
            )

            logger.info("NFStreamer inizializzato")
            logger.info(f"  idle_timeout:    {self.idle_timeout}s")
            logger.info(f"  active_timeout:  {self.active_timeout}s")
            logger.info(f"  accounting_mode: 3 (header L2/L3/L4 inclusi)")
            logger.info(f"  promiscuous:     attivo")

            for flow in self.streamer:
                if self.stop_event.is_set():
                    break

                self._update_stats(flow)

                try:
                    self.flow_queue.put(flow, timeout=1.0)
                except Exception:
                    self.stats.dropped_flows += 1
                    logger.warning(
                        f"Coda flow piena, flow scartato "
                        f"(totale scartati: {self.stats.dropped_flows})"
                    )
                    continue

                if self.flow_callback is not None:
                    try:
                        self.flow_callback(flow)
                    except Exception as exc:
                        logger.error(f"Errore nel flow callback: {exc}")

        except KeyboardInterrupt:
            logger.info("Cattura interrotta dall'utente")

        except Exception as exc:
            logger.error(f"Errore nel loop di cattura: {exc}")
            import traceback
            traceback.print_exc()

        finally:
            # Rilascia lo streamer esplicitamente per liberare le risorse C.
            if self.streamer is not None:
                try:
                    del self.streamer
                    self.streamer = None
                except Exception:
                    pass
            logger.info("Loop di cattura terminato")

    # ------------------------------------------------------------------
    # Statistiche
    # ------------------------------------------------------------------

    def _update_stats(self, flow: Any) -> None:
        """Aggiorna le statistiche di cattura con finestra temporale scorrevole."""
        if isinstance(flow, dict):
            packets = flow.get("bidirectional_packets", 0)
        else:
            packets = getattr(flow, "bidirectional_packets", 0)

        self._window_flows += 1
        self._window_packets += int(packets)

        elapsed = time.time() - self._stats_window_start
        if elapsed >= 1.0:
            self.stats.flows_per_second = self._window_flows / elapsed
            self.stats.packets_per_second = self._window_packets / elapsed
            self.stats.total_flows += self._window_flows
            self.stats.total_packets += self._window_packets
            self._window_flows = 0
            self._window_packets = 0
            self._stats_window_start = time.time()

        try:
            proc = psutil.Process(os.getpid())
            self.stats.memory_usage_mb = proc.memory_info().rss / 1_048_576
            self.stats.cpu_usage_percent = proc.cpu_percent(interval=None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_interface() -> str:
        """Rileva automaticamente l'interfaccia di rete principale attiva."""
        try:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()

            for iface, addr_list in addrs.items():
                if iface.startswith("lo"):
                    continue
                if iface not in stats or not stats[iface].isup:
                    continue
                for addr in addr_list:
                    if addr.family == 2:  # AF_INET
                        logger.info(f"Interfaccia rilevata automaticamente: {iface} ({addr.address})")
                        return iface
        except Exception as exc:
            logger.warning(f"Auto-detect interfaccia fallito: {exc}")

        logger.warning("Impossibile rilevare interfaccia, uso 'eth0' come fallback")
        return "eth0"