"""
Logger strutturato per Live NIDS Sniffer.

Gestisce logging in formato CSV e JSON con:
- Buffering e scrittura asincrona per non bloccare il processing loop
- File separati per alert (attacchi) e flow benigni campionati
- Rotazione automatica dei file di testo
- Shutdown pulito con flush completo prima dell'uscita
"""

import csv
import json
import logging
import queue
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    LOG_BACKUP_COUNT,
    LOG_FORMAT_TYPE,
    LOG_LEVEL,
    LOG_MAX_SIZE_MB,
    LOG_PREFIX,
    LOGS_DIR,
    LogFormat,
)

_WRITE_QUEUE_MAXSIZE: int = 10_000
_FLUSH_INTERVAL_SEC: float = 2.0
_BUFFER_SIZE: int = 50


class StructuredLogger:
    """
    Logger strutturato per eventi di sicurezza.

    La scrittura su CSV e JSON avviene su thread separati tramite code
    interne, in modo che il processing loop non venga mai bloccato da I/O.

    File prodotti:
        nids_sniffer.log          - log testuale con rotazione
        nids_sniffer_alerts.csv   - alert di attacco (CSV)
        nids_sniffer_alerts.jsonl - alert di attacco (JSONL)
        nids_sniffer_benign.csv   - flow benigni campionati (CSV)
        nids_sniffer_benign.jsonl - flow benigni campionati (JSONL)
        nids_sniffer_stats.jsonl  - statistiche operative periodiche
    """

    def __init__(self, name: str = "sniffer") -> None:
        self.name = name
        self._logger = self._setup_stdlib_logger()

        self._csv_alert_writer:   Optional[AsyncCSVWriter]  = None
        self._json_alert_writer:  Optional[AsyncJSONWriter] = None
        self._csv_benign_writer:  Optional[AsyncCSVWriter]  = None
        self._json_benign_writer: Optional[AsyncJSONWriter] = None

        self._setup_writers()

    def _setup_stdlib_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(LOG_LEVEL)
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setLevel(LOG_LEVEL)
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = RotatingFileHandler(
            LOGS_DIR / f"{LOG_PREFIX}.log",
            maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _setup_writers(self) -> None:
        if LOG_FORMAT_TYPE in (LogFormat.CSV, LogFormat.BOTH):
            self._csv_alert_writer = AsyncCSVWriter(
                LOGS_DIR / f"{LOG_PREFIX}_alerts.csv"
            )
            self._csv_alert_writer.start()

            self._csv_benign_writer = AsyncCSVWriter(
                LOGS_DIR / f"{LOG_PREFIX}_benign.csv"
            )
            self._csv_benign_writer.start()

        if LOG_FORMAT_TYPE in (LogFormat.JSON, LogFormat.BOTH):
            self._json_alert_writer = AsyncJSONWriter(
                LOGS_DIR / f"{LOG_PREFIX}_alerts.jsonl"
            )
            self._json_alert_writer.start()

            self._json_benign_writer = AsyncJSONWriter(
                LOGS_DIR / f"{LOG_PREFIX}_benign.jsonl"
            )
            self._json_benign_writer.start()

    # ------------------------------------------------------------------
    # Log stdlib
    # ------------------------------------------------------------------

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(message, *args)

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(message, *args)

    def critical(self, message: str, *args: Any) -> None:
        self._logger.critical(message, *args)

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(message, *args)

    # ------------------------------------------------------------------
    # Log strutturato
    # ------------------------------------------------------------------

    def log_alert(self, alert_data: Dict[str, Any]) -> None:
        """
        Registra un alert di attacco in modo asincrono.

        Scrive una riga sincrona su console/log testuale per visibilita'
        immediata, poi accoda la scrittura su CSV/JSON al writer thread.

        Args:
            alert_data: Dizionario con i campi dell'alert.
        """
        if "timestamp" not in alert_data:
            alert_data["timestamp"] = datetime.now().isoformat()

        self._logger.warning(
            "ALERT %s:%s -> %s:%s proto=%s conf=%.4f action=%s",
            alert_data.get("src_ip", "?"),
            alert_data.get("src_port", "?"),
            alert_data.get("dst_ip", "?"),
            alert_data.get("dst_port", "?"),
            alert_data.get("protocol", "?"),
            float(alert_data.get("confidence", 0.0)),
            alert_data.get("action", "?"),
        )

        if self._csv_alert_writer:
            self._csv_alert_writer.enqueue(alert_data)
        if self._json_alert_writer:
            self._json_alert_writer.enqueue(alert_data)

    def log_benign(self, flow_data: Dict[str, Any]) -> None:
        """
        Registra un flow benigno campionato in modo asincrono.

        I flow benigni vengono scritti in file separati dagli alert per
        non mescolare i dataset nell'analisi successiva. Questi dati
        permettono di calcolare il False Positive Rate in produzione.

        Args:
            flow_data: Dizionario con i campi del flow benigno.
        """
        if "timestamp" not in flow_data:
            flow_data["timestamp"] = datetime.now().isoformat()

        if self._csv_benign_writer:
            self._csv_benign_writer.enqueue(flow_data)
        if self._json_benign_writer:
            self._json_benign_writer.enqueue(flow_data)

    def log_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Registra le statistiche operative su file JSONL dedicato.

        La scrittura e' sincrona (una riga ogni 60 secondi): il costo
        I/O e' trascurabile rispetto ai vantaggi di non usare un quinto
        writer thread.

        Args:
            stats: Dict con metriche operative.
        """
        stats["timestamp"] = datetime.now().isoformat()

        self._logger.info(
            "STATS flows=%d pred=%d alerts=%d blocks=%d benign_seen=%d fps=%.1f mem=%.0fMB",
            stats.get("total_flows", 0),
            stats.get("total_predictions", 0),
            stats.get("total_alerts", 0),
            stats.get("total_blocks", 0),
            stats.get("total_benign_seen", 0),
            float(stats.get("flows_per_second", 0.0)),
            float(stats.get("memory_usage_mb", 0.0)),
        )

        try:
            with open(LOGS_DIR / f"{LOG_PREFIX}_stats.jsonl", "a") as fh:
                fh.write(json.dumps(stats) + "\n")
        except OSError as exc:
            self._logger.error("Impossibile scrivere stats file: %s", exc)

    def shutdown(self) -> None:
        """
        Attende il flush completo di tutti i writer e li termina.

        Da chiamare nello stop() dello sniffer per garantire che nessun
        record venga perso prima dell'uscita del processo.
        """
        writers: List[Optional[AsyncStructuredWriter]] = [
            self._csv_alert_writer,
            self._json_alert_writer,
            self._csv_benign_writer,
            self._json_benign_writer,
        ]
        for writer in writers:
            if writer is not None:
                writer.stop()
        self._logger.info("Logger shutdown completato")


# --------------------------------------------------------------------------
# Writer asincroni
# --------------------------------------------------------------------------

class AsyncStructuredWriter:
    """
    Writer asincrono con buffer e flush periodico.

    Un thread daemon consuma la coda, accumula i record in buffer interno
    e scrive su disco in batch ogni _FLUSH_INTERVAL_SEC secondi o quando
    il buffer raggiunge _BUFFER_SIZE record.
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(
            maxsize=_WRITE_QUEUE_MAXSIZE
        )
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"nids-writer-{self.filepath.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logging.getLogger("sniffer").warning(
                "Writer thread %s non terminato entro 10s", self.filepath.name
            )

    def enqueue(self, record: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            logging.getLogger("sniffer").warning(
                "Coda scrittura piena (%d), record scartato", _WRITE_QUEUE_MAXSIZE
            )

    def _writer_loop(self) -> None:
        buffer: List[Dict[str, Any]] = []

        while True:
            try:
                record = self._queue.get(timeout=_FLUSH_INTERVAL_SEC)
            except queue.Empty:
                if buffer:
                    self._flush(buffer)
                    buffer = []
                continue

            if record is None:
                break

            buffer.append(record)
            if len(buffer) >= _BUFFER_SIZE:
                self._flush(buffer)
                buffer = []

        while not self._queue.empty():
            try:
                r = self._queue.get_nowait()
                if r is not None:
                    buffer.append(r)
            except queue.Empty:
                break

        if buffer:
            self._flush(buffer)

    def _flush(self, records: List[Dict[str, Any]]) -> None:
        raise NotImplementedError


class AsyncCSVWriter(AsyncStructuredWriter):

    FIELDS = [
        "timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
        "protocol", "l7_proto", "prediction", "confidence", "action",
        "duration_ms", "bytes_in", "bytes_out", "packets_in", "packets_out",
    ]

    def __init__(self, filepath: Path) -> None:
        super().__init__(filepath)
        if not filepath.exists():
            try:
                with open(filepath, "w", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=self.FIELDS).writeheader()
            except OSError as exc:
                logging.getLogger("sniffer").error("Impossibile creare %s: %s", filepath, exc)

    def _flush(self, records: List[Dict[str, Any]]) -> None:
        try:
            with open(self.filepath, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
                for record in records:
                    writer.writerow({k: record.get(k, "") for k in self.FIELDS})
        except OSError as exc:
            logging.getLogger("sniffer").error("Errore CSV %s: %s", self.filepath, exc)


class AsyncJSONWriter(AsyncStructuredWriter):

    def _flush(self, records: List[Dict[str, Any]]) -> None:
        try:
            with open(self.filepath, "a") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logging.getLogger("sniffer").error("Errore JSONL %s: %s", self.filepath, exc)


# --------------------------------------------------------------------------
# Singleton globale
# --------------------------------------------------------------------------

_global_logger: Optional[StructuredLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> StructuredLogger:
    """Restituisce il logger globale (singleton thread-safe)."""
    global _global_logger
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = StructuredLogger()
    return _global_logger
