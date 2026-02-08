"""
Logger strutturato per Live NIDS Sniffer.

Gestisce logging in formato CSV e JSON con rotazione automatica.
"""

import logging
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler

from config import (
    LOGS_DIR, LOG_LEVEL, LOG_MAX_SIZE_MB, LOG_BACKUP_COUNT,
    LOG_PREFIX, LOG_FORMAT_TYPE, LogFormat
)


class StructuredLogger:
    """Logger strutturato per eventi di sicurezza."""
    
    def __init__(self, name: str = "sniffer"):
        self.name = name
        self.logger = self._setup_logger()
        
        # File handlers per CSV e JSON
        self.csv_handler: Optional[CSVLogHandler] = None
        self.json_handler: Optional[JSONLogHandler] = None
        
        self._setup_structured_handlers()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger standard."""
        
        logger = logging.getLogger(self.name)
        logger.setLevel(LOG_LEVEL)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(LOG_LEVEL)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console.setFormatter(formatter)
        
        logger.addHandler(console)
        
        # File handler (plain text)
        log_file = LOGS_DIR / f"{LOG_PREFIX}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_structured_handlers(self) -> None:
        """Setup handlers per CSV e JSON."""
        
        if LOG_FORMAT_TYPE in (LogFormat.CSV, LogFormat.BOTH):
            self.csv_handler = CSVLogHandler(LOGS_DIR / f"{LOG_PREFIX}_alerts.csv")
        
        if LOG_FORMAT_TYPE in (LogFormat.JSON, LogFormat.BOTH):
            self.json_handler = JSONLogHandler(LOGS_DIR / f"{LOG_PREFIX}_alerts.json")
    
    def info(self, message: str) -> None:
        """Log messaggio informativo."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log errore."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critico."""
        self.logger.critical(message)
    
    def debug(self, message: str) -> None:
        """Log debug."""
        self.logger.debug(message)
    
    def log_alert(self, alert_data: Dict[str, Any]) -> None:
        """
        Log alert strutturato.
        
        Args:
            alert_data: Dizionario con dati dell'alert
                       Deve contenere almeno: timestamp, src_ip, dst_ip, 
                       prediction, confidence, action
        """
        
        # Aggiungi timestamp se mancante
        if 'timestamp' not in alert_data:
            alert_data['timestamp'] = datetime.now().isoformat()
        
        # Log su console
        self.logger.warning(
            f"ALERT: {alert_data['src_ip']}:{alert_data.get('src_port', '?')} -> "
            f"{alert_data['dst_ip']}:{alert_data.get('dst_port', '?')} | "
            f"Prediction: {alert_data['prediction']} | "
            f"Confidence: {alert_data['confidence']:.4f} | "
            f"Action: {alert_data['action']}"
        )
        
        # Log strutturato
        if self.csv_handler:
            self.csv_handler.write(alert_data)
        
        if self.json_handler:
            self.json_handler.write(alert_data)
    
    def log_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Log statistiche operative.
        
        Args:
            stats: Dizionario con statistiche (flows, predictions, alerts, etc.)
        """
        
        stats['timestamp'] = datetime.now().isoformat()
        
        self.logger.info(
            f"STATS: Flows={stats.get('total_flows', 0)} | "
            f"Predictions={stats.get('total_predictions', 0)} | "
            f"Alerts={stats.get('total_alerts', 0)} | "
            f"Blocks={stats.get('total_blocks', 0)} | "
            f"FPS={stats.get('flows_per_second', 0):.2f}"
        )
        
        # Salva statistiche in JSON
        stats_file = LOGS_DIR / f"{LOG_PREFIX}_stats.json"
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')


class CSVLogHandler:
    """Handler per logging CSV strutturato."""
    
    # Campi standard per CSV
    FIELDS = [
        'timestamp',
        'src_ip',
        'src_port',
        'dst_ip',
        'dst_port',
        'protocol',
        'l7_proto',
        'prediction',
        'confidence',
        'action',
        'duration_ms',
        'bytes_in',
        'bytes_out',
        'packets_in',
        'packets_out',
    ]
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._init_file()
    
    def _init_file(self) -> None:
        """Inizializza file CSV con header."""
        
        # Se file non esiste, scrivi header
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()
    
    def write(self, data: Dict[str, Any]) -> None:
        """Scrivi record CSV."""
        
        # Filtra solo campi richiesti
        filtered = {k: data.get(k, '') for k in self.FIELDS}
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(filtered)


class JSONLogHandler:
    """Handler per logging JSON strutturato (JSONL format)."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
    
    def write(self, data: Dict[str, Any]) -> None:
        """Scrivi record JSON (una riga per record)."""
        
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(data) + '\n')


# Singleton logger globale
_global_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Ottieni logger globale (singleton)."""
    
    global _global_logger
    
    if _global_logger is None:
        _global_logger = StructuredLogger()
    
    return _global_logger
