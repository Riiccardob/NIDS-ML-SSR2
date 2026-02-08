"""
Utilities condivise per la pipeline NIDS.

Include funzioni per logging, metrics, e memory management.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger con output console e opzionale file."""
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler opzionale
    if log_file:
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_memory_usage() -> float:
    """
    Ritorna percentuale di RAM utilizzata.
    
    Returns:
        Percentuale (0-100) di RAM in uso
    """
    return psutil.virtual_memory().percent


def get_available_memory_gb() -> float:
    """
    Ritorna GB di RAM disponibile.
    
    Returns:
        GB di RAM disponibile
    """
    return psutil.virtual_memory().available / (1024**3)


def check_memory_availability(required_mb: float, safety_margin: float = 0.2) -> bool:
    """
    Verifica se c'è abbastanza memoria disponibile.
    
    Args:
        required_mb: MB richiesti
        safety_margin: Margine di sicurezza (default 20%)
    
    Returns:
        True se c'è abbastanza memoria
    """
    available_mb = get_available_memory_gb() * 1024
    required_with_margin = required_mb * (1 + safety_margin)
    
    return available_mb >= required_with_margin


def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: list,
    name: str = "DataFrame"
) -> None:
    """Valida presenza colonne richieste."""
    
    missing = set(required_columns) - set(df.columns)
    
    if missing:
        raise ValueError(
            f"{name} manca le seguenti colonne: {sorted(missing)}\n"
            f"Colonne presenti: {sorted(df.columns.tolist())}"
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcola metriche di classificazione binaria.
    
    Args:
        y_true: Label vere
        y_pred: Label predette
    
    Returns:
        Dict con metriche dettagliate
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    
    return metrics


def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    """
    Stampa metriche in formato leggibile.
    
    Args:
        metrics: Dict con metriche
        title: Titolo da visualizzare
    """
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print(f"FNR:       {metrics['fnr']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:>8,}  FN: {metrics['fn']:>8,}")
    print(f"  FP: {metrics['fp']:>8,}  TN: {metrics['tn']:>8,}")
    print(f"{'='*60}\n")


def save_dataset_info(df: pd.DataFrame, filepath: Path) -> None:
    """
    Salva informazioni sul dataset in JSON.
    
    Args:
        df: DataFrame da analizzare
        filepath: Path del file JSON di output
    """
    
    import json
    
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    if 'Label' in df.columns:
        info['label_distribution'] = df['Label'].value_counts().to_dict()
    
    if 'Label_Binary' in df.columns:
        info['label_binary_distribution'] = df['Label_Binary'].value_counts().to_dict()
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)


def format_bytes(bytes_value: int) -> str:
    """
    Formatta bytes in formato human-readable.
    
    Args:
        bytes_value: Valore in bytes
    
    Returns:
        Stringa formattata (es. "1.5 GB")
    """
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"


def estimate_dataframe_memory(n_rows: int, n_features: int, dtype_bytes: int = 8) -> float:
    """
    Stima memoria richiesta per un DataFrame.
    
    Args:
        n_rows: Numero di righe
        n_features: Numero di feature
        dtype_bytes: Byte per valore (default 8 per float64)
    
    Returns:
        MB stimati
    """
    
    return (n_rows * n_features * dtype_bytes) / (1024**2)


def log_system_info(logger: logging.Logger) -> None:
    """
    Logga informazioni di sistema.
    
    Args:
        logger: Logger da usare
    """
    
    import platform
    
    logger.info("="*70)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*70)
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    
    # RAM info
    mem = psutil.virtual_memory()
    logger.info(f"\nRAM:")
    logger.info(f"  Total: {mem.total / (1024**3):.1f} GB")
    logger.info(f"  Available: {mem.available / (1024**3):.1f} GB")
    logger.info(f"  Used: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
    
    # CPU info
    logger.info(f"\nCPU:")
    logger.info(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"  Logical cores: {psutil.cpu_count(logical=True)}")
    
    logger.info("="*70)


def get_worker_count(cpu_usage_percent: float = 0.75, min_workers: int = 2, max_workers: int = 16) -> int:
    """
    Calcola numero ottimale di worker per parallel processing.
    
    Args:
        cpu_usage_percent: Percentuale di CPU da usare (0.0-1.0)
        min_workers: Minimo numero di worker
        max_workers: Massimo numero di worker
    
    Returns:
        Numero di worker da utilizzare
    """
    
    total_cores = psutil.cpu_count(logical=True)
    
    if total_cores is None:
        return min_workers
    
    # Calcola worker basato su percentuale
    workers = int(total_cores * cpu_usage_percent)
    
    # Applica limiti
    workers = max(min_workers, workers)
    workers = min(max_workers, workers)
    
    return workers
