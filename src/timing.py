"""
================================================================================
NIDS-ML - Sistema di Timing e Performance Logging
================================================================================

Traccia tempistiche di esecuzione per tutti i moduli del progetto.

UTILIZZO:
---------
In ogni script, importare e usare il TimingLogger:

    from src.timing import TimingLogger
    
    timer = TimingLogger("preprocessing")
    
    timer.start("caricamento_csv")
    # ... operazioni ...
    timer.stop("caricamento_csv")
    
    timer.start("pulizia")
    # ... operazioni ...
    timer.stop("pulizia")
    
    timer.save()  # Salva su file

REPORT:
-------
Generare report con:
    python src/timing.py --report

================================================================================
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import time
import argparse
import threading
from contextlib import contextmanager

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_project_root


# ==============================================================================
# TIMING LOGGER
# ==============================================================================

class TimingLogger:
    """
    Logger per tracciare tempistiche di esecuzione.
    
    Salva i dati in formato JSON per analisi successiva.
    """
    
    def __init__(self, 
                 module_name: str,
                 parameters: Dict[str, Any] = None,
                 log_dir: Path = None):
        """
        Inizializza timing logger.
        
        Args:
            module_name: Nome modulo (preprocessing, training, etc.)
            parameters: Parametri utilizzati (n_jobs, n_iter, etc.)
            log_dir: Directory per file timing
        """
        self.module_name = module_name
        self.parameters = parameters or {}
        
        if log_dir is None:
            log_dir = get_project_root() / "logs" / "timing"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp sessione
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_timestamp = datetime.now().isoformat()
        
        # Storage tempi
        self.timings: Dict[str, Dict[str, Any]] = {}
        self.active_timers: Dict[str, float] = {}
        
        # Lock per thread safety
        self.lock = threading.Lock()
        
        # Tempo totale
        self.total_start = time.time()
    
    def start(self, operation: str) -> None:
        """
        Avvia timer per un'operazione.
        
        Args:
            operation: Nome operazione (es. "caricamento_csv", "training")
        """
        with self.lock:
            self.active_timers[operation] = time.time()
    
    def stop(self, operation: str, extra_info: Dict[str, Any] = None) -> float:
        """
        Ferma timer e registra durata.
        
        Args:
            operation: Nome operazione
            extra_info: Informazioni aggiuntive da salvare
        
        Returns:
            Durata in secondi
        """
        end_time = time.time()
        
        with self.lock:
            if operation not in self.active_timers:
                raise ValueError(f"Timer '{operation}' non avviato")
            
            start_time = self.active_timers.pop(operation)
            duration = end_time - start_time
            
            self.timings[operation] = {
                'duration_seconds': duration,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'extra_info': extra_info or {}
            }
            
            return duration
    
    @contextmanager
    def time_operation(self, operation: str, extra_info: Dict[str, Any] = None):
        """
        Context manager per timing automatico.
        
        Usage:
            with timer.time_operation("training"):
                # ... codice ...
        """
        self.start(operation)
        try:
            yield
        finally:
            self.stop(operation, extra_info)
    
    def add_metric(self, name: str, value: Any) -> None:
        """Aggiunge metrica custom."""
        with self.lock:
            if '_metrics' not in self.timings:
                self.timings['_metrics'] = {}
            self.timings['_metrics'][name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Restituisce riepilogo completo."""
        total_time = time.time() - self.total_start
        
        return {
            'module': self.module_name,
            'session_id': self.session_id,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': datetime.now().isoformat(),
            'total_duration_seconds': total_time,
            'parameters': self.parameters,
            'operations': self.timings,
            'system_info': {
                'cpu_count': os.cpu_count(),
                'python_version': sys.version.split()[0]
            }
        }
    
    def save(self) -> Path:
        """
        Salva timing su file JSON.
        
        Returns:
            Path al file salvato
        """
        filename = f"{self.module_name}_{self.session_id}.json"
        filepath = self.log_dir / filename
        
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filepath
    
    def print_summary(self) -> None:
        """Stampa riepilogo a video."""
        summary = self.get_summary()
        
        print("\n" + "-" * 50)
        print(f"TIMING SUMMARY - {self.module_name}")
        print("-" * 50)
        print(f"Session: {self.session_id}")
        print(f"Total time: {summary['total_duration_seconds']:.2f}s")
        print("\nOperations:")
        
        for op_name, op_data in self.timings.items():
            if op_name.startswith('_'):
                continue
            duration = op_data['duration_seconds']
            pct = (duration / summary['total_duration_seconds']) * 100
            print(f"  {op_name:30}: {duration:8.2f}s ({pct:5.1f}%)")
        
        if '_metrics' in self.timings:
            print("\nMetrics:")
            for name, value in self.timings['_metrics'].items():
                print(f"  {name}: {value}")
        
        print("-" * 50)


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================

def load_all_timing_logs(log_dir: Path = None) -> List[Dict[str, Any]]:
    """Carica tutti i file di timing."""
    if log_dir is None:
        log_dir = get_project_root() / "logs" / "timing"
    
    if not log_dir.exists():
        return []
    
    logs = []
    for filepath in sorted(log_dir.glob("*.json")):
        with open(filepath, 'r') as f:
            logs.append(json.load(f))
    
    return logs


def generate_timing_report(output_dir: Path = None) -> Path:
    """
    Genera report statistico delle esecuzioni.
    
    Returns:
        Path al report generato
    """
    if output_dir is None:
        output_dir = get_project_root() / "reports" / "timing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs = load_all_timing_logs()
    
    if not logs:
        print("Nessun log di timing trovato")
        return None
    
    # Raggruppa per modulo
    by_module = {}
    for log in logs:
        module = log['module']
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(log)
    
    # Genera report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("REPORT TEMPISTICHE ESECUZIONE")
    report_lines.append("=" * 70)
    report_lines.append(f"\nData generazione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Esecuzioni totali: {len(logs)}")
    report_lines.append(f"Moduli: {list(by_module.keys())}")
    
    # Statistiche per modulo
    for module, module_logs in by_module.items():
        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"MODULO: {module.upper()}")
        report_lines.append(f"{'='*70}")
        report_lines.append(f"Esecuzioni: {len(module_logs)}")
        
        # Tempi totali
        total_times = [log['total_duration_seconds'] for log in module_logs]
        report_lines.append(f"\nTempo totale:")
        report_lines.append(f"  Min:    {min(total_times):.2f}s")
        report_lines.append(f"  Max:    {max(total_times):.2f}s")
        report_lines.append(f"  Media:  {sum(total_times)/len(total_times):.2f}s")
        
        # Operazioni comuni
        all_operations = set()
        for log in module_logs:
            all_operations.update(log.get('operations', {}).keys())
        
        if all_operations:
            report_lines.append(f"\nOperazioni:")
            
            for op in sorted(all_operations):
                if op.startswith('_'):
                    continue
                
                op_times = []
                for log in module_logs:
                    if op in log.get('operations', {}):
                        op_times.append(log['operations'][op]['duration_seconds'])
                
                if op_times:
                    report_lines.append(f"\n  {op}:")
                    report_lines.append(f"    Esecuzioni: {len(op_times)}")
                    report_lines.append(f"    Min:        {min(op_times):.2f}s")
                    report_lines.append(f"    Max:        {max(op_times):.2f}s")
                    report_lines.append(f"    Media:      {sum(op_times)/len(op_times):.2f}s")
        
        # Parametri utilizzati
        report_lines.append(f"\nParametri utilizzati:")
        all_params = {}
        for log in module_logs:
            for k, v in log.get('parameters', {}).items():
                if k not in all_params:
                    all_params[k] = []
                all_params[k].append(v)
        
        for param, values in all_params.items():
            unique_values = list(set(str(v) for v in values))
            report_lines.append(f"  {param}: {unique_values}")
    
    # Esecuzioni recenti
    report_lines.append(f"\n{'='*70}")
    report_lines.append("ESECUZIONI RECENTI")
    report_lines.append(f"{'='*70}")
    
    recent = sorted(logs, key=lambda x: x['start_timestamp'], reverse=True)[:10]
    for log in recent:
        report_lines.append(f"\n{log['module']} - {log['session_id']}")
        report_lines.append(f"  Inizio: {log['start_timestamp']}")
        report_lines.append(f"  Durata: {log['total_duration_seconds']:.2f}s")
        if log.get('parameters'):
            params_str = ", ".join(f"{k}={v}" for k, v in list(log['parameters'].items())[:5])
            report_lines.append(f"  Params: {params_str}")
    
    # Salva report
    report_text = "\n".join(report_lines)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"timing_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # Salva anche JSON per analisi programmatica
    json_path = output_dir / f"timing_report_{timestamp}.json"
    
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'total_executions': len(logs),
        'by_module': {}
    }
    
    for module, module_logs in by_module.items():
        total_times = [log['total_duration_seconds'] for log in module_logs]
        report_data['by_module'][module] = {
            'executions': len(module_logs),
            'total_time_min': min(total_times),
            'total_time_max': max(total_times),
            'total_time_avg': sum(total_times) / len(total_times)
        }
    
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Report salvato: {report_path}")
    print(f"JSON salvato: {json_path}")
    
    # Stampa anche a video
    print("\n" + report_text)
    
    return report_path


# ==============================================================================
# DECORATORE PER TIMING AUTOMATICO
# ==============================================================================

def timed(operation_name: str = None):
    """
    Decoratore per timing automatico di funzioni.
    
    Usage:
        @timed("mia_operazione")
        def mia_funzione():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                print(f"[TIMING] {op_name}: {duration:.2f}s")
        
        return wrapper
    return decorator


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gestione timing NIDS-ML')
    parser.add_argument('--report', action='store_true', help='Genera report tempistiche')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory output report')
    
    args = parser.parse_args()
    
    if args.report:
        generate_timing_report(args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()