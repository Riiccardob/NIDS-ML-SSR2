"""
================================================================================
NIDS-ML - Comparazione Modelli con Scorecard Approach
================================================================================

Confronta modelli usando approccio "Scorecard con Hard Constraints" ottimizzato
per deployment NIDS in produzione.

LOGICA DI SELEZIONE:
--------------------
1. HARD CONSTRAINTS (eliminatori):
   - FPR (False Positive Rate) <= soglia (default: 1%)
   - Latenza media predizione <= soglia (default: 1ms)
   
2. SOFT RANKING (tra i modelli che passano):
   - Priorita 1: Recall (detection rate) - vogliamo catturare tutti gli attacchi
   - Priorita 2: F1 Score - bilanciamento precision/recall
   - Priorita 3: Latenza - piu veloce e meglio

MOTIVAZIONE:
------------
- FPR basso e CRITICO: 1% FPR su 1M flussi/giorno = 10.000 falsi allarmi
- Recall alto e CRITICO: 1% FNR significa che l'1% degli attacchi passa
- Latenza e CRITICA: Se predict() impiega 10ms e arrivano 1000 pkt/s, il sistema crolla

GUIDA PARAMETRI:
----------------
    python src/compare_models.py [opzioni]

Opzioni:
    --task STR              'binary' o 'multiclass' (default: binary)
    --max-fpr FLOAT         Soglia massima FPR (default: 0.01 = 1%)
    --max-latency-ms FLOAT  Soglia massima latenza ms (default: 1.0)
    --output-dir PATH       Directory output

ESEMPI:
-------
# Standard (FPR <= 1%, Latency <= 1ms)
python src/compare_models.py

# Piu restrittivo (FPR <= 0.5%)
python src/compare_models.py --max-fpr 0.005

# Meno restrittivo su latenza (per modelli complessi)
python src/compare_models.py --max-latency-ms 5.0

================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import shutil
from datetime import datetime
import time

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import get_logger, get_project_root, suppress_warnings
from src.feature_engineering import load_artifacts
from src.timing import TimingLogger

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE DEFAULT
# ==============================================================================

MODEL_NAMES = ['random_forest', 'xgboost', 'lightgbm']

# Hard constraints per produzione
DEFAULT_MAX_FPR = 0.01        # 1% massimo
DEFAULT_MAX_LATENCY_MS = 1.0  # 1ms massimo per predizione


# ==============================================================================
# CARICAMENTO RISULTATI
# ==============================================================================

def load_evaluation_results(model_name: str, task: str) -> Optional[Dict[str, Any]]:
    """Carica risultati evaluation per un modello."""
    reports_dir = get_project_root() / "reports" / model_name
    report_path = reports_dir / f"report_{model_name}_{task}.json"
    
    if not report_path.exists():
        logger.warning(f"Report non trovato: {report_path}")
        return None
    
    with open(report_path, 'r') as f:
        return json.load(f)


def load_model(model_name: str, task: str):
    """Carica modello salvato."""
    model_path = get_project_root() / "models" / model_name / f"model_{task}.pkl"
    
    if not model_path.exists():
        return None
    
    return joblib.load(model_path)


# ==============================================================================
# BENCHMARK LATENZA
# ==============================================================================

def benchmark_model_latency(model, n_samples: int = 1000, n_iterations: int = 5) -> Dict[str, float]:
    """
    Misura latenza di predizione del modello.
    
    Args:
        model: Modello sklearn/xgboost/lightgbm
        n_samples: Campioni per batch
        n_iterations: Ripetizioni per media stabile
    
    Returns:
        Dict con latency_mean_ms, latency_std_ms, latency_per_sample_ms
    """
    # Carica artifacts per ottenere numero feature
    _, selected_features, _ = load_artifacts()
    n_features = len(selected_features)
    
    # Genera dati sintetici (non importa il contenuto, solo la dimensione)
    X_dummy = np.random.randn(n_samples, n_features)
    
    # Warmup (JIT compilation per XGBoost/LightGBM)
    _ = model.predict(X_dummy[:10])
    
    # Benchmark
    latencies = []
    
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_dummy)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Converti in ms
    
    total_latency = np.mean(latencies)
    
    return {
        'latency_total_ms': total_latency,
        'latency_per_sample_ms': total_latency / n_samples,
        'latency_std_ms': np.std(latencies),
        'samples_per_second': n_samples / (total_latency / 1000)
    }


# ==============================================================================
# SCORECARD EVALUATION
# ==============================================================================

def evaluate_model_scorecard(model_name: str, 
                             task: str,
                             max_fpr: float,
                             max_latency_ms: float) -> Dict[str, Any]:
    """
    Valuta modello con approccio scorecard.
    
    Returns:
        Dict con metriche, latenza, e status (PASS/FAIL per ogni constraint)
    """
    result = {
        'model_name': model_name,
        'task': task,
        'metrics': {},
        'latency': {},
        'constraints': {
            'fpr_threshold': max_fpr,
            'latency_threshold_ms': max_latency_ms,
            'fpr_pass': False,
            'latency_pass': False,
            'all_pass': False
        },
        'score': 0.0,
        'status': 'FAIL'
    }
    
    # Carica risultati evaluation
    eval_results = load_evaluation_results(model_name, task)
    if eval_results is None:
        result['error'] = "Evaluation results not found"
        return result
    
    # Estrai metriche
    metrics = eval_results.get('metrics', {})
    result['metrics'] = metrics
    
    # Carica modello per benchmark latenza
    model = load_model(model_name, task)
    if model is None:
        result['error'] = "Model file not found"
        return result
    
    # Benchmark latenza
    print(f"   Benchmarking {model_name}...")
    latency_results = benchmark_model_latency(model)
    result['latency'] = latency_results
    
    # Verifica hard constraints
    fpr = metrics.get('false_positive_rate', 1.0)
    latency_per_sample = latency_results['latency_per_sample_ms']
    
    result['constraints']['fpr_pass'] = fpr <= max_fpr
    result['constraints']['latency_pass'] = latency_per_sample <= max_latency_ms
    result['constraints']['all_pass'] = (
        result['constraints']['fpr_pass'] and 
        result['constraints']['latency_pass']
    )
    
    # Calcola score (solo se passa i constraints)
    if result['constraints']['all_pass']:
        result['status'] = 'PASS'
        
        # Score composito: priorita a Recall, poi F1, poi velocita
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        # Normalizza latenza (piu bassa = meglio, max 1.0)
        latency_score = max(0, 1 - (latency_per_sample / max_latency_ms))
        
        # Score pesato: 50% Recall + 30% F1 + 20% Velocita
        result['score'] = (0.5 * recall) + (0.3 * f1) + (0.2 * latency_score)
    
    return result


def compare_models_scorecard(task: str,
                             max_fpr: float,
                             max_latency_ms: float) -> Tuple[List[Dict], Optional[str]]:
    """
    Confronta tutti i modelli con approccio scorecard.
    
    Returns:
        Tuple (lista risultati, nome modello migliore o None)
    """
    results = []
    
    for model_name in MODEL_NAMES:
        print(f"\n   Valutazione {model_name}...")
        result = evaluate_model_scorecard(model_name, task, max_fpr, max_latency_ms)
        results.append(result)
    
    # Filtra modelli che passano
    passing = [r for r in results if r['status'] == 'PASS']
    
    if not passing:
        return results, None
    
    # Ordina per score (decrescente)
    passing.sort(key=lambda x: x['score'], reverse=True)
    
    return results, passing[0]['model_name']


# ==============================================================================
# VISUALIZZAZIONI
# ==============================================================================

def plot_scorecard_comparison(results: List[Dict], 
                              best_model: Optional[str],
                              output_dir: Path) -> None:
    """Genera grafico comparativo scorecard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = [r['model_name'] for r in results]
    colors = ['green' if r['status'] == 'PASS' else 'red' for r in results]
    
    # 1. FPR con soglia
    ax1 = axes[0, 0]
    fprs = [r['metrics'].get('false_positive_rate', 0) * 100 for r in results]
    threshold = results[0]['constraints']['fpr_threshold'] * 100
    
    bars = ax1.bar(models, fprs, color=colors, alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Soglia: {threshold}%')
    ax1.set_ylabel('False Positive Rate (%)')
    ax1.set_title('FPR (piu basso = meglio)')
    ax1.legend()
    
    for bar, fpr in zip(bars, fprs):
        ax1.annotate(f'{fpr:.3f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # 2. Latenza con soglia
    ax2 = axes[0, 1]
    latencies = [r['latency'].get('latency_per_sample_ms', 0) for r in results]
    threshold_lat = results[0]['constraints']['latency_threshold_ms']
    
    bars = ax2.bar(models, latencies, color=colors, alpha=0.7)
    ax2.axhline(y=threshold_lat, color='red', linestyle='--', label=f'Soglia: {threshold_lat}ms')
    ax2.set_ylabel('Latenza per sample (ms)')
    ax2.set_title('Latenza (piu bassa = meglio)')
    ax2.legend()
    
    for bar, lat in zip(bars, latencies):
        ax2.annotate(f'{lat:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # 3. Recall
    ax3 = axes[1, 0]
    recalls = [r['metrics'].get('recall', 0) for r in results]
    
    bars = ax3.bar(models, recalls, color=colors, alpha=0.7)
    ax3.set_ylabel('Recall (Detection Rate)')
    ax3.set_title('Recall (piu alto = meglio)')
    ax3.set_ylim(0.95, 1.0)
    
    for bar, rec in zip(bars, recalls):
        ax3.annotate(f'{rec:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # 4. Score finale
    ax4 = axes[1, 1]
    scores = [r['score'] for r in results]
    
    bars = ax4.bar(models, scores, color=colors, alpha=0.7)
    ax4.set_ylabel('Score Composito')
    ax4.set_title('Score Finale (solo modelli PASS)')
    
    for bar, score, r in zip(bars, scores, results):
        label = f'{score:.3f}' if r['status'] == 'PASS' else 'FAIL'
        ax4.annotate(label, xy=(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.05)),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Titolo generale
    title = f"Scorecard Comparison"
    if best_model:
        title += f" - Best: {best_model.upper()}"
    else:
        title += " - NESSUN MODELLO PASSA I CONSTRAINTS"
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "scorecard_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Grafico salvato: {output_path}")


# ==============================================================================
# COPIA BEST MODEL
# ==============================================================================

def copy_best_model(best_model_name: str, 
                    task: str,
                    results: List[Dict],
                    output_dir: Path) -> None:
    """Copia modello migliore e genera report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    project_root = get_project_root()
    
    # Trova risultati del best model
    best_result = next((r for r in results if r['model_name'] == best_model_name), None)
    
    # Copia modello
    src_model = project_root / "models" / best_model_name / f"model_{task}.pkl"
    dst_model = output_dir / f"model_{task}.pkl"
    if src_model.exists():
        shutil.copy2(src_model, dst_model)
        logger.info(f"Modello copiato: {dst_model}")
    
    # Copia risultati training
    src_results = project_root / "models" / best_model_name / f"results_{task}.json"
    if src_results.exists():
        shutil.copy2(src_results, output_dir / f"training_results.json")
    
    # Copia grafici evaluation
    src_report_dir = project_root / "reports" / best_model_name
    if src_report_dir.exists():
        for f in src_report_dir.glob(f"*_{task}.*"):
            shutil.copy2(f, output_dir / f.name)
    
    # Salva metadata
    metadata = {
        'best_model': best_model_name,
        'task': task,
        'timestamp': datetime.now().isoformat(),
        'selection_method': 'scorecard',
        'constraints': best_result['constraints'] if best_result else {},
        'metrics': best_result['metrics'] if best_result else {},
        'latency': best_result['latency'] if best_result else {},
        'score': best_result['score'] if best_result else 0
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salva tutti i risultati comparazione
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


def generate_comparison_report(results: List[Dict],
                               best_model: Optional[str],
                               output_dir: Path) -> None:
    """Genera report testuale dettagliato."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SCORECARD COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metodo: Scorecard con Hard Constraints\n\n")
        
        # Constraints
        if results:
            c = results[0]['constraints']
            f.write("HARD CONSTRAINTS:\n")
            f.write(f"  - FPR massimo:     {c['fpr_threshold']*100:.2f}%\n")
            f.write(f"  - Latenza massima: {c['latency_threshold_ms']:.2f}ms per sample\n\n")
        
        # Risultati per modello
        f.write("=" * 70 + "\n")
        f.write("RISULTATI PER MODELLO\n")
        f.write("=" * 70 + "\n")
        
        for r in results:
            is_best = " [BEST]" if r['model_name'] == best_model else ""
            status_icon = "PASS" if r['status'] == 'PASS' else "FAIL"
            
            f.write(f"\n{r['model_name'].upper()}{is_best} - {status_icon}\n")
            f.write("-" * 50 + "\n")
            
            # Constraints check
            f.write("  Constraints:\n")
            fpr = r['metrics'].get('false_positive_rate', 0)
            lat = r['latency'].get('latency_per_sample_ms', 0)
            
            fpr_status = "PASS" if r['constraints']['fpr_pass'] else "FAIL"
            lat_status = "PASS" if r['constraints']['latency_pass'] else "FAIL"
            
            f.write(f"    FPR:     {fpr*100:.4f}% [{fpr_status}]\n")
            f.write(f"    Latency: {lat:.4f}ms [{lat_status}]\n")
            
            # Metriche
            f.write("  Metriche:\n")
            for k, v in r['metrics'].items():
                if isinstance(v, float):
                    f.write(f"    {k:25}: {v:.4f}\n")
            
            # Latenza dettagliata
            f.write("  Latenza dettagliata:\n")
            for k, v in r['latency'].items():
                if isinstance(v, float):
                    f.write(f"    {k:25}: {v:.4f}\n")
            
            # Score
            f.write(f"  Score finale: {r['score']:.4f}\n")
        
        # Verdetto finale
        f.write("\n" + "=" * 70 + "\n")
        f.write("VERDETTO FINALE\n")
        f.write("=" * 70 + "\n\n")
        
        passing = [r for r in results if r['status'] == 'PASS']
        
        if best_model:
            f.write(f"Modello selezionato: {best_model.upper()}\n")
            f.write(f"Modelli che passano: {len(passing)}/{len(results)}\n")
        else:
            f.write("ATTENZIONE: Nessun modello soddisfa i constraints!\n")
            f.write("Considerare:\n")
            f.write("  1. Rilassare i constraints (--max-fpr, --max-latency-ms)\n")
            f.write("  2. Riaddestrare i modelli con focus su FPR\n")
            f.write("  3. Ottimizzare i modelli per latenza\n")
    
    logger.info(f"Report salvato: {report_path}")


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Confronta modelli NIDS con approccio Scorecard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python src/compare_models.py
  python src/compare_models.py --max-fpr 0.005
  python src/compare_models.py --max-latency-ms 0.5
        """
    )
    
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                        default='binary')
    parser.add_argument('--max-fpr', type=float, default=DEFAULT_MAX_FPR,
                        help=f'FPR massimo (default: {DEFAULT_MAX_FPR})')
    parser.add_argument('--max-latency-ms', type=float, default=DEFAULT_MAX_LATENCY_MS,
                        help=f'Latenza massima in ms (default: {DEFAULT_MAX_LATENCY_MS})')
    parser.add_argument('--output-dir', type=Path, default=None)
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    # Timing
    timer = TimingLogger("compare_models", parameters={
        'task': args.task,
        'max_fpr': args.max_fpr,
        'max_latency_ms': args.max_latency_ms
    })
    
    print("\n" + "=" * 60)
    print("SCORECARD MODEL COMPARISON")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"Hard Constraints:")
    print(f"  - FPR massimo:     {args.max_fpr*100:.2f}%")
    print(f"  - Latenza massima: {args.max_latency_ms:.2f}ms")
    print()
    
    output_dir = args.output_dir or get_project_root() / "models" / "best_model"
    
    try:
        print("1. Valutazione modelli con scorecard...")
        with timer.time_operation("scorecard_evaluation"):
            results, best_model = compare_models_scorecard(
                args.task, args.max_fpr, args.max_latency_ms
            )
        
        # Mostra tabella risultati
        print("\n" + "-" * 70)
        print(f"{'Model':<15} {'FPR':<10} {'Latency':<12} {'Recall':<10} {'Score':<10} {'Status'}")
        print("-" * 70)
        
        for r in results:
            fpr = r['metrics'].get('false_positive_rate', 0) * 100
            lat = r['latency'].get('latency_per_sample_ms', 0)
            recall = r['metrics'].get('recall', 0)
            score = r['score']
            status = r['status']
            
            print(f"{r['model_name']:<15} {fpr:<10.4f} {lat:<12.4f} {recall:<10.4f} {score:<10.4f} {status}")
        
        print("-" * 70)
        
        if best_model:
            print(f"\nModello migliore: {best_model.upper()}")
        else:
            print("\nATTENZIONE: Nessun modello passa i constraints!")
        
        print("\n2. Generazione grafici...")
        with timer.time_operation("generate_plots"):
            plot_scorecard_comparison(results, best_model, output_dir)
        
        print("\n3. Generazione report...")
        with timer.time_operation("generate_report"):
            generate_comparison_report(results, best_model, output_dir)
        
        if best_model:
            print("\n4. Copia best model...")
            with timer.time_operation("copy_best_model"):
                copy_best_model(best_model, args.task, results, output_dir)
        
        # Salva timing
        timer.save()
        
        print("\n" + "=" * 60)
        print("CONFRONTO COMPLETATO")
        print("=" * 60)
        
        if best_model:
            print(f"\nBest model: {best_model.upper()}")
            print(f"Output: {output_dir}")
        else:
            print("\nNessun modello selezionato. Rilassare i constraints.")
        
        timer.print_summary()
        
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()