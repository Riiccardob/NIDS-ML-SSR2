"""
================================================================================
NIDS-ML - Comparazione Modelli con Versionamento
================================================================================

Confronta TUTTE le versioni di modelli (es. xgboost/cv3_iter20, xgboost/cv5_iter100)
usando approccio Scorecard con Hard Constraints.

NUOVE FUNZIONALITA:
-------------------
1. Confronta tutte le versioni, non solo i 3 tipi base
2. Ranking intra-algoritmo (es. tutte le versioni XGBoost ordinate)
3. Grafico plateau per vedere quando ulteriore training non migliora
4. Selezione automatica best_model tra tutte le versioni

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
from src.model_versioning import list_model_versions, generate_version_id

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

DEFAULT_MAX_FPR = 0.01
DEFAULT_MAX_LATENCY_MS = 1.0


# ==============================================================================
# CARICAMENTO RISULTATI
# ==============================================================================

def load_all_model_versions(task: str = 'binary') -> List[Dict]:
    """
    Carica tutte le versioni di tutti i modelli.
    
    Returns:
        Lista di dict con info complete per ogni versione
    """
    versions = list_model_versions(task=task)
    
    # Carica anche modelli nella root (backward compatibility)
    models_dir = get_project_root() / "models"
    
    for model_type in ['random_forest', 'xgboost', 'lightgbm']:
        root_model = models_dir / model_type / f"model_{task}.pkl"
        root_results = models_dir / model_type / f"results_{task}.json"
        
        if root_model.exists() and root_results.exists():
            # Verifica che non sia già contato come versione
            already_listed = any(
                v['path'] == models_dir / model_type 
                for v in versions
            )
            
            if not already_listed:
                try:
                    with open(root_results) as f:
                        results = json.load(f)
                    
                    versions.append({
                        'model_type': model_type,
                        'version_id': 'default',
                        'path': models_dir / model_type,
                        'model_path': root_model,
                        'results': results,
                        'validation_metrics': results.get('validation_metrics', {}),
                        'train_time': results.get('train_time_seconds', 0),
                        'n_iter': results.get('best_params', {}).get('n_iter', 0),
                        'cv': results.get('cv', 0)
                    })
                except Exception as e:
                    logger.warning(f"Errore caricamento {root_results}: {e}")
    
    return versions


# ==============================================================================
# BENCHMARK LATENZA
# ==============================================================================

def benchmark_model_latency(model, n_samples: int = 1000, n_iterations: int = 10) -> Dict[str, float]:
    """Misura latenza di predizione del modello."""
    _, selected_features, _, _ = load_artifacts()
    n_features = len(selected_features)
    
    np.random.seed(42)
    X_dummy = np.random.randn(n_samples, n_features)
    
    for _ in range(3):
        _ = model.predict(X_dummy[:100])
    
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_dummy)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies_sorted = sorted(latencies)
    if len(latencies_sorted) > 4:
        latencies_trimmed = latencies_sorted[1:-1]
    else:
        latencies_trimmed = latencies_sorted
    
    total_latency = np.mean(latencies_trimmed)
    
    return {
        'latency_total_ms': total_latency,
        'latency_per_sample_ms': total_latency / n_samples,
        'latency_std_ms': np.std(latencies_trimmed),
        'samples_per_second': n_samples / (total_latency / 1000)
    }


# ==============================================================================
# SCORECARD EVALUATION
# ==============================================================================

def evaluate_version_scorecard(version: Dict, 
                               task: str,
                               max_fpr: float,
                               max_latency_ms: float) -> Dict[str, Any]:
    """Valuta una singola versione con approccio scorecard (F2-Score based)."""
    
    full_id = f"{version['model_type']}/{version['version_id']}"
    
    result = {
        'full_id': full_id,
        'model_type': version['model_type'],
        'version_id': version['version_id'],
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
        'status': 'FAIL',
        'training_mode': version.get('training_mode', 'unknown')
    }
    
    metrics = version.get('validation_metrics', {})
    if not metrics:
        result['error'] = "No validation metrics found"
        return result
    
    result['metrics'] = metrics
    
    model_path = version.get('model_path')
    if not model_path or not Path(model_path).exists():
        result['error'] = "Model file not found"
        return result
    
    try:
        model = joblib.load(model_path)
        latency_results = benchmark_model_latency(model)
        result['latency'] = latency_results
    except Exception as e:
        result['error'] = f"Error loading model: {e}"
        return result
    
    fpr = metrics.get('false_positive_rate')
    if fpr is None:
        precision = metrics.get('precision', 0.99)
        if precision >= 0.98:
            fpr = 0.005
        elif precision >= 0.95:
            fpr = 0.01
        else:
            fpr = 1 - precision
        result['fpr_estimated'] = True
    else:
        result['fpr_estimated'] = False
    
    latency_per_sample = latency_results['latency_per_sample_ms']
    
    result['constraints']['fpr_pass'] = fpr <= max_fpr
    result['constraints']['latency_pass'] = latency_per_sample <= max_latency_ms
    result['constraints']['all_pass'] = (
        result['constraints']['fpr_pass'] and 
        result['constraints']['latency_pass']
    )
    
    if result['constraints']['all_pass']:
        result['status'] = 'PASS'
        
        f2 = metrics.get('f2', 0)
        latency_score = max(0, 1 - (latency_per_sample / max_latency_ms))
        
        result['score'] = (0.70 * f2) + (0.30 * latency_score)
    
    return result


def compare_all_versions(task: str, max_fpr: float, max_latency_ms: float) -> Tuple[List[Dict], Optional[str]]:
    """
    Confronta tutte le versioni di tutti i modelli.
    
    Returns:
        Tuple (lista risultati, full_id del migliore)
    """
    versions = load_all_model_versions(task)
    
    if not versions:
        logger.warning("Nessuna versione modello trovata")
        return [], None
    
    results = []
    
    print(f"\nValutazione {len(versions)} versioni...")
    for i, version in enumerate(versions, 1):
        full_id = f"{version['model_type']}/{version['version_id']}"
        print(f"  [{i}/{len(versions)}] {full_id}...")
        
        result = evaluate_version_scorecard(version, task, max_fpr, max_latency_ms)
        results.append(result)
    
    # Filtra chi passa
    passing = [r for r in results if r['status'] == 'PASS']
    
    if not passing:
        return results, None
    
    # Ordina per score
    passing.sort(key=lambda x: x['score'], reverse=True)
    
    return results, passing[0]['full_id']


# ==============================================================================
# RANKING INTRA-ALGORITMO
# ==============================================================================

def compute_algorithm_rankings(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Calcola ranking per ogni tipo di algoritmo.
    
    Returns:
        Dict {model_type: [versioni ordinate per score]}
    """
    rankings = {}
    
    for r in results:
        model_type = r['model_type']
        if model_type not in rankings:
            rankings[model_type] = []
        rankings[model_type].append(r)
    
    # Ordina ogni gruppo per score (F1 se non ha score)
    for model_type in rankings:
        rankings[model_type].sort(
            key=lambda x: (x['score'] if x['status'] == 'PASS' else x['metrics'].get('f1', 0)),
            reverse=True
        )
        
        # Aggiungi rank
        for i, r in enumerate(rankings[model_type], 1):
            r['intra_rank'] = i
            r['total_versions'] = len(rankings[model_type])
    
    return rankings


# ==============================================================================
# VISUALIZZAZIONI
# ==============================================================================

def plot_plateau_analysis(results: List[Dict], output_dir: Path):
    """
    Genera grafico per visualizzare il plateau di training.
    
    Mostra come F1/Recall cambiano al variare di n_iter*cv (effort totale).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Raggruppa per model_type
    by_type = {}
    for r in results:
        mt = r['model_type']
        if mt not in by_type:
            by_type[mt] = []
        
        effort = r.get('n_iter', 0) * r.get('cv', 1)
        f1 = r['metrics'].get('f1', 0)
        recall = r['metrics'].get('recall', 0)
        
        if effort > 0:
            by_type[mt].append({
                'effort': effort,
                'f1': f1,
                'recall': recall,
                'version': r['version_id'],
                'score': r['score']
            })
    
    if not by_type:
        return
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'random_forest': 'green', 'xgboost': 'blue', 'lightgbm': 'orange'}
    markers = {'random_forest': 'o', 'xgboost': 's', 'lightgbm': '^'}
    
    for mt, data in by_type.items():
        if not data:
            continue
        
        data.sort(key=lambda x: x['effort'])
        efforts = [d['effort'] for d in data]
        f1s = [d['f1'] for d in data]
        recalls = [d['recall'] for d in data]
        
        color = colors.get(mt, 'gray')
        marker = markers.get(mt, 'o')
        
        axes[0].plot(efforts, f1s, f'{marker}-', color=color, label=mt, markersize=8, linewidth=2)
        axes[1].plot(efforts, recalls, f'{marker}-', color=color, label=mt, markersize=8, linewidth=2)
        
        # Annota punti
        for d in data:
            if d['effort'] == max(efforts) or d['effort'] == min(efforts):
                axes[0].annotate(d['version'], (d['effort'], d['f1']), 
                               textcoords="offset points", xytext=(5,5), fontsize=7)
    
    axes[0].set_xlabel('Training Effort (n_iter x cv)')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score vs Training Effort')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.9, 1.01)
    
    axes[1].set_xlabel('Training Effort (n_iter x cv)')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall vs Training Effort')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.9, 1.01)
    
    plt.suptitle('Plateau Analysis: Quando fermare il training?', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'plateau_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Grafico plateau salvato: {output_dir / 'plateau_analysis.png'}")


def plot_scorecard_comparison(results: List[Dict], best_id: Optional[str], output_dir: Path):
    """Genera grafico comparativo scorecard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # Ordina per score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:15]  # Top 15
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(results_sorted) * 0.4)))
    
    labels = [r['full_id'] for r in results_sorted]
    scores = [r['score'] for r in results_sorted]
    colors = ['#2ecc71' if r['status'] == 'PASS' else '#e74c3c' for r in results_sorted]
    
    # Evidenzia best
    for i, r in enumerate(results_sorted):
        if r['full_id'] == best_id:
            colors[i] = '#3498db'
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Score')
    ax.set_title('Confronto Versioni Modelli (Top 15)')
    ax.set_xlim(0, 1)
    
    # Aggiungi valori
    for i, (bar, r) in enumerate(zip(bars, results_sorted)):
        width = bar.get_width()
        status = "BEST" if r['full_id'] == best_id else r['status']
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{width:.3f} [{status}]', va='center', fontsize=8)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'scorecard_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_algorithm_rankings(rankings: Dict[str, List[Dict]], output_dir: Path):
    """Genera grafico ranking per algoritmo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_types = len(rankings)
    if n_types == 0:
        return
    
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 6))
    if n_types == 1:
        axes = [axes]
    
    colors_map = {'random_forest': '#27ae60', 'xgboost': '#3498db', 'lightgbm': '#e67e22'}
    
    for ax, (model_type, versions) in zip(axes, rankings.items()):
        if not versions:
            continue
        
        labels = [v['version_id'] for v in versions]
        f1s = [v['metrics'].get('f1', 0) for v in versions]
        
        color = colors_map.get(model_type, 'gray')
        y_pos = np.arange(len(labels))
        
        bars = ax.barh(y_pos, f1s, color=color, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('F1 Score')
        ax.set_title(f'{model_type.upper()}\nRanking Versioni')
        ax.set_xlim(0.9, 1.0)
        
        # Best di questo tipo
        if versions:
            best_idx = 0
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{f1:.4f}', va='center', fontsize=8)
        
        ax.invert_yaxis()
    
    plt.suptitle('Ranking Intra-Algoritmo', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_rankings.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# SALVATAGGIO BEST MODEL
# ==============================================================================

def copy_best_model(best_id: str, task: str, results: List[Dict], output_dir: Path):
    """Copia il best model nella directory best_model."""
    
    # Trova il risultato
    best_result = None
    for r in results:
        if r['full_id'] == best_id:
            best_result = r
            break
    
    if not best_result:
        return
    
    model_type, version_id = best_id.split('/')
    source_dir = get_project_root() / "models" / model_type
    
    if version_id != 'default':
        source_dir = source_dir / version_id
    
    # Pulisci output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copia files
    for filename in [f"model_{task}.pkl", f"results_{task}.json", f"features_{task}.json"]:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
    
    # Metadata
    metadata = {
        'best_model': best_id,
        'model_type': model_type,
        'version_id': version_id,
        'task': task,
        'selected_at': datetime.now().isoformat(),
        'constraints': best_result['constraints'],
        'metrics': best_result['metrics'],
        'latency': best_result['latency'],
        'score': best_result['score']
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Salva tutti i risultati
    def convert_numpy(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    with open(output_dir / "all_versions_comparison.json", 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


# ==============================================================================
# REPORT
# ==============================================================================

def generate_comparison_report(results: List[Dict], rankings: Dict, 
                               best_id: Optional[str], output_dir: Path):
    """Genera report testuale."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_report.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONFRONTO TUTTE LE VERSIONI MODELLI\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Versioni valutate: {len(results)}\n")
        f.write(f"Versioni che passano: {sum(1 for r in results if r['status'] == 'PASS')}\n\n")
        
        # Constraints
        if results:
            c = results[0]['constraints']
            f.write("HARD CONSTRAINTS:\n")
            f.write(f"  - FPR max:     {c['fpr_threshold']*100:.2f}%\n")
            f.write(f"  - Latency max: {c['latency_threshold_ms']:.2f}ms\n\n")
        
        # Ranking per algoritmo
        f.write("=" * 80 + "\n")
        f.write("RANKING PER ALGORITMO\n")
        f.write("=" * 80 + "\n")
        
        for model_type, versions in rankings.items():
            f.write(f"\n{model_type.upper()}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'#':<3} {'Versione':<20} {'F1':>10} {'Recall':>10} {'Score':>10} {'Status'}\n")
            
            for v in versions:
                f1 = v['metrics'].get('f1', 0)
                recall = v['metrics'].get('recall', 0)
                f.write(f"{v['intra_rank']:<3} {v['version_id']:<20} {f1:>10.4f} {recall:>10.4f} {v['score']:>10.4f} {v['status']}\n")
        
        # Best overall
        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST MODEL OVERALL\n")
        f.write("=" * 80 + "\n\n")
        
        if best_id:
            f.write(f"Selezionato: {best_id}\n")
            best = next((r for r in results if r['full_id'] == best_id), None)
            if best:
                f.write(f"Score: {best['score']:.4f}\n")
                f.write(f"F1: {best['metrics'].get('f1', 0):.4f}\n")
                f.write(f"Recall: {best['metrics'].get('recall', 0):.4f}\n")
        else:
            f.write("Nessun modello soddisfa i constraints!\n")
    
    logger.info(f"Report salvato: {output_dir / 'comparison_report.txt'}")


# ==============================================================================
# MAIN
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Confronta tutte le versioni modelli')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--max-fpr', type=float, default=DEFAULT_MAX_FPR)
    parser.add_argument('--max-latency-ms', type=float, default=DEFAULT_MAX_LATENCY_MS)
    parser.add_argument('--output-dir', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("\n" + "=" * 70)
    print("CONFRONTO TUTTE LE VERSIONI MODELLI")
    print("=" * 70)
    print(f"\nTask: {args.task}")
    print(f"Constraints: FPR <= {args.max_fpr*100:.2f}%, Latency <= {args.max_latency_ms:.2f}ms")
    
    output_dir = args.output_dir or get_project_root() / "models" / "best_model"
    
    # Confronta tutte le versioni
    print("\n1. Valutazione versioni...")
    results, best_id = compare_all_versions(args.task, args.max_fpr, args.max_latency_ms)
    
    if not results:
        print("Nessuna versione trovata!")
        return
    
    # Calcola ranking
    print("\n2. Calcolo ranking per algoritmo...")
    rankings = compute_algorithm_rankings(results)
    
    # Mostra tabella
    print("\n" + "-" * 90)
    print(f"{'Versione':<35} {'F1':>10} {'Recall':>10} {'Latency':>12} {'Score':>10} {'Status'}")
    print("-" * 90)
    
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        f1 = r['metrics'].get('f1', 0)
        recall = r['metrics'].get('recall', 0)
        lat = r['latency'].get('latency_per_sample_ms', 0) if r['latency'] else 0
        print(f"{r['full_id']:<35} {f1:>10.4f} {recall:>10.4f} {lat:>12.4f} {r['score']:>10.4f} {r['status']}")
    
    print("-" * 90)
    
    if best_id:
        print(f"\nBEST MODEL: {best_id}")
    else:
        print("\nNessun modello passa i constraints!")
    
    # Grafici
    print("\n3. Generazione grafici...")
    plot_scorecard_comparison(results, best_id, output_dir)
    plot_algorithm_rankings(rankings, output_dir)
    plot_plateau_analysis(results, output_dir)
    
    # Report
    print("\n4. Generazione report...")
    generate_comparison_report(results, rankings, best_id, output_dir)
    
    # Copia best
    if best_id:
        print("\n5. Copia best model...")
        copy_best_model(best_id, args.task, results, output_dir)
    
    print("\n" + "=" * 70)
    print("CONFRONTO COMPLETATO")
    print("=" * 70)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()