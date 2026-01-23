"""
================================================================================
NIDS-ML - Model Versioning System
================================================================================

Sistema di versionamento per salvare modelli con parametri di tuning diversi.

STRUTTURA DIRECTORY:
--------------------
models/
├── xgboost/
│   ├── random_iter50/        # Versione con random search, 50 iter
│   │   ├── model_binary.pkl
│   │   ├── results_binary.json
│   │   └── features_binary.json
│   ├── bayesian_trials100/   # Versione con bayesian, 100 trials
│   │   ├── model_binary.pkl
│   │   ├── results_binary.json
│   │   └── features_binary.json
│   └── latest -> bayesian_trials100
├── lightgbm/
│   └── ...
└── best_model/               # Migliore in assoluto

================================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import shutil
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_logger, get_project_root

logger = get_logger(__name__)


def generate_version_id(training_mode: str = 'random_search', 
                       n_iter: int = None, 
                       n_trials: int = None,
                       extra_params: Dict = None) -> str:
    """
    Genera un ID versione basato sui parametri di tuning.
    
    Args:
        training_mode: 'tuned_params', 'random_search', etc.
        n_iter: Numero iterazioni (per random search)
        n_trials: Numero trials (per bayesian)
        extra_params: Parametri aggiuntivi
    
    Returns:
        Version ID, es. "random_iter50" o "bayesian_trials100"
    """
    if training_mode == 'tuned_params':
        tuning_file = extra_params.get('tuning_file') if extra_params else None
        if tuning_file:
            try:
                with open(tuning_file) as f:
                    data = json.load(f)
                method = data.get('tuning_method', 'unknown')
                if 'random' in method:
                    n = data.get('search_config', {}).get('n_iterations', 0)
                    return f"random_iter{n}"
                elif 'bayesian' in method:
                    n = data.get('search_config', {}).get('n_iterations', 0)
                    return f"bayesian_trials{n}"
            except Exception:
                pass
        return "tuned_params"
    
    if n_trials is not None:
        version_id = f"bayesian_trials{n_trials}"
    elif n_iter is not None:
        version_id = f"random_iter{n_iter}"
    else:
        version_id = "default"
    
    if extra_params:
        if extra_params.get('gpu'):
            version_id += "_gpu"
        if extra_params.get('early_stopping') is False:
            version_id += "_noes"
    
    return version_id


def get_version_dir(model_type: str, 
                    training_mode: str = 'random_search',
                    n_iter: int = None,
                    n_trials: int = None,
                    extra_params: Dict = None, 
                    create: bool = True) -> Path:
    """
    Ottiene la directory per una specifica versione del modello.
    
    Args:
        model_type: 'xgboost', 'lightgbm', 'random_forest'
        training_mode: Modalità training
        n_iter: Numero iterazioni random search
        n_trials: Numero trials bayesian
        extra_params: Parametri aggiuntivi
        create: Se True, crea la directory
    
    Returns:
        Path alla directory della versione
    """
    version_id = generate_version_id(training_mode, n_iter, n_trials, extra_params)
    version_dir = get_project_root() / "models" / model_type / version_id
    
    if create:
        version_dir.mkdir(parents=True, exist_ok=True)
    
    return version_dir


def save_versioned_model(model, results: Dict, selected_features: List[str],
                         model_type: str, n_iter: int = None, cv: int = None,
                         extra_params: Dict = None) -> Tuple[Path, str]:
    """
    Salva modello con versionamento.
    
    Args:
        model: Modello trained
        results: Risultati training
        selected_features: Lista feature usate
        model_type: Tipo modello
        n_iter: Iterazioni (per backward compatibility, deprecato)
        cv: Fold CV (per backward compatibility, deprecato)
        extra_params: Parametri extra
    
    Returns:
        Tuple (path directory, version_id)
    """
    import joblib
    
    training_mode = results.get('training_mode', 'random_search')
    task = results.get('task', 'binary')
    
    if training_mode == 'tuned_params':
        tuning_file = get_project_root() / "tuning_results" / f"{model_type}_best.json"
        if extra_params is None:
            extra_params = {}
        extra_params['tuning_file'] = tuning_file
        version_dir = get_version_dir(
            model_type, 
            training_mode=training_mode,
            extra_params=extra_params,
            create=True
        )
        version_id = generate_version_id(training_mode, extra_params=extra_params)
    else:
        n_iterations = results.get('n_iter') or n_iter
        n_trials = results.get('n_trials')
        version_dir = get_version_dir(
            model_type,
            training_mode=training_mode,
            n_iter=n_iterations,
            n_trials=n_trials,
            extra_params=extra_params,
            create=True
        )
        version_id = generate_version_id(
            training_mode,
            n_iter=n_iterations,
            n_trials=n_trials,
            extra_params=extra_params
        )
    
    results['version'] = {
        'version_id': version_id,
        'model_type': model_type,
        'training_mode': training_mode,
        'extra_params': extra_params or {},
        'created_at': datetime.now().isoformat()
    }
    
    model_path = version_dir / f"model_{task}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")
    
    results_path = version_dir / f"results_{task}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Risultati salvati: {results_path}")
    
    features_path = version_dir / f"features_{task}.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Feature salvate: {features_path}")
    
    latest_link = get_project_root() / "models" / model_type / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        shutil.rmtree(latest_link)
    
    try:
        latest_link.symlink_to(version_dir.name)
        logger.info(f"Symlink 'latest' aggiornato -> {version_id}")
    except OSError:
        pass
    
    return version_dir, version_id


def list_model_versions(model_type: str = None, task: str = 'binary') -> List[Dict]:
    """
    Lista tutte le versioni disponibili per un tipo di modello.
    
    Args:
        model_type: Tipo modello (None = tutti)
        task: 'binary' o 'multiclass'
    
    Returns:
        Lista di dict con info versioni
    """
    models_dir = get_project_root() / "models"
    versions = []
    
    model_types = [model_type] if model_type else ['random_forest', 'xgboost', 'lightgbm']
    
    for mtype in model_types:
        type_dir = models_dir / mtype
        if not type_dir.exists():
            continue
        
        root_model = type_dir / f"model_{task}.pkl"
        root_results = type_dir / f"results_{task}.json"
        
        if root_model.exists():
            version_info = {
                'model_type': mtype,
                'version_id': 'default',
                'path': type_dir,
                'model_path': root_model
            }
            
            if root_results.exists():
                try:
                    with open(root_results) as f:
                        results = json.load(f)
                    version_info['results'] = results
                    version_info['validation_metrics'] = results.get('validation_metrics', {})
                    version_info['train_time'] = results.get('train_time_seconds', 0)
                    version_info['training_mode'] = results.get('training_mode', 'unknown')
                except Exception as e:
                    logger.warning(f"Errore lettura results {root_results}: {e}")
            
            versions.append(version_info)
        
        for version_dir in type_dir.iterdir():
            if version_dir.is_symlink() or version_dir.is_file():
                continue
            
            model_file = version_dir / f"model_{task}.pkl"
            if not model_file.exists():
                continue
            
            version_info = {
                'model_type': mtype,
                'version_id': version_dir.name,
                'path': version_dir,
                'model_path': model_file
            }
            
            results_file = version_dir / f"results_{task}.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    version_info['results'] = results
                    version_info['validation_metrics'] = results.get('validation_metrics', {})
                    version_info['train_time'] = results.get('train_time_seconds', 0)
                    version_info['training_mode'] = results.get('training_mode', 'unknown')
                    
                    version_meta = results.get('version', {})
                    version_info['n_iter'] = version_meta.get('n_iter', _parse_n_iter(version_dir.name))
                    version_info['cv'] = version_meta.get('cv', 0)
                except Exception as e:
                    logger.warning(f"Errore lettura results {results_file}: {e}")
            
            versions.append(version_info)
    
    versions.sort(key=lambda x: (
        x['model_type'],
        -x.get('validation_metrics', {}).get('f1', 0)
    ))
    
    return versions


def _parse_n_iter(version_id: str) -> int:
    """Estrae n_iter dal version_id."""
    import re
    match = re.search(r'iter(\d+)', version_id)
    if match:
        return int(match.group(1))
    match = re.search(r'trials(\d+)', version_id)
    if match:
        return int(match.group(1))
    return 0


def get_best_version(model_type: str = None, task: str = 'binary') -> Optional[Dict]:
    """
    Trova la migliore versione tra tutte quelle disponibili.
    
    Args:
        model_type: Filtra per tipo (None = tutti)
        task: 'binary' o 'multiclass'
    
    Returns:
        Dict con info migliore versione, o None
    """
    versions = list_model_versions(model_type, task)
    
    if not versions:
        return None
    
    best = max(versions, key=lambda x: x.get('validation_metrics', {}).get('f1', 0))
    return best


def print_versions_summary(task: str = 'binary'):
    """Stampa riepilogo di tutte le versioni disponibili."""
    versions = list_model_versions(task=task)
    
    if not versions:
        print("Nessuna versione trovata.")
        return
    
    print("\n" + "=" * 90)
    print("VERSIONI MODELLI DISPONIBILI")
    print("=" * 90)
    print(f"\n{'Tipo':<15} {'Versione':<25} {'F1':>10} {'Recall':>10} {'Mode':<15}")
    print("-" * 90)
    
    current_type = None
    for v in versions:
        if v['model_type'] != current_type:
            if current_type is not None:
                print("-" * 90)
            current_type = v['model_type']
        
        metrics = v.get('validation_metrics', {})
        f1 = metrics.get('f1', 0)
        recall = metrics.get('recall', 0)
        mode = v.get('training_mode', 'unknown')
        
        print(f"{v['model_type']:<15} {v['version_id']:<25} {f1:>10.4f} {recall:>10.4f} {mode:<15}")
    
    print("=" * 90)
    
    best = get_best_version(task=task)
    if best:
        print(f"\nMigliore: {best['model_type']}/{best['version_id']} (F1={best.get('validation_metrics', {}).get('f1', 0):.4f})")


if __name__ == "__main__":
    print_versions_summary()