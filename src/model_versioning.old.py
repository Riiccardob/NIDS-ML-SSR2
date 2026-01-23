"""
================================================================================
NIDS-ML - Model Versioning System
================================================================================

Sistema di versionamento per salvare multiple versioni di modelli con parametri
diversi e confrontarle successivamente.

STRUTTURA DIRECTORY:
--------------------
models/
├── xgboost/
│   ├── cv3_iter20/           # Versione con cv=3, n_iter=20
│   │   ├── model_binary.pkl
│   │   ├── results_binary.json
│   │   └── features_binary.json
│   ├── cv5_iter100/          # Versione con cv=5, n_iter=100
│   │   ├── model_binary.pkl
│   │   ├── results_binary.json
│   │   └── features_binary.json
│   └── latest -> cv5_iter100  # Symlink all'ultima versione
├── lightgbm/
│   └── ...
└── best_model/               # Migliore in assoluto tra tutte le versioni

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


def generate_version_id(n_iter: int, cv: int, extra_params: Dict = None) -> str:
    """
    Genera un ID versione basato sui parametri di training.
    
    Args:
        n_iter: Numero iterazioni hyperparameter search
        cv: Numero fold cross-validation
        extra_params: Parametri aggiuntivi (es. gpu=True)
    
    Returns:
        Version ID, es. "cv5_iter100" o "cv5_iter100_gpu"
    """
    version_id = f"cv{cv}_iter{n_iter}"
    
    if extra_params:
        if extra_params.get('gpu'):
            version_id += "_gpu"
        if extra_params.get('early_stopping') is False:
            version_id += "_noes"
    
    return version_id


def get_version_dir(model_type: str, n_iter: int, cv: int, 
                    extra_params: Dict = None, create: bool = True) -> Path:
    """
    Ottiene la directory per una specifica versione del modello.
    
    Args:
        model_type: 'xgboost', 'lightgbm', 'random_forest'
        n_iter: Numero iterazioni
        cv: Numero fold CV
        extra_params: Parametri aggiuntivi
        create: Se True, crea la directory
    
    Returns:
        Path alla directory della versione
    """
    version_id = generate_version_id(n_iter, cv, extra_params)
    version_dir = get_project_root() / "models" / model_type / version_id
    
    if create:
        version_dir.mkdir(parents=True, exist_ok=True)
    
    return version_dir


def save_versioned_model(model, results: Dict, selected_features: List[str],
                         model_type: str, n_iter: int, cv: int,
                         extra_params: Dict = None) -> Tuple[Path, str]:
    """
    Salva modello con versionamento.
    
    Args:
        model: Modello trained
        results: Risultati training
        selected_features: Lista feature usate
        model_type: Tipo modello
        n_iter: Iterazioni
        cv: Fold CV
        extra_params: Parametri extra
    
    Returns:
        Tuple (path directory, version_id)
    """
    import joblib
    
    version_id = generate_version_id(n_iter, cv, extra_params)
    version_dir = get_version_dir(model_type, n_iter, cv, extra_params, create=True)
    
    task = results.get('task', 'binary')
    
    # Aggiungi metadata versione
    results['version'] = {
        'version_id': version_id,
        'model_type': model_type,
        'n_iter': n_iter,
        'cv': cv,
        'extra_params': extra_params or {},
        'created_at': datetime.now().isoformat()
    }
    
    # Salva modello
    model_path = version_dir / f"model_{task}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")
    
    # Salva risultati
    results_path = version_dir / f"results_{task}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Risultati salvati: {results_path}")
    
    # Salva feature
    features_path = version_dir / f"features_{task}.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Feature salvate: {features_path}")
    
    # Aggiorna symlink "latest"
    latest_link = get_project_root() / "models" / model_type / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        shutil.rmtree(latest_link)
    
    try:
        latest_link.symlink_to(version_dir.name)
        logger.info(f"Symlink 'latest' aggiornato -> {version_id}")
    except OSError:
        # Windows potrebbe non supportare symlink
        pass
    
    return version_dir, version_id


def list_model_versions(model_type: str = None, task: str = 'binary') -> List[Dict]:
    """
    Lista tutte le versioni disponibili per un tipo di modello.
    
    Include sia versioni nelle sottocartelle che modelli "default" nella root.
    
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
        
        # Prima cerca modello "default" direttamente nella root del tipo
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
                    version_info['n_iter'] = results.get('n_iter', 0)
                    version_info['cv'] = results.get('cv_folds', results.get('cv', 0))
                except Exception as e:
                    logger.warning(f"Errore lettura results {root_results}: {e}")
            
            versions.append(version_info)
        
        # Poi cerca versioni nelle sottocartelle
        for version_dir in type_dir.iterdir():
            # Salta symlink e file
            if version_dir.is_symlink() or version_dir.is_file():
                continue
            
            # Salta se non contiene il modello
            model_file = version_dir / f"model_{task}.pkl"
            if not model_file.exists():
                continue
            
            version_info = {
                'model_type': mtype,
                'version_id': version_dir.name,
                'path': version_dir,
                'model_path': model_file
            }
            
            # Carica risultati se presenti
            results_file = version_dir / f"results_{task}.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    version_info['results'] = results
                    version_info['validation_metrics'] = results.get('validation_metrics', {})
                    version_info['train_time'] = results.get('train_time_seconds', 0)
                    
                    # Estrai parametri dalla versione
                    version_meta = results.get('version', {})
                    version_info['n_iter'] = version_meta.get('n_iter', _parse_n_iter(version_dir.name))
                    version_info['cv'] = version_meta.get('cv', _parse_cv(version_dir.name))
                except Exception as e:
                    logger.warning(f"Errore lettura results {results_file}: {e}")
            
            versions.append(version_info)
    
    # Ordina per model_type, poi per score
    versions.sort(key=lambda x: (
        x['model_type'],
        -x.get('validation_metrics', {}).get('f1', 0)
    ))
    
    return versions


def _parse_n_iter(version_id: str) -> int:
    """Estrae n_iter dal version_id (es. cv5_iter100 -> 100)."""
    import re
    match = re.search(r'iter(\d+)', version_id)
    return int(match.group(1)) if match else 0


def _parse_cv(version_id: str) -> int:
    """Estrae cv dal version_id (es. cv5_iter100 -> 5)."""
    import re
    match = re.search(r'cv(\d+)', version_id)
    return int(match.group(1)) if match else 0


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
    
    # Trova la migliore per F1 score
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
    print(f"\n{'Tipo':<15} {'Versione':<20} {'F1':>10} {'Recall':>10} {'FPR':>10} {'Tempo':>10}")
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
        fpr = metrics.get('false_positive_rate', 0)
        train_time = v.get('train_time', 0)
        
        print(f"{v['model_type']:<15} {v['version_id']:<20} {f1:>10.4f} {recall:>10.4f} {fpr:>10.4f} {train_time/60:>9.1f}m")
    
    print("=" * 90)
    
    best = get_best_version(task=task)
    if best:
        print(f"\nMigliore: {best['model_type']}/{best['version_id']} (F1={best.get('validation_metrics', {}).get('f1', 0):.4f})")


def cleanup_old_versions(model_type: str, keep_n: int = 5, task: str = 'binary'):
    """
    Rimuove versioni vecchie mantenendo le N migliori.
    
    Args:
        model_type: Tipo modello
        keep_n: Numero versioni da mantenere
        task: 'binary' o 'multiclass'
    """
    versions = list_model_versions(model_type, task)
    
    if len(versions) <= keep_n:
        return
    
    # Ordina per F1 (migliori prima)
    versions.sort(key=lambda x: x.get('validation_metrics', {}).get('f1', 0), reverse=True)
    
    # Rimuovi le peggiori
    to_remove = versions[keep_n:]
    
    for v in to_remove:
        try:
            shutil.rmtree(v['path'])
            logger.info(f"Rimossa versione: {v['path']}")
        except Exception as e:
            logger.warning(f"Errore rimozione {v['path']}: {e}")


if __name__ == "__main__":
    print_versions_summary()