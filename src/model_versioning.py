"""
================================================================================
NIDS-ML - Model Versioning System (NEW)
================================================================================

Sistema di versionamento che rispecchia la struttura di tuning_results/.

STRUTTURA:
----------
models/
├── xgboost/
│   ├── random_iter50_cv5_2026-01-24_20.02/
│   │   ├── model_binary.pkl
│   │   ├── results_binary.json
│   │   ├── features_binary.json
│   │   └── tuning_source.json       # ← Link alla config tuning usata
│   └── bayesian_trials100_cv5_2026-01-25_15.30/
│       └── ...
├── lightgbm/
│   └── ...
└── random_forest/
    └── ...

CORRELAZIONE 1:1:
-----------------
tuning_results/xgboost/random_iter50_cv5_2026-01-24_20.02.json
        ↓
models/xgboost/random_iter50_cv5_2026-01-24_20.02/
        ↓ contiene tuning_source.json che punta al file tuning

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


def extract_version_id_from_tuning_file(tuning_filepath: Path) -> str:
    """
    Estrae version_id dal nome file tuning.
    
    Es: "random_iter50_cv5_2026-01-24_20.02.json" 
        → "random_iter50_cv5_2026-01-24_20.02"
    """
    return tuning_filepath.stem  # Rimuove .json


def get_version_dir_from_tuning(
    model_type: str,
    tuning_filepath: Path,
    create: bool = True
) -> Path:
    """
    Crea directory versione basata sul file tuning.
    
    Args:
        model_type: 'xgboost', 'lightgbm', 'random_forest'
        tuning_filepath: Path al file JSON tuning usato
        create: Se True, crea la directory
    
    Returns:
        Path alla directory versione
    """
    version_id = extract_version_id_from_tuning_file(tuning_filepath)
    version_dir = get_project_root() / "models" / model_type / version_id
    
    if create:
        version_dir.mkdir(parents=True, exist_ok=True)
    
    return version_dir


def save_tuned_model(
    model,
    results: Dict,
    selected_features: List[str],
    model_type: str,
    tuning_filepath: Path
) -> Tuple[Path, str]:
    """
    Salva modello trainato con parametri tuned.
    
    Args:
        model: Modello trained
        results: Risultati training (metriche, etc.)
        selected_features: Feature usate
        model_type: Tipo modello
        tuning_filepath: Path al file tuning usato
    
    Returns:
        Tuple (version_dir, version_id)
    """
    import joblib
    
    version_dir = get_version_dir_from_tuning(model_type, tuning_filepath, create=True)
    version_id = extract_version_id_from_tuning_file(tuning_filepath)
    task = results.get('task', 'binary')
    
    # Aggiungi metadata versione
    results['version'] = {
        'version_id': version_id,
        'model_type': model_type,
        'training_mode': 'tuned_params',
        'tuning_source': str(tuning_filepath),
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
    
    # Salva link al tuning source
    tuning_source_path = version_dir / "tuning_source.json"
    tuning_source_data = {
        'tuning_file': str(tuning_filepath),
        'tuning_file_relative': str(tuning_filepath.relative_to(get_project_root())),
        'version_id': version_id,
        'linked_at': datetime.now().isoformat()
    }
    with open(tuning_source_path, 'w') as f:
        json.dump(tuning_source_data, f, indent=2)
    logger.info(f"Tuning source salvato: {tuning_source_path}")
    
    return version_dir, version_id


def list_model_versions(model_type: str = None, task: str = 'binary') -> List[Dict]:
    """
    Lista tutte le versioni disponibili (nuova struttura).
    
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
        
        # Cerca versioni nelle sottocartelle
        for version_dir in type_dir.iterdir():
            if not version_dir.is_dir():
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
            
            # Carica risultati
            results_file = version_dir / f"results_{task}.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    version_info['results'] = results
                    version_info['validation_metrics'] = results.get('validation_metrics', {})
                    version_info['train_time'] = results.get('train_time_seconds', 0)
                    version_info['training_mode'] = results.get('training_mode', 'unknown')
                except Exception as e:
                    logger.warning(f"Errore lettura results {results_file}: {e}")
            
            # Carica tuning source
            tuning_source_file = version_dir / "tuning_source.json"
            if tuning_source_file.exists():
                try:
                    with open(tuning_source_file) as f:
                        tuning_source = json.load(f)
                    version_info['tuning_source'] = tuning_source
                except Exception:
                    pass
            
            versions.append(version_info)
    
    # Ordina per model_type, poi per score
    versions.sort(key=lambda x: (
        x['model_type'],
        -x.get('validation_metrics', {}).get('f1', 0)
    ))
    
    return versions


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
    """Stampa riepilogo versioni disponibili."""
    versions = list_model_versions(task=task)
    
    if not versions:
        print("Nessuna versione trovata.")
        return
    
    print("\n" + "=" * 90)
    print("VERSIONI MODELLI DISPONIBILI")
    print("=" * 90)
    print(f"\n{'Tipo':<15} {'Versione':<45} {'F1':>10} {'Recall':>10}")
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
        
        print(f"{v['model_type']:<15} {v['version_id']:<45} {f1:>10.4f} {recall:>10.4f}")
    
    print("=" * 90)
    
    best = get_best_version(task=task)
    if best:
        print(f"\nMigliore: {best['model_type']}/{best['version_id']} (F1={best.get('validation_metrics', {}).get('f1', 0):.4f})")


if __name__ == "__main__":
    print_versions_summary()