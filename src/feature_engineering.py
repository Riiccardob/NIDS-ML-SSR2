"""
================================================================================
NIDS-ML - Modulo Feature Engineering
================================================================================

Scaling e selezione feature per preparare i dati al training.

AGGIORNAMENTO: RandomForest → XGBoost per feature selection
-------------------------------------------------------------
Basato su test empirici su CICIDS-2017 (706k samples, 77 feature):
- XGBoost:      F2 = 0.9988, Tempo = 4.6s  ✅ MIGLIORE
- LightGBM:     F2 = 0.9987, Tempo = 5.3s
- RandomForest: F2 = 0.9977, Tempo = 49.0s

XGBoost scelto perché:
1. Performance migliore (+0.0011 F2 vs RandomForest)
2. 10x più veloce di RandomForest
3. Gestione nativa class imbalance (scale_pos_weight)
4. Industry standard per IDS/NIDS
5. Robusto su dataset sbilanciati
6. Probabile alignment con best model production

GUIDA PARAMETRI:
----------------
    python src/feature_engineering.py [opzioni]

Opzioni disponibili:
    --n-features INT      Numero feature da selezionare (default: 30)
    --n-estimators INT    Estimatori XGBoost (default: 100)
    --n-jobs INT          Core CPU da usare (default: auto, lascia 2 liberi)
    --max-ram INT         Limite RAM percentuale (default: 85)

ESEMPI:
-------
# Esecuzione standard
python src/feature_engineering.py

# Limita a 4 core
python src/feature_engineering.py --n-jobs 4

# Test veloce con pochi estimatori
python src/feature_engineering.py --n-estimators 50

================================================================================
"""

# ==============================================================================
# IMPORTANTE: Setup limiti risorse PRIMA di altri import
# ==============================================================================
import sys
import os
import argparse
from pathlib import Path

# Parse args prima di tutto per ottenere n_jobs
def _get_n_jobs_from_args():
    """Estrae n_jobs dagli argomenti senza fare parsing completo."""
    for i, arg in enumerate(sys.argv):
        if arg == '--n-jobs' and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return None
    return None

# Applica limiti PRIMA di importare sklearn
_n_jobs_arg = _get_n_jobs_from_args()
_n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

# Imposta variabili d'ambiente per limitare thread
os.environ['OMP_NUM_THREADS'] = str(_n_cores)
os.environ['MKL_NUM_THREADS'] = str(_n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(_n_cores)
os.environ['LOKY_MAX_CPU_COUNT'] = str(_n_cores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(_n_cores)

# Applica affinity CPU (il metodo piu efficace)
import psutil
try:
    p = psutil.Process()
    p.cpu_affinity(list(range(_n_cores)))
    p.nice(10)  # Priorita bassa
except Exception:
    pass

# Ora possiamo importare il resto
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, List
import joblib
import json
import gc
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    LABEL_COLUMNS,
    ResourceLimiter,
    suppress_warnings
)
from src.preprocessing import load_processed_data
from src.timing import TimingLogger

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE DEFAULT
# ==============================================================================

DEFAULT_N_FEATURES = 30
DEFAULT_RF_ESTIMATORS = 100  # Mantenuto per backward compatibility
DEFAULT_MAX_RAM = 85


# ==============================================================================
# PREPARAZIONE FEATURE
# ==============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Estrae nomi colonne feature, escludendo label."""
    feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    return sorted(feature_cols)


def prepare_xy(df: pd.DataFrame,
               label_col: str = 'Label_Binary',
               feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa feature (X) e target (y)."""
    if label_col not in df.columns:
        raise KeyError(f"Colonna target '{label_col}' non trovata")
    
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    
    return X, y


# ==============================================================================
# SCALING
# ==============================================================================

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler sui dati di training."""
    logger.info(f"Fitting scaler su {len(X_train):,} campioni, "
                f"{X_train.shape[1]} feature")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return scaler


def transform_data(X: pd.DataFrame, 
                   scaler: StandardScaler) -> pd.DataFrame:
    """Applica trasformazione scaler."""
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# ==============================================================================
# SELEZIONE FEATURE - XGBOOST
# ==============================================================================

def select_features_by_importance(X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  n_features: int = DEFAULT_N_FEATURES,
                                  n_estimators: int = DEFAULT_RF_ESTIMATORS,
                                  n_jobs: int = None,
                                  random_state: int = RANDOM_STATE
                                  ) -> Tuple[List[str], dict]:
    """
    Seleziona feature più importanti usando XGBoost.
    
    EMPIRICAL EVIDENCE (test su CICIDS-2017):
    - XGBoost selection: F2 = 0.9988 (migliore)
    - LightGBM selection: F2 = 0.9987
    - RandomForest selection: F2 = 0.9977
    
    XGBoost scelto per:
    - Performance: +0.0011 F2 vs RandomForest
    - Velocità: 4.6s vs 49s RandomForest
    - Robustezza: Gestione migliore di class imbalance
    - Industry standard: Più usato in IDS/NIDS
    
    Args:
        X_train: Feature training scalate
        y_train: Target training
        n_features: Numero feature da selezionare
        n_estimators: Numero estimators XGBoost
        n_jobs: Core CPU (None = usa default da environment)
        random_state: Seed
    
    Returns:
        Tuple (lista feature selezionate, dict importanze)
    """
    from xgboost import XGBClassifier
    
    # Usa n_jobs configurato o quello dalle variabili d'ambiente
    if n_jobs is None:
        n_jobs = int(os.environ.get('OMP_NUM_THREADS', -1))
    
    logger.info(f"Training XGBoost per selezione feature "
                f"(n_estimators={n_estimators}, n_jobs={n_jobs})...")
    
    # Calcola scale_pos_weight per dataset sbilanciati
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method='hist',  # Veloce per grandi dataset
        scale_pos_weight=scale_pos_weight,  # Gestione class imbalance
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Training con progress manuale
    print(f"   Training XGBoost: {n_estimators} estimators su {len(X_train):,} campioni...")
    print(f"   Core CPU in uso: {n_jobs}")
    print(f"   Class imbalance ratio: {scale_pos_weight:.2f}:1")
    
    xgb.fit(X_train, y_train, verbose=False)
    
    print("   ✓ XGBoost training completato")
    
    # Estrai importanze (native in XGBoost)
    importances = xgb.feature_importances_
    feature_names = X_train.columns.tolist()
    
    importance_dict = {
        name: float(imp) 
        for name, imp in zip(feature_names, importances)
    }
    
    sorted_features = sorted(importance_dict.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
    
    selected_features = [name for name, _ in sorted_features[:n_features]]
    
    logger.info(f"Selezionate {n_features} feature su {len(feature_names)}")
    
    del xgb
    gc.collect()
    
    return selected_features, importance_dict


def apply_feature_selection(X: pd.DataFrame,
                            selected_features: List[str]) -> pd.DataFrame:
    """Filtra DataFrame mantenendo solo feature selezionate."""
    missing = set(selected_features) - set(X.columns)
    if missing:
        raise KeyError(f"Feature mancanti: {missing}")
    
    return X[selected_features].copy()


# ==============================================================================
# SALVATAGGIO/CARICAMENTO ARTIFACTS
# ==============================================================================

def save_artifacts(scaler: StandardScaler,
                   selected_features: List[str],
                   feature_importances: dict,
                   scaler_columns: List[str] = None,
                   output_dir: Path = None) -> None:
    """Salva artifacts del feature engineering."""
    if output_dir is None:
        output_dir = get_project_root() / "artifacts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, output_dir / "scaler.pkl")
    logger.info(f"Salvato: scaler.pkl")
    
    with open(output_dir / "selected_features.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Salvato: selected_features.json")
    
    # Salva le colonne usate per fittare lo scaler
    if scaler_columns is not None:
        with open(output_dir / "scaler_columns.json", 'w') as f:
            json.dump(scaler_columns, f, indent=2)
        logger.info(f"Salvato: scaler_columns.json ({len(scaler_columns)} colonne)")
    
    sorted_importances = dict(sorted(feature_importances.items(),
                                     key=lambda x: x[1], 
                                     reverse=True))
    with open(output_dir / "feature_importances.json", 'w') as f:
        json.dump(sorted_importances, f, indent=2)
    logger.info(f"Salvato: feature_importances.json")


def load_artifacts(artifacts_dir: Path = None
                   ) -> Tuple[StandardScaler, List[str], dict, List[str]]:
    """
    Carica artifacts salvati.
    
    Returns:
        Tuple (scaler, selected_features, importances, scaler_columns)
        scaler_columns puo essere None se non salvato (vecchi artifacts)
    """
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "artifacts"
    
    required = ['scaler.pkl', 'selected_features.json', 'feature_importances.json']
    for f in required:
        if not (artifacts_dir / f).exists():
            raise FileNotFoundError(f"Artifact non trovato: {artifacts_dir / f}")
    
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    
    with open(artifacts_dir / "selected_features.json", 'r') as f:
        selected_features = json.load(f)
    
    with open(artifacts_dir / "feature_importances.json", 'r') as f:
        importances = json.load(f)
    
    # Carica scaler_columns se presente
    scaler_columns = None
    scaler_columns_path = artifacts_dir / "scaler_columns.json"
    if scaler_columns_path.exists():
        with open(scaler_columns_path, 'r') as f:
            scaler_columns = json.load(f)
    
    logger.info(f"Caricati artifacts da {artifacts_dir}")
    
    return scaler, selected_features, importances, scaler_columns


# ==============================================================================
# PIPELINE COMPLETA
# ==============================================================================

def run_feature_engineering(train: pd.DataFrame,
                            val: pd.DataFrame,
                            test: pd.DataFrame,
                            label_col: str = 'Label_Binary',
                            n_features: int = DEFAULT_N_FEATURES,
                            n_estimators: int = DEFAULT_RF_ESTIMATORS,
                            n_jobs: int = None,
                            random_state: int = RANDOM_STATE
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                       pd.Series, pd.Series, pd.Series]:
    """Esegue pipeline completa di feature engineering."""
    feature_cols = get_feature_columns(train)
    logger.info(f"Feature iniziali: {len(feature_cols)}")
    
    X_train, y_train = prepare_xy(train, label_col, feature_cols)
    X_val, y_val = prepare_xy(val, label_col, feature_cols)
    X_test, y_test = prepare_xy(test, label_col, feature_cols)
    
    scaler = fit_scaler(X_train)
    
    X_train_scaled = transform_data(X_train, scaler)
    X_val_scaled = transform_data(X_val, scaler)
    X_test_scaled = transform_data(X_test, scaler)
    
    del X_train, X_val, X_test
    gc.collect()
    
    selected_features, importances = select_features_by_importance(
        X_train_scaled, y_train,
        n_features=n_features,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    X_train_final = apply_feature_selection(X_train_scaled, selected_features)
    X_val_final = apply_feature_selection(X_val_scaled, selected_features)
    X_test_final = apply_feature_selection(X_test_scaled, selected_features)
    
    del X_train_scaled, X_val_scaled, X_test_scaled
    gc.collect()
    
    save_artifacts(scaler, selected_features, importances, scaler_columns=feature_cols)
    
    logger.info(f"Feature engineering completato")
    
    return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    """Parse argomenti CLI."""
    parser = argparse.ArgumentParser(
        description='Feature Engineering per NIDS-ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python src/feature_engineering.py
  python src/feature_engineering.py --n-jobs 4
  python src/feature_engineering.py --n-estimators 50  # Test veloce
        """
    )
    
    parser.add_argument('--n-features', type=int, default=DEFAULT_N_FEATURES,
                        help=f'Feature da selezionare (default: {DEFAULT_N_FEATURES})')
    parser.add_argument('--rf-estimators', type=int, default=DEFAULT_RF_ESTIMATORS,
                        help=f'Estimatori XGBoost (default: {DEFAULT_RF_ESTIMATORS})')
    parser.add_argument('--label-col', type=str, default='Label_Binary',
                        help='Colonna target')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Core CPU (default: auto, totale - 2)')
    parser.add_argument('--max-ram', type=int, default=DEFAULT_MAX_RAM,
                        help='Limite RAM %%')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                        help='Seed random')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funzione principale."""
    args = parse_arguments()
    
    # I limiti CPU sono gia stati applicati all'inizio del file
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    
    # Setup monitor risorse
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Algoritmo selezione:    XGBoost (empiricamente migliore)")
    print(f"  Feature da selezionare: {args.n_features}")
    print(f"  RF estimators:          {args.rf_estimators}")
    print(f"  CPU cores:              {n_jobs}/{os.cpu_count()}")
    print(f"  Max RAM:                {args.max_ram}%")
    print()
    
    # Inizializza timing logger
    timer = TimingLogger("feature_engineering", parameters={
        'algorithm': 'xgboost',
        'n_features': args.n_features,
        'rf_estimators': args.rf_estimators,
        'n_jobs': n_jobs,
        'max_ram': args.max_ram,
        'label_col': args.label_col,
        'random_state': args.random_state
    })
    
    try:
        print("1. Caricamento dati preprocessati...")
        with timer.time_operation("caricamento_dati"):
            train, val, test, mappings = load_processed_data()
        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        
        limiter.log_status(logger)
        
        print("\n2. Esecuzione pipeline feature engineering...")
        with timer.time_operation("feature_engineering_pipeline"):
            X_train, X_val, X_test, y_train, y_val, y_test = run_feature_engineering(
                train, val, test,
                label_col=args.label_col,
                n_features=args.n_features,
                n_estimators=args.rf_estimators,
                n_jobs=n_jobs,
                random_state=args.random_state
            )
        
        print("\n3. Salvataggio dataset pronti per training...")
        with timer.time_operation("salvataggio_dataset"):
            processed_dir = get_project_root() / "data" / "processed"
            
            train_ready = pd.concat([
                X_train.reset_index(drop=True),
                y_train.reset_index(drop=True).rename('target')
            ], axis=1)
            val_ready = pd.concat([
                X_val.reset_index(drop=True),
                y_val.reset_index(drop=True).rename('target')
            ], axis=1)
            test_ready = pd.concat([
                X_test.reset_index(drop=True),
                y_test.reset_index(drop=True).rename('target')
            ], axis=1)
            
            train_ready.to_parquet(processed_dir / "train_ready.parquet", index=False)
            val_ready.to_parquet(processed_dir / "val_ready.parquet", index=False)
            test_ready.to_parquet(processed_dir / "test_ready.parquet", index=False)
        
        print(f"   Salvati in {processed_dir}")
        
        print("\n   Top 10 feature selezionate:")
        _, selected_features, importances, _ = load_artifacts()
        for i, feat in enumerate(selected_features[:10]):
            print(f"   {i+1:2}. {feat}: {importances[feat]:.4f}")
        
        # Salva metriche timing
        timer.add_metric("train_samples", len(train))
        timer.add_metric("n_features_selected", len(selected_features))
        timing_path = timer.save()
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETATO")
        print("=" * 60)
        print(f"\nArtifacts: {get_project_root() / 'artifacts'}")
        print(f"Shape:     ({X_train.shape[0]:,}, {X_train.shape[1]})")
        print(f"Timing:    {timing_path}")
        print(f"\nProssimo step: python src/training/random_forest.py")
        
        timer.print_summary()
        limiter.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima: python src/preprocessing.py")
        sys.exit(1)


if __name__ == "__main__":
    main()