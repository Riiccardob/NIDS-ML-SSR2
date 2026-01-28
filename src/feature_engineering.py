"""
================================================================================
NIDS-ML - Feature Engineering v2
================================================================================

CHANGELOG v2:
-------------
Statistical Preprocessing (DEFAULT ON)
   - Remove low-variance features (quasi-costanti)
   - Remove high-correlation features (ridondanti)
   
RobustScaler (DEFAULT ON - migliore di StandardScaler per NIDS)
   - Usa mediana e IQR invece di media e std
   - Resistente agli outlier (critici in NIDS)

Feature Selection
   - Solo Random Forest Importance (metodo più affidabile)
   - Rimossi HistGradientBoosting e RFECV (risultati insoddisfacenti)

USAGE:
------
# Default (CONSIGLIATO - statistical + robust ON):
python src/feature_engineering.py

# Disabilita statistical:
python src/feature_engineering.py --no-statistical

# Disabilita robust (usa StandardScaler):
python src/feature_engineering.py --no-robust

# Configura soglie statistical:
python src/feature_engineering.py --variance-threshold 0.01 --correlation-threshold 0.90

================================================================================
"""

# Setup limiti CPU
import sys
import os
import argparse
from pathlib import Path

def _get_n_jobs_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--n-jobs' and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return None
    return None

_n_jobs_arg = _get_n_jobs_from_args()
_n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

os.environ['OMP_NUM_THREADS'] = str(_n_cores)
os.environ['MKL_NUM_THREADS'] = str(_n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(_n_cores)
os.environ['LOKY_MAX_CPU_COUNT'] = str(_n_cores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(_n_cores)

import psutil
try:
    p = psutil.Process()
    p.cpu_affinity(list(range(_n_cores)))
    p.nice(10)
except Exception:
    pass

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, List
import joblib
import json
import gc
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    LABEL_COLUMNS,
    ResourceLimiter,
    suppress_warnings,
    compute_column_checksum,
    validate_column_consistency,
    ensure_column_order
)
from src.preprocessing import load_processed_data
from src.timing import TimingLogger

suppress_warnings()
logger = get_logger(__name__)

DEFAULT_N_FEATURES = 30
DEFAULT_RF_ESTIMATORS = 100
DEFAULT_MAX_RAM = 85
DEFAULT_VARIANCE_THRESHOLD = 0.00
DEFAULT_CORRELATION_THRESHOLD = 0.95


# ==============================================================================
# STATISTICAL PREPROCESSING
# ==============================================================================

def statistical_preprocessing(X_train: pd.DataFrame,
                              X_val: pd.DataFrame,
                              X_test: pd.DataFrame,
                              variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
                              correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Statistical preprocessing: rimuove feature problematiche.
    
    Args:
        X_train, X_val, X_test: DataFrame feature
        variance_threshold: Soglia varianza (default: 0.00)
        correlation_threshold: Soglia correlazione (default: 0.95)
    
    Returns:
        Tuple (X_train_filtered, X_val_filtered, X_test_filtered, info_dict)
    """
    original_features = X_train.shape[1]
    info = {'original_features': original_features}
    
    logger.info("="*60)
    logger.info("STATISTICAL PREPROCESSING")
    logger.info("="*60)
    
    # Step 1: Low Variance Filter
    logger.info(f"Step 1: Removing low-variance features (threshold={variance_threshold})...")
    
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)
    X_test_var = variance_selector.transform(X_test)
    
    kept_features_var = X_train.columns[variance_selector.get_support()].tolist()
    removed_var = [f for f in X_train.columns if f not in kept_features_var]
    
    logger.info(f"  Removed {len(removed_var)} low-variance features")
    logger.info(f"  Remaining: {len(kept_features_var)}")
    
    if removed_var and len(removed_var) <= 10:
        logger.info(f"  Removed features: {removed_var}")
    
    info['step1_variance'] = {
        'removed_count': len(removed_var),
        'removed_features': removed_var,
        'kept_count': len(kept_features_var)
    }
    
    # Step 2: High Correlation Filter
    logger.info(f"Step 2: Removing high-correlation features (threshold={correlation_threshold})...")
    
    X_train_df = pd.DataFrame(X_train_var, columns=kept_features_var, index=X_train.index)
    corr_matrix = X_train_df.corr().abs()
    
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = []
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > correlation_threshold):
            to_drop.append(column)
    
    to_drop = list(set(to_drop))
    kept_features_corr = [f for f in kept_features_var if f not in to_drop]
    
    keep_indices = [i for i, f in enumerate(kept_features_var) if f in kept_features_corr]
    X_train_corr = X_train_var[:, keep_indices]
    X_val_corr = X_val_var[:, keep_indices]
    X_test_corr = X_test_var[:, keep_indices]
    
    logger.info(f"  Removed {len(to_drop)} high-correlation features")
    logger.info(f"  Remaining: {len(kept_features_corr)}")
    
    if to_drop and len(to_drop) <= 10:
        logger.info(f"  Removed features: {to_drop}")
    
    info['step2_correlation'] = {
        'removed_count': len(to_drop),
        'removed_features': to_drop,
        'kept_count': len(kept_features_corr)
    }
    
    # Summary
    final_features = len(kept_features_corr)
    reduction_pct = ((original_features - final_features) / original_features) * 100
    
    info['summary'] = {
        'final_features': final_features,
        'reduction_percent': reduction_pct,
        'kept_features': kept_features_corr
    }
    
    logger.info("="*60)
    logger.info(f"Statistical preprocessing completed:")
    logger.info(f"  {original_features} → {len(kept_features_corr)} features ({reduction_pct:.1f}% reduction)")
    logger.info("="*60)
    
    # Converti back to DataFrame
    X_train_final = pd.DataFrame(X_train_corr, columns=kept_features_corr, index=X_train.index)
    X_val_final = pd.DataFrame(X_val_corr, columns=kept_features_corr, index=X_val.index)
    X_test_final = pd.DataFrame(X_test_corr, columns=kept_features_corr, index=X_test.index)
    
    return X_train_final, X_val_final, X_test_final, info


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

def fit_scaler(X_train: pd.DataFrame, use_robust: bool = True):
    """
    Fit scaler su training set.
    
    Args:
        X_train: Training features
        use_robust: Se True usa RobustScaler (DEFAULT), altrimenti StandardScaler
    
    Returns:
        Scaler fitted
    """
    if use_robust:
        scaler = RobustScaler()
        logger.info("Using RobustScaler (outlier-resistant) - DEFAULT")
    else:
        scaler = StandardScaler()
        logger.info("Using StandardScaler")
    
    scaler.fit(X_train)
    logger.info(f"Scaler fitted su {X_train.shape[1]} feature")
    
    return scaler


def transform_data(X: pd.DataFrame, scaler) -> pd.DataFrame:
    """Trasforma dati con scaler."""
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

def select_features_by_importance(X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  n_features: int = DEFAULT_N_FEATURES,
                                  n_estimators: int = DEFAULT_RF_ESTIMATORS,
                                  n_jobs: int = None,
                                  random_state: int = RANDOM_STATE
                                  ) -> Tuple[List[str], dict]:
    """
    Feature selection con Random Forest importance.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_features: Numero feature da selezionare
        n_estimators: Alberi RF
        n_jobs: Core CPU
        random_state: Seed random
    
    Returns:
        Tuple (selected_features, feature_importances)
    """
    logger.info(f"Feature selection con Random Forest (n_estimators={n_estimators})...")
    
    if n_jobs is None:
        n_jobs = _n_cores
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight='balanced'
    )
    
    rf.fit(X_train, y_train)
    
    importances = dict(zip(X_train.columns, rf.feature_importances_))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    selected_features = [f for f, _ in sorted_features[:n_features]]
    
    logger.info(f"Selezionate {len(selected_features)} feature (top {n_features})")
    logger.info(f"Top 5: {selected_features[:5]}")
    
    return selected_features, importances


def apply_feature_selection(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Applica selezione feature."""
    missing = set(selected_features) - set(X.columns)
    if missing:
        raise KeyError(f"Feature mancanti in X: {missing}")
    
    return X[selected_features].copy()


# ==============================================================================
# SALVATAGGIO ARTIFACTS
# ==============================================================================

def save_artifacts(scaler,
                   selected_features: List[str],
                   feature_importances: dict,
                   scaler_columns: List[str] = None,
                   output_dir: Path = None,
                   statistical_info: dict = None) -> None:
    """
    Salva artifacts del feature engineering.
    
    IMPORTANTE:
    - scaler_columns deve contenere le colonne CHE LO SCALER SI ASPETTA
    - Se statistical preprocessing è abilitato, queste sono meno delle colonne originali
    """
    if output_dir is None:
        output_dir = get_project_root() / "artifacts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, output_dir / "scaler.pkl")
    logger.info(f"Salvato: scaler.pkl")
    
    with open(output_dir / "selected_features.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Salvato: selected_features.json")
    
    if scaler_columns is not None:
        scaler_columns_sorted = sorted(scaler_columns)
        
        with open(output_dir / "scaler_columns.json", 'w') as f:
            json.dump(scaler_columns_sorted, f, indent=2)
        logger.info(f"Salvato: scaler_columns.json ({len(scaler_columns_sorted)} colonne)")
        
        checksum = compute_column_checksum(scaler_columns_sorted)
        checksum_data = {
            'checksum': checksum,
            'n_columns': len(scaler_columns_sorted),
            'columns': scaler_columns_sorted[:10]
        }
        
        with open(output_dir / "column_checksum.json", 'w') as f:
            json.dump(checksum_data, f, indent=2)
        logger.info(f"✓ Salvato checksum colonne: {checksum}")
    
    sorted_importances = dict(sorted(feature_importances.items(),
                                     key=lambda x: x[1], 
                                     reverse=True))
    with open(output_dir / "feature_importances.json", 'w') as f:
        json.dump(sorted_importances, f, indent=2)
    logger.info(f"Salvato: feature_importances.json")
    
    if statistical_info is not None:
        with open(output_dir / "statistical_preprocessing_info.json", 'w') as f:
            json.dump(statistical_info, f, indent=2)
        logger.info(f"Salvato: statistical_preprocessing_info.json")


def load_artifacts(artifacts_dir: Path = None,
                   validate_columns: bool = True
                   ) -> Tuple:
    """Carica artifacts salvati."""
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
    
    scaler_columns = None
    scaler_columns_path = artifacts_dir / "scaler_columns.json"
    if scaler_columns_path.exists():
        with open(scaler_columns_path, 'r') as f:
            scaler_columns = json.load(f)
        
        if validate_columns:
            checksum_path = artifacts_dir / "column_checksum.json"
            if checksum_path.exists():
                with open(checksum_path, 'r') as f:
                    checksum_data = json.load(f)
                
                current_checksum = compute_column_checksum(scaler_columns)
                saved_checksum = checksum_data['checksum']
                
                if current_checksum != saved_checksum:
                    logger.warning(
                        f"⚠️  Checksum colonne non corrisponde!\n"
                        f"   Salvato: {saved_checksum}\n"
                        f"   Calcolato: {current_checksum}"
                    )
                else:
                    logger.info(f"✓ Checksum colonne verificato: {current_checksum}")
    
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
                            use_statistical: bool = True,
                            use_robust: bool = True,
                            variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
                            correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
                            n_jobs: int = None,
                            random_state: int = RANDOM_STATE
                            ) -> Tuple:
    """
    Pipeline completa feature engineering.
    
    DEFAULT v2:
    - use_statistical=True: Applica statistical preprocessing (DEFAULT)
    - use_robust=True: Usa RobustScaler (DEFAULT)
    
    Args:
        variance_threshold: Soglia per low-variance filter
        correlation_threshold: Soglia per correlation filter
    """
    feature_cols = get_feature_columns(train)
    logger.info(f"Feature iniziali: {len(feature_cols)}")
    
    X_train, y_train = prepare_xy(train, label_col, feature_cols)
    X_val, y_val = prepare_xy(val, label_col, feature_cols)
    X_test, y_test = prepare_xy(test, label_col, feature_cols)
    
    statistical_info = None
    
    # Statistical preprocessing (DEFAULT ON)
    if use_statistical:
        X_train, X_val, X_test, statistical_info = statistical_preprocessing(
            X_train, X_val, X_test,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold
        )
        logger.info(f"Dopo statistical: {X_train.shape[1]} feature")
    
    # FIX CRITICO: Usa le colonne CORRETTE (dopo statistical preprocessing)
    scaler_columns = X_train.columns.tolist()
    logger.info(f"✓ Scaler verrà fittato su {len(scaler_columns)} colonne")
    
    # Scaling (DEFAULT: RobustScaler)
    scaler = fit_scaler(X_train, use_robust=use_robust)
    
    X_train_scaled = transform_data(X_train, scaler)
    X_val_scaled = transform_data(X_val, scaler)
    X_test_scaled = transform_data(X_test, scaler)
    
    del X_train, X_val, X_test
    gc.collect()
    
    # Feature selection con Random Forest importance
    logger.info(f"Selezione feature con Random Forest importance")
    
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
    
    # Salva artifacts con colonne corrette
    save_artifacts(scaler, selected_features, importances, 
                   scaler_columns=scaler_columns,
                   statistical_info=statistical_info)
    
    logger.info(f"✓ Feature engineering completato")
    logger.info(f"✓ Artifacts salvati con {len(scaler_columns)} scaler_columns")
    
    return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Feature Engineering v2 per NIDS-ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v2 - DEFAULT:
  Statistical preprocessing: ON (usa --no-statistical per disabilitare)
  RobustScaler: ON (usa --no-robust per StandardScaler)

Esempi:
  # Default (CONSIGLIATO):
  python src/feature_engineering.py
  
  # Disabilita statistical:
  python src/feature_engineering.py --no-statistical
  
  # Configura soglie:
  python src/feature_engineering.py --variance-threshold 0.01 --correlation-threshold 0.90
        """
    )
    
    parser.add_argument('--n-features', type=int, default=DEFAULT_N_FEATURES,
                        help=f'Feature da selezionare (default: {DEFAULT_N_FEATURES})')
    parser.add_argument('--rf-estimators', type=int, default=DEFAULT_RF_ESTIMATORS,
                        help=f'Alberi RF per feature importance (default: {DEFAULT_RF_ESTIMATORS})')
    parser.add_argument('--label-col', type=str, default='Label_Binary',
                        help='Colonna target')
    
    # Statistical preprocessing (DEFAULT ON)
    parser.add_argument('--no-statistical', dest='use_statistical', action='store_false',
                        help='Disabilita statistical preprocessing')
    parser.add_argument('--variance-threshold', type=float, default=DEFAULT_VARIANCE_THRESHOLD,
                        help=f'Soglia varianza (default: {DEFAULT_VARIANCE_THRESHOLD})')
    parser.add_argument('--correlation-threshold', type=float, default=DEFAULT_CORRELATION_THRESHOLD,
                        help=f'Soglia correlazione (default: {DEFAULT_CORRELATION_THRESHOLD})')
    parser.set_defaults(use_statistical=True)
    
    # RobustScaler (DEFAULT ON)
    parser.add_argument('--no-robust', dest='use_robust', action='store_false',
                        help='Usa StandardScaler invece di RobustScaler')
    parser.set_defaults(use_robust=True)
    
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Core CPU (default: auto)')
    parser.add_argument('--max-ram', type=int, default=DEFAULT_MAX_RAM,
                        help='Limite RAM %%')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                        help='Seed random')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    n_jobs = args.n_jobs if args.n_jobs else _n_cores
    limiter = ResourceLimiter(n_cores=n_jobs, max_ram_percent=args.max_ram)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING v2")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Statistical preprocessing: {'ON (DEFAULT)' if args.use_statistical else 'OFF'}")
    if args.use_statistical:
        print(f"    - Variance threshold:    {args.variance_threshold}")
        print(f"    - Correlation threshold: {args.correlation_threshold}")
    print(f"  Scaler:                    {'RobustScaler (DEFAULT)' if args.use_robust else 'StandardScaler'}")
    print(f"  Metodo selezione:          Random Forest Importance")
    print(f"  Feature da selezionare:    {args.n_features}")
    print(f"  RF estimators:             {args.rf_estimators}")
    print(f"  CPU cores:                 {n_jobs}/{os.cpu_count()}")
    print()
    
    timer = TimingLogger("feature_engineering_v2", parameters={
        'use_statistical': args.use_statistical,
        'variance_threshold': args.variance_threshold,
        'correlation_threshold': args.correlation_threshold,
        'use_robust': args.use_robust,
        'n_features': args.n_features,
        'rf_estimators': args.rf_estimators,
        'n_jobs': n_jobs,
        'max_ram': args.max_ram
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
                use_statistical=args.use_statistical,
                use_robust=args.use_robust,
                variance_threshold=args.variance_threshold,
                correlation_threshold=args.correlation_threshold,
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
        _, selected_features, importances, scaler_columns = load_artifacts()
        for i, feat in enumerate(selected_features[:10]):
            print(f"   {i+1:2}. {feat}: {importances[feat]:.4f}")
        
        print(f"\n✓ Scaler columns salvate: {len(scaler_columns)}")
        
        timer.add_metric("train_samples", len(train))
        timer.add_metric("n_features_selected", len(selected_features))
        timer.add_metric("n_scaler_columns", len(scaler_columns))
        timing_path = timer.save()
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETATO")
        print("=" * 60)
        print(f"\nArtifacts: {get_project_root() / 'artifacts'}")
        print(f"Shape:     ({X_train.shape[0]:,}, {X_train.shape[1]})")
        print(f"Timing:    {timing_path}")
        
        print(f"\n✓ Configurazione v2 (DEFAULT):")
        if args.use_statistical:
            print(f"   - Statistical preprocessing: ON")
        if args.use_robust:
            print(f"   - RobustScaler: ON")
        
        print(f"\nProssimo step: python src/training/random_forest.py")
        
        timer.print_summary()
        limiter.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima: python src/preprocessing.py")
        sys.exit(1)


if __name__ == "__main__":
    main()