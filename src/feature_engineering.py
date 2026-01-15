"""
================================================================================
NIDS-ML - Modulo Feature Engineering
================================================================================

Scaling e selezione feature per preparare i dati al training.

PERCHE QUESTO MODULO:
---------------------
Il preprocessing pulisce i dati grezzi. Il feature engineering li prepara
per il machine learning:

1. SCALING (StandardScaler):
   - Normalizza feature a media=0, std=1
   - Necessario per algoritmi gradient-based
   - Migliora convergenza e stabilita numerica
   - IMPORTANTE: fit SOLO su train, transform su tutti i set

2. SELEZIONE FEATURE:
   - Riduce dimensionalita (77 -> 30 feature)
   - Elimina feature ridondanti o rumorose
   - Velocizza training e inference
   - Riduce rischio overfitting
   - Usa Random Forest per calcolare importanza

GUIDA PARAMETRI:
----------------
    python src/feature_engineering.py [opzioni]

Opzioni disponibili:
    --n-features INT      Numero feature da selezionare (default: 30)
    --rf-estimators INT   Alberi RF per selezione (default: 100)
    --label-col STR       Colonna target (default: Label_Binary)
    --max-ram INT         Limite RAM percentuale (default: 85)
    --max-cpu INT         Limite CPU percentuale (default: 85)
    --n-jobs INT          Core CPU da usare (default: -1 = auto)

ESEMPI:
-------
# Esecuzione standard
python src/feature_engineering.py

# Seleziona 50 feature invece di 30
python src/feature_engineering.py --n-features 50

# Limita risorse
python src/feature_engineering.py --max-ram 70 --n-jobs 4

# Per classificazione multiclasse
python src/feature_engineering.py --label-col Label_Multiclass

================================================================================
"""

import sys
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import joblib
import json
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.utils import (
    get_logger,
    get_project_root,
    RANDOM_STATE,
    LABEL_COLUMNS,
    ResourceMonitor,
    limit_cpu_cores,
    suppress_warnings
)
from src.preprocessing import load_processed_data

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CONFIGURAZIONE DEFAULT
# ==============================================================================

DEFAULT_N_FEATURES = 30          # Feature da selezionare
DEFAULT_RF_ESTIMATORS = 100      # Alberi per RF di selezione
DEFAULT_MAX_RAM = 85             # Limite RAM %
DEFAULT_MAX_CPU = 85             # Limite CPU %


# ==============================================================================
# PREPARAZIONE FEATURE
# ==============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Estrae i nomi delle colonne feature, escludendo le label.
    
    Le colonne label (Label, Label_Binary, ecc.) non sono feature
    predittive e devono essere escluse.
    
    Args:
        df: DataFrame con tutte le colonne
    
    Returns:
        Lista ordinata dei nomi delle colonne feature
    """
    feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    return sorted(feature_cols)


def prepare_xy(df: pd.DataFrame,
               label_col: str = 'Label_Binary',
               feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa feature (X) e target (y) da un DataFrame.
    
    Args:
        df: DataFrame completo con feature e label
        label_col: Nome colonna target
        feature_cols: Lista colonne feature (se None, deduce automaticamente)
    
    Returns:
        Tuple (X, y) dove:
        - X: DataFrame con sole feature
        - y: Series con target
    
    Raises:
        KeyError: Se label_col non presente nel DataFrame
    """
    if label_col not in df.columns:
        raise KeyError(f"Colonna target '{label_col}' non trovata. "
                       f"Disponibili: {list(df.columns)}")
    
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    
    return X, y


# ==============================================================================
# SCALING
# ==============================================================================

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Addestra lo StandardScaler sui dati di training.
    
    IMPORTANTE: Lo scaler deve essere fittato SOLO sui dati di training
    per evitare data leakage. Le statistiche (media, std) calcolate
    sul training vengono poi applicate a validation e test.
    
    StandardScaler trasforma ogni feature in modo che abbia:
    - Media = 0
    - Deviazione standard = 1
    
    Formula: z = (x - mean) / std
    
    Args:
        X_train: DataFrame con feature di training
    
    Returns:
        StandardScaler fittato
    """
    logger.info(f"Fitting scaler su {len(X_train):,} campioni, "
                f"{X_train.shape[1]} feature")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return scaler


def transform_data(X: pd.DataFrame, 
                   scaler: StandardScaler) -> pd.DataFrame:
    """
    Applica trasformazione dello scaler ai dati.
    
    Usa le statistiche calcolate durante fit() per trasformare i dati.
    Mantiene i nomi delle colonne e l'indice originale.
    
    Args:
        X: DataFrame con feature da scalare
        scaler: StandardScaler gia fittato
    
    Returns:
        DataFrame con feature scalate (stessa struttura di X)
    """
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# ==============================================================================
# SELEZIONE FEATURE
# ==============================================================================

def select_features_by_importance(X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  n_features: int = DEFAULT_N_FEATURES,
                                  n_estimators: int = DEFAULT_RF_ESTIMATORS,
                                  n_jobs: int = -1,
                                  random_state: int = RANDOM_STATE
                                  ) -> Tuple[List[str], dict]:
    """
    Seleziona le feature piu importanti usando Random Forest.
    
    Il processo:
    1. Addestra un Random Forest leggero sui dati
    2. Estrae feature_importances_ (basato su riduzione impurita)
    3. Ordina feature per importanza decrescente
    4. Seleziona le top N feature
    
    Perche Random Forest per selezione:
    - Robusto a feature correlate
    - Gestisce relazioni non lineari
    - Veloce da addestrare
    - Feature importance interpretabile
    
    Args:
        X_train: Feature di training (scalate)
        y_train: Target di training
        n_features: Numero feature da selezionare
        n_estimators: Numero alberi nel RF (piu = piu stabile ma piu lento)
        n_jobs: Core CPU (-1 = tutti disponibili)
        random_state: Seed per reproducibilita
    
    Returns:
        Tuple contenente:
        - Lista nomi feature selezionate (ordinate per importanza)
        - Dizionario {nome_feature: importanza} per tutte le feature
    """
    logger.info(f"Training RF per selezione feature "
                f"(n_estimators={n_estimators}, n_jobs={n_jobs})...")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,              # Limita profondita per velocita
        min_samples_split=10,      # Evita overfitting
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight='balanced',   # Gestisce sbilanciamento classi
        verbose=0
    )
    
    rf.fit(X_train, y_train)
    
    # Estrai importanze
    importances = rf.feature_importances_
    feature_names = X_train.columns.tolist()
    
    # Crea dizionario completo
    importance_dict = {
        name: float(imp) 
        for name, imp in zip(feature_names, importances)
    }
    
    # Ordina per importanza decrescente
    sorted_features = sorted(importance_dict.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
    
    # Seleziona top N
    selected_features = [name for name, _ in sorted_features[:n_features]]
    
    logger.info(f"Selezionate {n_features} feature su {len(feature_names)}")
    
    # Log top 10
    logger.info("Top 10 feature per importanza:")
    for i, (name, imp) in enumerate(sorted_features[:10]):
        logger.info(f"  {i+1:2}. {name}: {imp:.4f}")
    
    # Libera memoria
    del rf
    gc.collect()
    
    return selected_features, importance_dict


def apply_feature_selection(X: pd.DataFrame,
                            selected_features: List[str]) -> pd.DataFrame:
    """
    Filtra un DataFrame mantenendo solo le feature selezionate.
    
    Args:
        X: DataFrame con tutte le feature
        selected_features: Lista feature da mantenere
    
    Returns:
        DataFrame con sole feature selezionate
    
    Raises:
        KeyError: Se alcune feature selezionate non sono presenti in X
    """
    missing = set(selected_features) - set(X.columns)
    if missing:
        raise KeyError(f"Feature mancanti nel DataFrame: {missing}")
    
    return X[selected_features].copy()


# ==============================================================================
# SALVATAGGIO E CARICAMENTO ARTIFACTS
# ==============================================================================

def save_artifacts(scaler: StandardScaler,
                   selected_features: List[str],
                   feature_importances: dict,
                   output_dir: Path = None) -> None:
    """
    Salva tutti gli artifacts del feature engineering.
    
    Artifacts salvati:
    - scaler.pkl: StandardScaler fittato (per trasformare nuovi dati)
    - selected_features.json: Lista feature selezionate (ordine importante)
    - feature_importances.json: Importanza di tutte le feature
    
    Args:
        scaler: StandardScaler fittato
        selected_features: Lista feature selezionate
        feature_importances: Dizionario importanze
        output_dir: Directory output (default: artifacts/)
    """
    if output_dir is None:
        output_dir = get_project_root() / "artifacts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva scaler
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Salvato: {scaler_path.name}")
    
    # Salva lista feature selezionate
    features_path = output_dir / "selected_features.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Salvato: {features_path.name}")
    
    # Salva importanze (ordinate)
    sorted_importances = dict(sorted(feature_importances.items(),
                                     key=lambda x: x[1], 
                                     reverse=True))
    importances_path = output_dir / "feature_importances.json"
    with open(importances_path, 'w') as f:
        json.dump(sorted_importances, f, indent=2)
    logger.info(f"Salvato: {importances_path.name}")


def load_artifacts(artifacts_dir: Path = None
                   ) -> Tuple[StandardScaler, List[str], dict]:
    """
    Carica artifacts salvati per inference o training successivi.
    
    Args:
        artifacts_dir: Directory con gli artifacts
    
    Returns:
        Tuple (scaler, selected_features, importances_dict)
    
    Raises:
        FileNotFoundError: Se artifacts non trovati
    """
    if artifacts_dir is None:
        artifacts_dir = get_project_root() / "artifacts"
    
    # Verifica esistenza
    required = ['scaler.pkl', 'selected_features.json', 'feature_importances.json']
    for f in required:
        if not (artifacts_dir / f).exists():
            raise FileNotFoundError(f"Artifact non trovato: {artifacts_dir / f}")
    
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    
    with open(artifacts_dir / "selected_features.json", 'r') as f:
        selected_features = json.load(f)
    
    with open(artifacts_dir / "feature_importances.json", 'r') as f:
        importances = json.load(f)
    
    logger.info(f"Caricati artifacts da {artifacts_dir}")
    logger.info(f"Feature selezionate: {len(selected_features)}")
    
    return scaler, selected_features, importances


# ==============================================================================
# PIPELINE COMPLETA
# ==============================================================================

def run_feature_engineering(train: pd.DataFrame,
                            val: pd.DataFrame,
                            test: pd.DataFrame,
                            label_col: str = 'Label_Binary',
                            n_features: int = DEFAULT_N_FEATURES,
                            n_estimators: int = DEFAULT_RF_ESTIMATORS,
                            n_jobs: int = -1,
                            random_state: int = RANDOM_STATE
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                       pd.Series, pd.Series, pd.Series]:
    """
    Esegue pipeline completa di feature engineering.
    
    Passi:
    1. Separazione X/y per ogni split
    2. Fit scaler su training (SOLO training)
    3. Transform su tutti i set
    4. Selezione feature basata su importanza
    5. Applicazione selezione a tutti i set
    6. Salvataggio artifacts
    
    Args:
        train: Training DataFrame
        val: Validation DataFrame
        test: Test DataFrame
        label_col: Colonna target
        n_features: Numero feature da selezionare
        n_estimators: Alberi per RF selezione
        n_jobs: Core CPU da usare
        random_state: Seed per reproducibilita
    
    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        con feature scalate e selezionate
    """
    # 1. Identifica colonne feature
    feature_cols = get_feature_columns(train)
    logger.info(f"Feature iniziali: {len(feature_cols)}")
    
    # 2. Separa X e y
    X_train, y_train = prepare_xy(train, label_col, feature_cols)
    X_val, y_val = prepare_xy(val, label_col, feature_cols)
    X_test, y_test = prepare_xy(test, label_col, feature_cols)
    
    # 3. Fit scaler solo su train
    scaler = fit_scaler(X_train)
    
    # 4. Transform tutti i set
    X_train_scaled = transform_data(X_train, scaler)
    X_val_scaled = transform_data(X_val, scaler)
    X_test_scaled = transform_data(X_test, scaler)
    
    # Libera memoria originali
    del X_train, X_val, X_test
    gc.collect()
    
    # 5. Selezione feature
    selected_features, importances = select_features_by_importance(
        X_train_scaled, y_train,
        n_features=n_features,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    # 6. Applica selezione
    X_train_final = apply_feature_selection(X_train_scaled, selected_features)
    X_val_final = apply_feature_selection(X_val_scaled, selected_features)
    X_test_final = apply_feature_selection(X_test_scaled, selected_features)
    
    # Libera memoria scalati completi
    del X_train_scaled, X_val_scaled, X_test_scaled
    gc.collect()
    
    # 7. Salva artifacts
    save_artifacts(scaler, selected_features, importances)
    
    logger.info(f"Feature engineering completato")
    logger.info(f"Shape finale: train={X_train_final.shape}, "
                f"val={X_val_final.shape}, test={X_test_final.shape}")
    
    return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    """Parse argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description='Feature Engineering per NIDS-ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python src/feature_engineering.py
  python src/feature_engineering.py --n-features 50
  python src/feature_engineering.py --max-ram 70 --n-jobs 4
        """
    )
    
    parser.add_argument(
        '--n-features',
        type=int,
        default=DEFAULT_N_FEATURES,
        help=f'Numero feature da selezionare (default: {DEFAULT_N_FEATURES})'
    )
    parser.add_argument(
        '--rf-estimators',
        type=int,
        default=DEFAULT_RF_ESTIMATORS,
        help=f'Alberi RF per selezione (default: {DEFAULT_RF_ESTIMATORS})'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='Label_Binary',
        help='Colonna target (default: Label_Binary)'
    )
    parser.add_argument(
        '--max-ram',
        type=int,
        default=DEFAULT_MAX_RAM,
        help=f'Limite RAM percentuale (default: {DEFAULT_MAX_RAM})'
    )
    parser.add_argument(
        '--max-cpu',
        type=int,
        default=DEFAULT_MAX_CPU,
        help=f'Limite CPU percentuale (default: {DEFAULT_MAX_CPU})'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Core CPU da usare (-1 = auto, lascia 1 libero)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=RANDOM_STATE,
        help=f'Seed random (default: {RANDOM_STATE})'
    )
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funzione principale per esecuzione da linea di comando."""
    args = parse_arguments()
    
    # Configura numero core
    if args.n_jobs == -1:
        n_jobs = limit_cpu_cores()
    else:
        n_jobs = args.n_jobs
    
    # Monitor risorse
    monitor = ResourceMonitor(max_cpu=args.max_cpu, max_ram=args.max_ram)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"\nParametri:")
    print(f"  Feature da selezionare: {args.n_features}")
    print(f"  RF estimators:          {args.rf_estimators}")
    print(f"  Label column:           {args.label_col}")
    print(f"  CPU cores:              {n_jobs}")
    print(f"  Max RAM:                {args.max_ram}%")
    print(f"  Random state:           {args.random_state}")
    print()
    
    try:
        # 1. Carica dati
        print("1. Caricamento dati preprocessati...")
        train, val, test, mappings = load_processed_data()
        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        
        # Verifica risorse
        monitor.log_status(logger)
        
        # 2. Esegui pipeline
        print("\n2. Esecuzione pipeline feature engineering...")
        X_train, X_val, X_test, y_train, y_val, y_test = run_feature_engineering(
            train, val, test,
            label_col=args.label_col,
            n_features=args.n_features,
            n_estimators=args.rf_estimators,
            n_jobs=n_jobs,
            random_state=args.random_state
        )
        
        # 3. Salva dataset pronti
        print("\n3. Salvataggio dataset pronti per training...")
        processed_dir = get_project_root() / "data" / "processed"
        
        # Combina X e y in singoli DataFrame
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
        
        # 4. Report feature selezionate
        print("\n   Top 10 feature selezionate:")
        _, selected_features, importances = load_artifacts()
        for i, feat in enumerate(selected_features[:10]):
            print(f"   {i+1:2}. {feat}: {importances[feat]:.4f}")
        
        # Report finale
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETATO")
        print("=" * 60)
        print(f"\nArtifacts: {get_project_root() / 'artifacts'}")
        print(f"Dataset:   {processed_dir}")
        print(f"Shape:     ({X_train.shape[0]:,}, {X_train.shape[1]})")
        print(f"\nProssimo step: python src/training/random_forest.py")
        
        # Log risorse finali
        monitor.log_status(logger)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        print("Eseguire prima: python src/preprocessing.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()