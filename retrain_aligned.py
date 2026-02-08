#!/usr/bin/env python3
"""
NIDS-ML Retraining Script - Pipeline Allineata allo Sniffer

Questa pipeline è IDENTICA a quella dello sniffer:
1. Raw CSV → DataFrame
2. RobustScaler (fit su raw data)
3. Clip(-10, 10)
4. Train LightGBM

Nessun statistical preprocessing = nessun mismatch!

Esegui: python retrain_aligned.py
"""

import gc
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, accuracy_score
)

# Richiede: pip install lightgbm
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: lightgbm not installed. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLIP_VALUE = 10.0

# Feature da usare (le 44 dopo statistical preprocessing - per compatibilità)
# Se vuoi usare tutte le 77, cambia questa lista
FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Bwd Packets/s",
    "Min Packet Length",
    "Packet Length Mean",
    "Packet Length Variance",
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Std"
]

# LightGBM hyperparameters (ottimizzati per NIDS)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    # Per gestire imbalance
    'is_unbalance': True,
}

# ==============================================================================
# UTILITIES
# ==============================================================================

def normalize_name(name: str) -> str:
    return name.strip().lower()


def load_all_csvs(data_dir: str = "data/raw") -> pd.DataFrame:
    """Carica tutti i CSV e li concatena."""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_dfs = []
    for csv_path in csv_files:
        print(f"  Loading {csv_path.name}...")
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Strip column names (CIC-IDS2017 ha spazi iniziali)
        df.columns = df.columns.str.strip()
        
        all_dfs.append(df)
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows: {len(df_all):,}")
    
    return df_all


def extract_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Estrae feature nell'ordine specificato."""
    csv_cols_norm = {normalize_name(c): c for c in df.columns}
    
    X = np.zeros((len(df), len(feature_cols)), dtype=np.float64)
    matched = 0
    
    for i, feat in enumerate(feature_cols):
        feat_norm = normalize_name(feat)
        
        csv_col = None
        if feat_norm in csv_cols_norm:
            csv_col = csv_cols_norm[feat_norm]
        else:
            for var in [feat_norm.replace(' ', '_'), feat_norm.replace('_', ' ')]:
                if var in csv_cols_norm:
                    csv_col = csv_cols_norm[var]
                    break
        
        if csv_col and csv_col in df.columns:
            X[:, i] = df[csv_col].values
            matched += 1
    
    print(f"Features matched: {matched}/{len(feature_cols)}")
    return X


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    """Estrae label binarie (0=BENIGN, 1=ATTACK)."""
    # Trova colonna label
    label_col = None
    for col in df.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break
    
    if not label_col:
        raise ValueError("Label column not found")
    
    labels = df[label_col].astype(str).str.strip().str.upper()
    y = (labels != 'BENIGN').astype(int).values
    
    print(f"Labels: {(y==0).sum():,} BENIGN, {(y==1).sum():,} ATTACK")
    return y


def preprocess(X: np.ndarray, scaler: RobustScaler = None, fit: bool = False, clip: float = CLIP_VALUE):
    """
    Pipeline di preprocessing IDENTICA allo sniffer.
    
    1. Handle inf/nan
    2. Scale (fit or transform)
    3. Clip
    """
    # 1. Handle inf/nan
    X = np.where(np.isinf(X), 0, X)
    X = np.where(np.isnan(X), 0, X)
    
    # 2. Scale
    if fit:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # 3. Clip
    X_clipped = np.clip(X_scaled, -clip, clip)
    
    return X_clipped, scaler


def select_features_by_importance(model, feature_names: list, n_features: int = 30) -> tuple:
    """Seleziona top N feature per importanza."""
    importances = model.feature_importance(importance_type='gain')
    indices = np.argsort(importances)[::-1][:n_features]
    
    selected_features = [feature_names[i] for i in indices]
    selected_indices = indices.tolist()
    
    return selected_features, selected_indices, dict(zip(feature_names, importances.tolist()))


# ==============================================================================
# MAIN TRAINING
# ==============================================================================

def main():
    print("=" * 70)
    print("NIDS-ML RETRAINING - ALIGNED PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Clip value: {CLIP_VALUE}")
    
    if not LGBM_AVAILABLE:
        print("\nERROR: LightGBM not installed!")
        print("Install with: pip install lightgbm")
        return
    
    # Output directories
    output_dir = Path("artifacts_new")
    model_dir = Path("models/retrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("\n" + "=" * 50)
    print("[1] LOADING DATA")
    print("=" * 50)
    
    df = load_all_csvs("data/raw")
    
    # 2. Extract features and labels
    print("\n" + "=" * 50)
    print("[2] EXTRACTING FEATURES")
    print("=" * 50)
    
    X = extract_features(df, FEATURE_COLUMNS)
    y = extract_labels(df)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Free memory
    del df
    gc.collect()
    
    # 3. Train/test split
    print("\n" + "=" * 50)
    print("[3] TRAIN/TEST SPLIT")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    print(f"Train attack ratio: {y_train.mean():.2%}")
    print(f"Test attack ratio:  {y_test.mean():.2%}")
    
    # Free memory
    del X, y
    gc.collect()
    
    # 4. Preprocess (FIT on train only!)
    print("\n" + "=" * 50)
    print("[4] PREPROCESSING")
    print("=" * 50)
    
    X_train_processed, scaler = preprocess(X_train, fit=True, clip=CLIP_VALUE)
    X_test_processed, _ = preprocess(X_test, scaler=scaler, fit=False, clip=CLIP_VALUE)
    
    print(f"Train scaled stats: min={X_train_processed.min():.4f}, max={X_train_processed.max():.4f}")
    print(f"Test scaled stats:  min={X_test_processed.min():.4f}, max={X_test_processed.max():.4f}")
    
    # 5. Train LightGBM
    print("\n" + "=" * 50)
    print("[5] TRAINING LIGHTGBM")
    print("=" * 50)
    
    start_time = time.time()
    
    train_data = lgb.Dataset(X_train_processed, label=y_train)
    valid_data = lgb.Dataset(X_test_processed, label=y_test, reference=train_data)
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.1f}s")
    print(f"Best iteration: {model.best_iteration}")
    
    # 6. Evaluate
    print("\n" + "=" * 50)
    print("[6] EVALUATION")
    print("=" * 50)
    
    y_pred = (model.predict(X_test_processed) > 0.5).astype(int)
    y_prob = model.predict(X_test_processed)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'training_time_seconds': training_time,
        'best_iteration': model.best_iteration
    }
    
    print(f"\nF1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Probability distribution
    print("\nProbability distribution (attacks):")
    attack_probs = y_prob[y_test == 1]
    print(f"  Min: {attack_probs.min():.4f}")
    print(f"  Max: {attack_probs.max():.4f}")
    print(f"  Mean: {attack_probs.mean():.4f}")
    print(f"  Median: {np.median(attack_probs):.4f}")
    
    # 7. Feature selection
    print("\n" + "=" * 50)
    print("[7] FEATURE SELECTION")
    print("=" * 50)
    
    selected_features, selected_indices, importances = select_features_by_importance(
        model, FEATURE_COLUMNS, n_features=30
    )
    
    print("Top 10 features by importance:")
    for i, feat in enumerate(selected_features[:10]):
        print(f"  {i+1}. {feat}: {importances[feat]:.4f}")
    
    # 8. Save artifacts
    print("\n" + "=" * 50)
    print("[8] SAVING ARTIFACTS")
    print("=" * 50)
    
    # Save scaler
    joblib.dump(scaler, output_dir / "scaler.pkl")
    print(f"Saved: {output_dir}/scaler.pkl")
    
    # Save scaler columns
    with open(output_dir / "scaler_columns.json", 'w') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"Saved: {output_dir}/scaler_columns.json")
    
    # Save selected features
    with open(output_dir / "selected_features.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"Saved: {output_dir}/selected_features.json")
    
    # Save feature importances
    with open(output_dir / "feature_importances.json", 'w') as f:
        json.dump(importances, f, indent=2)
    print(f"Saved: {output_dir}/feature_importances.json")
    
    # Save model
    model.save_model(str(model_dir / "model_binary.txt"))
    print(f"Saved: {model_dir}/model_binary.txt (LightGBM native)")
    
    # Also save as sklearn-compatible
    # LightGBM models need to be wrapped for sklearn compatibility
    joblib.dump(model, model_dir / "model_binary.pkl")
    print(f"Saved: {model_dir}/model_binary.pkl (joblib)")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'raw -> RobustScaler -> clip(-10,10) -> LightGBM',
        'clip_value': CLIP_VALUE,
        'n_features': len(FEATURE_COLUMNS),
        'n_selected_features': len(selected_features),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'metrics': metrics,
        'lgbm_params': LGBM_PARAMS
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {model_dir}/metadata.json")
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
Results:
  F1 Score:  {metrics['f1']:.4f}
  Precision: {metrics['precision']:.4f}
  Recall:    {metrics['recall']:.4f}
  Accuracy:  {metrics['accuracy']:.4f}

Artifacts saved to:
  {output_dir}/
    - scaler.pkl
    - scaler_columns.json
    - selected_features.json
    - feature_importances.json

Model saved to:
  {model_dir}/
    - model_binary.pkl
    - model_binary.txt
    - metadata.json

Next steps:
  1. Copy artifacts_new/* to artifacts/
  2. Copy models/retrained/* to models/best_model/
  3. Run fix_and_test.py to verify
  4. Run evaluation: python -m src.sniffer.main evaluate-all
""")


if __name__ == "__main__":
    main()
