#!/usr/bin/env python3
"""
NIDS-ML Artifact Regenerator

Rigenera gli artifacts (scaler.pkl) con la versione corrente di sklearn.
Questo risolve il problema di version mismatch.

ATTENZIONE: Esegui SOLO se vuoi rigenerare lo scaler.
Il modello NON viene toccato.

Esegui: python regenerate_scaler.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')


def normalize_name(name: str) -> str:
    return name.strip().lower()


def main():
    print("=" * 70)
    print("NIDS-ML SCALER REGENERATOR")
    print("=" * 70)
    
    artifacts_dir = Path("artifacts")
    data_dir = Path("data/raw")
    
    # 1. Load existing scaler_columns
    print("\n[1] Loading scaler_columns.json...")
    
    with open(artifacts_dir / "scaler_columns.json") as f:
        scaler_columns = json.load(f)
    print(f"Scaler columns: {len(scaler_columns)}")
    print(f"First 5: {scaler_columns[:5]}")
    
    # 2. Load training data (tutti i CSV)
    print("\n[2] Loading training data...")
    
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    for csv_path in csv_files:
        print(f"  Loading {csv_path.name}...")
        df = pd.read_csv(csv_path, low_memory=False)
        all_data.append(df)
    
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"Total rows: {len(df_all):,}")
    
    # 3. Extract features in correct order
    print("\n[3] Extracting features in scaler_columns order...")
    
    csv_cols_norm = {normalize_name(c): c for c in df_all.columns}
    
    features_data = {}
    matched = 0
    
    for scaler_col in scaler_columns:
        scaler_col_norm = normalize_name(scaler_col)
        
        csv_col = None
        if scaler_col_norm in csv_cols_norm:
            csv_col = csv_cols_norm[scaler_col_norm]
        else:
            for var in [scaler_col_norm.replace(' ', '_'), scaler_col_norm.replace('_', ' ')]:
                if var in csv_cols_norm:
                    csv_col = csv_cols_norm[var]
                    break
        
        if csv_col and csv_col in df_all.columns:
            features_data[scaler_col] = df_all[csv_col].values
            matched += 1
        else:
            print(f"  WARNING: '{scaler_col}' not found, using zeros")
            features_data[scaler_col] = np.zeros(len(df_all))
    
    print(f"Matched: {matched}/{len(scaler_columns)}")
    
    # Create DataFrame with correct column order
    X = pd.DataFrame(features_data)[scaler_columns]
    
    # 4. Clean data
    print("\n[4] Cleaning data...")
    
    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"Shape: {X.shape}")
    print(f"NaN count: {X.isna().sum().sum()}")
    
    # 5. Fit new scaler
    print("\n[5] Fitting new RobustScaler...")
    
    scaler = RobustScaler()
    scaler.fit(X)
    
    print(f"Scaler fitted on {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # 6. Verify scaling
    print("\n[6] Verifying scaling...")
    
    X_scaled = scaler.transform(X.head(10000))
    print(f"Scaled stats (sample): min={X_scaled.min():.4f}, max={X_scaled.max():.4f}, mean={X_scaled.mean():.4f}")
    
    extreme = np.abs(X_scaled) > 10
    extreme_pct = extreme.sum() / X_scaled.size * 100
    print(f"Extreme values (|x| > 10): {extreme_pct:.2f}%")
    
    # 7. Save new scaler
    print("\n[7] Saving new scaler...")
    
    # Backup old
    old_scaler_path = artifacts_dir / "scaler.pkl"
    if old_scaler_path.exists():
        backup_path = artifacts_dir / "scaler_backup.pkl"
        import shutil
        shutil.copy(old_scaler_path, backup_path)
        print(f"Backed up old scaler to {backup_path}")
    
    # Save new
    joblib.dump(scaler, old_scaler_path)
    print(f"Saved new scaler to {old_scaler_path}")
    
    # 8. Verify
    print("\n[8] Verification...")
    
    scaler_reloaded = joblib.load(old_scaler_path)
    X_test = scaler_reloaded.transform(X.head(100))
    print(f"Reloaded scaler works: shape={X_test.shape}")
    
    print("\n" + "=" * 70)
    print("SCALER REGENERATION COMPLETE")
    print("=" * 70)
    print("""
Next steps:
1. Run fix_and_test.py again to verify improvement
2. If still not working, you may need to retrain the model

Note: The model was trained with the OLD scaler distribution.
If results are still poor, the model needs retraining.
""")


if __name__ == "__main__":
    main()
