#!/usr/bin/env python3
"""
NIDS-ML Diagnostic Script - Trova la causa esatta del problema F1=0

Questo script analizza passo-passo la pipeline per identificare dove si rompe.
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def main():
    print("=" * 70)
    print("NIDS-ML DIAGNOSTIC SCRIPT")
    print("=" * 70)
    
    # 1. Carica artifacts
    print("\n[1] LOADING ARTIFACTS")
    print("-" * 50)
    
    artifacts_dir = Path("artifacts")
    model_dir = Path("models/best_model")
    
    # Scaler
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    print(f"Scaler: {type(scaler).__name__}")
    
    # Scaler columns
    with open(artifacts_dir / "scaler_columns.json") as f:
        scaler_columns = json.load(f)
    print(f"Scaler columns: {len(scaler_columns)}")
    print(f"  First 5: {scaler_columns[:5]}")
    
    # Selected features
    with open(artifacts_dir / "selected_features.json") as f:
        selected_features = json.load(f)
    print(f"Selected features: {len(selected_features)}")
    print(f"  First 5: {selected_features[:5]}")
    
    # Model
    model_path = model_dir / "model_binary.pkl"
    if not model_path.exists():
        model_path = model_dir / "model.pkl"
    model = joblib.load(model_path)
    print(f"Model: {type(model).__name__}")
    
    # 2. Calcola indici
    print("\n[2] COMPUTING INDICES")
    print("-" * 50)
    
    scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(scaler_columns)}
    selected_indices = []
    
    for feat in selected_features:
        feat_lower = feat.strip().lower()
        if feat_lower in scaler_cols_lower:
            idx = scaler_cols_lower[feat_lower]
            selected_indices.append(idx)
            print(f"  {feat} -> index {idx}")
        else:
            print(f"  {feat} -> NOT FOUND!")
    
    print(f"\nIndices: {selected_indices}")
    print(f"Total mapped: {len(selected_indices)}/{len(selected_features)}")
    
    # 3. Carica CSV di test (DDoS)
    print("\n[3] LOADING TEST CSV (Friday DDoS)")
    print("-" * 50)
    
    csv_path = "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = pd.read_csv(csv_path, nrows=10000)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    print(f"Rows: {len(df)}")
    print(f"Columns (first 10): {list(df.columns)[:10]}")
    
    # Check label distribution
    label_col = 'Label'
    print(f"\nLabel distribution:")
    print(df[label_col].value_counts())
    
    # 4. Estrai feature nell'ordine corretto
    print("\n[4] EXTRACTING FEATURES IN CORRECT ORDER")
    print("-" * 50)
    
    # Mapping CSV columns to scaler_columns
    csv_cols_lower = {col.strip().lower(): col for col in df.columns}
    
    features_data = []
    matched = 0
    missing = 0
    
    for scaler_col in scaler_columns:
        scaler_col_lower = scaler_col.strip().lower()
        
        if scaler_col_lower in csv_cols_lower:
            csv_col = csv_cols_lower[scaler_col_lower]
            features_data.append(df[csv_col].values)
            matched += 1
        else:
            # Try variants
            variants = [
                scaler_col_lower.replace(' ', '_'),
                scaler_col_lower.replace('_', ' ')
            ]
            found = False
            for var in variants:
                if var in csv_cols_lower:
                    csv_col = csv_cols_lower[var]
                    features_data.append(df[csv_col].values)
                    matched += 1
                    found = True
                    break
            
            if not found:
                features_data.append(np.zeros(len(df)))
                missing += 1
                print(f"  MISSING: {scaler_col}")
    
    print(f"\nMatched: {matched}, Missing: {missing}")
    
    # Create feature matrix
    X_raw = np.column_stack(features_data)
    print(f"X_raw shape: {X_raw.shape}")
    
    # 5. Clean data
    print("\n[5] CLEANING DATA")
    print("-" * 50)
    
    # Replace inf/nan
    X_clean = np.where(np.isinf(X_raw), 0, X_raw)
    X_clean = np.where(np.isnan(X_clean), 0, X_clean)
    
    print(f"Inf count before: {np.isinf(X_raw).sum()}")
    print(f"NaN count before: {np.isnan(X_raw).sum()}")
    print(f"Inf count after: {np.isinf(X_clean).sum()}")
    print(f"NaN count after: {np.isnan(X_clean).sum()}")
    
    # 6. Scale
    print("\n[6] SCALING")
    print("-" * 50)
    
    X_scaled = scaler.transform(X_clean)
    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"X_scaled stats: min={X_scaled.min():.2f}, max={X_scaled.max():.2f}, mean={X_scaled.mean():.2f}")
    
    # Check for extreme values
    extreme_mask = np.abs(X_scaled) > 10
    extreme_count = extreme_mask.sum()
    print(f"Extreme values (|x| > 10): {extreme_count}")
    
    if extreme_count > 0:
        # Find which features have extreme values
        extreme_cols = np.where(extreme_mask.any(axis=0))[0]
        print(f"Columns with extreme values: {extreme_cols[:10]}")
        for col_idx in extreme_cols[:5]:
            col_name = scaler_columns[col_idx]
            col_max = np.abs(X_scaled[:, col_idx]).max()
            print(f"  {col_name}: max |value| = {col_max:.2f}")
    
    # 7. Select features
    print("\n[7] SELECTING FEATURES")
    print("-" * 50)
    
    X_selected = X_scaled[:, selected_indices]
    print(f"X_selected shape: {X_selected.shape}")
    print(f"X_selected stats: min={X_selected.min():.2f}, max={X_selected.max():.2f}")
    
    # 8. Predict
    print("\n[8] PREDICTING")
    print("-" * 50)
    
    # Get true labels
    y_true = (df[label_col] != 'BENIGN').astype(int).values
    print(f"True attacks: {y_true.sum()}/{len(y_true)}")
    
    # Predict
    y_pred = model.predict(X_selected)
    print(f"Predicted attacks: {y_pred.sum()}/{len(y_pred)}")
    
    # Probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_selected)[:, 1]
        print(f"Prob stats: min={y_prob.min():.4f}, max={y_prob.max():.4f}, mean={y_prob.mean():.4f}")
        
        # Distribution of probabilities
        print(f"\nProbability distribution:")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            count = (y_prob >= threshold).sum()
            print(f"  prob >= {threshold}: {count} ({count/len(y_prob)*100:.1f}%)")
    
    # 9. Calculate metrics
    print("\n[9] METRICS")
    print("-" * 50)
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    # 10. Sample analysis - look at specific rows
    print("\n[10] SAMPLE ANALYSIS")
    print("-" * 50)
    
    # Find some attack rows
    attack_indices = np.where(y_true == 1)[0][:5]
    benign_indices = np.where(y_true == 0)[0][:5]
    
    print("Attack samples:")
    for idx in attack_indices:
        prob = y_prob[idx] if hasattr(model, 'predict_proba') else -1
        pred = y_pred[idx]
        print(f"  idx={idx}: pred={pred}, prob={prob:.4f}, true=ATTACK")
    
    print("\nBenign samples:")
    for idx in benign_indices:
        prob = y_prob[idx] if hasattr(model, 'predict_proba') else -1
        pred = y_pred[idx]
        print(f"  idx={idx}: pred={pred}, prob={prob:.4f}, true=BENIGN")
    
    # 11. Check a specific attack row in detail
    print("\n[11] DETAILED ATTACK ROW ANALYSIS")
    print("-" * 50)
    
    if len(attack_indices) > 0:
        idx = attack_indices[0]
        print(f"Analyzing row {idx}:")
        
        # Raw values for selected features
        print("\nSelected features raw vs scaled:")
        for i, feat_idx in enumerate(selected_indices[:10]):
            feat_name = selected_features[i]
            raw_val = X_clean[idx, feat_idx]
            scaled_val = X_scaled[idx, feat_idx]
            selected_val = X_selected[idx, i]
            print(f"  {feat_name}: raw={raw_val:.2f}, scaled={scaled_val:.4f}, selected={selected_val:.4f}")
    
    # 12. Compare with training data distribution
    print("\n[12] CONCLUSION")
    print("-" * 50)
    
    if f1 < 0.1:
        print("PROBLEM DETECTED: F1 is very low!")
        print("\nPossible causes:")
        
        if y_pred.sum() < 10:
            print("  1. Model predicts almost everything as BENIGN")
            print("     -> Check if scaled values are in expected range")
        
        if extreme_count > len(df) * 0.1:
            print("  2. Many extreme values after scaling")
            print("     -> Scaling may not match training distribution")
        
        if missing > 0:
            print(f"  3. {missing} features are missing (set to 0)")
            print("     -> Check column name matching")
    else:
        print(f"Model seems to work! F1={f1:.4f}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()