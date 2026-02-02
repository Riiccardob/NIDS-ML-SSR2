#!/usr/bin/env python3
"""
NIDS-ML Prediction Analysis - Vedi cosa predice il modello SENZA guardare le label
Versione aggiornata: Rileva automaticamente tutti i CSV in data/raw
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from glob import glob

warnings.filterwarnings('ignore')

CLIP_VALUE = 10.0


def normalize_name(name):
    return name.strip().lower()


def main():
    print("=" * 70)
    print("NIDS-ML PREDICTION ANALYSIS (Label-Blind)")
    print("=" * 70)
    
    # Configuration
    artifacts_dir = Path("artifacts")
    model_dir = Path("models/best_model")
    data_dir = Path("data/raw")
    
    # -------------------------------------------------------------------------
    # FIX: Rilevamento automatico di tutti i CSV
    # -------------------------------------------------------------------------
    print(f"Cercando file CSV in: {data_dir} ...")
    found_files = sorted(glob(str(data_dir / "*.csv")))
    
    if not found_files:
        print("ERRORE: Nessun file .csv trovato nella cartella data/raw!")
        print("Assicurati che i file siano lì.")
        return

    csv_files = {}
    print("\nSeleziona CSV da analizzare:")
    
    for i, fpath in enumerate(found_files, 1):
        filename = os.path.basename(fpath)
        # Creiamo una descrizione semplice basata sul nome del file
        desc = filename.replace('.pcap_ISCX.csv', '').replace('-WorkingHours', '').replace('-workingHours', '')
        csv_files[str(i)] = (fpath, desc)
        print(f"  {i}. {desc} ({filename})")
    
    choice = input(f"\nScelta (1-{len(csv_files)}): ").strip()
    
    if choice not in csv_files:
        print("Scelta non valida, uscita.")
        return
    
    csv_path, csv_desc = csv_files[choice]
    print(f"\nAnalizzando: {csv_desc}")
    print(f"File: {csv_path}")
    
    # -------------------------------------------------------------------------
    # Load artifacts
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Loading artifacts...")
    
    try:
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
        with open(artifacts_dir / "scaler_columns.json") as f:
            scaler_columns = json.load(f)
            
        model_path = model_dir / "model_binary.pkl"
        if not model_path.exists():
            model_path = model_dir / "model.pkl"
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Errore caricamento modelli/artifacts: {e}")
        return
    
    is_booster = hasattr(model, 'predict') and not hasattr(model, 'predict_proba')
    print(f"Model: {type(model).__name__}, Booster: {is_booster}")
    
    # Load CSV
    print("\n" + "-" * 50)
    print("Loading CSV...")
    
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding='latin-1') # Encoding fix per sicurezza
        df.columns = df.columns.str.strip()
        print(f"Rows: {len(df):,}")
    except Exception as e:
        print(f"Errore lettura CSV: {e}")
        return
    
    # Extract features (WITHOUT looking at labels)
    print("\n" + "-" * 50)
    print("Extracting features (label-blind)...")
    
    csv_cols_norm = {normalize_name(c): c for c in df.columns}
    
    X = np.zeros((len(df), len(scaler_columns)), dtype=np.float64)
    missing_cols = []
    
    for i, col in enumerate(scaler_columns):
        col_norm = normalize_name(col)
        csv_col = csv_cols_norm.get(col_norm)
        if csv_col and csv_col in df.columns:
            # Gestione valori non numerici se necessario
            val = df[csv_col]
            # Converti in numeric, forzando errori a NaN
            val_numeric = pd.to_numeric(val, errors='coerce').fillna(0).values
            X[:, i] = val_numeric
        else:
            missing_cols.append(col)
            
    if missing_cols:
        print(f"WARNING: {len(missing_cols)} feature mancanti nel CSV (riempite con 0).")
        # print(f"Mancanti: {missing_cols[:5]}...")
    
    # Preprocess
    X = np.where(np.isinf(X), 0, X)
    X = np.where(np.isnan(X), 0, X)
    X_scaled = scaler.transform(X)
    X_clipped = np.clip(X_scaled, -CLIP_VALUE, CLIP_VALUE)
    
    # =========================================================================
    # PHASE 1: PREDICTIONS (WITHOUT LABELS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: PREDICTIONS (Label-Blind)")
    print("=" * 70)
    
    if is_booster:
        y_prob = model.predict(X_clipped)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_prob = model.predict_proba(X_clipped)[:, 1]
        y_pred = model.predict(X_clipped)
    
    n_pred_attack = (y_pred == 1).sum()
    n_pred_benign = (y_pred == 0).sum()
    
    print(f"\nPrediction Summary:")
    print(f"  Predicted ATTACK: {n_pred_attack:,} ({n_pred_attack/len(y_pred)*100:.2f}%)")
    print(f"  Predicted BENIGN: {n_pred_benign:,} ({n_pred_benign/len(y_pred)*100:.2f}%)")
    
    print(f"\nProbability Distribution:")
    print(f"  Min probability:  {y_prob.min():.4f}")
    print(f"  Max probability:  {y_prob.max():.4f}")
    print(f"  Mean probability: {y_prob.mean():.4f}")
    print(f"  Median probability: {np.median(y_prob):.4f}")
    
    print(f"\nProbability Buckets:")
    buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(buckets)-1):
        low, high = buckets[i], buckets[i+1]
        count = ((y_prob >= low) & (y_prob < high)).sum()
        pct = count / len(y_prob) * 100
        bar = '' * int(pct / 2)
        print(f"  [{low:.1f}-{high:.1f}): {count:>10,} ({pct:>5.1f}%) {bar}")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 10 predicted as ATTACK):")
    attack_indices = np.where(y_pred == 1)[0][:10]
    if len(attack_indices) > 0:
        print(f"  {'Index':<10} {'Probability':<15} {'Prediction'}")
        print(f"  {'':<40}")
        for idx in attack_indices:
            print(f"  {idx:<10} {y_prob[idx]:<15.4f} ATTACK")
    else:
        print("  No flows predicted as ATTACK")
    
    print(f"\nSample Predictions (first 10 predicted as BENIGN):")
    benign_indices = np.where(y_pred == 0)[0][:10]
    print(f"  {'Index':<10} {'Probability':<15} {'Prediction'}")
    print(f"  {'':<40}")
    for idx in benign_indices:
        print(f"  {idx:<10} {y_prob[idx]:<15.4f} BENIGN")
    
    # =========================================================================
    # PHASE 2: COMPARE WITH LABELS (User choice)
    # =========================================================================
    print("\n" + "=" * 70)
    compare = input("Vuoi confrontare con le label reali? (y/n): ").strip().lower()
    
    if compare == 'y':
        print("=" * 70)
        print("PHASE 2: COMPARISON WITH GROUND TRUTH")
        print("=" * 70)
        
        # Find label column
        label_col = None
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if not label_col:
            print("Label column not found!")
            return
        
        # Extract true labels
        labels = df[label_col].astype(str).str.strip().str.upper()
        y_true = (labels != 'BENIGN').astype(int).values
        
        # Ground truth stats
        n_true_attack = (y_true == 1).sum()
        n_true_benign = (y_true == 0).sum()
        
        print(f"\nGround Truth:")
        print(f"  Actual ATTACK: {n_true_attack:,} ({n_true_attack/len(y_true)*100:.2f}%)")
        print(f"  Actual BENIGN: {n_true_benign:,} ({n_true_benign/len(y_true)*100:.2f}%)")
        
        # Confusion matrix
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        print(f"\nConfusion Matrix:")
        print(f"  {'':15} {'Pred BENIGN':>15} {'Pred ATTACK':>15}")
        print(f"  {'Actual BENIGN':<15} {tn:>15,} {fp:>15,}")
        print(f"  {'Actual ATTACK':<15} {fn:>15,} {tp:>15,}")
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  FPR:       {fpr:.4f}")
        
        # Probability analysis by true class
        if n_true_attack > 0:
            attack_probs = y_prob[y_true == 1]
            print(f"\nProbability by True Class:")
            print(f"  ATTACK flows: mean prob = {attack_probs.mean():.4f}, median = {np.median(attack_probs):.4f}")
        
        if n_true_benign > 0:
            benign_probs = y_prob[y_true == 0]
            print(f"  BENIGN flows: mean prob = {benign_probs.mean():.4f}, median = {np.median(benign_probs):.4f}")
        
        # Show misclassifications
        print(f"\nMisclassification Analysis:")
        print(f"  False Positives (Benign predicted as Attack): {fp:,}")
        print(f"  False Negatives (Attack predicted as Benign): {fn:,}")
        
        if fn > 0:
            fn_indices = np.where((y_true == 1) & (y_pred == 0))[0][:5]
            print(f"\n  Sample False Negatives (Attack missed):")
            for idx in fn_indices:
                actual_label = labels.iloc[idx]
                print(f"    idx={idx}, prob={y_prob[idx]:.4f}, actual={actual_label}")
        
        # Label distribution in predictions
        print(f"\nAttack Types in Predictions:")
        try:
            pred_attack_labels = labels[y_pred == 1].value_counts()
            for label, count in pred_attack_labels.items():
                print(f"    {label}: {count:,}")
        except:
            pass

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()