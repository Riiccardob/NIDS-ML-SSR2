#!/usr/bin/env python3
"""
NIDS-ML Fix Script - Risolve i problemi identificati

Problemi risolti:
1. Carica righe con ATTACCHI (non solo le prime N)
2. Applica clipping aggressivo per valori fuori distribuzione
3. Gestisce version mismatch sklearn

Esegui: python fix_and_test.py
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Sopprimi warning sklearn version
warnings.filterwarnings('ignore', category=UserWarning)

def normalize_name(name: str) -> str:
    return name.strip().lower()

def load_csv_with_attacks(csv_path: str, n_benign: int = 5000, n_attack: int = 5000) -> pd.DataFrame:
    """
    Carica CSV assicurandosi di avere sia BENIGN che ATTACK.
    
    Il problema: le prime N righe potrebbero essere tutte BENIGN.
    Soluzione: leggiamo tutto e campioniamo bilanciato.
    """
    print(f"Loading CSV with balanced sampling...")
    
    # Leggi tutto il CSV (o un chunk grande)
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Trova colonna label
    label_col = None
    for col in df.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break
    
    if not label_col:
        raise ValueError("Label column not found")
    
    # Separa benign e attack
    benign = df[df[label_col].str.strip().str.upper() == 'BENIGN']
    attack = df[df[label_col].str.strip().str.upper() != 'BENIGN']
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Benign: {len(benign):,}")
    print(f"  Attack: {len(attack):,}")
    
    # Campiona bilanciato
    n_benign = min(n_benign, len(benign))
    n_attack = min(n_attack, len(attack))
    
    if n_attack == 0:
        print("  WARNING: No attack rows found in this CSV!")
        return benign.sample(n=min(10000, len(benign)), random_state=42)
    
    sampled = pd.concat([
        benign.sample(n=n_benign, random_state=42),
        attack.sample(n=n_attack, random_state=42)
    ]).sample(frac=1, random_state=42)  # Shuffle
    
    print(f"  Sampled: {len(sampled):,} ({n_benign} benign + {n_attack} attack)")
    
    return sampled


def extract_features(df: pd.DataFrame, scaler_columns: list) -> np.ndarray:
    """Estrae feature nell'ordine corretto."""
    csv_cols_norm = {normalize_name(c): c for c in df.columns}
    
    n_rows = len(df)
    n_cols = len(scaler_columns)
    X = np.zeros((n_rows, n_cols), dtype=np.float64)
    
    matched = 0
    for i, scaler_col in enumerate(scaler_columns):
        scaler_col_norm = normalize_name(scaler_col)
        
        # Trova colonna CSV corrispondente
        csv_col = None
        if scaler_col_norm in csv_cols_norm:
            csv_col = csv_cols_norm[scaler_col_norm]
        else:
            for var in [scaler_col_norm.replace(' ', '_'), scaler_col_norm.replace('_', ' ')]:
                if var in csv_cols_norm:
                    csv_col = csv_cols_norm[var]
                    break
        
        if csv_col and csv_col in df.columns:
            X[:, i] = df[csv_col].values
            matched += 1
    
    print(f"  Features matched: {matched}/{n_cols}")
    return X


def preprocess_with_clipping(X: np.ndarray, scaler, selected_indices: list, clip_value: float = 5.0) -> np.ndarray:
    """
    Preprocessa con clipping aggressivo.
    
    Il problema: lo scaler produce valori tipo 51000000 che confondono il modello.
    Soluzione: clip a [-5, 5] DOPO lo scaling (range ragionevole per alberi).
    """
    # 1. Gestisci inf/nan
    X = np.where(np.isinf(X), 0, X)
    X = np.where(np.isnan(X), 0, X)
    
    # 2. Scala
    X_scaled = scaler.transform(X)
    
    # 3. CLIP AGGRESSIVO (cruciale!)
    X_clipped = np.clip(X_scaled, -clip_value, clip_value)
    
    # 4. Seleziona feature
    X_selected = X_clipped[:, selected_indices]
    
    return X_selected


def main():
    print("=" * 70)
    print("NIDS-ML FIX AND TEST")
    print("=" * 70)
    
    # Paths
    artifacts_dir = Path("artifacts")
    model_dir = Path("models/best_model")
    
    # Test su tutti i CSV con attacchi
    csv_files = {
        'Tuesday': 'data/raw/Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday': 'data/raw/Wednesday-workingHours.pcap_ISCX.csv',
        'Friday DDoS': 'data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday PortScan': 'data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    }
    
    # 1. Load artifacts
    print("\n[1] LOADING ARTIFACTS")
    print("-" * 50)
    
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    print(f"Scaler: {type(scaler).__name__}")
    
    with open(artifacts_dir / "scaler_columns.json") as f:
        scaler_columns = json.load(f)
    print(f"Scaler columns: {len(scaler_columns)}")
    
    with open(artifacts_dir / "selected_features.json") as f:
        selected_features = json.load(f)
    print(f"Selected features: {len(selected_features)}")
    
    model_path = model_dir / "model_binary.pkl"
    if not model_path.exists():
        model_path = model_dir / "model.pkl"
    model = joblib.load(model_path)
    print(f"Model: {type(model).__name__}")
    
    # Compute indices
    scaler_cols_norm = {normalize_name(c): i for i, c in enumerate(scaler_columns)}
    selected_indices = []
    for feat in selected_features:
        feat_norm = normalize_name(feat)
        if feat_norm in scaler_cols_norm:
            selected_indices.append(scaler_cols_norm[feat_norm])
    print(f"Selection indices: {len(selected_indices)}")
    
    # 2. Test each CSV
    print("\n[2] TESTING CSVs WITH BALANCED SAMPLING")
    print("-" * 50)
    
    results = {}
    
    for name, csv_path in csv_files.items():
        if not Path(csv_path).exists():
            print(f"\n{name}: FILE NOT FOUND")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print(f"{'=' * 60}")
        
        # Load with balanced sampling
        df = load_csv_with_attacks(csv_path, n_benign=5000, n_attack=5000)
        
        # Find label column
        label_col = None
        for col in df.columns:
            if col.strip().lower() == 'label':
                label_col = col
                break
        
        # Get true labels
        y_true = (df[label_col].str.strip().str.upper() != 'BENIGN').astype(int).values
        n_attacks = y_true.sum()
        print(f"True attacks in sample: {n_attacks}/{len(y_true)}")
        
        if n_attacks == 0:
            print("SKIPPING: No attacks in this CSV")
            continue
        
        # Extract features
        X_raw = extract_features(df, scaler_columns)
        
        # Test different clip values
        print("\nTesting different clip values:")
        
        for clip_val in [3.0, 5.0, 10.0, None]:
            if clip_val is None:
                # No clipping
                X_clean = np.where(np.isinf(X_raw), 0, X_raw)
                X_clean = np.where(np.isnan(X_clean), 0, X_clean)
                X_scaled = scaler.transform(X_clean)
                X_selected = X_scaled[:, selected_indices]
                clip_str = "None"
            else:
                X_selected = preprocess_with_clipping(X_raw, scaler, selected_indices, clip_val)
                clip_str = f"{clip_val}"
            
            # Predict
            y_pred = model.predict(X_selected)
            y_prob = model.predict_proba(X_selected)[:, 1]
            
            # Metrics
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Prob stats for attacks
            attack_probs = y_prob[y_true == 1]
            mean_attack_prob = attack_probs.mean() if len(attack_probs) > 0 else 0
            
            print(f"  Clip={clip_str:>5}: F1={f1:.4f} | Prec={precision:.4f} | Rec={recall:.4f} | "
                  f"TP={tp:>5} | FP={fp:>5} | FN={fn:>5} | Avg Attack Prob={mean_attack_prob:.4f}")
        
        results[name] = {
            'n_samples': len(df),
            'n_attacks': n_attacks
        }
    
    # 3. Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    print("""
Based on the test results:

1. If F1 is still ~0 with clipping:
   -> The model was trained on DIFFERENT data distribution
   -> Need to RETRAIN the model with the current preprocessing pipeline
   -> Or regenerate artifacts (scaler.pkl) with sklearn 1.8.0

2. If F1 improves with clip=5.0:
   -> Add clipping to the sniffer evaluation code
   -> Use: X_clipped = np.clip(X_scaled, -5.0, 5.0)

3. The sklearn version warning:
   -> Regenerate scaler.pkl with your current sklearn version
   -> Run the training pipeline again: python -m scripts.03_train_models

4. CRITICAL: Make sure training and inference use the SAME:
   - sklearn version
   - Feature columns (scaler_columns.json)
   - Preprocessing steps (statistical filtering)
""")


if __name__ == "__main__":
    main()
