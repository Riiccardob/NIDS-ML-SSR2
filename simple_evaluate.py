#!/usr/bin/env python3
"""
NIDS-ML Simple Evaluator - No Feature Selection

Questo script valuta il modello usando TUTTE le 44 feature.
Niente feature selection post-scaling.

Pipeline:
  Raw CSV → 44 features → RobustScaler → Clip → Model (44 features)

Esegui: python simple_evaluate.py
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

CLIP_VALUE = 10.0

CICIDS2017_FILES = {
    'monday': ('Monday-WorkingHours.pcap_ISCX.csv', []),
    'tuesday': ('Tuesday-WorkingHours.pcap_ISCX.csv', ['FTP-Patator', 'SSH-Patator']),
    'wednesday': ('Wednesday-workingHours.pcap_ISCX.csv', ['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye', 'Heartbleed']),
    'thursday_morning': ('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', ['Web Attack']),
    'thursday_afternoon': ('Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', ['Infiltration']),
    'friday_morning': ('Friday-WorkingHours-Morning.pcap_ISCX.csv', ['Bot']),
    'friday_portscan': ('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', ['PortScan']),
    'friday_ddos': ('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', ['DDoS']),
}


def normalize_name(name: str) -> str:
    return name.strip().lower()


def load_artifacts(artifacts_dir: str = "artifacts", model_dir: str = "models/best_model"):
    """Carica scaler e model."""
    artifacts_path = Path(artifacts_dir)
    model_path = Path(model_dir)
    
    # Scaler
    scaler = joblib.load(artifacts_path / "scaler.pkl")
    print(f"Scaler: {type(scaler).__name__}")
    
    # Scaler columns
    with open(artifacts_path / "scaler_columns.json") as f:
        scaler_columns = json.load(f)
    print(f"Scaler columns: {len(scaler_columns)}")
    
    # Model
    model_file = model_path / "model_binary.pkl"
    if not model_file.exists():
        model_file = model_path / "model.pkl"
    
    model = joblib.load(model_file)
    print(f"Model: {type(model).__name__}")
    
    # Check if it's a LightGBM Booster
    is_booster = hasattr(model, 'predict') and not hasattr(model, 'predict_proba')
    print(f"Is LightGBM Booster: {is_booster}")
    
    return scaler, scaler_columns, model, is_booster


def extract_features(df: pd.DataFrame, scaler_columns: list) -> np.ndarray:
    """Estrae feature nell'ordine di scaler_columns."""
    csv_cols_norm = {normalize_name(c): c for c in df.columns}
    
    X = np.zeros((len(df), len(scaler_columns)), dtype=np.float64)
    matched = 0
    
    for i, col in enumerate(scaler_columns):
        col_norm = normalize_name(col)
        
        csv_col = None
        if col_norm in csv_cols_norm:
            csv_col = csv_cols_norm[col_norm]
        else:
            for var in [col_norm.replace(' ', '_'), col_norm.replace('_', ' ')]:
                if var in csv_cols_norm:
                    csv_col = csv_cols_norm[var]
                    break
        
        if csv_col and csv_col in df.columns:
            X[:, i] = df[csv_col].values
            matched += 1
    
    return X, matched


def preprocess(X: np.ndarray, scaler, clip: float = CLIP_VALUE) -> np.ndarray:
    """Preprocessa: clean → scale → clip."""
    # Handle inf/nan
    X = np.where(np.isinf(X), 0, X)
    X = np.where(np.isnan(X), 0, X)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Clip
    X_clipped = np.clip(X_scaled, -clip, clip)
    
    return X_clipped


def predict(model, X: np.ndarray, is_booster: bool, threshold: float = 0.5):
    """Predice usando il modello."""
    if is_booster:
        # LightGBM Booster returns probabilities directly
        y_prob = model.predict(X)
        y_pred = (y_prob > threshold).astype(int)
    else:
        # Sklearn-style model
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
    
    return y_pred, y_prob


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Calcola metriche."""
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Prob stats
    attack_prob_mean = y_prob[y_true == 1].mean() if (y_true == 1).any() else 0
    benign_prob_mean = y_prob[y_true == 0].mean() if (y_true == 0).any() else 0
    
    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision': precision, 'recall': recall, 'f1': f1,
        'fpr': fpr, 'accuracy': accuracy,
        'attack_prob_mean': attack_prob_mean,
        'benign_prob_mean': benign_prob_mean
    }


def evaluate_csv(csv_path: str, scaler, scaler_columns: list, model, is_booster: bool,
                 sample_size: int = None, batch_size: int = 50000):
    """Valuta un singolo CSV."""
    # Load CSV
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Sample if needed
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Find label column
    label_col = None
    for col in df.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break
    
    if not label_col:
        raise ValueError("Label column not found")
    
    # Extract labels
    labels = df[label_col].astype(str).str.strip().str.upper()
    y_true = (labels != 'BENIGN').astype(int).values
    
    # Class distribution
    class_dist = df[label_col].value_counts().to_dict()
    
    # Extract features
    X_raw, matched = extract_features(df, scaler_columns)
    
    # Preprocess
    X_processed = preprocess(X_raw, scaler)
    
    # Predict in batches
    all_preds = []
    all_probs = []
    
    for i in tqdm(range(0, len(X_processed), batch_size), desc="Predicting"):
        batch = X_processed[i:i+batch_size]
        y_pred, y_prob = predict(model, batch, is_booster)
        all_preds.extend(y_pred)
        all_probs.extend(y_prob)
    
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics['total_samples'] = len(df)
    metrics['features_matched'] = matched
    metrics['class_distribution'] = class_dist
    
    return metrics


def main():
    print("=" * 70)
    print("NIDS-ML SIMPLE EVALUATOR (No Feature Selection)")
    print("=" * 70)
    
    # Load artifacts
    print("\n[1] Loading artifacts...")
    scaler, scaler_columns, model, is_booster = load_artifacts()
    
    # Evaluate all CSVs
    print("\n[2] Evaluating CSVs...")
    
    data_dir = Path("data/raw")
    results = {}
    
    for day, (csv_name, attacks) in CICIDS2017_FILES.items():
        csv_path = data_dir / csv_name
        
        if not csv_path.exists():
            print(f"\n{day}: FILE NOT FOUND")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Day: {day.upper()}")
        print(f"File: {csv_name}")
        print(f"Expected attacks: {attacks if attacks else 'Benign only'}")
        print(f"{'=' * 60}")
        
        try:
            metrics = evaluate_csv(str(csv_path), scaler, scaler_columns, model, is_booster)
            results[day] = metrics
            
            print(f"\nResults:")
            print(f"  Samples:   {metrics['total_samples']:,}")
            print(f"  Features:  {metrics['features_matched']}/{len(scaler_columns)}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  FPR:       {metrics['fpr']:.4f}")
            print(f"  TP={metrics['tp']:,} TN={metrics['tn']:,} FP={metrics['fp']:,} FN={metrics['fn']:,}")
            print(f"  Attack prob mean: {metrics['attack_prob_mean']:.4f}")
            print(f"  Benign prob mean: {metrics['benign_prob_mean']:.4f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[day] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Day':<22} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FPR':>8} {'Samples':>12}")
    print("-" * 70)
    
    for day, res in results.items():
        if 'error' in res:
            print(f"{day:<22} {'ERROR':>8}")
        else:
            print(f"{day:<22} {res['f1']:>8.4f} {res['precision']:>8.4f} "
                  f"{res['recall']:>8.4f} {res['fpr']:>8.4f} {res['total_samples']:>12,}")
    
    # Save results
    output_path = Path("reports/simple_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
