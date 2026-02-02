#!/usr/bin/env python3
"""
NIDS-ML Real Attack Test - Cerca e testa un VERO attacco
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def main():
    print("=" * 70)
    print("NIDS-ML REAL ATTACK HUNTER")
    print("=" * 70)
    
    # Paths
    artifacts_dir = Path("artifacts")
    model_dir = Path("models/best_model")
    # Usa il file DDoS che sappiamo contenere attacchi
    csv_path = "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    # 1. Load artifacts
    print("\n[1] Loading artifacts...")
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    with open(artifacts_dir / "scaler_columns.json") as f:
        scaler_columns = json.load(f)
    with open(artifacts_dir / "selected_features.json") as f:
        selected_features = json.load(f)
    model = joblib.load(model_dir / "model_binary.pkl")
    
    # 2. Cerca una riga di attacco (Legge a blocchi per non saturare la RAM)
    print("\n[2] Hunting for DDoS attack row...")
    attack_row = None
    chunk_size = 50000
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Rimuovi spazi dai nomi colonne
        chunk.columns = chunk.columns.str.strip()
        
        attacks = chunk[chunk['Label'] != 'BENIGN']
        if not attacks.empty:
            print(f"  -> FOUND! Found {len(attacks)} attacks in chunk.")
            attack_row = attacks.iloc[0]
            print(f"  -> Selected Attack Type: {attack_row['Label']}")
            break
        print(f"  -> Scanned chunk (all benign)...")

    if attack_row is None:
        print("ERROR: Nessun attacco trovato nell'intero file!")
        return

    # 3. Costruisci vettore feature (Logica Claude/Gemini)
    print("\n[3] Building Feature Vector...")
    
    # Estrai le 44 feature per lo scaler
    vector_raw = []
    print("  Mapping columns...")
    for col in scaler_columns:
        # Gestione nomi flessibile
        val = None
        if col in attack_row: val = attack_row[col]
        elif " " + col in attack_row: val = attack_row[" " + col] # Spazio iniziale
        elif col.strip() in attack_row: val = attack_row[col.strip()]
        
        if val is None:
            print(f"  WARNING: Missing col {col}, using 0.0")
            val = 0.0
        vector_raw.append(val)
    
    X_raw = np.array([vector_raw], dtype=np.float32)
    
    # Pulizia
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Scaling
    X_scaled = scaler.transform(X_raw)
    
    # Clipping (FIX ESSENZIALE per valori esplosi)
    X_scaled = np.clip(X_scaled, -10, 10)
    
    # Selezione 30 feature finali
    final_indices = [scaler_columns.index(f) for f in selected_features if f in scaler_columns]
    X_final = X_scaled[:, final_indices]
    
    # 4. Predizione
    print("\n[4] Prediction...")
    prob = model.predict_proba(X_final)[0][1]
    pred = model.predict(X_final)[0]
    
    print("-" * 30)
    print(f"TRUE LABEL:  {attack_row['Label']}")
    print(f"PREDICTION:  {'ATTACK' if pred==1 else 'BENIGN'} (Class {pred})")
    print(f"CONFIDENCE:  {prob:.4f}")
    print("-" * 30)
    
    if prob > 0.1: # Soglia bassa per test
        print(" SUCCESS: Il modello ha rilevato una minaccia (anche se debole)!")
    else:
        print(" FAILURE: Il modello è ancora cieco.")

if __name__ == "__main__":
    main()