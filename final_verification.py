import pandas as pd
import joblib
import json
import numpy as np
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix

# CONFIGURAZIONE
THRESHOLD = 0.02  # La soglia magica (2%)
CSV_PATH = "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

def main():
    print(f" AVVIO VERIFICA MANUALE CON SOGLIA {THRESHOLD}")
    print("=" * 60)

    # 1. Carica Artifacts
    print("[1] Caricamento Modello e Scaler...")
    scaler = joblib.load("artifacts/scaler.pkl")
    model = joblib.load("models/best_model/model_binary.pkl")
    
    with open("artifacts/scaler_columns.json") as f:
        scaler_cols = json.load(f)
    with open("artifacts/selected_features.json") as f:
        selected_cols = json.load(f)

    # 2. Carica Dati
    print(f"[2] Caricamento CSV: {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Pulisci nomi colonne
    df.columns = df.columns.str.strip()
    
    # Prepara le Label vere
    y_true = (df['Label'] != 'BENIGN').astype(int)
    print(f"    Totale Righe: {len(df)}")
    print(f"    Attacchi Veri: {y_true.sum()}")

    # 3. Preprocessing Manuale (Rigoroso)
    print("[3] Preprocessing...")
    
    # Estrai solo le 44 colonne dello scaler nell'ordine giusto
    X_raw = df[scaler_cols].copy()
    
    # Gestione infiniti e NaN
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scaling
    X_scaled = scaler.transform(X_raw)
    
    # Feature Selection (per indice)
    sel_indices = [scaler_cols.index(c) for c in selected_cols]
    X_final = X_scaled[:, sel_indices]

    # 4. Predizione Probabilistica
    print("[4] Calcolo Probabilità...")
    # Ottieni la probabilità grezza (colonna 1)
    y_prob = model.predict_proba(X_final)[:, 1]
    
    # STATISTICHE VITALI
    print("\n--- STATISTICHE PROBABILITÀ ---")
    print(f"Media prob. Attacchi: {y_prob[y_true==1].mean():.4f}")
    print(f"Media prob. Benigni:  {y_prob[y_true==0].mean():.4f}")
    print(f"Max prob. rilevata:   {y_prob.max():.4f}")
    
    # 5. Applicazione Soglia Manuale
    print(f"\n[5] Applicazione Soglia > {THRESHOLD}...")
    y_pred = (y_prob > THRESHOLD).astype(int)
    
    # 6. Risultati
    print("\n" + "="*60)
    print("RISULTATI FINALI")
    print("="*60)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"TP (Attacchi presi):     {tp}")
    print(f"FN (Attacchi persi):     {fn}")
    print(f"FP (Falsi allarmi):      {fp}")
    print(f"TN (Benigni corretti):   {tn}")
    print("-" * 30)
    print(f"RECALL:    {recall:.4f}  <-- QUESTO È IL NUMERO CHE CONTA")
    print(f"PRECISION: {precision:.4f}")
    print(f"F1 SCORE:  {f1:.4f}")
    
    if recall > 0.8:
        print("\n VITTORIA! Il sistema funziona con il tuning della soglia.")
    else:
        print("\n ANCORA BASSO. Serve riaddestramento.")

if __name__ == "__main__":
    main()