#!/usr/bin/env python3
"""
Script di verifica rapida del dataset NF-UQ-NIDS-v2.

Esegui questo prima del preprocessing per verificare:
- Che il file CSV sia presente
- Dimensione e formato del dataset
- La distribuzione delle label (sample veloce)

Ottimizzato per dataset grandi (>70M records).
"""

import pandas as pd
from pathlib import Path
import sys

# Percorsi
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Sample size per verifica veloce
SAMPLE_SIZE = 50_000


def check_dataset():
    """Verifica rapida del dataset."""
    
    print("="*70)
    print("VERIFICA DATASET NF-UQ-NIDS-v2")
    print("="*70)
    print()
    
    # ========================================================================
    # 1. Cerca file CSV
    # ========================================================================
    
    print(f" Cercando file CSV in: {RAW_DATA_DIR}")
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print()
        print(" ERRORE: Nessun file CSV trovato!")
        print(f"   Directory: {RAW_DATA_DIR}")
        print()
        print("SOLUZIONE:")
        print(f"  1. Scarica il dataset NF-UQ-NIDS-v2")
        print(f"  2. Copia il file CSV in: {RAW_DATA_DIR}")
        print()
        return False
    
    print(f" Trovati {len(csv_files)} file CSV:")
    total_size_mb = 0
    for f in csv_files:
        size_mb = f.stat().st_size / (1024**2)
        total_size_mb += size_mb
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print(f"\n  Dimensione totale: {total_size_mb:.1f} MB")
    
    if total_size_mb > 5000:  # >5GB
        print(f"    Dataset grande rilevato (>{total_size_mb/1024:.1f} GB)")
        print(f"  La pipeline userà chunk-based processing")
    
    print()
    
    # ========================================================================
    # 2. Carica sample per verifica
    # ========================================================================
    
    print(f" Caricamento sample ({SAMPLE_SIZE:,} righe) per verifica...")
    try:
        df = pd.read_csv(csv_files[0], nrows=SAMPLE_SIZE, low_memory=False)
        print(f" Sample caricato: {df.shape}")
        print()
    except Exception as e:
        print(f" ERRORE nel caricamento: {e}")
        return False
    
    # ========================================================================
    # 3. Verifica colonne
    # ========================================================================
    
    print(" Informazioni dataset:")
    print(f"   Colonne totali: {len(df.columns)}")
    
    # Cerca colonna Label
    label_col = None
    for col in df.columns:
        if col.strip().upper() == 'LABEL':
            label_col = col
            break
    
    if label_col is None:
        print()
        print(" ERRORE: Colonna 'Label' non trovata!")
        print(f"   Colonne disponibili: {list(df.columns)[:10]}...")
        return False
    
    print(f"    Colonna label trovata: '{label_col}'")
    
    # Conta feature numeriche
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(f"   Feature numeriche: {len(numeric_cols)}")
    print()
    
    # ========================================================================
    # 4. Analizza label (sample)
    # ========================================================================
    
    print("  ANALISI LABEL (sample):")
    print("-"*70)
    
    label_counts = df[label_col].value_counts()
    total = len(df)
    
    print(f"Campione analizzato: {total:,} righe")
    print(f"Label uniche: {len(label_counts)}")
    print()
    
    # Verifica se label binarie
    unique_labels = set(label_counts.index)
    
    if unique_labels == {0, 1}:
        print(" LABEL BINARIE RILEVATE (0/1)")
        print()
        
        benign = label_counts.get(0, 0)
        attack = label_counts.get(1, 0)
        
        print(f"  Classe 0 (Benign): {benign:>8,} ({benign/total*100:>5.1f}%)")
        print(f"  Classe 1 (Attack): {attack:>8,} ({attack/total*100:>5.1f}%)")
        print()
        
        # Stima sul dataset completo
        if len(csv_files) == 1:
            file_size = csv_files[0].stat().st_size
            sample_size_bytes = df.memory_usage(deep=True).sum()
            
            estimated_rows = int((file_size / sample_size_bytes) * total * 0.9)
            estimated_benign = int((benign / total) * estimated_rows)
            estimated_attack = int((attack / total) * estimated_rows)
            
            print(f" STIMA DATASET COMPLETO:")
            print(f"  Righe totali stimate: ~{estimated_rows:,}")
            print(f"  Benign stimati: ~{estimated_benign:,}")
            print(f"  Attack stimati: ~{estimated_attack:,}")
            print()
        
        # Verifica presenza Attack column
        if 'Attack' in df.columns:
            print("ℹ  Rilevata anche colonna 'Attack' con tipi di attacco")
            attack_types = df['Attack'].value_counts()
            print(f"   Tipi di attacco nel sample: {len(attack_types)}")
            if len(attack_types) <= 10:
                print(f"   Tipi: {list(attack_types.index)}")
            print()
        
        has_both_classes = True
        
    elif unique_labels == {0} or unique_labels == {1}:
        print(" ATTENZIONE: Dataset contiene solo una classe!")
        print(f"   Label presente: {unique_labels}")
        print()
        print("   NOTA: Questo potrebbe essere solo nel sample.")
        print("   La pipeline verificherà il dataset completo durante preprocessing.")
        print()
        has_both_classes = False
        
    else:
        print(" ATTENZIONE: Label non sono binarie 0/1!")
        print(f"   Label presenti: {list(unique_labels)}")
        print()
        print("Distribuzione:")
        for label, count in label_counts.items():
            pct = (count / total) * 100
            print(f"  {str(label):20s}: {count:>6,} ({pct:>5.1f}%)")
        print()
        has_both_classes = False
    
    # ========================================================================
    # 5. Info memoria
    # ========================================================================
    
    print(" INFORMAZIONI MEMORIA:")
    print("-"*70)
    
    sample_memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"  Memory del sample: {sample_memory_mb:.1f} MB")
    
    if len(csv_files) == 1 and total == SAMPLE_SIZE:
        # Stima memoria totale dataset
        file_size = csv_files[0].stat().st_size
        estimated_rows = int((file_size / (sample_memory_mb * 1024**2)) * total * 0.9)
        estimated_memory_gb = (estimated_rows / total) * sample_memory_mb / 1024
        
        print(f"  Memory stimata dataset completo: ~{estimated_memory_gb:.1f} GB")
        print()
        
        if estimated_memory_gb > 8:
            print(f"    Dataset grande: la pipeline userà chunk-based processing")
            print(f"  Requisiti: ~8-12 GB RAM disponibili")
        else:
            print(f"   Dataset gestibile con 16GB RAM")
    
    print("-"*70)
    print()
    
    # ========================================================================
    # 6. Riepilogo finale
    # ========================================================================
    
    print("="*70)
    if has_both_classes:
        print(" VERIFICA COMPLETATA")
    else:
        print("  VERIFICA COMPLETATA CON WARNING")
    print("="*70)
    print()
    
    if has_both_classes:
        print("RIEPILOGO:")
        print(f"   File CSV presente: {csv_files[0].name}")
        print(f"   Label binarie (0/1) rilevate")
        print(f"   Sample contiene entrambe le classi")
        print()
        print("PRONTO PER IL PREPROCESSING!")
        print()
        print("COMANDI DA ESEGUIRE:")
        print("  1. python srcNF/preprocessing.py")
        print("  2. python srcNF/feature_engineering.py")
        print("  3. python srcNF/training.py --model xgboost")
        print()
        print("OPPURE pipeline completa:")
        print("  python srcNF/pipeline.py --model xgboost")
    else:
        print("ATTENZIONE:")
        print("  Il sample mostra possibili problemi con le label")
        print("  Procedi comunque - la pipeline verificherà il dataset completo")
        print()
        print("COMANDI:")
        print("  python srcNF/preprocessing.py")
    
    print("="*70)
    print()
    
    return True


if __name__ == "__main__":
    success = check_dataset()
    sys.exit(0 if success else 1)
