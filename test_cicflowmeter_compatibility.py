#!/usr/bin/env python3
"""
CICFlowMeter Python vs CSV Originale - Test di Compatibilità

Questo script:
1. Processa un PCAP con cicflowmeter Python
2. Confronta le feature estratte con il CSV originale
3. Calcola le differenze per determinare se sono compatibili

Se le differenze sono < 5%, il pacchetto cicflowmeter è utilizzabile
con il modello trainato sui CSV.

Prerequisiti:
    pip install cicflowmeter pandas numpy

Uso:
    python test_cicflowmeter_compatibility.py
"""

import sys
import subprocess
import tempfile
from pathlib import Path
from scapy.all import rdpcap
from cicflowmeter.flow_session import FlowSession
import time
from cicflowmeter.features.flow_bytes import FlowBytes
from cicflowmeter.features.context import PacketDirection

def monkey_patch_cicflowmeter():
    """
    Sovrascrive la funzione buggata della libreria in memoria
    per evitare il crash 'min() iterable argument is empty'.
    """
    def fixed_get_min_forward_header_bytes(self) -> int:
        if not self.flow.packets:
            return 0
        
        # Filtriamo prima i valori
        forward_headers = [
            self._header_size(packet)
            for packet, direction in self.flow.packets
            if direction == PacketDirection.FORWARD
        ]
        
        # FIX: Se la lista è vuota, restituiamo 0 invece di crashare con min()
        if not forward_headers:
            return 0
            
        return min(forward_headers)

    # Applichiamo la patch alla classe originale
    print("PATCHING: Applicazione fix a FlowBytes.get_min_forward_header_bytes...")
    FlowBytes.get_min_forward_header_bytes = fixed_get_min_forward_header_bytes
# -------------------------------------------------

# Verifica se cicflowmeter è installato
try:
    from cicflowmeter.flow_session import FlowSession
    from cicflowmeter.sniffer import create_sniffer
    CICFLOWMETER_AVAILABLE = True
except ImportError:
    CICFLOWMETER_AVAILABLE = False
    print("=" * 60)
    print("cicflowmeter non installato!")
    print("=" * 60)
    print("\nInstalla con:")
    print("  pip install cicflowmeter")
    print("\nOppure clona da GitHub:")
    print("  git clone https://github.com/hieulw/cicflowmeter")
    print("  cd cicflowmeter")
    print("  pip install .")
    print("=" * 60)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def normalize_column_name(name):
    """Normalizza nome colonna per confronto."""
    return name.strip().lower().replace(' ', '_').replace('/', '_')


def extract_with_cicflowmeter(pcap_path, output_csv):
    """
    Estrai feature da PCAP usando cicflowmeter Python (Versione FIXATA da Source Code).
    Usa il metodo .process() che è quello reale della libreria 0.5.0.
    """
    if not CICFLOWMETER_AVAILABLE:
        return None
    
    print(f"Processing PCAP with cicflowmeter (Code Analysis Fix): {pcap_path}")
    print("Reading packets... (può richiedere tempo)")

    try:
        # 1. Creiamo la sessione
        # Parametro 'output' è confermato dal codice sorgente [cite: 125]
        flow_session = FlowSession(output_mode="csv", output=output_csv)
        
        # 2. Leggiamo i pacchetti con Scapy (50k per test rapido)
        packets = rdpcap(pcap_path, count=50000) 
        print(f"Read {len(packets)} packets. Processing...")

        start_time = time.time()
        
        # 3. METODO CORRETTO: Usiamo un loop con .process()
        # Analizzando il codice, 'process' accetta un singolo pacchetto 'pkt'
        for i, packet in enumerate(packets):
            flow_session.process(packet)
            
            if i % 10000 == 0:
                print(f"  Processed {i} packets...")

        # 4. Flush finale
        # Il codice sorgente [cite: 143] conferma che esiste flush_flows()
        print("Flushing flows...")
        flow_session.flush_flows()
        
        print(f"Analysis done in {time.time() - start_time:.2f}s.")

        # Verifica risultato
        if Path(output_csv).exists():
            df = pd.read_csv(output_csv)
            print(f"Extracted {len(df)} flows")
            return df
        else:
            print("Output CSV not created.")
            return None
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_original_csv(csv_path):
    """Carica CSV originale di CICFlowMeter."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def compare_feature_names(df_python, df_original):
    """Confronta i nomi delle feature."""
    python_cols = set(normalize_column_name(c) for c in df_python.columns)
    original_cols = set(normalize_column_name(c) for c in df_original.columns)
    
    common = python_cols & original_cols
    only_python = python_cols - original_cols
    only_original = original_cols - python_cols
    
    print(f"\nFeature Comparison:")
    print(f"  Common features:           {len(common)}")
    print(f"  Only in Python extractor:  {len(only_python)}")
    print(f"  Only in Original CSV:      {len(only_original)}")
    
    if only_python:
        print(f"\n  Features only in Python: {list(only_python)[:10]}...")
    if only_original:
        print(f"\n  Features only in Original: {list(only_original)[:10]}...")
    
    return common, only_python, only_original


def compare_feature_values(df_python, df_original, common_features, sample_size=1000):
    """
    Confronta i valori delle feature comuni.
    
    Non possiamo fare un confronto row-by-row perché:
    - I flow potrebbero essere in ordine diverso
    - I flow potrebbero essere aggregati diversamente
    
    Confrontiamo invece le DISTRIBUZIONI delle feature.
    """
    print(f"\nComparing feature value distributions...")
    
    # Sample per velocità
    if len(df_python) > sample_size:
        df_python = df_python.sample(n=sample_size, random_state=42)
    if len(df_original) > sample_size:
        df_original = df_original.sample(n=sample_size, random_state=42)
    
    # Normalizza nomi colonne per lookup
    python_col_map = {normalize_column_name(c): c for c in df_python.columns}
    original_col_map = {normalize_column_name(c): c for c in df_original.columns}
    
    results = []
    
    for feat in sorted(common_features):
        if feat in ['label', 'flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp']:
            continue
        
        python_col = python_col_map.get(feat)
        original_col = original_col_map.get(feat)
        
        if not python_col or not original_col:
            continue
        
        try:
            python_vals = pd.to_numeric(df_python[python_col], errors='coerce').dropna()
            original_vals = pd.to_numeric(df_original[original_col], errors='coerce').dropna()
            
            if len(python_vals) == 0 or len(original_vals) == 0:
                continue
            
            # Confronta statistiche
            python_mean = python_vals.mean()
            original_mean = original_vals.mean()
            python_std = python_vals.std()
            original_std = original_vals.std()
            
            # Calcola differenza percentuale
            mean_diff_pct = abs(python_mean - original_mean) / (abs(original_mean) + 1e-10) * 100
            std_diff_pct = abs(python_std - original_std) / (abs(original_std) + 1e-10) * 100
            
            results.append({
                'feature': feat,
                'python_mean': python_mean,
                'original_mean': original_mean,
                'mean_diff_pct': mean_diff_pct,
                'python_std': python_std,
                'original_std': original_std,
                'std_diff_pct': std_diff_pct
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)


def print_comparison_results(comparison_df):
    """Stampa risultati del confronto."""
    if comparison_df.empty:
        print("No features to compare!")
        return
    
    print(f"\n{'Feature':<35} {'Py Mean':>12} {'Orig Mean':>12} {'Diff %':>10}")
    print("-" * 75)
    
    # Sort by difference
    comparison_df = comparison_df.sort_values('mean_diff_pct', ascending=False)
    
    # Show top 10 most different
    print("\nTop 10 Most Different Features:")
    for _, row in comparison_df.head(10).iterrows():
        print(f"  {row['feature']:<33} {row['python_mean']:>12.2f} {row['original_mean']:>12.2f} {row['mean_diff_pct']:>9.1f}%")
    
    # Show top 10 most similar
    print("\nTop 10 Most Similar Features:")
    for _, row in comparison_df.tail(10).iterrows():
        print(f"  {row['feature']:<33} {row['python_mean']:>12.2f} {row['original_mean']:>12.2f} {row['mean_diff_pct']:>9.1f}%")
    
    # Overall stats
    avg_diff = comparison_df['mean_diff_pct'].mean()
    median_diff = comparison_df['mean_diff_pct'].median()
    features_under_5pct = (comparison_df['mean_diff_pct'] < 5).sum()
    features_under_10pct = (comparison_df['mean_diff_pct'] < 10).sum()
    
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"Average difference:     {avg_diff:.1f}%")
    print(f"Median difference:      {median_diff:.1f}%")
    print(f"Features with < 5% diff:  {features_under_5pct}/{len(comparison_df)}")
    print(f"Features with < 10% diff: {features_under_10pct}/{len(comparison_df)}")
    
    # Verdict
    print(f"\n{'='*75}")
    print("VERDICT")
    print(f"{'='*75}")
    
    if median_diff < 5:
        print("EXCELLENT: cicflowmeter Python produce feature molto simili!")
        print("Puoi usarlo come feature extractor per lo sniffer.")
    elif median_diff < 15:
        print("MODERATE: Ci sono differenze, ma potrebbero essere accettabili.")
        print("Consiglio: Retrain il modello sulle feature estratte da cicflowmeter Python.")
    else:
        print("POOR: Le feature sono troppo diverse.")
        print("Consiglio: Usa il CICFlowMeter Java originale o retrain su nuove feature.")


def main():
    print("=" * 75)
    print("CICFlowMeter Python vs Original CSV - Compatibility Test")
    print("=" * 75)
    
    if not CICFLOWMETER_AVAILABLE:
        print("\nInstalla cicflowmeter e riprova.")
        return
    
    monkey_patch_cicflowmeter()
    
    # Paths (modifica secondo i tuoi file)
    pcap_path = "data/pcap/Tuesday-WorkingHours.pcap"
    original_csv_path = "data/raw/Tuesday-WorkingHours.pcap_ISCX.csv"
    output_csv_path = "/tmp/cicflowmeter_output.csv"
    
    # Check files exist
    if not Path(original_csv_path).exists():
        print(f"CSV originale non trovato: {original_csv_path}")
        return
    
    if not Path(pcap_path).exists():
        print(f"PCAP non trovato: {pcap_path}")
        print("\nSe non hai il PCAP, puoi comunque confrontare i nomi delle feature:")
        print("  1. Genera un CSV di test con cicflowmeter su qualsiasi PCAP")
        print("  2. Confronta i nomi delle colonne con il CSV originale")
        return
    
    # Extract with Python cicflowmeter
    print("\n[1] Extracting features with Python cicflowmeter...")
    df_python = extract_with_cicflowmeter(pcap_path, output_csv_path)
    
    if df_python is None:
        print("Failed to extract features with cicflowmeter")
        return
    
    # Load original CSV
    print("\n[2] Loading original CICFlowMeter CSV...")
    df_original = load_original_csv(original_csv_path)
    print(f"Original CSV: {len(df_original)} flows")
    
    # Compare feature names
    print("\n[3] Comparing feature names...")
    common, only_python, only_original = compare_feature_names(df_python, df_original)
    
    # Compare feature values
    print("\n[4] Comparing feature value distributions...")
    comparison_df = compare_feature_values(df_python, df_original, common)
    
    # Print results
    print_comparison_results(comparison_df)
    
    # Cleanup
    if Path(output_csv_path).exists():
        Path(output_csv_path).unlink()


if __name__ == "__main__":
    main()
