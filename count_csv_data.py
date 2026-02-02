import pandas as pd
import os
from glob import glob
# ==========================================
# CONFIGURAZIONE
# ==========================================
# Modifica questo percorso se i tuoi CSV sono altrove
DATA_DIR = "data/raw" 
# ==========================================
def main():
    # Cerca tutti i file .csv nella cartella
    files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))

    if not files:
        print(f" Nessun file CSV trovato in: {os.path.abspath(DATA_DIR)}")
        print("Assicurati di aver scaricato i CSV e di aver impostato il percorso corretto.")
        return
    print(f"\n{'='*100}")
    print(f"ANALISI GROUND TRUTH (VERITÀ ASSOLUTA) DEI CSV")
    print(f"{'='*100}")
    print(f"{'NOME FILE':<55} | {'ATTACCHI':<12} | {'TOTALE':<12} | {'% ATTACCHI'}")
    print(f"{'-'*100}")
    total_attacks_global = 0
    for f in files:
        try:
            filename = os.path.basename(f)

            # Leggiamo solo le colonne che contengono 'Label' per essere velocissimi (evitiamo di caricare tutto in RAM)
            # CIC-IDS2017 ha spesso uno spazio davanti al nome colonna, es: " Label"
            df = pd.read_csv(f, nrows=1) # Leggiamo solo header per trovare la colonna giusta
            label_col = [c for c in df.columns if 'label' in c.lower()]

            if not label_col:
                print(f"{filename[:53]:<55} |  Colonna Label non trovata")
                continue

            col_name = label_col[0]

            # Ora leggiamo solo quella colonna
            df = pd.read_csv(f, usecols=[col_name], encoding='latin-1', low_memory=False)

            # Pulizia etichette (spazi vuoti, maiuscole/minuscole)
            df[col_name] = df[col_name].astype(str).str.strip().str.upper()

            # Conteggi
            counts = df[col_name].value_counts()
            n_benign = counts.get('BENIGN', 0)
            n_total = len(df)
            n_attacks = n_total - n_benign

            total_attacks_global += n_attacks

            pct = (n_attacks / n_total) * 100

            # Stampa riga tabella
            print(f"{filename[:53]:<55} | {n_attacks:<12,} | {n_total:<12,} | {pct:.2f}%")

            # Opzionale: stampa quali attacchi ci sono
            # attack_types = [idx for idx in counts.index if idx != 'BENIGN']
            # if attack_types:
            #    print(f"   -> Tipi: {attack_types}")
        except Exception as e:
            print(f"Errore su {filename}: {e}")
    print(f"{'-'*100}")
    print(f"TOTALE ATTACCHI NEL DATASET: {total_attacks_global:,}")
    print(f"{'='*100}\n")
if __name__ == "__main__":
    main()
