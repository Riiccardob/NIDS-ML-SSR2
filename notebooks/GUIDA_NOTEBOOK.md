# 📘 Guida Notebook NIDS-ML Pipeline

## 🎯 Overview

Il notebook `nids_pipeline_complete.ipynb` esegue l'intera pipeline di preparazione dati per NIDS-ML, dalla raccolta del dataset raw fino alla generazione dei dataset pronti per il training.

### 📋 Cosa fa il notebook:

1. ✅ **Environment Detection** - Rileva automaticamente Kaggle vs Locale
2. ✅ **Repository Setup** - Scarica solo `src/` da GitHub (su Kaggle)
3. ✅ **Dataset Import** - Copia dataset da Kaggle Input o verifica locale
4. ✅ **Preprocessing** - Esegue `preprocessing.py`
5. ✅ **Feature Engineering** - Esegue `feature_engineering.py` con Statistical + RobustScaler
6. ✅ **Validation** - Verifica artifacts e mostra summary
7. ✅ **Export** - Crea ZIP con artifacts (solo Kaggle)

---

## 🚀 Quick Start

### Su Kaggle:

1. **Crea nuovo Notebook** su Kaggle
2. **Aggiungi Dataset**: "Network Intrusion Dataset" (il tuo dataset pubblico)
3. **Import**: Copia il contenuto di `nids_pipeline_complete.ipynb`
4. **Configura** (prima cella):
   ```python
   CLEAN_RUN = True
   USE_STATISTICAL = True
   USE_ROBUST = True
   N_FEATURES = 30
   RF_ESTIMATORS = 100
   ```
5. **Run All**: Esegui tutte le celle

### In Locale:

1. **Posizionati** nella root del repository
2. **Verifica** che `data/raw/*.csv` contenga i dataset
3. **Apri** il notebook con Jupyter:
   ```bash
   jupyter notebook nids_pipeline_complete.ipynb
   ```
4. **Configura** e **Run All**

---

## ⚙️ Configurazione

### Parametri Principali (Cella 1):

```python
# Clean Run
CLEAN_RUN = True  # True = riparte da zero, False = riusa esistenti

# Repository (solo Kaggle)
REPO_URL = "https://github.com/riiccardob/nids-ml-ssr2"
BRANCH = "main"

# Dataset Paths
KAGGLE_DATASET_PATH = "/kaggle/input/network-intrusion-dataset/"
LOCAL_DATASET_PATH = "data/raw"

# Feature Engineering
USE_STATISTICAL = True   # Statistical preprocessing (CONSIGLIATO)
USE_ROBUST = True        # RobustScaler (CONSIGLIATO)
N_FEATURES = 30          # Feature da selezionare
RF_ESTIMATORS = 100      # Alberi Random Forest
```

### Opzioni Avanzate:

#### CLEAN_RUN

| Valore | Comportamento |
|--------|---------------|
| `True` | Cancella `data/processed` e `artifacts`, riparte da zero |
| `False` | Riusa file esistenti se presenti (utile per debugging) |

**Quando usare `True`:**
- Prima esecuzione
- Cambio configurazione feature engineering
- Problemi con artifacts corrotti

**Quando usare `False`:**
- Re-run dopo errore in fase successiva
- Test su step specifico
- Sviluppo/debugging

#### Feature Engineering Config

| Parametro | Range | Default | Descrizione |
|-----------|-------|---------|-------------|
| `USE_STATISTICAL` | bool | `True` | Statistical preprocessing (variance + correlation) |
| `USE_ROBUST` | bool | `True` | RobustScaler (migliore per outlier) |
| `N_FEATURES` | 5-100 | 30 | Numero feature da selezionare |
| `RF_ESTIMATORS` | 10-1000 | 100 | Alberi Random Forest per importance |

**Configurazione CONSIGLIATA:**
```python
USE_STATISTICAL = True
USE_ROBUST = True
N_FEATURES = 30
RF_ESTIMATORS = 100
```

---

## 📂 Struttura Output

Dopo l'esecuzione completa:

```
/kaggle/working/  (o directory locale)
├── data/
│   ├── raw/
│   │   ├── cicids2017_*.csv
│   │   └── ...
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── train_ready.parquet  ← Dataset pronti per training
│       ├── val_ready.parquet
│       └── test_ready.parquet
├── artifacts/
│   ├── scaler.pkl
│   ├── selected_features.json
│   ├── feature_importances.json
│   ├── scaler_columns.json
│   ├── column_checksum.json
│   └── statistical_preprocessing_info.json  ← Se USE_STATISTICAL=True
├── logs/
│   └── timing/
│       ├── preprocessing_*.json
│       └── feature_engineering_v2_*.json
├── src/  (solo Kaggle, scaricato da GitHub)
└── pipeline_artifacts.zip  (solo Kaggle, per download)
```

---

## 🔍 Validazione Output

Il notebook verifica automaticamente:

### 1. Dataset Ready
```
TRAIN - Shape:     706,632 samples x 31 features  (30 + target)
VAL   - Shape:     151,422 samples x 31 features
TEST  - Shape:     151,422 samples x 31 features
```

### 2. Feature Selection
```
Selezionate: 30 features
```

### 3. Top-10 Feature Importances
```
  1. Bwd Packet Length Std              0.107700
  2. Bwd Packet Length Max              0.075900
  3. Packet Length Variance             0.061300
  ...
```

### 4. Scaler Info
```
Tipo: RobustScaler
Statistical Preprocessing: ATTIVO
  - Feature ridotte: 15.2%
  - Varianza rimossa: 3
  - Correlazione rimossa: 9
```

---

## 🐛 Troubleshooting

### Problema: "Dataset non trovato"

**Kaggle:**
```python
# Verifica path in configurazione
KAGGLE_DATASET_PATH = "/kaggle/input/network-intrusion-dataset/"

# Controlla che il dataset sia aggiunto al notebook
# Kaggle UI: Add Data > Search "network intrusion"
```

**Locale:**
```bash
# Verifica presenza CSV
ls data/raw/*.csv

# Se mancano, scarica o copia i file CSV
```

### Problema: "src/ non trovata"

**Kaggle:**
```python
# Verifica URL e branch
REPO_URL = "https://github.com/riiccardob/nids-ml-ssr2"
BRANCH = "main"

# Forza re-download cancellando src/
import shutil
shutil.rmtree("src")  # Poi ri-esegui cella setup
```

**Locale:**
```bash
# Assicurati di essere nella root del repo
pwd  # Deve mostrare .../nids-ml-ssr2

# Verifica presenza src/
ls src/
```

### Problema: "Memory Error"

**Kaggle:**
- Usa notebook con **30GB RAM** (Settings > Accelerator > None, ma aumenta RAM)
- Riduci `RF_ESTIMATORS` (es. 50 invece di 100)
- Assicurati di non avere altri notebook running

**Locale:**
- Chiudi applicazioni pesanti
- Riduci `RF_ESTIMATORS`
- Monitora RAM con `htop` o Task Manager

### Problema: "Preprocessing fallito"

**Verifica:**
```python
# Controlla log dettagliato
!cat logs/*.log  # Se esiste
```

**Soluzioni:**
- CSV corrotti? Ri-scarica dataset
- Encoding problemi? Verifica CSV con `head -n 10 data/raw/file.csv`
- Memoria insufficiente? Vedi sezione Memory Error

### Problema: "Feature Engineering lento"

**Normale per grandi dataset:**
- RF con 100 alberi su 700k samples richiede 2-3 minuti
- Statistical preprocessing è veloce (<10s)

**Accelera:**
```python
RF_ESTIMATORS = 50  # Invece di 100
N_FEATURES = 20     # Invece di 30
```

---

## 📊 Metriche & Performance

### Tempi Attesi (Kaggle, 4 CPU, 30GB RAM):

| Step | Tempo | Dettagli |
|------|-------|----------|
| Setup Repo | ~30s | Download + unzip + pip install |
| Dataset Import | ~10s | Copy CSV (se non cached) |
| Preprocessing | ~60s | Caricamento + pulizia + split |
| Feature Engineering | ~150s | Statistical (10s) + Scaling (5s) + RF (135s) |
| **TOTALE** | **~4 min** | Prima esecuzione |

**Run successive (CLEAN_RUN=False):** ~10s (solo validation)

### Dimensioni Output:

| File | Dimensione | Comprimibile |
|------|------------|--------------|
| train.parquet | ~150 MB | No (già compresso) |
| val.parquet | ~35 MB | No |
| test.parquet | ~35 MB | No |
| train_ready.parquet | ~50 MB | No |
| artifacts/*.pkl | ~5 MB | Sì |
| **pipeline_artifacts.zip** | **~100 MB** | - |

---

## 🎓 Best Practices

### 1. Development Workflow

```python
# Prima iterazione: tutto pulito
CLEAN_RUN = True
# ... esegui pipeline completa

# Debugging: riusa preprocessing
CLEAN_RUN = False
# ... modifica solo feature engineering
```

### 2. Configurazione Ottimale

**Per massima qualità:**
```python
USE_STATISTICAL = True
USE_ROBUST = True
N_FEATURES = 30
RF_ESTIMATORS = 200  # Più alberi = più tempo ma migliori feature
```

**Per velocità (testing):**
```python
USE_STATISTICAL = True
USE_ROBUST = True
N_FEATURES = 20
RF_ESTIMATORS = 50
```

**Per baseline (confronto):**
```python
USE_STATISTICAL = False
USE_ROBUST = False
N_FEATURES = 30
RF_ESTIMATORS = 100
```

### 3. Su Kaggle

- ✅ Usa **Session Persistence** (salva automaticamente tra run)
- ✅ Abilita **Internet** per download repository
- ✅ Monitora **RAM usage** (Kaggle limite 30GB)
- ✅ Scarica **pipeline_artifacts.zip** al termine (backup)

### 4. In Locale

- ✅ Usa **ambiente virtuale** Python
- ✅ Installa requirements: `pip install -r requirements.txt`
- ✅ Verifica **versioni librerie** (potrebbero differire da Kaggle)
- ✅ Backup periodico di `artifacts/`

---

## 🔄 Integrazione con Training Notebooks

Dopo aver completato questo notebook, i prossimi step sono:

### 1. Random Forest Tuning
```python
# Nuovo notebook: nids_training_random_forest.ipynb
# - Carica artifacts/ e data/processed/
# - Hyperparameter tuning con Optuna
# - Salva best model in models/random_forest/
```

### 2. XGBoost Tuning
```python
# Nuovo notebook: nids_training_xgboost.ipynb
# - Stessa struttura
# - Tuning specifico per XGBoost
```

### 3. LightGBM Tuning
```python
# Nuovo notebook: nids_training_lightgbm.ipynb
# - Stessa struttura
# - Tuning specifico per LightGBM
```

**Vantaggi separazione:**
- ✅ Parallelizzazione: run su 3 notebook Kaggle simultanei
- ✅ Riutilizzo: stesso preprocessing per tutti i modelli
- ✅ Confronto: metriche indipendenti per ogni algoritmo
- ✅ Debugging: isolamento errori per modello

---

## 📝 Note Finali

### Differenze Kaggle vs Locale:

| Aspetto | Kaggle | Locale |
|---------|--------|--------|
| Download repo | Automatico | Manuale (git clone) |
| Dataset path | `/kaggle/input/...` | `data/raw/` |
| Working dir | `/kaggle/working/` | Repository root |
| Export artifacts | ZIP automatico | Non necessario |
| Persistenza | Session Persistence | Permanente |

### Compatibilità:

- ✅ **Python**: 3.8+
- ✅ **Pandas**: 1.3+
- ✅ **Scikit-learn**: 1.0+
- ✅ **NumPy**: 1.21+
- ✅ **Kaggle Docker**: `gcr.io/kaggle-gpu-images/python` (latest)

### Sicurezza:

- ⚠️ **Non committare** file `.ipynb` con output (contengono dati)
- ⚠️ **Non condividere** artifacts con dati sensibili
- ✅ **Usa** `.gitignore` per `*.parquet`, `artifacts/`

---

## ❓ FAQ

**Q: Posso modificare i file in `src/` su Kaggle?**
A: Sì, ma le modifiche vanno perse al restart. Meglio modificare nel repo e ri-scaricare.

**Q: Quanto spazio disco serve in locale?**
A: ~2GB (dataset raw ~1GB + processed ~1GB + artifacts ~100MB)

**Q: Posso usare GPU?**
A: No, la pipeline non usa GPU. CPU è sufficiente.

**Q: CLEAN_RUN cancella anche il dataset raw?**
A: No, cancella solo `data/processed` e `artifacts`. Il dataset raw è preservato.

**Q: Posso testare su subset del dataset?**
A: Sì, modifica `preprocessing.py` per usare `sample(frac=0.1)` durante lo sviluppo.

---

## 📚 Risorse

- **Repository**: https://github.com/riiccardob/nids-ml-ssr2
- **Dataset**: Kaggle Dataset "Network Intrusion Detection"
- **Documentazione Scikit-learn**: https://scikit-learn.org/stable/
- **Kaggle Notebooks Docs**: https://www.kaggle.com/docs/notebooks

---

**Creato da:** NIDS-ML Team  
**Ultima modifica:** 2026-01-28  
**Versione Notebook:** 2.0
