# NIDS-ML: Network Intrusion Detection System

**Machine Learning-based Network Intrusion Detection System con Pipeline Completa**

---

## Indice

1. [Panoramica](#panoramica)
2. [Architettura](#architettura)
3. [Requisiti di Sistema](#requisiti-di-sistema)
4. [Installazione](#installazione)
5. [Pipeline di Esecuzione](#pipeline-di-esecuzione)
6. [Moduli Principali](#moduli-principali)
7. [Modelli Implementati](#modelli-implementati)
8. [Sistema di Selezione Modelli](#sistema-di-selezione-modelli)
9. [Deployment e Monitoring](#deployment-e-monitoring)
10. [Performance e Ottimizzazioni](#performance-e-ottimizzazioni)
11. [Configurazione Avanzata](#configurazione-avanzata)
12. [Troubleshooting](#troubleshooting)

---

## Panoramica

NIDS-ML è un sistema completo di rilevamento intrusioni di rete basato su machine learning. Il progetto implementa una pipeline end-to-end che va dall'acquisizione dati al deployment in produzione, utilizzando il dataset CIC-IDS2017.

**Caratteristiche principali:**

- Classificazione binaria (BENIGN vs ATTACK) con accuracy >99.8%
- Supporto classificazione multiclasse per tipologie di attacco specifiche
- Pipeline automatizzata di preprocessing e feature engineering
- Training parallelo di tre algoritmi ensemble (Random Forest, XGBoost, LightGBM)
- Sistema di selezione automatica del modello migliore tramite scorecard con hard constraints
- Sniffer real-time con supporto cattura live e analisi PCAP
- Modalità prevention con blocco automatico IP malevoli via iptables
- Sistema di timing e performance logging integrato
- Gestione ottimizzata delle risorse (CPU affinity, RAM limits)

---

## Architettura

### Struttura del Progetto

```
NIDS-ML/
├── data/
│   ├── raw/                    # CSV originali CIC-IDS2017
│   └── processed/              # Dataset processati (train/val/test)
├── src/
│   ├── training/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── lightgbm_model.py
│   ├── preprocessing.py        # Pulizia, encoding, bilanciamento
│   ├── feature_engineering.py  # Scaling, selezione feature
│   ├── evaluation.py           # Metriche e visualizzazioni
│   ├── compare_models.py       # Scorecard e selezione best model
│   ├── sniffer.py              # Network sniffer real-time/PCAP
│   ├── timing.py               # Performance logging
│   └── utils.py                # Utility e gestione risorse
├── models/
│   ├── random_forest/
│   ├── xgboost/
│   ├── lightgbm/
│   └── best_model/             # Modello selezionato per produzione
├── artifacts/
│   ├── scaler.pkl
│   ├── selected_features.json
│   └── feature_importances.json
├── logs/
│   └── timing/                 # Log performance esecuzioni
├── reports/
│   └── timing/                 # Report aggregati
└── notebooks/
    └── nids_ml_pipeline.ipynb  # Pipeline notebook per Kaggle
```

### Flusso Dati

```
CSV Raw → Preprocessing → Feature Engineering → Training → Evaluation → Comparison → Best Model → Deployment
```

Ogni fase salva artifacts intermedi per permettere ripresa e debug.

---

## Requisiti di Sistema

### Hardware Minimo

- CPU: 4 core (8+ consigliati)
- RAM: 16 GB (32 GB consigliati per training completo)
- Storage: 10 GB liberi
- GPU: Opzionale (XGBoost supporta CUDA)

### Software

- Python 3.10+
- Sistema operativo: Linux (consigliato), macOS, Windows
- Privilegi root per funzionalità sniffer live e prevention mode

### Dataset

CIC-IDS2017 disponibile da:
- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- Formato: CSV (lunedì-venerdì, 5 file principali)
- Dimensione totale: ~8 GB

---

## Installazione

### Setup Locale

```bash
# Clone repository
git clone https://github.com/Riiccardob/NIDS-ML-SSR2.git
cd NIDS-ML-SSR2

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt

# Scarica dataset CIC-IDS2017
# Posiziona i CSV in data/raw/
```

### Setup Kaggle/Colab

Il notebook `notebooks/nids_ml_pipeline.ipynb` gestisce automaticamente:
- Clone repository da GitHub
- Installazione dipendenze
- Download dataset da Kaggle datasets
- Configurazione ambiente

---

## Pipeline di Esecuzione

### Esecuzione Standard (Locale)

```bash
# 1. Preprocessing
python src/preprocessing.py --balance-ratio 2.0 --n-jobs 4

# 2. Feature Engineering
python src/feature_engineering.py --n-features 30 --rf-estimators 100 --n-jobs 4

# 3. Training modelli
python src/training/random_forest.py --n-iter 20 --cv 3 --n-jobs 4
python src/training/xgboost_model.py --n-iter 50 --cv 5 --gpu  # se GPU disponibile
python src/training/lightgbm_model.py --n-iter 50 --cv 5 --n-jobs 4

# 4. Evaluation
python src/evaluation.py --model-path models/random_forest/model_binary.pkl
python src/evaluation.py --model-path models/xgboost/model_binary.pkl
python src/evaluation.py --model-path models/lightgbm/model_binary.pkl

# 5. Selezione best model
python src/compare_models.py --max-fpr 0.02 --max-latency-ms 2.0

# 6. Test sniffer
sudo python src/sniffer.py --interface eth0 --verbose
```

### Test Rapido

```bash
# Preprocessing standard
python src/preprocessing.py --n-jobs 4

# Feature engineering veloce
python src/feature_engineering.py --rf-estimators 20 --n-jobs 4

# Training test veloce (solo XGBoost)
python src/training/xgboost_model.py --n-iter 5 --cv 2 --n-jobs 4

# Evaluation e comparison
python src/evaluation.py --model-path models/xgboost/model_binary.pkl
python src/compare_models.py
```

---

## Moduli Principali

### preprocessing.py

**Funzioni:**
- Caricamento e concatenazione CSV multipli
- Pulizia dati (NaN, infiniti, duplicati)
- Encoding label (binario e multiclasse)
- Bilanciamento dataset tramite undersampling con ratio configurabile
- Split stratificato train/validation/test

**Parametri chiave:**
- `--balance-ratio`: Rapporto majority:minority (default 2.0)
- `--no-balance`: Disabilita bilanciamento
- `--chunk-size`: Elaborazione chunked per RAM limitata

**Output:** `data/processed/{train,val,test}.parquet`

### feature_engineering.py

**Funzioni:**
- Standardizzazione feature via StandardScaler
- Selezione feature tramite importanza Random Forest
- Salvataggio artifacts (scaler, feature selezionate, importanze)

**Parametri chiave:**
- `--n-features`: Numero feature da selezionare (default 30)
- `--rf-estimators`: Alberi RF per selezione (default 100)

**Output:** `artifacts/{scaler.pkl, selected_features.json, feature_importances.json}`

### training/{random_forest,xgboost_model,lightgbm_model}.py

**Funzioni:**
- RandomizedSearchCV per hyperparameter tuning
- Early stopping (XGBoost, LightGBM)
- Salvataggio modello e risultati
- Tracking timing e metriche

**Parametri chiave:**
- `--n-iter`: Iterazioni random search
- `--cv`: Fold cross-validation
- `--gpu`: Abilita GPU (solo XGBoost)

**Output:** `models/{nome_modello}/{model_binary.pkl, results_binary.json, features_binary.json}`

### evaluation.py

**Funzioni:**
- Calcolo metriche complete su test set
- Misurazione latenza inferenza (CRITICO per NIDS real-time)
- Generazione visualizzazioni (confusion matrix, ROC curve, PR curve)
- Report JSON e testo

**Output:** `reports/{nome_modello}/*.{json,txt,png}`

### compare_models.py

**Sistema Scorecard con Hard Constraints:**

1. **Hard Constraints (eliminatori):**
   - FPR (False Positive Rate) ≤ soglia (default 2%)
   - Latenza media predizione ≤ soglia (default 2.0 ms)

2. **Soft Ranking (tra i modelli PASS):**
   - Formula score: `0.50 * recall + 0.30 * f1 + 0.20 * latency_score`
   - Priorità: sicurezza (recall) > bilanciamento (F1) > velocità

**Parametri:**
- `--max-fpr`: Soglia FPR massima (default 0.02 = 2%)
- `--max-latency-ms`: Latenza massima ms (default 2.0)

**Output:** `models/best_model/` (copia completa del modello migliore)

### sniffer.py

**Architettura Producer-Consumer:**

```
Thread PRODUCER → Queue (10k max) → Thread CONSUMER → Analisi ML
                                  ↓
                            Thread EXPIRE (controllo timeout)
```

**Modalità:**

- **DETECTION:** Solo logging, nessuna modifica sistema
- **PREVENTION:** Logging + blocco IP malevoli via iptables

**Funzioni:**
- Cattura live da interfaccia di rete
- Analisi PCAP offline con streaming (memory-efficient)
- Aggregazione pacchetti in flussi
- Estrazione feature compatibili CIC-IDS2017
- Predizione real-time con threshold configurabile
- Logging avanzato (generale, attacchi, flussi, firewall)

**Parametri chiave:**
- `--interface`: Interfaccia per cattura live
- `--pcap`: File PCAP da analizzare
- `--mode`: detection o prevention
- `--threshold`: Soglia probabilità (default 0.5)
- `--timeout`: Timeout flusso secondi (default 15)

---

## Modelli Implementati

### Random Forest

**Configurazione:**
- Classificatore ensemble con votazione
- Parallelizzazione multi-core
- Bilanciamento classi tramite `class_weight='balanced'`

**Hyperparameter Search:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}
```

**Performance:**
- Accuracy: 99.90%
- F1: 99.85%
- Latenza: ~0.77 ms/sample
- Throughput: 1.3M samples/sec

### XGBoost

**Configurazione:**
- Gradient boosting ottimizzato
- Supporto GPU (CUDA)
- Early stopping su validation set
- Scale_pos_weight automatico per bilanciamento

**Hyperparameter Search:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}
```

**Performance:**
- Accuracy: 99.82%
- F1: 99.73%
- Latenza: ~0.76 ms/sample
- Throughput: 1.3M samples/sec

### LightGBM

**Configurazione:**
- Leaf-wise tree growth
- Parallelizzazione ottimizzata (n_jobs=1 per CV, LightGBM parallelizza internamente)
- Early stopping
- Force_col_wise per dataset wide

**Hyperparameter Search:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70, 100],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [10, 20, 30, 50],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1]
}
```

**Performance:**
- Accuracy: 99.85%
- F1: 99.78%
- Latenza: ~2.46 ms/sample
- Throughput: 406K samples/sec

---

## Sistema di Selezione Modelli

### Logica Scorecard

**Fase 1: Eliminazione Hard Constraints**

```python
if FPR > 0.02:  # 2% massimo
    REJECT  # Troppi falsi allarmi in produzione

if latency_per_sample > 2.0 ms:
    REJECT  # Troppo lento per real-time
```

**Fase 2: Ranking Modelli PASS**

```python
# Formula bilanciata per NIDS
score = 0.50 * recall + 0.30 * f1 + 0.20 * latency_score

# Latency score normalizzato [0,1]
latency_score = max(0, 1 - (latency_ms / threshold_ms))
```

**Motivazione:**
- **Recall 50%:** Sicurezza prioritaria. 1% FNR = 1% attacchi non rilevati.
- **F1 30%:** Bilanciamento precision/recall per performance generale.
- **Latenza 20%:** Efficienza necessaria per deployment real-time.

### Interpretazione Risultati

```
PASS - Modello soddisfa entrambi i constraints
FAIL - Modello violato almeno un constraint

Score finale valido solo per modelli PASS
```

**Esempio Output:**

```
Model            FPR        Latency    Recall     Score      Status
random_forest    0.0009     0.7677     0.9989     10085.83   PASS
xgboost          0.0024     0.7555     0.9995     10074.35   PASS
lightgbm         0.0018     2.4622     0.9992     10079.06   PASS

Best Model: RANDOM_FOREST
```

---

## Deployment e Monitoring

### Deployment Sniffer

**Modalità Detection (solo logging):**

```bash
# Cattura live
sudo python src/sniffer.py --interface eth0 --threshold 0.5 --verbose

# Analisi PCAP
python src/sniffer.py --pcap capture.pcap --threshold 0.3 --min-packets 1
```

**Modalità Prevention (blocco IP):**

```bash
# ATTENZIONE: Richiede root e modifica iptables
sudo python src/sniffer.py --interface eth0 --mode prevention --threshold 0.5
```

**Parametri Critici:**

- `--threshold`: Soglia probabilità. Valori più bassi = più sensibile, più falsi positivi.
  - 0.3: Molto sensibile (detection anticipata, più FP)
  - 0.5: Bilanciato (standard)
  - 0.8: Conservativo (solo attacchi evidenti)

- `--min-packets`: Minimo pacchetti per analisi flusso.
  - 1: Analizza tutto (overhead alto)
  - 2-3: Standard (bilanciamento)
  - 5+: Solo flussi significativi

- `--timeout`: Timeout flusso secondi.
  - 15s: Test/sviluppo (basso per analisi frequente)
  - 60s: Produzione (bilanciamento)
  - 300s: Flussi lunghi (P2P, trasferimenti)

### Logging e Monitoring

**File Log Generati:**

```
logs/
└── sniffer_YYYYMMDD_HHMMSS.log         # Log generale
    attacks_YYYYMMDD_HHMMSS.log         # Solo attacchi
    flows_YYYYMMDD_HHMMSS.jsonl         # Dettaglio flussi (JSONL)
    firewall_YYYYMMDD_HHMMSS.log        # Azioni iptables
    stats_YYYYMMDD_HHMMSS.json          # Statistiche sessione
```

**Statistiche Sessione:**

```json
{
  "session_id": "20260120_150000",
  "start_time": "2026-01-20T15:00:00",
  "end_time": "2026-01-20T15:30:00",
  "packets_captured": 1500000,
  "packets_processed": 1500000,
  "flows_analyzed": 25000,
  "attacks_detected": 350,
  "benign_flows": 24650,
  "ips_blocked": 15,
  "detection_rate": 1.4
}
```

### Sistema Timing

**Tracking Performance:**

```bash
# Ogni esecuzione salva timing in logs/timing/
# Report aggregato
python src/timing.py --report
```

**Output Report:**

```
MODULO: preprocessing
  Esecuzioni: 5
  Tempo totale:
    Min:    450.2s
    Max:    520.8s
    Media:  485.5s
  
  Operazioni:
    caricamento_csv:
      Min:    120.5s
      Media:  135.2s
    pulizia_dati:
      Min:    85.3s
      Media:  92.1s
```

---

## Performance e Ottimizzazioni

### Gestione Risorse CPU

**Problema:** Sklearn, XGBoost, LightGBM usano tutti i core disponibili per default, causando saturazione CPU e slowdown.

**Soluzione:** Limiti CPU applicati a tre livelli.

**1. CPU Affinity (più efficace):**

```python
import psutil
p = psutil.Process()
p.cpu_affinity(list(range(4)))  # Usa solo core 0-3
```

**2. Variabili Ambiente (thread pools):**

```python
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
```

**3. Parametri Modello:**

```python
RandomForestClassifier(n_jobs=4)
RandomizedSearchCV(n_jobs=4)
```

**Applicazione Automatica:**

Tutti gli script applicano limiti automaticamente:

```bash
python src/training/random_forest.py --n-jobs 4
# Imposta automaticamente affinity + env + parametri
```

### Ottimizzazione Memoria

**Problema:** Dataset CIC-IDS2017 completo richiede ~8-12 GB RAM.

**Soluzioni Implementate:**

1. **Chunked Loading:**
   ```bash
   python src/preprocessing.py --chunk-size 100000
   ```

2. **Garbage Collection Aggressivo:**
   ```python
   del large_dataframe
   gc.collect()
   ```

3. **Parquet invece di CSV:**
   - Compressione ~70%
   - Lettura 3-5x più veloce

4. **Feature Selection:**
   - 30 feature invece di 77 originali
   - Riduzione memoria ~60%

### Ottimizzazione Training

**LightGBM - Parallelizzazione:**

**Problema:** Con `n_jobs > 1` in RandomizedSearchCV + parallelismo interno LightGBM → conflitti thread → lentezza.

**Soluzione:**

```python
# RandomizedSearchCV: n_jobs=1 (serializza CV)
# LightGBM: n_jobs=N (parallelizza ogni singolo fit)

search = RandomizedSearchCV(
    estimator=LGBMClassifier(n_jobs=4, force_col_wise=True),
    n_jobs=1,  # Serializza CV, evita conflitti
    ...
)
```

**XGBoost - GPU:**

```bash
# Auto-detect e usa GPU se disponibile
python src/training/xgboost_model.py --gpu

# Forza CPU
python src/training/xgboost_model.py --no-gpu
```

---

## Configurazione Avanzata

### Bilanciamento Dataset

```bash
# Standard (2:1 majority:minority)
python src/preprocessing.py --balance-ratio 2.0

# Più bilanciato (1.5:1)
python src/preprocessing.py --balance-ratio 1.5

# Molto sbilanciato (5:1, per testare robustezza)
python src/preprocessing.py --balance-ratio 5.0

# Nessun bilanciamento (usa dataset originale)
python src/preprocessing.py --no-balance
```

### Feature Engineering

```bash
# Più feature (trade-off accuracy vs latenza)
python src/feature_engineering.py --n-features 50

# Meno feature (più veloce, leggermente meno accurato)
python src/feature_engineering.py --n-features 20

# Selezione più accurata (più alberi RF)
python src/feature_engineering.py --rf-estimators 200
```

### Hyperparameter Tuning

```bash
# Test rapido (5 iterazioni, 2 fold)
python src/training/xgboost_model.py --n-iter 5 --cv 2

# Standard (20 iterazioni, 3 fold = 60 fit)
python src/training/xgboost_model.py --n-iter 20 --cv 3

# Ricerca estensiva (50 iterazioni, 5 fold = 250 fit)
python src/training/xgboost_model.py --n-iter 50 --cv 5
```

### Scorecard Customization

```bash
# Più restrittivo su FPR (0.5% massimo)
python src/compare_models.py --max-fpr 0.005

# Meno restrittivo su latenza (modelli più complessi OK)
python src/compare_models.py --max-latency-ms 5.0

# Combinazione
python src/compare_models.py --max-fpr 0.01 --max-latency-ms 1.0
```

---

## Troubleshooting

### Errore: "Modello non trovato"

**Causa:** Esecuzione evaluation/comparison prima di training.

**Soluzione:**

```bash
# Verifica modelli disponibili
ls models/*/model_binary.pkl

# Se mancanti, esegui training
python src/training/xgboost_model.py
```

### Errore: "Artifacts non trovati"

**Causa:** Feature engineering non eseguito.

**Soluzione:**

```bash
python src/feature_engineering.py
```

### Errore: "Data processed non trovati"

**Causa:** Preprocessing non eseguito.

**Soluzione:**

```bash
python src/preprocessing.py
```

### CPU Saturation (100% uso)

**Causa:** Limiti CPU non applicati correttamente.

**Soluzione:**

```bash
# Forza limiti espliciti
python src/training/random_forest.py --n-jobs 4

# Verifica affinity
python -c "import psutil; print(psutil.Process().cpu_affinity())"
```

### Out of Memory

**Soluzione 1: Chunked Loading**

```bash
python src/preprocessing.py --chunk-size 50000
```

**Soluzione 2: Riduci Feature**

```bash
python src/feature_engineering.py --n-features 20
```

**Soluzione 3: Libera RAM**

```bash
# Chiudi altre applicazioni
# Aumenta swap se disponibile
```

### Sniffer: "Permission denied"

**Causa:** Cattura live richiede privilegi root.

**Soluzione:**

```bash
sudo python src/sniffer.py --interface eth0
```

### Sniffer: "Scapy not available"

**Soluzione:**

```bash
pip install scapy
```

### PCAP Analysis Too Slow

**Causa:** Analisi completa di PCAP grandi.

**Soluzione:**

```bash
# Aumenta min-packets (analizza solo flussi significativi)
python src/sniffer.py --pcap large.pcap --min-packets 5

# Aumenta timeout (meno controlli scadenza)
python src/sniffer.py --pcap large.pcap --timeout 300
```

---

## Autori

Progetto sviluppato per il corso di Sicurezza dei Sistemi e delle Reti 2.

**Repository:** [NIDS-ML-SSR2](https://github.com/Riiccardob/NIDS-ML-SSR2)

**Licenza:** MIT