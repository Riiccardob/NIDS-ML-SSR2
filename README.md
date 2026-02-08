# NIDS NetFlow Pipeline - Chunk-Based Processing

Pipeline ottimizzata per training di modelli NIDS su dataset molto grandi (>70M records) come NF-UQ-NIDS-v2.

##  Caratteristiche Principali

### Parallel Processing 
- **Multiprocessing**: Processa chunk in parallelo usando pool di worker
- **Configurabile**: Imposta percentuale CPU da usare in `config.py`
- **Scalabile**: Automaticamente adatta worker count ai core disponibili
- **Speedup**: ~2-4x più veloce su CPU multi-core (es. 8 core)

### Chunk-Based Processing
- **Preprocessing**: Legge CSV a blocchi di 500k righe, pulisce e salva in Parquet compresso
- **Feature Engineering**: Fit scaler su sample rappresentativo (1M righe), applica scaling chunk-by-chunk
- **Training**: Carica train set in memoria, usa data loaders per validation/test

### Gestione Outlier Corretta
-  **Scaler fittato su dati "sporchi"** (outlier inclusi)
-  Sample rappresentativo del train set PRIMA di qualsiasi pulizia
-  Gestisce correttamente picchi di traffico in produzione

### Ottimizzazioni Memoria
- Mai più di 500k-1M righe in RAM contemporaneamente
- Parquet compresso (~10x più piccolo di CSV)
- Data loaders per evaluation su grandi dataset

##  Requisiti

```bash
# Python 3.8+
pip install pandas numpy scikit-learn xgboost lightgbm pyarrow psutil
```

**Hardware raccomandato per NF-UQ-NIDS-v2 (76M records):**
- RAM: 16GB (usa ~8GB durante processing)
- Storage: ~10GB liberi
- CPU: Multi-core raccomandato (XGBoost/LightGBM usano parallelismo)

##  Struttura Dataset

```
project/
 data/
    raw/              # CSV del dataset (NF-UQ-NIDS-v2)
    processed/        # Parquet generati dalla pipeline
 models/               # Modelli salvati
 artifacts/            # Scaler e feature list
 logs/                 # Log di esecuzione
```

##  Setup Iniziale

### 1. Scarica il Dataset

Scarica NF-UQ-NIDS-v2 da: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

```bash
# Copia il CSV in data/raw/
cp NF-UQ-NIDS-v2.csv data/raw/
```

### 2. Verifica Dataset

```bash
python srcNF/check_dataset.py
```

Questo script:
- Verifica presenza del CSV
- Analizza un sample per controllare le label
- Stima memoria richiesta
- Verifica bilanciamento classi

##  Esecuzione Pipeline

### Opzione 1: Pipeline Completa (Raccomandato)

```bash
# Con XGBoost (default)
python srcNF/pipeline.py

# Con LightGBM
python srcNF/pipeline.py --model lightgbm

# Con Random Forest
python srcNF/pipeline.py --model random_forest
```

**Tempi stimati** (76M records, 16GB RAM, 8 core @ 75% usage):
- Preprocessing: **15-30 min** (parallel) vs 30-60 min (sequential)
- Feature Engineering: 15-30 min
- Training XGBoost: 30-90 min
- **Totale: ~1-2.5 ore** (parallel) vs 1.5-3 ore (sequential)

### Opzione 2: Step-by-Step

```bash
# 1. Preprocessing (CSV → Parquet)
python srcNF/preprocessing.py

# 2. Feature Engineering (Scaling)
python srcNF/feature_engineering.py

# 3. Training
python srcNF/training.py --model xgboost
```

### Opzione 3: Solo Training (Dati già processati)

```bash
python srcNF/pipeline.py --model xgboost --skip-preprocessing
```

##  Output della Pipeline

### 1. Preprocessing Output

```
data/processed/
 train.parquet          # Train set (70%)
 val.parquet            # Validation set (15%)
 test.parquet           # Test set (15%)
```

**Caratteristiche:**
- Split stratificato (mantiene distribuzione classi)
- Formato Parquet compresso (snappy)
- Pulizia dati (NaN, Inf rimossi)
- **Outlier mantenuti** (essenziale per scaler)

### 2. Feature Engineering Output

```
data/processed/
 train_scaled.parquet   # Train set scalato
 val_scaled.parquet     # Validation set scalato
 test_scaled.parquet    # Test set scalato

artifacts/
 scaler.pkl             # RobustScaler fitted
 features.json          # Lista feature + metadata
 features.txt           # Lista leggibile
```

**Note importanti:**
- Scaler fittato su **1M sample del train set CON outlier**
- Feature selection minimale (solo varianza zero e correlazione >0.95)
- Tutte le feature numeriche utilizzate (<50 feature OK per XGBoost/LightGBM)

### 3. Training Output

```
models/xgboost/            # (o lightgbm, random_forest)
 model.pkl              # Modello trained
 metrics.json           # Metriche validation + test

logs/
 preprocessing.log
 feature_engineering.log
 training.log
 pipeline.log
```

##  Configurazione Avanzata

### File: `config.py`

```python
# Chunk processing
CHUNK_SIZE = 500_000              # Righe per chunk (adatta a RAM)
SCALER_SAMPLE_SIZE = 1_000_000    # Sample per scaler fitting

# Parallel processing 
ENABLE_PARALLEL_PROCESSING = True # Abilita/disabilita parallelismo
CPU_USAGE_PERCENT = 0.75          # Usa 75% dei core disponibili
MIN_WORKERS = 2                    # Minimo worker anche se CPU_USAGE_PERCENT è basso
MAX_WORKERS = 16                   # Massimo worker (safety limit)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature selection
CORRELATION_THRESHOLD = 0.95      # Rimuovi feature correlate

# Scaler
SCALER_TYPE = 'robust'            # 'robust' o 'standard'
```

### Ottimizzazione Parallelismo

**CPU_USAGE_PERCENT** controlla quanti core utilizzare:

```python
# Esempi per sistema con 8 core:
CPU_USAGE_PERCENT = 0.50  # 4 workers (usa 50%)
CPU_USAGE_PERCENT = 0.75  # 6 workers (usa 75%, raccomandato)
CPU_USAGE_PERCENT = 1.00  # 8 workers (usa 100%, max performance)
```

**Quando usare 100%**: Se il PC è dedicato al task
**Quando usare 50-75%**: Se vuoi usare il PC durante il processing (raccomandato)

**Speedup atteso**:
- 4 core: ~2-3x più veloce
- 8 core: ~3-5x più veloce
- 16 core: ~5-8x più veloce

**Note**: Lo speedup non è lineare per overhead di sincronizzazione, ma è comunque significativo.

### Adatta Chunk Size alla tua RAM

Con 16GB RAM:
- `CHUNK_SIZE = 500_000`  (default, safe)
- `CHUNK_SIZE = 1_000_000`  (più veloce, usa ~1-2GB per chunk)

Con 8GB RAM:
- `CHUNK_SIZE = 250_000`  (più lento ma safe)

##  Monitoraggio Esecuzione

La pipeline logga costantemente:
- Progresso (% completamento)
- Uso RAM corrente
- Tempo elapsed

**Esempio output:**
```
[2024-02-04 10:30:15] | INFO | Chunk 50/153 (32.7%) - RAM: 45.2%
[2024-02-04 10:30:45] | INFO | Chunk 100/153 (65.4%) - RAM: 47.8%
```

##  Troubleshooting

### 1. Out of Memory durante Preprocessing

**Sintomo**: Crash del PC o processo killed

**Soluzione**:
```python
# In config.py, riduci chunk size
CHUNK_SIZE = 250_000  # Da 500_000
```

### 2. Out of Memory durante Training

**Sintomo**: Crash durante model.fit()

**Soluzione 1** - Usa LightGBM (più leggero):
```bash
python srcNF/training.py --model lightgbm
```

**Soluzione 2** - Riduci parametri XGBoost:
```python
# In config.py
XGBOOST_PARAMS = {
    ...
    'max_bin': 128,  # Da 256
    'max_depth': 5,  # Da 6
}
```

### 3. Training troppo lento

**Causa**: Random Forest su 76M records

**Soluzione**: Usa XGBoost o LightGBM (10-100x più veloci)
```bash
python srcNF/training.py --model xgboost
```

### 4. File Parquet corrotto

**Sintomo**: Errore durante lettura Parquet

**Soluzione**: Riesegui preprocessing
```bash
# Rimuovi file corrotti
rm data/processed/*.parquet

# Riesegui
python srcNF/preprocessing.py
```

### 5. Parallel processing troppo lento o crash

**Causa**: Troppi worker per RAM disponibile

**Soluzione**: Riduci percentuale CPU
```python
# In config.py
CPU_USAGE_PERCENT = 0.50  # Da 0.75 a 0.50
# Oppure disabilita parallelismo
ENABLE_PARALLEL_PROCESSING = False
```

### 6. CPU usage al 100% rallenta il sistema

**Causa**: Troppi worker dedicati

**Soluzione**: Riduci CPU_USAGE_PERCENT
```python
# In config.py
CPU_USAGE_PERCENT = 0.50  # Lascia metà CPU libera
```

##  Performance Attese

### Dataset: NF-UQ-NIDS-v2 (76M records)

**Distribuzione:**
- Benign: 25M (33%)
- Attack: 51M (67%)

**Metriche attese** (XGBoost/LightGBM):
- Accuracy: >99%
- Precision: >98%
- Recall: >99%
- F1: >98%
- FPR: <1%

**Nota**: Random Forest può avere performance leggermente inferiori ma training molto più lento.

##  Note Importanti sullo Scaler

**CRITICAL**: Lo scaler DEVE essere fittato su dati "sporchi" (outlier inclusi)

###  Approccio Corretto (implementato)

1. Prendi sample RAPPRESENTATIVO del train set (1M righe)
2. Sample include TUTTI i dati (outlier inclusi)
3. Fit RobustScaler su questo sample
4. Apply scaling a tutti i dataset

###  Approccio SBAGLIATO (evitato)

1. ~~Rimuovi outlier dal train set~~
2. ~~Fit scaler su dati "puliti"~~
3.  In produzione non funziona (picchi traffico)

### Perché è importante?

In produzione:
- Traffico ha picchi naturali (non outlier ma traffico legittimo)
- Se scaler fittato su dati "puliti", non gestisce picchi
- Risultato: false positive e degrado performance

**RobustScaler** è robusto agli outlier quindi gestisce bene entrambi:
- Dati normali (usa mediana e IQR)
- Picchi di traffico (non vengono distorti)

##  Log Files

I log contengono informazioni dettagliate su ogni step:

```bash
# Vedi progresso preprocessing
tail -f logs/preprocessing.log

# Vedi feature selection
tail -f logs/feature_engineering.log

# Vedi training metrics
tail -f logs/training.log

# Pipeline completa
tail -f logs/pipeline.log
```

##  Comandi Utili

```bash
# Verifica spazio disco
df -h data/

# Monitora RAM durante esecuzione
watch -n 1 free -h

# Monitora CPU usage (mostra worker in azione)
htop  # o top

# Conta numero di core disponibili
python -c "import psutil; print(f'Cores: {psutil.cpu_count()}')"

# Verifica worker count configurato
python -c "from utils import get_worker_count; from config import CPU_USAGE_PERCENT, MIN_WORKERS, MAX_WORKERS; print(f'Workers: {get_worker_count(CPU_USAGE_PERCENT, MIN_WORKERS, MAX_WORKERS)}')"

# Conta righe nei Parquet
python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('data/processed/train.parquet').metadata.num_rows)"

# Vedi feature selezionate
cat artifacts/features.txt
```

##  Riferimenti

- **Dataset**: NF-UQ-NIDS-v2 - https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- **Paper**: NetFlow-based intrusion detection systems
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/

## 🤝 Support

In caso di problemi:
1. Controlla i log in `logs/`
2. Verifica RAM disponibile con `free -h`
3. Riduci `CHUNK_SIZE` se necessario
4. Usa LightGBM se XGBoost usa troppa RAM

---

**Buon training! **
