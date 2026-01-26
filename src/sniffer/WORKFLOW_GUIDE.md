# NIDS-ML Complete Workflow Guide

## 📋 Indice

1. [Panoramica Sistema](#1-panoramica-sistema)
2. [Struttura Progetto](#2-struttura-progetto)
3. [Installazione](#3-installazione)
4. [Workflow Completo (6 Fasi)](#4-workflow-completo)
5. [Comandi CLI](#5-comandi-cli)
6. [Dataset Esterni per Testing Avanzato](#6-dataset-esterni)
7. [Troubleshooting](#7-troubleshooting)
8. [Checklist Pre-Production](#8-checklist)

---

## 1. Panoramica Sistema

### 1.1 Architettura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NIDS-ML Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Packets    │───▶│    Flows     │───▶│   Features   │───▶│   Model   │ │
│  │  (Scapy)     │    │ (FlowManager)│    │ (Extractor)  │    │ (ML)      │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                    │                   │                   │       │
│        ▼                    ▼                   ▼                   ▼       │
│  Live/PCAP            Aggregazione        77 Features          Prediction   │
│                       Bidirezionale       CIC-IDS2017          BENIGN/ATTACK│
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Validation & Calibration                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Validator   │    │  Calibrator  │    │  Evaluator   │                   │
│  │ (Coverage)   │    │ (Statistics) │    │ (Metrics)    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Componenti

| Componente | File | Funzione |
|------------|------|----------|
| **Config** | `config.py` | Configurazione centralizzata, path, modelli |
| **Flow** | `flow.py` | Aggregazione pacchetti → flussi bidirezionali |
| **Features** | `features.py` | Estrazione 77 feature CIC-IDS2017 |
| **Engine** | `engine.py` | Motore principale (live + PCAP) |
| **Validator** | `validator.py` | Validazione feature extraction |
| **Calibrator** | `calibration.py` | Diagnostica distribuzione feature |
| **Evaluator** | `evaluation.py` | Metriche performance (F1, FPR, latency) |
| **Main** | `main.py` | CLI entry point |

---

## 2. Struttura Progetto

```
nids-ml/
├── src/
│   └── sniffer/               # 📦 Package principale
│       ├── __init__.py
│       ├── config.py          # Configurazione centralizzata
│       ├── main.py            # CLI entry point
│       ├── flow.py            # Flow aggregation
│       ├── features.py        # Feature extraction
│       ├── engine.py          # Sniffer engine
│       ├── validator.py       # Feature validation
│       ├── evaluation.py      # Model evaluation
│       └── calibration.py     # Feature calibration
│
├── models/                    # 🤖 Modelli ML
│   ├── best_model/            # Modello migliore (auto-selected)
│   │   ├── model_binary.pkl
│   │   ├── results_binary.json
│   │   └── features_binary.json
│   ├── xgboost/               # Esperimenti XGBoost
│   │   ├── bayesian_trials50_cv5_20240115/
│   │   └── bayesian_trials100_cv5_20240120/
│   ├── lightgbm/              # Esperimenti LightGBM
│   └── random_forest/         # Esperimenti Random Forest
│
├── artifacts/                 # 🔧 Preprocessing artifacts
│   ├── scaler.pkl
│   ├── feature_selector.pkl
│   ├── selected_features.json
│   └── scaler_columns.json
│
├── data/
│   ├── raw/                   # 📊 CIC-IDS2017 CSV (training data)
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   ├── Wednesday-workingHours.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   │   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── pcap/                  # 📦 CIC-IDS2017 PCAP files
│   │   ├── Monday-WorkingHours.pcap
│   │   ├── Tuesday-WorkingHours.pcap
│   │   ├── Wednesday-workingHours.pcap
│   │   ├── Thursday-WorkingHours.pcap
│   │   └── Friday-WorkingHours.pcap
│   ├── processed/             # Dati preprocessati
│   └── external/              # Dataset esterni per testing
│
├── logs/                      # 📝 Runtime logs
└── reports/                   # 📈 Evaluation reports
```

---

## 3. Installazione

### 3.1 Dipendenze

```bash
# Core
pip install numpy pandas scikit-learn joblib tqdm scipy

# Packet capture
pip install scapy

# ML models (se non già installati)
pip install xgboost lightgbm

# Configuration
pip install pyyaml

# Optional: visualizzazioni
pip install matplotlib seaborn
```

### 3.2 Verifica Installazione

```bash
# Verifica che tutto funzioni
python -m src.sniffer.main config --show --list-models --list-data
```

---

## 4. Workflow Completo (6 Fasi)

### 📊 OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WORKFLOW DI VALIDAZIONE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1          FASE 2          FASE 3          FASE 4          FASE 5     │
│  ───────         ───────         ───────         ───────         ───────    │
│  Validate   ──▶  Calibrate  ──▶  Evaluate   ──▶  PCAP Test  ──▶  External   │
│  CSV             Features        Metrics         CIC-IDS          Dataset    │
│  Coverage        Statistics      F1/FPR          PCAP             Test       │
│                                                                    │         │
│                                                                    ▼         │
│                                                              FASE 6          │
│                                                              ───────         │
│                                                              Live            │
│                                                              Capture         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### FASE 1: Validazione CSV Coverage

**Obiettivo:** Verificare che i CSV abbiano tutte le feature richieste dal modello.

**Perché è importante:** Se mancano feature, il modello non può fare predizioni accurate.

```bash
# Valida un singolo giorno
python -m src.sniffer.main validate --day tuesday -o reports/validate_tuesday.json

# Valida tutti i giorni manualmente
for day in monday tuesday wednesday thursday_morning thursday_afternoon friday_morning friday_afternoon_portscan friday_afternoon_ddos; do
    echo "=== $day ==="
    python -m src.sniffer.main validate --day $day --sample 1000
done
```

**Criteri di successo:**
- ✅ Coverage ≥ 95% delle feature
- ✅ Nessuna feature CRITICAL mancante
- ✅ Score ≥ 80/100

**Output atteso:**
```
FEATURE VALIDATION REPORT
==========================
Coverage: 76/77 features (98.7%)
Missing: Init_Win_bytes_backward
Critical issues: 0
Score: 95/100 ✓
```

---

### FASE 2: Calibrazione Feature

**Obiettivo:** Analizzare la distribuzione delle feature per identificare anomalie.

**Perché è importante:** Feature con distribuzione anomala (tutti zero, varianza nulla) possono indicare problemi nell'estrazione o dati corrotti.

```bash
# Calibrazione base
python -m src.sniffer.main calibrate --day tuesday --sample 10000

# Analisi feature importance
python -m src.sniffer.main calibrate --day tuesday --importance --sample 20000

# Calibrazione completa con report
python -m src.sniffer.main calibrate --day wednesday --sample 20000 -o reports/calibration_wednesday.json
```

**Criteri di successo:**
- ✅ Score calibrazione ≥ 80/100
- ✅ Meno di 10 feature con zero variance
- ✅ Nessun critical issue

**Output atteso:**
```
FEATURE CALIBRATION REPORT
===========================
Samples: 20000
Overall Score: 87.5/100

📊 FEATURE ANALYSIS:
  Total features: 77
  Missing in CSV: 1
  Zero variance: 5
  High discrepancy: 3

💡 RECOMMENDATIONS:
  • Features with zero variance may not be useful: Active Mean, Active Std, ...
```

---

### FASE 3: Valutazione Metriche su CSV

**Obiettivo:** Misurare F1-Score, FPR, Precision, Recall sui dati di training.

**Perché è importante:** Conferma che il modello funziona correttamente sui dati originali prima di testarlo su dati nuovi.

```bash
# Valuta un singolo giorno con un modello specifico
python -m src.sniffer.main evaluate --day tuesday \
    --model-type xgboost --model-version best \
    --sample 50000 -o reports/eval_tuesday.json

# Valuta TUTTI i giorni
python -m src.sniffer.main evaluate --all-days \
    --model-type xgboost \
    --sample 10000 -o reports/eval_all_days.json

# Confronta modelli diversi
for model in xgboost lightgbm random_forest; do
    echo "=== $model ==="
    python -m src.sniffer.main evaluate --all-days --model-type $model --sample 10000
done
```

**Criteri di successo:**
| Metrica | Minimo | Target |
|---------|--------|--------|
| F1 Score | ≥ 0.90 | ≥ 0.95 |
| False Positive Rate | ≤ 0.05 | ≤ 0.02 |
| Recall | ≥ 0.85 | ≥ 0.95 |
| Precision | ≥ 0.85 | ≥ 0.90 |

**Note sui giorni CIC-IDS2017:**

| Giorno | Attacchi | Note |
|--------|----------|------|
| Monday | Nessuno (solo BENIGN) | F1=0 è corretto! Usa FPR come metrica |
| Tuesday | FTP-Patator, SSH-Patator | Brute force facili da rilevare |
| Wednesday | DoS, Heartbleed | DoS molto detectabili |
| Thursday AM | Web Attacks | Più difficili |
| Thursday PM | Infiltration | Rari, difficili |
| Friday AM | Botnet | Moderato |
| Friday PM | PortScan, DDoS | Facili |

---

### FASE 4: Test su PCAP CIC-IDS2017

**Obiettivo:** Testare la pipeline completa (packet → flow → features → prediction) sui PCAP originali.

**Perché è importante:** Verifica che l'estrazione feature dal PCAP produca risultati coerenti con i CSV (che sono stati generati dagli stessi PCAP da CICFlowMeter).

```bash
# Analizza PCAP di Tuesday (brute force attacks)
python -m src.sniffer.main pcap --day tuesday \
    --model-type xgboost \
    --max-packets 100000 \
    -v -o reports/pcap_tuesday.json

# Analizza Wednesday (DoS)
python -m src.sniffer.main pcap --day wednesday \
    --model-type xgboost \
    -v -o reports/pcap_wednesday.json

# Benchmark latenza
python -m src.sniffer.main benchmark --samples 5000 --iterations 20
```

**Cosa aspettarsi:**
- Il numero di attacchi rilevati dal PCAP dovrebbe essere simile a quello nel CSV
- La distribuzione delle label dovrebbe essere coerente
- Non aspettarti match perfetto: l'estrazione feature Python ≠ CICFlowMeter Java

**Criteri di successo:**
- ✅ Nessun crash durante l'analisi
- ✅ Latency < 500ms per flow
- ✅ Attacchi rilevati (non zero per giorni con attacchi)

---

### FASE 5: Test su Dataset Esterno

**Obiettivo:** Validare la generalizzazione del modello su dati MAI visti durante il training.

**Perché è fondamentale:** Un modello che funziona solo su CIC-IDS2017 è inutile nel mondo reale. Devi testare su dataset diversi.

#### 5.1 Dataset Consigliati

| Dataset | Anno | Dimensione | Link | Perché usarlo |
|---------|------|------------|------|---------------|
| **CSE-CIC-IDS2018** | 2018 | ~16GB CSV | [AWS](https://www.unb.ca/cic/datasets/ids-2018.html) | Evoluzione diretta di CIC-IDS2017, stesso formato |
| **CICIDS2017 (unseen days)** | 2017 | - | Già hai | Usa split temporale: train su Lun-Gio, test su Ven |
| **UNSW-NB15** | 2015 | ~2GB | [Link](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Dataset diverso, verifica robustezza |
| **CTU-13** | 2011 | Varia | [Link](https://www.stratosphereips.org/datasets-ctu13) | Botnet reali, molto challenging |
| **MAWI** | 2020+ | Ongoing | [Link](http://mawi.wide.ad.jp/mawi/) | Traffico reale anonimizzato |

#### 5.2 Procedura per CSE-CIC-IDS2018 (Consigliato)

```bash
# 1. Download (da AWS S3)
mkdir -p data/external/cse-cic-ids2018
cd data/external/cse-cic-ids2018
# Scarica da: https://www.unb.ca/cic/datasets/ids-2018.html

# 2. Struttura attesa:
# data/external/cse-cic-ids2018/
# ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
# ├── Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
# └── ...

# 3. Valida il formato
python -m src.sniffer.main validate \
    --csv data/external/cse-cic-ids2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv

# 4. Valuta
python -m src.sniffer.main evaluate \
    --csv data/external/cse-cic-ids2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv \
    --model-type xgboost \
    --sample 50000 \
    -o reports/eval_cse2018_wednesday.json
```

#### 5.3 Split Temporale CIC-IDS2017

Un'alternativa senza scaricare nuovi dati:

```python
# In un notebook o script Python

from src.sniffer import Config, SnifferEvaluator

config = Config()
evaluator = SnifferEvaluator(
    model_dir='models/best_model',
    artifacts_dir='artifacts'
)

# Train: Monday-Thursday, Test: Friday
# Questo simula la situazione reale dove il modello 
# è stato allenato sul passato e deve predire il futuro

print("=== FRIDAY (UNSEEN) ===")
for day in ['friday_morning', 'friday_afternoon_portscan', 'friday_afternoon_ddos']:
    csv_path = config.paths.raw_csv_dir / config.dataset.CICIDS2017_FILES[day]['csv']
    metrics = evaluator.evaluate_csv(str(csv_path), sample_size=20000)
    print(f"{day}: F1={metrics.f1_score:.4f}, FPR={metrics.false_positive_rate:.4f}")
```

**Criteri di successo su dataset esterni:**
- ✅ F1 Score ≥ 0.80 (accettabile calo di ~10% rispetto a training data)
- ✅ FPR ≤ 0.10 (può essere più alto su dati nuovi)
- ✅ Il modello non crasha su formati leggermente diversi

---

### FASE 6: Test Live

**Obiettivo:** Verificare il funzionamento in condizioni reali di produzione.

**Prerequisiti:**
- ✅ Tutte le fasi precedenti completate con successo
- ✅ Permessi root o capabilities per packet capture
- ✅ Interfaccia di rete configurata

#### 6.1 Test Controllato (Consigliato Prima)

```bash
# Genera traffico di test con tcpreplay (se hai un PCAP)
sudo tcpreplay -i eth0 --pps=1000 data/pcap/Tuesday-WorkingHours.pcap &

# In un altro terminale, avvia lo sniffer
sudo python -m src.sniffer.main live \
    --interface eth0 \
    --duration 60 \
    --model-type xgboost \
    -v -o reports/live_test_controlled.json
```

#### 6.2 Test su Traffico Reale

```bash
# Test breve (5 minuti)
sudo python -m src.sniffer.main live \
    --interface eth0 \
    --duration 300 \
    --model-type xgboost \
    -v -o reports/live_test_5min.json

# Test prolungato (1 ora)
sudo python -m src.sniffer.main live \
    --interface eth0 \
    --duration 3600 \
    --model-type xgboost \
    --log-dir /var/log/nids \
    -o reports/live_test_1hour.json
```

#### 6.3 Produzione (con Firewall)

```bash
# ATTENZIONE: Questo blocca realmente gli IP!
# Usa prima --firewall senza --firewall-execute (dry-run)

# Dry run (log only)
sudo python -m src.sniffer.main live \
    --interface eth0 \
    --firewall \
    --confidence 0.9 \
    -v

# Produzione (blocking attivo)
sudo python -m src.sniffer.main live \
    --interface eth0 \
    --firewall \
    --firewall-execute \
    --confidence 0.95
```

**Criteri di successo:**
- ✅ Zero crash durante l'esecuzione
- ✅ Latency < 500ms per flow
- ✅ Memory usage stabile (no memory leak)
- ✅ FPR accettabile su traffico reale (< 5% falsi allarmi)

---

## 5. Comandi CLI

### 5.1 Riferimento Rapido

```bash
# Configurazione
python -m src.sniffer.main config --show              # Mostra config
python -m src.sniffer.main config --list-models       # Lista modelli
python -m src.sniffer.main config --list-data         # Lista dataset

# Validazione
python -m src.sniffer.main validate --day tuesday     # Per giorno
python -m src.sniffer.main validate --csv path/to.csv # Per file

# Calibrazione
python -m src.sniffer.main calibrate --day tuesday
python -m src.sniffer.main calibrate --csv path/to.csv --importance

# Valutazione
python -m src.sniffer.main evaluate --day tuesday --sample 10000
python -m src.sniffer.main evaluate --all-days --model-type xgboost
python -m src.sniffer.main evaluate --csv path/to.csv

# Benchmark
python -m src.sniffer.main benchmark --samples 1000

# PCAP
python -m src.sniffer.main pcap --day tuesday
python -m src.sniffer.main pcap --file path/to.pcap

# Live
sudo python -m src.sniffer.main live --interface eth0 --duration 300
```

### 5.2 Opzioni Globali

```bash
-v, --verbose           # Output dettagliato
-o, --output FILE       # Salva risultati in JSON
-c, --config-file FILE  # Usa file configurazione YAML/JSON
--model-type TYPE       # xgboost, lightgbm, random_forest
--model-version VER     # Versione specifica o "best"
--task-type TYPE        # binary o multiclass
--confidence FLOAT      # Soglia confidenza (0-1)
```

---

## 6. Dataset Esterni

### 6.1 Dove Trovare Dataset

| Source | URL | Note |
|--------|-----|------|
| Canadian Institute for Cybersecurity | https://www.unb.ca/cic/datasets/ | CIC-IDS2017, CSE-CIC-IDS2018 |
| UNSW Sydney | https://research.unsw.edu.au/projects/unsw-nb15-dataset | UNSW-NB15 |
| Stratosphere Lab | https://www.stratosphereips.org/datasets-overview | CTU-13, Malware |
| MAWI Working Group | http://mawi.wide.ad.jp/mawi/ | Traffico reale |
| Kaggle | https://www.kaggle.com/search?q=network+intrusion | Vari |

### 6.2 Compatibilità Dataset

Il sistema è ottimizzato per dataset con formato CICFlowMeter. Per altri formati:

```python
# Verifica compatibilità
from src.sniffer import quick_feature_check

result = quick_feature_check('path/to/new_dataset.csv')
print(f"Compatible: {result['compatible']}")
print(f"Missing features: {result['missing_features']}")
```

---

## 7. Troubleshooting

### 7.1 Errori Comuni

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| `Model not found` | Path sbagliato | Verifica `--model-type` e `--model-version` |
| `Permission denied` | No root per capture | Usa `sudo` |
| `No module named 'scapy'` | Dipendenza mancante | `pip install scapy` |
| `F1 = 0` su Monday | Nessun attacco nel dataset | È corretto! Usa FPR |
| `KeyError: 'Label'` | Nome colonna diverso | Verifica con `validate` |

### 7.2 Debug Mode

```bash
# Verbose output
python -m src.sniffer.main evaluate --day tuesday -v

# Log to file
python -m src.sniffer.main evaluate --day tuesday \
    --log-dir logs/ -v 2>&1 | tee debug.log
```

---

## 8. Checklist Pre-Production

### ✅ Artifacts
- [ ] `models/best_model/model_binary.pkl` esiste
- [ ] `artifacts/scaler.pkl` esiste
- [ ] `artifacts/selected_features.json` esiste

### ✅ Fase 1: Validate
- [ ] Coverage ≥ 95%
- [ ] Zero critical issues

### ✅ Fase 2: Calibrate
- [ ] Score ≥ 80/100
- [ ] Zero critical issues

### ✅ Fase 3: Evaluate (CSV)
- [ ] F1 ≥ 0.90 su training data
- [ ] FPR ≤ 0.05

### ✅ Fase 4: PCAP Test
- [ ] Nessun crash
- [ ] Attacchi rilevati coerenti

### ✅ Fase 5: External Dataset
- [ ] F1 ≥ 0.80 su dati nuovi
- [ ] FPR ≤ 0.10

### ✅ Fase 6: Live Test
- [ ] Test controllato OK
- [ ] Test 5 minuti OK
- [ ] Memory stabile
- [ ] Latency < 500ms

### ✅ Sistema
- [ ] Permessi configurati
- [ ] Log directory scrivibile
- [ ] Monitoring attivo

---

## 📞 Quick Reference Card

```bash
# === SETUP ===
pip install numpy pandas scikit-learn joblib scapy tqdm scipy pyyaml

# === VALIDATION WORKFLOW ===
python -m src.sniffer.main config --show --list-models
python -m src.sniffer.main validate --day tuesday
python -m src.sniffer.main calibrate --day tuesday --sample 10000
python -m src.sniffer.main evaluate --all-days --model-type xgboost --sample 10000
python -m src.sniffer.main benchmark --samples 1000
python -m src.sniffer.main pcap --day tuesday -v

# === PRODUCTION ===
sudo python -m src.sniffer.main live --interface eth0 --model-type xgboost
```

---

*NIDS-ML v2.0.0 - Network Intrusion Detection System*
