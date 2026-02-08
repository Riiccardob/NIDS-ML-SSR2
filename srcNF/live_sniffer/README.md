# NIDS SNIFFER - FINAL CLEAN VERSION

## Struttura Corretta

```
NIDS-ML-SSR2/
 srcNF/
    pipeline.py
    preprocessing.py
    feature_engineering.py
    training.py
    utils.py
    config.py
   
    live_sniffer/                 # Sniffer + Testing
        README.md                 # Documentazione principale
        requirements.txt
        config.py
        main.py
        test_sniffer.py
       
        core/                     # Engine sniffer
           capture.py
           feature_mapper.py
           preprocessor.py
           predictor.py
       
        security/                 # Alert e firewall
           alert_manager.py
           firewall_controller.py
       
        utils/                    # Utilities
           logger.py
       
        testing/                  # Tool di testing
            DOCKER_TESTING_GUIDE.md  # Guida test Docker
            verify_alignment.py      # Script verifica feature (NON USARE CON CICIDS!)
            test_data/
                train_sample.csv     # Sample training per confronto

 artifacts/                        # Model artifacts (creati da training)
    scaler.pkl
    features.json

 models/                           # Trained models
    xgboost/
        model.pkl

 logs/                             # Log sniffer
     sniffer/
         nids_sniffer.log
         nids_sniffer_alerts.csv
         nids_sniffer_alerts.json
```

---

## FILE DA RIMUOVERE (Cleanup)

### In srcNF/live_sniffer/

```bash
cd srcNF/live_sniffer

# Rimuovi documentazione obsoleta
rm -f ARCHITECTURE.md FIX_GUIDE.md PRODUCTION.md QUICKSTART.md setup.py

# Rimuovi directory vuote/inutili
rm -rf artifacts data logs models
```

### In validation/ (SE ESISTE ANCORA)

```bash
# Se hai ancora validation/ nella root, rimuovila
cd NIDS-ML-SSR2
rm -rf validation/

# TUTTO è ora in srcNF/live_sniffer/testing/
```

---

## QUICK START PULITO

### 1. Test Componenti

```bash
cd srcNF/live_sniffer
sudo python3 test_sniffer.py

# Deve passare 10/10 test
```

### 2. Test Live con Docker

```bash
# Leggi guida completa
less testing/DOCKER_TESTING_GUIDE.md

# Quick test:
# 1. Crea rete e vittima
docker network create --subnet=172.20.0.0/16 nids_test_net
docker run -d --name vittima --net nids_test_net --ip 172.20.0.2 nginx

# 2. Trova interfaccia Docker
INTERFACE=$(ifconfig | grep -B 1 "inet 172.20.0.1" | head -1 | awk '{print $1}' | tr -d ':')

# 3. Avvia sniffer
sudo python3 main.py --mode alert --interface $INTERFACE

# 4. Genera attacco (altro terminal)
sudo nmap -sS -p 1-100 172.20.0.2

# 5. Verifica logs
tail -f ../../logs/sniffer/nids_sniffer_alerts.csv
```

---

## IMPORTANTE: Feature Alignment

**NON usare CICIDS** per verify_alignment.py!

**Motivo**: Training set è SCALED (valori normalizzati), CICIDS è RAW.

**Alternative valide**:
1.  Traffico live generato da te (Docker test)
2.  PCAP dal tuo stesso dataset NF-UQ-NIDS-v2 (se disponibile)
3.  CICIDS, UNSW-NB15, altri dataset pubblici (incompatibili)

**verify_alignment.py** è uno strumento **avanzato** che serve SOLO se:
- Hai PCAP del tuo dataset originale
- Vuoi verificare estrazione matematica delle feature

Per validazione pratica, usa **test Docker** (DOCKER_TESTING_GUIDE.md).

---

## FILE ESSENZIALI

### Core Sniffer
- `main.py` - Entry point
- `config.py` - Configurazione
- `core/*` - Engine (capture, feature extraction, prediction)
- `security/*` - Alert e firewall
- `utils/logger.py` - Logging strutturato

### Testing
- `test_sniffer.py` - Unit test componenti
- `testing/DOCKER_TESTING_GUIDE.md` - Guida test pratici
- `testing/verify_alignment.py` - Tool avanzato (opzionale)

### Documentazione
- `README.md` - Questo file
- `testing/DOCKER_TESTING_GUIDE.md` - Tutorial test completo

---

## RISOLUZIONE PROBLEMI COMUNI

### 1. Import Error in verify_alignment.py

**Fix applicato**:
```python
# In testing/verify_alignment.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # live_sniffer/testing -> live_sniffer
sys.path.insert(0, str(PROJECT_ROOT))
```

### 2. CICIDS produce risultati disastrosi

**Non è un bug**: CICIDS è RAW, training è SCALED.

**Soluzione**: Usa test Docker invece di verify_alignment.

### 3. Troppi file in giro

**Esegui cleanup**:
```bash
# Dalla root progetto
bash cleanup.sh
```

---

## WORKFLOW RACCOMANDATO

```bash
# 1. Verifica componenti
cd srcNF/live_sniffer
sudo python3 test_sniffer.py

# 2. Test live con Docker
# Segui testing/DOCKER_TESTING_GUIDE.md

# 3. Se detection OK → Deploy produzione
# 4. Se detection scarsa → Retraining necessario
```

---

## SUPPORTO

- **Test pratici**: `testing/DOCKER_TESTING_GUIDE.md`
- **Configurazione**: `config.py` (inline comments)
- **Log troubleshooting**: `logs/sniffer/nids_sniffer.log`

---

## VERSIONE

Clean Version: 2.0
Data: 2024-02-08
Compatibile: nfstream 6.5.4, Python 3.10+
