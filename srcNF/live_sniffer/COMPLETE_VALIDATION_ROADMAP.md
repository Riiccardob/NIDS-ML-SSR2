# VALIDATION ROADMAP - GUIDA COMPLETA E DEFINITIVA

## OVERVIEW

Questa guida copre **TUTTA** la validazione del sistema NIDS dopo il retraining con 24 feature.

**Prerequisiti**:
-  Training completato con successo (24 feature)
-  Artifacts generati in `/artifacts`
-  Modello salvato in `/models/xgboost`

---

## FASE 1: VALIDAZIONE COMPONENTI (30 min)

### Step 1.1: Aggiorna test_sniffer.py

```bash
cd srcNF/live_sniffer

# Backup vecchio test
cp test_sniffer.py test_sniffer.py.backup

# Applica versione corretta
cp test_sniffer_FIXED.py test_sniffer.py
```

### Step 1.2: Test Componenti

```bash
# IMPORTANTE: Esegui con sudo (serve per root check)
sudo python3 test_sniffer.py

# Output atteso:
# Test 1: External Libraries ... PASS
# Test 2: Sniffer Modules ... PASS
# Test 3: Artifacts ... PASS (24 features)
# Test 4: Configuration ... PASS
# Test 5: Feature Mapper ... PASS (24 features)
# Test 6: Preprocessor ... PASS (24 features)
# Test 7: Predictor ... PASS (24 features)
# Test 8: Logger ... PASS
# Test 9: Network Interfaces ... PASS
# Test 10: Root Privileges ... PASS
#
# TOTAL: 10/10 tests passed
# STATUS: ALL TESTS PASSED 
```

**CHECKPOINT CRITICO**:
- Se qualche test FAIL → **STOP e risolvi** prima di procedere
- Se 10/10 PASS → Procedi a FASE 2

---

## FASE 2: VALIDAZIONE FUNZIONALE RAPIDA (15 min)

### Step 2.1: Test Loopback (Traffico Locale)

```bash
# Terminal 1: Avvia sniffer su loopback
sudo python3 main.py --mode alert --interface lo

# Output atteso:
# ======================================================================
# NIDS LIVE SNIFFER STARTED
# ======================================================================
# Mode: ALERT
# Interface: lo
# Model: xgboost (24 features)
# Threshold: 0.5
# ======================================================================
# Waiting for traffic...
```

**Verifica output**:
-  "24 features" (NON 35!)
-  Nessun errore "Invalid feature shape"

```bash
# Terminal 2: Genera traffico
ping -c 100 127.0.0.1
curl http://localhost:80  # (se hai server locale)
```

```bash
# Terminal 3: Monitor logs
tail -f ../../logs/sniffer/nids_sniffer_alerts.csv

# Output atteso: Flow rilevati senza errori
```

**Verifica**:
-  File CSV creato
-  Flow processati senza crash
-  Nessun errore "feature shape"

```bash
# Terminal 1: Stop sniffer (Ctrl+C)
```

### Step 2.2: Verifica Logs

```bash
# Check alerts
wc -l ../../logs/sniffer/nids_sniffer_alerts.csv

# Deve avere >= 2 righe (header + flow)

# Check formato
head -5 ../../logs/sniffer/nids_sniffer_alerts.csv

# Output atteso (esempio):
# timestamp,src_ip,src_port,dst_ip,dst_port,protocol,prediction,confidence,action
# 2026-02-09 01:23:45,127.0.0.1,12345,127.0.0.1,80,6,0,0.3245,logged
```

**CHECKPOINT**:
-  Sniffer cattura traffico
-  Processa flow (24 feature)
-  Genera log corretti

---

## FASE 3: TEST DOCKER - ATTACK SIMULATION (2 ore)

### Step 3.1: Setup Ambiente Docker

```bash
# Crea rete isolata
docker network create --subnet=172.20.0.0/16 nids_test_net

# Avvia target (Nginx web server)
docker run -d \
    --name nids_target \
    --net nids_test_net \
    --ip 172.20.0.2 \
    nginx:alpine

# Verifica target attivo
docker ps | grep nids_target

# Trova interfaccia Docker bridge
INTERFACE=$(ifconfig | grep -B 1 "inet 172.20.0.1" | head -1 | awk '{print $1}' | tr -d ':')
echo "Docker interface: $INTERFACE"

# Output esempio: br-a1b2c3d4e5f6
```

### Step 3.2: Test 1 - Port Scan (nmap)

```bash
# Terminal 1: Avvia sniffer
cd srcNF/live_sniffer
sudo python3 main.py --mode alert --interface $INTERFACE

# Terminal 2: Monitor alerts
tail -f ../../logs/sniffer/nids_sniffer_alerts.csv

# Terminal 3: Execute port scan
sudo nmap -sS -p 1-100 172.20.0.2

# Lascia scannare tutte le 100 porte (~30 secondi)
```

**Verifica Detection**:
```bash
# Count attack flows rilevati
grep -c ",1," ../../logs/sniffer/nids_sniffer_alerts.csv

# Detection rate atteso: >30% (almeno 30/100)
# Con 24 feature: 50-70% detection è BUONO
```

**CHECKPOINT**:
-  Detection rate >30%
-  Confidence valori tra 0.5-1.0 per attack
-  Nessun crash durante scan

### Step 3.3: Test 2 - DoS SYN Flood (hping3)

```bash
# Terminal 1: Sniffer già attivo

# Terminal 3: SYN flood controllato
sudo hping3 -S -p 80 --flood -i u10000 172.20.0.2

# Lascia girare 10 secondi, poi Ctrl+C
```

**Verifica Detection**:
```bash
# Ultimi 200 flow
tail -200 ../../logs/sniffer/nids_sniffer_alerts.csv | grep ",1," | wc -l

# Detection attesa: >60% dei flow SYN flood
```

### Step 3.4: Test 3 - HTTP Slowloris

```bash
# Installa slowloris (se non già fatto)
git clone https://github.com/gkbrk/slowloris.git /tmp/slowloris
cd /tmp/slowloris

# Terminal 3: Esegui attack
python3 slowloris.py 172.20.0.2 -p 80 -s 50

# Lascia girare 60 secondi
```

**Verifica**:
```bash
cd srcNF/live_sniffer

# Check flow HTTP lunghi
grep ",80," ../../logs/sniffer/nids_sniffer_alerts.csv | tail -50

# Cerca flow con duration alta (slowloris pattern)
```

### Step 3.5: Cleanup Docker

```bash
# Terminal 1: Stop sniffer (Ctrl+C)

# Cleanup
docker stop nids_target
docker rm nids_target
docker network rm nids_test_net
```

**RISULTATI FASE 3**:

| Test | Detection Rate Atteso | Pass Condition |
|------|----------------------|----------------|
| Port Scan | 50-70% | >30% |
| DoS SYN Flood | 60-80% | >50% |
| HTTP Slowloris | 30-50% | >20% |

Se **tutti i test >pass condition** → Sistema VALIDATO 

---

## FASE 4: BLOCK MODE (OPZIONALE - 1 ora)

**ATTENZIONE**: Test PERICOLOSO. Rischio auto-blocco IP.

### Step 4.1: Setup Whitelist (CRITICO!)

```bash
cd srcNF/live_sniffer
nano config.py

# Trova WHITELIST_IPS e aggiungi:
WHITELIST_IPS = [
    "127.0.0.1",              # Localhost
    "::1",                    # IPv6 localhost
    "172.20.0.1",             # Docker gateway (TUO IP!)
    "192.168.1.1",            # Gateway LAN (TROVA IL TUO!)
    "<TUO_IP_MACCHINA>",      # IP amministratore
]

# Trova TUO IP:
# ip addr show | grep "inet " | grep -v 127.0.0.1
```

**SALVA E ESCI** (Ctrl+X, Y, Enter)

### Step 4.2: Test BLOCK Mode (Docker Isolato)

```bash
# Ricrea ambiente Docker
docker network create --subnet=172.20.0.0/16 nids_test_net
docker run -d --name nids_target --net nids_test_net --ip 172.20.0.2 nginx:alpine

INTERFACE=$(ifconfig | grep -B 1 "inet 172.20.0.1" | head -1 | awk '{print $1}' | tr -d ':')

# Terminal 1: BLOCK mode
sudo python3 main.py --mode block --interface $INTERFACE

# Output deve mostrare:
# Mode: BLOCK
# Firewall: iptables chain NIDS_BLOCK created
# Whitelist: 172.20.0.1 protected

# Terminal 2: Monitor iptables
watch -n 2 'sudo iptables -L NIDS_BLOCK -n -v'

# Terminal 3: Generate attack da container NON whitelisted
docker run -it --rm --net nids_test_net --ip 172.20.0.50 alpine sh

# Dentro container:
apk add nmap
nmap -sS -p 1-50 172.20.0.2
```

**Verifica Blocco**:
```bash
# Terminal 2: Controlla iptables
sudo iptables -L NIDS_BLOCK -n -v

# Deve mostrare regola DROP per 172.20.0.50
# pkts bytes target     prot opt in     out     source        destination
#    0     0 DROP       all  --  *      *       172.20.0.50   0.0.0.0/0
```

**Verifica Whitelist**:
```bash
# Terminal 3: Test da host (whitelisted)
nmap -sS -p 80 172.20.0.2

# Terminal 2: Check iptables
sudo iptables -L NIDS_BLOCK -n

# 172.20.0.1 NON deve apparire (protetto da whitelist)
```

**Cleanup**:
```bash
# Terminal 1: Stop sniffer (Ctrl+C)
# Deve rimuovere chain automaticamente

# Verifica cleanup
sudo iptables -L NIDS_BLOCK
# Errore: No chain/target/match by that name. (BUONO!)

# Cleanup Docker
docker stop nids_target && docker rm nids_target
docker network rm nids_test_net
```

---

## FASE 5: METRICHE FINALI (30 min)

### Step 5.1: Analisi Performance

```bash
cd srcNF/live_sniffer

# Estrai statistiche da logs
python3 << 'EOF'
import pandas as pd
from pathlib import Path

log_path = Path("../../logs/sniffer/nids_sniffer_alerts.csv")
df = pd.read_csv(log_path)

print("="*70)
print("PERFORMANCE METRICS")
print("="*70)

print(f"\nTotal flows: {len(df):,}")

if 'prediction' in df.columns:
    benign = (df['prediction'] == 0).sum()
    attack = (df['prediction'] == 1).sum()
    
    print(f"Benign: {benign:,} ({benign/len(df)*100:.1f}%)")
    print(f"Attack: {attack:,} ({attack/len(df)*100:.1f}%)")
    
    if 'confidence' in df.columns:
        avg_conf = df['confidence'].mean()
        print(f"\nAverage confidence: {avg_conf:.3f}")
        
        attack_df = df[df['prediction'] == 1]
        if len(attack_df) > 0:
            avg_attack_conf = attack_df['confidence'].mean()
            print(f"Attack confidence (avg): {avg_attack_conf:.3f}")
EOF
```

### Step 5.2: Summary Report

```bash
# Crea report finale
cat > ../../validation_report.md << 'EOF'
# NIDS VALIDATION REPORT

## System Configuration
- Features: 24 (nfstream-compatible)
- Model: XGBoost
- Scaler: RobustScaler (aggregate)

## Validation Results

### Component Tests
- [x] All 10/10 tests PASSED

### Functional Tests
- [x] Loopback traffic: OK
- [x] Log generation: OK

### Attack Detection (Docker)
- Port Scan: XX% detection rate (>30% required)
- DoS Flood: XX% detection rate (>50% required)
- Slowloris: XX% detection rate (>20% required)

### Block Mode (Optional)
- [x] Firewall integration: OK
- [x] Whitelist protection: OK
- [x] Auto-cleanup: OK

## Conclusion
System validated and ready for deployment.

Date: $(date)
EOF

cat ../../validation_report.md
```

---

## TROUBLESHOOTING

### Problema: "Invalid feature shape (35, expected 24)"

**Causa**: Artifacts vecchi (35 feature)

**Fix**:
```bash
# Verifica artifacts
cat ../../artifacts/features.json | grep n_features

# Se mostra 35:
cd ..
rm -rf artifacts/* models/*
python pipeline.py --model xgboost
```

### Problema: Detection Rate Bassa (<20%)

**Possibili cause**:
1. Feature mismatch (verifica 24 feature in artifacts)
2. Threshold troppo alto (prova ATTACK_THRESHOLD = 0.4)
3. Modello non converso (verifica training log)

**Diagnosi**:
```bash
# Check confidence distribution
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("../../logs/sniffer/nids_sniffer_alerts.csv")
print(df['confidence'].describe())
EOF

# Se avg confidence <0.3 → modello non sicuro
# Se avg confidence >0.8 → ottimo
```

### Problema: Sniffer Blocca su Ctrl+C

**Fix**: Genera traffico dummy
```bash
# Altro terminal:
ping -c 1 <target_ip>

# Loop si sblocca
```

---

## FILE STRUTTURA FINALE

```
srcNF/
 config.py                      # Config principale (24 feature drop)
 preprocessing.py
 feature_engineering.py
 training.py
 pipeline.py

 live_sniffer/
    config.py                  # Config sniffer (21 feature list)
    test_sniffer.py            # Test suite (24 feature)
    main.py
   
    core/
       feature_mapper.py      # MINIMAL (24 feature)
       preprocessor.py
       predictor.py
       capture.py
   
    security/
       alert_manager.py
       firewall_controller.py
   
    utils/
       logger.py
   
    testing/                   # OPZIONALE (per alignment test)
        test_data/
           train_sample.csv
        verify_alignment.py

 artifacts/
    scaler.pkl                 # 24 feature scaler
    features.json              # 24 feature list

 models/
     xgboost/
         model.pkl              # 24 feature model
```

**File da RIMUOVERE** (obsoleti):
```bash
# Cleanup opzionale
cd srcNF/live_sniffer

# Rimuovi validation/ (duplicato di testing/)
rm -rf validation/

# Rimuovi PCAP grandi se non servono
rm testing/test_data/Friday-WorkingHours.pcap
rm testing/test_data/Tuesday-WorkingHours.pcap

# Mantieni solo test_traffic.pcap (piccolo, generato)
```

---

## CHECKLIST FINALE

Prima di deployment produzione:

- [ ] Test componenti: 10/10 PASS
- [ ] Test loopback: OK
- [ ] Test Docker port scan: >30% detection
- [ ] Test Docker DoS: >50% detection
- [ ] Log format: CSV + JSON corretti
- [ ] Block mode: Whitelist + cleanup OK
- [ ] Feature count: 24 in tutti i componenti
- [ ] Documentation: README aggiornato

Se **TUTTI** check  → **READY FOR PRODUCTION** 

---

## COMANDI RAPIDI REFERENCE

```bash
# Test componenti
sudo python3 test_sniffer.py

# Avvia ALERT mode
sudo python3 main.py --mode alert --interface eth0

# Avvia BLOCK mode
sudo python3 main.py --mode block --interface eth0

# Monitor logs live
tail -f ../../logs/sniffer/nids_sniffer_alerts.csv

# Check iptables (BLOCK mode)
sudo iptables -L NIDS_BLOCK -n -v

# Cleanup manuale iptables
sudo iptables -D INPUT -j NIDS_BLOCK
sudo iptables -F NIDS_BLOCK
sudo iptables -X NIDS_BLOCK
```

---

## TEMPO TOTALE VALIDAZIONE

- Fase 1 (Componenti): 30 min
- Fase 2 (Funzionale): 15 min
- Fase 3 (Docker Attack): 2 ore
- Fase 4 (Block Mode): 1 ora (opzionale)
- Fase 5 (Metriche): 30 min

**Totale minimo**: 3 ore
**Totale completo**: 4.5 ore

**BUONA VALIDAZIONE!** 
