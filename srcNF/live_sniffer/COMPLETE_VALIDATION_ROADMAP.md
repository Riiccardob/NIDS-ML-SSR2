# ROADMAP COMPLETA - VALIDAZIONE SISTEMA NIDS-ML-SSR2

## PREREQUISITI

- Training completato (artifacts in `/artifacts`, model in `/models/xgboost`)
- Python 3.10+
- NFStream 5.1.0+ installato
- Root access (per packet capture)

---

## FASE 0: CORREZIONE FEATURE MAPPER (CRITICO!)

### Step 0.1: Backup File Originale

```bash
cd srcNF/live_sniffer/core
cp feature_mapper.py feature_mapper.py.BACKUP
```

### Step 0.2: Applica Feature Mapper Corretto

**OPZIONE A - Copia file corretto fornito**:
```bash
# Copia feature_mapper_FIXED.py → feature_mapper.py
cp /path/to/feature_mapper_FIXED.py feature_mapper.py
```

**OPZIONE B - Applica patch manualmente**:

Apri `feature_mapper.py` e applica queste modifiche:

**MODIFICA 1** (linea ~120):
```python
# PRIMA (SBAGLIATO):
if feature_name == "MIN_TTL":
    return float(self._get_value(flow, 'bidirectional_min_ps', 0))

# DOPO (CORRETTO):
if feature_name == "MIN_TTL":
    return 0.0  # Feature non disponibile in NFStream
```

**MODIFICA 2** (aggiungi dopo IN_PKTS, linea ~85):
```python
if feature_name == "OUT_PKTS":
    return float(self._get_value(flow, "dst2src_packets", 0))
```

**MODIFICA 3** (linea ~95):
```python
# AGGIUNGI:
if feature_name == "CLIENT_TCP_FLAGS":
    flags = self._get_value(flow, "client_tcp_flags", 0)
    return float(flags)
```

**MODIFICA 4** (linea ~140):
```python
# AGGIUNGI:
if feature_name == "RETRANSMITTED_OUT_PKTS":
    total_retrans = self._get_value(flow, "bidirectional_retrans_packets", 0)
    return float(total_retrans / 2.0)
```

**MODIFICA 5** (linea ~200):
```python
# AGGIUNGI:
if feature_name == "ICMP_IPV4_TYPE":
    protocol = self._get_value(flow, "protocol", 0)
    if protocol == 1:  # ICMP
        return float(self._get_value(flow, "icmp_type", 0))
    return 0.0
```

### Step 0.3: Verifica Modifiche

```bash
# Check syntax
python3 -m py_compile feature_mapper.py

# Se nessun errore:
echo "Feature mapper corretto!"
```

---

## FASE 1: TEST COMPONENTI (30 min)

### Step 1.1: Aggiorna Test Suite

```bash
cd srcNF/live_sniffer

# Backup vecchio test
cp test_sniffer.py test_sniffer.py.OLD

# Usa versione migliorata
cp /path/to/test_sniffer_IMPROVED.py test_sniffer.py
```

### Step 1.2: Esegui Test Componenti

```bash
# IMPORTANTE: Serve sudo per root check
sudo $(which python) test_sniffer.py
```

**Output Atteso:**
```
======================================================================
TEST SUMMARY
======================================================================
  [PASS] External Libraries
  [PASS] Sniffer Modules
  [PASS] Artifacts
  [PASS] Configuration
  [PASS] Feature Mapper
  [PASS] Preprocessor
  [PASS] Predictor
  [PASS] Logger
  [PASS] Network Interfaces
  [PASS] Root Privileges

======================================================================
TOTAL: 10/10 tests passed
STATUS: ALL TESTS PASSED
======================================================================
```

**CHECKPOINT CRITICO**:
- Se **ANY test FAIL** → STOP e risolvi prima di procedere
- Se **10/10 PASS** → Procedi a FASE 2

---

## FASE 2: TEST FUNZIONALE RAPIDO (15 min)

### Step 2.1: Test Loopback (Traffico Locale)

**Terminal 1 - Avvia Sniffer**:
```bash
cd srcNF/live_sniffer
sudo $(which python) main.py --mode alert --interface lo
```

**Output Atteso**:
```
======================================================================
NIDS LIVE SNIFFER STARTED
======================================================================
Mode: ALERT
Interface: lo
Model: xgboost (24 features)
Threshold: 0.5
======================================================================
Processing loop started

Waiting for traffic...
```

**Verifica Output**:
-  "24 features" (NON 35 o altro!)
-  Nessun errore "Invalid feature shape"
-  "Processing loop started"

**Terminal 2 - Genera Traffico**:
```bash
# Test 1: ICMP (ping)
ping -c 20 127.0.0.1

# Test 2: TCP (se hai server locale)
curl http://localhost:80

# Test 3: DNS (se hai resolver locale)
nslookup google.com 127.0.0.1
```

**Terminal 3 - Monitor Logs**:
```bash
tail -f logs/sniffer/nids_sniffer_alerts.csv
```

**Output Atteso (esempio)**:
```
timestamp,src_ip,src_port,dst_ip,dst_port,protocol,prediction,confidence,action
2026-02-09 12:34:56,127.0.0.1,54321,127.0.0.1,0,1,benign,0.2314,logged
2026-02-09 12:34:57,127.0.0.1,54322,127.0.0.1,0,1,benign,0.1823,logged
```

**Verifica**:
-  File CSV creato
-  Flow processati (almeno 10+)
-  Nessun crash
-  Confidence values tra 0-1

**Terminal 1 - Stop Sniffer**:
```
Ctrl+C
```

**Output Atteso**:
```
Stopping Live NIDS Sniffer...
Processing remaining batch...

======================================================================
FINAL STATISTICS
======================================================================
Total flows processed: 45
Total predictions: 45
Total alerts: 3
Total blocks: 0

======================================================================
SNIFFER STOPPED
======================================================================
```

### Step 2.2: Analisi Logs

```bash
cd srcNF

# Conta flow processati
wc -l logs/sniffer/nids_sniffer_alerts.csv

# Deve avere >= 2 righe (header + flow)

# Analizza distribuzione predizioni
awk -F',' 'NR>1 {print $8}' logs/sniffer/nids_sniffer_alerts.csv | sort | uniq -c

# Output esempio:
#   42 benign
#    3 attack

# Verifica formato
head -5 logs/sniffer/nids_sniffer_alerts.csv
```

**CHECKPOINT**:
-  Sniffer cattura traffico loopback
-  Processa flow (24 feature)
-  Genera log corretti (CSV + JSON)
-  Nessun crash o errore

---

## FASE 3: TEST ATTACK SIMULATION - DOCKER (2 ore)

### Step 3.1: Setup Ambiente Docker

```bash
# Crea rete isolata
docker network create --subnet=172.20.0.0/16 nids_test_net

# Verifica creazione
docker network ls | grep nids_test_net

# Avvia target (Nginx web server)
docker run -d \
    --name nids_target \
    --net nids_test_net \
    --ip 172.20.0.2 \
    --rm \
    nginx:alpine

# Verifica target attivo
docker ps | grep nids_target

# Output esempio:
# abc123def456   nginx:alpine   ... Up 2 seconds   nids_target

# Trova interfaccia Docker bridge
ip addr show | grep -A 2 "172.20.0.1"

# Output esempio:
# 5: br-abc123:  <BROADCAST,MULTICAST,UP,LOWER_UP> ...
#     inet 172.20.0.1/16 brd 172.20.255.255 scope global br-abc123

# Salva nome interfaccia
INTERFACE="br-abc123"  # Sostituisci con tuo valore
echo $INTERFACE
```

### Step 3.2: Test 1 - Port Scan (nmap)

**Terminal 1 - Avvia Sniffer**:
```bash
cd srcNF/live_sniffer
sudo $(which python) main.py --mode alert --interface $INTERFACE
```

**Terminal 2 - Monitor Alerts**:
```bash
cd srcNF
tail -f logs/sniffer/nids_sniffer_alerts.csv
```

**Terminal 3 - Execute Port Scan**:
```bash
# Installa nmap se non presente
sudo apt-get install nmap  # Ubuntu/Debian
# oppure
sudo yum install nmap      # CentOS/RHEL

# Esegui scan su 100 porte
sudo nmap -sS -p 1-100 172.20.0.2

# Lascia completare scan (~30 secondi)
```

**Output nmap atteso**:
```
Starting Nmap 7.80 ( https://nmap.org ) at 2026-02-09 12:00
Nmap scan report for 172.20.0.2
Host is up (0.00023s latency).
Not shown: 99 closed ports
PORT   STATE SERVICE
80/tcp open  http

Nmap done: 1 IP address (1 host up) scanned in 2.34 seconds
```

**Terminal 1 - Verifica Sniffer Output**:
Dovresti vedere flow processing in tempo reale:
```
[2026-02-09 12:00:15] | INFO | Processed: 50 flows...
[2026-02-09 12:00:16] | INFO | Processed: 100 flows...
```

**Terminal 1 - Stop Sniffer**:
```
Ctrl+C
```

**Analisi Detection Rate**:
```bash
cd srcNF

# Conta total flow
TOTAL=$(awk -F',' 'NR>1' logs/sniffer/nids_sniffer_alerts.csv | wc -l)

# Conta attack flow rilevati
ATTACKS=$(awk -F',' 'NR>1 && $8=="attack"' logs/sniffer/nids_sniffer_alerts.csv | wc -l)

# Calcola detection rate
python3 << EOF
total = $TOTAL
attacks = $ATTACKS
rate = (attacks / total * 100) if total > 0 else 0
print(f"Total flows: {total}")
print(f"Attacks detected: {attacks}")
print(f"Detection rate: {rate:.1f}%")
print()
if rate > 30:
    print("PASS: Detection rate > 30%")
else:
    print("FAIL: Detection rate too low")
EOF
```

**Detection Rate Atteso**: 30-70%

**CHECKPOINT**:
-  Sniffer cattura traffico Docker bridge
-  Processa flow scan nmap
-  Detection rate >30%
-  Confidence alta per attack (>0.6)

### Step 3.3: Test 2 - DoS SYN Flood (hping3)

**Installa hping3**:
```bash
sudo apt-get install hping3  # Ubuntu/Debian
# oppure
sudo yum install hping3      # CentOS/RHEL
```

**Terminal 1 - Avvia Sniffer**:
```bash
cd srcNF/live_sniffer

# Reset logs per test pulito
rm -f ../../logs/sniffer/nids_sniffer_alerts.csv

sudo $(which python) main.py --mode alert --interface $INTERFACE
```

**Terminal 2 - Monitor Logs**:
```bash
cd srcNF
watch -n 1 'tail -20 logs/sniffer/nids_sniffer_alerts.csv'
```

**Terminal 3 - SYN Flood Controllato**:
```bash
# SYN flood con rate controllato (100 pkt/sec)
sudo hping3 -S -p 80 --flood -i u10000 172.20.0.2

# Lascia girare 10 secondi
# Poi Ctrl+C
```

**Terminal 1 - Stop Sniffer**:
```
Ctrl+C
```

**Analisi Detection**:
```bash
cd srcNF

# Analizza ultimi 200 flow
tail -200 logs/sniffer/nids_sniffer_alerts.csv > /tmp/syn_test.csv

# Conta attack detection
awk -F',' 'NR>1 && $8=="attack"' /tmp/syn_test.csv | wc -l

# Calcola rate
python3 << EOF
import pandas as pd
df = pd.read_csv('/tmp/syn_test.csv')
total = len(df)
attacks = (df['prediction'] == 'attack').sum()
rate = (attacks / total * 100) if total > 0 else 0
print(f"Detection rate: {rate:.1f}%")
print("PASS" if rate > 50 else "FAIL")
EOF
```

**Detection Rate Atteso**: 50-80%

### Step 3.4: Test 3 - HTTP Slowloris

**Installa slowloris**:
```bash
cd /tmp
git clone https://github.com/gkbrk/slowloris.git
cd slowloris
```

**Terminal 1 - Avvia Sniffer**:
```bash
cd srcNF/live_sniffer

# Reset logs
rm -f ../../logs/sniffer/nids_sniffer_alerts.csv

sudo $(which python) main.py --mode alert --interface $INTERFACE
```

**Terminal 2 - Monitor**:
```bash
cd srcNF
tail -f logs/sniffer/nids_sniffer_alerts.csv
```

**Terminal 3 - Slowloris Attack**:
```bash
cd /tmp/slowloris

# Attack con 50 socket
python3 slowloris.py 172.20.0.2 -p 80 -s 50

# Lascia girare 60 secondi
# Poi Ctrl+C
```

**Terminal 1 - Stop Sniffer**:
```
Ctrl+C
```

**Analisi Flow Lunghi**:
```bash
cd srcNF

# Filtra flow HTTP con durata alta
awk -F',' 'NR>1 && $6==80 && $11>1000' logs/sniffer/nids_sniffer_alerts.csv | head -10

# Cerca pattern slowloris (durata alta, pochi byte)
python3 << EOF
import pandas as pd
df = pd.read_csv('logs/sniffer/nids_sniffer_alerts.csv')
slow = df[(df['dst_port'] == 80) & (df['duration_ms'] > 5000)]
print(f"Slow flows detected: {len(slow)}")
print(f"Attack detection: {(slow['prediction'] == 'attack').sum()}")
EOF
```

**Detection Rate Atteso**: 20-50% (slowloris è difficile)

### Step 3.5: Cleanup Docker

```bash
# Stop sniffer (Terminal 1)
Ctrl+C

# Cleanup Docker
docker stop nids_target
docker network rm nids_test_net

# Verifica cleanup
docker ps | grep nids_target  # Deve essere vuoto
docker network ls | grep nids_test_net  # Deve essere vuoto
```

**RISULTATI FASE 3**:

| Test | Detection Rate Atteso | Pass Condition |
|------|----------------------|----------------|
| Port Scan (nmap) | 30-70% | >30% |
| DoS SYN Flood | 50-80% | >50% |
| HTTP Slowloris | 20-50% | >20% |

**CHECKPOINT**:
-  Tutti i test >pass condition → Sistema VALIDATO
-  Qualche test sotto threshold → Rivedere feature mapping
-  Tutti i test fail → PROBLEMA CRITICO, ricontrollare FASE 0

---

## FASE 4: BLOCK MODE (OPZIONALE - 1 ora)

 **ATTENZIONE**: Test **PERICOLOSO**. Rischio auto-blocco IP.

### Step 4.1: Setup Whitelist (CRITICO!)

```bash
cd srcNF/live_sniffer
nano config.py
```

**Trova sezione WHITELIST_IPS e modifica**:
```python
WHITELIST_IPS: List[str] = [
    "127.0.0.1",              # Localhost
    "::1",                    # IPv6 localhost
    "172.20.0.1",             # Docker gateway (TUO IP!)
    "192.168.1.1",            # Gateway LAN
    "<TUO_IP_MACCHINA>",      # IP amministratore
]
```

**Trova TUO IP**:
```bash
# Metodo 1
ip addr show | grep "inet " | grep -v 127.0.0.1

# Metodo 2
hostname -I

# Output esempio:
# 192.168.1.50

# Aggiungi a WHITELIST_IPS!
```

**SALVA E ESCI**: `Ctrl+X`, `Y`, `Enter`

**Verifica Whitelist**:
```bash
python3 -c "from config import WHITELIST_IPS; print('Whitelist:', WHITELIST_IPS)"
```

### Step 4.2: Test BLOCK Mode (Docker Isolato)

```bash
# Ricrea ambiente Docker
docker network create --subnet=172.20.0.0/16 nids_test_net
docker run -d --name nids_target --net nids_test_net --ip 172.20.0.2 --rm nginx:alpine

# Trova interfaccia
INTERFACE=$(ip addr show | grep -B 1 "inet 172.20.0.1" | head -1 | awk '{print $2}' | tr -d ':')
echo "Interface: $INTERFACE"
```

**Terminal 1 - BLOCK Mode**:
```bash
cd srcNF/live_sniffer
sudo $(which python) main.py --mode block --interface $INTERFACE
```

**Output Atteso**:
```
======================================================================
NIDS LIVE SNIFFER STARTED
======================================================================
Mode: BLOCK
Interface: br-abc123
Firewall: iptables chain NIDS_BLOCK created
Whitelist: 172.20.0.1 protected
======================================================================
```

**Verifica Firewall Setup**:
```bash
# Terminal 2
sudo iptables -L NIDS_BLOCK -n -v

# Output atteso (vuoto all'inizio):
# Chain NIDS_BLOCK (1 references)
#  pkts bytes target     prot opt in     out     source        destination
```

**Terminal 3 - Generate Attack da Container NON Whitelisted**:
```bash
# Entra in container attacker
docker run -it --rm --net nids_test_net --ip 172.20.0.50 alpine sh

# Dentro container:
apk add nmap

# Scan target
nmap -sS -p 1-50 172.20.0.2

# Se bloccato vedrai timeout...
```

**Terminal 2 - Verifica Blocco**:
```bash
sudo iptables -L NIDS_BLOCK -n -v

# Deve mostrare regola DROP per 172.20.0.50:
#  pkts bytes target     prot opt in     out     source        destination
#     0     0 DROP       all  --  *      *       172.20.0.50   0.0.0.0/0
```

**Verifica Whitelist Protection**:
```bash
# Terminal 4 - Test da host (whitelisted)
nmap -sS -p 80 172.20.0.2

# Deve funzionare (no timeout)

# Verifica che 172.20.0.1 NON sia in iptables
sudo iptables -L NIDS_BLOCK -n | grep 172.20.0.1

# Deve essere vuoto (whitelisted, non bloccato)
```

**Terminal 1 - Stop Sniffer**:
```
Ctrl+C
```

**Output Atteso**:
```
Stopping Live NIDS Sniffer...
Cleaning up firewall...
Firewall chain removed
======================================================================
SNIFFER STOPPED
======================================================================
```

**Verifica Cleanup Automatico**:
```bash
sudo iptables -L NIDS_BLOCK

# Output atteso:
# iptables: No chain/target/match by that name.
# (BUONO! Chain rimossa)
```

**Cleanup Docker**:
```bash
docker stop nids_target
docker network rm nids_test_net
```

**CHECKPOINT BLOCK MODE**:
-  Firewall setup automatico
-  IP attacker bloccato
-  Whitelist protegge gateway
-  Cleanup automatico funziona

---

## FASE 5: METRICHE FINALI (30 min)

### Step 5.1: Analisi Performance

```bash
cd srcNF/live_sniffer

# Crea script analisi
python3 << 'EOF'
import pandas as pd
import json
from pathlib import Path

log_path = Path("../../logs/sniffer/nids_sniffer_alerts.csv")

if not log_path.exists():
    print("ERROR: No log file found")
    exit(1)

df = pd.read_csv(log_path)

print("="*70)
print("PERFORMANCE METRICS")
print("="*70)
print(f"\nTotal flows: {len(df):,}")

if 'prediction' in df.columns:
    benign = (df['prediction'] == 'benign').sum()
    attack = (df['prediction'] == 'attack').sum()
    
    print(f"Benign: {benign:,} ({benign/len(df)*100:.1f}%)")
    print(f"Attack: {attack:,} ({attack/len(df)*100:.1f}%)")
    
    if 'confidence' in df.columns:
        avg_conf = df['confidence'].mean()
        print(f"\nAverage confidence: {avg_conf:.3f}")
        
        if attack > 0:
            attack_df = df[df['prediction'] == 'attack']
            avg_attack_conf = attack_df['confidence'].mean()
            print(f"Attack confidence (avg): {avg_attack_conf:.3f}")
            
            # Top attacking IPs
            if 'src_ip' in attack_df.columns:
                top_ips = attack_df['src_ip'].value_counts().head(5)
                print(f"\nTop Attacking IPs:")
                for ip, count in top_ips.items():
                    print(f"  {ip}: {count} flows")

# Protocol distribution
if 'protocol' in df.columns:
    print(f"\nProtocol Distribution:")
    proto_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
    for proto, count in df['protocol'].value_counts().head(5).items():
        proto_name = proto_map.get(proto, f'Proto-{proto}')
        print(f"  {proto_name}: {count}")

print("="*70)
EOF
```

### Step 5.2: Summary Report

```bash
cd srcNF

# Crea report finale
cat > validation_report_$(date +%Y%m%d_%H%M%S).md << 'EOF'
# NIDS VALIDATION REPORT

## System Configuration
- Dataset: NF-UQ-NIDS-v2
- Features: 24 (nfstream-compatible)
- Model: XGBoost (incremental learning)
- Scaler: RobustScaler (aggregate, with outliers)

## Validation Results

### Component Tests
- [x] All 10/10 tests PASSED

### Functional Tests
- [x] Loopback traffic: OK
- [x] Log generation: OK (CSV + JSON)

### Attack Detection (Docker)
- Port Scan (nmap): XX% detection rate (>30% required)
- DoS SYN Flood: XX% detection rate (>50% required)
- HTTP Slowloris: XX% detection rate (>20% required)

### Block Mode (Optional)
- [x] Firewall integration: OK
- [x] Whitelist protection: OK
- [x] Auto-cleanup: OK

## Issues Identified

### Fixed
1. Feature mapper: MIN_TTL mapping corrected
2. Feature mapper: OUT_PKTS added
3. Feature mapper: CLIENT_TCP_FLAGS corrected
4. Feature mapper: ICMP_IPV4_TYPE added

### Known Limitations
1. Some features unavailable in NFStream (expected)
2. Detection rate lower than full-feature model (expected)
3. Slowloris detection challenging (known limitation)

## Conclusion
System validated and ready for deployment in ALERT mode.
BLOCK mode requires careful monitoring in production.

Date: $(date)
Validated by: [Your Name]
EOF

# Visualizza report
cat validation_report_*.md
```

---

## TROUBLESHOOTING

### Problema: "Invalid feature shape (35, expected 24)"

**Causa**: Artifacts vecchi con 35 feature

**Fix**:
```bash
cd srcNF

# Verifica feature count
cat artifacts/features.json | grep n_features

# Se mostra 35:
rm -rf artifacts/* models/*

# Ritraina con pipeline corretta
python3 pipeline.py --model xgboost
```

### Problema: Detection Rate Bassa (<20%)

**Possibili cause**:
1. Feature mapper non corretto (rivedi FASE 0)
2. Threshold troppo alto
3. Modello non converso

**Diagnosi**:
```bash
cd srcNF

# Check confidence distribution
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('logs/sniffer/nids_sniffer_alerts.csv')
print(df['confidence'].describe())

# Se avg confidence <0.3 → modello incerto
# Se avg confidence >0.8 → modello sicuro
EOF
```

**Fix**:
```bash
# Se confidence bassa → modello incerto, prova threshold più basso
nano live_sniffer/config.py

# Cambia:
ATTACK_THRESHOLD = 0.4  # Era 0.5

# Ritesta
```

### Problema: Sniffer Blocca su Ctrl+C

**Causa**: Loop attende traffico, non esce subito

**Fix**: Genera traffico dummy
```bash
# Altro terminal:
ping -c 1 <target_ip>

# Loop si sblocca e termina
```

### Problema: "Permission denied" su iptables

**Causa**: Non running con sudo

**Fix**:
```bash
# SEMPRE usa sudo per BLOCK mode
sudo $(which python) main.py --mode block --interface eth0
```

---

## COMANDI RAPIDI REFERENCE

```bash
# Test componenti
cd srcNF/live_sniffer
sudo $(which python) test_sniffer.py

# Avvia ALERT mode
sudo $(which python) main.py --mode alert --interface eth0

# Avvia BLOCK mode
sudo $(which python) main.py --mode block --interface eth0

# Monitor logs live
tail -f ../../logs/sniffer/nids_sniffer_alerts.csv

# Check iptables (BLOCK mode)
sudo iptables -L NIDS_BLOCK -n -v

# Cleanup manuale iptables
sudo iptables -D INPUT -j NIDS_BLOCK
sudo iptables -F NIDS_BLOCK
sudo iptables -X NIDS_BLOCK

# Analisi detection rate
cd srcNF
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('logs/sniffer/nids_sniffer_alerts.csv')
total = len(df)
attacks = (df['prediction'] == 'attack').sum()
print(f"Detection rate: {attacks/total*100:.1f}%")
EOF
```

---

## TEMPO TOTALE VALIDAZIONE

- Fase 0 (Correzione): **15 min**
- Fase 1 (Componenti): **30 min**
- Fase 2 (Funzionale): **15 min**
- Fase 3 (Docker Attack): **2 ore**
- Fase 4 (Block Mode): **1 ora** (opzionale)
- Fase 5 (Metriche): **30 min**

**Totale minimo**: 3.5 ore
**Totale completo**: 4.5 ore

---

## STATUS FINALE

Al completamento di tutte le fasi:

 **PASS**: Sistema validato e pronto per deployment
 **PASS CON WARNING**: Funziona ma detection rate sotto ottimale
 **FAIL**: Problemi critici, rivedere configurazione

**BUONA VALIDAZIONE!** 
EOF
