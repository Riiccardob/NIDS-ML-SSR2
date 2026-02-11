# NIDS-ML-SSR2 -- Setup Linode con Sniffer Remoto

**Versione**: 8.0 (architettura corretta NIDS)
**Data**: 2026-02-17

---

## ARCHITETTURA

```
[TUO PC]  ----attacchi HTTP/nmap---→  [SERVER LINODE]
(attaccante)                           (sniffer NIDS + target)
                                       ↓
                                    [Interfaccia eth0]
                                       ↓
                                    [nfstream cattura]
                                       ↓
                                    [Modello XGBoost]
                                       ↓
                                    [Alert log]
```

**Flusso**:
1. Tu lanci attacchi dal PC verso IP pubblico Linode
2. Traffico arriva sull'interfaccia `eth0` del server
3. Sniffer NIDS cattura tutto il traffico in ingresso
4. Modello classifica flow e genera alert
5. Tu vedi i risultati nei log del server

---

## PARTE 1: SETUP SERVER LINODE

### 1.1 - Rebuild e Accesso

**Linode Dashboard** → Server → **Rebuild**:
- Distribuzione: **Ubuntu 24.04 LTS**
- Root password: imposta password forte
- SSH key: aggiungi la tua (opzionale)

Attendi rebuild (2-3 minuti), poi:

```bash
# Dal tuo PC, connettiti
ssh root@<IP_LINODE>
```

### 1.2 - Installazione Dipendenze Sistema

```bash
# Aggiorna sistema
apt update && apt upgrade -y

# Dipendenze Python e librerie
apt install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    nginx \
    build-essential \
    libpcap-dev \
    pkg-config

# Verifica Python
python3 --version
# Atteso: Python 3.12.x
```

### 1.3 - Clona Progetto sul Server

```bash
# Crea directory
mkdir -p /opt/nids
cd /opt/nids

# Se hai il progetto su GitHub, clona:
# git clone https://github.com/TUO_USERNAME/NIDS-ML-SSR2.git
# cd NIDS-ML-SSR2

# OPPURE trasferisci da locale con scp (dal tuo PC):
# cd ~/Documents/NIDS-ML-SSR2
# tar czf nids.tar.gz srcNF/ artifacts/ models/ requirements.txt
# scp nids.tar.gz root@<IP_LINODE>:/opt/nids/
# 
# Poi sul server:
# cd /opt/nids
# tar xzf nids.tar.gz
```

**Metodo alternativo - Upload manuale artifacts**:

Sul server Linode, crea la struttura:
```bash
cd /opt/nids
mkdir -p artifacts models/xgboost srcNF/live_sniffer
```

Dal tuo PC, trasferisci i file essenziali:
```bash
# Artifacts
scp ~/Documents/NIDS-ML-SSR2/artifacts/features.json root@<IP_LINODE>:/opt/nids/artifacts/
scp ~/Documents/NIDS-ML-SSR2/artifacts/scaler.pkl root@<IP_LINODE>:/opt/nids/artifacts/

# Modello
scp ~/Documents/NIDS-ML-SSR2/models/xgboost/model.pkl root@<IP_LINODE>:/opt/nids/models/xgboost/

# Tutto srcNF/live_sniffer (comprimi prima)
cd ~/Documents/NIDS-ML-SSR2
tar czf sniffer.tar.gz srcNF/live_sniffer/
scp sniffer.tar.gz root@<IP_LINODE>:/opt/nids/
# Sul server:
cd /opt/nids
tar xzf sniffer.tar.gz
```

### 1.4 - Setup Virtual Environment Python

Sul server:
```bash
cd /opt/nids

# Crea venv
python3 -m venv venv

# Attiva
source venv/bin/activate

# Installa dipendenze
pip install --upgrade pip
pip install \
    nfstream==6.5.3 \
    scikit-learn==1.6.1 \
    xgboost==2.1.3 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    joblib==1.4.2 \
    psutil==6.1.0

# Verifica installazione
python - <<'EOF'
import nfstream
import xgboost
import sklearn
print("nfstream:", nfstream.__version__)
print("xgboost:", xgboost.__version__)
print("sklearn:", sklearn.__version__)
EOF
```

Atteso:
```
nfstream: 6.5.3
xgboost: 2.1.3
sklearn: 1.6.1
```

### 1.5 - Verifica Struttura Progetto

```bash
cd /opt/nids
tree -L 3 -I 'venv|__pycache__'
```

Atteso:
```
.
 artifacts
    features.json
    scaler.pkl
 models
    xgboost
        model.pkl
 srcNF
     live_sniffer
         main.py
         config.py
         core/
         security/
         utils/
```

### 1.6 - Identifica Interfaccia di Rete

```bash
# Lista interfacce
ip link show

# Output tipico Linode:
# 1: lo: <LOOPBACK,UP> ...
# 2: eth0: <BROADCAST,MULTICAST,UP> ...

# Verifica IP pubblico su eth0
ip addr show eth0 | grep "inet "
```

L'interfaccia principale è **eth0** (quasi sempre su Linode).

### 1.7 - Setup Target HTTP (Nginx)

```bash
# Nginx già installato, avvialo
systemctl start nginx
systemctl enable nginx

# Verifica locale
curl -I http://localhost/
# Atteso: HTTP/1.1 200 OK

# Verifica da remoto (dal tuo PC)
curl -I http://<IP_LINODE>/
# Deve rispondere 200 OK
```

### 1.8 - Configura Firewall Linode

**Linode Dashboard** → Firewalls:

Se hai un firewall attivo, aggiungi regole:
- **Inbound**: Accept TCP port 80 (HTTP) from 0.0.0.0/0
- **Inbound**: Accept TCP port 22 (SSH) from 0.0.0.0/0
- **Inbound**: Accept TCP port 8080 (HTTP alt) from 0.0.0.0/0 (opzionale)

---

## PARTE 2: TEST SNIFFER SUL SERVER

### 2.1 - Test Iniziale (cattura locale)

Sul server:
```bash
cd /opt/nids/srcNF/live_sniffer
source ../../venv/bin/activate

# Test cattura su loopback (veloce)
sudo $(which python) main.py \
    --mode alert \
    --interface lo \
    --fast \
    --verbose \
    --cooldown 3
```

**In un altro terminale SSH** (connettiti di nuovo al server):
```bash
# Genera traffico locale
curl http://localhost/ > /dev/null
curl http://localhost/ > /dev/null
curl http://localhost/ > /dev/null
```

**Nel terminale dello sniffer**, dovresti vedere:
```
[SAFE]   127.0.0.1:PORT -> 127.0.0.1:80  proto=HTTP  conf=0.XXXX
```

Premi `Ctrl+C` per fermare.

**Se questo funziona**, la pipeline è corretta. Procedi.

### 2.2 - Test su Interfaccia Pubblica (eth0)

Sul server:
```bash
cd /opt/nids/srcNF/live_sniffer
source ../../venv/bin/activate

# IMPORTANTE: cattura su eth0 (interfaccia pubblica)
sudo $(which python) main.py \
    --mode alert \
    --interface eth0 \
    --fast \
    --verbose \
    --cooldown 3
```

**Dal tuo PC** (Terminal separato):
```bash
# Genera traffico HTTP verso il server
curl http://<IP_LINODE>/
```

**Sul server**, dovresti vedere:
```
[SAFE]   <TUO_IP_PUBBLICO>:PORT -> <IP_LINODE>:80  proto=HTTP  conf=0.XXXX
```

**Se vedi questo**, lo sniffer cattura correttamente traffico da remoto!

---

## PARTE 3: DEMO LIVE

### 3.1 - Avvio Sniffer (Server - Terminal 1)

Sul server Linode:
```bash
cd /opt/nids/srcNF/live_sniffer
source ../../venv/bin/activate

# Avvia sniffer su interfaccia pubblica
sudo $(which python) main.py \
    --mode alert \
    --interface eth0 \
    --fast \
    --verbose \
    --cooldown 3
```

Lascia questo terminale aperto. Puoi usare `tmux` o `screen` per non perdere la sessione.

### 3.2 - Monitoraggio Log (Server - Terminal 2)

In un **secondo terminale SSH** al server:
```bash
# Segui alert in tempo reale
tail -f /opt/nids/logs/sniffer/nids_sniffer_alerts.csv

# In alternativa, segui log principale
tail -f /opt/nids/logs/sniffer/nids_sniffer.log
```

### 3.3 - Attacchi dal Tuo PC

Sul tuo PC:
```bash
# Salva IP per comodità
export TARGET="<IP_LINODE>"
```

#### Attacco 1: Traffico Benigno

```bash
# 10 richieste HTTP normali con pausa
for i in {1..10}; do
    curl -s http://${TARGET}/ > /dev/null
    echo "Richiesta $i inviata"
    sleep 1
done
```

**Sul server Terminal 1**, dovresti vedere prevalentemente:
```
[SAFE]   <TUO_IP>:PORT -> <IP_LINODE>:80  proto=HTTP  conf=0.XXXX
```

#### Attacco 2: HTTP Flood Moderato

```bash
# 500 richieste, 30 connessioni parallele
ab -n 500 -c 30 -q http://${TARGET}/
```

**Sul server**, dovresti vedere:
```
[ATTACK] <TUO_IP>:PORT -> <IP_LINODE>:80  proto=HTTP  conf=0.7XXX
[SAFE]   <TUO_IP>:PORT -> <IP_LINODE>:80  proto=HTTP  conf=0.6XXX
(mix di rosso e grigio)
```

**Pausa**:
```bash
sleep 5
```

#### Attacco 3: HTTP Flood Intenso (IL MIGLIORE)

```bash
# 2000 richieste, 80 connessioni parallele
ab -n 2000 -c 80 http://${TARGET}/
```

**Sul server**, dovresti vedere:
```
[ATTACK] <TUO_IP>:PORT -> <IP_LINODE>:80  proto=HTTP  conf=0.9XXX
[ATTACK] <TUO_IP>:PORT -> <IP_LINODE>:80  proto=HTTP  conf=0.8XXX
(muro rosso continuo)
```

Detection rate atteso: **80-95%**

#### Attacco 4 (Opzionale): Port Scan

```bash
# Scan porte comuni
sudo nmap -sS -p 22,80,443,8080,3306 ${TARGET}
```

**Sul server**, risultati variabili:
- Alcuni flow: `[ATTACK]` conf 0.7-0.9
- Altri flow: `[SAFE]` conf 0.9+

Detection rate atteso: **30-50%** (migliorato rispetto a loopback, ma non perfetto)

#### Attacco 5 (Opzionale): SYN Flood

```bash
# SYN flood verso porta 80 (BREVE, 5000 pacchetti)
sudo hping3 -S --flood -p 80 ${TARGET} -c 5000
```

**ATTENZIONE**: questo è un attacco reale verso il tuo server. Usalo solo brevemente e con count limitato (`-c 5000`).

**Sul server**, dovresti vedere alcuni flow classificati come `[ATTACK]`.

---

## PARTE 4: RISULTATI ATTESI

### Miglioramenti rispetto a Loopback

| Scenario | Loopback | Linode Remoto | Miglioramento |
|----------|----------|---------------|---------------|
| HTTP flood | 99% ATTACK | 85-95% ATTACK | Stabile |
| Traffico benigno | 30% SAFE | 70-80% SAFE | +133% |
| Port scan | 1% ATTACK | 30-50% ATTACK | +3000% |
| SYN flood | 1% ATTACK | 20-40% ATTACK | +2000% |

### Perché Funziona Meglio

1. **Latenza reale**: 20-100ms invece di 0.01ms
2. **MTU standard**: 1500 byte invece di 65536
3. **Routing internet**: packet loss, jitter, riordino naturali
4. **Throughput limitato**: banda 100Mbps-1Gbps invece di 10Gbps+

Tutte queste caratteristiche avvicinano il traffico catturato al dataset NF-UQ-NIDS-v2.

---

## PARTE 5: MONITORAGGIO E ANALISI

### 5.1 - Statistiche Real-time

Sul server (Terminal 3):
```bash
# Conta alert generati
grep -c "attack" /opt/nids/logs/sniffer/nids_sniffer_alerts.csv

# Ultimi 10 alert
tail -10 /opt/nids/logs/sniffer/nids_sniffer_alerts.csv

# Stats aggregate
tail -5 /opt/nids/logs/sniffer/nids_sniffer_stats.jsonl | python3 -m json.tool
```

### 5.2 - Analisi Post-Demo

```bash
cd /opt/nids/srcNF/live_sniffer

# Conta alert per IP sorgente
python3 - <<'EOF'
import csv
from collections import Counter

with open('logs/sniffer/nids_sniffer_alerts.csv') as f:
    reader = csv.DictReader(f)
    ips = [row['src_ip'] for row in reader if row['prediction'] == 'attack']
    
print("Top 5 IP attaccanti:")
for ip, count in Counter(ips).most_common(5):
    print(f"  {ip}: {count} alert")
EOF
```

### 5.3 - Download Log per Analisi Locale

Dal tuo PC:
```bash
# Scarica log
scp root@<IP_LINODE>:/opt/nids/logs/sniffer/nids_sniffer_alerts.csv ~/Downloads/
scp root@<IP_LINODE>:/opt/nids/logs/sniffer/nids_sniffer_stats.jsonl ~/Downloads/

# Analizza con pandas
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv('~/Downloads/nids_sniffer_alerts.csv')
print(df['prediction'].value_counts())
print(df.groupby('prediction')['confidence'].describe())
EOF
```

---

## PARTE 6: CLEANUP E PERSISTENZA

### 6.1 - Fermare Sniffer

Sul server, nel Terminal 1 (dove gira lo sniffer):
```bash
# Premi Ctrl+C
^C

# Output:
# Segnale 2 ricevuto, arresto in corso...
# STATISTICHE FINALI
# ...
# SNIFFER FERMATO
```

### 6.2 - Setup Persistente con Systemd (Opzionale)

Se vuoi far partire lo sniffer automaticamente al boot:

Sul server:
```bash
cat > /etc/systemd/system/nids-sniffer.service <<'EOF'
[Unit]
Description=NIDS Live Sniffer
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/nids/srcNF/live_sniffer
Environment="PATH=/opt/nids/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/nids/venv/bin/python main.py --mode alert --interface eth0 --fast
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Abilita e avvia
systemctl daemon-reload
systemctl enable nids-sniffer
systemctl start nids-sniffer

# Verifica stato
systemctl status nids-sniffer

# Log in tempo reale
journalctl -u nids-sniffer -f
```

---

## PARTE 7: TROUBLESHOOTING

### Problema 1: "Cannot open device eth0: Permission denied"

**Causa**: non hai usato `sudo`.

**Fix**:
```bash
sudo $(which python) main.py ...
```

### Problema 2: Sniffer non vede traffico remoto

**Causa**: stai catturando su interfaccia sbagliata.

**Fix**: verifica interfaccia con IP pubblico:
```bash
ip addr show | grep "inet " | grep -v "127.0.0.1"
# Identifica quella con IP pubblico Linode
```

### Problema 3: "ModuleNotFoundError: No module named 'nfstream'"

**Causa**: venv non attivato o pacchetti non installati.

**Fix**:
```bash
cd /opt/nids
source venv/bin/activate
pip install nfstream==6.5.3
```

### Problema 4: ab dal PC non riesce a connettersi

**Causa**: firewall Linode blocca la porta.

**Fix**: Dashboard Linode → Firewalls → Verifica regola TCP 80 Accept.

### Problema 5: Troppi alert su traffico benigno

**Causa**: threshold troppo basso.

**Fix**: aumenta threshold in `config.py`:
```python
ATTACK_THRESHOLD = 0.8  # invece di 0.7
```

---

## PARTE 8: CHECKLIST PRE-DEMO

Sul server Linode:
- [ ] Ubuntu 24.04 installato
- [ ] Python 3.12 + dipendenze installate
- [ ] Progetto in `/opt/nids` con artifacts, modello, codice
- [ ] Venv creato con nfstream, xgboost, sklearn
- [ ] Nginx funzionante su porta 80
- [ ] Firewall aperto su porta 80
- [ ] Test locale sniffer (loopback) funzionante
- [ ] Test remoto sniffer (eth0) funzionante

Sul tuo PC:
- [ ] Tool installati: `ab`, `curl`, `nmap`, `hping3`
- [ ] Connettività verificata: `curl http://<IP_LINODE>/` risponde
- [ ] IP Linode salvato: `export TARGET="<IP>"`

---

## COMANDO RAPIDO FINALE

**Server (Terminal 1 - Sniffer)**:
```bash
cd /opt/nids/srcNF/live_sniffer
source ../../venv/bin/activate
sudo $(which python) main.py --mode alert --interface eth0 --fast --verbose --cooldown 3
```

**Server (Terminal 2 - Monitor)**:
```bash
tail -f /opt/nids/logs/sniffer/nids_sniffer_alerts.csv
```

**PC (Attacchi)**:
```bash
export TARGET="<IP_LINODE>"

# Benigno
for i in {1..10}; do curl -s http://${TARGET}/ > /dev/null; sleep 1; done

# Moderato
ab -n 500 -c 30 -q http://${TARGET}/
sleep 5

# Intenso (QUESTO FUNZIONA SICURO)
ab -n 2000 -c 80 http://${TARGET}/
```

---

Questa è la configurazione corretta per un **vero NIDS** che protegge un server remoto.
