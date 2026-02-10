# PIANO TEST COMPLETO - PRE-PRODUZIONE

**Versione**: DEFINITIVA  
**Data**: 2026-02-10  
**Status modello**:  VALIDATO (98% accuracy su 100K test)

---

## FASE 1: VALIDAZIONE DATASET COMPLETO

### Test 1.1: Full Test Set (OBBLIGATORIO)

```bash
cd srcNF/live_sniffer/testing

# Test su TUTTO il test set (~11.4M righe)
python3 test_on_file.py \
    --parquet ../../data/processed/test_scaled.parquet \
    --output full_test_results.csv \
    --compare

# Tempo stimato: 10-15 minuti
```

**Metriche attese**:
- Accuracy: >0.97
- Precision: >0.95
- Recall: >0.95
- F1 Score: >0.96

**Se PASS**: Modello validato su dataset completo   
**Se FAIL (<95%)**: Verifica feature alignment

---

### Test 1.2: Validation Set

```bash
# Test su validation set (~1.7M righe)
python3 test_on_file.py \
    --parquet ../../data/processed/val_scaled.parquet \
    --output val_results.csv \
    --compare

# Tempo stimato: 2-3 minuti
```

**Scopo**: Confermare che performance è **consistente** tra test e val.

**Metriche attese**: Simili a Test 1.1 (differenza <2%)

---

## FASE 2: VALIDAZIONE SU PCAP REALI

### Test 2.1: Tuesday COMPLETO (Port Scan + DoS)

```bash
# Test su TUTTO il PCAP Tuesday
python3 test_on_file.py \
    --pcap ../../data/pcap/Tuesday-WorkingHours.pcap \
    --output tuesday_full_results.csv \
    --stats

# Tempo stimato: 30-60 minuti (dipende da size)
```

**Metriche attese**:
- Attack rate: 15-25%
- Avg confidence: >0.7
- Processing rate: >1000 flow/sec

**Analisi manuale**:
```bash
# Analizza distribuzione temporale attacchi
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tuesday_full_results.csv')
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

# Attack rate per ora
hourly = df.groupby('hour')['prediction'].apply(
    lambda x: (x == 'attack').sum() / len(x) * 100
)

print("Attack rate per ora:")
print(hourly)

# Plot
hourly.plot(kind='bar', title='Tuesday Attack Rate by Hour')
plt.ylabel('Attack Rate (%)')
plt.xlabel('Hour')
plt.savefig('tuesday_attack_timeline.png')
print("\nPlot salvato: tuesday_attack_timeline.png")
EOF
```

**Verifica**: Gli attacchi devono concentrarsi tra le 10:00-14:00 (Port Scan).

---

### Test 2.2: Wednesday (Normal + Web Attack)

```bash
# Se hai anche Wednesday PCAP
python3 test_on_file.py \
    --pcap ../../data/pcap/Wednesday-WorkingHours.pcap \
    --output wednesday_results.csv \
    --stats

# Tempo: 30-60 minuti
```

**Atteso**: Attack rate 5-15% (Web attacks)

---

## FASE 3: STRESS TEST & PERFORMANCE

### Test 3.1: Performance Benchmark

```bash
# Misura velocità processing su test set grande
cd srcNF/live_sniffer/testing

time python3 test_on_file.py \
    --parquet ../../data/processed/test_scaled.parquet \
    --output perf_test.csv \
    --max 1000000

# Calcola flow/sec
```

**Metriche target**:
- Processing rate: >1500 rows/sec (CSV/Parquet)
- Processing rate: >500 flow/sec (PCAP live extraction)
- RAM usage: <4GB durante processing

**Verifica RAM**:
```bash
# In altro terminale durante test
watch -n 1 "ps aux | grep python | grep test_on_file"
```

---

### Test 3.2: Confidence Distribution

```bash
# Analizza distribuzione confidence
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_csv('full_test_results.csv')

print("CONFIDENCE DISTRIBUTION")
print("="*50)
print(df['confidence'].describe())
print()

# Percentili
print("Percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(df['confidence'], p)
    print(f"  P{p:02d}: {val:.4f}")

print()

# Confidence per classe
benign_conf = df[df['prediction'] == 'benign']['confidence']
attack_conf = df[df['prediction'] == 'attack']['confidence']

print(f"Benign avg confidence: {benign_conf.mean():.4f}")
print(f"Attack avg confidence: {attack_conf.mean():.4f}")

# Low confidence cases (<0.6)
low_conf = df[df['confidence'] < 0.6]
print(f"\nLow confidence (<0.6): {len(low_conf)} flows ({len(low_conf)/len(df)*100:.2f}%)")
EOF
```

**Target**:
- Median confidence: >0.85
- <5% flows with confidence <0.6
- Attack avg conf > Benign avg conf

---

## FASE 4: EDGE CASES & ROBUSTNESS

### Test 4.1: PCAP Malformato

```bash
# Crea PCAP con pacchetti malformati (se disponibile)
# Testa robustezza error handling

python3 test_on_file.py \
    --pcap malformed_traffic.pcap \
    --output robust_test.csv \
    --stats
```

**Verifica**: Error rate <5%, no crash

---

### Test 4.2: ICMP Flood Detection

```bash
# Genera traffico ICMP per test ICMP_IPV4_TYPE feature
# Oppure usa PCAP con ICMP flood

# Su sistema Linux:
sudo hping3 -1 --flood -c 1000 <target_ip>

# Cattura in PCAP e testa
```

**Verifica**: ICMP flood rilevato come attack (grazie a ICMP_IPV4_TYPE)

---

## FASE 5: THRESHOLD TUNING (OPZIONALE)

### Test 5.1: ROC Curve Analysis

```bash
# Test vari threshold
cd srcNF/live_sniffer/testing

for thresh in 0.3 0.4 0.5 0.6 0.7 0.8; do
    # Modifica temporaneamente config.py
    sed -i.bak "s/ATTACK_THRESHOLD: float = .*/ATTACK_THRESHOLD: float = $thresh/" ../config.py
    
    python3 test_on_file.py \
        --parquet ../../data/processed/test_scaled.parquet \
        --output thresh_${thresh}_results.csv \
        --compare \
        --max 50000
    
    # Ripristina
    mv ../config.py.bak ../config.py
done

# Analizza risultati
python3 << 'EOF'
import pandas as pd
import json

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
results = []

for t in thresholds:
    with open(f'thresh_{t}_results_stats.json') as f:
        stats = json.load(f)
    results.append({
        'threshold': t,
        'accuracy': stats['accuracy'],
        'precision': stats['precision'],
        'recall': stats['recall'],
        'f1': stats['f1_score'],
        'fpr': stats['false_positives'] / (stats['false_positives'] + stats['true_negatives'])
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Best F1
best = df.loc[df['f1'].idxmax()]
print(f"\nBest threshold: {best['threshold']} (F1={best['f1']:.4f})")
EOF
```

**Scopo**: Trovare threshold ottimale per tuo use case.

---

## FASE 6: SIMULAZIONE LIVE (PRE-DEPLOYMENT)

### Test 6.1: Loopback Traffic

```bash
cd srcNF/live_sniffer

# Terminal 1: Avvia sniffer su loopback
sudo python3 main.py --mode alert --interface lo

# Terminal 2: Genera traffico normale
ping -c 100 127.0.0.1
curl http://127.0.0.1

# Terminal 3: Monitor log
tail -f ../../logs/sniffer/nids_sniffer_alerts.csv
```

**Verifica**:
- Sniffer processa flow
- No crash
- Log scritto correttamente
- Traffico normale → benign (no false positive)

---

### Test 6.2: Simulated Port Scan

```bash
# Terminal 1: Sniffer live
sudo python3 main.py --mode alert --interface lo

# Terminal 2: Port scan su loopback
nmap -sS -p 1-1000 127.0.0.1

# Verifica log
grep "attack" ../../logs/sniffer/nids_sniffer_alerts.csv | wc -l
```

**Atteso**: Port scan rilevato come attack (>50% dei flow)

---

### Test 6.3: Docker Attack Simulation

```bash
# Setup Docker network
docker network create --subnet=172.20.0.0/16 nids_test
docker run -d --name target --net nids_test --ip 172.20.0.2 nginx:alpine

# Trova interface Docker
IFACE=$(ip addr | grep "172.20.0.1" | awk '{print $NF}')

# Terminal 1: Sniffer
sudo python3 main.py --mode alert --interface $IFACE

# Terminal 2: Attack simulation
# Port scan
sudo nmap -sS -p 1-100 172.20.0.2

# SYN flood
sudo hping3 -S -p 80 --flood -c 1000 172.20.0.2

# HTTP flood
ab -n 1000 -c 50 http://172.20.0.2/
```

**Metriche target**:
- Port scan detection: >60%
- SYN flood detection: >70%
- HTTP flood detection: >40%

---

## FASE 7: BLOCK MODE TEST (SE RICHIESTO)

### Test 7.1: Block Mode Validation

** CRITICAL**: Configura whitelist prima!

```python
# In config.py
WHITELIST_IPS: List[str] = [
    "127.0.0.1",
    "::1",
    "172.20.0.1",  # Docker gateway
    "<YOUR_IP>",    # TUO IP admin
]
```

```bash
# Test BLOCK mode
sudo python3 main.py --mode block --interface $IFACE

# Simula attack da IP NON in whitelist
# Verifica che IP viene bloccato

# Check iptables
sudo iptables -L NIDS_BLOCK -v -n
```

**Verifica**:
- IP attaccante bloccato
- IP in whitelist mai bloccato
- Cleanup corretto on exit

---

## CHECKLIST FINALE PRE-PRODUZIONE

### Performance 
- [ ] Test set completo: Accuracy >97%
- [ ] Processing rate: >500 flow/sec (PCAP)
- [ ] RAM usage: <4GB
- [ ] No memory leak (test 1h+)

### Robustness 
- [ ] Error handling: <5% error rate
- [ ] PCAP malformati: no crash
- [ ] Edge cases: gestiti correttamente

### Detection 
- [ ] Port scan: >60% detection
- [ ] DoS/DDoS: >70% detection
- [ ] False positive: <5%
- [ ] Tuesday PCAP: attack rate 15-25%

### Operational 
- [ ] Loopback test: OK
- [ ] Docker simulation: OK
- [ ] Log scritto correttamente
- [ ] Alert manager funzionante

### Security (se BLOCK mode) 
- [ ] Whitelist configurata
- [ ] Block funzionante
- [ ] Cleanup iptables OK
- [ ] No self-block

---

## RISULTATI ATTUALI

**Test completato**: Test 1.1 (100K sample)
-  Accuracy: 98.02%
-  Precision: 98.81%
-  Recall: 98.23%
-  F1: 98.52%

**Status**: **READY FOR FULL VALIDATION**

**Next step**: Esegui Test 1.1 su FULL test set (11.4M righe)

---

## COMANDI QUICK START

```bash
cd srcNF/live_sniffer/testing

# Test rapido (già fatto)
python3 test_on_file.py --parquet ../../data/processed/test_scaled.parquet --output quick.csv --compare --max 100000

# Test COMPLETO (ESEGUI QUESTO)
python3 test_on_file.py --parquet ../../data/processed/test_scaled.parquet --output full_validation.csv --compare

# Test Tuesday COMPLETO
python3 test_on_file.py --pcap ../../data/pcap/Tuesday-WorkingHours.pcap --output tuesday_full.csv --stats

# Live test (loopback)
cd ..
sudo python3 main.py --mode alert --interface lo
```

---

**RACCOMANDAZIONE**: Esegui almeno Test 1.1 (full test set) prima di deployment live.
