# NIDS-ML-SSR2: Network Intrusion Detection System

Machine Learning-based Network Intrusion Detection System using NetFlow features extracted via nfstream. The system consists of two main components: an offline training pipeline for model development and a live deployment sniffer for real-time network monitoring.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Part 1: Model Training Pipeline](#part-1-model-training-pipeline)
- [Part 2: Live Network Sniffer](#part-2-live-network-sniffer)
- [Dataset](#dataset)
- [Performance](#performance)
- [Limitations and Known Issues](#limitations-and-known-issues)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

NIDS-ML-SSR2 implements a supervised machine learning approach to network intrusion detection based on NetFlow feature analysis. The system analyzes network flow characteristics (packet counts, byte counts, timing statistics, protocol information) to classify traffic as benign or malicious.

**Key Features**:
- NetFlow-based detection using nfstream for feature extraction
- XGBoost classifier trained on NF-UQ-NIDS-v2 dataset (76M+ flows)
- Real-time traffic analysis with sub-second latency
- Modular architecture separating training from deployment
- Two operational modes: ALERT (logging only) and BLOCK (with iptables integration)
- Comprehensive logging and statistics tracking
- Memory-efficient chunk-based processing for large datasets

## Architecture

The system is divided into two independent components:

### 1. Training Pipeline (`srcNF/*.py`)

Offline machine learning pipeline for model development:

```
Raw CSV Data → Preprocessing → Feature Engineering → Model Training → Artifacts
                (chunk-based)    (RobustScaler)      (XGBoost)       (pkl files)
```

**Input**: NF-UQ-NIDS-v2 dataset (CSV format, 76M records, 43 features)  
**Output**: Trained model, fitted scaler, feature metadata  
**Processing**: Chunk-based to handle large datasets without loading entire dataset into RAM

### 2. Live Sniffer (`srcNF/live_sniffer/`)

Real-time network monitoring system:

```
Network Interface → Packet Capture → Flow Generation → Feature Extraction → 
(eth0, wlan0)       (nfstream)        (NetFlow)         (18 features)

→ Preprocessing → Model Inference → Alert/Block → Logging
  (RobustScaler)   (XGBoost)         (iptables)     (CSV/JSONL)
```

**Input**: Live network traffic on specified interface  
**Output**: Alert logs (CSV), statistics (JSONL), optional firewall rules  
**Processing**: Real-time with configurable batch inference (default: 100 flows / 2s)

## Project Structure

```
NIDS-ML-SSR2/

 srcNF/                          # Main source directory
   
    config.py                   # Training pipeline configuration
    preprocessing.py            # Data cleaning and splitting
    feature_engineering.py     # Scaler fitting and feature selection
    training.py                 # Model training (XGBoost/LightGBM)
    pipeline.py                 # Orchestrator for full training pipeline
    utils.py                    # Shared utilities (logging, metrics)
   
    live_sniffer/               # Live deployment system
       
        main.py                 # Entry point for live sniffer
        config.py               # Sniffer configuration
        requirements.txt        # Python dependencies
       
        core/                   # Core inference components
           capture.py          # Network capture with nfstream
           feature_mapper.py   # NetFlow → model feature conversion
           preprocessor.py     # RobustScaler application
           predictor.py        # Model inference wrapper
       
        security/               # Security response modules
           alert_manager.py    # Alert generation and rate limiting
           firewall_controller.py  # iptables integration (BLOCK mode)
       
        utils/                  # Sniffer utilities
           logger.py           # Structured logging (CSV/JSONL)
       
        GUIDES/                 # Deployment documentation
           NIDS_TEST_ROADMAP.md       # Testing roadmap
           loopback/
              DEMO_LOOPBACK_GUIDE.md
           linode/
               LINODE_NIDS_SETUP.md
       
        test_on_file.py         # Offline validation tool (PCAP/CSV/Parquet)
        test_sniffer.py         # Component integration tests

 artifacts/                      # Generated artifacts
    features.json               # Feature metadata (18 features)
    scaler.pkl                  # Fitted RobustScaler
    model.pkl                   # Trained XGBoost model (if using sklearn API)

 models/                         # Trained models
    xgboost/
        model.pkl               # XGBoost classifier

 data/                           # Dataset directory (not in repo)
    raw/                        # Original CSV files
    processed/                  # Parquet files (train/val/test)
    pcap/                       # PCAP files for validation

 logs/                           # Log directory (created at runtime)
     training/                   # Training logs
     sniffer/                    # Live sniffer logs
         nids_sniffer.log        # General log
         nids_sniffer_alerts.csv # Attack alerts
         nids_sniffer_benign.csv # Benign samples
         nids_sniffer_stats.jsonl # System statistics
```

## Prerequisites

### Hardware Requirements

**Training Pipeline**:
- RAM: 8-16 GB (depends on chunk size configuration)
- CPU: Multi-core recommended (XGBoost benefits from parallelization)
- Disk: 50-100 GB free space (dataset + intermediate files)

**Live Sniffer**:
- RAM: 2-4 GB
- CPU: 2+ cores
- Network: Interface with promiscuous mode support

### Software Requirements

- Python 3.10 or higher
- Ubuntu 20.04+ / Debian 11+ (for live sniffer with iptables)
- Root privileges (for packet capture)

### Python Dependencies

Core libraries:
- `nfstream >= 6.5.3` - Network flow capture and feature extraction
- `xgboost >= 2.1.0` - Gradient boosting classifier
- `scikit-learn >= 1.6.0` - Preprocessing and metrics
- `pandas >= 2.2.0` - Data manipulation
- `numpy >= 1.26.0` - Numerical operations
- `pyarrow >= 10.0.0` - Parquet file handling
- `joblib >= 1.4.0` - Model serialization
- `psutil >= 6.0.0` - System monitoring

See `srcNF/live_sniffer/requirements.txt` for complete dependency list.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/NIDS-ML-SSR2.git
cd NIDS-ML-SSR2
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
# For training pipeline
pip install -r srcNF/requirements.txt

# For live sniffer
pip install -r srcNF/live_sniffer/requirements.txt
```

### 4. Download Dataset

Download the NF-UQ-NIDS-v2 dataset and place it in `data/raw/`:

```bash
mkdir -p data/raw
# Download NF-UQ-NIDS-v2.csv to data/raw/
```

Dataset source: [NF-UQ-NIDS-v2 on Kaggle](https://www.kaggle.com/datasets/your-dataset-link)

## Part 1: Model Training Pipeline

The training pipeline converts raw NetFlow CSV data into a trained machine learning model.

### Configuration

Edit `srcNF/config.py` to adjust:

```python
# Dataset paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Feature selection (18 features used)
REQUIRED_FEATURES = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES",
    "IN_PKTS", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    # ... (see config.py for full list)
]

# Model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # ...
}

# Processing
CHUNK_SIZE = 500_000  # Rows per chunk
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

### Pipeline Execution

#### Option 1: Full Pipeline (Recommended for First Run)

```bash
cd srcNF
python pipeline.py --model xgboost
```

This executes:
1. **Preprocessing**: CSV → Parquet, stratified train/val/test split
2. **Feature Engineering**: Fit RobustScaler on training data
3. **Training**: Train XGBoost classifier with incremental learning
4. **Validation**: Evaluate on validation and test sets

**Output**:
- `artifacts/features.json` - Feature metadata
- `artifacts/scaler.pkl` - Fitted RobustScaler
- `models/xgboost/model.pkl` - Trained XGBoost model

**Expected Duration**: 30-90 minutes (depends on hardware)

#### Option 2: Individual Steps

**Preprocessing only**:
```bash
python preprocessing.py
```

**Feature engineering only** (requires preprocessed data):
```bash
python feature_engineering.py
```

**Training only** (requires preprocessed + scaled data):
```bash
python training.py --model xgboost
```

**Full pipeline with LightGBM**:
```bash
python pipeline.py --model lightgbm
```

**Skip preprocessing** (if already done):
```bash
python pipeline.py --model xgboost --skip-preprocessing
```

### Model Performance Metrics

Expected results on NF-UQ-NIDS-v2 test set (11.4M flows):

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 97.4%  |
| Precision | 98.5%  |
| Recall    | 97.6%  |
| F1 Score  | 98.0%  |

Confusion matrix:
- True Positives: 7.44M (attacks correctly detected)
- True Negatives: 3.66M (benign correctly identified)
- False Positives: 115K (benign misclassified as attack, 1.5%)
- False Negatives: 183K (attacks missed, 2.4%)

## Part 2: Live Network Sniffer

Real-time network intrusion detection system.

### Prerequisites for Live Sniffer

1. **Trained model artifacts** (from Part 1):
   - `artifacts/features.json`
   - `artifacts/scaler.pkl`
   - `models/xgboost/model.pkl`

2. **Root privileges** (for packet capture)

3. **Network interface** in promiscuous mode

### Configuration

Edit `srcNF/live_sniffer/config.py`:

```python
# Network
NETWORK_INTERFACE = None  # None = auto-detect, or "eth0", "wlan0"

# Operation
OPERATION_MODE = OperationMode.ALERT  # ALERT or BLOCK
ATTACK_THRESHOLD = 0.7  # Classification threshold (0.5-0.9)

# Performance
INFERENCE_BATCH_SIZE = 100  # Flows per batch
INFERENCE_BATCH_TIMEOUT = 2.0  # Max seconds before forcing batch
FLOW_IDLE_TIMEOUT = 120  # Seconds before closing idle flow
FLOW_ACTIVE_TIMEOUT = 1800  # Max flow duration

# Logging
STATS_LOG_INTERVAL = 60  # Statistics log frequency (seconds)
```

### Running the Sniffer

#### Basic Usage (ALERT Mode)

```bash
cd srcNF/live_sniffer
sudo $(which python) main.py --mode alert --interface eth0
```

This will:
- Capture packets on interface `eth0`
- Extract NetFlow features
- Classify flows as benign/attack
- Log alerts to `logs/sniffer/nids_sniffer_alerts.csv`
- Display statistics every 60 seconds

#### Advanced Options

**Demo mode** (fast timeouts, verbose output):
```bash
sudo $(which python) main.py \
    --mode alert \
    --interface eth0 \
    --fast \
    --verbose \
    --cooldown 3
```

**BLOCK mode** (with iptables firewall integration):
```bash
sudo $(which python) main.py \
    --mode block \
    --interface eth0
```

**Custom batch configuration**:
```bash
sudo $(which python) main.py \
    --mode alert \
    --interface eth0 \
    --batch-size 200 \
    --batch-timeout 5.0
```

**All CLI options**:
```
--mode {alert,block}       Operation mode (default: alert)
--interface IFACE          Network interface (default: auto-detect)
--batch-size N             Flows per inference batch (default: 100)
--batch-timeout SEC        Max batch wait time (default: 2.0)
--idle-timeout SEC         Flow idle timeout (default: 120)
--active-timeout SEC       Flow active timeout (default: 1800)
--fast                     Demo mode: idle=1s, active=10s
--verbose                  Show benign flows in console
--cooldown SEC             Alert rate limiting per IP (default: 30)
```

### Output and Logs

The sniffer generates structured logs in `logs/sniffer/`:

**`nids_sniffer_alerts.csv`** - Attack alerts:
```csv
timestamp,src_ip,src_port,dst_ip,dst_port,protocol,l7_proto,prediction,confidence,action,duration_ms,bytes_in,bytes_out,packets_in,packets_out
2026-02-17 14:23:45,192.168.1.100,54321,8.8.8.8,80,6,HTTP,attack,0.9234,alert_logged,150,512,2048,5,8
```

**`nids_sniffer_benign.csv`** - Benign sample (1 every 100 flows):
```csv
timestamp,src_ip,src_port,dst_ip,dst_port,protocol,l7_proto,prediction,confidence,action,duration_ms,bytes_in,bytes_out,packets_in,packets_out
2026-02-17 14:23:50,192.168.1.100,54322,1.1.1.1,443,6,HTTPS,benign,0.0234,logged,200,1024,4096,10,12
```

**`nids_sniffer_stats.jsonl`** - System statistics (every 60s):
```json
{"timestamp": "2026-02-17T14:24:00", "total_flows": 1234, "total_predictions": 1234, "total_alerts": 45, "total_blocks": 0, "alert_rate_pct": 3.65, "flows_per_second": 20.5, "memory_usage_mb": 245, "cpu_usage_percent": 12.3}
```

### Testing and Validation

#### 1. Component Tests

```bash
cd srcNF/live_sniffer
python test_sniffer.py
```

This runs integration tests for:
- Artifact loading
- Feature mapper initialization
- Preprocessor functionality
- Model predictor
- Logger components

#### 2. Offline Validation

Test the model on PCAP files, CSV, or Parquet data:

```bash
# Test on PCAP file
python test_on_file.py \
    --pcap ../../data/pcap/Tuesday-WorkingHours.pcap \
    --output ../../results/tuesday_results.csv \
    --stats

# Test on scaled Parquet (with ground truth comparison)
python test_on_file.py \
    --parquet ../../data/processed/test_scaled.parquet \
    --output ../../results/test_validation.csv \
    --compare

# Test on raw CSV
python test_on_file.py \
    --csv ../../data/raw/sample.csv \
    --output ../../results/sample_results.csv \
    --max 10000
```

#### 3. Live Testing

See comprehensive testing roadmaps in `srcNF/live_sniffer/GUIDES/`:

- **`NIDS_TEST_ROADMAP.md`**: Complete 3-phase validation (components, offline, live)
- **`loopback/DEMO_LOOPBACK_GUIDE.md`**: Testing on localhost
- **`linode/LINODE_NIDS_SETUP.md`**: Production deployment on remote server

## Dataset

**NF-UQ-NIDS-v2**: NetFlow-based intrusion detection dataset

- **Size**: 76,187,588 flows (train: 53.3M, val: 11.4M, test: 11.4M)
- **Features**: 43 original features (18 selected for model)
- **Classes**: Binary (benign / attack)
- **Attacks**: DDoS, DoS, Reconnaissance, Brute Force, Injection, XSS, Infiltration, Exploits, and more
- **Benign**: 25.2M flows (33%)
- **Malicious**: 51.0M flows (67%)

**Feature Selection** (18 features used):

Network layer:
- `PROTOCOL`: IP protocol number
- `L7_PROTO`: Application layer protocol ID

Volume metrics:
- `IN_BYTES`, `OUT_BYTES`: Byte counts per direction
- `IN_PKTS`, `OUT_PKTS`: Packet counts per direction

Timing:
- `DURATION_IN`, `DURATION_OUT`: Flow duration per direction (seconds)

Size statistics:
- `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`: Packet size range (bytes)

TCP-specific:
- `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`: TCP flag bitmasks

Throughput:
- `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`: Bytes per second

Bidirectional:
- `FLOW_DURATION_MILLISECONDS`: Total flow duration
- `MIN_TTL`, `MAX_TTL`: Time-to-live range

## Performance

### Training Pipeline

**Test Environment**: 
- CPU: Intel i7-9700K (8 cores)
- RAM: 16 GB
- Disk: NVMe SSD

**Results**:
- Preprocessing: 15-20 minutes
- Feature engineering: 5-10 minutes
- Training (XGBoost, 200 estimators): 20-30 minutes
- Total pipeline: 45-60 minutes

**Memory Usage**: Peak 8-10 GB during training

### Live Sniffer

**Test Environment**:
- CPU: Intel i5-8250U (4 cores)
- RAM: 8 GB
- Network: 1 Gbps Ethernet

**Results**:
- Throughput: 500-800 flows/second
- Latency: <2 seconds (with batch_size=100, batch_timeout=2.0)
- Memory: 200-300 MB steady state
- CPU: 10-20% average

**Scalability**:
- Handles 100 Mbps sustained traffic without dropped flows
- Suitable for small to medium network segments (100-500 devices)

## Limitations and Known Issues

### 1. Feature Mismatch (nProbe vs nfstream)

**Issue**: The model was trained on NF-UQ-NIDS-v2 dataset generated with nProbe/nfdump, but the live sniffer uses nfstream. These libraries calculate some features slightly differently.

**Impact**:
- Higher false positive rate on benign traffic (5-10% instead of 1.5%)
- Reduced detection rate on micro-flows (<5 packets)
- Port scans may be misclassified as benign (flow-level vs temporal aggregation)

**Mitigation**:
- Use `ATTACK_THRESHOLD=0.7` instead of 0.5 to reduce false positives
- For production deployment, consider retraining on data captured with nfstream
- Implement temporal aggregation layer for port scan detection

### 2. Loopback Interface Not Supported

**Issue**: Testing on loopback interface (127.0.0.1) produces unreliable results due to unrealistic network characteristics (zero latency, 65K MTU, unlimited throughput).

**Impact**:
- Port scans appear 99% benign (confidence 0.999)
- SYN floods not detected
- Only HTTP flood detection works reliably

**Solution**: Always test on real network interfaces (eth0, wlan0) or remote targets. See `GUIDES/linode/LINODE_NIDS_SETUP.md` for production deployment.

### 3. BLOCK Mode Requires Careful Configuration

**Issue**: Firewall integration with iptables can block legitimate traffic if misconfigured.

**Requirements**:
- Add management IPs to `WHITELIST_IPS` in config.py
- Test thoroughly in ALERT mode before enabling BLOCK mode
- Ensure iptables is installed and accessible

### 4. Model Limitations

**Not Detected Well**:
- Application-layer attacks (SQL injection, XSS) - insufficient feature representation
- Low-and-slow attacks (slowloris) - temporal aggregation needed
- Encrypted payload attacks - no deep packet inspection

**Well Detected**:
- Volumetric attacks (DDoS, DoS) - 95%+ detection rate
- Port scans (with temporal aggregation) - 80%+ detection rate
- Brute force attempts - 90%+ detection rate

## Deployment

### Recommended Production Setup

1. **Server-side deployment**: Install sniffer on gateway/firewall server
2. **Mirror port**: Configure switch to mirror traffic to sniffer interface
3. **ALERT mode initially**: Run in monitoring mode for 1-2 weeks
4. **Baseline establishment**: Analyze false positive rate on production traffic
5. **Threshold tuning**: Adjust `ATTACK_THRESHOLD` based on observed FPR
6. **BLOCK mode (optional)**: Enable firewall integration after validation

### Systemd Service Example

Create `/etc/systemd/system/nids-sniffer.service`:

```ini
[Unit]
Description=NIDS Live Sniffer
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/nids/srcNF/live_sniffer
Environment="PATH=/opt/nids/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/opt/nids/venv/bin/python main.py --mode alert --interface eth0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nids-sniffer
sudo systemctl start nids-sniffer
sudo systemctl status nids-sniffer
```

### Monitoring

**Real-time logs**:
```bash
tail -f /opt/nids/logs/sniffer/nids_sniffer.log
```

**Alert monitoring**:
```bash
tail -f /opt/nids/logs/sniffer/nids_sniffer_alerts.csv | \
    awk -F',' 'NR>1 {print $1, $2, $3, $4, $9}'
```

**Statistics dashboard** (requires jq):
```bash
tail -1 /opt/nids/logs/sniffer/nids_sniffer_stats.jsonl | jq .
```

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

**Areas for Improvement**:
- Temporal aggregation for port scan detection
- Support for additional ML models (LSTM, Transformer)
- Dashboard/visualization interface
- Multi-class classification (attack type identification)
- Integration with SIEM systems

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- NF-UQ-NIDS-v2 dataset creators
- nfstream library developers
- XGBoost project contributors

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your email or contact info]

---

**Last Updated**: February 2026  
**Version**: 2.0  
**Status**: Production-ready with known limitations
