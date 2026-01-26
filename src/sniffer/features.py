"""
================================================================================
NIDS-ML - Feature Definitions & Extraction
================================================================================

Definizione centralizzata delle 77 feature CIC-IDS2017.
Questo modulo garantisce consistenza tra training e inference.

IMPORTANTE:
-----------
Le feature DEVONO essere estratte esattamente nello stesso modo
in cui CICFlowMeter le estrae, altrimenti il modello non funzionerà.

Unità di misura CIC-IDS2017:
- Durata: MICROSECONDI
- IAT (Inter-Arrival Time): MICROSECONDI
- Bytes: BYTES
- Rates: BYTES/SECONDO o PACKETS/SECONDO

================================================================================
"""

from typing import Dict, List, Optional
import numpy as np


# ==============================================================================
# FEATURE NAMES - Ordine esatto del dataset CIC-IDS2017
# ==============================================================================

FEATURE_NAMES = [
    # Flow identifiers (da rimuovere prima del training)
    # 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 
    # 'Protocol', 'Timestamp',
    
    # Flow duration
    'Flow Duration',
    
    # Packet counts
    'Total Fwd Packets',
    'Total Backward Packets',
    
    # Byte counts
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    
    # Forward packet length statistics
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    
    # Backward packet length statistics
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    
    # Flow rates
    'Flow Bytes/s',
    'Flow Packets/s',
    
    # Flow IAT (Inter-Arrival Time)
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    
    # Forward IAT
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    
    # Backward IAT
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    
    # PSH Flags
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    
    # URG Flags
    'Fwd URG Flags',
    'Bwd URG Flags',
    
    # Header lengths
    'Fwd Header Length',
    'Bwd Header Length',
    
    # Packet rates
    'Fwd Packets/s',
    'Bwd Packets/s',
    
    # Packet length statistics (combined)
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    
    # Flags
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    
    # Down/Up Ratio
    'Down/Up Ratio',
    
    # Average packet/segment size
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    
    # Duplicate header (CIC-IDS2017 quirk)
    'Fwd Header Length.1',
    
    # Bulk features
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    
    # Subflow features
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    
    # Init window
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    
    # Active data packets
    'act_data_pkt_fwd',
    
    # Min segment size
    'min_seg_size_forward',
    
    # Active/Idle statistics
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min',
    
    # Label (da gestire separatamente)
    # 'Label'
]

# Feature critiche per la detection (top 30 da feature importance)
CRITICAL_FEATURES = [
    'Bwd Packet Length Std',
    'Bwd Packet Length Max',
    'Avg Bwd Segment Size',
    'Packet Length Variance',
    'Bwd Packet Length Mean',
    'Packet Length Std',
    'Subflow Bwd Bytes',
    'Total Fwd Packets',
    'Average Packet Size',
    'Packet Length Mean',
    'Max Packet Length',
    'Subflow Fwd Packets',
    'Fwd Header Length.1',
    'Total Length of Bwd Packets',
    'Total Backward Packets',
    'Fwd Packet Length Max',
    'Bwd Packets/s',
    'Subflow Bwd Packets',
    'Subflow Fwd Bytes',
    'Fwd Header Length',
    'Init_Win_bytes_forward',
    'Bwd Packet Length Min',
    'Bwd Header Length',
    'Init_Win_bytes_backward',
    'Avg Fwd Segment Size',
    'Total Length of Fwd Packets',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd Packet Length Mean',
    'Fwd IAT Max',
]


# ==============================================================================
# FEATURE EXTRACTOR
# ==============================================================================

class FeatureExtractor:
    """
    Estrae feature CIC-IDS2017 compatibili da un flusso di rete.
    
    Questa classe centralizza la logica di estrazione per garantire
    che lo sniffer produca feature identiche a quelle del training.
    """
    
    def __init__(self, validate: bool = True):
        """
        Args:
            validate: Se True, valida che tutte le feature siano presenti
        """
        self.validate = validate
        self._warnings = []
    
    @staticmethod
    def safe_mean(values: List[float]) -> float:
        """Calcola media in modo sicuro."""
        if not values:
            return 0.0
        return float(np.mean(values))
    
    @staticmethod
    def safe_std(values: List[float]) -> float:
        """Calcola deviazione standard in modo sicuro (ddof=0 come CICFlowMeter)."""
        if not values or len(values) < 1:
            return 0.0
        return float(np.std(values, ddof=0))
    
    @staticmethod
    def safe_max(values: List[float]) -> float:
        """Calcola massimo in modo sicuro."""
        if not values:
            return 0.0
        return float(max(values))
    
    @staticmethod
    def safe_min(values: List[float]) -> float:
        """Calcola minimo in modo sicuro."""
        if not values:
            return 0.0
        return float(min(values))
    
    @staticmethod
    def safe_sum(values: List[float]) -> float:
        """Calcola somma in modo sicuro."""
        if not values:
            return 0.0
        return float(sum(values))
    
    @staticmethod
    def safe_var(values: List[float]) -> float:
        """Calcola varianza in modo sicuro (ddof=0)."""
        if not values or len(values) < 1:
            return 0.0
        return float(np.var(values, ddof=0))
    
    def extract(self, flow: 'Flow') -> Dict[str, float]:
        """
        Estrae tutte le feature da un oggetto Flow.
        
        Args:
            flow: Oggetto Flow con dati aggregati
        
        Returns:
            Dict con tutte le 77 feature CIC-IDS2017
        """
        features = {}
        
        # Duration in MICROSECONDI
        duration_sec = flow.duration
        duration_us = duration_sec * 1e6
        features['Flow Duration'] = duration_us
        
        # Packet counts
        features['Total Fwd Packets'] = flow.fwd_packets
        features['Total Backward Packets'] = flow.bwd_packets
        
        # Byte counts
        features['Total Length of Fwd Packets'] = flow.fwd_bytes
        features['Total Length of Bwd Packets'] = flow.bwd_bytes
        
        # Forward packet length stats
        features['Fwd Packet Length Max'] = self.safe_max(flow.fwd_lengths)
        features['Fwd Packet Length Min'] = self.safe_min(flow.fwd_lengths)
        features['Fwd Packet Length Mean'] = self.safe_mean(flow.fwd_lengths)
        features['Fwd Packet Length Std'] = self.safe_std(flow.fwd_lengths)
        
        # Backward packet length stats
        features['Bwd Packet Length Max'] = self.safe_max(flow.bwd_lengths)
        features['Bwd Packet Length Min'] = self.safe_min(flow.bwd_lengths)
        features['Bwd Packet Length Mean'] = self.safe_mean(flow.bwd_lengths)
        features['Bwd Packet Length Std'] = self.safe_std(flow.bwd_lengths)
        
        # Flow rates (bytes/sec, packets/sec)
        if duration_sec > 0:
            features['Flow Bytes/s'] = flow.total_bytes / duration_sec
            features['Flow Packets/s'] = flow.total_packets / duration_sec
            features['Fwd Packets/s'] = flow.fwd_packets / duration_sec
            features['Bwd Packets/s'] = flow.bwd_packets / duration_sec
        else:
            features['Flow Bytes/s'] = 0.0
            features['Flow Packets/s'] = 0.0
            features['Fwd Packets/s'] = 0.0
            features['Bwd Packets/s'] = 0.0
        
        # IAT in MICROSECONDI
        all_iats = flow.fwd_iats + flow.bwd_iats
        
        # Flow IAT
        features['Flow IAT Mean'] = self.safe_mean(all_iats) * 1e6
        features['Flow IAT Std'] = self.safe_std(all_iats) * 1e6
        features['Flow IAT Max'] = self.safe_max(all_iats) * 1e6
        features['Flow IAT Min'] = self.safe_min(all_iats) * 1e6
        
        # Forward IAT
        features['Fwd IAT Total'] = self.safe_sum(flow.fwd_iats) * 1e6
        features['Fwd IAT Mean'] = self.safe_mean(flow.fwd_iats) * 1e6
        features['Fwd IAT Std'] = self.safe_std(flow.fwd_iats) * 1e6
        features['Fwd IAT Max'] = self.safe_max(flow.fwd_iats) * 1e6
        features['Fwd IAT Min'] = self.safe_min(flow.fwd_iats) * 1e6
        
        # Backward IAT
        features['Bwd IAT Total'] = self.safe_sum(flow.bwd_iats) * 1e6
        features['Bwd IAT Mean'] = self.safe_mean(flow.bwd_iats) * 1e6
        features['Bwd IAT Std'] = self.safe_std(flow.bwd_iats) * 1e6
        features['Bwd IAT Max'] = self.safe_max(flow.bwd_iats) * 1e6
        features['Bwd IAT Min'] = self.safe_min(flow.bwd_iats) * 1e6
        
        # PSH/URG Flags per direzione
        features['Fwd PSH Flags'] = flow.fwd_psh_flags
        features['Bwd PSH Flags'] = flow.bwd_psh_flags
        features['Fwd URG Flags'] = flow.fwd_urg_flags
        features['Bwd URG Flags'] = flow.bwd_urg_flags
        
        # Header lengths
        # NOTA: CICFlowMeter somma tutti gli header length dei pacchetti
        features['Fwd Header Length'] = flow.fwd_header_bytes
        features['Bwd Header Length'] = flow.bwd_header_bytes
        features['Fwd Header Length.1'] = flow.fwd_header_bytes  # Duplicato in CIC-IDS2017
        
        # Packet length stats (combined)
        all_lengths = flow.fwd_lengths + flow.bwd_lengths
        features['Min Packet Length'] = self.safe_min(all_lengths)
        features['Max Packet Length'] = self.safe_max(all_lengths)
        features['Packet Length Mean'] = self.safe_mean(all_lengths)
        features['Packet Length Std'] = self.safe_std(all_lengths)
        features['Packet Length Variance'] = self.safe_var(all_lengths)
        
        # TCP Flags counts
        features['FIN Flag Count'] = flow.fin_count
        features['SYN Flag Count'] = flow.syn_count
        features['RST Flag Count'] = flow.rst_count
        features['PSH Flag Count'] = flow.psh_count
        features['ACK Flag Count'] = flow.ack_count
        features['URG Flag Count'] = flow.urg_count
        features['CWE Flag Count'] = flow.cwe_count
        features['ECE Flag Count'] = flow.ece_count
        
        # Down/Up Ratio
        if flow.fwd_packets > 0:
            features['Down/Up Ratio'] = flow.bwd_packets / flow.fwd_packets
        else:
            features['Down/Up Ratio'] = 0.0
        
        # Average sizes
        if flow.total_packets > 0:
            features['Average Packet Size'] = flow.total_bytes / flow.total_packets
        else:
            features['Average Packet Size'] = 0.0
        
        features['Avg Fwd Segment Size'] = features['Fwd Packet Length Mean']
        features['Avg Bwd Segment Size'] = features['Bwd Packet Length Mean']
        
        # Bulk features (placeholder - difficili da calcolare senza info bulk)
        features['Fwd Avg Bytes/Bulk'] = 0.0
        features['Fwd Avg Packets/Bulk'] = 0.0
        features['Fwd Avg Bulk Rate'] = 0.0
        features['Bwd Avg Bytes/Bulk'] = 0.0
        features['Bwd Avg Packets/Bulk'] = 0.0
        features['Bwd Avg Bulk Rate'] = 0.0
        
        # Subflow (uguale ai totali per singolo subflow)
        features['Subflow Fwd Packets'] = flow.fwd_packets
        features['Subflow Fwd Bytes'] = flow.fwd_bytes
        features['Subflow Bwd Packets'] = flow.bwd_packets
        features['Subflow Bwd Bytes'] = flow.bwd_bytes
        
        # Init window
        features['Init_Win_bytes_forward'] = flow.init_win_fwd if flow.init_win_fwd else 65535
        features['Init_Win_bytes_backward'] = flow.init_win_bwd if flow.init_win_bwd else 65535
        
        # Active data packets
        features['act_data_pkt_fwd'] = flow.act_data_pkt_fwd
        
        # Min segment size
        features['min_seg_size_forward'] = self.safe_min(flow.fwd_lengths) if flow.fwd_lengths else 0
        
        # Active/Idle statistics
        # NOTA: Questi richiedono tracking di periodi attivi/idle
        # Per ora placeholder, ma potrebbero essere calcolati
        features['Active Mean'] = self.safe_mean(flow.active_times) * 1e6 if flow.active_times else 0.0
        features['Active Std'] = self.safe_std(flow.active_times) * 1e6 if flow.active_times else 0.0
        features['Active Max'] = self.safe_max(flow.active_times) * 1e6 if flow.active_times else 0.0
        features['Active Min'] = self.safe_min(flow.active_times) * 1e6 if flow.active_times else 0.0
        features['Idle Mean'] = self.safe_mean(flow.idle_times) * 1e6 if flow.idle_times else 0.0
        features['Idle Std'] = self.safe_std(flow.idle_times) * 1e6 if flow.idle_times else 0.0
        features['Idle Max'] = self.safe_max(flow.idle_times) * 1e6 if flow.idle_times else 0.0
        features['Idle Min'] = self.safe_min(flow.idle_times) * 1e6 if flow.idle_times else 0.0
        
        # Validazione
        if self.validate:
            self._validate_features(features)
        
        return features
    
    def _validate_features(self, features: Dict[str, float]) -> None:
        """Valida che le feature siano ragionevoli."""
        for name, value in features.items():
            # Check per NaN o Inf
            if np.isnan(value) or np.isinf(value):
                self._warnings.append(f"Feature '{name}' ha valore invalido: {value}")
                features[name] = 0.0
            
            # Check per valori negativi dove non dovrebbero esserci
            if value < 0 and 'Ratio' not in name:
                self._warnings.append(f"Feature '{name}' ha valore negativo: {value}")
    
    @property
    def warnings(self) -> List[str]:
        """Restituisce warning accumulati."""
        return self._warnings
    
    def clear_warnings(self) -> None:
        """Pulisce i warning."""
        self._warnings = []


# ==============================================================================
# UTILITY
# ==============================================================================

def get_feature_columns_ordered() -> List[str]:
    """
    Restituisce i nomi delle feature nell'ordine corretto.
    
    Questo ordine DEVE corrispondere all'ordine usato durante il training.
    """
    return FEATURE_NAMES.copy()


def validate_feature_dict(features: Dict[str, float], 
                          required_features: List[str] = None) -> List[str]:
    """
    Valida un dizionario di feature.
    
    Args:
        features: Dizionario feature name -> value
        required_features: Lista feature richieste (default: FEATURE_NAMES)
    
    Returns:
        Lista di feature mancanti
    """
    if required_features is None:
        required_features = FEATURE_NAMES
    
    missing = []
    for feat in required_features:
        if feat not in features:
            missing.append(feat)
    
    return missing
