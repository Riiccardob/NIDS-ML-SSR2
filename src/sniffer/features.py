"""
NIDS-ML Sniffer - Feature Extraction (77 feature CIC-IDS2017)

Unità di misura CIC-IDS2017:
- Durata: MICROSECONDI
- IAT: MICROSECONDI  
- Bytes: BYTES
- Rates: BYTES/SECONDO o PACKETS/SECONDO
"""

import numpy as np
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .flow import Flow


FEATURE_NAMES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]


class FeatureExtractor:
    """Estrae le 77 feature CIC-IDS2017 da un Flow."""
    
    @staticmethod
    def _mean(vals): return float(np.mean(vals)) if vals else 0.0
    @staticmethod
    def _std(vals): return float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0
    @staticmethod
    def _min(vals): return float(min(vals)) if vals else 0.0
    @staticmethod
    def _max(vals): return float(max(vals)) if vals else 0.0
    @staticmethod
    def _sum(vals): return float(sum(vals)) if vals else 0.0
    @staticmethod
    def _var(vals): return float(np.var(vals, ddof=0)) if vals else 0.0
    
    def extract(self, flow: 'Flow') -> Dict[str, float]:
        """Estrae tutte le 77 feature."""
        f = {}
        
        duration_sec = flow.duration
        duration_us = duration_sec * 1_000_000
        f['Flow Duration'] = duration_us
        
        f['Total Fwd Packets'] = float(flow.fwd_packets)
        f['Total Backward Packets'] = float(flow.bwd_packets)
        f['Total Length of Fwd Packets'] = float(flow.fwd_bytes)
        f['Total Length of Bwd Packets'] = float(flow.bwd_bytes)
        
        f['Fwd Packet Length Max'] = self._max(flow.fwd_lengths)
        f['Fwd Packet Length Min'] = self._min(flow.fwd_lengths)
        f['Fwd Packet Length Mean'] = self._mean(flow.fwd_lengths)
        f['Fwd Packet Length Std'] = self._std(flow.fwd_lengths)
        
        f['Bwd Packet Length Max'] = self._max(flow.bwd_lengths)
        f['Bwd Packet Length Min'] = self._min(flow.bwd_lengths)
        f['Bwd Packet Length Mean'] = self._mean(flow.bwd_lengths)
        f['Bwd Packet Length Std'] = self._std(flow.bwd_lengths)
        
        if duration_sec > 0:
            f['Flow Bytes/s'] = flow.total_bytes / duration_sec
            f['Flow Packets/s'] = flow.total_packets / duration_sec
            f['Fwd Packets/s'] = flow.fwd_packets / duration_sec
            f['Bwd Packets/s'] = flow.bwd_packets / duration_sec
        else:
            f['Flow Bytes/s'] = f['Flow Packets/s'] = f['Fwd Packets/s'] = f['Bwd Packets/s'] = 0.0
        
        # IAT in microsecondi
        all_iats = [iat * 1e6 for iat in flow.iats]
        f['Flow IAT Mean'] = self._mean(all_iats)
        f['Flow IAT Std'] = self._std(all_iats)
        f['Flow IAT Max'] = self._max(all_iats)
        f['Flow IAT Min'] = self._min(all_iats)
        
        fwd_iats = [iat * 1e6 for iat in flow.fwd_iats]
        f['Fwd IAT Total'] = self._sum(fwd_iats)
        f['Fwd IAT Mean'] = self._mean(fwd_iats)
        f['Fwd IAT Std'] = self._std(fwd_iats)
        f['Fwd IAT Max'] = self._max(fwd_iats)
        f['Fwd IAT Min'] = self._min(fwd_iats)
        
        bwd_iats = [iat * 1e6 for iat in flow.bwd_iats]
        f['Bwd IAT Total'] = self._sum(bwd_iats)
        f['Bwd IAT Mean'] = self._mean(bwd_iats)
        f['Bwd IAT Std'] = self._std(bwd_iats)
        f['Bwd IAT Max'] = self._max(bwd_iats)
        f['Bwd IAT Min'] = self._min(bwd_iats)
        
        f['Fwd PSH Flags'] = float(flow.fwd_psh_flags)
        f['Bwd PSH Flags'] = float(flow.bwd_psh_flags)
        f['Fwd URG Flags'] = float(flow.fwd_urg_flags)
        f['Bwd URG Flags'] = float(flow.bwd_urg_flags)
        
        f['Fwd Header Length'] = float(flow.fwd_header_length)
        f['Bwd Header Length'] = float(flow.bwd_header_length)
        
        all_lengths = flow.fwd_lengths + flow.bwd_lengths
        f['Min Packet Length'] = self._min(all_lengths)
        f['Max Packet Length'] = self._max(all_lengths)
        f['Packet Length Mean'] = self._mean(all_lengths)
        f['Packet Length Std'] = self._std(all_lengths)
        f['Packet Length Variance'] = self._var(all_lengths)
        
        f['FIN Flag Count'] = float(flow.fin_flag_count)
        f['SYN Flag Count'] = float(flow.syn_flag_count)
        f['RST Flag Count'] = float(flow.rst_flag_count)
        f['PSH Flag Count'] = float(flow.psh_flag_count)
        f['ACK Flag Count'] = float(flow.ack_flag_count)
        f['URG Flag Count'] = float(flow.urg_flag_count)
        f['CWE Flag Count'] = float(flow.cwe_flag_count)
        f['ECE Flag Count'] = float(flow.ece_flag_count)
        
        f['Down/Up Ratio'] = flow.bwd_packets / flow.fwd_packets if flow.fwd_packets > 0 else 0.0
        f['Average Packet Size'] = flow.total_bytes / flow.total_packets if flow.total_packets > 0 else 0.0
        f['Avg Fwd Segment Size'] = f['Fwd Packet Length Mean']
        f['Avg Bwd Segment Size'] = f['Bwd Packet Length Mean']
        f['Fwd Header Length.1'] = f['Fwd Header Length']
        
        # Bulk features (tipicamente 0)
        for key in ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']:
            f[key] = 0.0
        
        f['Subflow Fwd Packets'] = float(flow.fwd_packets)
        f['Subflow Fwd Bytes'] = float(flow.fwd_bytes)
        f['Subflow Bwd Packets'] = float(flow.bwd_packets)
        f['Subflow Bwd Bytes'] = float(flow.bwd_bytes)
        
        f['Init_Win_bytes_forward'] = float(flow.init_win_bytes_forward)
        f['Init_Win_bytes_backward'] = float(flow.init_win_bytes_backward)
        f['act_data_pkt_fwd'] = float(flow.act_data_pkt_fwd)
        f['min_seg_size_forward'] = self._min(flow.fwd_lengths) if flow.fwd_lengths else 0.0
        
        active_us = [t * 1e6 for t in flow.active_times]
        idle_us = [t * 1e6 for t in flow.idle_times]
        f['Active Mean'] = self._mean(active_us)
        f['Active Std'] = self._std(active_us)
        f['Active Max'] = self._max(active_us)
        f['Active Min'] = self._min(active_us)
        f['Idle Mean'] = self._mean(idle_us)
        f['Idle Std'] = self._std(idle_us)
        f['Idle Max'] = self._max(idle_us)
        f['Idle Min'] = self._min(idle_us)
        
        return f