"""
NIDS-ML Sniffer - Flow Aggregation Module

Aggrega pacchetti in flussi bidirezionali (5-tupla).

FIXES:
- Migliorata gestione timeout con expire_flows
- Aggiunto tracking eta flusso per garbage collection
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_FLOW_TIMEOUT = 60.0
DEFAULT_MAX_PACKETS = 500
IDLE_THRESHOLD = 1.0


@dataclass
class Flow:
    """Flusso di rete bidirezionale."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    
    start_time: float = field(default_factory=time.time)
    last_time: float = field(default_factory=time.time)
    
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    
    fwd_lengths: List[int] = field(default_factory=list)
    bwd_lengths: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    fwd_timestamps: List[float] = field(default_factory=list)
    bwd_timestamps: List[float] = field(default_factory=list)
    
    fwd_header_length: int = 0
    bwd_header_length: int = 0
    
    fwd_psh_flags: int = 0
    bwd_psh_flags: int = 0
    fwd_urg_flags: int = 0
    bwd_urg_flags: int = 0
    
    fin_flag_count: int = 0
    syn_flag_count: int = 0
    rst_flag_count: int = 0
    psh_flag_count: int = 0
    ack_flag_count: int = 0
    urg_flag_count: int = 0
    cwe_flag_count: int = 0
    ece_flag_count: int = 0
    
    init_win_bytes_forward: int = 0
    init_win_bytes_backward: int = 0
    _fwd_win_set: bool = field(default=False, repr=False)
    _bwd_win_set: bool = field(default=False, repr=False)
    
    act_data_pkt_fwd: int = 0
    active_times: List[float] = field(default_factory=list)
    idle_times: List[float] = field(default_factory=list)
    _last_active_start: Optional[float] = field(default=None, repr=False)
    _last_packet_time: Optional[float] = field(default=None, repr=False)
    
    @property
    def flow_key(self) -> Tuple:
        if (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port):
            return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
        return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)
    
    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets
    
    @property
    def total_bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes
    
    @property
    def duration(self) -> float:
        return max(0.0, self.last_time - self.start_time)
    
    @property
    def age(self) -> float:
        """Eta del flusso dall'ultimo pacchetto."""
        return time.time() - self.last_time
    
    @property
    def iats(self) -> List[float]:
        if len(self.timestamps) < 2:
            return []
        return [
            self.timestamps[i] - self.timestamps[i-1] 
            for i in range(1, len(self.timestamps))
        ]
    
    @property
    def fwd_iats(self) -> List[float]:
        if len(self.fwd_timestamps) < 2:
            return []
        return [
            self.fwd_timestamps[i] - self.fwd_timestamps[i-1] 
            for i in range(1, len(self.fwd_timestamps))
        ]
    
    @property
    def bwd_iats(self) -> List[float]:
        if len(self.bwd_timestamps) < 2:
            return []
        return [
            self.bwd_timestamps[i] - self.bwd_timestamps[i-1] 
            for i in range(1, len(self.bwd_timestamps))
        ]
    
    def add_packet(
        self, 
        timestamp: float, 
        is_forward: bool, 
        payload_size: int, 
        header_length: int, 
        tcp_flags: Optional[Dict[str, bool]] = None, 
        window_size: int = 0
    ):
        """Aggiunge un pacchetto al flusso."""
        if self._last_packet_time is not None:
            gap = timestamp - self._last_packet_time
            if gap > IDLE_THRESHOLD:
                self.idle_times.append(gap)
                if self._last_active_start is not None:
                    active_duration = self._last_packet_time - self._last_active_start
                    if active_duration > 0:
                        self.active_times.append(active_duration)
                self._last_active_start = timestamp
            elif self._last_active_start is None:
                self._last_active_start = timestamp
        else:
            self._last_active_start = timestamp
        
        self._last_packet_time = timestamp
        self.last_time = timestamp
        self.timestamps.append(timestamp)
        packet_size = header_length + payload_size
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_size
            self.fwd_lengths.append(packet_size)
            self.fwd_timestamps.append(timestamp)
            self.fwd_header_length += header_length
            if not self._fwd_win_set and window_size > 0:
                self.init_win_bytes_forward = window_size
                self._fwd_win_set = True
            if payload_size > 0:
                self.act_data_pkt_fwd += 1
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_size
            self.bwd_lengths.append(packet_size)
            self.bwd_timestamps.append(timestamp)
            self.bwd_header_length += header_length
            if not self._bwd_win_set and window_size > 0:
                self.init_win_bytes_backward = window_size
                self._bwd_win_set = True
        
        if tcp_flags:
            self._process_flags(tcp_flags, is_forward)
    
    def _process_flags(self, flags: Dict[str, bool], is_forward: bool):
        if flags.get('FIN'): 
            self.fin_flag_count += 1
        if flags.get('SYN'): 
            self.syn_flag_count += 1
        if flags.get('RST'): 
            self.rst_flag_count += 1
        if flags.get('PSH'):
            self.psh_flag_count += 1
            if is_forward: 
                self.fwd_psh_flags += 1
            else: 
                self.bwd_psh_flags += 1
        if flags.get('ACK'): 
            self.ack_flag_count += 1
        if flags.get('URG'):
            self.urg_flag_count += 1
            if is_forward: 
                self.fwd_urg_flags += 1
            else: 
                self.bwd_urg_flags += 1
        if flags.get('ECE'): 
            self.ece_flag_count += 1
        if flags.get('CWR'): 
            self.cwe_flag_count += 1
    
    def is_complete(self) -> bool:
        """Verifica se il flusso e' completo (FIN o RST ricevuto)."""
        return self.fin_flag_count > 0 or self.rst_flag_count > 0


@dataclass
class PacketInfo:
    """Info estratte da un pacchetto."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    payload_size: int
    header_length: int
    tcp_flags: Dict[str, bool] = field(default_factory=dict)
    window_size: int = 0


class FlowManager:
    """
    Gestisce flussi attivi.
    
    FIX: Migliorata gestione memoria con expire_flows() per garbage collection
    proattivo durante cattura live prolungata.
    """
    
    def __init__(
        self, 
        flow_timeout: float = DEFAULT_FLOW_TIMEOUT, 
        max_packets: int = DEFAULT_MAX_PACKETS
    ):
        self.flow_timeout = flow_timeout
        self.max_packets = max_packets
        self._flows: Dict[Tuple, Flow] = {}
        self.flows_created = 0
        self.flows_completed = 0
        self.flows_expired = 0
    
    def get_flow_count(self) -> int:
        """Numero flussi attivi."""
        return len(self._flows)
    
    def add_packet(self, pkt: PacketInfo) -> Optional[Flow]:
        """
        Aggiunge pacchetto. Restituisce Flow se pronto per analisi.
        
        Un flusso viene restituito (e rimosso) quando:
        - Raggiunge max_packets
        - Riceve flag FIN o RST
        """
        key = self._make_key(
            pkt.src_ip, pkt.dst_ip, 
            pkt.src_port, pkt.dst_port, 
            pkt.protocol
        )
        
        if key not in self._flows:
            self._flows[key] = Flow(
                src_ip=pkt.src_ip, 
                dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, 
                dst_port=pkt.dst_port,
                protocol=pkt.protocol,
                start_time=pkt.timestamp, 
                last_time=pkt.timestamp
            )
            self.flows_created += 1
        
        flow = self._flows[key]
        is_forward = (pkt.src_ip == flow.src_ip and pkt.src_port == flow.src_port)
        
        flow.add_packet(
            timestamp=pkt.timestamp, 
            is_forward=is_forward,
            payload_size=pkt.payload_size, 
            header_length=pkt.header_length,
            tcp_flags=pkt.tcp_flags, 
            window_size=pkt.window_size
        )
        
        if flow.total_packets >= self.max_packets or flow.is_complete():
            del self._flows[key]
            self.flows_completed += 1
            return flow
        
        return None
    
    def expire_flows(self, current_time: float) -> List[Flow]:
        """
        Rimuove e restituisce flussi scaduti (inattivi per piu di flow_timeout).
        
        Chiamare periodicamente per prevenire memory leak durante:
        - Cattura live prolungata
        - Attacchi con IP spoofing
        - Connessioni half-open (slowloris)
        
        Args:
            current_time: Timestamp corrente (dal pacchetto o time.time())
        
        Returns:
            Lista di flussi scaduti da analizzare
        """
        expired = []
        to_remove = []
        
        for key, flow in self._flows.items():
            if (current_time - flow.last_time) > self.flow_timeout:
                expired.append(flow)
                to_remove.append(key)
        
        for key in to_remove:
            del self._flows[key]
            self.flows_expired += 1
        
        return expired
    
    def get_all_flows(self) -> List[Flow]:
        """Restituisce tutti i flussi e svuota il manager."""
        flows = list(self._flows.values())
        self._flows.clear()
        return flows
    
    def get_stats(self) -> Dict:
        """Statistiche del manager."""
        return {
            'active_flows': len(self._flows),
            'flows_created': self.flows_created,
            'flows_completed': self.flows_completed,
            'flows_expired': self.flows_expired
        }
    
    @staticmethod
    def _make_key(src_ip, dst_ip, src_port, dst_port, protocol) -> Tuple:
        """Crea chiave bidirezionale per il flusso."""
        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        return (dst_ip, src_ip, dst_port, src_port, protocol)