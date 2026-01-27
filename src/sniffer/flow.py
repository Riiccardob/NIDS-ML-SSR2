"""
================================================================================
NIDS-ML - Flow Class
================================================================================

Rappresentazione di un flusso di rete bidirezionale.

Un flusso aggrega tutti i pacchetti tra una coppia (src_ip:src_port, dst_ip:dst_port)
e calcola statistiche per la classificazione ML.

MIGLIORAMENTI rispetto alla versione precedente:
- Tracking separato PSH/URG flags per direzione
- Calcolo reale header bytes invece di approssimazione
- Tracking Active/Idle times
- Tracking act_data_pkt_fwd (pacchetti con payload > 0)
- CWE/ECE flags

================================================================================
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ==============================================================================
# COSTANTI
# ==============================================================================

# Soglia idle in secondi (CICFlowMeter usa 1 secondo)
IDLE_THRESHOLD = 1.0

# Timeout default per flussi (idle timeout)
DEFAULT_FLOW_TIMEOUT = 60.0

# Active timeout: forza analisi dopo N secondi di attività continua
# Importante per attacchi come Slowloris o port scan
ACTIVE_TIMEOUT = 30.0

# Max pacchetti prima di forzare analisi
MAX_PACKETS_PER_FLOW = 500


# ==============================================================================
# FLOW CLASS
# ==============================================================================

@dataclass
class Flow:
    """
    Rappresenta un flusso di rete bidirezionale.
    
    Raccoglie statistiche sui pacchetti e supporta l'estrazione
    di feature CIC-IDS2017 compatibili.
    
    Attributes:
        src_ip: IP sorgente (direzione forward)
        dst_ip: IP destinazione
        src_port: Porta sorgente
        dst_port: Porta destinazione
        protocol: Protocollo IP (6=TCP, 17=UDP, etc.)
    """
    
    # Identificatori flusso
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    
    # Timing
    start_time: float = field(default_factory=time.time)
    last_time: float = field(default_factory=time.time)
    
    # Contatori pacchetti
    fwd_packets: int = 0
    bwd_packets: int = 0
    
    # Contatori bytes
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    
    # Contatori header bytes (reali, non approssimati)
    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0
    
    # Liste per statistiche lunghezza
    fwd_lengths: List[int] = field(default_factory=list)
    bwd_lengths: List[int] = field(default_factory=list)
    
    # Liste per IAT (Inter-Arrival Time) in SECONDI
    fwd_iats: List[float] = field(default_factory=list)
    bwd_iats: List[float] = field(default_factory=list)
    
    # Timestamp ultimo pacchetto per direzione (per IAT)
    last_fwd_time: Optional[float] = None
    last_bwd_time: Optional[float] = None
    
    # TCP Flags totali
    fin_count: int = 0
    syn_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0
    cwe_count: int = 0
    ece_count: int = 0
    
    # PSH/URG per direzione
    fwd_psh_flags: int = 0
    bwd_psh_flags: int = 0
    fwd_urg_flags: int = 0
    bwd_urg_flags: int = 0
    
    # Window size iniziali
    init_win_fwd: Optional[int] = None
    init_win_bwd: Optional[int] = None
    
    # Active data packets (pacchetti con payload > 0)
    act_data_pkt_fwd: int = 0
    
    # Active/Idle tracking
    active_times: List[float] = field(default_factory=list)
    idle_times: List[float] = field(default_factory=list)
    _last_active_start: Optional[float] = None
    _in_idle_period: bool = False
    
    def __post_init__(self):
        """Inizializzazione post-creazione."""
        self.last_time = self.start_time
        self._last_active_start = self.start_time
    
    @property
    def flow_id(self) -> str:
        """Identificatore unico del flusso."""
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}:{self.protocol}"
    
    @property
    def flow_key(self) -> str:
        """Chiave normalizzata per lookup bidirezionale."""
        if (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port):
            return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}-{self.protocol}"
        return f"{self.dst_ip}:{self.dst_port}-{self.src_ip}:{self.src_port}-{self.protocol}"
    
    @property
    def duration(self) -> float:
        """Durata del flusso in secondi."""
        return max(self.last_time - self.start_time, 0.000001)
    
    @property
    def total_packets(self) -> int:
        """Numero totale di pacchetti."""
        return self.fwd_packets + self.bwd_packets
    
    @property
    def total_bytes(self) -> int:
        """Numero totale di bytes."""
        return self.fwd_bytes + self.bwd_bytes
    
    def add_packet(self, 
                   packet_len: int,
                   header_len: int,
                   payload_len: int,
                   is_forward: bool,
                   timestamp: float,
                   tcp_flags: Optional[Dict[str, bool]] = None,
                   window_size: Optional[int] = None) -> None:
        """
        Aggiunge un pacchetto al flusso.
        
        Args:
            packet_len: Lunghezza totale pacchetto (include header)
            header_len: Lunghezza header IP+TCP/UDP
            payload_len: Lunghezza payload applicativo
            is_forward: True se direzione forward (src->dst)
            timestamp: Timestamp del pacchetto
            tcp_flags: Dict con flag TCP {'F': bool, 'S': bool, ...}
            window_size: TCP window size
        """
        # Update timing
        old_last_time = self.last_time
        self.last_time = timestamp
        
        # Track Active/Idle
        time_since_last = timestamp - old_last_time
        self._update_active_idle(time_since_last, timestamp)
        
        if is_forward:
            self._add_forward_packet(packet_len, header_len, payload_len, 
                                     timestamp, tcp_flags, window_size)
        else:
            self._add_backward_packet(packet_len, header_len, payload_len,
                                      timestamp, tcp_flags, window_size)
    
    def _add_forward_packet(self, packet_len: int, header_len: int, payload_len: int,
                            timestamp: float, tcp_flags: Optional[Dict], 
                            window_size: Optional[int]) -> None:
        """Aggiunge pacchetto forward."""
        self.fwd_packets += 1
        self.fwd_bytes += packet_len
        self.fwd_header_bytes += header_len
        self.fwd_lengths.append(packet_len)
        
        # IAT
        if self.last_fwd_time is not None:
            iat = timestamp - self.last_fwd_time
            self.fwd_iats.append(iat)
        self.last_fwd_time = timestamp
        
        # Window size iniziale
        if self.init_win_fwd is None and window_size is not None:
            self.init_win_fwd = window_size
        
        # Active data packet
        if payload_len > 0:
            self.act_data_pkt_fwd += 1
        
        # TCP Flags
        if tcp_flags:
            self._update_flags(tcp_flags, is_forward=True)
    
    def _add_backward_packet(self, packet_len: int, header_len: int, payload_len: int,
                             timestamp: float, tcp_flags: Optional[Dict],
                             window_size: Optional[int]) -> None:
        """Aggiunge pacchetto backward."""
        self.bwd_packets += 1
        self.bwd_bytes += packet_len
        self.bwd_header_bytes += header_len
        self.bwd_lengths.append(packet_len)
        
        # IAT
        if self.last_bwd_time is not None:
            iat = timestamp - self.last_bwd_time
            self.bwd_iats.append(iat)
        self.last_bwd_time = timestamp
        
        # Window size iniziale
        if self.init_win_bwd is None and window_size is not None:
            self.init_win_bwd = window_size
        
        # TCP Flags
        if tcp_flags:
            self._update_flags(tcp_flags, is_forward=False)
    
    def _update_flags(self, tcp_flags: Dict[str, bool], is_forward: bool) -> None:
        """Aggiorna contatori flag TCP."""
        if tcp_flags.get('F'):
            self.fin_count += 1
        if tcp_flags.get('S'):
            self.syn_count += 1
        if tcp_flags.get('R'):
            self.rst_count += 1
        if tcp_flags.get('P'):
            self.psh_count += 1
            if is_forward:
                self.fwd_psh_flags += 1
            else:
                self.bwd_psh_flags += 1
        if tcp_flags.get('A'):
            self.ack_count += 1
        if tcp_flags.get('U'):
            self.urg_count += 1
            if is_forward:
                self.fwd_urg_flags += 1
            else:
                self.bwd_urg_flags += 1
        if tcp_flags.get('C'):  # CWR/CWE
            self.cwe_count += 1
        if tcp_flags.get('E'):  # ECE
            self.ece_count += 1
    
    def _update_active_idle(self, time_since_last: float, current_time: float) -> None:
        """
        Aggiorna tracking periodi attivi e idle.
        
        CICFlowMeter considera un flusso "idle" se passa più di 1 secondo
        senza pacchetti.
        """
        if time_since_last > IDLE_THRESHOLD:
            # Periodo idle rilevato
            if self._last_active_start is not None:
                # Fine periodo attivo
                active_duration = (current_time - time_since_last) - self._last_active_start
                if active_duration > 0:
                    self.active_times.append(active_duration)
            
            # Registra periodo idle
            self.idle_times.append(time_since_last)
            
            # Nuovo periodo attivo inizia ora
            self._last_active_start = current_time
            self._in_idle_period = False
    
    def finalize(self) -> None:
        """
        Finalizza il flusso quando scade il timeout.
        
        Registra l'ultimo periodo attivo se presente.
        """
        if self._last_active_start is not None:
            active_duration = self.last_time - self._last_active_start
            if active_duration > 0:
                self.active_times.append(active_duration)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte flusso in dizionario per logging/JSON."""
        return {
            'flow_id': self.flow_id,
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'duration_sec': self.duration,
            'fwd_packets': self.fwd_packets,
            'bwd_packets': self.bwd_packets,
            'total_packets': self.total_packets,
            'fwd_bytes': self.fwd_bytes,
            'bwd_bytes': self.bwd_bytes,
            'total_bytes': self.total_bytes
        }
    
    def is_expired(self, current_time: float, timeout: float = DEFAULT_FLOW_TIMEOUT) -> bool:
        """Verifica se il flusso è scaduto (idle timeout)."""
        return (current_time - self.last_time) > timeout
    
    def is_active_timeout(self, current_time: float, active_timeout: float = ACTIVE_TIMEOUT) -> bool:
        """
        Verifica se il flusso ha superato l'active timeout.
        
        IMPORTANTE per attacchi come:
        - Slowloris: connessioni aperte a lungo con poco traffico
        - Port Scan: tanti SYN senza ACK
        - DoS lenti: traffico costante per lungo tempo
        
        Forza l'analisi anche se il flusso non è "completo".
        """
        flow_duration = current_time - self.start_time
        return flow_duration > active_timeout
    
    def should_analyze(self, current_time: float = None) -> bool:
        """
        Verifica se il flusso deve essere analizzato.
        
        Condizioni per analisi:
        1. Raggiunto MAX_PACKETS_PER_FLOW
        2. Superato ACTIVE_TIMEOUT (flusso attivo da troppo tempo)
        3. Visto flag FIN o RST (chiusura connessione)
        """
        # Max packets reached
        if self.total_packets >= MAX_PACKETS_PER_FLOW:
            return True
        
        # Active timeout check
        if current_time is not None:
            if self.is_active_timeout(current_time):
                return True
        
        # Connection termination flags
        if self.fin_count > 0 or self.rst_count > 0:
            # Aspetta un po' dopo FIN/RST per catturare eventuali ACK finali
            if self.total_packets >= 4:  # Minimo per una connessione base
                return True
        
        return False


# ==============================================================================
# FLOW MANAGER
# ==============================================================================

class FlowManager:
    """
    Gestisce l'aggregazione dei pacchetti in flussi bidirezionali.
    
    Thread-safe per uso in ambiente multi-thread.
    """
    
    def __init__(self, timeout: float = DEFAULT_FLOW_TIMEOUT):
        """
        Args:
            timeout: Secondi prima che un flusso scada
        """
        self.flows: Dict[str, Flow] = {}
        self.timeout = timeout
        self.lock = threading.Lock()
        
        # Statistics
        self.total_packets_processed = 0
        self.total_flows_created = 0
        self.total_flows_completed = 0
    
    @staticmethod
    def _make_flow_key(src_ip: str, dst_ip: str, 
                       src_port: int, dst_port: int, 
                       protocol: int) -> Tuple[str, bool]:
        """
        Genera chiave normalizzata per flusso bidirezionale.
        
        Returns:
            Tuple (key, is_forward)
        """
        if (src_ip, src_port) < (dst_ip, dst_port):
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}", True
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}", False
    
    def add_packet(self,
                   src_ip: str, dst_ip: str,
                   src_port: int, dst_port: int,
                   protocol: int,
                   packet_len: int,
                   header_len: int,
                   payload_len: int,
                   timestamp: float,
                   tcp_flags: Optional[Dict[str, bool]] = None,
                   window_size: Optional[int] = None) -> Optional[Flow]:
        """
        Aggiunge un pacchetto al flusso appropriato.
        
        Args:
            src_ip, dst_ip: Indirizzi IP
            src_port, dst_port: Porte
            protocol: Protocollo IP
            packet_len: Lunghezza totale pacchetto
            header_len: Lunghezza header
            payload_len: Lunghezza payload
            timestamp: Timestamp
            tcp_flags: Flag TCP
            window_size: TCP window size
        
        Returns:
            Flow se il flusso è completo (max packets raggiunto), None altrimenti
        """
        key, is_forward = self._make_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
        
        with self.lock:
            self.total_packets_processed += 1
            
            # Crea nuovo flusso se non esiste
            if key not in self.flows:
                if is_forward:
                    flow = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                else:
                    flow = Flow(dst_ip, src_ip, dst_port, src_port, protocol)
                flow.start_time = timestamp
                self.flows[key] = flow
                self.total_flows_created += 1
            
            flow = self.flows[key]
            flow.add_packet(
                packet_len=packet_len,
                header_len=header_len,
                payload_len=payload_len,
                is_forward=is_forward,
                timestamp=timestamp,
                tcp_flags=tcp_flags,
                window_size=window_size
            )
            
            # Verifica se flusso completo (max packets, active timeout, o FIN/RST)
            if flow.should_analyze(current_time=timestamp):
                del self.flows[key]
                self.total_flows_completed += 1
                flow.finalize()
                return flow
        
        return None
    
    def get_expired_flows(self, reference_time: Optional[float] = None) -> List[Flow]:
        """
        Restituisce e rimuove flussi scaduti.
        
        Args:
            reference_time: Tempo di riferimento (default: time.time())
        
        Returns:
            Lista di Flow scaduti
        """
        if reference_time is None:
            reference_time = time.time()
        
        expired = []
        
        with self.lock:
            expired_keys = [
                key for key, flow in self.flows.items()
                if flow.is_expired(reference_time, self.timeout)
            ]
            
            for key in expired_keys:
                flow = self.flows.pop(key)
                flow.finalize()
                expired.append(flow)
                self.total_flows_completed += 1
        
        return expired
    
    def get_all_flows(self) -> List[Flow]:
        """
        Restituisce tutti i flussi rimanenti (per shutdown).
        
        Returns:
            Lista di tutti i Flow
        """
        with self.lock:
            flows = []
            for flow in self.flows.values():
                flow.finalize()
                flows.append(flow)
            self.flows.clear()
            self.total_flows_completed += len(flows)
        
        return flows
    
    def get_flow_count(self) -> int:
        """Numero di flussi attivi."""
        with self.lock:
            return len(self.flows)
    
    def add_packet_from_info(self, pkt_info) -> Optional[Flow]:
        """
        Aggiunge un pacchetto da un oggetto PacketInfo.
        
        Wrapper per add_packet che accetta un oggetto con attributi.
        
        Args:
            pkt_info: Oggetto con attributi (src_ip, dst_ip, src_port, etc.)
            
        Returns:
            Flow se il flusso è completo, None altrimenti
        """
        # Calcola packet_len come header + payload
        packet_len = pkt_info.header_length + pkt_info.payload_size
        
        return self.add_packet(
            src_ip=pkt_info.src_ip,
            dst_ip=pkt_info.dst_ip,
            src_port=pkt_info.src_port,
            dst_port=pkt_info.dst_port,
            protocol=pkt_info.protocol,
            packet_len=packet_len,
            header_len=pkt_info.header_length,
            payload_len=pkt_info.payload_size,
            timestamp=pkt_info.timestamp,
            tcp_flags=getattr(pkt_info, 'tcp_flags', None),
            window_size=getattr(pkt_info, 'window_size', None)
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Statistiche del manager."""
        with self.lock:
            return {
                'active_flows': len(self.flows),
                'total_packets': self.total_packets_processed,
                'flows_created': self.total_flows_created,
                'flows_completed': self.total_flows_completed
            }