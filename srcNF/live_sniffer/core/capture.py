"""
Network Capture Engine con nfstream.

Gestisce cattura pacchetti e generazione flow NetFlow.
FIXED per nfstream 6.5.4 API (parametri corretti).
"""

import time
from typing import Iterator, Optional, Callable, Any
from threading import Thread, Event
from queue import Queue, Empty
from dataclasses import dataclass
import psutil

from nfstream import NFStreamer

from config import (
    NETWORK_INTERFACE, FLOW_IDLE_TIMEOUT,
    FLOW_ACTIVE_TIMEOUT, MAX_FLOWS_IN_MEMORY
)
from utils.logger import get_logger


logger = get_logger()


@dataclass
class CaptureStats:
    """Statistiche di cattura."""
    
    total_flows: int = 0
    total_packets: int = 0
    flows_per_second: float = 0.0
    packets_per_second: float = 0.0
    active_flows: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class FlowCaptureEngine:
    """Engine per cattura e processing flow NetFlow."""
    
    def __init__(
        self,
        interface: Optional[str] = None,
        flow_callback: Optional[Callable[[Any], None]] = None,
        idle_timeout: int = FLOW_IDLE_TIMEOUT,
        active_timeout: int = FLOW_ACTIVE_TIMEOUT,
    ):
        """
        Inizializza capture engine.
        
        Args:
            interface: Nome interfaccia di rete (None = auto-detect)
            flow_callback: Funzione chiamata per ogni flow completo
            idle_timeout: Timeout idle flow (secondi)
            active_timeout: Timeout active flow (secondi)
        """
        
        self.interface = interface or self._auto_detect_interface()
        self.flow_callback = flow_callback
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        
        # Streamer nfstream
        self.streamer: Optional[NFStreamer] = None
        
        # Queue per flow processing
        self.flow_queue: Queue[Any] = Queue(maxsize=MAX_FLOWS_IN_MEMORY)
        
        # Thread control
        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None
        
        # Statistiche
        self.stats = CaptureStats()
        self._stats_start_time = time.time()
        
        logger.info(f"FlowCaptureEngine initialized on interface: {self.interface}")
    
    def _auto_detect_interface(self) -> str:
        """Auto-detect interfaccia di rete principale."""
        
        net_if_stats = psutil.net_if_stats()
        net_if_addrs = psutil.net_if_addrs()
        
        for iface, addrs in net_if_addrs.items():
            if iface.startswith('lo'):
                continue
            
            if iface in net_if_stats and net_if_stats[iface].isup:
                for addr in addrs:
                    if addr.family == 2:  # AF_INET (IPv4)
                        logger.info(f"Auto-detected interface: {iface} ({addr.address})")
                        return iface
        
        logger.warning("Could not auto-detect interface, using 'eth0'")
        return "eth0"
    
    def start(self) -> None:
        """Avvia cattura in background thread."""
        
        if self.capture_thread is not None and self.capture_thread.is_alive():
            logger.warning("Capture already running")
            return
        
        logger.info(f"Starting capture on {self.interface}")
        
        self.stop_event.clear()
        
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Capture thread started")
    
    def stop(self) -> None:
        """Ferma cattura."""
        
        logger.info("Stopping capture...")
        self.stop_event.set()
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")
        
        logger.info("Capture stopped")
    
    def _capture_loop(self) -> None:
        """Loop principale di cattura."""
        
        try:
            # Inizializza NFStreamer con parametri corretti per 6.5.4
            # Parametri supportati: source, idle_timeout, active_timeout, 
            # accounting_mode, decode_tunnels, bpf_filter, promiscuous_mode,
            # n_dissections, statistical_analysis, splt_analysis
            
            self.streamer = NFStreamer(
                source=self.interface,
                idle_timeout=self.idle_timeout,
                active_timeout=self.active_timeout,
                accounting_mode=1,  # IP accounting
                decode_tunnels=True,
                bpf_filter=None,  # Cattura tutto
                promiscuous_mode=True,
                n_dissections=20,  # DPI per L7 protocols
                statistical_analysis=True,
                splt_analysis=0,
            )
            
            logger.info("NFStreamer initialized successfully")
            logger.info(f"  Idle timeout: {self.idle_timeout}s")
            logger.info(f"  Active timeout: {self.active_timeout}s")
            logger.info(f"  Promiscuous mode: enabled")
            
            # Loop di cattura
            for flow in self.streamer:
                if self.stop_event.is_set():
                    break
                
                # Aggiorna statistiche
                self._update_stats(flow)
                
                # Metti flow in queue
                try:
                    self.flow_queue.put(flow, timeout=1.0)
                except:
                    logger.warning("Flow queue full, dropping flow")
                    continue
                
                # Callback opzionale
                if self.flow_callback:
                    try:
                        self.flow_callback(flow)
                    except Exception as e:
                        logger.error(f"Flow callback error: {e}")
        
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")
        
        except Exception as e:
            logger.error(f"Capture error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            if self.streamer:
                del self.streamer
            logger.info("Capture loop terminated")
    
    def _update_stats(self, flow: Any) -> None:
        """Aggiorna statistiche di cattura."""
        
        self.stats.total_flows += 1
        
        # Get packets count (dict or attribute access)
        if isinstance(flow, dict):
            packets = flow.get('bidirectional_packets', 0)
        else:
            packets = getattr(flow, 'bidirectional_packets', 0)
        
        self.stats.total_packets += packets
        
        # Calcola rate
        elapsed = time.time() - self._stats_start_time
        if elapsed >= 1.0:
            self.stats.flows_per_second = self.stats.total_flows / elapsed
            self.stats.packets_per_second = self.stats.total_packets / elapsed
            
            self.stats.total_flows = 0
            self.stats.total_packets = 0
            self._stats_start_time = time.time()
        
        # Memory e CPU
        process = psutil.Process()
        self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.stats.cpu_usage_percent = process.cpu_percent()
        
        # Active flows
        self.stats.active_flows = self.flow_queue.qsize()
    
    def get_flow(self, timeout: float = 1.0) -> Optional[Any]:
        """Ottieni prossimo flow dalla queue."""
        
        try:
            return self.flow_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def iter_flows(self, timeout: float = 1.0) -> Iterator[Any]:
        """Itera sui flow."""
        
        while not self.stop_event.is_set():
            flow = self.get_flow(timeout=timeout)
            if flow is not None:
                yield flow
    
    def get_stats(self) -> CaptureStats:
        """Ottieni statistiche correnti."""
        return self.stats
    
    def is_running(self) -> bool:
        """Check se cattura e attiva."""
        return self.capture_thread is not None and self.capture_thread.is_alive()
