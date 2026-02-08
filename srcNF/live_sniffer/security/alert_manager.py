"""
Alert Manager per gestione e logging alert di sicurezza.

Coordina logging strutturato e azioni di risposta.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from config import OperationMode, OPERATION_MODE
from utils.logger import get_logger
from security.firewall_controller import FirewallController


logger = get_logger()


class AlertManager:
    """Manager per gestione alert e azioni di risposta."""
    
    def __init__(
        self,
        operation_mode: OperationMode = OPERATION_MODE,
        firewall: Optional[FirewallController] = None,
    ):
        """
        Inizializza alert manager.
        
        Args:
            operation_mode: Modalita operativa (ALERT o BLOCK)
            firewall: FirewallController (richiesto se mode=BLOCK)
        """
        
        self.operation_mode = operation_mode
        self.firewall = firewall
        
        # Statistiche
        self.total_alerts = 0
        self.total_blocks = 0
        self.alerts_by_ip: Dict[str, int] = defaultdict(int)
        self.alerts_by_protocol: Dict[str, int] = defaultdict(int)
        
        # Validazione
        if self.operation_mode == OperationMode.BLOCK and self.firewall is None:
            raise ValueError("FirewallController required for BLOCK mode")
        
        logger.info(f"AlertManager initialized in {operation_mode.value.upper()} mode")
    
    def process_alert(
        self,
        src_ip: str,
        dst_ip: str,
        prediction: int,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Processa un alert e decide azione.
        
        Args:
            src_ip: IP sorgente
            dst_ip: IP destinazione
            prediction: Predizione (0=benign, 1=attack)
            confidence: Confidenza predizione
            metadata: Metadata flow (porte, protocollo, etc.)
        
        Returns:
            Azione presa ("logged", "blocked", "whitelisted")
        """
        
        self.total_alerts += 1
        
        # Tracking statistiche
        self.alerts_by_ip[src_ip] += 1
        protocol = metadata.get('protocol', 'unknown')
        self.alerts_by_protocol[str(protocol)] += 1
        
        # Determina azione
        action = "logged"
        
        if self.operation_mode == OperationMode.BLOCK:
            # Verifica whitelist
            if self.firewall.is_blocked(src_ip):
                action = "already_blocked"
            elif src_ip in self.firewall.whitelist:
                action = "whitelisted"
            else:
                # Blocca IP
                reason = f"attack_detected_confidence_{confidence:.2f}"
                if self.firewall.block_ip(src_ip, reason=reason):
                    action = "blocked"
                    self.total_blocks += 1
        
        # Costruisci alert data
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'src_ip': src_ip,
            'src_port': metadata.get('src_port', 0),
            'dst_ip': dst_ip,
            'dst_port': metadata.get('dst_port', 0),
            'protocol': protocol,
            'l7_proto': metadata.get('l7_proto', 'unknown'),
            'prediction': 'attack' if prediction == 1 else 'benign',
            'confidence': confidence,
            'action': action,
            'duration_ms': metadata.get('duration_ms', 0),
            'bytes_in': metadata.get('bytes_in', 0),
            'bytes_out': metadata.get('bytes_out', 0),
            'packets_in': metadata.get('packets_in', 0),
            'packets_out': metadata.get('packets_out', 0),
        }
        
        # Log strutturato
        logger.log_alert(alert_data)
        
        return action
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Ottieni statistiche alert.
        
        Returns:
            Dict con statistiche
        """
        
        stats = {
            'total_alerts': self.total_alerts,
            'total_blocks': self.total_blocks,
            'operation_mode': self.operation_mode.value,
            'top_alerting_ips': self._get_top_n(self.alerts_by_ip, 10),
            'alerts_by_protocol': dict(self.alerts_by_protocol),
        }
        
        # Aggiungi stats firewall se disponibile
        if self.firewall:
            stats['blocked_ips_count'] = self.firewall.get_block_count()
            stats['whitelist_count'] = len(self.firewall.whitelist)
        
        return stats
    
    def _get_top_n(self, counter: Dict[str, int], n: int) -> Dict[str, int]:
        """Ottieni top N elementi da counter."""
        
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:n])
    
    def reset_statistics(self) -> None:
        """Reset statistiche."""
        
        self.total_alerts = 0
        self.total_blocks = 0
        self.alerts_by_ip.clear()
        self.alerts_by_protocol.clear()
        
        logger.info("Alert statistics reset")
