"""
Live NIDS Sniffer - Main Entry Point.

Network Intrusion Detection System basato su ML con nfstream.
"""

import sys
import time
import signal
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import argparse

import numpy as np

# Import moduli sniffer
from config import (
    validate_config, get_config_summary, OperationMode,
    OPERATION_MODE, NETWORK_INTERFACE, INFERENCE_BATCH_SIZE,
    INFERENCE_BATCH_TIMEOUT, STATS_LOG_INTERVAL, FLOW_EXPIRATION_CHECK_INTERVAL
)
from utils.logger import get_logger
from core.capture import FlowCaptureEngine
from core.feature_mapper import FeatureMapper
from core.preprocessor import FeaturePreprocessor
from core.predictor import ModelPredictor
from security.alert_manager import AlertManager
from security.firewall_controller import FirewallController


logger = get_logger()


class LiveNIDSSniffer:
    """Live Network Intrusion Detection System Sniffer."""
    
    def __init__(
        self,
        interface: Optional[str] = None,
        operation_mode: OperationMode = OPERATION_MODE,
        batch_size: int = INFERENCE_BATCH_SIZE,
        batch_timeout: float = INFERENCE_BATCH_TIMEOUT,
    ):
        """
        Inizializza sniffer.
        
        Args:
            interface: Interfaccia di rete (None = auto-detect)
            operation_mode: ALERT o BLOCK
            batch_size: Batch size per inference
            batch_timeout: Timeout batch (secondi)
        """
        
        self.interface = interface or NETWORK_INTERFACE
        self.operation_mode = operation_mode
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Componenti
        self.capture_engine: Optional[FlowCaptureEngine] = None
        self.feature_mapper: Optional[FeatureMapper] = None
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.predictor: Optional[ModelPredictor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.firewall: Optional[FirewallController] = None
        
        # Batch processing
        self.feature_batch: List[np.ndarray] = []
        self.metadata_batch: List[dict] = []
        self.batch_start_time: float = time.time()
        
        # Statistiche
        self.total_flows_processed = 0
        self.total_predictions = 0
        self.last_stats_time = time.time()
        self.last_cleanup_time = time.time()
        
        # Controllo interruzione
        self.running = False
        
        logger.info("LiveNIDSSniffer initialized")
    
    def setup(self) -> None:
        """Setup componenti."""
        
        logger.info("="*70)
        logger.info("LIVE NIDS SNIFFER - SETUP")
        logger.info("="*70)
        
        # Valida config
        validate_config()
        logger.info("Configuration validated")
        
        # Log config summary
        config = get_config_summary()
        logger.info(f"Operation mode: {config['operation_mode']}")
        logger.info(f"Model: {config['model_type']}")
        logger.info(f"Interface: {config['network_interface']}")
        logger.info(f"Attack threshold: {config['attack_threshold']}")
        logger.info(f"Batch size: {config['inference_batch_size']}")
        
        # Setup firewall se BLOCK mode
        if self.operation_mode == OperationMode.BLOCK:
            logger.info("\nInitializing firewall controller...")
            self.firewall = FirewallController()
        
        # Setup componenti ML
        logger.info("\nInitializing ML components...")
        self.feature_mapper = FeatureMapper()
        self.preprocessor = FeaturePreprocessor()
        self.predictor = ModelPredictor()
        
        # Alert manager
        self.alert_manager = AlertManager(
            operation_mode=self.operation_mode,
            firewall=self.firewall
        )
        
        # Capture engine
        logger.info("\nInitializing capture engine...")
        self.capture_engine = FlowCaptureEngine(interface=self.interface)
        
        logger.info("\n" + "="*70)
        logger.info("SETUP COMPLETED")
        logger.info("="*70 + "\n")
    
    def start(self) -> None:
        """Avvia sniffer."""
        
        logger.info("Starting Live NIDS Sniffer...")
        logger.info(f"Mode: {self.operation_mode.value.upper()}")
        logger.info(f"Interface: {self.interface}")
        logger.info("Press Ctrl+C to stop\n")
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Avvia capture
        self.capture_engine.start()
        self.running = True
        
        # Main loop
        try:
            self._processing_loop()
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            self.stop()
    
    def _processing_loop(self) -> None:
        """Loop principale di processing."""
        
        logger.info("Processing loop started\n")
        
        while self.running:
            # Processa flow dalla queue
            flow = self.capture_engine.get_flow(timeout=0.1)
            
            if flow is not None:
                self._process_flow(flow)
            
            # Check batch timeout
            elapsed = time.time() - self.batch_start_time
            if elapsed >= self.batch_timeout and len(self.feature_batch) > 0:
                self._process_batch()
            
            # Check batch size
            if len(self.feature_batch) >= self.batch_size:
                self._process_batch()
            
            # Log statistiche periodiche
            if time.time() - self.last_stats_time >= STATS_LOG_INTERVAL:
                self._log_statistics()
            
            # Cleanup blocchi scaduti
            if self.firewall and time.time() - self.last_cleanup_time >= FLOW_EXPIRATION_CHECK_INTERVAL:
                self.firewall.cleanup_expired_blocks()
                self.last_cleanup_time = time.time()
    
    def _process_flow(self, flow) -> None:
        """
        Processa singolo flow.
        
        Args:
            flow: NFFlow da processare
        """
        
        try:
            # Estrai feature
            features = self.feature_mapper.extract_features(flow)
            
            # Valida
            if not self.feature_mapper.validate_feature_vector(features):
                logger.warning("Invalid feature vector, skipping flow")
                return
            
            # Estrai metadata
            metadata = self.feature_mapper.extract_flow_metadata(flow)
            
            # Aggiungi a batch
            self.feature_batch.append(features)
            self.metadata_batch.append(metadata)
            
            self.total_flows_processed += 1
        
        except Exception as e:
            logger.error(f"Error processing flow: {e}")
    
    def _process_batch(self) -> None:
        """Processa batch di feature."""
        
        if len(self.feature_batch) == 0:
            return
        
        try:
            # Stack features in matrice
            feature_matrix = np.vstack(self.feature_batch)
            
            # Preprocessing (scaling)
            scaled_features = self.preprocessor.preprocess_batch(feature_matrix)
            
            # Prediction batch
            results = self.predictor.predict_batch(scaled_features)
            
            # Processa risultati
            for i, result in enumerate(results):
                metadata = self.metadata_batch[i]
                
                # Se e' un attacco, genera alert
                if result.prediction == 1:
                    self.alert_manager.process_alert(
                        src_ip=metadata['src_ip'],
                        dst_ip=metadata['dst_ip'],
                        prediction=result.prediction,
                        confidence=result.confidence,
                        metadata=metadata
                    )
                
                self.total_predictions += 1
            
            # Reset batch
            self.feature_batch.clear()
            self.metadata_batch.clear()
            self.batch_start_time = time.time()
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Reset batch comunque
            self.feature_batch.clear()
            self.metadata_batch.clear()
    
    def _log_statistics(self) -> None:
        """Log statistiche operative."""
        
        # Capture stats
        capture_stats = self.capture_engine.get_stats()
        
        # Alert stats
        alert_stats = self.alert_manager.get_statistics()
        
        # Combina
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_flows': self.total_flows_processed,
            'total_predictions': self.total_predictions,
            'total_alerts': alert_stats['total_alerts'],
            'total_blocks': alert_stats['total_blocks'],
            'flows_per_second': capture_stats.flows_per_second,
            'memory_usage_mb': capture_stats.memory_usage_mb,
            'cpu_usage_percent': capture_stats.cpu_usage_percent,
            'batch_pending': len(self.feature_batch),
        }
        
        logger.log_statistics(stats)
        self.last_stats_time = time.time()
    
    def _signal_handler(self, signum, frame) -> None:
        """Handler per segnali (Ctrl+C)."""
        logger.info(f"\nReceived signal {signum}, stopping...")
        self.running = False
    
    def stop(self) -> None:
        """Ferma sniffer."""
        
        logger.info("\nStopping Live NIDS Sniffer...")
        
        self.running = False
        
        # Processa batch rimanenti
        if len(self.feature_batch) > 0:
            logger.info("Processing remaining batch...")
            self._process_batch()
        
        # Stop capture
        if self.capture_engine:
            self.capture_engine.stop()
        
        # Log statistiche finali
        self._log_statistics()
        
        # Log alert stats dettagliate
        alert_stats = self.alert_manager.get_statistics()
        logger.info("\n" + "="*70)
        logger.info("FINAL STATISTICS")
        logger.info("="*70)
        logger.info(f"Total flows processed: {self.total_flows_processed:,}")
        logger.info(f"Total predictions: {self.total_predictions:,}")
        logger.info(f"Total alerts: {alert_stats['total_alerts']:,}")
        logger.info(f"Total blocks: {alert_stats['total_blocks']:,}")
        
        if alert_stats.get('top_alerting_ips'):
            logger.info("\nTop Alerting IPs:")
            for ip, count in list(alert_stats['top_alerting_ips'].items())[:5]:
                logger.info(f"  {ip}: {count} alerts")
        
        # Cleanup firewall
        if self.firewall:
            logger.info("\nCleaning up firewall...")
            self.firewall.teardown()
        
        logger.info("\n" + "="*70)
        logger.info("SNIFFER STOPPED")
        logger.info("="*70)


def main():
    """Entry point CLI."""
    
    parser = argparse.ArgumentParser(
        description='Live Network Intrusion Detection System (ML-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

1. Run in ALERT mode (logging only, no blocking):
   sudo python -m live_sniffer.main --mode alert

2. Run in BLOCK mode (automatic IP blocking):
   sudo python -m live_sniffer.main --mode block

3. Specify network interface:
   sudo python -m live_sniffer.main --interface eth0 --mode alert

4. Custom batch size:
   sudo python -m live_sniffer.main --batch-size 50 --mode alert

NOTES:
- Requires ROOT privileges (sudo) for packet capture and iptables
- Press Ctrl+C to stop gracefully
- Logs are saved in logs/sniffer/ directory
        """
    )
    
    parser.add_argument(
        '--interface', '-i',
        type=str,
        default=None,
        help='Network interface to monitor (default: auto-detect)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['alert', 'block'],
        default='alert',
        help='Operation mode: alert (log only) or block (log + firewall) (default: alert)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=INFERENCE_BATCH_SIZE,
        help=f'Batch size for inference (default: {INFERENCE_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--batch-timeout', '-t',
        type=float,
        default=INFERENCE_BATCH_TIMEOUT,
        help=f'Batch timeout in seconds (default: {INFERENCE_BATCH_TIMEOUT})'
    )
    
    args = parser.parse_args()
    
    # Parse operation mode
    operation_mode = OperationMode.ALERT if args.mode == 'alert' else OperationMode.BLOCK
    
    # Check root privileges
    import os
    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges")
        print("Please run with: sudo python -m live_sniffer.main")
        sys.exit(1)
    
    # Crea e avvia sniffer
    try:
        sniffer = LiveNIDSSniffer(
            interface=args.interface,
            operation_mode=operation_mode,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
        )
        
        sniffer.setup()
        sniffer.start()
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
