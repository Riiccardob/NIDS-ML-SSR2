"""
Firewall Controller per blocco automatico IP malicious.

Supporta iptables/nftables con gestione whitelist e blocchi temporanei.
"""

import subprocess
import time
from typing import Set, Dict, Optional
from threading import Lock
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import (
    IPTABLES_CHAIN, IPTABLES_JUMP_RULE, WHITELIST_IPS,
    BLOCK_DURATION_SECONDS, FIREWALL_TYPE
)
from utils.logger import get_logger


logger = get_logger()


@dataclass
class BlockedIP:
    """Info su IP bloccato."""
    
    ip: str
    blocked_at: datetime
    expires_at: Optional[datetime]
    reason: str
    flow_count: int = 1


class FirewallController:
    """Controller per gestione firewall (iptables/nftables)."""
    
    def __init__(
        self,
        chain_name: str = IPTABLES_CHAIN,
        whitelist: Optional[Set[str]] = None,
        block_duration: int = BLOCK_DURATION_SECONDS,
    ):
        """
        Inizializza firewall controller.
        
        Args:
            chain_name: Nome chain custom per NIDS
            whitelist: Set di IP da non bloccare mai
            block_duration: Durata blocco in secondi (0 = permanente)
        """
        
        self.chain_name = chain_name
        self.whitelist = set(whitelist or WHITELIST_IPS)
        self.block_duration = block_duration
        self.firewall_type = FIREWALL_TYPE
        
        # Tracking IP bloccati
        self.blocked_ips: Dict[str, BlockedIP] = {}
        self.lock = Lock()
        
        # Inizializza firewall
        self._setup_firewall()
        
        logger.info(f"FirewallController initialized with chain: {chain_name}")
        logger.info(f"Whitelist: {len(self.whitelist)} IPs")
        logger.info(f"Block duration: {block_duration}s ({'permanent' if block_duration == 0 else 'temporary'})")
    
    def _setup_firewall(self) -> None:
        """Setup firewall (crea chain se non esiste)."""
        
        if self.firewall_type != "iptables":
            logger.warning(f"Firewall type {self.firewall_type} not fully supported, using iptables")
        
        try:
            # Check se chain esiste
            result = subprocess.run(
                ["iptables", "-L", self.chain_name, "-n"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                # Crea chain
                logger.info(f"Creating iptables chain: {self.chain_name}")
                subprocess.run(
                    ["iptables", "-N", self.chain_name],
                    check=True
                )
                
                # Aggiungi jump rule da INPUT se richiesto
                if IPTABLES_JUMP_RULE:
                    subprocess.run(
                        ["iptables", "-I", "INPUT", "-j", self.chain_name],
                        check=True
                    )
                    logger.info(f"Added jump rule: INPUT -> {self.chain_name}")
            else:
                logger.info(f"Chain {self.chain_name} already exists")
            
            # Flush chain per partire puliti
            subprocess.run(["iptables", "-F", self.chain_name], check=True)
            logger.info("Firewall chain flushed")
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup firewall: {e}")
            logger.error("Make sure you have root privileges (sudo)")
            raise
        
        except FileNotFoundError:
            logger.error("iptables not found. Install iptables package.")
            raise
    
    def block_ip(self, ip: str, reason: str = "malicious_traffic") -> bool:
        """
        Blocca IP via firewall.
        
        Args:
            ip: Indirizzo IP da bloccare
            reason: Motivo del blocco
        
        Returns:
            True se bloccato, False se in whitelist o gia' bloccato
        """
        
        # Check whitelist
        if ip in self.whitelist:
            logger.debug(f"IP {ip} in whitelist, not blocking")
            return False
        
        with self.lock:
            # Check se gia' bloccato
            if ip in self.blocked_ips:
                # Incrementa counter
                self.blocked_ips[ip].flow_count += 1
                logger.debug(f"IP {ip} already blocked (count: {self.blocked_ips[ip].flow_count})")
                return False
            
            try:
                # Aggiungi regola iptables
                subprocess.run(
                    ["iptables", "-I", self.chain_name, "-s", ip, "-j", "DROP"],
                    check=True,
                    capture_output=True
                )
                
                # Tracking
                now = datetime.now()
                expires = None if self.block_duration == 0 else now + timedelta(seconds=self.block_duration)
                
                self.blocked_ips[ip] = BlockedIP(
                    ip=ip,
                    blocked_at=now,
                    expires_at=expires,
                    reason=reason
                )
                
                logger.warning(
                    f"BLOCKED IP: {ip} | Reason: {reason} | "
                    f"Duration: {'permanent' if expires is None else f'{self.block_duration}s'}"
                )
                
                return True
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to block IP {ip}: {e}")
                return False
    
    def unblock_ip(self, ip: str) -> bool:
        """
        Sblocca IP.
        
        Args:
            ip: Indirizzo IP da sbloccare
        
        Returns:
            True se sbloccato, False se non era bloccato
        """
        
        with self.lock:
            if ip not in self.blocked_ips:
                logger.debug(f"IP {ip} not blocked")
                return False
            
            try:
                # Rimuovi regola iptables
                subprocess.run(
                    ["iptables", "-D", self.chain_name, "-s", ip, "-j", "DROP"],
                    check=True,
                    capture_output=True
                )
                
                # Rimuovi tracking
                del self.blocked_ips[ip]
                
                logger.info(f"UNBLOCKED IP: {ip}")
                return True
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to unblock IP {ip}: {e}")
                return False
    
    def cleanup_expired_blocks(self) -> int:
        """
        Rimuove blocchi scaduti.
        
        Returns:
            Numero di IP sbloccati
        """
        
        if self.block_duration == 0:
            return 0  # Blocchi permanenti
        
        now = datetime.now()
        expired = []
        
        with self.lock:
            for ip, block_info in self.blocked_ips.items():
                if block_info.expires_at and now >= block_info.expires_at:
                    expired.append(ip)
        
        # Sblocca IPs scaduti
        count = 0
        for ip in expired:
            if self.unblock_ip(ip):
                count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired blocks")
        
        return count
    
    def is_blocked(self, ip: str) -> bool:
        """Check se IP e' bloccato."""
        with self.lock:
            return ip in self.blocked_ips
    
    def get_blocked_ips(self) -> Dict[str, BlockedIP]:
        """Ottieni dizionario IP bloccati."""
        with self.lock:
            return self.blocked_ips.copy()
    
    def get_block_count(self) -> int:
        """Numero di IP bloccati."""
        with self.lock:
            return len(self.blocked_ips)
    
    def add_to_whitelist(self, ip: str) -> None:
        """
        Aggiungi IP a whitelist.
        
        Args:
            ip: IP da aggiungere
        """
        
        self.whitelist.add(ip)
        logger.info(f"Added to whitelist: {ip}")
        
        # Se era bloccato, sblocca
        if self.is_blocked(ip):
            self.unblock_ip(ip)
    
    def remove_from_whitelist(self, ip: str) -> None:
        """Rimuovi IP da whitelist."""
        
        if ip in self.whitelist:
            self.whitelist.remove(ip)
            logger.info(f"Removed from whitelist: {ip}")
    
    def flush_all_blocks(self) -> int:
        """
        Rimuove tutti i blocchi.
        
        Returns:
            Numero di IP sbloccati
        """
        
        with self.lock:
            ips = list(self.blocked_ips.keys())
        
        count = 0
        for ip in ips:
            if self.unblock_ip(ip):
                count += 1
        
        logger.warning(f"Flushed all blocks: {count} IPs unblocked")
        return count
    
    def teardown(self) -> None:
        """Cleanup firewall (rimuove chain e regole)."""
        
        logger.info("Tearing down firewall...")
        
        try:
            # Flush blocchi
            self.flush_all_blocks()
            
            # Rimuovi jump rule se esiste
            if IPTABLES_JUMP_RULE:
                subprocess.run(
                    ["iptables", "-D", "INPUT", "-j", self.chain_name],
                    check=False,  # Non errore se non esiste
                    capture_output=True
                )
            
            # Flush e delete chain
            subprocess.run(["iptables", "-F", self.chain_name], check=False)
            subprocess.run(["iptables", "-X", self.chain_name], check=False)
            
            logger.info("Firewall chain removed")
        
        except Exception as e:
            logger.error(f"Error during firewall teardown: {e}")
