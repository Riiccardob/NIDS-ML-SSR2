"""
Preprocessor per feature scaling.

Applica RobustScaler addestrato su dataset completo.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from config import SCALER_PATH, N_FEATURES
from utils.logger import get_logger


logger = get_logger()


class FeaturePreprocessor:
    """Preprocessor per scaling feature con RobustScaler."""
    
    def __init__(self, scaler_path: Optional[Path] = None):
        """
        Inizializza preprocessor.
        
        Args:
            scaler_path: Path al file scaler.pkl (default: da config)
        """
        
        self.scaler_path = scaler_path or SCALER_PATH
        self.scaler = None
        self._load_scaler()
    
    def _load_scaler(self) -> None:
        """Carica RobustScaler da file."""
        
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        
        try:
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
            
            # Verifica scaler
            if not hasattr(self.scaler, 'transform'):
                raise ValueError("Invalid scaler: missing transform method")
            
            if self.scaler.n_features_in_ != N_FEATURES:
                logger.warning(
                    f"Scaler expects {self.scaler.n_features_in_} features, "
                    f"but config has {N_FEATURES}"
                )
        
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            raise
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Applica scaling alle feature.
        
        Args:
            features: Feature vector raw (shape: (n_features,) o (n_samples, n_features))
        
        Returns:
            Feature vector scalate
        """
        
        # Gestisci singolo sample vs batch
        single_sample = False
        if features.ndim == 1:
            features = features.reshape(1, -1)
            single_sample = True
        
        # Valida shape
        if features.shape[1] != N_FEATURES:
            raise ValueError(
                f"Invalid feature shape: {features.shape}, expected (*, {N_FEATURES})"
            )
        
        # Applica scaling
        try:
            scaled = self.scaler.transform(features)
            
            # Gestisci inf/nan post-scaling (come in training)
            scaled = self._handle_inf_nan(scaled)
            
            # Se era singolo sample, ritorna flat
            if single_sample:
                return scaled.flatten()
            
            return scaled
        
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise
    
    def _handle_inf_nan(self, X: np.ndarray) -> np.ndarray:
        """
        Gestisce inf/nan post-scaling (stesso metodo di training).
        
        Args:
            X: Feature matrix scalate
        
        Returns:
            Feature matrix con inf/nan gestiti
        """
        
        # Limiti per Float32
        max_val = np.finfo(np.float32).max
        min_val = np.finfo(np.float32).min
        
        # Sostituisci inf con valori grandi ma finiti
        X[np.isposinf(X)] = 1e10
        X[np.isneginf(X)] = -1e10
        
        # Sostituisci nan con 0 (valore neutro per scaler)
        X[np.isnan(X)] = 0.0
        
        # Clamp a limiti Float32
        X = np.clip(X, min_val, max_val)
        
        return X
    
    def preprocess_batch(self, feature_batch: np.ndarray) -> np.ndarray:
        """
        Preprocessa batch di feature.
        
        Args:
            feature_batch: Batch di feature (shape: (batch_size, n_features))
        
        Returns:
            Batch scalato
        """
        
        return self.preprocess(feature_batch)
    
    def get_scaler_info(self) -> dict:
        """
        Restituisce informazioni sullo scaler.
        
        Returns:
            Dict con info scaler
        """
        
        if self.scaler is None:
            return {}
        
        return {
            "scaler_type": type(self.scaler).__name__,
            "n_features": self.scaler.n_features_in_,
            "center": self.scaler.center_[:5].tolist() if hasattr(self.scaler, 'center_') else None,
            "scale": self.scaler.scale_[:5].tolist() if hasattr(self.scaler, 'scale_') else None,
        }
