"""
Predictor per inference con modello addestrato.

Supporta XGBoost e LightGBM con batch inference.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

from config import MODEL_PATH, MODEL_TYPE, ATTACK_THRESHOLD, N_FEATURES
from utils.logger import get_logger


logger = get_logger()


@dataclass
class PredictionResult:
    """Risultato di una predizione."""
    
    prediction: int  # 0 = benign, 1 = attack
    confidence: float  # Probabilita classe predetta
    probabilities: np.ndarray  # [prob_benign, prob_attack]


class ModelPredictor:
    """Predictor per classificazione binaria attack/benign."""
    
    def __init__(self, model_path: Optional[Path] = None, threshold: float = ATTACK_THRESHOLD):
        """
        Inizializza predictor.
        
        Args:
            model_path: Path al modello (default: da config)
            threshold: Threshold per classificazione attack (default: 0.5)
        """
        
        self.model_path = model_path or MODEL_PATH
        self.threshold = threshold
        self.model = None
        self.model_type = MODEL_TYPE
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Carica modello da file."""
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Verifica modello
            if not hasattr(self.model, 'predict'):
                raise ValueError("Invalid model: missing predict method")
            
            if not hasattr(self.model, 'predict_proba'):
                logger.warning("Model does not support predict_proba, using predict only")
            
            logger.info(f"Model type: {type(self.model).__name__}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Predici singolo sample.
        
        Args:
            features: Feature vector scalate (shape: (n_features,))
        
        Returns:
            PredictionResult
        """
        
        # Valida input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[1]}")
        
        # Predizione
        try:
            # Probabilita (se supportate)
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                prediction = 1 if proba[1] >= self.threshold else 0
                confidence = proba[prediction]
            else:
                # Fallback a predict
                prediction = self.model.predict(features)[0]
                proba = np.array([0.0, 1.0] if prediction == 1 else [1.0, 0.0])
                confidence = 1.0
            
            return PredictionResult(
                prediction=int(prediction),
                confidence=float(confidence),
                probabilities=proba
            )
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, feature_batch: np.ndarray) -> List[PredictionResult]:
        """
        Predici batch di sample.
        
        Args:
            feature_batch: Batch di feature scalate (shape: (batch_size, n_features))
        
        Returns:
            Lista di PredictionResult
        """
        
        if feature_batch.ndim != 2:
            raise ValueError("Expected 2D array for batch prediction")
        
        if feature_batch.shape[1] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {feature_batch.shape[1]}")
        
        try:
            # Batch prediction
            if hasattr(self.model, 'predict_proba'):
                proba_batch = self.model.predict_proba(feature_batch)
                predictions = (proba_batch[:, 1] >= self.threshold).astype(int)
                confidences = proba_batch[np.arange(len(predictions)), predictions]
            else:
                predictions = self.model.predict(feature_batch)
                proba_batch = np.zeros((len(predictions), 2))
                proba_batch[np.arange(len(predictions)), predictions] = 1.0
                confidences = np.ones(len(predictions))
            
            # Costruisci risultati
            results = []
            for i in range(len(predictions)):
                results.append(PredictionResult(
                    prediction=int(predictions[i]),
                    confidence=float(confidences[i]),
                    probabilities=proba_batch[i]
                ))
            
            return results
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Restituisce informazioni sul modello.
        
        Returns:
            Dict con info modello
        """
        
        if self.model is None:
            return {}
        
        info = {
            "model_type": type(self.model).__name__,
            "threshold": self.threshold,
            "supports_proba": hasattr(self.model, 'predict_proba'),
        }
        
        # Info specifiche per XGBoost
        if hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                info["n_trees"] = len(booster.get_dump())
            except:
                pass
        
        # Info specifiche per LightGBM
        if hasattr(self.model, 'booster_'):
            try:
                info["n_trees"] = self.model.booster_.num_trees()
            except:
                pass
        
        return info
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Aggiorna threshold di classificazione.
        
        Args:
            new_threshold: Nuovo threshold (0.0 - 1.0)
        """
        
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Invalid threshold: {new_threshold} (must be 0.0-1.0)")
        
        old_threshold = self.threshold
        self.threshold = new_threshold
        
        logger.info(f"Threshold updated: {old_threshold:.3f} -> {new_threshold:.3f}")
