"""
Preprocessor per feature scaling.

Applica il RobustScaler addestrato durante la feature engineering.
Espone sia preprocess() (singolo sample) che preprocess_batch() (matrice).
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from config import SCALER_PATH, N_FEATURES
from utils.logger import get_logger


logger = get_logger()


class FeaturePreprocessor:
    """
    Wrapper intorno al RobustScaler serializzato.

    Gestisce automaticamente:
    - Reshape singolo sample / batch
    - Sostituzione di inf/nan post-scaling (stesso comportamento del training)
    - Validazione della shape in input
    """

    def __init__(self, scaler_path: Optional[Path] = None) -> None:
        """
        Args:
            scaler_path: Percorso al file scaler.pkl.
                         Se None, usa il valore da config.SCALER_PATH.
        """
        self.scaler_path = scaler_path or SCALER_PATH
        self.scaler = None
        self._load_scaler()

    def _load_scaler(self) -> None:
        """Carica il RobustScaler dal file serializzato."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler non trovato: {self.scaler_path}")

        try:
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler caricato da {self.scaler_path}")

            if not hasattr(self.scaler, "transform"):
                raise ValueError("Scaler invalido: metodo transform assente")

            # Se lo scaler e' stato fittato con un DataFrame (feature_names_in_
            # presente), sklearn emette un UserWarning ad ogni transform() su
            # array numpy. Rimuovere l'attributo sopprime il warning senza
            # alterare il comportamento numerico dello scaler.
            if hasattr(self.scaler, "feature_names_in_"):
                del self.scaler.feature_names_in_

            scaler_n = getattr(self.scaler, "n_features_in_", None)
            if scaler_n is not None and scaler_n != N_FEATURES:
                # Questo e' un errore bloccante: il modello si aspetta N_FEATURES
                # ma lo scaler e' stato fittato su un numero diverso.
                # Il messaggio dirige esplicitamente alla root cause.
                raise ValueError(
                    f"Mismatch critico: scaler.n_features_in_={scaler_n} "
                    f"ma N_FEATURES da features.json={N_FEATURES}. "
                    "Gli artifacts (scaler.pkl, features.json, model) devono "
                    "essere generati dalla stessa esecuzione della pipeline. "
                    "Rigenera tutto con: python srcNF/pipeline.py"
                )

        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            logger.error(f"Errore nel caricamento dello scaler: {exc}")
            raise

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Scala un singolo vettore feature.

        Args:
            features: Array 1-D di shape (N_FEATURES,) oppure
                      2-D di shape (1, N_FEATURES).

        Returns:
            Array 1-D di shape (N_FEATURES,) scalato.

        Raises:
            ValueError: Se la shape e' incompatibile con N_FEATURES.
        """
        if features.ndim == 1:
            matrix = features.reshape(1, -1)
            return self._transform(matrix).flatten()

        if features.ndim == 2 and features.shape[0] == 1:
            return self._transform(features).flatten()

        raise ValueError(
            f"preprocess() accetta solo vettori 1-D o matrici (1, N). "
            f"Per batch usa preprocess_batch(). Shape ricevuta: {features.shape}"
        )

    def preprocess_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Scala una matrice di feature (batch).

        Args:
            features: Array 2-D di shape (n_samples, N_FEATURES).

        Returns:
            Array 2-D di shape (n_samples, N_FEATURES) scalato.

        Raises:
            ValueError: Se la shape e' incompatibile.
        """
        if features.ndim != 2:
            raise ValueError(
                f"preprocess_batch() richiede un array 2-D. "
                f"Shape ricevuta: {features.shape}"
            )
        return self._transform(features)

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Applica lo scaler e gestisce inf/nan post-scaling.

        Args:
            matrix: Array 2-D (n_samples, N_FEATURES).

        Returns:
            Array 2-D scalato senza inf/nan.
        """
        if matrix.shape[1] != N_FEATURES:
            raise ValueError(
                f"Shape incompatibile: {matrix.shape}, atteso (*, {N_FEATURES})"
            )

        scaled = self.scaler.transform(matrix)
        return self._sanitize(scaled)

    @staticmethod
    def _sanitize(X: np.ndarray) -> np.ndarray:
        """
        Sostituisce inf e nan con valori finiti.

        Comportamento identico a handle_inf_after_scaling() nel training:
          +inf  -> +1e10
          -inf  -> -1e10
          nan   ->  0.0
          valori float finiti ma fuori range float32 -> clipped

        Args:
            X: Array numpy (qualsiasi shape).

        Returns:
            Array con gli stessi dtype/shape, senza inf/nan.
        """
        X[np.isposinf(X)] = 1e10
        X[np.isneginf(X)] = -1e10
        X[np.isnan(X)] = 0.0

        float32_max = np.finfo(np.float32).max
        X = np.clip(X, -float32_max, float32_max)
        return X

    def get_scaler_info(self) -> dict:
        """Restituisce informazioni sullo scaler caricato."""
        if self.scaler is None:
            return {"loaded": False}

        return {
            "loaded":       True,
            "scaler_type":  type(self.scaler).__name__,
            "n_features":   getattr(self.scaler, "n_features_in_", N_FEATURES),
            "scaler_path":  str(self.scaler_path),
        }