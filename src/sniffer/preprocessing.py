"""
NIDS-ML Sniffer - Preprocessing Module v4 (WORKING VERSION)

Pipeline CORRETTA (identica al training):
1. Extract 44 features (scaler_columns.json)
2. RobustScaler
3. Clip(-10, 10)

NO feature selection post-scaling!
Il modello è stato trainato su 44 feature, quindi passiamo 44 feature.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

CLIP_VALUE = 10.0


def normalize_name(name: str) -> str:
    """Normalizza nome colonna."""
    return name.strip().lower()


@dataclass
class PipelineArtifacts:
    """Container per artifacts."""
    scaler: Any
    scaler_columns: List[str]  # 44 features
    is_booster: bool = False  # True if LightGBM Booster
    
    def __post_init__(self):
        logger.info(f"Pipeline: {len(self.scaler_columns)} features -> scale -> clip -> predict")


def load_pipeline_artifacts(
    artifacts_dir: str = 'artifacts',
    model_dir: Optional[str] = None
) -> PipelineArtifacts:
    """Carica artifacts."""
    artifacts_path = Path(artifacts_dir)
    
    # Scaler
    scaler_path = artifacts_path / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        scaler = joblib.load(scaler_path)
    logger.info(f"Scaler caricato: {type(scaler).__name__}")
    
    # Scaler columns
    scaler_cols_path = artifacts_path / 'scaler_columns.json'
    if not scaler_cols_path.exists():
        raise FileNotFoundError(f"scaler_columns.json not found")
    
    with open(scaler_cols_path, 'r') as f:
        scaler_columns = json.load(f)
    logger.info(f"Scaler columns: {len(scaler_columns)} features")
    
    return PipelineArtifacts(
        scaler=scaler,
        scaler_columns=scaler_columns
    )


class InferencePipeline:
    """
    Pipeline di preprocessing per inference.
    
    Pipeline:
        features dict → 44 values → scale → clip → ready for model
    """
    
    def __init__(self, artifacts: PipelineArtifacts, clip_value: float = CLIP_VALUE):
        self.artifacts = artifacts
        self.scaler = artifacts.scaler
        self.scaler_columns = artifacts.scaler_columns
        self.clip_value = clip_value
        
        # Build lookup index
        self._cols_index = {}
        for i, col in enumerate(self.scaler_columns):
            norm = normalize_name(col)
            self._cols_index[norm] = i
            self._cols_index[norm.replace(' ', '_')] = i
            self._cols_index[norm.replace('_', ' ')] = i
        
        logger.info(f"Pipeline: {len(self.scaler_columns)} -> scale -> clip")
    
    def transform(self, features: Dict[str, float]) -> np.ndarray:
        """
        Trasforma dict feature in array per predizione.
        
        Args:
            features: Dict {feature_name: value}
        
        Returns:
            Array shape (1, 44)
        """
        # Normalize feature names
        features_norm = {normalize_name(k): v for k, v in features.items()}
        
        # Build value vector in correct order
        values = []
        for col in self.scaler_columns:
            col_norm = normalize_name(col)
            
            if col_norm in features_norm:
                val = features_norm[col_norm]
            else:
                # Try variants
                val = 0.0
                for var in [col_norm.replace(' ', '_'), col_norm.replace('_', ' ')]:
                    if var in features_norm:
                        val = features_norm[var]
                        break
            
            # Handle inf/nan
            if isinstance(val, (int, float)):
                if np.isinf(val) or np.isnan(val):
                    val = 0.0
            
            values.append(float(val))
        
        # Scale
        X = np.array([values], dtype=np.float64)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_scaled = self.scaler.transform(X)
        
        # Clip
        X_clipped = np.clip(X_scaled, -self.clip_value, self.clip_value)
        
        return X_clipped
    
    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Trasforma DataFrame in array per predizione batch.
        
        Args:
            df: DataFrame con colonne feature
        
        Returns:
            Array shape (n_samples, 44)
        """
        # Build column name lookup
        df_cols_norm = {normalize_name(c): c for c in df.columns}
        
        # Extract values in correct order
        n_rows = len(df)
        n_cols = len(self.scaler_columns)
        X = np.zeros((n_rows, n_cols), dtype=np.float64)
        
        for i, col in enumerate(self.scaler_columns):
            col_norm = normalize_name(col)
            
            df_col = None
            if col_norm in df_cols_norm:
                df_col = df_cols_norm[col_norm]
            else:
                for var in [col_norm.replace(' ', '_'), col_norm.replace('_', ' ')]:
                    if var in df_cols_norm:
                        df_col = df_cols_norm[var]
                        break
            
            if df_col and df_col in df.columns:
                X[:, i] = df[df_col].values
        
        # Handle inf/nan
        X = np.where(np.isinf(X), 0, X)
        X = np.where(np.isnan(X), 0, X)
        
        # Scale
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_scaled = self.scaler.transform(X)
        
        # Clip
        X_clipped = np.clip(X_scaled, -self.clip_value, self.clip_value)
        
        return X_clipped
    
    def get_info(self) -> Dict[str, Any]:
        """Info sulla pipeline."""
        return {
            'scaler_type': type(self.scaler).__name__,
            'n_features': len(self.scaler_columns),
            'clip_value': self.clip_value,
            'features': self.scaler_columns[:5]
        }


def create_inference_pipeline(
    artifacts_dir: str = 'artifacts',
    model_dir: Optional[str] = None
) -> InferencePipeline:
    """Factory function."""
    artifacts = load_pipeline_artifacts(artifacts_dir, model_dir)
    return InferencePipeline(artifacts)


def validate_artifacts_consistency(artifacts_dir: str = 'artifacts') -> Dict[str, Any]:
    """Valida coerenza artifacts."""
    results = {
        'valid': True,
        'errors': [],
        'info': {}
    }
    
    try:
        artifacts = load_pipeline_artifacts(artifacts_dir)
        
        results['info'] = {
            'scaler_type': type(artifacts.scaler).__name__,
            'n_features': len(artifacts.scaler_columns)
        }
        
        # Check scaler expects same number of features
        if hasattr(artifacts.scaler, 'n_features_in_'):
            expected = artifacts.scaler.n_features_in_
            actual = len(artifacts.scaler_columns)
            if expected != actual:
                results['errors'].append(
                    f"Scaler expects {expected} features but scaler_columns has {actual}"
                )
                results['valid'] = False
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(str(e))
    
    return results