"""
NIDS-ML Sniffer - Preprocessing Module per Pipeline

Gestisce il preprocessing delle feature per allinearsi al pipeline di training:
1. Estrazione 77 feature CIC-IDS2017 -> filtra a 44 (scaler_columns.json)
2. RobustScaler su 44 feature
3. Selezione 30 feature per indice (selected_features.json, ordine importanza)

CRITICO: L'ordine delle feature deve essere ESATTAMENTE quello che il modello si aspetta!

CORREZIONI:
- Gestione robusta feature mancanti
- Normalizzazione nomi case-insensitive
- Validazione coerenza artifacts
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Container per tutti gli artifacts necessari all'inference."""
    scaler: Any
    scaler_columns: List[str]
    selected_features: List[str]
    selected_indices: List[int]
    statistical_info: Optional[Dict] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if len(self.selected_indices) != len(self.selected_features):
            raise ValueError(
                f"Mismatch: {len(self.selected_indices)} indices vs "
                f"{len(self.selected_features)} features"
            )


def load_pipeline_artifacts(
    artifacts_dir: str = 'artifacts',
    model_dir: Optional[str] = None
) -> PipelineArtifacts:
    """
    Carica tutti gli artifacts necessari per l'inference.
    
    Args:
        artifacts_dir: Directory contenente scaler e metadata
        model_dir: Directory modello (opzionale, per features specifiche)
    
    Returns:
        PipelineArtifacts con tutti i componenti
    """
    artifacts_path = Path(artifacts_dir)
    
    scaler_path = artifacts_path / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler non trovato: {scaler_path}")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*version.*')
        scaler = joblib.load(scaler_path)
    logger.info(f"Scaler caricato: {type(scaler).__name__}")
    
    scaler_columns_path = artifacts_path / 'scaler_columns.json'
    if scaler_columns_path.exists():
        with open(scaler_columns_path, 'r') as f:
            scaler_columns = json.load(f)
    elif hasattr(scaler, 'feature_names_in_'):
        scaler_columns = list(scaler.feature_names_in_)
        logger.warning("scaler_columns.json non trovato, uso feature_names_in_ da scaler")
    else:
        raise FileNotFoundError(f"scaler_columns.json non trovato: {scaler_columns_path}")
    logger.info(f"Scaler columns: {len(scaler_columns)} feature")
    
    selected_features = None
    search_paths = [
        artifacts_path / 'selected_features.json',
    ]
    if model_dir:
        search_paths.append(Path(model_dir) / 'features_binary.json')
    
    for path in search_paths:
        if path and path.exists():
            with open(path, 'r') as f:
                selected_features = json.load(f)
            logger.info(f"Selected features caricati da: {path}")
            break
    
    if selected_features is None:
        raise FileNotFoundError("selected_features.json non trovato")
    logger.info(f"Selected features: {len(selected_features)}")
    
    scaler_cols_map = {col.strip().lower(): idx for idx, col in enumerate(scaler_columns)}
    selected_indices = []
    missing_features = []
    
    for feat in selected_features:
        feat_lower = feat.strip().lower()
        if feat_lower in scaler_cols_map:
            selected_indices.append(scaler_cols_map[feat_lower])
        else:
            feat_underscore = feat_lower.replace(' ', '_')
            feat_space = feat_lower.replace('_', ' ')
            found = False
            for variant in [feat_underscore, feat_space]:
                if variant in scaler_cols_map:
                    selected_indices.append(scaler_cols_map[variant])
                    found = True
                    break
            if not found:
                missing_features.append(feat)
    
    if missing_features:
        raise ValueError(f"Feature non trovate in scaler_columns: {missing_features}")
    
    statistical_info = None
    stat_path = artifacts_path / 'statistical_preprocessing_info.json'
    if stat_path.exists():
        with open(stat_path, 'r') as f:
            statistical_info = json.load(f)
    
    checksum = None
    checksum_path = artifacts_path / 'column_checksum.json'
    if checksum_path.exists():
        with open(checksum_path, 'r') as f:
            checksum = json.load(f).get('checksum')
    
    return PipelineArtifacts(
        scaler=scaler,
        scaler_columns=scaler_columns,
        selected_features=selected_features,
        selected_indices=selected_indices,
        statistical_info=statistical_info,
        checksum=checksum
    )


class InferencePipeline:
    """Pipeline di preprocessing per inference in tempo reale."""
    
    def __init__(self, artifacts: PipelineArtifacts):
        self.artifacts = artifacts
        self.scaler = artifacts.scaler
        self.scaler_columns = artifacts.scaler_columns
        self.selected_features = artifacts.selected_features
        self.selected_indices = artifacts.selected_indices
        
        self._scaler_cols_lower = {}
        for i, col in enumerate(self.scaler_columns):
            col_lower = col.strip().lower()
            self._scaler_cols_lower[col_lower] = i
            self._scaler_cols_lower[col_lower.replace(' ', '_')] = i
            self._scaler_cols_lower[col_lower.replace('_', ' ')] = i
        
        logger.info(
            f"Pipeline: {len(self.scaler_columns)} -> scale -> "
            f"select {len(self.selected_features)} features"
        )
    
    def transform(self, features: Dict[str, float]) -> np.ndarray:
        """
        Trasforma dict feature in array per predizione.
        
        Args:
            features: Dizionario {feature_name: value}
        
        Returns:
            Array shape (1, n_selected_features)
        """
        features_lower = {k.strip().lower(): v for k, v in features.items()}
        
        for k, v in list(features_lower.items()):
            features_lower[k.replace(' ', '_')] = v
            features_lower[k.replace('_', ' ')] = v
        
        values = []
        for col in self.scaler_columns:
            col_lower = col.strip().lower()
            if col_lower in features_lower:
                values.append(features_lower[col_lower])
            else:
                values.append(0.0)
        
        arr = np.array([values], dtype=np.float64)
        
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            scaled = self.scaler.transform(arr)
        
        return scaled[:, self.selected_indices]
    
    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Trasforma DataFrame in array per predizione batch.
        
        Args:
            df: DataFrame con colonne feature
        
        Returns:
            Array shape (n_samples, n_selected_features)
        """
        col_map = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            col_map[col_lower] = col
            col_map[col_lower.replace(' ', '_')] = col
            col_map[col_lower.replace('_', ' ')] = col
        
        cols_to_use = []
        for scaler_col in self.scaler_columns:
            scaler_col_lower = scaler_col.strip().lower()
            
            if scaler_col_lower in col_map:
                cols_to_use.append(col_map[scaler_col_lower])
            else:
                df[scaler_col] = 0.0
                cols_to_use.append(scaler_col)
        
        df_ordered = df[cols_to_use].copy()
        
        df_ordered = df_ordered.replace([np.inf, -np.inf], np.nan)
        df_ordered = df_ordered.fillna(0.0)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            scaled = self.scaler.transform(df_ordered.values)
        
        return scaled[:, self.selected_indices]
    
    def get_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sulla pipeline."""
        return {
            'scaler_type': type(self.scaler).__name__,
            'n_scaler_columns': len(self.scaler_columns),
            'n_selected_features': len(self.selected_features),
            'top_5_features': self.selected_features[:5],
            'checksum': self.artifacts.checksum
        }


def create_inference_pipeline(
    artifacts_dir: str = 'artifacts',
    model_dir: Optional[str] = None
) -> InferencePipeline:
    """Factory function per creare InferencePipeline."""
    return InferencePipeline(load_pipeline_artifacts(artifacts_dir, model_dir))


def validate_artifacts_consistency(artifacts_dir: str = 'artifacts') -> Dict[str, Any]:
    """
    Valida coerenza degli artifacts.
    
    Returns:
        Dict con risultati validazione
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        artifacts = load_pipeline_artifacts(artifacts_dir)
        
        results['info'] = {
            'scaler_type': type(artifacts.scaler).__name__,
            'n_scaler_columns': len(artifacts.scaler_columns),
            'n_selected_features': len(artifacts.selected_features),
            'checksum': artifacts.checksum,
        }
        
        max_idx = max(artifacts.selected_indices) if artifacts.selected_indices else 0
        if max_idx >= len(artifacts.scaler_columns):
            results['errors'].append(
                f"Index {max_idx} out of range (max: {len(artifacts.scaler_columns)-1})"
            )
            results['valid'] = False
        
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