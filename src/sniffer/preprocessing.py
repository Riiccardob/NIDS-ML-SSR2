"""
NIDS-ML Sniffer - Preprocessing Module per Pipeline v2

Gestisce il preprocessing delle feature per allinearsi al pipeline di training v2:
1. Estrazione 77 feature CIC-IDS2017 → filtra a 44 (scaler_columns.json)
2. RobustScaler su 44 feature
3. Selezione 30 feature per indice (selected_features.json, ordine importanza)

CRITICO: L'ordine delle feature deve essere ESATTAMENTE quello che il modello si aspetta!
"""

import json
import logging
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
    scaler_columns: List[str]  # 44 feature dopo statistical preprocessing
    selected_features: List[str]  # 30 feature selezionate (ordine importanza)
    selected_indices: List[int]  # Indici in scaler_columns (preserva ordine!)
    statistical_info: Optional[Dict] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if len(self.selected_indices) != len(self.selected_features):
            raise ValueError(f"Mismatch: {len(self.selected_indices)} indices vs {len(self.selected_features)} features")


def load_pipeline_artifacts(artifacts_dir: str = 'artifacts', model_dir: Optional[str] = None) -> PipelineArtifacts:
    """Carica tutti gli artifacts necessari per l'inference."""
    artifacts_path = Path(artifacts_dir)
    
    # 1. Scaler
    scaler_path = artifacts_path / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler non trovato: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info(f"Caricato scaler: {type(scaler).__name__}")
    
    # 2. Scaler columns (44 feature)
    scaler_columns_path = artifacts_path / 'scaler_columns.json'
    if not scaler_columns_path.exists():
        # Fallback per artifacts vecchi
        if hasattr(scaler, 'feature_names_in_'):
            scaler_columns = list(scaler.feature_names_in_)
        else:
            raise FileNotFoundError(f"scaler_columns.json non trovato: {scaler_columns_path}")
    else:
        with open(scaler_columns_path, 'r') as f:
            scaler_columns = json.load(f)
    logger.info(f"Scaler columns: {len(scaler_columns)} feature")
    
    # 3. Selected features (30 feature in ordine di importanza)
    selected_features = None
    search_paths = [
        artifacts_path / 'selected_features.json',
        Path(model_dir) / 'features_binary.json' if model_dir else None,
    ]
    for path in search_paths:
        if path and path.exists():
            with open(path, 'r') as f:
                selected_features = json.load(f)
            break
    
    if selected_features is None:
        raise FileNotFoundError("selected_features.json non trovato")
    logger.info(f"Selected features: {len(selected_features)}")
    
    # 4. Calcola indici (CRITICO: preserva ordine di selected_features!)
    scaler_cols_map = {col.strip().lower(): idx for idx, col in enumerate(scaler_columns)}
    selected_indices = []
    for feat in selected_features:
        feat_lower = feat.strip().lower()
        if feat_lower in scaler_cols_map:
            selected_indices.append(scaler_cols_map[feat_lower])
        else:
            raise ValueError(f"Feature '{feat}' non trovata in scaler_columns")
    
    # 5. Info opzionali
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
    """Pipeline di preprocessing per inference."""
    
    def __init__(self, artifacts: PipelineArtifacts):
        self.artifacts = artifacts
        self.scaler = artifacts.scaler
        self.scaler_columns = artifacts.scaler_columns
        self.selected_features = artifacts.selected_features
        self.selected_indices = artifacts.selected_indices
        self._scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(self.scaler_columns)}
        
        logger.info(f"Pipeline: {len(self.scaler_columns)} → scale → select {len(self.selected_features)} features")
    
    def transform(self, features: Dict[str, float]) -> np.ndarray:
        """Trasforma dict feature in array per predizione. Returns shape (1, n_features)."""
        # Normalizza nomi
        features_lower = {k.strip().lower(): v for k, v in features.items()}
        
        # Estrai valori nell'ordine di scaler_columns
        values = []
        for col in self.scaler_columns:
            col_lower = col.strip().lower()
            if col_lower in features_lower:
                values.append(features_lower[col_lower])
            else:
                # Prova varianti
                found = False
                for variant in [col_lower.replace(' ', '_'), col_lower.replace('_', ' ')]:
                    if variant in features_lower:
                        values.append(features_lower[variant])
                        found = True
                        break
                if not found:
                    values.append(0.0)
        
        # Scale
        arr = np.array([values])
        scaled = self.scaler.transform(arr)
        
        # Select features by index (preserva ordine!)
        return scaled[:, self.selected_indices]
    
    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Trasforma DataFrame. Returns shape (n_samples, n_features)."""
        col_map = {c.strip().lower(): c for c in df.columns}
        
        # Seleziona colonne nell'ordine di scaler_columns
        cols_to_use = []
        for scaler_col in self.scaler_columns:
            scaler_col_lower = scaler_col.strip().lower()
            if scaler_col_lower in col_map:
                cols_to_use.append(col_map[scaler_col_lower])
            else:
                # Cerca varianti
                found = False
                for variant in [scaler_col_lower.replace(' ', '_'), scaler_col_lower.replace('_', ' ')]:
                    if variant in col_map:
                        cols_to_use.append(col_map[variant])
                        found = True
                        break
                if not found:
                    df[scaler_col] = 0.0
                    cols_to_use.append(scaler_col)
        
        df_ordered = df[cols_to_use]
        scaled = self.scaler.transform(df_ordered.values)
        return scaled[:, self.selected_indices]
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'scaler_type': type(self.scaler).__name__,
            'n_scaler_columns': len(self.scaler_columns),
            'n_selected_features': len(self.selected_features),
            'top_5_features': self.selected_features[:5],
        }


def create_inference_pipeline(artifacts_dir: str = 'artifacts', model_dir: Optional[str] = None) -> InferencePipeline:
    """Factory function."""
    return InferencePipeline(load_pipeline_artifacts(artifacts_dir, model_dir))


def validate_artifacts_consistency(artifacts_dir: str = 'artifacts') -> Dict[str, Any]:
    """Valida coerenza artifacts."""
    results = {'valid': True, 'errors': [], 'warnings': [], 'info': {}}
    
    try:
        artifacts = load_pipeline_artifacts(artifacts_dir)
        results['info'] = {
            'scaler_type': type(artifacts.scaler).__name__,
            'n_scaler_columns': len(artifacts.scaler_columns),
            'n_selected_features': len(artifacts.selected_features),
            'checksum': artifacts.checksum,
        }
        
        # Verifica indici
        max_idx = max(artifacts.selected_indices) if artifacts.selected_indices else 0
        if max_idx >= len(artifacts.scaler_columns):
            results['errors'].append(f"Index {max_idx} out of range")
            results['valid'] = False
            
    except Exception as e:
        results['valid'] = False
        results['errors'].append(str(e))
    
    return results