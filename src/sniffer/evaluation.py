"""
NIDS-ML Sniffer - Evaluation Module

Valuta il modello su CSV CIC-IDS2017.

Pipeline corretta:
1. Carica CSV con ~77 feature
2. Filtra alle scaler_columns (44 feature dopo statistical preprocessing)
3. Scala con RobustScaler
4. Seleziona le 30 feature finali (indici in ordine di importanza)
5. Predici
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, 
    recall_score, accuracy_score
)


logger = logging.getLogger(__name__)


def find_column(columns: List[str], target: str) -> Optional[str]:
    """Trova colonna con matching case-insensitive."""
    target_lower = target.strip().lower()
    for col in columns:
        if col.strip().lower() == target_lower:
            return col
        # Varianti
        col_lower = col.strip().lower()
        if col_lower.replace(' ', '_') == target_lower.replace(' ', '_'):
            return col
        if col_lower.replace('_', ' ') == target_lower.replace('_', ' '):
            return col
    return None


@dataclass
class EvaluationResult:
    """Risultato valutazione."""
    samples: int
    benign: int
    attack: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    tp: int
    tn: int
    fp: int
    fn: int
    latency_ms: float = 0.0
    
    def print_summary(self):
        print("\n" + "="*60)
        print("RISULTATI VALUTAZIONE")
        print("="*60)
        print(f"Campioni: {self.samples:,} (Benign: {self.benign:,}, Attack: {self.attack:,})")
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1:        {self.f1:.4f}")
        print(f"FPR:       {self.fpr:.4f}")
        print(f"TP: {self.tp:,} | TN: {self.tn:,} | FP: {self.fp:,} | FN: {self.fn:,}")
        if self.latency_ms > 0:
            print(f"Latency:   {self.latency_ms:.4f} ms/sample")
        print("="*60)
    
    def to_dict(self) -> Dict:
        return {
            'samples': self.samples, 'benign': self.benign, 'attack': self.attack,
            'accuracy': self.accuracy, 'precision': self.precision, 
            'recall': self.recall, 'f1': self.f1, 'fpr': self.fpr,
            'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn,
            'latency_ms': self.latency_ms
        }


class SnifferEvaluator:
    """Valutatore per modelli NIDS su CSV."""
    
    def __init__(
        self, 
        model_dir: str = 'models/best_model',
        artifacts_dir: str = 'artifacts',
        label_column: str = 'Label'
    ):
        self.model_dir = Path(model_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.label_column = label_column
        self.logger = logging.getLogger('sniffer.evaluator')
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Carica modello e artifacts."""
        self.logger.info("Caricamento artifacts...")
        
        # 1. Modello
        model_path = self.model_dir / 'model_binary.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato in {self.model_dir}")
        
        self.model = joblib.load(model_path)
        self.logger.info(f"Modello: {type(self.model).__name__}")
        
        # 2. Scaler
        scaler_path = self.artifacts_dir / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if self.scaler:
            self.logger.info(f"Scaler: {type(self.scaler).__name__}")
        
        # 3. Feature selector (opzionale)
        selector_path = self.artifacts_dir / 'feature_selector.pkl'
        self.selector = joblib.load(selector_path) if selector_path.exists() else None
        
        # 4. Scaler columns (feature che lo scaler si aspetta - 44)
        scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
        if scaler_cols_path.exists():
            with open(scaler_cols_path, 'r') as f:
                self.scaler_columns = json.load(f)
            self.logger.info(f"Scaler columns: {len(self.scaler_columns)} features")
        else:
            self.scaler_columns = None
        
        # 5. Selected features (feature finali - 30, in ordine importanza)
        features_path = self.artifacts_dir / 'selected_features.json'
        if not features_path.exists():
            features_path = self.model_dir / 'features_binary.json'
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            self.logger.info(f"Selected features: {len(self.selected_features)} features")
        else:
            self.selected_features = None
        
        # 6. Determina strategia pipeline
        self._selected_indices = None
        self.use_selector = False
        
        if self.selector is not None and self.scaler_columns is not None:
            # Caso migliore: abbiamo selector pickle
            self.features_to_load = self.scaler_columns
            self.use_selector = True
            self.logger.info(f"Pipeline: {len(self.scaler_columns)} -> scale -> selector -> predict")
            
        elif self.selector is None and self.scaler_columns is not None and self.selected_features is not None:
            # No selector pickle, ma abbiamo entrambe le liste
            # Creiamo selezione per indice
            self.logger.info("Creazione index-based selector...")
            
            scaler_cols_lower = {col.strip().lower(): i for i, col in enumerate(self.scaler_columns)}
            selected_indices = []
            for feat in self.selected_features:
                feat_lower = feat.strip().lower()
                if feat_lower in scaler_cols_lower:
                    selected_indices.append(scaler_cols_lower[feat_lower])
                else:
                    self.logger.warning(f"Feature '{feat}' non trovata in scaler_columns")
            
            if len(selected_indices) == len(self.selected_features):
                # CRITICO: NON ordinare! L'ordine in selected_features.json è quello che il modello si aspetta
                self._selected_indices = selected_indices
                self.features_to_load = self.scaler_columns
                self.use_selector = True
                self.logger.info(f"Pipeline: {len(self.scaler_columns)} -> scale -> select[{len(self._selected_indices)}] -> predict")
            else:
                # Fallback
                self.features_to_load = self.selected_features
                self.use_selector = False
                self.logger.warning("Fallback: uso selected_features direttamente")
                
        elif self.selected_features is not None:
            # No scaler_columns: assumiamo scaler fittato su selected_features
            self.features_to_load = self.selected_features
            self.use_selector = False
            self.logger.info(f"Pipeline: {len(self.selected_features)} -> scale -> predict")
        else:
            raise ValueError("Nessun artifact feature trovato!")
        
        self.logger.info("Artifacts caricati")
    
    def _prepare_csv_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepara dati CSV per predizione."""
        # Trova colonna label
        label_col = find_column(df.columns.tolist(), self.label_column)
        if not label_col:
            raise ValueError(f"Colonna label '{self.label_column}' non trovata")
        
        # Converti label a binario
        labels = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        labels = labels.values
        
        # Mappa colonne CSV alle feature richieste
        feature_cols = self.features_to_load
        col_mapping = {}
        for target_col in feature_cols:
            found = find_column(df.columns.tolist(), target_col)
            col_mapping[target_col] = found
        
        # Crea DataFrame features
        features_data = {}
        missing = 0
        for target_col, source_col in col_mapping.items():
            if source_col and source_col in df.columns:
                features_data[target_col] = df[source_col].values
            else:
                features_data[target_col] = np.zeros(len(df))
                missing += 1
        
        if missing > 0:
            self.logger.warning(f"{missing} feature non trovate, impostate a 0")
        
        features_df = pd.DataFrame(features_data)
        features_df = features_df[feature_cols]  # Ordine corretto
        
        # Gestisci NaN/Inf
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df, labels
    
    def _predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Esegue predizione su batch."""
        # 1. Scala
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        # 2. Seleziona feature
        if self.use_selector:
            if self.selector is not None:
                features_selected = self.selector.transform(features_scaled)
            elif self._selected_indices is not None:
                features_selected = features_scaled[:, self._selected_indices]
            else:
                features_selected = features_scaled
        else:
            features_selected = features_scaled
        
        # 3. Predici
        return self.model.predict(features_selected)
    
    def evaluate_csv(
        self, 
        csv_path: str, 
        sample_size: Optional[int] = None,
        batch_size: int = 10000,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Valuta modello su file CSV.
        
        Args:
            csv_path: Path al CSV
            sample_size: Numero campioni (None = tutti)
            batch_size: Dimensione batch
            verbose: Output verboso
        
        Returns:
            EvaluationResult con metriche
        """
        self.logger.info(f"Valutazione: {csv_path}")
        
        # Carica CSV
        df = pd.read_csv(csv_path, low_memory=False)
        self.logger.info(f"Righe caricate: {len(df):,}")
        
        # Sampling
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Campionate: {sample_size:,}")
        
        # Prepara dati
        features_df, y_true = self._prepare_csv_data(df)
        
        # Predici con timing
        start_time = time.perf_counter()
        
        all_predictions = []
        for i in range(0, len(features_df), batch_size):
            batch = features_df.iloc[i:i+batch_size]
            preds = self._predict_batch(batch)
            all_predictions.extend(preds)
            
            if verbose and (i + batch_size) % 50000 == 0:
                self.logger.info(f"Processed {i+batch_size:,}/{len(features_df):,}")
        
        elapsed = time.perf_counter() - start_time
        y_pred = np.array(all_predictions)
        
        # Calcola metriche
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        result = EvaluationResult(
            samples=len(y_true),
            benign=int((y_true == 0).sum()),
            attack=int((y_true == 1).sum()),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            fpr=float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
            latency_ms=float(elapsed / len(y_true) * 1000)
        )
        
        if verbose:
            result.print_summary()
        
        return result


def quick_evaluate(
    csv_path: str,
    model_dir: str = 'models/best_model',
    artifacts_dir: str = 'artifacts',
    sample_size: Optional[int] = None
) -> EvaluationResult:
    """Funzione rapida per valutazione."""
    evaluator = SnifferEvaluator(model_dir=model_dir, artifacts_dir=artifacts_dir)
    return evaluator.evaluate_csv(csv_path, sample_size=sample_size)