"""
================================================================================
NIDS-ML - Feature Validation System
================================================================================

Valida che le feature estratte dallo sniffer siano allineate con quelle
del dataset CIC-IDS2017 originale.

PERCHE' E' IMPORTANTE:
---------------------
Il modello è stato trainato su feature estratte da CICFlowMeter (Java).
Se il nostro estrattore (Python) produce feature diverse, le predizioni
saranno SBAGLIATE anche se il modello è buono.

TIPI DI VALIDAZIONE:
-------------------
1. STATISTICAL: Confronta distribuzioni feature tra sniffer e CSV
2. FLOW-LEVEL: Confronta singoli flussi (richiede match IP)
3. COVERAGE: Verifica che tutte le feature richieste siano presenti

USO:
----
from src.sniffer.validator import FeatureValidator, ValidationMode

validator = FeatureValidator()
report = validator.validate_csv_compatibility(csv_path, pcap_path)
report.print_summary()

================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .features import FEATURE_NAMES, CRITICAL_FEATURES, FeatureExtractor


# ==============================================================================
# VALIDATION MODES
# ==============================================================================

class ValidationMode(Enum):
    """Validation modes for feature comparison."""
    STATISTICAL = auto()   # Compare feature distributions
    FLOW_LEVEL = auto()    # Compare individual flows (requires IP match)
    CSV_COVERAGE = auto()  # Check if CSV has required features


# ==============================================================================
# CONFIGURAZIONE TOLLERANZE
# ==============================================================================

# Tolleranze per confronto feature
TOLERANCES = {
    # Feature temporali (IAT, Duration) - alta variabilità
    'temporal': {
        'relative': 0.30,  # 30% differenza relativa accettabile
        'features': [
            'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
    },
    # Feature di conteggio - bassa variabilità
    'count': {
        'relative': 0.05,  # 5% differenza relativa
        'features': [
            'Total Fwd Packets', 'Total Backward Packets',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
        ]
    },
    # Feature di bytes/dimensione - media variabilità
    'size': {
        'relative': 0.15,  # 15% differenza relativa
        'features': [
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
        ]
    },
    # Feature rate - alta variabilità (dipende da duration)
    'rate': {
        'relative': 0.50,  # 50% differenza relativa
        'features': [
            'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s'
        ]
    }
}


# ==============================================================================
# VALIDATION REPORT
# ==============================================================================

@dataclass
class FeatureComparisonResult:
    """Risultato confronto singola feature."""
    feature_name: str
    csv_value: float
    sniffer_value: float
    difference: float
    relative_diff: float
    tolerance: float
    passed: bool
    category: str


@dataclass
class ValidationReport:
    """Report completo validazione."""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_type: str = ""
    
    # Statistiche generali
    total_features: int = 0
    features_passed: int = 0
    features_failed: int = 0
    
    # Dettagli per feature
    feature_results: List[FeatureComparisonResult] = field(default_factory=list)
    
    # Feature mancanti
    missing_features: List[str] = field(default_factory=list)
    
    # Warning e errori
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metadati
    csv_path: str = ""
    pcap_path: str = ""
    flows_compared: int = 0
    
    @property
    def pass_rate(self) -> float:
        """Percentuale feature che passano."""
        if self.total_features == 0:
            return 0.0
        return self.features_passed / self.total_features * 100
    
    @property
    def is_valid(self) -> bool:
        """True se validazione OK (>80% pass rate e nessun errore critico)."""
        return self.pass_rate >= 80 and len(self.errors) == 0
    
    def get_failed_features(self) -> List[FeatureComparisonResult]:
        """Restituisce feature che non passano."""
        return [r for r in self.feature_results if not r.passed]
    
    def get_critical_failures(self) -> List[FeatureComparisonResult]:
        """Restituisce feature critiche che non passano."""
        failed = self.get_failed_features()
        return [r for r in failed if r.feature_name in CRITICAL_FEATURES]
    
    def print_summary(self) -> None:
        """Stampa riepilogo a console."""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT - {self.validation_type}")
        print(f"{'='*60}")
        
        print(f"\nTimestamp: {self.timestamp}")
        if self.csv_path:
            print(f"CSV: {self.csv_path}")
        if self.pcap_path:
            print(f"PCAP: {self.pcap_path}")
        
        print(f"\n{'─'*40}")
        print(f"RISULTATI")
        print(f"{'─'*40}")
        print(f"Feature totali:    {self.total_features}")
        print(f"Feature passate:   {self.features_passed} ({self.pass_rate:.1f}%)")
        print(f"Feature fallite:   {self.features_failed}")
        
        if self.missing_features:
            print(f"\nFeature mancanti ({len(self.missing_features)}):")
            for f in self.missing_features[:10]:
                print(f"  - {f}")
            if len(self.missing_features) > 10:
                print(f"  ... e altre {len(self.missing_features) - 10}")
        
        failed = self.get_failed_features()
        if failed:
            print(f"\n{'─'*40}")
            print(f"FEATURE NON ALLINEATE")
            print(f"{'─'*40}")
            print(f"{'Feature':<35} {'CSV':>12} {'Sniffer':>12} {'Diff%':>8}")
            print("-" * 70)
            
            for r in sorted(failed, key=lambda x: abs(x.relative_diff), reverse=True)[:20]:
                print(f"{r.feature_name:<35} {r.csv_value:>12.2f} {r.sniffer_value:>12.2f} {r.relative_diff*100:>7.1f}%")
        
        critical = self.get_critical_failures()
        if critical:
            print(f"\n⚠️  ATTENZIONE: {len(critical)} feature CRITICHE non allineate!")
            for r in critical:
                print(f"  - {r.feature_name}: diff {r.relative_diff*100:.1f}%")
        
        if self.warnings:
            print(f"\nWarning ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                print(f"  ⚠️  {w}")
        
        if self.errors:
            print(f"\nErrori ({len(self.errors)}):")
            for e in self.errors:
                print(f"  ❌ {e}")
        
        print(f"\n{'='*60}")
        if self.is_valid:
            print("✅ VALIDAZIONE OK - Lo sniffer è allineato con il dataset")
        else:
            print("❌ VALIDAZIONE FALLITA - Controllare le feature non allineate")
        print(f"{'='*60}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte in dizionario per JSON."""
        return {
            'timestamp': self.timestamp,
            'validation_type': self.validation_type,
            'total_features': self.total_features,
            'features_passed': self.features_passed,
            'features_failed': self.features_failed,
            'pass_rate': self.pass_rate,
            'is_valid': self.is_valid,
            'missing_features': self.missing_features,
            'warnings': self.warnings,
            'errors': self.errors,
            'csv_path': self.csv_path,
            'pcap_path': self.pcap_path,
            'flows_compared': self.flows_compared,
            'failed_features': [
                {
                    'name': r.feature_name,
                    'csv': r.csv_value,
                    'sniffer': r.sniffer_value,
                    'diff_pct': r.relative_diff * 100
                }
                for r in self.get_failed_features()
            ]
        }
    
    def save(self, path: Path) -> None:
        """Salva report su file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_json(self, path: str) -> None:
        """Alias for save() that accepts string path."""
        self.save(Path(path))


# ==============================================================================
# FEATURE VALIDATOR
# ==============================================================================

class FeatureValidator:
    """
    Valida che le feature estratte dallo sniffer siano allineate
    con quelle del dataset CIC-IDS2017.
    """
    
    def __init__(self, tolerances: Dict = None, artifacts_dir: str = None):
        """
        Args:
            tolerances: Override tolleranze default
            artifacts_dir: Path to artifacts directory (for loading expected features)
        """
        self.tolerances = tolerances or TOLERANCES
        self.extractor = FeatureExtractor(validate=True)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        
        # Load expected features from artifacts if available
        self.expected_features = None
        if self.artifacts_dir:
            features_path = self.artifacts_dir / 'selected_features.json'
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.expected_features = json.load(f)
            else:
                # Fallback to scaler_columns
                scaler_cols_path = self.artifacts_dir / 'scaler_columns.json'
                if scaler_cols_path.exists():
                    with open(scaler_cols_path, 'r') as f:
                        self.expected_features = json.load(f)
    
    def _get_tolerance(self, feature_name: str) -> float:
        """Ottiene tolleranza per una feature specifica."""
        for category, config in self.tolerances.items():
            if feature_name in config.get('features', []):
                return config['relative']
        return 0.20  # Default 20%
    
    def _get_category(self, feature_name: str) -> str:
        """Ottiene categoria di una feature."""
        for category, config in self.tolerances.items():
            if feature_name in config.get('features', []):
                return category
        return 'other'
    
    def _compare_values(self, csv_val: float, sniffer_val: float, 
                        feature_name: str) -> FeatureComparisonResult:
        """Confronta due valori di feature."""
        diff = abs(sniffer_val - csv_val)
        
        if csv_val != 0:
            rel_diff = diff / abs(csv_val)
        elif sniffer_val != 0:
            rel_diff = 1.0  # 100% se csv è 0 ma sniffer no
        else:
            rel_diff = 0.0  # Entrambi 0
        
        tolerance = self._get_tolerance(feature_name)
        passed = rel_diff <= tolerance
        category = self._get_category(feature_name)
        
        return FeatureComparisonResult(
            feature_name=feature_name,
            csv_value=csv_val,
            sniffer_value=sniffer_val,
            difference=diff,
            relative_diff=rel_diff,
            tolerance=tolerance,
            passed=passed,
            category=category
        )
    
    def validate_statistical(self, 
                            csv_path: Path,
                            sniffer_features_list: List[Dict[str, float]],
                            sample_size: int = 5000) -> ValidationReport:
        """
        Validazione statistica: confronta distribuzioni feature.
        
        Non richiede match esatto dei flussi, confronta statistiche aggregate.
        
        Args:
            csv_path: Path al CSV CIC-IDS2017
            sniffer_features_list: Lista di dict feature estratte dallo sniffer
            sample_size: Righe CSV da campionare
        
        Returns:
            ValidationReport
        """
        report = ValidationReport(
            validation_type='STATISTICAL',
            csv_path=str(csv_path)
        )
        
        # Carica CSV
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = df.columns.str.strip()
        except Exception as e:
            report.errors.append(f"Errore lettura CSV: {e}")
            return report
        
        # Sample se necessario
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Converti sniffer features in DataFrame
        if not sniffer_features_list:
            report.errors.append("Nessuna feature sniffer fornita")
            return report
        
        df_sniffer = pd.DataFrame(sniffer_features_list)
        
        # Confronta feature per feature
        for feature_name in FEATURE_NAMES:
            # Trova colonna nel CSV (gestisce spazi)
            csv_col = None
            for col in df.columns:
                if col.strip().lower() == feature_name.lower():
                    csv_col = col
                    break
            
            if csv_col is None:
                report.missing_features.append(feature_name)
                continue
            
            if feature_name not in df_sniffer.columns:
                report.missing_features.append(f"{feature_name} (sniffer)")
                continue
            
            # Calcola medie
            csv_mean = df[csv_col].replace([np.inf, -np.inf], np.nan).mean()
            sniffer_mean = df_sniffer[feature_name].mean()
            
            if pd.isna(csv_mean):
                report.warnings.append(f"{feature_name}: CSV ha tutti NaN")
                csv_mean = 0.0
            
            result = self._compare_values(csv_mean, sniffer_mean, feature_name)
            report.feature_results.append(result)
            report.total_features += 1
            
            if result.passed:
                report.features_passed += 1
            else:
                report.features_failed += 1
        
        report.flows_compared = len(df_sniffer)
        
        return report
    
    def validate_from_csv_only(self, csv_path: Path, 
                               selected_features: List[str] = None) -> ValidationReport:
        """
        Valida che il CSV abbia tutte le feature necessarie per il modello.
        
        Usa questa validazione PRIMA di testare con PCAP.
        
        Args:
            csv_path: Path al CSV CIC-IDS2017
            selected_features: Feature richieste dal modello
        
        Returns:
            ValidationReport
        """
        report = ValidationReport(
            validation_type='CSV_COVERAGE',
            csv_path=str(csv_path)
        )
        
        if selected_features is None:
            selected_features = FEATURE_NAMES
        
        # Carica CSV
        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=100)
            df.columns = df.columns.str.strip()
        except Exception as e:
            report.errors.append(f"Errore lettura CSV: {e}")
            return report
        
        csv_cols_lower = {col.lower(): col for col in df.columns}
        
        for feature_name in selected_features:
            feature_lower = feature_name.lower()
            
            if feature_lower in csv_cols_lower:
                report.features_passed += 1
            else:
                report.missing_features.append(feature_name)
                report.features_failed += 1
            
            report.total_features += 1
        
        # Check label
        label_found = False
        for col in df.columns:
            if 'label' in col.lower():
                label_found = True
                break
        
        if not label_found:
            report.warnings.append("Colonna 'Label' non trovata nel CSV")
        
        return report
    
    def validate_single_flow(self, 
                            csv_row: pd.Series,
                            sniffer_features: Dict[str, float]) -> ValidationReport:
        """
        Valida singolo flusso confrontando riga CSV con feature sniffer.
        
        Args:
            csv_row: Riga del CSV (pd.Series)
            sniffer_features: Dict feature estratte dallo sniffer
        
        Returns:
            ValidationReport
        """
        report = ValidationReport(
            validation_type='SINGLE_FLOW',
            flows_compared=1
        )
        
        for feature_name in FEATURE_NAMES:
            # Trova valore CSV
            csv_val = None
            for col in csv_row.index:
                if col.strip().lower() == feature_name.lower():
                    csv_val = csv_row[col]
                    break
            
            if csv_val is None or pd.isna(csv_val):
                report.missing_features.append(f"{feature_name} (CSV)")
                continue
            
            # Trova valore sniffer
            sniffer_val = sniffer_features.get(feature_name)
            if sniffer_val is None:
                report.missing_features.append(f"{feature_name} (sniffer)")
                continue
            
            result = self._compare_values(float(csv_val), float(sniffer_val), feature_name)
            report.feature_results.append(result)
            report.total_features += 1
            
            if result.passed:
                report.features_passed += 1
            else:
                report.features_failed += 1
        
        return report
    
    def validate_csv_coverage(self, df: pd.DataFrame, 
                              selected_features: List[str] = None) -> ValidationReport:
        """
        Valida che un DataFrame abbia tutte le feature necessarie.
        
        Alias più user-friendly per la validazione CSV.
        
        Args:
            df: DataFrame da validare
            selected_features: Feature richieste (default: FEATURE_NAMES)
        
        Returns:
            ValidationReport
        """
        report = ValidationReport(
            validation_type='CSV_COVERAGE'
        )
        
        # Use expected features from artifacts if available
        if selected_features is None:
            if self.expected_features:
                selected_features = self.expected_features
            else:
                selected_features = FEATURE_NAMES
        
        # Normalize column names
        df_cols_normalized = {col.strip().lower(): col for col in df.columns}
        
        for feature_name in selected_features:
            feature_lower = feature_name.strip().lower()
            
            if feature_lower in df_cols_normalized:
                report.features_passed += 1
            else:
                report.missing_features.append(feature_name)
                report.features_failed += 1
            
            report.total_features += 1
        
        # Check for critical features
        critical_missing = [f for f in report.missing_features if f in CRITICAL_FEATURES]
        if critical_missing:
            report.errors.append(f"Critical features missing: {', '.join(critical_missing)}")
        
        # Check label column
        label_found = False
        for col in df.columns:
            if 'label' in col.lower():
                label_found = True
                break
        
        if not label_found:
            report.warnings.append("Label column not found in CSV")
        
        return report


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def quick_validate_csv(csv_path: Path, selected_features: List[str] = None) -> bool:
    """
    Quick check se un CSV è compatibile.
    
    Returns:
        True se il CSV ha almeno 80% delle feature
    """
    validator = FeatureValidator()
    report = validator.validate_from_csv_only(csv_path, selected_features)
    return report.pass_rate >= 80


def find_column(columns, *possible_names: str) -> Optional[str]:
    """
    Trova una colonna cercando tra vari nomi possibili.
    Gestisce spazi iniziali/finali nei nomi colonne.
    
    Args:
        columns: Either a pd.DataFrame, a list of column names, or df.columns
        possible_names: Names to search for
        
    Returns:
        The actual column name if found, None otherwise
    """
    # Handle different input types
    if isinstance(columns, pd.DataFrame):
        col_list = columns.columns.tolist()
    elif hasattr(columns, 'tolist'):
        col_list = columns.tolist()
    else:
        col_list = list(columns)
    
    col_map = {col.strip().lower(): col for col in col_list}
    
    for name in possible_names:
        name_lower = name.strip().lower()
        if name_lower in col_map:
            return col_map[name_lower]
    
    return None