# """
# ================================================================================
# NIDS-ML - Modulo Evaluation
# ================================================================================

# Valutazione modelli su test set con metriche complete e visualizzazioni.

# GUIDA PARAMETRI:
# ----------------
#     python src/evaluation.py [opzioni]

# Opzioni:
#     --model-path PATH     Path al modello .pkl (obbligatorio)
#     --task STR            'binary' o 'multiclass' (default: binary)
#     --output-dir PATH     Directory per report (default: reports/)
#     --n-jobs INT          Core CPU (default: auto)

# ESEMPI:
# -------
# # Valuta Random Forest binario
# python src/evaluation.py --model-path models/random_forest/model_binary.pkl

# # Valuta XGBoost multiclasse
# python src/evaluation.py --model-path models/xgboost/model_multiclass.pkl --task multiclass

# # Valuta tutti i modelli binari
# python src/evaluation.py --model-path models/random_forest/model_binary.pkl
# python src/evaluation.py --model-path models/xgboost/model_binary.pkl
# python src/evaluation.py --model-path models/lightgbm/model_binary.pkl

# ================================================================================
# """

# # ==============================================================================
# # SETUP LIMITI CPU
# # ==============================================================================
# import sys
# import os
# import argparse
# from pathlib import Path

# def _get_arg(name, default=None, arg_type=str):
#     for i, arg in enumerate(sys.argv):
#         if arg == f'--{name}' and i + 1 < len(sys.argv):
#             try:
#                 return arg_type(sys.argv[i + 1])
#             except ValueError:
#                 return default
#     return default

# _n_jobs_arg = _get_arg('n-jobs', None, int)
# _n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

# os.environ['OMP_NUM_THREADS'] = str(_n_cores)
# os.environ['MKL_NUM_THREADS'] = str(_n_cores)
# os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)

# import psutil
# try:
#     p = psutil.Process()
#     p.cpu_affinity(list(range(_n_cores)))
#     p.nice(10)
# except Exception:
#     pass

# ROOT_DIR = Path(__file__).parent.parent
# sys.path.insert(0, str(ROOT_DIR))

# import pandas as pd
# import numpy as np
# from typing import Dict, Any, Optional, Tuple
# import joblib
# import json
# from datetime import datetime
# import warnings

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
#     average_precision_score
# )

# import matplotlib
# matplotlib.use('Agg')  # Backend non-interattivo
# import matplotlib.pyplot as plt
# import seaborn as sns

# from src.utils import get_logger, get_project_root, suppress_warnings
# from src.preprocessing import load_processed_data
# from src.feature_engineering import (
#     load_artifacts,
#     get_feature_columns,
#     prepare_xy,
#     transform_data,
#     apply_feature_selection
# )
# from src.timing import TimingLogger

# suppress_warnings()
# logger = get_logger(__name__)


# # ==============================================================================
# # CARICAMENTO MODELLO E DATI
# # ==============================================================================

# def load_model(model_path: Path):
#     """Carica modello salvato."""
#     if not model_path.exists():
#         raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
#     model = joblib.load(model_path)
#     logger.info(f"Modello caricato: {model_path}")
    
#     return model


# def prepare_test_data(task: str = 'binary', 
#                       model_features: list[str] = None) -> Tuple[pd.DataFrame, pd.Series, dict]:
#     """
#     Prepara dati di test applicando scaler e feature selection.
    
#     Args:
#         task: 'binary' o 'multiclass'
#         model_features: Lista feature del modello (se None usa artifacts globali)
    
#     Returns:
#         Tuple (X_test, y_test, mappings) pronti per predizione
#     """
#     # Carica dati
#     _, _, test, mappings = load_processed_data()
    
#     # Carica artifacts
#     scaler, selected_features, _, _ = load_artifacts()
    
#     # Usa feature specifiche del modello se fornite
#     if model_features is not None:
#         selected_features = model_features
    
#     # Prepara X e y
#     label_col = 'Label_Binary' if task == 'binary' else 'Label_Multiclass'
#     feature_cols = get_feature_columns(test)
    
#     X_test, y_test = prepare_xy(test, label_col, feature_cols)
    
#     # Applica trasformazioni
#     X_test_scaled = transform_data(X_test, scaler)
#     X_test_final = apply_feature_selection(X_test_scaled, selected_features)
    
#     logger.info(f"Test set preparato: {X_test_final.shape}")
    
#     return X_test_final, y_test, mappings


# # ==============================================================================
# # CALCOLO METRICHE
# # ==============================================================================

# def compute_metrics(y_true: np.ndarray,
#                     y_pred: np.ndarray,
#                     y_prob: Optional[np.ndarray] = None,
#                     task: str = 'binary') -> Dict[str, float]:
#     """
#     Calcola metriche complete di classificazione.
    
#     Args:
#         y_true: Label reali
#         y_pred: Predizioni
#         y_prob: Probabilita predette (per ROC-AUC)
#         task: 'binary' o 'multiclass'
    
#     Returns:
#         Dizionario con tutte le metriche
#     """
#     metrics = {}
    
#     # Metriche base
#     metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
#     if task == 'binary':
#         metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
#         metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
#         metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        
#         # Specificity (True Negative Rate)
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#         metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
#         metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
#         metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
#         # ROC-AUC se abbiamo probabilita
#         if y_prob is not None:
#             try:
#                 metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
#                 metrics['average_precision'] = float(average_precision_score(y_true, y_prob))
#             except Exception:
#                 pass
#     else:
#         # Multiclass
#         metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
#         metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
#         metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
#         metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
#         metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
#         metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
#     return metrics


# def analyze_errors(y_true: np.ndarray,
#                    y_pred: np.ndarray,
#                    X_test: pd.DataFrame,
#                    task: str = 'binary') -> Dict[str, Any]:
#     """
#     Analizza errori di classificazione.
    
#     Returns:
#         Dizionario con statistiche sugli errori
#     """
#     errors = {}
    
#     # Indici errori
#     wrong_mask = y_true != y_pred
#     n_errors = wrong_mask.sum()
    
#     errors['total_errors'] = int(n_errors)
#     errors['error_rate'] = float(n_errors / len(y_true))
    
#     if task == 'binary':
#         # False Positives (predetto 1, era 0)
#         fp_mask = (y_pred == 1) & (y_true == 0)
#         errors['false_positives'] = int(fp_mask.sum())
        
#         # False Negatives (predetto 0, era 1)
#         fn_mask = (y_pred == 0) & (y_true == 1)
#         errors['false_negatives'] = int(fn_mask.sum())
        
#         # Statistiche feature per FP e FN
#         if fp_mask.sum() > 0:
#             fp_features = X_test[fp_mask].describe().to_dict()
#         else:
#             fp_features = {}
        
#         if fn_mask.sum() > 0:
#             fn_features = X_test[fn_mask].describe().to_dict()
#         else:
#             fn_features = {}
    
#     return errors


# # ==============================================================================
# # VISUALIZZAZIONI
# # ==============================================================================

# def plot_confusion_matrix(y_true: np.ndarray,
#                           y_pred: np.ndarray,
#                           labels: list = None,
#                           output_path: Path = None,
#                           title: str = "Confusion Matrix") -> None:
#     """Genera e salva confusion matrix."""
#     cm = confusion_matrix(y_true, y_pred)
    
#     plt.figure(figsize=(10, 8))
    
#     if labels is None:
#         labels = [str(i) for i in range(cm.shape[0])]
    
#     # Normalizza per visualizzazione percentuale
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
#     # Heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=labels, yticklabels=labels)
    
#     plt.title(title)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         logger.info(f"Confusion matrix salvata: {output_path}")
    
#     plt.close()


# def plot_roc_curve(y_true: np.ndarray,
#                    y_prob: np.ndarray,
#                    output_path: Path = None,
#                    title: str = "ROC Curve") -> None:
#     """Genera e salva ROC curve (solo binary)."""
#     fpr, tpr, thresholds = roc_curve(y_true, y_prob)
#     roc_auc = roc_auc_score(y_true, y_prob)
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         logger.info(f"ROC curve salvata: {output_path}")
    
#     plt.close()


# def plot_precision_recall_curve(y_true: np.ndarray,
#                                 y_prob: np.ndarray,
#                                 output_path: Path = None,
#                                 title: str = "Precision-Recall Curve") -> None:
#     """Genera e salva Precision-Recall curve (solo binary)."""
#     precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
#     ap = average_precision_score(y_true, y_prob)
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.4f})')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(title)
#     plt.legend(loc="lower left")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         logger.info(f"PR curve salvata: {output_path}")
    
#     plt.close()


# def plot_feature_importance(model,
#                             feature_names: list,
#                             output_path: Path = None,
#                             top_n: int = 20,
#                             title: str = "Feature Importance") -> None:
#     """Genera e salva grafico feature importance."""
#     # Estrai importanze (funziona per RF, XGB, LGBM)
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#     else:
#         logger.warning("Modello non ha feature_importances_")
#         return
    
#     # Ordina per importanza
#     indices = np.argsort(importances)[::-1][:top_n]
#     top_features = [feature_names[i] for i in indices]
#     top_importances = importances[indices]
    
#     plt.figure(figsize=(10, 8))
#     plt.barh(range(len(top_features)), top_importances[::-1], color='steelblue')
#     plt.yticks(range(len(top_features)), top_features[::-1])
#     plt.xlabel('Importance')
#     plt.title(title)
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         logger.info(f"Feature importance salvata: {output_path}")
    
#     plt.close()


# # ==============================================================================
# # REPORT
# # ==============================================================================

# def generate_report(model_name: str,
#                     task: str,
#                     metrics: Dict[str, float],
#                     errors: Dict[str, Any],
#                     y_true: np.ndarray,
#                     y_pred: np.ndarray,
#                     output_dir: Path) -> Path:
#     """
#     Genera report completo in formato JSON e TXT.
    
#     Returns:
#         Path al report JSON
#     """
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     report = {
#         'model_name': model_name,
#         'task': task,
#         'timestamp': timestamp,
#         'test_samples': int(len(y_true)),
#         'metrics': metrics,
#         'errors': errors,
#         'classification_report': classification_report(y_true, y_pred, output_dict=True)
#     }
    
#     # Salva JSON
#     json_path = output_dir / f"report_{model_name}_{task}.json"
#     with open(json_path, 'w') as f:
#         json.dump(report, f, indent=2, default=str)
    
#     # Salva TXT leggibile
#     txt_path = output_dir / f"report_{model_name}_{task}.txt"
#     with open(txt_path, 'w') as f:
#         f.write("=" * 60 + "\n")
#         f.write(f"EVALUATION REPORT - {model_name.upper()}\n")
#         f.write("=" * 60 + "\n\n")
        
#         f.write(f"Task: {task}\n")
#         f.write(f"Timestamp: {timestamp}\n")
#         f.write(f"Test samples: {len(y_true):,}\n\n")
        
#         f.write("-" * 40 + "\n")
#         f.write("METRICS\n")
#         f.write("-" * 40 + "\n")
#         for name, value in metrics.items():
#             f.write(f"  {name:25}: {value:.4f}\n")
        
#         f.write("\n" + "-" * 40 + "\n")
#         f.write("ERROR ANALYSIS\n")
#         f.write("-" * 40 + "\n")
#         for name, value in errors.items():
#             if isinstance(value, float):
#                 f.write(f"  {name:25}: {value:.4f}\n")
#             else:
#                 f.write(f"  {name:25}: {value}\n")
        
#         f.write("\n" + "-" * 40 + "\n")
#         f.write("CLASSIFICATION REPORT\n")
#         f.write("-" * 40 + "\n")
#         f.write(classification_report(y_true, y_pred))
    
#     logger.info(f"Report salvato: {json_path}")
#     logger.info(f"Report TXT: {txt_path}")
    
#     return json_path


# # ==============================================================================
# # PIPELINE COMPLETA
# # ==============================================================================

# def evaluate_model(model_path: Path,
#                    task: str = 'binary',
#                    output_dir: Path = None) -> Dict[str, Any]:
#     """
#     Pipeline completa di valutazione modello.
    
#     Args:
#         model_path: Path al modello .pkl
#         task: 'binary' o 'multiclass'
#         output_dir: Directory per output
    
#     Returns:
#         Dizionario con tutti i risultati
#     """
#     # Setup output directory
#     if output_dir is None:
#         model_name = model_path.parent.name
#         output_dir = get_project_root() / "reports" / model_name
    
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Carica modello
#     model = load_model(model_path)
#     model_name = model_path.parent.name
    
#     # Cerca file features specifico del modello
#     model_features = None
#     features_path = model_path.parent / f"features_{task}.json"
#     if features_path.exists():
#         with open(features_path, 'r') as f:
#             model_features = json.load(f)
#         logger.info(f"Caricate feature specifiche del modello: {len(model_features)}")
    
#     # Prepara dati test (usa feature del modello se disponibili)
#     X_test, y_test, mappings = prepare_test_data(task, model_features=model_features)
    
#     # Predizioni
#     print("   Generazione predizioni...")
#     y_pred = model.predict(X_test)
    
#     # Probabilita (se disponibili)
#     y_prob = None
#     if hasattr(model, 'predict_proba'):
#         y_prob_all = model.predict_proba(X_test)
#         if task == 'binary':
#             y_prob = y_prob_all[:, 1]  # Probabilita classe positiva
    
#     # Calcola metriche
#     print("   Calcolo metriche...")
#     metrics = compute_metrics(y_test.values, y_pred, y_prob, task)
    
#     # Analisi errori
#     print("   Analisi errori...")
#     errors = analyze_errors(y_test.values, y_pred, X_test, task)
    
#     # Genera visualizzazioni
#     print("   Generazione grafici...")
    
#     # Confusion Matrix
#     if task == 'binary':
#         labels = ['Benign', 'Attack']
#     else:
#         labels = list(mappings['multiclass'].keys())
    
#     plot_confusion_matrix(
#         y_test.values, y_pred, labels,
#         output_dir / f"confusion_matrix_{task}.png",
#         f"Confusion Matrix - {model_name}"
#     )
    
#     # ROC e PR curve (solo binary)
#     if task == 'binary' and y_prob is not None:
#         plot_roc_curve(
#             y_test.values, y_prob,
#             output_dir / f"roc_curve_{task}.png",
#             f"ROC Curve - {model_name}"
#         )
        
#         plot_precision_recall_curve(
#             y_test.values, y_prob,
#             output_dir / f"pr_curve_{task}.png",
#             f"Precision-Recall Curve - {model_name}"
#         )
    
#     # Feature importance
#     _, selected_features, _, _ = load_artifacts()
#     plot_feature_importance(
#         model, selected_features,
#         output_dir / f"feature_importance_{task}.png",
#         title=f"Feature Importance - {model_name}"
#     )
    
#     # Genera report
#     print("   Generazione report...")
#     report_path = generate_report(
#         model_name, task, metrics, errors,
#         y_test.values, y_pred, output_dir
#     )
    
#     return {
#         'model_name': model_name,
#         'task': task,
#         'metrics': metrics,
#         'errors': errors,
#         'output_dir': str(output_dir),
#         'report_path': str(report_path)
#     }


# # ==============================================================================
# # ARGUMENT PARSER
# # ==============================================================================

# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         description='Valutazione modelli NIDS',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Esempi:
#   python src/evaluation.py --model-path models/random_forest/model_binary.pkl
#   python src/evaluation.py --model-path models/xgboost/model_binary.pkl
#   python src/evaluation.py --model-path models/lightgbm/model_multiclass.pkl --task multiclass
#         """
#     )
    
#     parser.add_argument('--model-path', type=Path, required=True,
#                         help='Path al modello .pkl')
#     parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
#                         default='binary', help='Tipo classificazione')
#     parser.add_argument('--output-dir', type=Path, default=None,
#                         help='Directory output (default: reports/<model_name>)')
#     parser.add_argument('--n-jobs', type=int, default=None,
#                         help='Core CPU')
    
#     return parser.parse_args()


# # ==============================================================================
# # MAIN
# # ==============================================================================

# def main():
#     args = parse_arguments()
    
#     print("\n" + "=" * 60)
#     print("MODEL EVALUATION")
#     print("=" * 60)
#     print(f"\nModello: {args.model_path}")
#     print(f"Task:    {args.task}")
#     print()
    
#     try:
#         print("1. Caricamento modello e dati...")
#         results = evaluate_model(
#             args.model_path,
#             args.task,
#             args.output_dir
#         )
        
#         print("\n" + "=" * 60)
#         print("EVALUATION COMPLETATA")
#         print("=" * 60)
        
#         print(f"\nMetriche Test Set:")
#         for name, value in results['metrics'].items():
#             print(f"  {name:25}: {value:.4f}")
        
#         print(f"\nErrori:")
#         print(f"  Totali:          {results['errors']['total_errors']:,}")
#         print(f"  Error rate:      {results['errors']['error_rate']:.4f}")
#         if 'false_positives' in results['errors']:
#             print(f"  False Positives: {results['errors']['false_positives']:,}")
#             print(f"  False Negatives: {results['errors']['false_negatives']:,}")
        
#         print(f"\nOutput: {results['output_dir']}")
#         print(f"Report: {results['report_path']}")
        
#     except FileNotFoundError as e:
#         print(f"\nERRORE: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\nERRORE: {e}")
#         raise


# if __name__ == "__main__":
#     main()


"""
================================================================================
NIDS-ML - Modulo Evaluation
================================================================================

Valutazione modelli su test set con metriche complete e visualizzazioni.

GUIDA PARAMETRI:
----------------
    python src/evaluation.py [opzioni]

Opzioni:
    --model-path PATH     Path al modello .pkl (obbligatorio)
    --task STR            'binary' o 'multiclass' (default: binary)
    --output-dir PATH     Directory per report (default: reports/)
    --n-jobs INT          Core CPU (default: auto)

ESEMPI:
-------
# Valuta Random Forest binario
python src/evaluation.py --model-path models/random_forest/model_binary.pkl

# Valuta XGBoost multiclasse
python src/evaluation.py --model-path models/xgboost/model_multiclass.pkl --task multiclass

# Valuta tutti i modelli binari
python src/evaluation.py --model-path models/random_forest/model_binary.pkl
python src/evaluation.py --model-path models/xgboost/model_binary.pkl
python src/evaluation.py --model-path models/lightgbm/model_binary.pkl

================================================================================
"""

# ==============================================================================
# SETUP LIMITI CPU
# ==============================================================================
import sys
import os
import argparse
from pathlib import Path

def _get_arg(name, default=None, arg_type=str):
    for i, arg in enumerate(sys.argv):
        if arg == f'--{name}' and i + 1 < len(sys.argv):
            try:
                return arg_type(sys.argv[i + 1])
            except ValueError:
                return default
    return default

_n_jobs_arg = _get_arg('n-jobs', None, int)
_n_cores = _n_jobs_arg if _n_jobs_arg else max(1, (os.cpu_count() or 4) - 2)

os.environ['OMP_NUM_THREADS'] = str(_n_cores)
os.environ['MKL_NUM_THREADS'] = str(_n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(_n_cores)

import psutil
try:
    p = psutil.Process()
    p.cpu_affinity(list(range(_n_cores)))
    p.nice(10)
except Exception:
    pass

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import joblib
import json
from datetime import datetime
import warnings
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_logger, get_project_root, suppress_warnings
from src.preprocessing import load_processed_data
from src.feature_engineering import (
    load_artifacts,
    get_feature_columns,
    prepare_xy,
    transform_data,
    apply_feature_selection
)
from src.timing import TimingLogger

suppress_warnings()
logger = get_logger(__name__)


# ==============================================================================
# CARICAMENTO MODELLO E DATI
# ==============================================================================

def load_model(model_path: Path):
    """Carica modello salvato."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Modello caricato: {model_path}")
    
    return model


def prepare_test_data(task: str = 'binary', 
                      model_features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Prepara dati di test applicando scaler e feature selection.
    
    Args:
        task: 'binary' o 'multiclass'
        model_features: Lista feature del modello (se None usa artifacts globali)
    
    Returns:
        Tuple (X_test, y_test, mappings) pronti per predizione
    """
    # Carica dati
    _, _, test, mappings = load_processed_data()
    
    # Carica artifacts
    scaler, selected_features, _, _ = load_artifacts()
    
    # Usa feature specifiche del modello se fornite
    if model_features is not None:
        selected_features = model_features
    
    # Prepara X e y
    label_col = 'Label_Binary' if task == 'binary' else 'Label_Multiclass'
    feature_cols = get_feature_columns(test)
    
    X_test, y_test = prepare_xy(test, label_col, feature_cols)
    
    # Applica trasformazioni
    X_test_scaled = transform_data(X_test, scaler)
    X_test_final = apply_feature_selection(X_test_scaled, selected_features)
    
    logger.info(f"Test set preparato: {X_test_final.shape}")
    
    return X_test_final, y_test, mappings


# ==============================================================================
# CALCOLO METRICHE
# ==============================================================================

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None,
                    task: str = 'binary') -> Dict[str, float]:
    """
    Calcola metriche complete di classificazione.
    
    Args:
        y_true: Label reali
        y_pred: Predizioni
        y_prob: Probabilita predette (per ROC-AUC)
        task: 'binary' o 'multiclass'
    
    Returns:
        Dizionario con tutte le metriche
    """
    metrics = {}
    
    # Metriche base
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    if task == 'binary':
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        # ROC-AUC se abbiamo probabilita
        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['average_precision'] = float(average_precision_score(y_true, y_prob))
            except Exception:
                pass
    else:
        # Multiclass
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    return metrics


def measure_latency(model, X_test: pd.DataFrame, n_runs: int = 5) -> Dict[str, float]:
    """
    Misura la latenza di inferenza del modello.
    
    Questa metrica è CRITICA per un NIDS in produzione:
    - Se la latenza è troppo alta, non puoi fare detection real-time
    - La scorecard usa questo valore per scartare modelli troppo lenti
    
    Args:
        model: Modello da testare
        X_test: Dati di test
        n_runs: Numero di run per calcolare statistiche
    
    Returns:
        Dict con latenza in millisecondi:
        - mean_ms: Media
        - std_ms: Deviazione standard
        - min_ms: Minimo
        - max_ms: Massimo
        - per_sample_ms: Tempo medio per singolo campione
    """
    n_samples = len(X_test)
    times = []
    
    # Warm-up (prima esecuzione spesso più lenta)
    _ = model.predict(X_test.iloc[:100])
    
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Converti in ms
    
    times = np.array(times)
    
    return {
        'total_samples': n_samples,
        'n_runs': n_runs,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'per_sample_ms': float(np.mean(times) / n_samples),
        'throughput_samples_per_sec': float(n_samples / (np.mean(times) / 1000))
    }


def analyze_errors(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   X_test: pd.DataFrame,
                   task: str = 'binary') -> Dict[str, Any]:
    """
    Analizza errori di classificazione.
    
    Returns:
        Dizionario con statistiche sugli errori
    """
    errors = {}
    
    # Indici errori
    wrong_mask = y_true != y_pred
    n_errors = wrong_mask.sum()
    
    errors['total_errors'] = int(n_errors)
    errors['error_rate'] = float(n_errors / len(y_true))
    
    if task == 'binary':
        # False Positives (predetto 1, era 0)
        fp_mask = (y_pred == 1) & (y_true == 0)
        errors['false_positives'] = int(fp_mask.sum())
        
        # False Negatives (predetto 0, era 1)
        fn_mask = (y_pred == 0) & (y_true == 1)
        errors['false_negatives'] = int(fn_mask.sum())
        
        # Statistiche feature per FP e FN
        if fp_mask.sum() > 0:
            fp_features = X_test[fp_mask].describe().to_dict()
        else:
            fp_features = {}
        
        if fn_mask.sum() > 0:
            fn_features = X_test[fn_mask].describe().to_dict()
        else:
            fn_features = {}
    
    return errors


# ==============================================================================
# VISUALIZZAZIONI
# ==============================================================================

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: list = None,
                          output_path: Path = None,
                          title: str = "Confusion Matrix") -> None:
    """Genera e salva confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    # Normalizza per visualizzazione percentuale
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix salvata: {output_path}")
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   output_path: Path = None,
                   title: str = "ROC Curve") -> None:
    """Genera e salva ROC curve (solo binary)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve salvata: {output_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                output_path: Path = None,
                                title: str = "Precision-Recall Curve") -> None:
    """Genera e salva Precision-Recall curve (solo binary)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"PR curve salvata: {output_path}")
    
    plt.close()


def plot_feature_importance(model,
                            feature_names: list,
                            output_path: Path = None,
                            top_n: int = 20,
                            title: str = "Feature Importance") -> None:
    """Genera e salva grafico feature importance."""
    # Estrai importanze (funziona per RF, XGB, LGBM)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning("Modello non ha feature_importances_")
        return
    
    # Ordina per importanza
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances[::-1], color='steelblue')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance salvata: {output_path}")
    
    plt.close()


# ==============================================================================
# REPORT
# ==============================================================================

def generate_report(model_name: str,
                    task: str,
                    metrics: Dict[str, float],
                    errors: Dict[str, Any],
                    y_true: np.ndarray,
                    y_pred: np.ndarray,
                    output_dir: Path) -> Path:
    """
    Genera report completo in formato JSON e TXT.
    
    Returns:
        Path al report JSON
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'model_name': model_name,
        'task': task,
        'timestamp': timestamp,
        'test_samples': int(len(y_true)),
        'metrics': metrics,
        'errors': errors,
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Salva JSON
    json_path = output_dir / f"report_{model_name}_{task}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Salva TXT leggibile
    txt_path = output_dir / f"report_{model_name}_{task}.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVALUATION REPORT - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Task: {task}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Test samples: {len(y_true):,}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("METRICS\n")
        f.write("-" * 40 + "\n")
        for name, value in metrics.items():
            f.write(f"  {name:25}: {value:.4f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for name, value in errors.items():
            if isinstance(value, float):
                f.write(f"  {name:25}: {value:.4f}\n")
            else:
                f.write(f"  {name:25}: {value}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred))
    
    logger.info(f"Report salvato: {json_path}")
    logger.info(f"Report TXT: {txt_path}")
    
    return json_path


# ==============================================================================
# PIPELINE COMPLETA
# ==============================================================================

def evaluate_model(model_path: Path,
                   task: str = 'binary',
                   output_dir: Path = None) -> Dict[str, Any]:
    """
    Pipeline completa di valutazione modello.
    
    Args:
        model_path: Path al modello .pkl
        task: 'binary' o 'multiclass'
        output_dir: Directory per output
    
    Returns:
        Dizionario con tutti i risultati
    """
    # Setup output directory
    if output_dir is None:
        model_name = model_path.parent.name
        output_dir = get_project_root() / "reports" / model_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carica modello
    model = load_model(model_path)
    model_name = model_path.parent.name
    
    # Cerca file features specifico del modello
    model_features = None
    features_path = model_path.parent / f"features_{task}.json"
    if features_path.exists():
        with open(features_path, 'r') as f:
            model_features = json.load(f)
        logger.info(f"Caricate feature specifiche del modello: {len(model_features)}")
    
    # Prepara dati test (usa feature del modello se disponibili)
    X_test, y_test, mappings = prepare_test_data(task, model_features=model_features)
    
    # Predizioni
    print("   Generazione predizioni...")
    y_pred = model.predict(X_test)
    
    # Probabilita (se disponibili)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob_all = model.predict_proba(X_test)
        if task == 'binary':
            y_prob = y_prob_all[:, 1]  # Probabilita classe positiva
    
    # Calcola metriche
    print("   Calcolo metriche...")
    metrics = compute_metrics(y_test.values, y_pred, y_prob, task)
    
    # Misura latenza (CRITICO per NIDS real-time)
    print("   Misurazione latenza...")
    latency = measure_latency(model, X_test, n_runs=5)
    
    # Analisi errori
    print("   Analisi errori...")
    errors = analyze_errors(y_test.values, y_pred, X_test, task)
    
    # Genera visualizzazioni
    print("   Generazione grafici...")
    
    # Confusion Matrix
    if task == 'binary':
        labels = ['Benign', 'Attack']
    else:
        labels = list(mappings['multiclass'].keys())
    
    plot_confusion_matrix(
        y_test.values, y_pred, labels,
        output_dir / f"confusion_matrix_{task}.png",
        f"Confusion Matrix - {model_name}"
    )
    
    # ROC e PR curve (solo binary)
    if task == 'binary' and y_prob is not None:
        plot_roc_curve(
            y_test.values, y_prob,
            output_dir / f"roc_curve_{task}.png",
            f"ROC Curve - {model_name}"
        )
        
        plot_precision_recall_curve(
            y_test.values, y_prob,
            output_dir / f"pr_curve_{task}.png",
            f"Precision-Recall Curve - {model_name}"
        )
    
    # Feature importance - USA LE FEATURE DEL MODELLO, non quelle globali!
    # Questo corregge il bug segnalato da Gemini
    features_for_plot = model_features if model_features is not None else None
    if features_for_plot is None:
        _, features_for_plot, _, _ = load_artifacts()
    
    plot_feature_importance(
        model, features_for_plot,
        output_dir / f"feature_importance_{task}.png",
        title=f"Feature Importance - {model_name}"
    )
    
    # Genera report
    print("   Generazione report...")
    report_path = generate_report(
        model_name, task, metrics, errors,
        y_test.values, y_pred, output_dir
    )
    
    # Salva anche latenza nel report JSON
    latency_path = output_dir / f"latency_{task}.json"
    with open(latency_path, 'w') as f:
        json.dump(latency, f, indent=2)
    logger.info(f"Latenza salvata: {latency_path}")
    
    return {
        'model_name': model_name,
        'task': task,
        'metrics': metrics,
        'latency': latency,
        'errors': errors,
        'output_dir': str(output_dir),
        'report_path': str(report_path)
    }


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Valutazione modelli NIDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python src/evaluation.py --model-path models/random_forest/model_binary.pkl
  python src/evaluation.py --model-path models/xgboost/model_binary.pkl
  python src/evaluation.py --model-path models/lightgbm/model_multiclass.pkl --task multiclass
        """
    )
    
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path al modello .pkl')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                        default='binary', help='Tipo classificazione')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory output (default: reports/<model_name>)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Core CPU')
    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nModello: {args.model_path}")
    print(f"Task:    {args.task}")
    print()
    
    try:
        print("1. Caricamento modello e dati...")
        results = evaluate_model(
            args.model_path,
            args.task,
            args.output_dir
        )
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETATA")
        print("=" * 60)
        
        print(f"\nMetriche Test Set:")
        for name, value in results['metrics'].items():
            print(f"  {name:25}: {value:.4f}")
        
        print(f"\nLatenza Inferenza:")
        latency = results.get('latency', {})
        print(f"  Totale ({latency.get('total_samples', 0):,} samples): {latency.get('mean_ms', 0):.2f} ms")
        print(f"  Per sample:              {latency.get('per_sample_ms', 0)*1000:.4f} µs")
        print(f"  Throughput:              {latency.get('throughput_samples_per_sec', 0):,.0f} samples/sec")
        
        print(f"\nErrori:")
        print(f"  Totali:          {results['errors']['total_errors']:,}")
        print(f"  Error rate:      {results['errors']['error_rate']:.4f}")
        if 'false_positives' in results['errors']:
            print(f"  False Positives: {results['errors']['false_positives']:,}")
            print(f"  False Negatives: {results['errors']['false_negatives']:,}")
        
        print(f"\nOutput: {results['output_dir']}")
        print(f"Report: {results['report_path']}")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRORE: {e}")
        raise


if __name__ == "__main__":
    main()