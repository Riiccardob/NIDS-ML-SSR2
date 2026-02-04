"""
Pipeline completa per NIDS NetFlow-based con chunk-based processing.

Orchestrazione di:
1. Preprocessing (chunk-based)
2. Feature Engineering (chunk-based)
3. Training (con data loaders)
4. Evaluation

Ottimizzato per dataset grandi (>70M records).
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add srcNF to path
sys.path.insert(0, str(Path(__file__).parent))

import preprocessing
import feature_engineering
import training
from utils import setup_logger, log_system_info
from config import DEFAULT_MODEL


logger = setup_logger(__name__, 'pipeline.log')


def run_pipeline(model_type: str = None, skip_preprocessing: bool = False):
    """
    Esegue pipeline completa con chunk-based processing.
    
    Args:
        model_type: 'xgboost', 'lightgbm', o 'random_forest'
        skip_preprocessing: Se True, salta preprocessing e feature engineering
    """
    
    if model_type is None:
        model_type = DEFAULT_MODEL
    
    logger.info("="*70)
    logger.info("NIDS NETFLOW PIPELINE (CHUNK-BASED)")
    logger.info("="*70)
    logger.info(f"Model: {model_type}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    # Log system info
    log_system_info(logger)
    
    start_time = datetime.now()
    
    try:
        if not skip_preprocessing:
            # ================================================================
            # STEP 1: Preprocessing
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("STEP 1: PREPROCESSING (CHUNK-BASED)")
            logger.info("="*70)
            logger.info("CSV → Parquet con chunk-based processing")
            logger.info("Stratified split senza caricare tutto in RAM")
            logger.info("-"*70)
            
            step_start = datetime.now()
            preprocessing.main()
            step_time = (datetime.now() - step_start).total_seconds()
            
            logger.info(f"\n Preprocessing completato in {step_time:.1f}s ({step_time/60:.1f} min)")
            
            # ================================================================
            # STEP 2: Feature Engineering
            # ================================================================
            logger.info("\n" + "="*70)
            logger.info("STEP 2: FEATURE ENGINEERING (CHUNK-BASED)")
            logger.info("="*70)
            logger.info("Scaler fitted su sample rappresentativo (CON outlier)")
            logger.info("Scaling applicato chunk-by-chunk")
            logger.info("-"*70)
            
            step_start = datetime.now()
            feature_engineering.main()
            step_time = (datetime.now() - step_start).total_seconds()
            
            logger.info(f"\n Feature engineering completato in {step_time:.1f}s ({step_time/60:.1f} min)")
            
        else:
            logger.info("\n⏭  Skipping preprocessing (usando dati esistenti)")
        
        # ====================================================================
        # STEP 3: Training
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 3: TRAINING")
        logger.info("="*70)
        logger.info(f"Modello: {model_type}")
        logger.info("Training data caricato in memoria")
        logger.info("Validation/Test con chunk-based data loaders")
        logger.info("-"*70)
        
        step_start = datetime.now()
        training.main(model_type)
        step_time = (datetime.now() - step_start).total_seconds()
        
        logger.info(f"\n Training completato in {step_time:.1f}s ({step_time/60:.1f} min)")
        
        # ====================================================================
        # Summary
        # ====================================================================
        elapsed = datetime.now() - start_time
        
        logger.info("\n" + "="*70)
        logger.info(" PIPELINE COMPLETATA CON SUCCESSO")
        logger.info("="*70)
        logger.info(f"Tempo totale: {elapsed.total_seconds():.1f}s ({elapsed.total_seconds()/60:.1f} min)")
        logger.info(f"Modello: {model_type}")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n ERRORE nella pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Entry point con argomenti CLI."""
    
    parser = argparse.ArgumentParser(
        description='Pipeline completa NIDS NetFlow-based con chunk-based processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI D'USO:

1. Pipeline completa con XGBoost (default):
   python pipeline.py

2. Pipeline completa con LightGBM:
   python pipeline.py --model lightgbm

3. Solo training (se preprocessing già fatto):
   python pipeline.py --model xgboost --skip-preprocessing

4. Pipeline completa con Random Forest:
   python pipeline.py --model random_forest

NOTE:
- La pipeline usa chunk-based processing per gestire dataset grandi (>70M records)
- Lo scaler viene fittato su un sample rappresentativo CON outlier
- Il training carica i dati in memoria, ma validation/test usano data loaders
- Requisiti: ~8-12 GB RAM disponibili per dataset da 76M records
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgboost', 'random_forest', 'lightgbm'],
        default=DEFAULT_MODEL,
        help=f'Tipo modello (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Salta preprocessing e feature engineering (usa dati esistenti)'
    )
    
    args = parser.parse_args()
    
    success = run_pipeline(
        model_type=args.model,
        skip_preprocessing=args.skip_preprocessing
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
