#!/usr/bin/env python3
"""
Test script per verificare componenti Live NIDS Sniffer.

IMPORTANTE: Esegui dalla directory live_sniffer:
  cd srcNF/live_sniffer
  sudo python3 test_sniffer.py
"""

import sys
from pathlib import Path
import traceback

# Fix: Aggiungi directory corrente e parent al path
SNIFFER_DIR = Path(__file__).parent
sys.path.insert(0, str(SNIFFER_DIR))
sys.path.insert(0, str(SNIFFER_DIR.parent.parent))  # PROJECT_ROOT


def print_header(title: str) -> None:
    """Stampa header formattato."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def test_imports() -> bool:
    """Test import librerie esterne."""
    
    print_header("TEST 1: External Libraries")
    
    libraries = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('joblib', 'joblib'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('nfstream', 'nfstream'),
        ('psutil', 'psutil'),
        ('pyarrow', 'pyarrow'),
    ]
    
    all_ok = True
    
    for name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"  OK: {name}")
        except ImportError as e:
            print(f"  FAIL: {name} - {e}")
            all_ok = False
    
    return all_ok


def test_module_imports() -> bool:
    """Test import moduli sniffer."""
    
    print_header("TEST 2: Sniffer Modules")
    
    modules = [
        ('config', 'config'),
        ('utils.logger', 'utils.logger'),
        ('core.feature_mapper', 'core.feature_mapper'),
        ('core.preprocessor', 'core.preprocessor'),
        ('core.predictor', 'core.predictor'),
        ('core.capture', 'core.capture'),
        ('security.alert_manager', 'security.alert_manager'),
        ('security.firewall_controller', 'security.firewall_controller'),
    ]
    
    all_ok = True
    
    for display_name, module_name in modules:
        try:
            __import__(module_name)
            print(f"  OK: {display_name}")
        except Exception as e:
            print(f"  FAIL: {display_name} - {e}")
            traceback.print_exc()
            all_ok = False
    
    return all_ok


def test_artifacts() -> bool:
    """Test presenza artifacts."""
    
    print_header("TEST 3: Artifacts")
    
    project_root = SNIFFER_DIR.parent.parent
    
    artifacts = [
        project_root / 'artifacts' / 'scaler.pkl',
        project_root / 'artifacts' / 'features.json',
    ]
    
    models = [
        project_root / 'models' / 'xgboost' / 'model.pkl',
        project_root / 'models' / 'lightgbm' / 'model.pkl',
    ]
    
    all_ok = True
    
    print("\n  Required artifacts:")
    for artifact in artifacts:
        if artifact.exists():
            size_mb = artifact.stat().st_size / (1024**2)
            print(f"  OK: {artifact.name} ({size_mb:.2f} MB)")
        else:
            print(f"  FAIL: {artifact.name} - NOT FOUND")
            print(f"       Expected at: {artifact}")
            all_ok = False
    
    print("\n  Model artifacts (at least one required):")
    model_found = False
    for model in models:
        if model.exists():
            size_mb = model.stat().st_size / (1024**2)
            print(f"  OK: {model.parent.name}/{model.name} ({size_mb:.2f} MB)")
            model_found = True
        else:
            print(f"  MISSING: {model.parent.name}/{model.name}")
    
    if not model_found:
        print("\n  ERROR: No model found! At least one model required.")
        print("  Run training first: cd ../.. && python srcNF/pipeline.py --model xgboost")
        all_ok = False
    
    return all_ok


def test_config_validation() -> bool:
    """Test validazione configurazione."""
    
    print_header("TEST 4: Configuration Validation")
    
    try:
        import config
        
        # Check paths
        print(f"  Project root: {config.PROJECT_ROOT}")
        print(f"  Artifacts dir: {config.ARTIFACTS_DIR}")
        print(f"  Models dir: {config.MODELS_DIR}")
        
        # Validate (questo fallira se mancano artifacts)
        try:
            config.validate_config()
            print("  OK: Configuration valid")
        except ValueError as e:
            print(f"  WARNING: Configuration has missing files")
            print(f"  {e}")
            return False
        
        summary = config.get_config_summary()
        print("\n  Configuration summary:")
        for key, value in summary.items():
            print(f"    {key}: {value}")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_feature_mapper() -> bool:
    """Test feature mapper."""
    
    print_header("TEST 5: Feature Mapper")
    
    try:
        from core.feature_mapper import FeatureMapper
        import numpy as np
        
        mapper = FeatureMapper()
        print(f"  OK: Mapper initialized with {mapper.n_features} features")
        print(f"  OK: Feature mapping built ({len(mapper.feature_mapping)} mappings)")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_preprocessor() -> bool:
    """Test preprocessor."""
    
    print_header("TEST 6: Preprocessor (Scaler)")
    
    try:
        from core.preprocessor import FeaturePreprocessor
        import numpy as np
        
        preprocessor = FeaturePreprocessor()
        print("  OK: Scaler loaded")
        
        scaler_info = preprocessor.get_scaler_info()
        print(f"  OK: Scaler type: {scaler_info['scaler_type']}")
        print(f"  OK: Features: {scaler_info['n_features']}")
        
        # Test preprocessing
        dummy_features = np.random.randn(35)
        scaled = preprocessor.preprocess(dummy_features)
        print(f"  OK: Test preprocessing successful (shape: {scaled.shape})")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_predictor() -> bool:
    """Test predictor."""
    
    print_header("TEST 7: Model Predictor")
    
    try:
        from core.predictor import ModelPredictor
        import numpy as np
        
        predictor = ModelPredictor()
        print("  OK: Model loaded")
        
        model_info = predictor.get_model_info()
        print(f"  OK: Model type: {model_info['model_type']}")
        print(f"  OK: Supports proba: {model_info['supports_proba']}")
        
        # Test prediction
        dummy_features = np.random.randn(35)
        result = predictor.predict(dummy_features)
        print(f"  OK: Test prediction successful")
        print(f"      Prediction: {result.prediction} (0=benign, 1=attack)")
        print(f"      Confidence: {result.confidence:.4f}")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_logger() -> bool:
    """Test logger."""
    
    print_header("TEST 8: Structured Logger")
    
    try:
        from utils.logger import get_logger
        
        logger = get_logger()
        logger.info("Test log message")
        print("  OK: Logger initialized")
        
        # Test structured alert
        test_alert = {
            'timestamp': '2024-02-05T12:00:00',
            'src_ip': '192.168.1.100',
            'src_port': 12345,
            'dst_ip': '8.8.8.8',
            'dst_port': 53,
            'protocol': 17,
            'l7_proto': 'DNS',
            'prediction': 'attack',
            'confidence': 0.95,
            'action': 'logged',
        }
        
        logger.log_alert(test_alert)
        print("  OK: Alert logging successful")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_network_interface() -> bool:
    """Test rilevamento interfacce di rete."""
    
    print_header("TEST 9: Network Interfaces")
    
    try:
        import psutil
        
        interfaces = psutil.net_if_stats()
        
        print("  Available network interfaces:")
        for iface, stats in interfaces.items():
            status = "UP" if stats.isup else "DOWN"
            print(f"    {iface}: {status}")
        
        print("\n  OK: Network interface detection successful")
        
        return True
    
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_permissions() -> bool:
    """Test permessi root."""
    
    print_header("TEST 10: Root Privileges")
    
    import os
    
    if os.geteuid() == 0:
        print("  OK: Running with root privileges")
        return True
    else:
        print("  WARNING: Not running as root")
        print("  Packet capture and firewall require root (sudo)")
        print("  Run with: sudo python3 test_sniffer.py")
        return False


def main():
    """Main test runner."""
    
    print("="*70)
    print("LIVE NIDS SNIFFER - COMPONENT TEST")
    print("="*70)
    print(f"Working directory: {Path.cwd()}")
    print(f"Sniffer directory: {SNIFFER_DIR}")
    print(f"Project root: {SNIFFER_DIR.parent.parent}")
    
    tests = [
        ("External Libraries", test_imports),
        ("Sniffer Modules", test_module_imports),
        ("Artifacts", test_artifacts),
        ("Configuration", test_config_validation),
        ("Feature Mapper", test_feature_mapper),
        ("Preprocessor", test_preprocessor),
        ("Predictor", test_predictor),
        ("Logger", test_logger),
        ("Network Interfaces", test_network_interface),
        ("Root Privileges", test_permissions),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  CRITICAL ERROR: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("STATUS: ALL TESTS PASSED")
        print("\nYou can now run the sniffer:")
        print("  sudo python3 main.py --mode alert")
    else:
        print("STATUS: SOME TESTS FAILED")
        print("\nCommon fixes:")
        if not any("Artifacts" in name and result for name, result in results):
            print("  - Run training pipeline first:")
            print("    cd ../.. && python srcNF/pipeline.py --model xgboost")
        print("  - Make sure you're in srcNF/live_sniffer directory")
        print("  - Run with sudo for root privileges")
    
    print("="*70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
