#!/usr/bin/env python3
"""
Test suite per Live NIDS Sniffer.

Verifica tutti i componenti prima del deployment.
AGGIORNATO: Supporta 24 feature (post-drop).
"""

import sys
import os
from pathlib import Path
import numpy as np

# Setup path
SNIFFER_DIR = Path(__file__).parent
PROJECT_ROOT = SNIFFER_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("LIVE NIDS SNIFFER - COMPONENT TEST")
print("=" * 70)
print(f"Working directory: {Path.cwd()}")
print(f"Sniffer directory: {SNIFFER_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print()


def test_external_libraries():
    """Test 1: Librerie esterne."""
    print("=" * 70)
    print("TEST 1: External Libraries")
    print("=" * 70)
    
    libraries = [
        'numpy',
        'pandas',
        'sklearn',
        'joblib',
        'xgboost',
        'lightgbm',
        'nfstream',
        'psutil',
        'pyarrow',
    ]
    
    all_ok = True
    for lib in libraries:
        try:
            if lib == 'sklearn':
                __import__('sklearn')
                print(f"  OK: scikit-learn")
            else:
                __import__(lib)
                print(f"  OK: {lib}")
        except ImportError as e:
            print(f"  FAIL: {lib} - {e}")
            all_ok = False
    
    print()
    return all_ok


def test_sniffer_modules():
    """Test 2: Moduli sniffer."""
    print("=" * 70)
    print("TEST 2: Sniffer Modules")
    print("=" * 70)
    
    modules = [
        'config',
        'utils.logger',
        'core.feature_mapper',
        'core.preprocessor',
        'core.predictor',
        'core.capture',
        'security.alert_manager',
        'security.firewall_controller',
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  OK: {module}")
        except ImportError as e:
            print(f"  FAIL: {module} - {e}")
            all_ok = False
    
    print()
    return all_ok


def test_artifacts():
    """Test 3: Artifacts necessari."""
    print("=" * 70)
    print("TEST 3: Artifacts")
    print("=" * 70)
    print()
    
    from config import ARTIFACTS_DIR, MODELS_DIR, SCALER_PATH, FEATURES_PATH, MODEL_PATH
    
    print("  Required artifacts:")
    all_ok = True
    
    # Scaler
    if SCALER_PATH.exists():
        size = SCALER_PATH.stat().st_size / (1024**2)
        print(f"  OK: scaler.pkl ({size:.2f} MB)")
    else:
        print(f"  MISSING: scaler.pkl")
        all_ok = False
    
    # Features
    if FEATURES_PATH.exists():
        size = FEATURES_PATH.stat().st_size / (1024**2)
        print(f"  OK: features.json ({size:.2f} MB)")
    else:
        print(f"  MISSING: features.json")
        all_ok = False
    
    print()
    print("  Model artifacts (at least one required):")
    
    model_found = False
    for model_type in ['xgboost', 'lightgbm']:
        model_path = MODELS_DIR / model_type / "model.pkl"
        if model_path.exists():
            size = model_path.stat().st_size / (1024**2)
            print(f"  OK: {model_type}/model.pkl ({size:.2f} MB)")
            model_found = True
        else:
            print(f"  MISSING: {model_type}/model.pkl")
    
    if not model_found:
        all_ok = False
    
    print()
    return all_ok


def test_config():
    """Test 4: Configurazione."""
    print("=" * 70)
    print("TEST 4: Configuration Validation")
    print("=" * 70)
    
    from config import (
        PROJECT_ROOT, ARTIFACTS_DIR, MODELS_DIR,
        validate_config, get_config_summary
    )
    
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Artifacts dir: {ARTIFACTS_DIR}")
    print(f"  Models dir: {MODELS_DIR}")
    
    try:
        validate_config()
        print("  OK: Configuration valid")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    
    summary = get_config_summary()
    print()
    print("  Configuration summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    print()
    return True


def test_feature_mapper():
    """Test 5: Feature mapper."""
    print("=" * 70)
    print("TEST 5: Feature Mapper")
    print("=" * 70)
    
    from core.feature_mapper import FeatureMapper
    from config import N_FEATURES
    
    try:
        mapper = FeatureMapper()
        print(f"  OK: Mapper initialized with {mapper.n_features} features")
        
        # Test extraction su dummy flow
        # IMPORTANTE: Usa N_FEATURES dal config (24, non 35!)
        dummy_flow = {
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 6,
            'application_name': 'HTTP',
            'src2dst_bytes': 1000,
            'src2dst_packets': 10,
            'dst2src_bytes': 2000,
            'bidirectional_duration_ms': 5000,
            'src2dst_duration_ms': 4000,
            'dst2src_duration_ms': 1000,
            'bidirectional_min_ps': 60,
            'bidirectional_max_ps': 1500,
            'bidirectional_packets': 15,
            'bidirectional_bytes': 3000,
        }
        
        features = mapper.extract_features(dummy_flow)
        
        if features.shape == (N_FEATURES,):
            print(f"  OK: Feature extraction successful ({N_FEATURES} features)")
        else:
            print(f"  FAIL: Expected {N_FEATURES} features, got {features.shape[0]}")
            return False
        
        if mapper.validate_feature_vector(features):
            print(f"  OK: Feature vector valid")
        else:
            print(f"  FAIL: Feature vector validation failed")
            return False
        
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_preprocessor():
    """Test 6: Preprocessor."""
    print("=" * 70)
    print("TEST 6: Preprocessor (Scaler)")
    print("=" * 70)
    
    from core.preprocessor import FeaturePreprocessor
    from config import N_FEATURES
    
    try:
        preprocessor = FeaturePreprocessor()
        print("  OK: Scaler loaded")
        
        info = preprocessor.get_scaler_info()
        print(f"  OK: Scaler type: {info['scaler_type']}")
        print(f"  OK: Features: {info['n_features']}")
        
        # CRITICO: Usa N_FEATURES corretto (24)
        dummy_features = np.random.rand(N_FEATURES)
        scaled = preprocessor.preprocess(dummy_features)
        
        if scaled.shape == (N_FEATURES,):
            print(f"  OK: Scaling successful ({N_FEATURES} features)")
        else:
            print(f"  FAIL: Expected {N_FEATURES} features, got {scaled.shape[0]}")
            return False
        
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_predictor():
    """Test 7: Predictor."""
    print("=" * 70)
    print("TEST 7: Model Predictor")
    print("=" * 70)
    
    from core.predictor import ModelPredictor
    from config import N_FEATURES
    
    try:
        predictor = ModelPredictor()
        print("  OK: Model loaded")
        
        info = predictor.get_model_info()
        print(f"  OK: Model type: {info['model_type']}")
        print(f"  OK: Supports proba: {info['supports_proba']}")
        
        # CRITICO: Usa N_FEATURES corretto (24)
        dummy_features = np.random.rand(N_FEATURES)
        result = predictor.predict(dummy_features)
        
        print(f"  OK: Prediction: {result.prediction} (confidence: {result.confidence:.4f})")
        
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_logger():
    """Test 8: Logger."""
    print("=" * 70)
    print("TEST 8: Structured Logger")
    print("=" * 70)
    
    from utils.logger import get_logger
    
    try:
        logger = get_logger()
        logger.info("Test log message")
        print("  OK: Logger initialized")
        
        # Test alert logging
        logger.warning(
            "ALERT: 192.168.1.100:12345 -> 8.8.8.8:53 | "
            "Prediction: attack | Confidence: 0.9500 | Action: logged"
        )
        print("  OK: Alert logging successful")
        
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    
    print()
    return True


def test_network_interfaces():
    """Test 9: Network interfaces."""
    print("=" * 70)
    print("TEST 9: Network Interfaces")
    print("=" * 70)
    
    import psutil
    
    try:
        interfaces = psutil.net_if_stats()
        
        print("  Available network interfaces:")
        for name, stats in interfaces.items():
            status = "UP" if stats.isup else "DOWN"
            print(f"    {name}: {status}")
        
        print()
        print("  OK: Network interface detection successful")
        
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    
    print()
    return True


def test_root_privileges():
    """Test 10: Root privileges."""
    print("=" * 70)
    print("TEST 10: Root Privileges")
    print("=" * 70)
    
    if os.geteuid() == 0:
        print("  OK: Running with root privileges")
        result = True
    else:
        print("  WARNING: Not running with root privileges")
        print("  Network capture will require sudo")
        result = True  # Non è un errore critico
    
    print()
    return result


def main():
    """Run all tests."""
    
    tests = [
        ("External Libraries", test_external_libraries),
        ("Sniffer Modules", test_sniffer_modules),
        ("Artifacts", test_artifacts),
        ("Configuration", test_config),
        ("Feature Mapper", test_feature_mapper),
        ("Preprocessor", test_preprocessor),
        ("Predictor", test_predictor),
        ("Logger", test_logger),
        ("Network Interfaces", test_network_interfaces),
        ("Root Privileges", test_root_privileges),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"CRITICAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print()
    print("=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("STATUS: ALL TESTS PASSED ")
    else:
        print("STATUS: SOME TESTS FAILED")
        print()
        print("Common fixes:")
        print("  - Make sure you're in srcNF/live_sniffer directory")
        print("  - Run with sudo for root privileges")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
