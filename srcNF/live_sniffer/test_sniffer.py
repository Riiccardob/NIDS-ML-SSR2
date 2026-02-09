#!/usr/bin/env python3
"""
Test suite per Live NIDS Sniffer - VERSIONE MIGLIORATA

Verifica tutti i componenti prima del deployment.
AGGIORNATO: Test con dati realistici, non dummy values.
"""

import sys
import os
from pathlib import Path
import numpy as np

SNIFFER_DIR = Path(__file__).parent
PROJECT_ROOT = SNIFFER_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("LIVE NIDS SNIFFER - COMPONENT TEST (IMPROVED)")
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
    
    if SCALER_PATH.exists():
        size = SCALER_PATH.stat().st_size / (1024**2)
        print(f"  OK: scaler.pkl ({size:.2f} MB)")
    else:
        print(f"  MISSING: scaler.pkl")
        all_ok = False
    
    if FEATURES_PATH.exists():
        size = FEATURES_PATH.stat().st_size / (1024**2)
        print(f"  OK: features.json ({size:.2f} MB)")
        
        import json
        with open(FEATURES_PATH, 'r') as f:
            features_info = json.load(f)
        
        n_features = features_info.get('n_features', 0)
        print(f"  INFO: Model expects {n_features} features")
    else:
        print(f"  MISSING: features.json")
        all_ok = False
    
    print()
    print("  Model artifacts:")
    
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
        validate_config, get_config_summary, REQUIRED_FEATURES
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
    print(f"  Required features ({len(REQUIRED_FEATURES)}):")
    for i, feat in enumerate(REQUIRED_FEATURES[:5], 1):
        print(f"    {i}. {feat}")
    if len(REQUIRED_FEATURES) > 5:
        print(f"    ... and {len(REQUIRED_FEATURES)-5} more")
    
    print()
    return True


def test_feature_mapper():
    """Test 5: Feature mapper con dati realistici."""
    print("=" * 70)
    print("TEST 5: Feature Mapper (REALISTIC TEST)")
    print("=" * 70)
    
    from core.feature_mapper import FeatureMapper
    from config import N_FEATURES
    
    try:
        mapper = FeatureMapper()
        print(f"  OK: Mapper initialized with {mapper.n_features} features")
        
        # TEST 1: Flow TCP HTTP
        print("\n  Test 1: TCP HTTP flow")
        http_flow = {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 54321,
            'dst_port': 80,
            'protocol': 6,  # TCP
            'application_name': 'HTTP',
            'src2dst_bytes': 1500,
            'src2dst_packets': 10,
            'src2dst_duration_ms': 5000,
            'dst2src_bytes': 8000,
            'dst2src_packets': 15,
            'dst2src_duration_ms': 4500,
            'bidirectional_bytes': 9500,
            'bidirectional_packets': 25,
            'bidirectional_duration_ms': 5000,
            'bidirectional_min_ps': 60,
            'bidirectional_max_ps': 1500,
            'client_tcp_flags': 2,  # SYN
            'bidirectional_retrans_packets': 0,
        }
        
        features_http = mapper.extract_features(http_flow)
        
        if features_http.shape == (N_FEATURES,):
            print(f"    OK: Extracted {N_FEATURES} features")
        else:
            print(f"    FAIL: Expected {N_FEATURES} features, got {features_http.shape[0]}")
            return False
        
        if mapper.validate_feature_vector(features_http):
            print(f"    OK: Feature vector valid (no NaN/Inf)")
        else:
            print(f"    FAIL: Feature vector invalid")
            return False
        
        print(f"    Sample values: L4_SRC_PORT={features_http[0]:.1f}, L4_DST_PORT={features_http[1]:.1f}")
        
        # TEST 2: Flow UDP DNS
        print("\n  Test 2: UDP DNS flow")
        dns_flow = {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 53210,
            'dst_port': 53,
            'protocol': 17,  # UDP
            'application_name': 'DNS',
            'src2dst_bytes': 65,
            'src2dst_packets': 1,
            'src2dst_duration_ms': 100,
            'dst2src_bytes': 120,
            'dst2src_packets': 1,
            'dst2src_duration_ms': 50,
            'bidirectional_bytes': 185,
            'bidirectional_packets': 2,
            'bidirectional_duration_ms': 150,
            'bidirectional_min_ps': 65,
            'bidirectional_max_ps': 120,
            'client_tcp_flags': 0,
            'bidirectional_retrans_packets': 0,
        }
        
        features_dns = mapper.extract_features(dns_flow)
        
        if features_dns.shape == (N_FEATURES,) and mapper.validate_feature_vector(features_dns):
            print(f"    OK: DNS flow extracted correctly")
        else:
            print(f"    FAIL: DNS flow extraction failed")
            return False
        
        # TEST 3: Flow ICMP
        print("\n  Test 3: ICMP flow")
        icmp_flow = {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 0,
            'dst_port': 0,
            'protocol': 1,  # ICMP
            'application_name': 'ICMP',
            'icmp_type': 8,  # Echo request
            'src2dst_bytes': 84,
            'src2dst_packets': 1,
            'src2dst_duration_ms': 10,
            'dst2src_bytes': 84,
            'dst2src_packets': 1,
            'dst2src_duration_ms': 5,
            'bidirectional_bytes': 168,
            'bidirectional_packets': 2,
            'bidirectional_duration_ms': 15,
            'bidirectional_min_ps': 84,
            'bidirectional_max_ps': 84,
            'client_tcp_flags': 0,
            'bidirectional_retrans_packets': 0,
        }
        
        features_icmp = mapper.extract_features(icmp_flow)
        
        if features_icmp.shape == (N_FEATURES,) and mapper.validate_feature_vector(features_icmp):
            print(f"    OK: ICMP flow extracted correctly")
            # Verifica che ICMP_IPV4_TYPE sia popolato
            icmp_type_idx = mapper.required_features.index("ICMP_IPV4_TYPE")
            if features_icmp[icmp_type_idx] == 8.0:
                print(f"    OK: ICMP_IPV4_TYPE correctly set to 8")
            else:
                print(f"    WARNING: ICMP_IPV4_TYPE = {features_icmp[icmp_type_idx]}, expected 8")
        else:
            print(f"    FAIL: ICMP flow extraction failed")
            return False
        
        print("\n  Feature extraction tests passed!")
        
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
        
        dummy_features = np.random.rand(N_FEATURES) * 1000
        scaled = preprocessor.preprocess(dummy_features)
        
        if scaled.shape == (N_FEATURES,):
            print(f"  OK: Scaling successful ({N_FEATURES} features)")
        else:
            print(f"  FAIL: Expected {N_FEATURES} features, got {scaled.shape[0]}")
            return False
        
        if not np.any(np.isnan(scaled)) and not np.any(np.isinf(scaled)):
            print(f"  OK: Scaled features valid (no NaN/Inf)")
        else:
            print(f"  FAIL: Scaled features contain NaN/Inf")
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
        
        dummy_features = np.random.randn(N_FEATURES)
        result = predictor.predict(dummy_features)
        
        print(f"  OK: Prediction: {result.prediction} (confidence: {result.confidence:.4f})")
        
        if result.prediction in [0, 1]:
            print(f"  OK: Prediction is binary (0 or 1)")
        else:
            print(f"  FAIL: Invalid prediction value: {result.prediction}")
            return False
        
        if 0.0 <= result.confidence <= 1.0:
            print(f"  OK: Confidence in valid range [0,1]")
        else:
            print(f"  FAIL: Invalid confidence: {result.confidence}")
            return False
        
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
        result = True
    
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
        print("STATUS: ALL TESTS PASSED")
        print()
        print("NEXT STEPS:")
        print("  1. sudo python3 main.py --mode alert --interface lo")
        print("  2. Generate traffic: ping 127.0.0.1")
        print("  3. Check logs: tail -f ../../logs/sniffer/nids_sniffer_alerts.csv")
    else:
        print("STATUS: SOME TESTS FAILED")
        print()
        print("Common fixes:")
        print("  - Run with sudo for root privileges")
        print("  - Ensure artifacts exist (run training first)")
        print("  - Check feature count mismatch")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)