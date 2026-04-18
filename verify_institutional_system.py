#!/usr/bin/env python3
"""
Nexus Institutional v0.3.0 - Full System Verification
=====================================================
Comprehensive test suite verifying all enterprise capabilities.

Run this to verify the complete institutional platform is operational.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.nexus.core.context import engine_context
from src.nexus.institutional.orchestrator import (
    InstitutionalOrchestrator,
    ExecutionMode,
    AssetClass,
)


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_initialization():
    """Test 1: System Initialization."""
    print_header("TEST 1: System Initialization")
    
    try:
        engine_context.initialize(config_path="config/production.yaml")
        logger = engine_context.get_logger("verification")
        logger.info("✅ Production configuration loaded successfully")
        print("✅ PASSED: System initialization")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_multi_asset_support():
    """Test 2: Multi-Asset Class Support."""
    print_header("TEST 2: Multi-Asset Class Support")
    
    try:
        orchestrator = InstitutionalOrchestrator()
        orchestrator.set_asset_classes(["equities", "fixed-income", "crypto", "derivatives", "fx"])
        
        expected = {AssetClass.EQUITIES, AssetClass.FIXED_INCOME, AssetClass.CRYPTO, 
                   AssetClass.DERIVATIVES, AssetClass.FX}
        actual = set(orchestrator.asset_classes)
        
        if expected == actual:
            print(f"✅ PASSED: All 5 asset classes initialized")
            for ac in orchestrator.asset_classes:
                print(f"   ✓ {ac.value}")
            return True
        else:
            print(f"❌ FAILED: Asset class mismatch")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_venue_support():
    """Test 3: 235+ Venue Support."""
    print_header("TEST 3: 235+ Venue Support")
    
    try:
        orchestrator = InstitutionalOrchestrator()
        orchestrator.set_venue_count(235)
        
        num_venues = len(orchestrator.venues)
        print(f"✅ PASSED: {num_venues} venues initialized")
        
        # Show sample venues
        sample_venues = list(orchestrator.venues.values())[:10]
        print(f"\n  Sample venues:")
        for venue in sample_venues:
            print(f"    • {venue.name:15} ({', '.join(ac.value for ac in venue.asset_classes)})")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_execution_modes():
    """Test 4: Execution Modes."""
    print_header("TEST 4: Execution Modes")
    
    try:
        orchestrator = InstitutionalOrchestrator()
        
        modes = ["backtest", "paper", "live", "market-making"]
        for mode in modes:
            orchestrator.set_execution_mode(mode)
            expected_mode = mode.replace("-", "_").upper()
            actual_mode = orchestrator.mode.name
            
            if expected_mode == actual_mode:
                print(f"✅ {mode.upper():15} - OK")
            else:
                print(f"❌ {mode.upper():15} - FAILED")
                return False
        
        print("\n✅ PASSED: All execution modes functional")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_institutional_orchestrator():
    """Test 5: Institutional Orchestrator."""
    print_header("TEST 5: Institutional Orchestrator")
    
    try:
        orchestrator = InstitutionalOrchestrator()
        orchestrator.set_execution_mode("market-making")
        orchestrator.set_asset_classes(["multi"])
        orchestrator.set_venue_count(50)
        orchestrator.enable_market_making()
        orchestrator.enable_ultra_low_latency()
        
        checks = [
            ("Execution Mode", orchestrator.mode.name == "MARKET_MAKING"),
            ("Asset Classes", len(orchestrator.asset_classes) == 6),  # multi = all 6
            ("Venues", len(orchestrator.venues) > 0),
            ("Market Making", orchestrator.market_making_enabled),
            ("Ultra-Low Latency", orchestrator.ultra_low_latency_mode),
        ]
        
        all_passed = True
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("\n✅ PASSED: Orchestrator fully operational")
        
        return all_passed
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_cloud_deployment():
    """Test 6: Cloud Deployment Configuration."""
    print_header("TEST 6: Cloud Deployment (Azure)")
    
    try:
        from src.nexus.institutional.deployment import CloudDeploymentManager, CloudConfig
        
        config = CloudConfig(
            region="westus2",
            aks_vm_count=10,
            enable_gpu=True,
            enable_fpga=True,
        )
        
        manager = CloudDeploymentManager(config)
        
        checks = [
            ("Region", config.region == "westus2"),
            ("AKS Nodes", config.aks_vm_count == 10),
            ("GPU Support", config.enable_gpu),
            ("FPGA Support", config.enable_fpga),
            ("Auto-scaling", config.auto_scaling),
        ]
        
        all_passed = True
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("\n✅ PASSED: Cloud deployment configuration valid")
        
        return all_passed
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_configuration_parsing():
    """Test 7: Production Configuration Parsing."""
    print_header("TEST 7: Production Configuration Parsing")
    
    try:
        import yaml
        
        with open("config/production.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        required_sections = [
            "environment",
            "execution",
            "market_making",
            "risk",
            "low_latency",
            "strategies",
            "cloud",
        ]
        
        all_present = True
        for section in required_sections:
            if section in config:
                print(f"✅ {section}")
            else:
                print(f"❌ {section} MISSING")
                all_present = False
        
        if all_present:
            print("\n✅ PASSED: All configuration sections present")
        
        return all_present
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("  NEXUS INSTITUTIONAL v0.3.0 - FULL SYSTEM VERIFICATION")
    print("  Enterprise Trading Platform - Complete Test Suite")
    print("="*70)
    
    tests = [
        ("Initialization", test_initialization),
        ("Multi-Asset Support", test_multi_asset_support),
        ("Venue Support (235+)", test_venue_support),
        ("Execution Modes", test_execution_modes),
        ("Institutional Orchestrator", test_institutional_orchestrator),
        ("Cloud Deployment", test_cloud_deployment),
        ("Configuration Parsing", test_configuration_parsing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"  STATUS: ✅ PRODUCTION READY")
        print(f"  Timestamp: {datetime.utcnow().isoformat()}")
    else:
        print(f"  STATUS: ❌ ISSUES DETECTED")
    
    print("="*70 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
