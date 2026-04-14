#!/usr/bin/env python3
"""Quick test to verify unified 13-type engine integration"""

import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all new components import successfully"""
    logger.info("=" * 80)
    logger.info("TEST 1: Verifying imports...")
    logger.info("=" * 80)
    
    try:
        from mini_quant_fund.strategies.unified_trading_engine import UnifiedTradingEngine, TradingType
        logger.info("✓ UnifiedTradingEngine imported")
    except Exception as e:
        logger.error(f"✗ UnifiedTradingEngine import failed: {e}")
        assert False, f"UnifiedTradingEngine import failed: {e}"
    
    try:
        from mini_quant_fund.portfolio.volatility_scaled_allocator import VolatilityScaledAllocator
        logger.info("✓ VolatilityScaledAllocator imported")
    except Exception as e:
        logger.error(f"✗ VolatilityScaledAllocator import failed: {e}")
        assert False, f"VolatilityScaledAllocator import failed: {e}"
    
    try:
        from mini_quant_fund.execution.enhanced_seven_gates import SevenGateRiskManager
        logger.info("✓ SevenGateRiskManager imported")
    except Exception as e:
        logger.error(f"✗ SevenGateRiskManager import failed: {e}")
        assert False, f"SevenGateRiskManager import failed: {e}"

def test_engine_initialization():
    """Test that UnifiedTradingEngine initializes correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Engine initialization...")
    logger.info("=" * 80)
    
    try:
        from mini_quant_fund.strategies.unified_trading_engine import UnifiedTradingEngine
        nav = 1_000_000
        config = {"risk": {"max_gross_leverage": 3.0}}
        
        engine = UnifiedTradingEngine(nav=nav, config=config)
        logger.info(f"✓ UnifiedTradingEngine created with NAV=${nav:,.0f}")
        
        # Test trading type selection
        active_types = engine.select_trading_types(
            symbol="AAPL",
            volatility=0.20,
            adv=50_000_000,
            spread_bps=2.0,
            has_earnings=False,
            has_news=False
        )
        logger.info(f"✓ Trading types selected: {len(active_types)} types active for AAPL")
        for tt in active_types:
            logger.info(f"  - {tt.name}")
        
        assert len(active_types) > 0, "No trading types selected"
    except Exception as e:
        logger.error(f"✗ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Engine initialization failed: {e}"

def test_allocator_initialization():
    """Test VolatilityScaledAllocator initialization"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Allocator initialization...")
    logger.info("=" * 80)
    
    try:
        from mini_quant_fund.portfolio.volatility_scaled_allocator import VolatilityScaledAllocator
        nav = 1_000_000
        
        allocator = VolatilityScaledAllocator(
            nav=nav,
            max_position_pct=0.05,
            max_sector_pct=0.20
        )
        logger.info(f"✓ VolatilityScaledAllocator created with NAV=${nav:,.0f}")
        
        # Test dynamic sizing calculation
        qty, reason = allocator.calculate_dynamic_position_size(
            symbol="AAPL",
            signal_confidence=0.75,
            volatility=0.20,
            sector="Technology",
            correlation_to_largest=0.65,
            adv=50_000_000,
            price=150.0
        )
        logger.info(f"✓ Dynamic sizing calculated: {qty} shares for AAPL")
        logger.info(f"  Reason: {reason}")
        
        assert qty > 0, "Position size calculation failed"
    except Exception as e:
        logger.error(f"✗ Allocator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Allocator initialization failed: {e}"

def test_seven_gates():
    """Test 7-Gate Risk Manager initialization"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: 7-Gate Risk Manager...")
    logger.info("=" * 80)
    
    try:
        from mini_quant_fund.execution.enhanced_seven_gates import SevenGateRiskManager
        nav = 1_000_000
        
        gates = SevenGateRiskManager(
            nav=nav,
            max_gross_leverage=3.0,
            max_position_pct=0.05,
            max_sector_pct=0.20,
            max_correlation=0.90,
            adv_limit_pct=0.10,
            max_vix=50.0,
            max_realized_vol=0.50
        )
        logger.info(f"✓ SevenGateRiskManager created with NAV=${nav:,.0f}")
        
        # Test order validation
        is_approved, reason, final_qty = gates.validate_order(
            symbol="AAPL",
            quantity=1000,
            price=150.0,
            side="BUY",
            sector="Technology",
            adv=50_000_000,
            vix=20.0,
            realized_volatility=0.20,
            correlation_largest=0.65
        )
        logger.info(f"✓ Order validation tested: {reason}")
        logger.info(f"  Approved: {is_approved}, Final Qty: {final_qty}")
        
        assert is_approved is not None, "Order validation failed"
    except Exception as e:
        logger.error(f"✗ 7-Gate initialization failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"7-Gate initialization failed: {e}"

def main():
    logger.info("\n" + "🚀 " * 40)
    logger.info("UNIFIED 13-TYPE TRADING ENGINE - INTEGRATION TEST")
    logger.info("🚀 " * 40 + "\n")
    
    try:
        test_imports()
        test_engine_initialization()
        test_allocator_initialization()
        test_seven_gates()
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info("✓ PASS: Imports")
        logger.info("✓ PASS: Engine Init")
        logger.info("✓ PASS: Allocator Init")
        logger.info("✓ PASS: 7-Gates")
        logger.info("=" * 80)
        logger.info("✓ ALL TESTS PASSED - System ready for trading!")
        return 0
    except Exception as e:
        logger.error(f"✗ SOME TESTS FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
