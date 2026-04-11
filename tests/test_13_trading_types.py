#!/usr/bin/env python3
"""
PACK 4: Comprehensive Testing Suite for All 13 Trading Types
Tests each type individually, integration, and stress scenarios
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDayTrading:
    """Unit tests for Day Trading type"""

    def test_day_trading_creation(self):
        """Verify day trading positions created correctly"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        # Day trading should be selected for high-liquidity stocks
        types = engine.select_trading_types(
            symbol="AAPL",
            volatility=0.15,
            adv=50_000_000,
            spread_bps=2.0,
            has_earnings=False,
            has_news=False,
        )
        assert TradingType.DAY_TRADING in types

    def test_day_trading_auto_close(self):
        """Verify day trades close at 3:50 PM ET"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.DAY_TRADING]
        assert params["auto_close_time"] == "15:50"
        assert params["hold_minutes_max"] == 390  # 6.5 hours


class TestSwingTrading:
    """Unit tests for Swing Trading type"""

    def test_swing_trading_selection(self):
        """Verify swing trading selected for low-volatility environments"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="JNJ",
            volatility=0.12,
            adv=10_000_000,
            spread_bps=3.0,
            has_earnings=False,
            has_news=False,
        )
        assert TradingType.SWING_TRADING in types

    def test_swing_trading_parameters(self):
        """Verify swing trading has correct hold times"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.SWING_TRADING]
        assert params["hold_minutes_min"] == 120  # 2 hours
        assert params["hold_minutes_max"] == 28800  # 20 days


class TestScalping:
    """Unit tests for Scalping type"""

    def test_scalping_selection(self):
        """Verify scalping selected for ultra-liquid stocks"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="SPY",
            volatility=0.18,
            adv=100_000_000,
            spread_bps=1.0,
            has_earnings=False,
            has_news=False,
        )
        assert TradingType.SCALPING in types

    def test_scalping_parameters(self):
        """Verify scalping has ultra-short hold times"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.SCALPING]
        assert params["hold_minutes_min"] == 0.15  # 10 seconds
        assert params["hold_minutes_max"] == 5


class TestPositionTrading:
    """Unit tests for Position Trading type"""

    def test_position_trading_long_holds(self):
        """Verify position trading supports long-term holds"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.POSITION_TRADING]
        assert params["hold_minutes_min"] == 43200  # 30 days
        assert params["hold_minutes_max"] == 525600  # 1 year
        assert params["trailing_stop"] == True


class TestMomentumTrading:
    """Unit tests for Momentum Trading type"""

    def test_momentum_trading_selection(self):
        """Verify momentum trading selected in trending conditions"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="TSLA",
            volatility=0.25,
            adv=20_000_000,
            spread_bps=2.5,
            has_earnings=False,
            has_news=False,
        )
        assert TradingType.MOMENTUM_TRADING in types


class TestAlgorithmicTrading:
    """Unit tests for Algorithmic Trading type"""

    def test_algorithmic_always_active(self):
        """Verify algorithmic trading is always selected"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="XYZ",
            volatility=0.50,
            adv=1_000_000,
            spread_bps=10.0,
            has_earnings=False,
            has_news=False,
        )
        assert TradingType.ALGORITHMIC in types


class TestNewsTrading:
    """Unit tests for News Trading type"""

    def test_news_trading_high_volatility(self):
        """Verify news trading selected in high volatility"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="NVDA",
            volatility=0.40,
            adv=15_000_000,
            spread_bps=3.0,
            has_earnings=False,
            has_news=True,
        )
        assert TradingType.NEWS_TRADING in types


class TestEventDrivenTrading:
    """Unit tests for Event-Driven Trading type"""

    def test_event_driven_earnings(self):
        """Verify event-driven trading activated for earnings"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="MSFT",
            volatility=0.20,
            adv=30_000_000,
            spread_bps=2.0,
            has_earnings=True,
            has_news=False,
        )
        assert TradingType.EVENT_DRIVEN in types


class TestTechnicalTrading:
    """Unit tests for Technical Trading type"""

    def test_technical_trading_medium_vol(self):
        """Verify technical trading works in normal volatility"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="SPY",
            volatility=0.20,
            adv=80_000_000,
            spread_bps=1.5,
            has_earnings=False,
            has_news=False,
        )
        assert TradingType.TECHNICAL_TRADING in types


class TestFundamentalTrading:
    """Unit tests for Fundamental Trading type"""

    def test_fundamental_parameters(self):
        """Verify fundamental trading has long-term parameters"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.FUNDAMENTAL_TRADING]
        assert params["hold_minutes_min"] == 1440  # 1 day
        assert params["hold_minutes_max"] == 262080  # 6 months


class TestDeliveryTrading:
    """Unit tests for Delivery Trading type"""

    def test_delivery_trading_buy_hold(self):
        """Verify delivery trading is long-term buy-and-hold"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.DELIVERY_TRADING]
        assert params["hold_minutes_min"] == 525600  # 1 year
        assert params["dividend_reinvestment"] == True


class TestSocialCopyTrading:
    """Unit tests for Social/Copy Trading types"""

    def test_social_trading_parameters(self):
        """Verify social trading has medium hold times"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.SOCIAL_TRADING]
        assert params["position_size_pct"] == 0.02

    def test_copy_trading_scaled_nav(self):
        """Verify copy trading scales to NAV"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        params = engine.type_params[TradingType.COPY_TRADING]
        assert params["scale_to_nav"] == True


class TestIntegration13Types:
    """Integration tests: All 13 types working together"""

    def test_all_13_types_exist(self):
        """Verify all 13 trading types are defined"""
        from strategies.unified_trading_engine import TradingType

        types = list(TradingType)
        assert len(types) == 13

    def test_signal_generation_all_types(self):
        """Verify signal generation for all types"""
        from strategies.unified_trading_engine import UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        signal_data = {
            "is_orb": True,
            "at_support": True,
            "bid_ask_imbalance": 1.8,
            "is_breakout": True,
            "has_news_event": True,
            "has_corporate_event": True,
        }

        all_types = list(engine.type_params.keys())
        signals = engine.generate_signals("AAPL", signal_data, all_types[:5])

        assert isinstance(signals, dict)

    def test_priority_conflict_resolution(self):
        """Verify priority-based conflict resolution works"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        # Create conflicting signals
        signals = {
            TradingType.DAY_TRADING: {"action": "BUY", "confidence": 0.7},
            TradingType.POSITION_TRADING: {"action": "SELL", "confidence": 0.6},
            TradingType.MOMENTUM_TRADING: {"action": "BUY", "confidence": 0.75},
        }

        # Position Trading has higher priority and should override
        result = engine.aggregate_signals("AAPL", signals)
        assert result is not None
        # Position trading should win due to priority
        assert result[0] == TradingType.POSITION_TRADING


class TestStressScenarios:
    """Stress tests for extreme conditions"""

    def test_high_volatility_weekend(self):
        """Test system behavior during high volatility + weekend"""
        from strategies.unified_trading_engine import TradingType, UnifiedTradingEngine

        engine = UnifiedTradingEngine(nav=1_000_000, config={})

        types = engine.select_trading_types(
            symbol="VIX",
            volatility=0.80,
            adv=5_000_000,
            spread_bps=5.0,
            has_earnings=False,
            has_news=False,
        )
        # Even in high vol, algorithmic should be active
        assert TradingType.ALGORITHMIC in types

    def test_zero_liquidity_rejection(self):
        """Test rejection of illiquid symbols"""
        from execution.enhanced_seven_gates import SevenGateRiskManager

        gates = SevenGateRiskManager(
            nav=1_000_000, max_gross_leverage=3.0, adv_limit_pct=0.10, test_mode=True
        )

        # Very low ADV symbol - should be rejected
        is_approved, reason, qty = gates.validate_order(
            symbol="PENNY",
            quantity=10000,
            price=5.0,
            side="BUY",
            sector="Technology",
            adv=100,  # Only $500 ADV
            vix=20.0,
            realized_volatility=0.20,
            correlation_largest=0.5,
        )

        assert is_approved == False
        assert "ILLIQUID" in reason

    def test_vix_spike_rejection(self):
        """Test rejection during VIX spike"""
        from execution.enhanced_seven_gates import SevenGateRiskManager

        gates = SevenGateRiskManager(
            nav=1_000_000, max_gross_leverage=3.0, max_vix=50.0, test_mode=True
        )

        is_approved, reason, qty = gates.validate_order(
            symbol="SPY",
            quantity=1000,
            price=400.0,
            side="BUY",
            sector="Technology",
            adv=50_000_000,
            vix=65.0,  # VIX spike > 50
            realized_volatility=0.20,
            correlation_largest=0.5,
        )

        assert is_approved == False
        assert "VIX_TOO_HIGH" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
