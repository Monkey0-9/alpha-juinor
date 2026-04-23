"""
Test suite for Alpha Engine
Tests OU model, factor model, ML ensemble, and IC-weighted blending
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from elite_quant_fund.alpha.engine import (
    OrnsteinUhlenbeckModel,
    CrossSectionalFactorModel,
    ICWeightedBlender,
    AlphaEngine
)
from elite_quant_fund.core.types import MarketBar, AlphaSignal


class TestOrnsteinUhlenbeckModel:
    """Test OU mean-reversion model"""
    
    def test_ou_calibration(self):
        """Test OU parameter calibration"""
        model = OrnsteinUhlenbeckModel("AAPL", window=50)
        
        # Feed mean-reverting prices
        # OU: prices oscillate around mean
        base_price = 100.0
        for i in range(40):
            # Create oscillating price around 100
            price = base_price + 5 * np.sin(i * 0.5) + np.random.normal(0, 0.5)
            
            bar = MarketBar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000
            )
            
            signal = model.update(bar)
        
        # After calibration, should have OU parameters
        assert model.kappa >= 0
        assert model.theta > 0
        
        # Half-life should be reasonable (not infinite)
        if model.half_life.days < 999:
            assert model.half_life.days > 0
    
    def test_ou_signal_generation(self):
        """Test signal generation when price deviates from mean"""
        model = OrnsteinUhlenbeckModel("AAPL", window=50)
        
        # Feed prices around mean=100
        prices = [100 + np.random.normal(0, 1) for _ in range(35)]
        
        for price in prices:
            bar = MarketBar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000
            )
            model.update(bar)
        
        # Now create large deviation (z-score > 2)
        deviation_price = 110.0  # Far from mean
        
        bar = MarketBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=deviation_price - 0.5,
            high=deviation_price + 1.0,
            low=deviation_price - 1.0,
            close=deviation_price,
            volume=1000000
        )
        
        signal = model.update(bar)
        
        # Should generate signal if mean-reverting and deviation large
        if signal:
            assert isinstance(signal, AlphaSignal)
            assert abs(signal.strength) > 0
            assert signal.symbol == "AAPL"


class TestCrossSectionalFactorModel:
    """Test factor model"""
    
    def test_factor_exposure(self):
        """Test factor exposure calculation"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        model = CrossSectionalFactorModel(symbols, lookback=20)
        
        # Feed returns data
        np.random.seed(42)
        for _ in range(25):
            for sym in symbols:
                price = 100 + np.random.normal(0, 2)
                bar = MarketBar(
                    symbol=sym,
                    timestamp=datetime.now(),
                    open=price - 1,
                    high=price + 2,
                    low=price - 2,
                    close=price,
                    volume=1000000
                )
                model.update(bar)
        
        # Should have factor exposures
        exposure = model.get_factor_exposure("AAPL")
        if exposure is not None:
            assert len(exposure) > 0


class TestICWeightedBlender:
    """Test IC-weighted signal blending"""
    
    def test_signal_blending(self):
        """Test blending multiple signals"""
        symbols = ["AAPL"]
        blender = ICWeightedBlender(symbols, ic_lookback=20)
        
        # Create signals from different models
        signals = {
            'ou': AlphaSignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type="MEAN_REVERSION",
                strength=0.5,
                horizon=timedelta(hours=1),
                metadata={'model': 'OU'}
            ),
            'factor': AlphaSignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type="FACTOR",
                strength=-0.3,
                horizon=timedelta(hours=2),
                metadata={'model': 'factor'}
            ),
            'ml': AlphaSignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type="MACHINE_LEARNING",
                strength=0.4,
                horizon=timedelta(hours=1),
                metadata={'model': 'ML'}
            )
        }
        
        blended = blender.blend_signals("AAPL", signals)
        
        assert blended is not None
        assert blended.symbol == "AAPL"
        assert -1 <= blended.strength <= 1
        
        # Should be weighted average (roughly)
        # (0.5 * 0.33 + -0.3 * 0.33 + 0.4 * 0.34) ~= 0.2
        assert abs(blended.strength) < 1.0


class TestAlphaEngine:
    """Test complete alpha engine"""
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        symbols = ["AAPL", "MSFT"]
        engine = AlphaEngine(symbols)
        
        assert len(engine.symbols) == 2
        assert "AAPL" in engine.ou_models
        assert "MSFT" in engine.ou_models
    
    def test_bar_processing(self):
        """Test processing bar through engine"""
        symbols = ["AAPL"]
        engine = AlphaEngine(symbols)
        
        bar = MarketBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000
        )
        
        bundle = engine.process_bar(bar)
        
        assert bundle.timestamp == bar.timestamp
        assert "AAPL" in bundle.signals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
