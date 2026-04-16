import pytest
import numpy as np
from src.options.greeks_calculator import RealTimeGreeksCalculator
from src.options.volatility_surface import VolatilitySurfaceEngine

def test_black_scholes_delta():
    calc = RealTimeGreeksCalculator()
    # At-the-money call should have delta close to 0.5
    g = calc.calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert 0.5 < g.delta < 0.7 # Approx BS delta for ATM

def test_black_scholes_put_delta():
    calc = RealTimeGreeksCalculator()
    g = calc.calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert -0.5 < g.delta < -0.3

def test_svi_calibration_smoke():
    engine = VolatilitySurfaceEngine()
    strikes = [90, 95, 100, 105, 110]
    vols = [0.25, 0.22, 0.20, 0.21, 0.24]
    res = engine.build_surface("TEST", strikes, vols, 100)
    assert res["status"] == "success"
    
    interp_vol = engine.get_vol("TEST", 100, 100)
    assert 0.1 < interp_vol < 0.3
