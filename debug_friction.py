#!/usr/bin/env python3
"""
DEBUG FRICTION
=============
"""
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf

# Import System
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy_factory.manager import StrategyManager
from allocator.meta_controller import MetaController
from institutional_portfolio_validation import run_portfolio_backtest

def debug():
    print("DEBUGGING FRICTION ON SPY")
    data = yf.download("SPY", start="2005-01-01", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        spy = data['Close']['SPY']
    else:
        spy = data['Close']
    spy = spy.dropna()

    mgr = StrategyManager()
    controller = MetaController()

    # 1. Run with 0 Friction (Gross)
    gross = run_portfolio_backtest(spy, mgr, controller, friction_mult=0.0)
    print(f"Gross Sharpe: {gross['sharpe']:.2f}")

    # 2. Run with 1x Friction (15bps)
    net_1x = run_portfolio_backtest(spy, mgr, controller, friction_mult=1.0)
    print(f"Net (1x - 15bps) Sharpe: {net_1x['sharpe']:.2f}")

    # 3. Run with 2x Friction (30bps - The Stress Test)
    net_2x = run_portfolio_backtest(spy, mgr, controller, friction_mult=2.0)
    print(f"Net (2x - 30bps) Sharpe: {net_2x['sharpe']:.2f}")

    decay_1x = (gross['sharpe'] - net_1x['sharpe']) / gross['sharpe']
    decay_2x = (gross['sharpe'] - net_2x['sharpe']) / gross['sharpe']

    print(f"Decay 1x: {decay_1x:.1%}")
    print(f"Decay 2x: {decay_2x:.1%}")

    if net_2x['sharpe'] > 0.3:
        print("PASS: Strategy is robust enough.")
    else:
        print("FAIL: Strategy killed by costs.")

if __name__ == "__main__":
    debug()
