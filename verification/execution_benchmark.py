#!/usr/bin/env python3
"""
VERIFICATION: EXECUTION BENCHMARK (S-Class Test 2)
==================================================

Benchmarks TWAP Algorithm vs Naive Market Orders.
Simulates market impact and volatility risk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_brownian_motion(start_price, sigma, steps, dt=1/390): # 1 min steps
    t = np.linspace(0, steps*dt, steps)
    w = np.random.standard_normal(size=steps)
    w = np.cumsum(w)*np.sqrt(dt)
    x = (0 - 0.5*sigma**2)*t + sigma*w
    s = start_price*np.exp(x)
    return s

def calculate_impact(quantity, volatility, daily_volume=1000000):
    """Square-root market impact model."""
    # Alpha * Vol * sqrt(Qty / DailyVol)
    # Simplified: 10bps * sqrt(pct_adv)
    pct_adv = quantity / daily_volume
    impact_bps = 50 * np.sqrt(pct_adv) # 50bps coefficient
    return impact_bps / 10000

def run_benchmark():
    print("=" * 60)
    print("     EXECUTION BENCHMARK: TWAP vs MARKET")
    print("=" * 60)

    # Config
    START_PRICE = 100.0
    SIGMA = 0.20 # 20% annualized vol
    ORDER_SIZE = 100000 # Shares (10% of ADV)
    DAILY_VOL = 1000000
    SIMULATIONS = 1000

    market_shortfalls = []
    twap_shortfalls = []

    print(f"Running {SIMULATIONS} simulations...")
    print(f"Order Size: {ORDER_SIZE/DAILY_VOL:.1%} of ADV")

    for i in range(SIMULATIONS):
        # Generate 15-minute price path (15 steps)
        # TWAP duration = 15 mins
        prices = generate_brownian_motion(START_PRICE, SIGMA, 16)
        arrival_price = prices[0]

        # 1. Market Order (Immediate)
        impact = calculate_impact(ORDER_SIZE, SIGMA, DAILY_VOL)
        fill_price_mkt = arrival_price * (1 + impact)
        shortfall_mkt = (fill_price_mkt - arrival_price) / arrival_price
        market_shortfalls.append(shortfall_mkt)

        # 2. TWAP (4 slices over 15 mins: t=0, 5, 10, 15)
        # Slices at indices 0, 5, 10, 15
        slice_indices = [0, 5, 10, 15]
        slice_qty = ORDER_SIZE / 4

        fills = []
        for idx in slice_indices:
            p = prices[idx]
            imp = calculate_impact(slice_qty, SIGMA, DAILY_VOL)
            fill = p * (1 + imp)
            fills.append(fill)

        avg_fill_twap = np.mean(fills)
        shortfall_twap = (avg_fill_twap - arrival_price) / arrival_price
        twap_shortfalls.append(shortfall_twap)

    # Analysis
    avg_mkt = np.mean(market_shortfalls) * 10000 # bps
    avg_twap = np.mean(twap_shortfalls) * 10000 # bps

    print("-" * 60)
    print(f"Avg Market Slippage: {avg_mkt:.2f} bps")
    print(f"Avg TWAP Slippage:   {avg_twap:.2f} bps")
    print(f"Savings:             {avg_mkt - avg_twap:.2f} bps")
    print("-" * 60)

    if avg_twap < avg_mkt:
        print("✅ VERIFICATION PASSED: TWAP reduces slippage.")
    else:
        print("❌ VERIFICATION FAILED: TWAP increased cost (Volatility Risk > Impact Savings).")

if __name__ == "__main__":
    run_benchmark()
