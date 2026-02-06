"""
Enhanced Portfolio Assessment Tool
Fetches live prices and calculates detailed P&L for each position.
"""

import logging
from datetime import datetime

import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def assess_portfolio(target="sharpe"):
    print("=" * 70)
    print(f"VAST INTELLIGENCE: Portfolio Assessment (Target: {target.upper()})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Mocked positions for now (should ideally load from DB)
    positions = [
        {"symbol": "SIRI", "qty": 9057.8359, "cost": 20.60, "priority": "HIGH"},
        {"symbol": "BKNG", "qty": 296.8295, "cost": 5118.60, "priority": "MEDIUM"},
        {"symbol": "AAPL", "qty": 6.0, "cost": 257.86, "priority": "LOW"},
    ]

    # In a real scenario, we would load from database:
    # positions = db.get_active_positions()

    total_invested = 0
    total_current = 0
    total_pnl = 0

    print(
        f"\n{'Symbol':<8} {'Qty':>12} {'Cost':>10} {'Current':>10} "
        f"{'P&L ($)':>14} {'P&L (%)':>10} {'Priority':<8}"
    )
    print("-" * 80)

    results = []

    for pos in positions:
        symbol = pos["symbol"]
        qty = pos["qty"]
        cost = pos["cost"]
        invested = qty * cost
        total_invested += invested

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")

            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
                current_val = qty * current_price
                pnl = current_val - invested
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0

                total_current += current_val
                total_pnl += pnl

                results.append(
                    {
                        "symbol": symbol,
                        "qty": qty,
                        "cost": cost,
                        "current": current_price,
                        "invested": invested,
                        "current_val": current_val,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "priority": pos["priority"],
                    }
                )

                pnl_indicator = "🔴" if pnl < 0 else "🟢"
                print(
                    f"{symbol:<8} {qty:>12.4f} ${cost:>9.2f} ${current_price:>9.2f} "
                    f"{pnl_indicator} ${pnl:>12,.2f} {pnl_pct:>9.2f}% {pos['priority']:<8}"
                )
            else:
                print(f"{symbol:<8} Data unavailable (Using Cost basis for estim)")
                current_price = cost
                results.append(
                    {
                        "symbol": symbol,
                        "qty": qty,
                        "current": cost,
                        "pnl": 0,
                        "pnl_pct": 0,
                    }
                )

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            print(f"{symbol:<8} Error: {e}")

    print("-" * 80)
    total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
    pnl_indicator = "🔴" if total_pnl < 0 else "🟢"

    print(
        f"\n{'TOTALS':<8} {'':>12} {'':>10} {'':>10} "
        f"{pnl_indicator} ${total_pnl:>12,.2f} {total_pnl_pct:>9.2f}%"
    )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total Invested:      ${total_invested:>15,.2f}")
    print(f"Total Current Value: ${total_current:>15,.2f}")
    print(f"Total P&L:           ${total_pnl:>15,.2f} ({total_pnl_pct:.2f}%)")
    print(f"{'='*70}")

    # Optimization Suggestions
    print(f"\nOPTIMIZATION ANALYSIS (Target: {target.upper()})")
    print("-" * 40)
    suggestions = []

    # Mock efficient frontier logic
    if target == "sharpe":
        # Suggest reducing largest singular risk if correlation is unknown
        # Suggest diversifying away from SIRI if it dominates
        siri_weight = (
            (positions[0]["qty"] * positions[0]["cost"]) / total_invested
            if total_invested
            else 0
        )
        if siri_weight > 0.5:
            suggestions.append(f"REDUCE SIRI by 20% (Concentration Risk > 50%)")
            suggestions.append(f"ALLOCATE to Low-Correlated Assets (e.g., GLD, TLT)")
        else:
            suggestions.append("Portfolio is well balanced.")
    elif target == "volatility":
        suggestions.append("REDUCE High Volatility Assets (BKNG)")

    for s in suggestions:
        print(f"» {s}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization-target", type=str, default="sharpe")
    args = parser.parse_args()
    assess_portfolio(target=args.optimization_target)
