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


def assess_portfolio():
    print("=" * 70)
    print("VAST INTELLIGENCE: Portfolio Assessment")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Known positions from system
    positions = [
        {"symbol": "SIRI", "qty": 9057.8359, "cost": 20.60, "priority": "HIGH"},
        {"symbol": "BKNG", "qty": 296.8295, "cost": 5118.60, "priority": "MEDIUM"},
        {"symbol": "AAPL", "qty": 6.0, "cost": 257.86, "priority": "LOW"},
    ]

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
                print(f"{symbol:<8} Data unavailable")

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

    # Position-level recommendations
    print("\nEXIT PRIORITY ANALYSIS:")
    print("-" * 40)

    # Sort by P&L (worst first)
    sorted_results = sorted(results, key=lambda x: x.get("pnl", 0))

    for i, r in enumerate(sorted_results, 1):
        action = "EXIT" if r["pnl"] < 0 else "HOLD/TRIM"
        urgency = (
            "URGENT"
            if r["pnl_pct"] < -10
            else ("MODERATE" if r["pnl_pct"] < 0 else "LOW")
        )
        print(f"{i}. {r['symbol']}: {action} ({urgency})")
        print(f"   P&L: ${r['pnl']:,.2f} ({r['pnl_pct']:.2f}%)")
        print(f"   Value: ${r['current_val']:,.2f}")
        print()

    return results


if __name__ == "__main__":
    assess_portfolio()
