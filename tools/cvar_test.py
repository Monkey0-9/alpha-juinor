import sqlite3
import numpy as np
import pandas as pd
import json
import os

db_path = r"C:\mini-quant-fund\runtime\institutional_trading.db"
output_path = r"C:\mini-quant-fund\runtime\agent_results\cvartest_report.json"

def run_cvar_test():
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Load ACTIVE symbols
    cur.execute("SELECT symbol FROM trading_eligibility WHERE state='ACTIVE'")
    symbols = [r[0] for r in cur.fetchall()][:20]  # limit for speed and stability

    if not symbols:
        print("No ACTIVE symbols found for CVaR test.")
        return

    # Build returns matrix
    rets = {}
    for s in symbols:
        df = pd.read_sql_query(f"SELECT date, adjusted_close FROM price_history WHERE symbol='{s}' ORDER BY date DESC LIMIT 1260", con)
        if df.empty or len(df) < 50:
            continue
        df = df.sort_values('date')
        df['ret'] = df['adjusted_close'].pct_change()
        rets[s] = df['ret'].dropna()

    if not rets:
        print("No returns data found for CVaR test.")
        return

    # Align returns on dates
    returns_df = pd.DataFrame(rets).dropna()
    if returns_df.empty:
        print("Aligned returns matrix is empty.")
        return

    R = returns_df.values
    T, N = R.shape

    # assume equal weight portfolio
    w = np.ones(N) / N
    port_r = R.dot(w)

    def historical_cvar(returns, alpha=0.95):
        losses = -returns
        var = np.percentile(losses, alpha*100)
        tail = losses[losses >= var]
        cvar = tail.mean() if len(tail) > 0 else var
        return float(var), float(cvar)

    var, cvar = historical_cvar(port_r, alpha=0.95)

    # marginal cvar
    marginal = {}
    for i, s in enumerate(returns_df.columns):
        eps = 1e-3
        w2 = w.copy()
        w2[i] += eps
        # Re-normalize? The user template didn't re-normalize, just added eps.
        port2 = R.dot(w2)
        var2, cvar2 = historical_cvar(port2, 0.95)
        marginal[s] = float(cvar2 - cvar)

    # Status checks
    # portfolio.cvar <= CVAR_LIMIT (default 0.06)
    # marginal_cvar <= marginal_limit (default 0.01)

    CVAR_LIMIT = 0.06
    MARGINAL_LIMIT = 0.01

    pass_fail = {
        "portfolio_cvar_pass": cvar <= CVAR_LIMIT,
        "marginal_cvar_passes": {s: m <= MARGINAL_LIMIT for s, m in marginal.items()}
    }

    out = {
        "portfolio": {"var": var, "cvar": cvar},
        "marginal": marginal,
        "pass_fail": pass_fail,
        "limits": {"cvar": CVAR_LIMIT, "marginal": MARGINAL_LIMIT}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"CVaR test report written to {output_path}")
    con.close()

if __name__ == "__main__":
    run_cvar_test()
