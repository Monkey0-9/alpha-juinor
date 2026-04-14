# execution/calibrator.py
"""
Execution calibrator that consumes fills log and expected-order log and computes realized slippage
statistics (mean, std, tail) and outputs updated tc model parameters to configs/tc_params.json.
Run this weekly or after 5k fills.
Expected input: CSV or JSON lines containing expected_price, fill_price, symbol, qty, ts.
"""
import csv, json, os, math
from typing import List, Dict
import numpy as np

DEFAULT_PATH = "data/fills.csv"
TC_PARAMS_PATH = "configs/tc_params.json"

def analyze_fills(fills_path: str = DEFAULT_PATH) -> Dict[str,any]:
    if not os.path.exists(fills_path):
        raise FileNotFoundError(f"{fills_path} missing")
    deltas_bps = []
    with open(fills_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp = float(r.get("expected_price", r.get("expected", 0)))
            fill = float(r.get("fill_price", r.get("fill", 0)))
            if exp <= 0: continue
            deltas_bps.append((fill/exp - 1.0) * 10000.0)
    if not deltas_bps:
        return {"count":0}
    arr = np.array(deltas_bps)
    stats = {"count": len(arr), "mean_bps": float(np.mean(arr)), "std_bps": float(np.std(arr)),
             "p50": float(np.percentile(arr,50)), "p95": float(np.percentile(arr,95)), "p99": float(np.percentile(arr,99))}
    return stats

def update_tc_params(stats: Dict[str,any], out_path: str = TC_PARAMS_PATH):
    # naive mapping: base_spread = p50/2 half-spread; impact_coeff tuned from mean difference
    base_spread = max(0.2, abs(stats.get("p50", 0.0))/2.0)
    # impact_coeff scale: shrink mean relative to adv proxy; keep conservative floor
    impact = max(0.00005, abs(stats.get("mean_bps",0.0))/10000.0 / 10.0)
    params = {"base_spread_bps": base_spread, "impact_coeff": impact, "min_slippage_bps": max(0.2, base_spread*0.5)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return params

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fills", default=DEFAULT_PATH)
    p.add_argument("--out", default=TC_PARAMS_PATH)
    args = p.parse_args()
    s = analyze_fills(args.fills)
    print("stats:", s)
    if s.get("count",0) > 0:
        new = update_tc_params(s, args.out)
        print("updated tc params:", new)
