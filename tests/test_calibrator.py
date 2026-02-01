# tests/test_calibrator.py
from execution.calibrator import analyze_fills, update_tc_params
import csv, os

def test_calibrator(tmp_path):
    f = tmp_path / "fills.csv"
    with open(f, "w", newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=["expected_price","fill_price","qty"])
        w.writeheader()
        w.writerow({"expected_price":100.0, "fill_price":100.2, "qty":10})
        w.writerow({"expected_price":100.0, "fill_price":99.8, "qty":5})
    stats = analyze_fills(str(f))
    assert stats["count"] == 2
    assert "mean_bps" in stats
    assert "std_bps" in stats
    params = update_tc_params(stats, out_path=str(tmp_path/"tc.json"))
    assert "base_spread_bps" in params
    assert "impact_coeff" in params

def test_calibrator_empty_file(tmp_path):
    f = tmp_path / "fills_empty.csv"
    with open(f, "w", newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=["expected_price","fill_price","qty"])
        w.writeheader()
    stats = analyze_fills(str(f))
    assert stats["count"] == 0

def test_calibrator_multiple_fills(tmp_path):
    f = tmp_path / "fills_multi.csv"
    with open(f, "w", newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=["expected_price","fill_price","qty"])
        w.writeheader()
        for i in range(100):
            exp = 100.0
            fill = 100.0 + (i % 10 - 5) * 0.01  # slippage between -0.05 and +0.04
            w.writerow({"expected_price":exp, "fill_price":fill, "qty":i+1})
    stats = analyze_fills(str(f))
    assert stats["count"] == 100
    assert "p50" in stats
    assert "p95" in stats
    assert "p99" in stats
