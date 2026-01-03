
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from main import main, BacktestRegistry

def run_regimes():
    print("="*60)
    print("RUNNING REGIME ANALYSIS")
    print("="*60)

    regimes = {
        "Full History": {"start_date": "2018-01-01", "end_date": None},
        "Bull Market (2020-2021)": {"start_date": "2020-01-01", "end_date": "2021-12-31"},
        "Bear Market (2022)": {"start_date": "2022-01-01", "end_date": "2022-12-31"},
        "Choppy Recovery (2023)": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "Modern Bull (2024-Present)": {"start_date": "2024-01-01", "end_date": None},
    }

    results = []

    for name, dates in regimes.items():
        print(f"\n>>> Running Regime: {name} ({dates['start_date']} to {dates['end_date'] or 'Present'})")
        
        try:
            # Override configuration for this run
            # Note: main() prints to stdout, we let it flow
            main(config_override={
                "start_date": dates["start_date"],
                "end_date": dates["end_date"],
                "strategy_name": f"regime_{name.lower().replace(' ', '_').split('(')[0]}",
                "validation_mode": True
            })
            
            # Find the latest run to get metrics
            # This is a bit hacky, normally main() would return the run object
            # We assume the registry has the latest run
            registry = BacktestRegistry()
            manifest = registry.list_runs()
            if not manifest:
                print("   [Error] No runs found in registry.")
                continue
                
            latest = manifest[0]
            metrics = latest.get("metrics", {})
            
            results.append({
                "Regime": name,
                "RunID": latest["run_id"],
                "Return": metrics.get("total_return", 0.0),
                "Ann. Return": metrics.get("annualized_return", 0.0) if "annualized_return" in metrics else 0.0, # Manifest might not have this, check meta
                "Ann. Vol": metrics.get("annualized_volatility", 0.0) if "annualized_volatility" in metrics else 0.0,
                "Sharpe": metrics.get("sharpe_ratio", 0.0) if "sharpe_ratio" in metrics else metrics.get("sharpe", 0.0), # handle inconsistent naming
                "MaxDD": metrics.get("max_drawdown", 0.0),
                "Final Eq": metrics.get("final_equity", 0.0)
            })
            
        except Exception as e:
            print(f"   [Error] Failed run for regime {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("REGIME ANALYSIS RESULTS")
    print("="*60)
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Format columns
        df["Return"] = df["Return"].map("{:,.1%}".format)
        df["Ann. Return"] = df["Ann. Return"].map("{:,.1%}".format) # Assuming access to this metric
        df["Ann. Vol"] = df["Ann. Vol"].map("{:,.1%}".format)
        df["Sharpe"] = df["Sharpe"].map("{:.2f}".format)
        df["MaxDD"] = df["MaxDD"].map("{:,.1%}".format)
        df["Final Eq"] = df["Final Eq"].map("${:,.0f}".format)
        
        print(df.to_string(index=False))
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("output/analysis")
        report_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path / f"regime_analysis_{timestamp}.csv", index=False)
        print(f"\nReport saved to output/analysis/regime_analysis_{timestamp}.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_regimes()
