import os
import sys
import numpy as np
import pandas as pd

mingw_bin = os.path.expanduser(r"~\scoop\apps\mingw\current\bin")
if os.path.exists(mingw_bin):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(mingw_bin)
    else:
        os.environ["PATH"] = mingw_bin + os.pathsep + os.environ["PATH"]

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "nexus", "cpp_extensions")),
)

from nexus.portfolio.optimization import PortfolioOptimizer  # noqa: E402

opt = PortfolioOptimizer()
symbols = ["AAPL", "MSFT"]
signals = [0.8, -0.6]

# Needs historical data to trigger correlation penalty
df_aapl = pd.DataFrame({"close": np.linspace(100, 150, 50)})
df_msft = pd.DataFrame({"close": np.linspace(200, 300, 50)})  # Highly correlated
hist_data = {"AAPL": df_aapl, "MSFT": df_msft}

weights = opt.optimize_weights(symbols, signals, hist_data)
print("WEIGHTS:", weights)
