import pandas as pd
import matplotlib.pyplot as plt

def plot_equity_curve(equity: pd.Series, title="Fund Equity Curve"):
    plt.figure(figsize=(10, 5))
    plt.plot(equity, linewidth=2)
    plt.title(title)
    plt.ylabel("Equity")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()

def plot_drawdown(equity: pd.Series, title="Fund Drawdown"):
    peak = equity.cummax()
    drawdown = equity / peak - 1

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown, color="red")
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()

def allocation_table(weights: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({
        "Weight": weights.round(3),
        "Allocation %": (weights * 100).round(1)
    })
    return table
