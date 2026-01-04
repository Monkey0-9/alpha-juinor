import pandas as pd
import numpy as np

TRANSACTION_COST = 0.0005   # 0.05%
SLIPPAGE = 0.0002           # 0.02%

def run_backtest(
    prices: pd.Series,
    invest_permission: pd.Series,
    capital: float = 1_000_000,
    position_fraction: float = 0.25
) -> pd.DataFrame:
    """
    Long-only, cash-or-invest backtest
    """

    # Ensure Series (defensive)
    prices = prices.squeeze()
    invest_permission = invest_permission.squeeze()

    # Align data
    df = pd.concat(
        [
            prices.rename("price"),
            invest_permission.rename("permission")
        ],
        axis=1
    ).dropna()

    # Target exposure
    df["target_exposure"] = df["permission"] * position_fraction

    # Exposure changes (trades)
    df["prev_exposure"] = df["target_exposure"].shift(1).fillna(0)
    df["exposure_change"] = df["target_exposure"] - df["prev_exposure"]

    # Trading costs
    df["trade_cost"] = (
        df["exposure_change"].abs()
        * capital
        * (TRANSACTION_COST + SLIPPAGE)
    )

    # Market returns
    df["market_return"] = df["price"].pct_change(fill_method=None).fillna(0)

    # Strategy returns
    df["strategy_return"] = df["prev_exposure"] * df["market_return"]

    # Net returns after costs
    df["net_return"] = df["strategy_return"] - (df["trade_cost"] / capital)

    # Equity curve
    df["equity"] = capital * (1 + df["net_return"]).cumprod()

    return df
