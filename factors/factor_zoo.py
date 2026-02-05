"""
Factor Zoo - 50+ Orthogonal Alpha Factors.

Based on academic research and institutional practices:
- Fama-French factors (value, size, momentum)
- Quality factors (ROE, margins, leverage)
- Technical factors (RSI, MACD, Bollinger)
- Liquidity factors (Amihud, turnover)
- Alternative factors (sentiment, insider activity)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FactorResult:
    """Result of factor calculation."""
    name: str
    value: float
    z_score: float
    percentile: float
    category: str


class FactorZoo:
    """
    Comprehensive factor library for alpha generation.

    Categories:
    - VALUE: Book-to-market, earnings yield, cash flow yield
    - MOMENTUM: 12-1, 6-1, 3-1 month returns
    - QUALITY: ROE, ROA, margins, leverage
    - VOLATILITY: Realized vol, idiosyncratic vol, beta
    - LIQUIDITY: Amihud ratio, turnover, spread
    - SIZE: Market cap, log market cap
    - TECHNICAL: RSI, MACD, Bollinger bands
    - ALTERNATIVE: Sentiment, insider activity, short interest
    """

    FACTOR_DEFINITIONS = {
        # VALUE FACTORS
        "book_to_market": {"category": "VALUE", "higher_is_better": True},
        "earnings_yield": {"category": "VALUE", "higher_is_better": True},
        "cash_flow_yield": {"category": "VALUE", "higher_is_better": True},
        "dividend_yield": {"category": "VALUE", "higher_is_better": True},
        "sales_to_price": {"category": "VALUE", "higher_is_better": True},

        # MOMENTUM FACTORS
        "momentum_12_1": {"category": "MOMENTUM", "higher_is_better": True},
        "momentum_6_1": {"category": "MOMENTUM", "higher_is_better": True},
        "momentum_3_1": {"category": "MOMENTUM", "higher_is_better": True},
        "momentum_1_0": {"category": "MOMENTUM", "higher_is_better": True},
        "momentum_reversal": {"category": "MOMENTUM", "higher_is_better": False},

        # QUALITY FACTORS
        "roe": {"category": "QUALITY", "higher_is_better": True},
        "roa": {"category": "QUALITY", "higher_is_better": True},
        "gross_margin": {"category": "QUALITY", "higher_is_better": True},
        "operating_margin": {"category": "QUALITY", "higher_is_better": True},
        "net_margin": {"category": "QUALITY", "higher_is_better": True},
        "asset_turnover": {"category": "QUALITY", "higher_is_better": True},
        "debt_to_equity": {"category": "QUALITY", "higher_is_better": False},
        "current_ratio": {"category": "QUALITY", "higher_is_better": True},
        "interest_coverage": {"category": "QUALITY", "higher_is_better": True},
        "earnings_stability": {"category": "QUALITY", "higher_is_better": True},

        # VOLATILITY FACTORS
        "realized_vol_20": {"category": "VOLATILITY", "higher_is_better": False},
        "realized_vol_60": {"category": "VOLATILITY", "higher_is_better": False},
        "idiosyncratic_vol": {"category": "VOLATILITY", "higher_is_better": False},
        "beta": {"category": "VOLATILITY", "higher_is_better": None},
        "downside_beta": {"category": "VOLATILITY", "higher_is_better": False},
        "vol_of_vol": {"category": "VOLATILITY", "higher_is_better": False},

        # LIQUIDITY FACTORS
        "amihud_illiquidity": {"category": "LIQUIDITY", "higher_is_better": False},
        "turnover": {"category": "LIQUIDITY", "higher_is_better": True},
        "dollar_volume": {"category": "LIQUIDITY", "higher_is_better": True},
        "bid_ask_spread": {"category": "LIQUIDITY", "higher_is_better": False},
        "volume_volatility": {"category": "LIQUIDITY", "higher_is_better": False},

        # SIZE FACTORS
        "market_cap": {"category": "SIZE", "higher_is_better": None},
        "log_market_cap": {"category": "SIZE", "higher_is_better": None},
        "enterprise_value": {"category": "SIZE", "higher_is_better": None},

        # TECHNICAL FACTORS
        "rsi_14": {"category": "TECHNICAL", "higher_is_better": None},
        "rsi_divergence": {"category": "TECHNICAL", "higher_is_better": True},
        "macd_signal": {"category": "TECHNICAL", "higher_is_better": True},
        "macd_histogram": {"category": "TECHNICAL", "higher_is_better": True},
        "bollinger_pct": {"category": "TECHNICAL", "higher_is_better": None},
        "atr_ratio": {"category": "TECHNICAL", "higher_is_better": False},
        "adx": {"category": "TECHNICAL", "higher_is_better": True},
        "obv_trend": {"category": "TECHNICAL", "higher_is_better": True},
        "vwap_deviation": {"category": "TECHNICAL", "higher_is_better": None},
        "support_distance": {"category": "TECHNICAL", "higher_is_better": True},
        "resistance_distance": {"category": "TECHNICAL", "higher_is_better": True},

        # ALTERNATIVE FACTORS
        "sentiment_score": {"category": "ALTERNATIVE", "higher_is_better": True},
        "news_sentiment": {"category": "ALTERNATIVE", "higher_is_better": True},
        "insider_buying": {"category": "ALTERNATIVE", "higher_is_better": True},
        "short_interest": {"category": "ALTERNATIVE", "higher_is_better": False},
        "analyst_revisions": {"category": "ALTERNATIVE", "higher_is_better": True},
        "earnings_surprise": {"category": "ALTERNATIVE", "higher_is_better": True},
        "institutional_ownership": {"category": "ALTERNATIVE", "higher_is_better": True},
    }

    def __init__(self):
        self.factor_cache: Dict[str, pd.DataFrame] = {}
        self.z_score_lookback = 252  # 1 year for z-score calculation

    # =========================================================================
    # MOMENTUM FACTORS
    # =========================================================================

    def calc_momentum_12_1(self, prices: pd.Series) -> float:
        """12-month momentum excluding last month (classic Jegadeesh-Titman)."""
        if len(prices) < 252:
            return np.nan
        ret_12m = prices.iloc[-21] / prices.iloc[-252] - 1
        return float(ret_12m)

    def calc_momentum_6_1(self, prices: pd.Series) -> float:
        """6-month momentum excluding last month."""
        if len(prices) < 126:
            return np.nan
        ret_6m = prices.iloc[-21] / prices.iloc[-126] - 1
        return float(ret_6m)

    def calc_momentum_3_1(self, prices: pd.Series) -> float:
        """3-month momentum excluding last month."""
        if len(prices) < 63:
            return np.nan
        ret_3m = prices.iloc[-21] / prices.iloc[-63] - 1
        return float(ret_3m)

    def calc_momentum_1_0(self, prices: pd.Series) -> float:
        """1-month momentum (short-term reversal candidate)."""
        if len(prices) < 21:
            return np.nan
        ret_1m = prices.iloc[-1] / prices.iloc[-21] - 1
        return float(ret_1m)

    # =========================================================================
    # VOLATILITY FACTORS
    # =========================================================================

    def calc_realized_vol_20(self, prices: pd.Series) -> float:
        """20-day realized volatility (annualized)."""
        if len(prices) < 21:
            return np.nan
        returns = prices.pct_change().dropna().iloc[-20:]
        return float(returns.std() * np.sqrt(252))

    def calc_realized_vol_60(self, prices: pd.Series) -> float:
        """60-day realized volatility (annualized)."""
        if len(prices) < 61:
            return np.nan
        returns = prices.pct_change().dropna().iloc[-60:]
        return float(returns.std() * np.sqrt(252))

    def calc_beta(
        self, prices: pd.Series, market_prices: pd.Series, window: int = 60
    ) -> float:
        """Beta relative to market."""
        if len(prices) < window or len(market_prices) < window:
            return np.nan

        stock_ret = prices.pct_change().dropna().iloc[-window:]
        mkt_ret = market_prices.pct_change().dropna().iloc[-window:]

        # Align indices
        common = stock_ret.index.intersection(mkt_ret.index)
        if len(common) < 20:
            return np.nan

        cov = np.cov(stock_ret.loc[common], mkt_ret.loc[common])[0, 1]
        var = mkt_ret.loc[common].var()

        return float(cov / var) if var > 0 else np.nan

    def calc_idiosyncratic_vol(
        self, prices: pd.Series, market_prices: pd.Series, window: int = 60
    ) -> float:
        """Idiosyncratic volatility (residual from market model)."""
        if len(prices) < window or len(market_prices) < window:
            return np.nan

        stock_ret = prices.pct_change().dropna().iloc[-window:]
        mkt_ret = market_prices.pct_change().dropna().iloc[-window:]

        common = stock_ret.index.intersection(mkt_ret.index)
        if len(common) < 20:
            return np.nan

        beta = self.calc_beta(prices, market_prices, window)
        if np.isnan(beta):
            return np.nan

        residuals = stock_ret.loc[common] - beta * mkt_ret.loc[common]
        return float(residuals.std() * np.sqrt(252))

    # =========================================================================
    # TECHNICAL FACTORS
    # =========================================================================

    def calc_rsi_14(self, prices: pd.Series) -> float:
        """14-period RSI."""
        if len(prices) < 15:
            return np.nan

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def calc_macd_signal(self, prices: pd.Series) -> float:
        """MACD histogram (MACD - Signal)."""
        if len(prices) < 35:
            return np.nan

        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal

        return float(histogram.iloc[-1])

    def calc_bollinger_pct(self, prices: pd.Series, window: int = 20) -> float:
        """Bollinger band percentage (0 = lower, 1 = upper)."""
        if len(prices) < window:
            return np.nan

        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std

        pct = (prices.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        return float(np.clip(pct, 0, 1))

    def calc_vwap_deviation(
        self, prices: pd.Series, volumes: pd.Series, window: int = 20
    ) -> float:
        """Deviation from VWAP."""
        if len(prices) < window or len(volumes) < window:
            return np.nan

        typical = prices.iloc[-window:]
        vol = volumes.iloc[-window:]

        vwap = (typical * vol).sum() / vol.sum()
        deviation = (prices.iloc[-1] - vwap) / vwap
        return float(deviation)

    # =========================================================================
    # LIQUIDITY FACTORS
    # =========================================================================

    def calc_amihud_illiquidity(
        self, prices: pd.Series, volumes: pd.Series, window: int = 20
    ) -> float:
        """Amihud illiquidity ratio (|return| / dollar volume)."""
        if len(prices) < window or len(volumes) < window:
            return np.nan

        returns = prices.pct_change().abs().iloc[-window:]
        dollar_vol = (prices * volumes).iloc[-window:]

        # Avoid division by zero
        dollar_vol = dollar_vol.replace(0, np.nan)
        ratio = (returns / dollar_vol).mean()

        return float(ratio * 1e6)  # Scale for readability

    def calc_turnover(
        self, volumes: pd.Series, shares_outstanding: float, window: int = 20
    ) -> float:
        """Average daily turnover ratio."""
        if len(volumes) < window or shares_outstanding <= 0:
            return np.nan

        avg_vol = volumes.iloc[-window:].mean()
        return float(avg_vol / shares_outstanding)

    # =========================================================================
    # COMPOSITE FACTOR CALCULATION
    # =========================================================================

    def calculate_all_factors(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        market_prices: Optional[pd.Series] = None,
        fundamentals: Optional[Dict[str, float]] = None
    ) -> Dict[str, FactorResult]:
        """
        Calculate all available factors for a symbol.

        Returns:
            Dict of factor_name -> FactorResult
        """
        results = {}

        # Momentum
        for name, func in [
            ("momentum_12_1", lambda: self.calc_momentum_12_1(prices)),
            ("momentum_6_1", lambda: self.calc_momentum_6_1(prices)),
            ("momentum_3_1", lambda: self.calc_momentum_3_1(prices)),
            ("momentum_1_0", lambda: self.calc_momentum_1_0(prices)),
        ]:
            try:
                value = func()
                results[name] = self._create_result(name, value)
            except Exception as e:
                logger.debug(f"Factor {name} failed: {e}")

        # Volatility
        results["realized_vol_20"] = self._create_result(
            "realized_vol_20", self.calc_realized_vol_20(prices)
        )
        results["realized_vol_60"] = self._create_result(
            "realized_vol_60", self.calc_realized_vol_60(prices)
        )

        if market_prices is not None:
            results["beta"] = self._create_result(
                "beta", self.calc_beta(prices, market_prices)
            )
            results["idiosyncratic_vol"] = self._create_result(
                "idiosyncratic_vol",
                self.calc_idiosyncratic_vol(prices, market_prices)
            )

        # Technical
        results["rsi_14"] = self._create_result(
            "rsi_14", self.calc_rsi_14(prices)
        )
        results["macd_signal"] = self._create_result(
            "macd_signal", self.calc_macd_signal(prices)
        )
        results["bollinger_pct"] = self._create_result(
            "bollinger_pct", self.calc_bollinger_pct(prices)
        )

        # Liquidity (if volumes provided)
        if volumes is not None:
            results["amihud_illiquidity"] = self._create_result(
                "amihud_illiquidity",
                self.calc_amihud_illiquidity(prices, volumes)
            )
            results["vwap_deviation"] = self._create_result(
                "vwap_deviation",
                self.calc_vwap_deviation(prices, volumes)
            )

        # Fundamentals (if provided)
        if fundamentals:
            for key in ["roe", "roa", "gross_margin", "debt_to_equity"]:
                if key in fundamentals:
                    results[key] = self._create_result(key, fundamentals[key])

        return results

    def _create_result(self, name: str, value: float) -> FactorResult:
        """Create a FactorResult with z-score and percentile placeholders."""
        defn = self.FACTOR_DEFINITIONS.get(name, {})
        return FactorResult(
            name=name,
            value=value if not np.isnan(value) else 0.0,
            z_score=0.0,  # Computed cross-sectionally
            percentile=0.5,  # Computed cross-sectionally
            category=defn.get("category", "UNKNOWN")
        )

    def compute_composite_score(
        self,
        factors: Dict[str, FactorResult],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute weighted composite factor score.

        Default weights favor momentum and quality.
        """
        if weights is None:
            weights = {
                "momentum_12_1": 0.15,
                "momentum_6_1": 0.10,
                "quality_composite": 0.20,
                "realized_vol_20": -0.15,  # Negative = lower is better
                "rsi_14": 0.05,
                "macd_signal": 0.10,
            }

        score = 0.0
        total_weight = 0.0

        for factor_name, weight in weights.items():
            if factor_name in factors:
                val = factors[factor_name].value
                if not np.isnan(val):
                    score += val * weight
                    total_weight += abs(weight)

        return score / total_weight if total_weight > 0 else 0.0


# Global singleton
_factor_zoo: Optional[FactorZoo] = None


def get_factor_zoo() -> FactorZoo:
    """Get or create global FactorZoo instance."""
    global _factor_zoo
    if _factor_zoo is None:
        _factor_zoo = FactorZoo()
    return _factor_zoo
