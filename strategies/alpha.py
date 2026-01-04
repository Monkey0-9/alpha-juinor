# strategies/alpha.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np


def _minmax_scale(s: pd.Series) -> pd.Series:
    """Scale series to 0..1 using min/max (safe)."""
    s = pd.to_numeric(s, errors='coerce').astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def safe_pct_change(series: pd.Series) -> pd.Series:
    """Strict pct_change with no implicit forward-fill and inf/nan removal."""
    r = series.pct_change(fill_method=None)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return pd.to_numeric(r, errors='coerce').astype(float)


def safe_clip(raw: pd.Series, prices_index: pd.Index) -> pd.Series:
    """Safe clipping with empty-series guard returning aligned zeros."""
    raw = pd.to_numeric(raw, errors='coerce').astype(float)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return pd.Series(0.0, index=prices_index)
    return raw.clip(lower=0.0, upper=1.0)


def _ensure_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """
    FAIL FAST: Extracts 'Close' column if DataFrame provided.
    Ensures alpha models operate on deterministic scalar prices.
    """
    if isinstance(data, pd.DataFrame):
        if "Close" not in data.columns:
            raise ValueError("Institutional Abort: Market data missing required 'Close' column.")
        return data["Close"]
    return data


class Alpha(ABC):
    """Abstract Alpha. Implement compute(prices) -> conviction series (0..1)."""

    @abstractmethod
    def compute(self, prices: pd.Series) -> pd.Series:
        pass


class TrendAlpha(Alpha):
    """
    Trend alpha: uses spread of short / long EMA and recent slope.
    Produces a smooth 0..1 conviction series (higher => stronger trend).
    """

    def __init__(self, short: int = 50, long: int = 200):
        self.short = short
        self.long = long

    def compute(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        prices = _ensure_series(prices)
        if len(prices) < self.long:
            return pd.Series(0.0, index=prices.index)

        ema_s = prices.ewm(span=self.short, adjust=False).mean()
        ema_l = prices.ewm(span=self.long, adjust=False).mean()

        # raw score = relative distance between short and long EMAs
        score = (ema_s - ema_l) / ema_l
        # also increase score when slope of short EMA is positive
        slope = ema_s.diff(5) / ema_s.shift(5)
        raw = score + 0.5 * slope

        return safe_clip(raw, prices.index)


class MeanReversionAlpha(Alpha):
    """
    Mean reversion alpha:
    When price is stretched vs short MA (zscore), expect reversion.
    The signal is larger when price is far from short MA in standardized units.
    """

    def __init__(self, short: int = 5, lookback_std: int = 21):
        self.short = short
        self.lookback_std = lookback_std

    def compute(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        prices = _ensure_series(prices)
        if len(prices) < max(self.short, self.lookback_std):
            return pd.Series(0.0, index=prices.index)

        sma_short = prices.rolling(self.short).mean()
        dev = prices - sma_short
        vol = safe_pct_change(prices).rolling(self.lookback_std).std().replace(0, np.nan)
        z = (dev / (prices * vol)).fillna(0)  # normalized stretch

        # mean reversion expects negative signal when price is far above SMA (we want to short),
        # but we want a long-only conviction scale. We'll use symmetric reversion magnitude.
        # Ideally, high z (overbought) -> sell signal (low conviction for long only layout)
        # But if we treat it as "reversion opportunity", maybe higher is better?
        # Standard approach:
        # If z < -2 (oversold) -> Buy (High Conviction)
        # If z > 2 (overbought) -> Sell (Low Conviction)
        
        # Invert z so that low z (oversold) becomes high value
        raw = -z 
        
        return safe_clip(raw, prices.index)


class RSIAlpha(Alpha):
    """
    RSI Alpha:
    RSI < 30 -> High Conviction (Oversold, potential bounce)
    RSI > 70 -> Low Conviction (Overbought)
    """
    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        prices = _ensure_series(prices)
        if len(prices) < self.period + 1:
            return pd.Series(0.0, index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        # We want Alpha 1.0 when RSI is low (oversold)
        # Alpha 0.0 when RSI is high (overbought)
        # Linear map: (100 - RSI) / 100
        raw = (100 - rsi) / 100.0
        return safe_clip(raw, prices.index)


class MACDAlpha(Alpha):
    """
    MACD Crossover Alpha:
    Signal = MACD - SignalLine
    Positive -> Bullish Trend -> High Conviction
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        prices = _ensure_series(prices)
        if len(prices) < self.slow + self.signal:
            return pd.Series(0.0, index=prices.index)

        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=self.signal, adjust=False).mean()

        hist = macd - sig
        # Higher histogram -> stronger bullish momentum
        return safe_clip(hist, prices.index)


class BollingerBandAlpha(Alpha):
    """
    Bollinger Band Squeeze/Breakout Alpha.
    Here we focus on Mean Reversion: Price touching lower band -> Buy.
    """
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def compute(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        prices = _ensure_series(prices)
        if len(prices) < self.window:
            return pd.Series(0.0, index=prices.index)

        sma = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        
        upper = sma + (std * self.num_std)
        lower = sma - (std * self.num_std)

        # %B indicator = (Price - Lower) / (Upper - Lower)
        # %B < 0 -> Price below lower band (Oversold) -> High Conviction
        # %B > 1 -> Price above upper band (Overbought) -> Low Conviction
        
        pct_b = (prices - lower) / (upper - lower).replace(0, np.nan)
        
        # Invert: 1 - %B
        # If pct_b is 0 (at lower band), score is 1.0
        # If pct_b is 1 (at upper band), score is 0.0
        score = 1.0 - pct_b
        return safe_clip(score, prices.index)


class CompositeAlpha(Alpha):
    """
    Composite Alpha with Dynamic Probabilistic Weighting (Rolling IC/Sharpe).
    """

    def __init__(self, alphas: List[Alpha], window: int = 60):
        self.alphas = alphas
        self.window = window
        self.weights = np.array([1.0 / len(alphas)] * len(alphas))
        
        # History tracking
        self.history_signals: List[np.array] = [] # List of [alpha_1_sig, alpha_2_sig, ...]
        self.history_prices: List[float] = []
        
    def _update_weights(self):
        """
        Recalculate weights based on Rolling IC (Information Coefficient).
        IC = Correlation(Signal_t-1, Return_t)
        """
        if len(self.history_prices) < 30:
            return # Keep equal weights until enough history
            
        # 1. Calculate Returns (INSTITUTIONAL: Explicit fill_method)
        prices = pd.Series(self.history_prices)
        returns = (
            prices
            .pct_change(fill_method=None)
            .shift(-1)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        
        if len(returns) < 20:
            return

        # 2. Align Signals
        # self.history_signals[i] corresponds to price[i]. 
        # We want corr(signal[i], returns[i]) where returns[i] is (p[i+1]/p[i] - 1)
        # So we need signals up to len(returns)
        
        n_points = len(returns)
        sigs_mat = np.array(self.history_signals[:n_points]) # shape (n_points, n_alphas)
        
        # 3. Compute Performance Metrics (IC and Sharpe)
        # Pearson correlation
        ics = []
        sharpes = []
        target = returns.values
        
        if np.std(target) < 1e-8:
            return

        for i in range(len(self.alphas)):
            alpha_sigs = sigs_mat[:, i]
            
            # IC
            if np.std(alpha_sigs) < 1e-8:
                ics.append(0.0)
            else:
                ic = np.corrcoef(alpha_sigs, target)[0, 1]
                ics.append(0.0 if np.isnan(ic) else ic)
            
            # Sharpe (Implied PnL)
            # Implied PnL = Signal * Return
            pnl = alpha_sigs * target
            mean_pnl = np.mean(pnl)
            std_pnl = np.std(pnl)
            
            if std_pnl < 1e-9:
                sharpes.append(0.0)
            else:
                # Annualized? No, just raw ratio is fine for relative weighting
                sharpes.append(mean_pnl / std_pnl)

        # 4. Weighting Logic
        # Requirement: weight ∝ max(sharpe, 0)
        # We can mix IC and Sharpe, or just use Sharpe.
        # Let's use Sharpe as the primary requirement for "Meta-Alpha Allocation".
        
        raw_scores = np.maximum(sharpes, 0.0) # Clip negative Sharpe
        
        # If all scores zero (e.g. neg sharpe), fallback to IC
        if np.sum(raw_scores) < 1e-6:
             raw_scores = np.maximum(ics, 0.0)
        
        # If still zero, equal weight
        if np.sum(raw_scores) < 1e-6:
            self.weights = np.array([1.0 / len(self.alphas)] * len(self.alphas))
            return
            
        # Normalize
        self.weights = raw_scores / np.sum(raw_scores)

    def compute(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        # Institutional Fix: Select only Close column before scalar extraction
        ser = _ensure_series(prices)
        current_price = float(ser.iloc[-1]) if not ser.empty else 0.0
        
        # 1. Capture history for *next* weight update
        self.history_prices.append(current_price)
        
        # Limit history size
        if len(self.history_prices) > self.window + 5:
             self.history_prices.pop(0)
             if self.history_signals:
                 self.history_signals.pop(0)

        # 2. Compute current component signals
        current_sigs = []
        series_list = []
        
        for a in self.alphas:
            s = a.compute(prices).fillna(0)
            series_list.append(s)
            
            # Store scalar signal for history (for IC calc)
            # We assume the last value is the 'actionable' signal for this bar
            val = float(s.iloc[-1]) if not s.empty else 0.0
            current_sigs.append(val)
            
        self.history_signals.append(np.array(current_sigs))
        
        # 3. Update Weights (based on history accumulated so far)
        # Note: We update weights *before* applying them to current signal?
        # Actually, we can only correlate past signal with *known* return.
        # So we update weights using closed history, then apply to current.
        self._update_weights()
            
        if not series_list:
             return pd.Series(0.0, index=prices.index)

        # align indices
        df = pd.concat(series_list, axis=1).fillna(0)
        
        # 4. Apply Dynamic Weights
        w = self.weights
        w_sum = w.sum()
        if w_sum <= 0:
            return pd.Series(0.0, index=prices.index)
            
        # Apply weights
        # df columns are alphas. w is vector.
        # df is (T, N), w is (N,)
        conv = (df * w).sum(axis=1)
        return safe_clip(conv, prices.index)
