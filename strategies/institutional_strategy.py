# strategies/institutional_strategy.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
import time
from .base import BaseStrategy
from .alpha import CompositeAlpha, TrendAlpha, MeanReversionAlpha, RSIAlpha, MACDAlpha, BollingerBandAlpha
from .ml_models.ml_alpha import MLAlpha
from data.processors.features import FeatureEngineer

# New Institutional Modules
from regime.markov import RegimeModel
from market_structure.wyckoff import structure_filter
from market_structure.auction import AuctionConfidence
from market_structure.market_profile import compute_market_profile
from timing.gann_cycles import GannTimeFilter
from micro.vpin import compute_vpin

logger = logging.getLogger(__name__)

class InstitutionalStrategy(BaseStrategy):
    """
    Maximized High-Frequency Alpha Strategy.
    Enriched with Institutional Regimes, Structures, and Filters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tickers = config.get('tickers', [])
        
        # Feature Flags
        features = config.get('features', {})
        self.use_regime = features.get('use_regime_detection', True)
        self.use_wyckoff = features.get('use_wyckoff_filter', True)
        self.use_auction = features.get('use_auction_market_confidence', True)
        self.use_market_profile = features.get('use_market_profile_value_area', True)
        self.use_gann = features.get('use_gann_time_filter', True)
        self.use_vpin = features.get('use_vpin_filter', True)

        # Ensemble of 5 technical alphas
        self.technical_alpha = CompositeAlpha(alphas=[
            TrendAlpha(short=5, long=20),
            MeanReversionAlpha(),
            RSIAlpha(),
            MACDAlpha(),
            BollingerBandAlpha()
        ], window=30)
        # Per-Ticker ML Models (Institutional Hardening)
        fe = FeatureEngineer()
        self.ml_models: Dict[str, MLAlpha] = {
            tk: MLAlpha(feature_engineer=fe) for tk in self.tickers
        }
        
        # Initialize Filters
        self.regime_engine = RegimeModel()
        self.auction_engine = AuctionConfidence()
        self.gann_filter = GannTimeFilter()
        
        # Optimization: Pre-allocate Fallback Logging Set & Throttle
        self._logged_fallback = set()
        self._log_throttle = 0

    def generate_signals(self, market_data: pd.DataFrame, macro_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        t_start = time.perf_counter()
        signals = {}
        ts = market_data.index[-1]
        
        # 0. Global Timing Filter (Gann) - O(1) Check
        if self.use_gann and not self.gann_filter.can_trade(ts):
            # Fast Exit
            for tk in self.tickers: signals[tk] = 0.5
            return pd.DataFrame([signals], index=[ts])

        # 0b. Macro Risk Overlay (Multi-Source Logic)
        # Uses FRED VIX data if provided to gate trading
        risk_off_mode = False
        if macro_context:
            vix = macro_context.get("VIX", 20.0)
            if vix > 32.0: # Institutional Panic Threshold
                risk_off_mode = True
                logger.info(f"MACRO ALERT: VIX {vix:.1f} > 32. Trading Halted (Risk Off).")
                for tk in self.tickers: signals[tk] = 0.5
                return pd.DataFrame([signals], index=[ts])

        # LATENCY OPTIMIZATION:
        # Enforce "Time Budget" of 50ms (soft target)
        
        # Rate Limiting for Logging (Prevent I/O storms during spikes)
        if not hasattr(self, '_log_throttle'): self._log_throttle = 0
        
        # GLOBAL PRE-SLICING (The "Still More Better" Optimization)
        # Instead of slicing deeper in the loop, we slice the whole panel once.
        # This is strictly O(1) for the loop.
        if len(market_data) > 60:
            market_window = market_data.iloc[-60:]
        else:
            market_window = market_data
            
        for tk in self.tickers:
            try:
                # OPTIMIZATION: Check column existence cheaply
                if tk not in market_window.columns.levels[0]: continue
                
                # Access pre-sliced window. 
                # Still need dropna() for specific ticker in case of partial data
                df_slice = market_window[tk].dropna()
                
                if df_slice.empty or len(df_slice) < 20:
                    signals[tk] = 0.5
                    continue
                
                # 1. Alpha (Hybrid)
                # Technical (Vectorized inside needs more data? TrendAlpha long=20. 60 is fine.)
                tech_score = float(self.technical_alpha.compute(df_slice).iloc[-1])
                
                # ML (Inference)
                # CRITICAL: Do not alter ML models as per user instruction.
                ml_model = self.ml_models.get(tk)
                conviction = 0.5
                if ml_model and ml_model.is_trained and ml_model.ml_ready:
                    ml_score = ml_model.predict_conviction(df_slice)
                    conviction = 0.5 * tech_score + 0.5 * ml_score
                else:
                    conviction = 0.5 + (tech_score - 0.5) * 0.7
                    
                # 2. Institutional Filters (Conditional Execution - Fail Fast)
                if abs(conviction - 0.5) < 0.01:
                    signals[tk] = 0.5
                    continue
                
                # Regime
                if self.use_regime:
                    label = self.regime_engine.infer(df_slice)
                    if label == "high_vol": conviction = 0.5 + (conviction - 0.5) * 0.5
                    elif label == "neutral": conviction = 0.5 + (conviction - 0.5) * 0.8
                    
                # Wyckoff
                if self.use_wyckoff:
                    # FIX: Cache result, don't call multiple times!
                    wyckoff_s = structure_filter(df_slice)
                    if conviction > 0.5 and not wyckoff_s['allow_long']: conviction = 0.5
                    if conviction < 0.5 and not wyckoff_s['allow_short']: conviction = 0.5

                # Auction
                if self.use_auction:
                    auc = self.auction_engine.compute_confidence(
                        df_slice["Close"], df_slice["Volume"], df_slice["High"], df_slice["Low"]
                    )
                    conviction = 0.5 + (conviction - 0.5) * auc

                # Market Profile
                if self.use_market_profile:
                    mp = compute_market_profile(df_slice)
                    if mp['inside_value_area']: conviction = 0.5 + (conviction - 0.5) * 0.5
                    
                # VPIN (Skipped for now or stub)
                
                # --- PHASE 4: INSTITUTIONAL HARDENING (EV & CONSENSUS) ---
                
                # 3. Expected Value (EV) Gate
                # EV = P(Win) * Gain - P(Loss) * Loss - Cost
                # Assumptions: Reward:Risk = 2:1, Cost = 5bps
                daily_vol = df_slice['Close'].pct_change().std()
                if np.isnan(daily_vol): daily_vol = 0.01
                
                # Estimated Probabilities derived from Conviction
                # Conviction 0.6 -> P(Win) ~ 0.55? Let's treat Conviction as raw Probability for simplicity
                if conviction > 0.5:
                    p_win = conviction
                    p_loss = 1.0 - p_win
                    est_reward = 2.0 * daily_vol
                    est_risk = 1.0 * daily_vol
                    transaction_cost = 0.0005 # 5bps
                    
                    ev = (p_win * est_reward) - (p_loss * est_risk) - transaction_cost
                    
                    if ev <= 0:
                        # logger.debug(f"EV REJECT {tk}: EV={ev:.5f} <= 0")
                        conviction = 0.5
                
                # 4. Multi-Horizon Consensus (Weekly/Daily alignment)
                # Validates if the 'Micro' signal aligns with 'Macro' trend of the window
                if conviction != 0.5:
                    # Daily Slope
                    y_daily = df_slice['Close'].values
                    slope_daily = np.polyfit(np.arange(len(y_daily)), y_daily, 1)[0]
                    
                    # Weekly Estimate (Simulated by resampling last 20 days -> ~4 weeks)
                    # If we have 60 days, we can resample.
                    if len(df_slice) >= 20:
                        weekly_closes = df_slice['Close'].iloc[::5] # Approx weekly
                        if len(weekly_closes) > 2:
                            slope_weekly = np.polyfit(np.arange(len(weekly_closes)), weekly_closes.values, 1)[0]
                            
                            # CONSENSUS CHECK:
                            # If Daily UP but Weekly DOWN -> Chop Risk -> Reduce Signal
                            if (slope_daily > 0 and slope_weekly < 0) or (slope_daily < 0 and slope_weekly > 0):
                                logger.info(f"Timeframe Conflict {tk}: Daily/Weekly divergence. Halving conviction.")
                                conviction = 0.5 + (conviction - 0.5) * 0.5

                # 6. Loss Shape Control (ATR Based Volatility Stop)
                # "Winning systems lose small"
                # If volatility is high, we require a WIDER stop (lower leverage), effectively reducing convexity of loss
                atr_period = 14
                if len(df_slice) > atr_period:
                    high_low = df_slice['High'] - df_slice['Low']
                    high_close = np.abs(df_slice['High'] - df_slice['Close'].shift())
                    low_close = np.abs(df_slice['Low'] - df_slice['Close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = np.max(ranges, axis=1)
                    atr = true_range.rolling(atr_period).mean().iloc[-1]
                    
                    if atr > 0:
                        # Dynamic Stop Distance
                        # Calm (Low Vol) -> Tight Stop (1x ATR)
                        # Chaos (High Vol) -> Loose Stop (2x ATR) -> Requires smaller size
                        current_price = df_slice['Close'].iloc[-1]
                        vol_regime_scalar = (atr / current_price) / 0.01 # Normalized to 1% daily vol
                        
                        # If highly volatile, penalize conviction to enforce "Shape Control"
                        # We don't want to take full size positions in 4-ATR markets
                        if vol_regime_scalar > 2.0:
                             conviction = 0.5 + (conviction - 0.5) * (1.0 / vol_regime_scalar)
                
                # 8. Liquidity Impact (Market Impact Model)
                # Impact ~ Eta * sqrt(Size / ADV)
                # We enforce: Size < 1% ADV to keep impact negligible
                recent_vol_avg = df_slice['Volume'].iloc[-5:].mean()
                if recent_vol_avg > 0:
                     # Assume we want to trade roughly $10k (mini fund size)
                     # In real engine, 'Allocator' knows size, but Strategy must verify tradeability
                     # $10k / Price = Shares
                     est_shares = 10000.0 / df_slice['Close'].iloc[-1]
                     participation_rate = est_shares / recent_vol_avg
                     
                     if participation_rate > 0.01: # > 1% of volume
                          # Too big for liquidity -> Impact Penalty
                          penalty = 1.0 - (participation_rate * 10) # decay fast
                          conviction = 0.5 + (conviction - 0.5) * max(0.1, penalty)
                
                signals[tk] = conviction
                
            except Exception as e:
                # ERROR SAFETY: Catch all, log only once per 100 errors to avoid flooding
                # signals[tk] = 0.5 is the safe default
                if self._log_throttle % 100 == 0:
                     logger.error(f"Signal failed {tk}: {e}")
                self._log_throttle += 1
                signals[tk] = 0.5

        t_end = time.perf_counter()
        duration_ms = (t_end - t_start) * 1000.0
        
        # Performance Monitoring (Throttled)
        if duration_ms > 50:
             self._log_throttle += 1
             if self._log_throttle % 50 == 0: # Log only every 50th slow occurrence
                 logger.warning(f"SLOW SIGNAL GENERATION: {duration_ms:.2f}ms")
             
        return pd.DataFrame([signals], index=[ts])

    def calculate_risk(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Pass-through for risk calculation.
        Actual risk logic is handled by RiskManager in the Engine, but Strategy contract requires this.
        """
        return signals

    def train_models(self, data: pd.DataFrame):
        """Train independent ML models."""
        for tk in self.tickers:
            if tk in data.columns.levels[0]:
                df_tk = data[tk].dropna()
                if not df_tk.empty:
                    ml_model = self.ml_models.get(tk)
                    if ml_model:
                        ml_model.train(df_tk)
