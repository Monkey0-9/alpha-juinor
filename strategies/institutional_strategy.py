import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
from strategies.regime_engine import RegimeEngine
from alpha_families import get_alpha_families
from strategies.ml_referee import MLReferee
from strategies.filters import InstitutionalFilters
from portfolio.allocator import InstitutionalAllocator
from risk.engine import RiskManager
from strategies.nlp_engine import InstitutionalNLPEngine
from concurrent.futures import ThreadPoolExecutor

class InstitutionalStrategy:
    """
    Institutional-grade strategy combining alpha families, regime detection,
    ML referee, institutional filters, and portfolio construction.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.regime_engine = RegimeEngine()
        self.alpha_families = get_alpha_families()
        self.ml_referee = MLReferee()
        self.filters = InstitutionalFilters(self.config)
        risk_manager = RiskManager()
        self.allocator = InstitutionalAllocator(risk_manager)
        self.nlp_engine = InstitutionalNLPEngine()
        self.executor = ThreadPoolExecutor(max_workers=16)

    def generate_signals(self, market_data, context=None, macro_context=None):
        """
        Generate institutional signals using parallel symbol processing.
        Ensures consistent behavior for single or multi-asset universes.
        """
        from data.utils.schema import ensure_dataframe
        market_data = ensure_dataframe(market_data)
        
        if market_data.empty:
            logger.warning("InstitutionalStrategy: market_data is empty. Returning neutral signals.")
            return pd.DataFrame()

        # Extract tickers robustly
        if isinstance(market_data.columns, pd.MultiIndex):
            tickers = market_data.columns.get_level_values(0).unique()
        else:
            # If not MultiIndex, assume columns are tickers (Close, Open, etc might be columns if single asset)
            # Standard institutional format is MultiIndex (Ticker, Field)
            # If it's just OHLCV for one ticker, we treat the ticker as "Asset"
            tickers = ["Asset"] if "Close" in market_data.columns else market_data.columns
        
        # Parallel execution across tickers
        # Parallel execution across tickers
        def _process_ticker(symbol):
            try:
                # 1. Data Extraction & Contract Enforcement
                try:
                    if symbol == "Asset" and "Close" in market_data.columns:
                        df = market_data
                    else:
                        df = market_data[symbol] if isinstance(market_data.columns, pd.MultiIndex) else market_data[[symbol]]
                    
                    if isinstance(df, pd.Series):
                        df = df.to_frame(name="Close")

                    if df is None or df.empty:
                        logger.warning(f"Data for {symbol} is empty. Returning neutral.")
                        return symbol, 0.5

                    if 'Close' not in df.columns:
                        logger.error(f"SCHEMA BREACH: {symbol} missing 'Close' column.")
                        return symbol, 0.5
                        
                except Exception as de:
                    logger.error(f"DATA EXTRACTION FAILED for {symbol}: {de}")
                    return symbol, 0.5
                
                # 2. Context (Regime)
                try:
                    regime_context = self.regime_engine.detect_regime(df)
                except Exception as re:
                    logger.warning(f"Regime detection failed for {symbol}: {re}")
                    regime_context = {'regime_tag': 'NORMAL', 'vol_target_multiplier': 1.0}
                
                # 3. Alpha Generation
                alpha_values = []
                for alpha in self.alpha_families:
                    try:
                        res = alpha.generate_signal(df, regime_context)
                        if isinstance(res, dict) and 'signal' in res:
                            alpha_values.append(res['signal'])
                    except Exception as ae:
                        logger.debug(f"Alpha {alpha.__class__.__name__} failed for {symbol}: {ae}")
                
                # 4. News Sentiment Integration
                news_modifier = 0.0
                try:
                    news_articles = context.get('news', []) if context else []
                    if news_articles:
                        nlp_impact = self.nlp_engine.analyze_market_impact(news_articles, symbol)
                        if nlp_impact.direction == 'positive':
                            news_modifier = 0.1 * nlp_impact.magnitude
                        elif nlp_impact.direction == 'negative':
                            news_modifier = -0.1 * nlp_impact.magnitude
                except Exception:
                    pass

                final_val = np.mean(alpha_values) if alpha_values else 0.5
                final_val = np.clip(final_val + news_modifier, 0.0, 1.0)
                
                return symbol, final_val
                
            except Exception as e:
                logger.error(f"UNHANDLED STRATEGY ERROR for {symbol}: {e}")
                return symbol, 0.5

        results = list(self.executor.map(_process_ticker, tickers))
        signals = {symbol: val for symbol, val in results}

        # Apply ML referee
        refined_signals = self.ml_referee.refine_signals(signals, market_data) if hasattr(self.ml_referee, 'refine_signals') else signals

        # Apply institutional filters
        filtered_signals, _ = self.filters.apply_filters(refined_signals, market_data, {})

        # Return DataFrame
        timestamp = market_data.index[-1]
        
        # Ensure filtered_signals is not empty and contains all tickers
        if not filtered_signals:
            # Provide neutral signals if everything was filtered out
            filtered_signals = {tk: 0.5 for tk in tickers}
            
        return pd.DataFrame([filtered_signals], index=[timestamp])

    def construct_portfolio(self, signals, data, current_portfolio):
        """
        Construct portfolio using institutional allocator.
        """
        return self.allocator.allocate(signals, data, current_portfolio)

    def train_models(self, train_panel):
        """
        Train ML models if needed. For InstitutionalStrategy, this is primarily
        for the ML referee to learn from historical data.
        """
        if train_panel is not None and not train_panel.empty:
            # The ML referee can train on historical alpha signals and returns
            # For now, we'll skip training as the referee trains internally when needed
            pass
