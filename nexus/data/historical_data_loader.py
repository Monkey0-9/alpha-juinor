import os
import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """
    Downloads and caches maximum available historical data for macro regime modeling.
    Addresses the '100-year' requirement by downloading legacy indices
    (like S&P 500 ^GSPC which goes back to 1927).
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load_macro_data(self, symbol: str = "^GSPC") -> pd.DataFrame:
        """
        Loads macro data (S&P 500) as far back as possible (1927+).
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"{symbol.replace('^', '')}_macro.csv")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached macro data for {symbol}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            return df

        logger.info(f"Downloading deep historical macro data for {symbol}...")
        try:
            # yfinance max period pulls all available data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max")

            if df.empty:
                logger.error(f"No data found for {symbol}")
                return pd.DataFrame()

            # Lowercase columns to match system expectations
            df.columns = [c.lower() for c in df.columns]

            # Basic feature engineering for macro regimes
            df['returns'] = df['close'].pct_change()
            df['vol_30d'] = df['returns'].rolling(30).std() * (252 ** 0.5)
            df['vol_252d'] = df['returns'].rolling(252).std() * (252 ** 0.5)
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()

            # Regime classification (1 = Bull/Low Vol, -1 = Bear/High Vol)
            df['regime'] = 1  # Default Bull
            # If price below 200 SMA and Volatility is high -> Bear
            df.loc[(df['close'] < df['sma_200']) & (
                df['vol_30d'] > df['vol_252d']), 'regime'] = -1

            df.dropna(inplace=True)
            df.to_csv(cache_path)
            logger.info(f"Saved {len(df)} days of macro data to {cache_path}")

            return df
        except Exception as e:
            logger.error(f"Failed to fetch macro data: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = HistoricalDataLoader()
    df = loader.load_macro_data()
    print(f"Loaded macro data from {df.index.min()} to {df.index.max()}")
    print(df.tail())
