"""
Free Market Data Sources - Alpha Vantage, Yahoo Finance, and Public APIs
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

class AlphaVantageData:
    """Free Alpha Vantage API for market data (500 calls/day)"""
    
    def __init__(self, api_key: str = "YOUR_ALPHA_VANTAGE_KEY"):
        """Initialize Alpha Vantage client"""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_real_time_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote from Alpha Vantage (free tier)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Real-time quote data
        """
        try:
            if self.api_key == "YOUR_ALPHA_VANTAGE_KEY":
                # Mock data for demo
                return {
                    "symbol": symbol,
                    "price": np.random.uniform(100, 200),
                    "change": np.random.uniform(-5, 5),
                    "change_percent": np.random.uniform(-3, 3),
                    "volume": np.random.randint(1000000, 10000000),
                    "source": "Alpha_Vantage_Mock"
                }
            
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    "symbol": quote.get("01. symbol"),
                    "price": float(quote.get("05. price", 0)),
                    "change": float(quote.get("09. change", 0)),
                    "change_percent": float(quote.get("10. change percent", "0").replace("%", "")),
                    "volume": int(quote.get("06. volume", 0)),
                    "source": "Alpha_Vantage"
                }
            else:
                return {"error": "No quote data available"}
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage quote: {e}")
            return {"error": str(e)}
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Dict:
        """
        Get historical data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            days: Number of days of data
            
        Returns:
            Historical price data
        """
        try:
            if self.api_key == "YOUR_ALPHA_VANTAGE_KEY":
                # Generate mock historical data
                dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
                prices = []
                
                base_price = np.random.uniform(100, 200)
                for i in range(days):
                    change = np.random.normal(0, 0.02)  # 2% daily volatility
                    base_price *= (1 + change)
                    prices.append(base_price)
                
                return {
                    "symbol": symbol,
                    "dates": [d.strftime('%Y-%m-%d') for d in dates],
                    "prices": prices,
                    "source": "Alpha_Vantage_Mock"
                }
            
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "compact"
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                dates = []
                prices = []
                
                for date_str, values in list(time_series.items())[:days]:
                    dates.append(date_str)
                    prices.append(float(values["4. close"]))
                
                return {
                    "symbol": symbol,
                    "dates": dates,
                    "prices": prices,
                    "source": "Alpha_Vantage"
                }
            else:
                return {"error": "No historical data available"}
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {"error": str(e)}

class YahooFinanceData:
    """Free Yahoo Finance data using yfinance library"""
    
    def __init__(self):
        """Initialize Yahoo Finance client"""
        pass
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """
        Get real-time data from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Real-time market data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            history = ticker.history(period="1d")
            if not history.empty:
                current_price = history['Close'].iloc[-1]
                previous_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
            else:
                current_price = info.get('currentPrice', 0)
                previous_close = info.get('previousClose', current_price)
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
            
            return {
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "change_percent": change_percent,
                "volume": info.get('volume', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "source": "Yahoo_Finance"
            }
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return {"error": str(e)}
    
    def get_options_chain(self, symbol: str) -> Dict:
        """
        Get options chain from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Options chain data
        """
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.options
            
            if not options:
                return {"error": "No options available"}
            
            # Get nearest expiry
            expiry = options[0]
            opt = ticker.option_chain(expiry)
            
            calls = opt.calls
            puts = opt.puts
            
            return {
                "symbol": symbol,
                "expiry": expiry,
                "calls": {
                    "count": len(calls),
                    "sample": calls.head(5).to_dict('records') if len(calls) > 0 else []
                },
                "puts": {
                    "count": len(puts),
                    "sample": puts.head(5).to_dict('records') if len(puts) > 0 else []
                },
                "source": "Yahoo_Finance"
            }
            
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return {"error": str(e)}

class FreeMarketDataAggregator:
    """Aggregator for free market data sources"""
    
    def __init__(self, alpha_vantage_key: str = None):
        """Initialize aggregator"""
        self.alpha_vantage = AlphaVantageData(alpha_vantage_key or "YOUR_ALPHA_VANTAGE_KEY")
        self.yahoo_finance = YahooFinanceData()
        
    def get_comprehensive_quote(self, symbol: str) -> Dict:
        """
        Get comprehensive quote from multiple free sources
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Comprehensive market data
        """
        try:
            # Get data from multiple sources
            alpha_data = self.alpha_vantage.get_real_time_quote(symbol)
            yahoo_data = self.yahoo_finance.get_real_time_data(symbol)
            
            # Combine data
            combined_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "sources": []
            }
            
            if "error" not in alpha_data:
                combined_data["sources"].append(alpha_data["source"])
                combined_data.update({
                    "alpha_price": alpha_data.get("price"),
                    "alpha_volume": alpha_data.get("volume")
                })
            
            if "error" not in yahoo_data:
                combined_data["sources"].append(yahoo_data["source"])
                combined_data.update({
                    "yahoo_price": yahoo_data.get("price"),
                    "yahoo_volume": yahoo_data.get("volume"),
                    "market_cap": yahoo_data.get("market_cap"),
                    "pe_ratio": yahoo_data.get("pe_ratio")
                })
            
            # Use Yahoo Finance as primary if available
            if "error" not in yahoo_data:
                combined_data["price"] = yahoo_data.get("price")
                combined_data["change"] = yahoo_data.get("change")
                combined_data["change_percent"] = yahoo_data.get("change_percent")
                combined_data["volume"] = yahoo_data.get("volume")
            elif "error" not in alpha_data:
                combined_data["price"] = alpha_data.get("price")
                combined_data["change"] = alpha_data.get("change")
                combined_data["change_percent"] = alpha_data.get("change_percent")
                combined_data["volume"] = alpha_data.get("volume")
            else:
                return {"error": "No data available from any source"}
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error aggregating market data: {e}")
            return {"error": str(e)}
    
    def get_historical_analysis(self, symbol: str, days: int = 30) -> Dict:
        """
        Get historical price analysis
        
        Args:
            symbol: Stock symbol
            days: Number of days to analyze
            
        Returns:
            Historical analysis
        """
        try:
            # Get historical data
            alpha_data = self.alpha_vantage.get_historical_data(symbol, days)
            
            if "error" not in alpha_data:
                prices = alpha_data["prices"]
                dates = alpha_data["dates"]
                
                # Calculate statistics
                returns = np.diff(prices) / prices[:-1]
                
                analysis = {
                    "symbol": symbol,
                    "period": f"{days} days",
                    "start_price": prices[0],
                    "end_price": prices[-1],
                    "total_return": (prices[-1] - prices[0]) / prices[0],
                    "volatility": np.std(returns),
                    "max_price": max(prices),
                    "min_price": min(prices),
                    "avg_volume": np.mean(alpha_data.get("volumes", [0])),
                    "source": alpha_data["source"]
                }
                
                return analysis
            else:
                return {"error": "No historical data available"}
                
        except Exception as e:
            logger.error(f"Error analyzing historical data: {e}")
            return {"error": str(e)}

def get_free_market_data(symbol: str, data_type: str = "quote") -> Dict:
    """
    Get free market data for a symbol
    
    Args:
        symbol: Stock symbol
        data_type: Type of data (quote, historical, options)
        
    Returns:
        Market data
    """
    try:
        aggregator = FreeMarketDataAggregator()
        
        if data_type == "quote":
            return aggregator.get_comprehensive_quote(symbol)
        elif data_type == "historical":
            return aggregator.get_historical_analysis(symbol)
        elif data_type == "options":
            return aggregator.yahoo_finance.get_options_chain(symbol)
        else:
            return {"error": "Invalid data type"}
            
    except Exception as e:
        logger.error(f"Error getting free market data: {e}")
        return {"error": str(e)}
