import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple

class VolatilitySurfaceEngine:
    """Real-time vol surface from market data"""
    
    def __init__(self):
        # SVI Parameters: (a, b, rho, m, sigma)
        self.params = {} # ticker -> params
        
    def svi_function(self, k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        """SVI (Stochastic Volatility Inspired) model"""
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
    def build_surface(self, ticker: str, strikes: List[float], vols: List[float], forward_price: float) -> Dict:
        """
        Calibrate SVI surface for a single maturity.
        k: log-moneyness log(K/F)
        """
        k = np.log(np.array(strikes) / forward_price)
        w = np.array(vols)**2 # total variance or just variance? SVI usually fits variance
        
        def objective(p):
            a, b, rho, m, sig = p
            # Constraints
            if b < 0 or abs(rho) >= 1 or sig < 0:
                return 1e10
            preds = self.svi_function(k, a, b, rho, m, sig)
            return np.sum((preds - w)**2)
            
        # Initial guess
        x0 = [0.1, 0.1, 0, 0, 0.1]
        res = minimize(objective, x0, method="Nelder-Mead")
        
        if res.success:
            self.params[ticker] = res.x
            return {
                "ticker": ticker,
                "params": res.x.tolist(),
                "status": "success"
            }
        else:
            return {"status": "failed", "error": res.message}

    def get_vol(self, ticker: str, strike: float, forward_price: float) -> float:
        """Interpolate vol using calibrated SVI"""
        if ticker not in self.params:
            return 0.0
        
        k = np.log(strike / forward_price)
        a, b, rho, m, sigma = self.params[ticker]
        var = self.svi_function(k, a, b, rho, m, sigma)
        return np.sqrt(max(0, var))
