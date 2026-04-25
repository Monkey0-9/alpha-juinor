from numba import jit

class SovereignFractalEngine:
    """
    Calculates the Fractal Dimension (Hurst-like) of the market structure.
    Detects "Roughness" or "Jaggedness" in the price to predict breaks.
    """
    @staticmethod
    @jit(nopython=True)
    def _fast_fd(prices):
        n = len(prices)
        if n < 30: return 1.5
        
        diffs = np.abs(prices[1:] - prices[:-1])
        variation = np.sum(diffs)
        
        if variation == 0: return 1.0
        fd = 1.0 + (np.log(variation) / np.log(n))
        return min(2.0, max(1.0, fd))

    def calculate_fractal_dimension(self, prices):
        return self._fast_fd(prices)

    def detect_hidden_vol(self, fd):
        if fd > 1.7: return "HIGH_ROUGHNESS"
        if fd < 1.3: return "HIGH_PERSISTENCE"
        return "STABLE"
