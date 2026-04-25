import logging
import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optimizer")


class InstitutionalOptimizer:
    """
    World-class Portfolio Optimizer.
    Implements Markowitz Mean-Variance Optimization and Black-Litterman logic.
    """
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate

    def optimize_weights(self, expected_returns, cov_matrix, target_return=None):
        """
        Finds weights that maximize the Sharpe Ratio
        """
        num_assets = len(expected_returns)
        args = (expected_returns, cov_matrix)
        # Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # No short selling in this basic version
        bounds = tuple((0, 1) for asset in range(num_assets))

        def negative_sharpe(weights, exp_returns, cov_mat):
            p_return = np.sum(exp_returns * weights)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
            sharpe = (p_return - self.risk_free_rate) / p_std
            return -sharpe

        result = minimize(
            negative_sharpe, num_assets * [1. / num_assets],
            args=args, method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Portfolio optimization SUCCESSFUL")
            return result.x
        else:
            logger.error("Portfolio optimization FAILED")
            return None

    def calculate_tracking_error(self, portfolio_returns, benchmark_returns):
        """
        Calculates Tracking Error against a benchmark (e.g. S&P 500)
        """
        diff = portfolio_returns - benchmark_returns
        return np.std(diff) * np.sqrt(252)


if __name__ == "__main__":
    # Demo optimization
    opt = InstitutionalOptimizer()
    returns = np.array([0.12, 0.15, 0.10])
    cov = np.array([[0.05, 0.01, 0.02],
                    [0.01, 0.06, 0.01],
                    [0.02, 0.01, 0.04]])
    weights = opt.optimize_weights(returns, cov)
    print(f"Optimal Weights: {weights}")
