"""
Opportunity Cost Module for PM Brain.
Computes marginal contribution and compares against portfolio alternatives.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class OpportunityCostEngine:
    """
    Computes opportunity cost for portfolio decisions.
    Answers: "Is this position better than what we could do with the capital elsewhere?"
    """

    def __init__(self, replacement_threshold: float = 0.20):
        """
        Args:
            replacement_threshold: Minimum improvement required to replace existing position (default 20%)
        """
        self.replacement_threshold = replacement_threshold

    def compute_marginal_contribution(
        self,
        symbol: str,
        mu: float,
        sigma: float,
        confidence: float,
        current_portfolio: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute marginal Sharpe contribution of adding this position.

        Args:
            symbol: Symbol to evaluate
            mu: Expected return
            sigma: Expected volatility
            confidence: Forecast confidence
            current_portfolio: Current portfolio state (optional)

        Returns:
            Marginal Sharpe ratio
        """
        # Risk-adjusted score (Sharpe-like)
        if sigma <= 0:
            return 0.0

        # Confidence-adjusted Sharpe
        sharpe = (mu / sigma) * confidence

        # If we have portfolio state, compute true marginal contribution
        # accounting for correlations
        if current_portfolio and len(current_portfolio) > 0:
            try:
                # Extract portfolio positions and their characteristics
                portfolio_mus = []
                portfolio_sigmas = []
                portfolio_weights = []

                for sym, pos_data in current_portfolio.items():
                    if isinstance(pos_data, dict):
                        portfolio_mus.append(pos_data.get('mu', 0.0))
                        portfolio_sigmas.append(pos_data.get('sigma', 0.01))
                        portfolio_weights.append(pos_data.get('weight', 0.0))

                if portfolio_weights and sum(abs(w) for w in portfolio_weights) > 0:
                    # Calculate portfolio Sharpe
                    portfolio_mu = sum(w * m for w, m in zip(portfolio_weights, portfolio_mus))
                    # Simplified variance (assuming low correlation)
                    portfolio_var = sum((w * s) ** 2 for w, s in zip(portfolio_weights, portfolio_sigmas))
                    portfolio_sigma = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.01

                    portfolio_sharpe = portfolio_mu / portfolio_sigma if portfolio_sigma > 0 else 0.0

                    # Marginal contribution: how much does adding this position improve Sharpe?
                    # Simplified: compare standalone Sharpe to portfolio Sharpe
                    marginal_improvement = sharpe - portfolio_sharpe

                    # Weight the marginal contribution by confidence
                    return marginal_improvement * confidence

            except Exception as e:
                logger.warning(f"Could not compute marginal contribution for {symbol}: {e}")
                # Fall back to standalone Sharpe
                pass

        # Default: use standalone Sharpe if no portfolio state
        return sharpe

    def get_next_best_alternative(
        self,
        universe_scores: Dict[str, float],
        current_holdings: Optional[List[str]] = None,
        exclude_symbols: Optional[List[str]] = None
    ) -> Tuple[Optional[str], float]:
        """
        Find the next-best alternative position from the universe.

        Args:
            universe_scores: Dict of {symbol: sharpe_score}
            current_holdings: Symbols currently held (optional)
            exclude_symbols: Symbols to exclude from consideration

        Returns:
            (best_symbol, best_score) or (None, 0.0) if no alternatives
        """
        if not universe_scores:
            return None, 0.0

        exclude = set(exclude_symbols or [])

        # Filter out excluded symbols
        candidates = {sym: score for sym, score in universe_scores.items()
                     if sym not in exclude}

        if not candidates:
            return None, 0.0

        # Find best alternative
        best_symbol = max(candidates, key=candidates.get)
        best_score = candidates[best_symbol]

        return best_symbol, best_score

    def should_replace(
        self,
        current_symbol: str,
        current_score: float,
        alternative_symbol: Optional[str],
        alternative_score: float
    ) -> Tuple[bool, str]:
        """
        Decide if we should replace current position with alternative.

        Args:
            current_symbol: Current symbol being evaluated
            current_score: Sharpe score of current symbol
            alternative_symbol: Best alternative symbol
            alternative_score: Sharpe score of alternative

        Returns:
            (should_replace, reason)
        """
        if alternative_symbol is None:
            return False, "NO_ALTERNATIVE"

        # Calculate improvement
        if current_score <= 0:
            improvement = float('inf') if alternative_score > 0 else 0.0
        else:
            improvement = (alternative_score - current_score) / abs(current_score)

        if improvement > self.replacement_threshold:
            reason = f"OPPORTUNITY_COST_FAIL: Alternative '{alternative_symbol}' is {improvement:.1%} better"
            return True, reason

        return False, "OPPORTUNITY_COST_OK"

    def evaluate_position(
        self,
        symbol: str,
        mu: float,
        sigma: float,
        confidence: float,
        universe_scores: Optional[Dict[str, float]] = None,
        current_portfolio: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full opportunity cost evaluation for a position.

        Returns:
            Dict with {
                'marginal_contribution': float,
                'best_alternative': str,
                'alternative_score': float,
                'should_hold': bool,
                'reason': str
            }
        """
        # Compute marginal contribution
        marginal_sharpe = self.compute_marginal_contribution(
            symbol, mu, sigma, confidence, current_portfolio
        )

        # Find best alternative
        if universe_scores:
            best_alt, alt_score = self.get_next_best_alternative(
                universe_scores,
                exclude_symbols=[symbol]
            )
        else:
            best_alt, alt_score = None, 0.0

        # Decide if we should hold this position
        should_replace, reason = self.should_replace(
            symbol, marginal_sharpe, best_alt, alt_score
        )

        should_hold = not should_replace

        return {
            'marginal_contribution': marginal_sharpe,
            'best_alternative': best_alt,
            'alternative_score': alt_score,
            'should_hold': should_hold,
            'reason': reason,
            'improvement_required': self.replacement_threshold
        }
