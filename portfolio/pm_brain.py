"""
portfolio/pm_brain.py

Portfolio Manager Brain - Capital Competition Engine
Implements Mean-Variance optimization with CVaR blocking.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
from contracts.alpha_model import AlphaOutput
from alpha_families.normalization import AlphaNormalizer
from risk.quantum.state_space import RegimeStateSpace
from risk.quantum.entanglement_detector import EntanglementDetector
from contracts.allocation import RejectedAsset


logger = logging.getLogger(__name__)

# Initialize Normalizer
normalizer = AlphaNormalizer()


# Try optional optimizer import
try:
    from portfolio.optimizer import PortfolioOptimizer, Constraint
except ImportError:
    PortfolioOptimizer = None
    Constraint = None


class PMBrain:
    """
    Capital Competition Engine with risk-adjusted scoring.
    Enriched with Quantum Physics Modules:
    1. Regime State Space (μ adjustment)
    2. Entanglement Detector (Gating w_max)
    """

    def __init__(self, cvar_limit: float = -0.05, max_positions: int = 20):
        self.cvar_limit = cvar_limit
        self.max_positions = max_positions

        # Quantum Modules
        self.regime_space = RegimeStateSpace()
        self.entanglement = EntanglementDetector(threshold=0.7)

        # Initialize regime belief
        self.regime_space.update()

    def get_rejected_assets(self) -> List[RejectedAsset]:
        """Return list of rejected assets (placeholder for stateful impl)."""
        return []

    def allocate(
        self,
        alphas: Dict[str, AlphaOutput],
        current_positions: Dict[str, float],
        cov: np.array = None,
        liquidity: Dict[str, float] = None,
        prices: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, float], List[RejectedAsset]]:
        """
        Institutional Allocation using Constrained Optimizer & Quantum Modules.
        Returns: (weights, rejected_assets)
        """
        rejected_assets = []
        liquidity = liquidity or {}

        # 1. Enforce Alpha Contract
        alphas = self.enforce_alpha_contract(alphas)

        tickers = list(alphas.keys())
        if not tickers:
            return {}, []

        # 2. Quantum Updates & Regime Adjustment
        regime_compatible_mus = []

        for t in tickers:
            # Mock compatibility profile (should come from metadata)
            c_profile = np.ones(5) / 5.0
            r_i = self.regime_space.get_compatibility(c_profile)

            # Apply to mu (Regime Adjustment)
            adj_mu = alphas[t].mu * alphas[t].confidence * r_i
            regime_compatible_mus.append(adj_mu)

            if adj_mu <= 0:
                rejected_assets.append(RejectedAsset(
                    symbol=t,
                    reason="Negative Expectancy after Regime Adj",
                    mu=alphas[t].mu,
                    sigma=alphas[t].sigma,
                    score=adj_mu
                ))

        # B. Entanglement Gating (Updates w_max)
        # Dynamic cap: Ensure we can reach 100% exposure even with few assets
        base_cap = max(0.20, 1.1 / len(tickers))
        w_max_map = {t: base_cap for t in tickers}

        if prices is not None and not prices.empty:
            ent_report = self.entanglement.compute_metric(prices)
            if ent_report.threshold_breach:
                msg = f"[QUANTUM] Entanglement Breach: " \
                      f"{ent_report.global_index:.2f} > 0.7"
                logger.warning(msg)

                for t in tickers:
                    centrality = ent_report.asset_centrality.get(t, 0.0)
                    beta = 0.5
                    new_max = 0.20 * (1.0 - beta * centrality)
                    w_max_map[t] = max(0.01, new_max)

                    if new_max < 0.02:
                        reason = f"Entanglement Gating " \
                                 f"(Cap {new_max:.2%} < 2%)"
                        rejected_assets.append(RejectedAsset(
                            symbol=t,
                            reason=reason,
                            mu=alphas[t].mu,
                            sigma=alphas[t].sigma,
                            score=ent_report.global_index
                        ))

        # 4. Optimization Setup
        # Filter out rejected
        rejected_symbols = {r.symbol for r in rejected_assets}
        active_tickers = [t for t in tickers if t not in rejected_symbols]

        if not active_tickers:
            return {}, rejected_assets

        # Re-index for optimizer
        active_indices = [tickers.index(t) for t in active_tickers]
        mu_vec = np.array([regime_compatible_mus[i] for i in active_indices])

        # Handle COV
        if cov is None:
            sigmas = np.array([alphas[t].sigma for t in active_tickers])
            sub_cov = np.diag(sigmas ** 2)
        else:
            sub_cov = cov[np.ix_(active_indices, active_indices)]

        current_w = np.array(
            [current_positions.get(t, 0.0) for t in active_tickers]
        )

        # Generate Scenarios (Monte Carlo) for CVaR optimizer
        # N_scenarios = 1000
        # If sub_cov is valid, sample. Else diagonal.
        try:
            # Ensure PSD
            scenarios = np.random.multivariate_normal(mu_vec, sub_cov, 1000)
        except Exception:
            # Fallback to diagonal
            sigmas = np.sqrt(np.diag(sub_cov))
            scenarios = np.random.normal(
                mu_vec, sigmas, (1000, len(active_tickers))
            )

        # Bounds
        w_min = np.zeros(len(active_tickers))
        w_max = np.array([w_max_map[t] for t in active_tickers])

        # Params
        opt_params = {
            "lambda": 1.0,
            "gamma": 5.0,  # CVaR aversion
            "alpha": 0.95
        }

        # Run Optimization
        if PortfolioOptimizer:
            opt = PortfolioOptimizer()
            # Signature: mu, Sigma, scenario_returns, w_prev, w_min, w_max,
            #            sector_map, params
            res = opt.optimize(
                mu=mu_vec,
                Sigma=sub_cov,
                scenario_returns=scenarios,
                w_prev=current_w,
                w_min=w_min,
                w_max=w_max,
                sector_map=None,
                params=opt_params
            )

            if isinstance(res, dict) and "w" in res:
                # Extract weights
                w_opt = res["w"]
                final_weights = {
                    active_tickers[i]: float(w_opt[i])
                    for i in range(len(active_tickers))
                    if w_opt[i] > 0.001
                }

                # Log non-selected opportunity cost
                for t in active_tickers:
                    if t not in final_weights:
                        rejected_assets.append(RejectedAsset(
                            symbol=t,
                            reason="Optimizer Zero Weight",
                            mu=alphas[t].mu,
                            sigma=alphas[t].sigma,
                            score=0.0
                        ))
                return final_weights, rejected_assets
            else:
                msg = f"Optimizer failed ({res.status}), using fallback."
                logger.warning(msg)

        # Fallback
        res_fallback = self.allocate_capital(
            {t: alphas[t] for t in active_tickers}
        )
        return res_fallback, rejected_assets

    def allocate_capital(
        self,
        alphas: Dict[str, AlphaOutput],
        total_capital: float = 1.0
    ) -> Dict[str, float]:
        """Heuristic Fallback Allocation."""
        # Simple risk-parity-like or mu-weighted
        valid = []
        for sym, alpha in alphas.items():
            if alpha.mu > 0 and alpha.cvar_95 > self.cvar_limit:
                score = alpha.mu / alpha.sigma  # Sharpe proxy
                valid.append((sym, score))

        if not valid:
            return {}

        total_score = sum(s for _, s in valid)
        if total_score <= 0:
            return {}

        return {
            sym: (s / total_score) * total_capital
            for sym, s in valid
        }

    def enforce_alpha_contract(
        self,
        alphas: Dict[str, AlphaOutput]
    ) -> Dict[str, AlphaOutput]:
        """Enforce strict contract."""
        clean = {}
        for sym, alpha in alphas.items():
            # Already validated by Pydantic on creation?
            # We can re-validate or trust.
            # But let's check basic sanity if something slipped or if we want
            # to filter specific anomalies.
            if alpha.confidence < 0.2:
                msg = f"Rejecting {sym} due to low confidence " \
                      f"{alpha.confidence}"
                logger.debug(msg)
                continue

            clean[sym] = alpha
        return clean
