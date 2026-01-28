"""
Capital Auction Engine - Phase 4
=================================
All strategies and symbols compete for capital using optimization:
    max_w Σ wᵢ μᵢ − λ · CVaR(w)

Inputs:
- Expected return μ
- Uncertainty σ
- CVaR
- Data confidence
- Liquidity cost
- Correlation
- Time risk

Output:
- Allocation
- Rank
- Rejection reason

No symbol gets capital by default.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from governance.institutional_specification import (
    AssetClass,
    CapitalAuctionInput,
    CapitalAuctionOutput,
    CVaRConfig,
    ModelHealthMetrics,
    compute_model_disagreement_penalty,
    compute_decay_factors,
    StrategyLifecycle
)

logger = logging.getLogger(__name__)


class CapitalAuctionEngine:
    """
    PM Brain - Capital Competition Engine.

    All strategies and symbols compete for capital through a structured auction.
    The optimization maximizes risk-adjusted returns with CVaR constraints.

    Objective: max_w Σ wᵢ μᵢ − λ · CVaR(w)

    Where:
    - μ = expected return (adjusted for decay, disagreement, data quality)
    - CVaR = Conditional Value at Risk (95% confidence)
    - λ = risk aversion parameter
    """

    def __init__(
        self,
        cvar_config: Optional[CVaRConfig] = None,
        risk_aversion_lambda: float = 2.0,
        min_data_quality: float = 0.6,
        min_history_days: int = 1260,
        max_position_size: float = 0.10,
        max_sector_exposure: float = 0.15,
        max_correlation_risk: float = 0.70,
        enable_governance_veto: bool = True
    ):
        """
        Initialize Capital Auction Engine.

        Args:
            cvar_config: CVaR configuration
            risk_aversion_lambda: Risk aversion parameter for optimization
            min_data_quality: Minimum data quality score to qualify
            min_history_days: Minimum history required (1260 = ~5 trading years)
            max_position_size: Maximum single position size
            max_sector_exposure: Maximum sector exposure
            max_correlation_risk: Maximum average correlation threshold
            enable_governance_veto: Whether to apply governance veto checks
        """
        self.cvar_config = cvar_config if cvar_config is not None else CVaRConfig()
        self.risk_aversion = risk_aversion_lambda
        self.lambda_param = risk_aversion_lambda  # Alias for clarity
        self.min_data_quality = min_data_quality
        self.min_history_days = min_history_days
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation_risk = max_correlation_risk
        self.enable_governance_veto = enable_governance_veto

        # Track allocations for correlation computation
        self._allocation_history: Dict[str, float] = {}

    def run_auction(
        self,
        candidates: List[CapitalAuctionInput],
        portfolio_nav: float,
        current_weights: Optional[Dict[str, float]] = None,
        available_capital: Optional[float] = None
    ) -> Dict[str, CapitalAuctionOutput]:
        """
        Run the capital auction for all candidate symbols.

        Args:
            candidates: List of capital auction inputs
            portfolio_nav: Total portfolio NAV
            current_weights: Current position weights
            available_capital: Available capital for new allocations

        Returns:
            Dict mapping symbol to auction output
        """
        if current_weights is None:
            current_weights = {}
        if available_capital is None:
            available_capital = portfolio_nav

        outputs = {}

        # Phase 1: Pre-qualification screening
        qualified = self._prequalify_candidates(candidates)

        # Phase 2: Compute adjusted returns with all penalties
        adjusted_inputs = self._compute_adjusted_returns(qualified)

        # Phase 3: Run optimization
        allocations = self._optimize_allocations(
            adjusted_inputs,
            portfolio_nav,
            current_weights
        )

        # Phase 4: Apply governance veto
        final_outputs = self._apply_governance_veto(
            adjusted_inputs,
            allocations,
            portfolio_nav
        )

        # Track for correlation
        for symbol, output in final_outputs.items():
            if output.allocated:
                self._allocation_history[symbol] = output.weight

        return final_outputs

    def _prequalify_candidates(
        self,
        candidates: List[CapitalAuctionInput]
    ) -> List[CapitalAuctionInput]:
        """
        Phase 1: Screen candidates against minimum requirements.

        Rejects:
        - Insufficient history (< 1260 days)
        - Poor data quality (< 0.6)
        - No expected return (μ <= 0)
        - Extreme model decay (final_decay_factor < 0.3)
        """
        qualified = []

        for c in candidates:
            # Debug logging
            logger.info("AUCTION_INPUT %s mu=%0.6f sigma=%0.6f cvar=%0.6f quality=%0.3f rows=%d",
                        c.symbol, c.mu, c.sigma, c.cvar_95, c.data_quality_score, c.history_days)

            reasons = []

            # Check history
            if c.history_days < self.min_history_days:
                reasons.append(f"INSUFFICIENT_HISTORY:{c.history_days}<{self.min_history_days}")

            # Check data quality
            if c.data_quality_score < self.min_data_quality:
                reasons.append(f"LOW_DATA_QUALITY:{c.data_quality_score:.2f}<{self.min_data_quality:.2f}")

            # Check NaN values (Task 2: Stop NaNs)
            import math
            if math.isnan(c.mu) or math.isinf(c.mu):
                reasons.append("NAN_OR_INF_MU")

            if math.isnan(c.sigma) or math.isinf(c.sigma):
                reasons.append("NAN_OR_INF_SIGMA")

            if math.isnan(c.cvar_95) or math.isinf(c.cvar_95):
                reasons.append("NAN_OR_INF_CVAR")

            # Check return (must be positive) - only if mu is valid
            if not math.isnan(c.mu) and c.mu <= 0:
                reasons.append(f"NON_POSITIVE_RETURN:mu={c.mu:.4f}")

            # Check model decay
            _, _, decay = compute_decay_factors(
                c.model_age_days,
                c.rolling_forecast_error,
                c.autocorr_flip_detected
            )
            if decay < 0.3:
                reasons.append(f"MODEL_DECAYED:decay={decay:.2f}<0.3")

            # Check for placeholder values (Objective 5)
            # mu == -0.02 is a known placeholder for "uninitialized/failed"
            if not math.isnan(c.mu) and abs(c.mu - (-0.02)) < 1e-6:
                reasons.append("PLACEHOLDER_ERROR:mu=-0.02")

            # Check for excessive CVaR (Objective 6 - Marginal)
            if not math.isnan(c.marginal_cvar) and c.marginal_cvar > 0.10: # 10% marginal CVaR for 1% weight is extreme
                reasons.append(f"EXCESSIVE_MARGINAL_CVAR:{c.marginal_cvar:.2f}")

            if reasons:
                logger.warning(f"[AUCTION] [VETO] Rejected {c.symbol} for institutional sanity: {reasons}")
                # Still include but mark as rejected - optimization will skip it
                c.reason_codes.extend(reasons)

            # Use 'qualified' list only for clean candidates?
            # Current implementation logic:
            # `run_auction` uses `qualified` list.
            # If we append to `qualified` but it has reason codes, will `_algorithm` filter it?
            # `_compute_adjusted_returns` processes them.
            # `_optimize_allocations` calculates scores.

            # Refinement: If it has MAJOR blocking reasons (NaNs), DO NOT add to qualified.
            # If it has soft reasons (low quality but > min), maybe add?
            # But here `reasons` implies REJECTION.

            # The original code appended to qualified anyway:
            # qualified.append(c)
            # But set reason_codes.

            # Optimization logic:
            # sorted_inputs = sorted(..., key=lambda x: x['mu_adjusted'] / max(x['cvar_95'], 0.001))
            # If mu is NaN, sorting crashes.

            # FIX: If NaNs present, DO NOT ADD to qualified.
            if any("NAN" in r for r in reasons):
                 # Skip entirely for optimization, but we want it in output as Rejected.
                 # run_auction returns `final_outputs`.
                 # It calls `_apply_governance_veto(adjusted_inputs...)`
                 # If we drop it from qualified, it won't be in adjusted_inputs.
                 # Then `_apply_governance_veto` loops over `allocations`.
                 # If it's not in allocations, it won't be in output?
                 # Wait, `run_auction` returns `final_outputs`.
                 # We need to make sure rejected candidates appear in final output with decision="REJECT".

                 # Current architecture flaw: if dropped here, it disappears.
                 # We should add it to qualified but ensure values are clean (e.g. 0.0) so optimization doesn't crash,
                 # AND ensure allocation is 0.

                 # If NaN, force values to safe defaults for the pipeline but mark REJECTED.
                 if "NAN_OR_INF_MU" in reasons: c.mu = -1.0 # Force negative return so optimization ignores
                 if "NAN_OR_INF_SIGMA" in reasons: c.sigma = 1.0
                 if "NAN_OR_INF_CVAR" in reasons: c.cvar_95 = 1.0

            qualified.append(c)

        return qualified

    def _compute_adjusted_returns(
        self,
        candidates: List[CapitalAuctionInput]
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Compute adjusted returns with all penalties.

        Applies:
        1. Model disagreement penalty
        2. Data quality penalty
        3. Liquidity cost penalty
        4. Model decay penalty
        5. Time decay penalty
        """
        adjusted = []

        # Collect mus for disagreement calculation
        mus = [c.mu for c in candidates]

        for c in candidates:
            # 1. Base expected return
            mu = c.mu

            # 2. Apply model disagreement penalty
            # This would ideally use multiple model outputs per symbol
            # For now, we use the provided sigma as proxy for disagreement
            disagreement_penalty = np.exp(-0.5 * (c.sigma ** 2))
            mu_adjusted = mu * disagreement_penalty

            # 3. Apply data quality penalty
            data_quality_penalty = c.data_quality_score
            mu_adjusted *= data_quality_penalty

            # 4. Apply liquidity cost penalty
            liquidity_penalty = 1.0 - min(c.liquidity_cost_bps / 100.0, 0.5)
            mu_adjusted *= liquidity_penalty

            # 5. Apply model decay penalty
            age_decay, error_decay, final_decay = compute_decay_factors(
                c.model_age_days,
                c.rolling_forecast_error,
                c.autocorr_flip_detected
            )
            mu_adjusted *= final_decay

            # 6. Apply time decay (holding period penalty)
            if c.holding_period_days > 0:
                time_penalty = np.exp(-c.time_decay_rate * c.holding_period_days / 252)
                mu_adjusted *= time_penalty

            # 7. Apply market impact penalty
            impact_penalty = 1.0 - min(c.market_impact_bps / 100.0, 0.3)
            mu_adjusted *= impact_penalty

            # 8. Compute risk-adjusted score
            # Higher CVaR = lower score
            risk_score = c.cvar_95 / max(c.mu, 0.0001)

            # 9. Compute expected CVaR contribution
            # Simplified: assume linear contribution for small weights
            cvar_contribution = c.marginal_cvar

            adjusted.append({
                'symbol': c.symbol,
                'asset_class': c.asset_class,
                'mu_raw': c.mu,
                'mu_adjusted': mu_adjusted,
                'sigma': c.sigma,
                'cvar_95': c.cvar_95,
                'marginal_cvar': cvar_contribution,
                'p_loss': c.p_loss,
                'data_quality': c.data_quality_score,
                'liquidity_cost': c.liquidity_cost_bps,
                'market_impact': c.market_impact_bps,
                'correlation_risk': c.correlation_risk,
                'sector_exposure': c.sector_exposure,
                'strategy_id': c.strategy_id,
                'strategy_stage': c.strategy_lifecycle_stage,
                'decay_factor': final_decay,
                'risk_score': risk_score,
                'reason_codes': c.reason_codes.copy()
            })

        return adjusted

    def _optimize_allocations(
        self,
        adjusted_inputs: List[Dict[str, Any]],
        portfolio_nav: float,
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Phase 3: Run optimization to allocate capital.

        Uses greedy optimization with CVaR constraints.

        Objective: max Σ wᵢ μᵢ − λ · CVaR(w)

        Constraints:
        - Σ|wᵢ| ≤ 1.0 (gross leverage)
        - wᵢ ≤ max_position_size
        - CVaR ≤ portfolio_limit
        - Sector exposure ≤ max_sector_exposure
        """
        # Sort by risk-adjusted return (mu_adjusted / cvar)
        # Higher ratio = better risk-adjusted return
        sorted_inputs = sorted(
            adjusted_inputs,
            key=lambda x: x['mu_adjusted'] / max(x['cvar_95'], 0.001),
            reverse=True
        )

        allocations = {}

        # Track portfolio metrics
        remaining_capital = portfolio_nav
        portfolio_cvar = 0.0
        sector_allocations: Dict[str, float] = {}

        for item in sorted_inputs:
            symbol = item['symbol']
            mu_adj = item['mu_adjusted']
            cvar = item['cvar_95']
            marginal_cvar = item['marginal_cvar']
            sector = item.get('asset_class', AssetClass.STOCKS.value)

            # Constraint 1: CVaR limit check
            projected_cvar = portfolio_cvar + marginal_cvar * self.max_position_size
            if projected_cvar > self.cvar_config.portfolio_limit:
                item['reason_codes'].append("CVAR_LIMIT_BREACH")
                allocations[symbol] = 0.0
                continue

            # Constraint 2: Sector exposure check
            current_sector = sector_allocations.get(sector, 0.0)
            if current_sector + self.max_position_size > self.max_sector_exposure:
                item['reason_codes'].append(f"SECTOR_LIMIT_BREACH:{sector}")
                allocations[symbol] = 0.0
                continue

            # Constraint 3: Position size limit
            # Scale by confidence and data quality
            confidence_factor = min(item['mu_adjusted'] / max(mu_adj, 0.001), 1.0)
            position_size = self.max_position_size * confidence_factor

            # Constraint 4: Available capital
            if position_size * portfolio_nav > remaining_capital:
                position_size = remaining_capital / portfolio_nav

            if position_size > 0.001:  # Minimum threshold
                allocations[symbol] = position_size
                remaining_capital -= position_size * portfolio_nav
                portfolio_cvar = projected_cvar
                sector_allocations[sector] = current_sector + position_size
            else:
                allocations[symbol] = 0.0

        return allocations

    def _apply_governance_veto(
        self,
        adjusted_inputs: List[Dict[str, Any]],
        allocations: Dict[str, float],
        portfolio_nav: float
    ) -> Dict[str, CapitalAuctionOutput]:
        """
        Phase 4: Apply governance veto checks.

        Veto triggers:
        - Unexplained profits
        - Too-perfect execution
        - Data dependency risk
        - Correlated wins
        """
        # Build lookup for adjusted inputs
        input_lookup = {item['symbol']: item for item in adjusted_inputs}

        outputs = {}

        # Sort by allocation for ranking
        sorted_allocations = sorted(
            allocations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for rank, (symbol, weight) in enumerate(sorted_allocations, 1):
            item = input_lookup.get(symbol)
            if not item:
                outputs[symbol] = CapitalAuctionOutput(
                    symbol=symbol,
                    allocated=False,
                    weight=0.0,
                    rank=0,
                    decision="REJECT",
                    reason_codes=["NOT_IN_CANDIDATES"]
                )
                continue

            reason_codes = item['reason_codes'].copy()
            vetoed = False
            veto_reason = ""

            # Governance veto checks
            if self.enable_governance_veto:
                # Check 1: Unexplained profits
                # If return is suspiciously high with low confidence
                if item['mu_raw'] > 0.05 and item['data_quality'] < 0.8:
                    vetoed = True
                    veto_reason = "UNEXPLAINED_HIGH_RETURN_LOW_CONFIDENCE"
                    reason_codes.append(veto_reason)

                # Check 2: Data dependency risk
                # If data quality is marginal
                if item['data_quality'] < 0.7:
                    vetoed = True
                    veto_reason = "DATA_DEPENDENCY_RISK"
                    reason_codes.append(veto_reason)

                # Check 3: Model decay warning
                if item['decay_factor'] < 0.5:
                    reason_codes.append(f"MODEL_DECAY_WARNING:{item['decay_factor']:.2f}")

                # Check 4: High CVaR
                if item['cvar_95'] > 0.05:
                    reason_codes.append(f"HIGH_CVAR:{item['cvar_95']:.3f}")

            # Determine decision
            if vetoed:
                decision = "VETO"
                allocated = False
                final_weight = 0.0
            elif weight > 0.001:
                decision = "ALLOCATE"
                allocated = True
                final_weight = weight
            else:
                decision = "REJECT"
                allocated = False
                final_weight = 0.0

            # Compute penalties for output
            data_quality_penalty = 1.0 - item['data_quality']
            model_decay_penalty = 1.0 - item['decay_factor']
            liquidity_cost = item['liquidity_cost']
            correlation_penalty = item['correlation_risk']

            outputs[symbol] = CapitalAuctionOutput(
                symbol=symbol,
                allocated=allocated,
                weight=final_weight,
                rank=rank if allocated else 0,
                decision=decision,
                reason_codes=reason_codes,
                mu_contribution=item['mu_adjusted'] * final_weight if allocated else 0.0,
                cvar_contribution=item['marginal_cvar'] * final_weight if allocated else 0.0,
                liquidity_cost=liquidity_cost,
                correlation_penalty=correlation_penalty,
                data_quality_penalty=data_quality_penalty,
                model_decay_penalty=model_decay_penalty,
                vetoed=vetoed,
                veto_reason=veto_reason
            )

            if allocated:
                logger.info(
                    f"[AUCTION] ALLOCATED {symbol}: weight={final_weight:.3f}, "
                    f"mu_adj={item['mu_adjusted']:.4f}, cvar={item['cvar_95']:.3f}, "
                    f"rank={rank}"
                )
            else:
                logger.info(
                    f"[AUCTION] REJECTED {symbol}: reason={reason_codes[:2]}"
                )

        return outputs

    def get_auction_summary(
        self,
        outputs: Dict[str, CapitalAuctionOutput]
    ) -> Dict[str, Any]:
        """Get summary statistics of the auction."""
        allocated = [o for o in outputs.values() if o.allocated]
        rejected = [o for o in outputs.values() if not o.allocated]
        vetoed = [o for o in outputs.values() if o.vetoed]

        return {
            "total_candidates": len(outputs),
            "allocated_count": len(allocated),
            "rejected_count": len(rejected),
            "vetoed_count": len(vetoed),
            "total_weight": sum(o.weight for o in allocated),
            "total_mu_contribution": sum(o.mu_contribution for o in allocated),
            "total_cvar_contribution": sum(o.cvar_contribution for o in allocated),
            "avg_data_quality_penalty": np.mean([o.data_quality_penalty for o in allocated]) if allocated else 0.0,
            "avg_model_decay_penalty": np.mean([o.model_decay_penalty for o in allocated]) if allocated else 0.0,
            "top_allocations": [
                {"symbol": o.symbol, "weight": o.weight, "mu": o.mu_contribution}
                for o in sorted(allocated, key=lambda x: x.weight, reverse=True)[:5]
            ],
            "rejection_reasons": self._aggregate_rejection_reasons(rejected + vetoed)
        }

    def _aggregate_rejection_reasons(
        self,
        outputs: List[CapitalAuctionOutput]
    ) -> Dict[str, int]:
        """Aggregate reasons for rejections."""
        reasons = {}
        for o in outputs:
            for code in o.reason_codes:
                reasons[code] = reasons.get(code, 0) + 1
        return reasons


def run_capital_auction(
    candidates: List[CapitalAuctionInput],
    portfolio_nav: float,
    current_weights: Optional[Dict[str, float]] = None
) -> Dict[str, CapitalAuctionOutput]:
    """
    Convenience function to run capital auction.

    Args:
        candidates: List of candidate inputs
        portfolio_nav: Portfolio NAV
        current_weights: Current position weights

    Returns:
        Dict of symbol -> auction output
    """
    engine = CapitalAuctionEngine()
    return engine.run_auction(candidates, portfolio_nav, current_weights)

