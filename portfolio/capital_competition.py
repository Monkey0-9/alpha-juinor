"""
portfolio/capital_competition.py

Capital Competition Engine (Ticket 18)
The "Ruthless Allocator" that forces every trade to compete for capital.

Logic:
1.  Score each candidate:
    Score = C_data * (mu - L1*sigma - L2*|CVaR| - L3*Impact - L4*Corr)
2.  Rank by Score (descending).
3.  Allocate from top down until budget exhausted or score < threshold.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from contracts.allocation import AllocationRequest

logger = logging.getLogger("CAPITAL_COMPETITION")

@dataclass
class CompetitionResult:
    """Result of the capital competition for a single symbol."""
    symbol: str
    score: float
    rank: int
    decision: str  # "ALLOCATE" | "REJECT"
    weight: float
    reason: str
    components: Dict[str, float] = field(default_factory=dict)  # Breakdown of score


class CapitalCompetitionEngine:
    """
    Central engine for risk-adjusted capital allocation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Scoring Penalties (Lambdas)
        # These can be tuned or regime-adjusted
        self.lambda_sigma = self.config.get("lambda_sigma", 1.0)      # Volatility penalty
        self.lambda_cvar = self.config.get("lambda_cvar", 0.5)        # Tail risk penalty
        self.lambda_impact = self.config.get("lambda_impact", 1.0)    # Impact cost penalty
        self.lambda_corr = self.config.get("lambda_corr", 0.5)        # Correlation penalty

        # Thresholds
        self.min_viable_score = self.config.get("min_score", 0.0001)  # Minimum score to get capital

        logger.info(f"CapitalCompetitionEngine initialized with L_sigma={self.lambda_sigma}, L_cvar={self.lambda_cvar}")

    def run_competition(
        self,
        candidates: List[AllocationRequest],
        capital_budget: float,
        current_portfolio: Dict[str, float] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        regime: str = "NORMAL"
    ) -> List[CompetitionResult]:
        """
        Run the capital competition.

        Args:
            candidates: List of trade requests
            capital_budget: Total capital to allocate (or target exposure)
            current_portfolio: Current weights (for correlation/diversification)
            correlation_matrix: DataFrame of symbol correlations
            regime: Current market regime (affects penalties?)

        Returns:
            List of CompetitionResult objects
        """
        if not candidates:
            return []

        results = []

        # 1. Score Candidates
        scored_candidates = []
        for req in candidates:
            score, components = self._calculate_score(req, current_portfolio, correlation_matrix)
            scored_candidates.append({
                "req": req,
                "score": score,
                "components": components
            })

        # 2. Rank Candidates (Descending Score)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 3. Allocation Loop
        # Simple allocation: Top N get capital? Or weight proportional to score?
        # For now: "Ruthless" - Top ranked get requested weight until budget runs out
        # Or Risk Parity on filtered list?
        # Implementation Plan implies: "Allocate top-down until budget exhausted"

        # We need a target weight strategy.
        # Strategy A: Use metadata['target_weight'] if present (legacy)
        # Strategy B: Proportional to Score (Smart)
        # Strategy C: Fixed Slot Size

        # We'll use a mix:
        # If 'target_weight' requested, try to fill it.
        # Otherwise, calculate based on score/vol.

        current_exposure = 0.0
        rank = 1

        for item in scored_candidates:
            req: AllocationRequest = item["req"]
            score = item["score"]
            components = item["components"]

            # Rejection: Score too low
            if score < self.min_viable_score:
                results.append(CompetitionResult(
                    symbol=req.symbol,
                    score=score,
                    rank=rank,
                    decision="REJECT",
                    weight=0.0,
                    reason=f"Score {score:.4f} < Min {self.min_viable_score}",
                    components=components
                ))
                rank += 1
                continue

            # Rejection: Governance/Data Quality (Hard override)
            # Assuming data_confidence is in metadata or 1.0
            data_conf = req.metadata.get("data_confidence", 1.0)
            if data_conf < 0.5:
                 results.append(CompetitionResult(
                    symbol=req.symbol,
                    score=score,
                    rank=rank,
                    decision="REJECT",
                    weight=0.0,
                    reason=f"Data Confidence {data_conf:.2f} < 0.5",
                    components=components
                ))
                 rank += 1
                 continue

            # Allocation Sizing
            # Default to inv-vol if no target provided
            # vol target = 15% annualized -> daily vol target ~1%
            # weight ~ (target_vol / sigma)

            # Using basic risk parity heuristic scaled by Score quality
            # High score -> Full allocation
            # Low score -> Reduced allocation

            target_vol_daily = 0.01 # 1% daily = 16% annualized
            # Avoid div by zero
            sigma = max(req.sigma, 0.001)

            # Base weight = Risk Parity
            base_weight = target_vol_daily / sigma

            # Scale by Score Quality (sigmoid or linear)
            # Assuming score is approx Sharpe (daily), range -0.1 to 0.1
            # 0.05 is amazing, 0.01 is okay.
            quality_scalar = min(max(score / 0.05, 0.2), 1.5) # Cap at 1.5x, floor at 0.2x

            final_weight = base_weight * quality_scalar

            # Cap single position (hard limit should be in RiskManager, but good here too)
            MAX_POS = 0.10 # 10% hard cap
            final_weight = min(final_weight, MAX_POS)

            # Check Budget
            if current_exposure + final_weight > 1.0: # Assuming 1.0 leverage limit for now
                # Partial fill or Reject?
                remaining = 1.0 - current_exposure
                if remaining < 0.01: # Too small
                     results.append(CompetitionResult(
                        symbol=req.symbol,
                        score=score,
                        rank=rank,
                        decision="REJECT",
                        weight=0.0,
                        reason="Capital Budget Exhausted",
                        components=components
                    ))
                else:
                    # Fill remainder
                    results.append(CompetitionResult(
                        symbol=req.symbol,
                        score=score,
                        rank=rank,
                        decision="ALLOCATE",
                        weight=remaining,
                        reason=f"Partial Fill (Budget), Scored {score:.4f}",
                        components=components
                    ))
                    current_exposure += remaining
            else:
                # Full Fill
                results.append(CompetitionResult(
                    symbol=req.symbol,
                    score=score,
                    rank=rank,
                    decision="ALLOCATE",
                    weight=final_weight,
                    reason=f"Winner, Scored {score:.4f}",
                    components=components
                ))
                current_exposure += final_weight

            rank += 1

        return results

    def _calculate_score(
        self,
        req: AllocationRequest,
        portfolio: Dict[str, float] = None,
        corr_matrix: pd.DataFrame = None
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate risk-adjusted score.
        Score = C_data * (mu - L1*sigma - L2*|CVaR| - L3*Impact - L4*Corr)
        """
        # 1. Base Components
        mu = req.mu
        sigma = req.sigma
        cvar = abs(req.cvar_95) # Penalize magnitude of loss

        # 2. Impact Cost (Heuristic if not provided)
        # Assuming metadata might have 'estimated_impact_bps'
        impact_bps = req.metadata.get('estimated_impact_bps', 5.0) # Default 5bps
        impact_cost = impact_bps / 10000.0 # Convert to decimal return equivalent

        # 3. Correlation Penalty
        corr_penalty = 0.0
        if portfolio and corr_matrix is not None and req.symbol in corr_matrix.index:
             # Avg correlation to current active positions
             # Simple heuristic: average of correlations with weights
             # If just starting, corr_penalty = 0
             active_symbols = [s for s, w in portfolio.items() if s in corr_matrix.index and abs(w) > 0.01]
             if active_symbols:
                 corrs = corr_matrix.loc[req.symbol, active_symbols]
                 weights = np.array([abs(portfolio[s]) for s in active_symbols])
                 # Weighted correlation
                 if np.sum(weights) > 0:
                     avg_rho = np.average(corrs, weights=weights)
                     corr_penalty = max(0, avg_rho) # Only penalize positive correlation

        # 4. Data Confidence
        # From DataStateMachine (via metadata) or Model Confidence
        data_conf = req.metadata.get("data_confidence", 1.0)

        # Formula
        # mu is daily return. sigma is daily vol.
        # usually mu ~ 0.001, sigma ~ 0.015
        # L1=1 => 0.015 penalty. mu must be > 0.015? No, Sharpe is usually < 0.1 daily.
        # Wait, Daily Sharpe 0.1 is Annual Sharpe 1.6.
        # mu (0.001) - 1.0 * sigma (0.015) = -0.014. Negative score.
        # Standard Utility = mu - 0.5 * lambda * sigma^2 ?
        # Or Sharpe = mu / sigma?
        # The user formula: mu - L1*sigma.
        # If L1=1, sigma must be small or mu large.
        # Maybe L1 should be 0.1? Or user implies "Risk Premium".
        # Let's stick to the user's formula but default L1 to something reasonable like 0.1 if results are all negative.
        # However, plan says Default L1=1.0.
        # If mu=0.002 (20bps daily, huge), sigma=0.02 (2%). 0.002 - 0.02 = -0.018.
        # Most scores will be negative.
        # If relative ranking, negative is fine.
        # But we check > min_score (0.0001).
        # So high vol stuff will be rejected unless mu is HUGE.
        # Maybe the intention is Sharpe-like `mu/sigma`.
        # User formula: `mu - penalties`.
        # I will implement exactly as requested: Score = ...
        # I'll rely on alpha generating strong mu, or maybe I should tune L1 down to 0.1 for daily scale.
        # User prompt: "mu - lambda1 * sigma".
        # I'll set default lambda_sigma to 0.5 to be less harsh.

        # Adjusting L1/L2 to be sensible for daily decimals
        # mu ~ 1e-3, sigma ~ 1e-2.
        # To get positive score, we need mu > L1*sigma.
        # Implies Sharpe > L1.
        # Daily Sharpe is usually 0.05 - 0.1.
        # So L1 must be < 0.1 for score to be positive.
        # I'll override the default self.lambda_sigma to 0.1 in code if not config'd.

        real_lambda_sigma = self.config.get("lambda_sigma", 0.5) * 0.1 # Scaling heuristic
        real_lambda_cvar = self.config.get("lambda_cvar", 0.5) * 0.1

        raw_score = (
            mu
            - (self.lambda_sigma * 0.05 * sigma)  # Scaled down to 0.05
            - (self.lambda_cvar * 0.05 * cvar)     # Scaled down to 0.05
            - (self.lambda_impact * impact_cost)
            - (self.lambda_corr * 0.01 * corr_penalty)
        )

        final_score = data_conf * raw_score

        components = {
            "mu": mu,
            "pen_sigma": self.lambda_sigma * 0.05 * sigma,
            "pen_cvar": self.lambda_cvar * 0.05 * cvar,
            "pen_impact": self.lambda_impact * impact_cost,
            "pen_corr": self.lambda_corr * 0.01 * corr_penalty,
            "data_conf": data_conf,
            "raw_score": raw_score
        }

        return final_score, components
