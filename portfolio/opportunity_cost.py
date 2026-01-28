"""
portfolio/opportunity_cost.py

Opportunity Cost Engine (Ticket 19)
Enforces "Why this trade?" logic by auditing marginal utility.

Logic:
Rejects trades that do not demonstrate a significant edge over the next best alternative.
"""

import logging
from typing import List
from portfolio.capital_competition import CompetitionResult

logger = logging.getLogger("OPPORTUNITY_COST")

class OpportunityCostManager:
    """
    Manages opportunity cost checks.
    """

    def __init__(self, epsilon: float = 0.001):
        self.epsilon = epsilon  # Minimum score separation required
        logger.info(f"OpportunityCostManager initialized with Epsilon={self.epsilon}")

    def filter_candidates(self, ranked_results: List[CompetitionResult]) -> List[CompetitionResult]:
        """
        Apply opportunity cost filter.

        Logic:
        Iterate through ranked candidates.
        For Candidate i vs Candidate i+1:
             If Score[i] - Score[i+1] < Epsilon:
                 Reject Candidate i (Marginal Utility Fail)
                 Audit reason: "Score gap {diff} < {epsilon} vs {Symbol i+1}"
        """
        if not ranked_results or len(ranked_results) < 2:
            return ranked_results

        # Assume input is already sorted by score desc
        filtered_results = []

        # Working copy to modify decisions without breaking iteration if needed
        # But here we just iterate and modify in place or rebuild list?
        # We need to look ahead.

        # If we reject top candidate, do we compare the *next* candidate to the one after it?
        # Yes. The comparison is pairwise down the list.
        # But if i is rejected, does i+1 become the new "Top" to compare against i+2?
        # The prompt says "Reject any trade that does not beat... score_i - score_i+1".
        # Independent pairwise checks on the original rank seems the most robust interpretation.

        for i in range(len(ranked_results) - 1):
            current = ranked_results[i]
            next_best = ranked_results[i+1]

            # Skip if already rejected by Competition Engine
            if current.decision == "REJECT":
                filtered_results.append(current)
                continue

            # Skip if next is rejected?
            # If next is rejected (e.g. negative score), then current is competing against "nothing valid".
            # So opportunity cost is satisfied (it beats the void).
            if next_best.decision == "REJECT" or next_best.score < 0:
                 current.components["opp_cost_pass"] = "beats_rejected"
                 filtered_results.append(current)
                 continue

            diff = current.score - next_best.score

            if diff < self.epsilon:
                # REJECT due to Opportunity Cost
                current.decision = "REJECT"
                current.reason = f"OPP_COST: Gap {diff:.5f} < {self.epsilon} vs {next_best.symbol}"
                current.weight = 0.0
                current.components["opp_cost_diff"] = diff
                logger.info(f"Rejecting {current.symbol}: {current.reason}")
            else:
                # PASS
                current.components["why_this_trade"] = f"Rank #{i+1} beats #{i+2} ({next_best.symbol}) by +{diff:.4f}"

            filtered_results.append(current)

        # Append last item (nothing to compare against, so passes if valid)
        last = ranked_results[-1]
        if last.decision != "REJECT":
             last.components["why_this_trade"] = "Last valid candidate"
        filtered_results.append(last)

        return filtered_results
