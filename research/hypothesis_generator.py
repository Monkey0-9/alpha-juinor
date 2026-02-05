#!/usr/bin/env python3
"""
AUTOMATED RESEARCH: HYPOTHESIS GENERATOR
========================================

S-Class Initiative 3: Continuous Alpha Discovery.
Generates testable trading hypotheses from factor permutations.
"""

import itertools
import random
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Hypothesis:
    id: str
    factor_type: str # 'MOMENTUM', 'MEAN_REVERSION', 'VOLATILITY'
    params: Dict
    rationale: str

class HypothesisGenerator:
    """
    Generates trading hypotheses for the research farm.
    """

    def __init__(self):
        self.lookbacks = [5, 10, 20, 50, 100, 200, 252]
        self.universes = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']

    def generate_momentum_hypotheses(self) -> List[Hypothesis]:
        """Generate Momentum ideas (Trend Following)."""
        hypotheses = []

        # Simple Price Momentum
        for lb in self.lookbacks:
            h = Hypothesis(
                id=f"MOM_{lb}D",
                factor_type="MOMENTUM",
                params={"lookback": lb, "threshold": 0.0},
                rationale=f"Assets that went up over {lb} days will continue to rise."
            )
            hypotheses.append(h)

        return hypotheses

    def generate_mean_reversion_hypotheses(self) -> List[Hypothesis]:
        """Generate Mean Reversion ideas."""
        hypotheses = []

        # RSI Variations
        rsi_periods = [7, 14, 21]
        thresholds = [(30, 70), (20, 80), (10, 90)]

        for p, (os, ob) in itertools.product(rsi_periods, thresholds):
            h = Hypothesis(
                id=f"MR_RSI_{p}_{os}_{ob}",
                factor_type="MEAN_REVERSION",
                params={"indicator": "RSI", "period": p, "oversold": os, "overbought": ob},
                rationale=f"RSI({p}) < {os} indicates oversold conditions."
            )
            hypotheses.append(h)

        return hypotheses

    def generate_all(self) -> List[Hypothesis]:
        """Generate all permutations."""
        all_h = []
        all_h.extend(self.generate_momentum_hypotheses())
        all_h.extend(self.generate_mean_reversion_hypotheses())
        return all_h

def demo():
    print("="*60)
    print("     AUTOMATED HYPOTHESIS GENERATOR")
    print("="*60)

    gen = HypothesisGenerator()
    hypotheses = gen.generate_all()

    print(f"Generated {len(hypotheses)} testable hypotheses.")
    print("-" * 60)

    # Sample 5 random ideas
    for h in random.sample(hypotheses, 5):
        print(f"[{h.id}] {h.factor_type}: {h.rationale}")
        print(f"   Params: {h.params}")
        print("")

if __name__ == "__main__":
    demo()
