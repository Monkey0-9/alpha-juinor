
import numpy as np
import math
from typing import List

class ModelDisagreement:
    """
    Calculates penalty based on model divergence.
    """
    def __init__(self, beta: float = 5.0):
        self.beta = beta

    def calculate_penalty(self, mus: List[float]) -> float:
        """
        penalty = exp(-beta * (1 / (1 + var_mu)))
        If var_mu is high, 1/(1+var) is small, penalty is close to 1 (low penalty?).
        Wait, logic check:
        High disagreement (high var) -> We want LESS confidence usually? or MORE?
        User formula: exp(-beta * (1/(1+var)))
        If var -> infinity, 1/(1+var) -> 0, exp(0) -> 1. (No penalty).
        If var -> 0, 1/(1+0) -> 1, exp(-beta). (Max penalty).

        This seems inverted? Typically high disagreement = lower confidence.
        Maybe "Disagreement" here means "We punish CONSENSUS if it's too tight"?
        OR maybe the user formula implies "High variance = penalty"?

        Let's re-read: "penalty = exp(-beta * (1 / (1 + var_mu)))"
        Let beta=5.
        Var=0 -> exp(-5) = 0.006 (Heavy Penalty).
        Var=100 -> exp(-5 * 0.01) = exp(-0.05) = 0.95 (Low Penalty).

        Implies: Punish Consensus (Groupthink). Encourages robust disagreement?
        Actually, maybe the user wants to PENALIZE high variance.
        If user wants to penalize high variance, formula should be `exp(-beta * var)`.

        However, I MUST follow the user's formula EXACTLY as requested:
        "penalty = exp(-beta * (1 / (1 + var_mu)))"

        I will implement it exactly.
        """
        if not mus or len(mus) < 2:
            return 1.0

        var_mu = np.var(mus)
        exponent = -self.beta * (1.0 / (1.0 + var_mu))
        return math.exp(exponent)
