
import pandas as pd
from typing import Dict
from risk.cvar import compute_cvar
from risk.evt import fit_pot_tail

def compute_tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Wrapper that returns {cvar, evtrisk_score}
    Safe, neutral defaults on error.
    """
    try:
        cvar = compute_cvar(returns)
        evt_res = fit_pot_tail(returns)
        
        return {
            "cvar": cvar,
            "evtrisk_score": evt_res.get('tail_risk_score', 0.0),
            "xi": evt_res.get('shape_xi', 0.0)
        }
    except Exception:
        return {"cvar": 0.0, "evtrisk_score": 0.0, "xi": 0.0}
