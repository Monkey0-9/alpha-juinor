import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.governance_engine import GovernanceEngine, VetoTrigger
from governance.institutional_specification import GovernanceDecision, CVaRConfig

# Setup logging to file
log_path = r"C:\mini-quant-fund\runtime\agent_results\20260121_091700\governance_test_log.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GOV_TEST")

def run_governance_test():
    engine = GovernanceEngine(enable_all_vetoes=True)
    results = []

    # Helper to create a base decision
    def base_decision(symbol="AAPL"):
        return GovernanceDecision(
            decision="EXECUTE_BUY",
            reason_codes=[],
            mu=0.01,
            sigma=0.02,
            cvar=0.03,
            data_quality=0.8,
            model_confidence=0.8,
            symbol=symbol,
            position_size=0.05,
            cycle_id="test_cycle",
            timestamp=datetime.utcnow().isoformat(),
            vetoed=False,
            veto_reason="",
            veto_triggers={}
        )

    test_cases = [
        ("UNEXPLAINED_PROFIT", lambda d: setattr(d, 'mu', 0.05) or setattr(d, 'model_confidence', 0.3)),
        ("HIGH_CVAR", lambda d: setattr(d, 'cvar', 0.08)), # limit is 0.06 in our cvar_test if using defaults
        ("MODEL_DECAY", lambda d: setattr(d, 'model_confidence', 0.4)),
        ("LOW_DATA_QUALITY", lambda d: setattr(d, 'data_quality', 0.4)),
        ("INSUFFICIENT_HISTORY", lambda d: setattr(d, 'symbol', "")),
        ("DATA_DEPENDENCY_RISK", lambda d: setattr(d, 'data_quality', 0.65)),
        ("TOO_PERFECT_EXECUTION", lambda d: setattr(d, 'mu', 0.1) or setattr(d, 'sigma', 0.001)),
        ("DRAWDOWN_LIMIT", lambda d: None), # portfolio_state check
        ("LEVERAGE_BREACH", lambda d: setattr(d, 'position_size', 1.5)),
        ("CORRELATED_WINS", lambda d: None) # requires repeating decisions
    ]

    with open(log_path, "w") as f:
        f.write("GOVERNANCE VETO SIMULATION REPORT\n")
        f.write("="*40 + "\n\n")

        for name, modifier in test_cases:
            d = base_decision()
            modifier(d)

            # Special case for drawdown/leverage
            portfolio_state = {}
            if name == "DRAWDOWN_LIMIT":
                portfolio_state = {"drawdown": 0.25}
            elif name == "LEVERAGE_BREACH":
                portfolio_state = {"leverage": 0.0} # 1.5 position size will breach 1.0 limit

            # Special case for correlated wins
            if name == "CORRELATED_WINS":
                for _ in range(6):
                    engine.evaluate_decision(base_decision("AAPL_WIN"))

            out = engine.evaluate_decision(d, portfolio_state)

            status = "PASS" if out.veto_triggers.get(VetoTrigger[name].value) else "FAIL"
            # Note: GovernanceEngine might not veto for "info/warning" severities but should trigger.
            # We check if 'triggered' is true in the internal logic.

            line = f"Trigger: {name:<25} | Triggered: {out.veto_triggers.get(VetoTrigger[name].value)} | Vetoed: {out.vetoed} | Status: {status}\n"
            f.write(line)
            print(line, end='')

    print(f"\nGovernance test log written to {log_path}")

if __name__ == "__main__":
    run_governance_test()
