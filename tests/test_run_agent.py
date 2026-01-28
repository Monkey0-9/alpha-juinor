
import sys
import os
sys.path.append(os.getcwd())

from alpha_families.agent_runner import run_agent
import pandas as pd

class MockAgent:
    def evaluate(self, df, **kwargs):
        print(f"MockAgent evaluated with kwargs: {kwargs}")
        return {'mu': 0.5}

agent = MockAgent()
df = pd.DataFrame()
res = run_agent(agent, df, regime_context={'regime': 'NORMAL'}, features={'test': 1})
print(f"Result: {res}")
