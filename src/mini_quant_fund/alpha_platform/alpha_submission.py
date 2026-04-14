import os
from .alpha_dsl import AlphaDSL
from .backtest_engine import DistributedBacktestEngine

class AlphaIDE:
    """Web-based Research IDE for Alpha Development (Mock)"""
    
    def __init__(self):
        self.engine = DistributedBacktestEngine()
        
    def render_editor(self):
        return """
        # Alpha Research IDE
        alpha = (close - ts_mean(close, 20)) / ts_std(close, 20)
        backtest(alpha)
        """

class AlphaSubmissionAPI:
    """API for submitting alphas to the production pipeline"""
    
    def __init__(self, repo_path: str = "src/mini_quant_fund/alpha_platform/repository/"):
        self.repo_path = repo_path
        os.makedirs(repo_path, exist_ok=True)
        
    def submit(self, researcher_id: str, alpha_name: str, expression: str):
        """Save alpha expression to repository"""
        filename = f"{researcher_id}_{alpha_name}.alpha"
        with open(os.path.join(self.repo_path, filename), "w") as f:
            f.write(expression)
        return {"status": "success", "alpha_id": filename}
