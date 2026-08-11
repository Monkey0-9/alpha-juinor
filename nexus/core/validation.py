import pandas as pd
from typing import List, Tuple
from nexus.research.backtest import BacktestEngine, BacktestResult

class ValidationFramework:
    """
    Handles walk-forward testing (WFT) and out-of-sample validation 
    to minimize overfitting.
    """
    def __init__(self, n_splits=5, train_size=0.7):
        self.n_splits = n_splits
        self.train_size = train_size
        self.backtester = BacktestEngine()

    def create_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Creates rolling train/test splits for Walk-Forward Testing.
        """
        splits = []
        n_samples = len(data)
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_samples)
            
            # Simple expanding window train, fixed window test
            train_end = start_idx + int((end_idx - start_idx) * self.train_size)
            
            train = data.iloc[:train_end]
            test = data.iloc[train_end:end_idx]
            splits.append((train, test))
            
        return splits
        
    def evaluate_model(self, model, data: pd.DataFrame):
        """
        Evaluates a model across WFT splits.
        """
        splits = self.create_walk_forward_splits(data)
        metrics = []
        
        for i, (train, test) in enumerate(splits):
            # Generate signals on test data
            signals = test.apply(lambda row: model.predict(pd.DataFrame([row])), axis=1)
            
            # Run backtest
            result = self.backtester.run(test['close'], signals)
            metrics.append(result.metrics)
            
        return metrics
