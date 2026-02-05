import numpy as np
import pandas as pd
import pytest

from ml.baselines import get_baselines


class TestClassicalBaselines:

    def setup_method(self):
        self.baselines = get_baselines()

        # Synthetic Data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = 2 * self.X_train[:, 0] + 0.5 * self.X_train[:, 1] + 0.1 * np.random.randn(100)

        self.X_test = np.random.randn(20, 5)
        self.y_test = 2 * self.X_test[:, 0] + 0.5 * self.X_test[:, 1] + 0.1 * np.random.randn(20)

    def test_linear_regression(self):
        model = self.baselines.train_linear(self.X_train, self.y_train)
        metrics = self.baselines.evaluate(model, self.X_test, self.y_test, "linear")

        assert "mse" in metrics
        assert metrics["mse"] < 0.1  # Should be low for linear data

    def test_ridge_reproducibility(self):
        """Ensure model training is deterministic."""
        model1 = self.baselines.train_ridge(self.X_train, self.y_train)
        metrics1 = self.baselines.evaluate(model1, self.X_test, self.y_test, "ridge")

        model2 = self.baselines.train_ridge(self.X_train, self.y_train)
        metrics2 = self.baselines.evaluate(model2, self.X_test, self.y_test, "ridge")

        assert metrics1["mse"] == metrics2["mse"]

    def test_arima_integration(self):
        """Test ARIMA if installed."""
        try:
            import statsmodels
            series = pd.Series(np.random.randn(100) + np.linspace(0, 10, 100))
            model = self.baselines.train_arima(series)

            if model:
                # Basic check that it returns a result wrapper
                assert hasattr(model, "forecast")
        except ImportError:
            pytest.skip("statsmodels not installed")
