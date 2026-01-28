import pytest
import pandas as pd
import numpy as np
from alpha_families.ml_alpha import MLAlpha
from utils.metrics import metrics

class MockModel:
    def __init__(self, feature_names=None):
        if feature_names:
            self.feature_names_in_ = feature_names
    def predict(self, X):
        # Return 0.5 for any input
        return np.array([0.5])

def test_ml_predict_safe_success():
    ml = MLAlpha()
    model = MockModel(feature_names=['f1', 'f2'])
    X = pd.DataFrame({'f1': [1.0], 'f2': [2.0]})

    pred = ml.ml_predict_safe(model, X, symbol="TEST")
    assert pred is not None
    assert pred[0] == 0.5
    assert ml.model_errors == 0

def test_missing_feature_returns_none():
    metrics.model_errors = 0
    ml = MLAlpha()
    model = MockModel(feature_names=['f1', 'f2'])
    X = pd.DataFrame({'f1': [1.0]}) # f2 missing

    pred = ml.ml_predict_safe(model, X, symbol="TEST")
    assert pred is None
    assert ml.model_errors == 1
    assert metrics.model_errors == 1

def test_ml_predict_safe_no_metadata():
    ml = MLAlpha()
    model = MockModel() # No feature_names_in_
    X = pd.DataFrame({'f1': [1.0], 'f2': [2.0]})

    # Should convert to numpy and predict without raising
    pred = ml.ml_predict_safe(model, X, symbol="TEST")
    assert pred is not None
    assert pred[0] == 0.5

def test_exception_in_predict_returns_none():
    class ExplodingModel:
        def predict(self, X):
            raise ValueError("BOOM")

    ml = MLAlpha()
    pred = ml.ml_predict_safe(ExplodingModel(), pd.DataFrame({'A': [1]}), symbol="FAILTEST")
    assert pred is None
    assert ml.model_errors > 0
