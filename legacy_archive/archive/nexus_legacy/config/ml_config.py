"""
Machine Learning Configuration
Centralizes thresholds, paths, and hyperparameters for the
Predictive Alpha engine.
"""


class MLConfig:
    # Model Paths
    MODEL_PATH = "models/lgbm_predictor_v1.pkl"
    FEATURE_STORE_PATH = "data/feature_store/"

    # Prediction Thresholds
    # Probability > BULLISH_THRESHOLD -> High Confidence Buy Boost
    BULLISH_THRESHOLD = 0.65

    # Probability < BEARISH_THRESHOLD -> High Confidence Sell/Short Boost
    BEARISH_THRESHOLD = 0.35

    # Impact Multipliers
    BOOST_MULTIPLIER = 1.25  # Increase score by 25%
    PENALTY_MULTIPLIER = 0.70  # Decrease score by 30%

    # Training Params
    TRAIN_TEST_SPLIT = 0.15
    FORECAST_HORIZON_HOURS = 24
    TARGET_RETURN_THRESHOLD = 0.01  # 1%

    @classmethod
    def get_adaptive_thresholds(cls, current_volatility: float) -> tuple[float, float]:
        """
        Returns dynamic thresholds based on market volatility.
        If volatility is high (> 3%), require higher confidence.

        Args:
            current_volatility: Standard deviation of returns (e.g., 20-period rolling std).

        Returns:
            (bullish_threshold, bearish_threshold)
        """
        # Base values
        bull = cls.BULLISH_THRESHOLD
        bear = cls.BEARISH_THRESHOLD

        # Adaptive Logic: High Volatility -> Stricter Standards
        if current_volatility > 0.03:  # > 3% volatility
            bull += 0.05  # 0.70
            bear -= 0.05  # 0.30

        return bull, bear
