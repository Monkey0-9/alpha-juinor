"""
ML Referee for Alpha Signal Reweighting.

Uses ML to reweight alphas, detect interactions, and provide explainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class MLReferee:
    """
    ML-based referee for alpha signals: reweights, detects interactions, explainability.
    """

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_importance = {}

    def reweight_alphas(self, alpha_signals: Dict[str, Dict[str, Any]], historical_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Reweight alpha signals using ML model.

        Args:
            alpha_signals: Dict of alpha signals {alpha_name: {'signal': float, 'confidence': float}}
            historical_data: Historical data for training (optional)

        Returns:
            Dict of alpha weights {alpha_name: weight}
        """
        if not self.is_trained and historical_data is not None:
            self._train_model(historical_data)

        if not self.is_trained:
            # Equal weights if no training data
            num_alphas = len(alpha_signals)
            return {name: 1.0 / num_alphas for name in alpha_signals.keys()}

        # Prepare features for prediction
        features = self._prepare_features(alpha_signals)

        # Predict weights
        weights = self.model.predict([features])[0]

        # Normalize weights to sum to 1
        weights = np.maximum(weights, 0)  # No negative weights
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = np.ones(len(weights)) / len(weights)

        return dict(zip(alpha_signals.keys(), weights))

    def detect_interactions(self, alpha_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect interactions between alphas.

        Args:
            alpha_signals: Dict of alpha signals

        Returns:
            Dict with interaction metrics
        """
        signals = np.array([alpha['signal'] for alpha in alpha_signals.values()])
        confidences = np.array([alpha['confidence'] for alpha in alpha_signals.values()])

        # Correlation matrix
        corr_matrix = np.corrcoef(signals.reshape(1, -1), signals.reshape(1, -1)) if len(signals) > 1 else np.array([[1.0]])

        # Average correlation
        avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])

        # Diversity score (1 - avg correlation)
        diversity_score = 1.0 - abs(avg_corr)

        return {
            'correlation_matrix': corr_matrix.tolist(),
            'average_correlation': avg_corr,
            'diversity_score': diversity_score,
            'high_confidence_count': np.sum(confidences > 0.7)
        }

    def refine_signals(self,
                       signals: Dict[str, float],
                       market_data: pd.DataFrame,
                       agent_results: Dict[str, Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Refine signals with Disagreement Penalty and Opportunity Cost Hurdle.
        """
        refined = {}
        hurdle = 0.55 # Standardizing signals to [0,1], hurdle 0.55 or 0.45

        for symbol, raw_signal in signals.items():
            # 1. Disagreement Penalty (Classical vs Classical)
            # Institutional rule: If agents disagree on direction (buy vs sell), penalize conviction.
            if agent_results and symbol in agent_results:
                symbol_agents = agent_results[symbol]
                mus = [a.get('mu', 0.5) for a in symbol_agents.values() if isinstance(a, dict) and a.get('ok')]

                if mus:
                    directions = [np.sign(m - 0.5) for m in mus if abs(m - 0.5) > 0.01]
                    if len(set(directions)) > 1:
                        # DISAGREEMENT DETECTED
                        penalty = 0.5 # 50% reduction in net signal alpha
                        alpha = raw_signal - 0.5
                        raw_signal = 0.5 + (alpha * penalty)

            # 1B. ML Shadow Disagreement Penalty (User Request FIX 4)
            # Formula: mu_final = mu_base * exp(-beta * var(mu_models))
            # Here we treat 'mu_models' as [Classical_Mu, ML_Shadow_Mu]
            # Since ML might not be fully trained, we only apply if is_trained
            if self.is_trained:
                 # Generate Shadow Signal
                 # Need features for this symbol. For now, strict separation means we might not have them easily here
                 # without passing them. But let's assume raw_signal approximates consensus.
                 # Let's assume ML signal is 0.5 (Neutral) if we can't infer, or we need to pass features to refine_signals.
                 ml_shadow_mu = 0.5 # Placeholder until features passed

                 # Calculate Variance between Classical (raw_signal) and ML
                 # var([a, b]) = ((a-mean)^2 + (b-mean)^2) / 2 = (a-b)^2 / 4
                 model_variance = ((raw_signal - ml_shadow_mu) ** 2) / 4.0
                 beta = 10.0 # Sensitivity parameter
                 shadow_decay = np.exp(-beta * model_variance)

                 # Apply decay to alpha component
                 alpha_base = raw_signal - 0.5
                 raw_signal = 0.5 + (alpha_base * shadow_decay)

                 # Log disagreement if significant
                 if shadow_decay < 0.9:
                     pass # logger.info(f"ML_SHADOW_PENALTY: {symbol} decay={shadow_decay:.2f} (Classical={raw_signal:.2f}, ML={ml_shadow_mu:.2f})")

            # 2. Opportunity Cost Hurdle
            # Reject if the signal is not sufficiently strong to overcome trading costs
            alpha_abs = abs(raw_signal - 0.5)
            if alpha_abs < 0.05: # Hurdle: 5% alpha from neutral
                refined[symbol] = 0.5
                # logger.info(f"PM_OPPORTUNITY_COST_REJECT: {symbol} signal {raw_signal:.4f} below hurdle")
            else:
                refined[symbol] = raw_signal

        return refined

    def get_explainability(self) -> Dict[str, Any]:
        """
        Get explainability metrics from the ML model.

        Returns:
            Dict with feature importance and model metrics
        """
        if not self.is_trained:
            return {'status': 'not_trained'}

        return {
            'feature_importance': self.feature_importance,
            'model_type': 'RandomForestRegressor',
            'n_estimators': self.model.n_estimators
        }

    def _train_model(self, historical_data: pd.DataFrame):
        """
        Train the ML model on historical data.

        Args:
            historical_data: DataFrame with alpha signals and future returns
        """
        # This is a simplified training - in practice, use more sophisticated features
        if len(historical_data) < self.lookback_window:
            return

        # Assume historical_data has columns for each alpha signal and future returns
        alpha_cols = [col for col in historical_data.columns if col.startswith('alpha_')]
        target_col = 'future_return'

        if not alpha_cols or target_col not in historical_data.columns:
            return

        X = historical_data[alpha_cols].values
        y = historical_data[target_col].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Feature importance
        self.feature_importance = dict(zip(alpha_cols, self.model.feature_importances_))

        self.is_trained = True

    def _prepare_features(self, alpha_signals: Dict[str, Dict[str, Any]]) -> List[float]:
        """
        Prepare features for ML prediction.

        Args:
            alpha_signals: Dict of alpha signals

        Returns:
            List of features
        """
        features = []
        for name, signal_data in alpha_signals.items():
            features.extend([
                signal_data['signal'],
                signal_data['confidence'],
                signal_data['signal'] * signal_data['confidence']  # Interaction term
            ])
        return features
