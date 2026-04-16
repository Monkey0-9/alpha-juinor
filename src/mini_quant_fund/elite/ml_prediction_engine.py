"""
Advanced ML Prediction Engine
Deep learning and reinforcement learning models matching top quant firms
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import threading
import time

# Add deep learning frameworks if available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

logger = logging.getLogger(__name__)

@dataclass
class PredictionFeatures:
    """Features for ML prediction models"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    volatility: float
    momentum: float
    mean_reversion: float
    order_flow: int
    inventory: float
    time_of_day: int
    day_of_week: int
    market_regime: str
    technical_indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionTarget:
    """Target variables for ML models"""
    future_return_1min: float
    future_return_5min: float
    future_return_15min: float
    future_return_1hr: float
    future_volatility: float
    future_spread: float
    price_direction: int  # 1 for up, -1 for down, 0 for flat

class DeepNeuralNetwork(nn.Module):
    """Deep neural network for price prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ReinforcementLearningTrader:
    """Reinforcement learning trader using Deep Q-Network"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for reinforcement learning")
        
        # Q-Network
        self.q_network = DeepNeuralNetwork(state_size, [256, 128, 64])
        self.target_network = DeepNeuralNetwork(state_size, [256, 128, 64])
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []  # Experience replay
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_action(self, state, training: bool = True):
        """Get action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Limit memory size
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def replay(self, batch_size: int = 32):
        """Train the model using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch])
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if len(self.memory) % 1000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class MLPredictionEngine:
    """
    Machine Learning Prediction Engine
    
    Features:
    - Deep learning price prediction
    - Reinforcement learning trading
    - Ensemble methods
    - Real-time feature engineering
    - Model ensembling
    """
    
    def __init__(self):
        """Initialize ML prediction engine"""
        
        # Feature engineering
        self.feature_scaler = StandardScaler()
        
        # Traditional ML models
        self.price_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42)
        }
        
        # Deep learning models
        self.deep_models = {}
        self.rl_trader = None
        
        # Model performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Training data
        self.training_data = []
        self.feature_history = []
        
        # Real-time prediction cache
        self.prediction_cache = {}
        self.cache_ttl = 60  # seconds
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("ML Prediction Engine initialized with advanced models")
    
    def extract_features(self, market_data: Dict[str, Any]) -> PredictionFeatures:
        """Extract advanced features from market data"""
        
        symbol = market_data.get('symbol', 'UNKNOWN')
        timestamp = market_data.get('timestamp', datetime.now())
        price = market_data.get('price', 0.0)
        volume = market_data.get('volume', 0)
        bid = market_data.get('bid', 0.0)
        ask = market_data.get('ask', 0.0)
        
        # Basic features
        spread = ask - bid if bid > 0 and ask > 0 else 0.0
        
        # Time-based features
        time_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Technical indicators
        technical_indicators = self._calculate_technical_indicators(market_data)
        
        # Market regime
        market_regime = self._detect_market_regime(market_data)
        
        # Advanced features
        volatility = market_data.get('volatility', 0.02)
        momentum = technical_indicators.get('momentum', 0.0)
        mean_reversion = technical_indicators.get('mean_reversion', 0.0)
        order_flow = market_data.get('order_flow', 0)
        inventory = market_data.get('inventory', 0.0)
        
        return PredictionFeatures(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            spread=spread,
            volatility=volatility,
            momentum=momentum,
            mean_reversion=mean_reversion,
            order_flow=order_flow,
            inventory=inventory,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            market_regime=market_regime,
            technical_indicators=technical_indicators
        )
    
    def _calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate technical indicators"""
        
        price = market_data.get('price', 0.0)
        volume = market_data.get('volume', 0)
        
        # Get historical data (simplified)
        historical_prices = market_data.get('historical_prices', [price] * 20)
        
        if len(historical_prices) < 5:
            return {
                'momentum': 0.0,
                'mean_reversion': 0.0,
                'rsi': 50.0,
                'macd': 0.0,
                'bollinger_position': 0.0
            }
        
        # Momentum
        momentum = (price - historical_prices[0]) / historical_prices[0] if historical_prices[0] > 0 else 0.0
        
        # Mean reversion
        mean_price = np.mean(historical_prices)
        std_price = np.std(historical_prices)
        mean_reversion = (mean_price - price) / std_price if std_price > 0 else 0.0
        
        # RSI
        gains = [max(0, historical_prices[i] - historical_prices[i-1]) for i in range(1, len(historical_prices))]
        losses = [max(0, historical_prices[i-1] - historical_prices[i]) for i in range(1, len(historical_prices))]
        
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100.0
        
        # MACD
        ema_12 = pd.Series(historical_prices).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(historical_prices).ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        
        # Bollinger Bands
        bb_upper = mean_price + 2 * std_price
        bb_lower = mean_price - 2 * std_price
        bollinger_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        
        return {
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'rsi': rsi,
            'macd': macd,
            'bollinger_position': bollinger_position
        }
    
    def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect market regime"""
        
        volatility = market_data.get('volatility', 0.02)
        momentum = market_data.get('momentum', 0.0)
        
        if volatility > 0.03:
            if momentum > 0.01:
                return "HIGH_VOL_BULL"
            else:
                return "HIGH_VOL_BEAR"
        elif volatility < 0.01:
            if abs(momentum) < 0.005:
                return "LOW_VOL_SIDEWAYS"
            else:
                return "LOW_VOL_TRENDING"
        else:
            return "NORMAL"
    
    def train_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train all ML models"""
        
        if len(training_data) < 100:
            logger.warning("Insufficient training data")
            return {}
        
        # Extract features and targets
        features = []
        targets = []
        
        for data_point in training_data:
            features_data = self.extract_features(data_point)
            
            # Convert to numerical features
            feature_vector = self._features_to_vector(features_data)
            target_vector = self._targets_to_vector(data_point)
            
            features.append(feature_vector)
            targets.append(target_vector)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train each model
        model_performance = {}
        
        for model_name, model in self.price_models.items():
            logger.info(f"Training {model_name} model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_performance[model_name] = {
                'mse': mse,
                'r2': r2,
                'accuracy': 1 - mse / np.var(y_test)
            }
            
            logger.info(f"{model_name} - MSE: {mse:.6f}, R2: {r2:.3f}")
        
        # Store performance
        self.model_performance = model_performance
        
        # Get feature importance for tree-based models
        if 'random_forest' in self.price_models:
            rf_model = self.price_models['random_forest']
            feature_names = self._get_feature_names()
            importance = rf_model.feature_importances_
            
            self.feature_importance = dict(zip(feature_names, importance))
        
        return model_performance
    
    def _features_to_vector(self, features: PredictionFeatures) -> np.ndarray:
        """Convert features to numerical vector"""
        
        # Basic features
        vector = [
            features.price,
            features.volume,
            features.spread,
            features.volatility,
            features.momentum,
            features.mean_reversion,
            features.order_flow,
            features.inventory,
            features.time_of_day,
            features.day_of_week
        ]
        
        # Technical indicators
        for indicator, value in features.technical_indicators.items():
            vector.append(value)
        
        # Market regime (one-hot encoded)
        regimes = ['HIGH_VOL_BULL', 'HIGH_VOL_BEAR', 'LOW_VOL_SIDEWAYS', 'LOW_VOL_TRENDING', 'NORMAL']
        for regime in regimes:
            vector.append(1.0 if features.market_regime == regime else 0.0)
        
        return np.array(vector)
    
    def _targets_to_vector(self, data_point: Dict[str, Any]) -> np.ndarray:
        """Convert targets to numerical vector"""
        
        return np.array([
            data_point.get('future_return_1min', 0.0),
            data_point.get('future_return_5min', 0.0),
            data_point.get('future_return_15min', 0.0),
            data_point.get('future_return_1hr', 0.0),
            data_point.get('future_volatility', 0.02),
            data_point.get('future_spread', 0.001),
            data_point.get('price_direction', 0)
        ])
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance tracking"""
        
        names = [
            'price', 'volume', 'spread', 'volatility', 'momentum', 
            'mean_reversion', 'order_flow', 'inventory', 'time_of_day', 'day_of_week'
        ]
        
        # Technical indicators
        technical_names = ['rsi', 'macd', 'bollinger_position']
        names.extend(technical_names)
        
        # Market regimes
        regime_names = ['HIGH_VOL_BULL', 'HIGH_VOL_BEAR', 'LOW_VOL_SIDEWAYS', 'LOW_VOL_TRENDING', 'NORMAL']
        names.extend(regime_names)
        
        return names
    
    def predict_price_movement(self, market_data: Dict[str, Any], horizon_minutes: int = 5) -> Dict[str, Any]:
        """Predict price movement using ensemble models"""
        
        # Check cache
        cache_key = f"{market_data.get('symbol', 'UNKNOWN')}_{horizon_minutes}"
        current_time = time.time()
        
        if cache_key in self.prediction_cache:
            cached_prediction, cache_time = self.prediction_cache[cache_key]
            if current_time - cache_time < self.cache_ttl:
                return cached_prediction
        
        # Extract features
        features = self.extract_features(market_data)
        feature_vector = self._features_to_vector(features)
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform([feature_vector])
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.price_models.items():
            try:
                pred = model.predict(feature_vector_scaled)[0]
                predictions[model_name] = pred[horizon_minutes - 1] if horizon_minutes <= 4 else pred[3]
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                predictions[model_name] = 0.0
        
        # Ensemble prediction (weighted average)
        if self.model_performance:
            # Weight by model performance
            weights = {}
            total_weight = 0.0
            
            for model_name in predictions.keys():
                if model_name in self.model_performance:
                    r2 = self.model_performance[model_name]['r2']
                    weight = max(0.1, r2)  # Minimum weight of 0.1
                    weights[model_name] = weight
                    total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
            
            # Calculate weighted prediction
            ensemble_pred = 0.0
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 1.0 / len(predictions))
                ensemble_pred += pred * weight
            
            predictions['ensemble'] = ensemble_pred
        
        # Cache prediction
        self.prediction_cache[cache_key] = (predictions, current_time)
        
        return {
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'horizon_minutes': horizon_minutes,
            'predictions': predictions,
            'confidence': self._calculate_prediction_confidence(predictions),
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now()
        }
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        
        if not predictions:
            return 0.0
        
        # Calculate standard deviation of predictions
        pred_values = list(predictions.values())
        if len(pred_values) < 2:
            return 0.5
        
        std_dev = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        
        # Confidence based on agreement
        if std_dev == 0:
            return 1.0
        
        # Lower standard deviation = higher confidence
        confidence = 1.0 / (1.0 + std_dev * 100)
        
        return max(0.1, min(0.9, confidence))
    
    def initialize_reinforcement_learning(self, state_size: int, action_size: int):
        """Initialize reinforcement learning trader"""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for reinforcement learning")
            return None
        
        self.rl_trader = ReinforcementLearningTrader(state_size, action_size)
        
        logger.info(f"Reinforcement Learning Trader initialized - State size: {state_size}, Action size: {action_size}")
        
        return self.rl_trader
    
    def get_trading_action(self, state: List[float], training: bool = True) -> int:
        """Get trading action from reinforcement learning model"""
        
        if not self.rl_trader:
            return 0  # Hold action
        
        return self.rl_trader.get_action(state, training)
    
    def update_reinforcement_learning(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool):
        """Update reinforcement learning model"""
        
        if not self.rl_trader:
            return
        
        self.rl_trader.remember(state, action, reward, next_state, done)
        self.rl_trader.replay()
    
    def save_models(self, filepath: str):
        """Save trained models"""
        
        model_data = {
            'scaler': self.feature_scaler,
            'models': self.price_models,
            'performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        
        try:
            model_data = joblib.load(filepath)
            
            self.feature_scaler = model_data['scaler']
            self.price_models = model_data['models']
            self.model_performance = model_data['performance']
            self.feature_importance = model_data['feature_importance']
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

def run_ml_prediction_demo():
    """Demonstrate ML prediction capabilities"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ADVANCED ML PREDICTION ENGINE DEMO")
    print("=" * 60)
    
    # Initialize ML engine
    ml_engine = MLPredictionEngine()
    
    # Generate sample training data
    print("\n1. GENERATING TRAINING DATA...")
    training_data = []
    
    for i in range(1000):
        # Simulate market data
        base_price = 150.0 + np.random.normal(0, 5)
        volume = np.random.randint(100000, 1000000)
        volatility = np.random.uniform(0.01, 0.04)
        
        data_point = {
            'symbol': 'AAPL',
            'timestamp': datetime.now() + timedelta(minutes=i),
            'price': base_price,
            'volume': volume,
            'bid': base_price - 0.01,
            'ask': base_price + 0.01,
            'volatility': volatility,
            'momentum': np.random.normal(0, 0.02),
            'inventory': np.random.normal(0, 1000),
            'order_flow': np.random.randint(-10000, 10000),
            'historical_prices': [base_price * (1 + np.random.normal(0, 0.01)) for _ in range(20)]
        }
        
        # Add targets (future returns)
        future_return_1min = np.random.normal(0, 0.001)
        future_return_5min = np.random.normal(0, 0.002)
        future_return_15min = np.random.normal(0, 0.005)
        future_return_1hr = np.random.normal(0, 0.01)
        
        data_point.update({
            'future_return_1min': future_return_1min,
            'future_return_5min': future_return_5min,
            'future_return_15min': future_return_15min,
            'future_return_1hr': future_return_1hr,
            'future_volatility': volatility * np.random.uniform(0.8, 1.2),
            'future_spread': 0.001 * np.random.uniform(0.5, 2.0),
            'price_direction': np.sign(future_return_5min)
        })
        
        training_data.append(data_point)
    
    print(f"Generated {len(training_data)} training samples")
    
    # Train models
    print("\n2. TRAINING ML MODELS...")
    performance = ml_engine.train_models(training_data)
    
    print("\nModel Performance:")
    for model_name, perf in performance.items():
        print(f"  {model_name}:")
        print(f"    MSE: {perf['mse']:.6f}")
        print(f"    R2: {perf['r2']:.3f}")
        print(f"    Accuracy: {perf['accuracy']:.3f}")
    
    # Test predictions
    print("\n3. TESTING PREDICTIONS...")
    
    # Test market data
    test_market_data = {
        'symbol': 'AAPL',
        'timestamp': datetime.now(),
        'price': 150.0,
        'volume': 500000,
        'bid': 149.99,
        'ask': 150.01,
        'volatility': 0.025,
        'momentum': 0.01,
        'inventory': 500.0,
        'order_flow': 5000,
        'historical_prices': [149.5, 149.8, 150.2, 149.9, 150.1] * 4
    }
    
    # Predict different horizons
    horizons = [1, 5, 15, 60]  # minutes
    
    for horizon in horizons:
        prediction = ml_engine.predict_price_movement(test_market_data, horizon)
        
        print(f"\nPrediction Horizon: {horizon} minutes")
        print(f"  Ensemble Prediction: {prediction['predictions']['ensemble']:.6f}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        
        # Show individual model predictions
        for model_name, pred in prediction['predictions'].items():
            if model_name != 'ensemble':
                print(f"  {model_name}: {pred:.6f}")
    
    # Feature importance
    print(f"\nFeature Importance (Top 10):")
    importance = prediction['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for feature, importance_score in sorted_features:
        print(f"  {feature}: {importance_score:.4f}")
    
    # Initialize reinforcement learning
    print("\n4. INITIALIZING REINFORCEMENT LEARNING...")
    
    # State: [price, volume, volatility, momentum, inventory, spread]
    state_size = 5
    # Action: [strong_sell, sell, hold, buy, strong_buy]
    action_size = 5
    
    rl_trader = ml_engine.initialize_reinforcement_learning(state_size, action_size)
    
    if rl_trader:
        print("Reinforcement Learning Trader initialized")
        print("State size: 5 (price, volume, volatility, momentum, inventory, spread)")
        print("Action size: 5 (strong_sell, sell, hold, buy, strong_buy)")
        
        # Test trading action
        test_state = [150.0, 500000, 0.025, 0.01, 500.0, 0.02]
        action = ml_engine.get_trading_action(test_state, training=False)
        
        action_names = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        print(f"Test State: {test_state}")
        print(f"Recommended Action: {action_names[action]}")
    
    print("\n" + "=" * 60)
    print("ML PREDICTION ENGINE DEMO COMPLETE")
    print("=" * 60)
    
    return ml_engine

if __name__ == "__main__":
    ml_engine = run_ml_prediction_demo()
    
    # Save models
    ml_engine.save_models('ml_models.joblib')
    print("\nModels saved to: ml_models.joblib")
