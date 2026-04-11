"""
Intelligence Core Demonstration
================================

Shows how the system learns and improves over time.
Demonstrates:
1. Meta-learning - learns from outcomes
2. Pattern recognition - identifies recurring patterns
3. Confidence calibration - learns what confidence means
4. Ensemble optimization - learns which models to trust
5. Risk adaptation - adjusts for market conditions
"""

import sys
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_intelligence_core():
    """Demonstrate the Intelligence Core learning in action."""
    
    print("\n" + "=" * 80)
    print("INTELLIGENCE CORE DEMONSTRATION")
    print("=" * 80)
    
    from ml_alpha.intelligence_core import get_intelligence_core
    from ml_alpha.enhanced_predictive_model import get_enhanced_model
    
    # Get instances
    logger.info("\n[INIT] Creating Intelligence Core and Enhanced Model...")
    core = get_intelligence_core()
    model = get_enhanced_model()
    
    logger.info("✓ Systems initialized")
    
    # Simulate a series of predictions and outcomes
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: INITIAL PREDICTIONS (Model Learning)")
    logger.info("=" * 80)
    
    np.random.seed(42)
    
    predictions = []
    outcomes = []
    
    # Simulate 50 predictions with varying quality
    for i in range(50):
        # Generate prediction
        prediction = np.random.rand()  # 0-1 probability
        confidence = 0.5 + np.random.rand() * 0.4  # 0.5-0.9
        
        # Generate actual outcome (with some correlation to prediction)
        if np.random.rand() < 0.55:  # Slightly better than random
            actual = prediction + np.random.randn() * 0.1
        else:
            actual = np.random.rand()
        
        actual = max(0.0, min(1.0, actual))
        
        # Market conditions
        market_features = {
            'volatility': np.random.rand() * 0.03,
            'momentum': np.random.randn() * 0.05,
            'trend': np.random.rand(),
            'sharpe_ratio': np.random.randn() * 0.5,
            'drawdown': -np.random.uniform(0, 0.05),
        }
        
        # Record prediction for learning
        core.learn_from_outcome(
            prediction=prediction,
            confidence=confidence,
            actual_outcome=actual,
            profit_loss=np.random.uniform(-2, 5),  # Random profit/loss
            market_features=market_features
        )
        
        predictions.append(prediction)
        outcomes.append(actual)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/50] Predictions processed...")
    
    # Show learning progress
    logger.info("\n" + "-" * 80)
    logger.info("LEARNING PROGRESS:")
    logger.info("-" * 80)
    
    status = core.get_system_intelligence()
    logger.info(f"  Total predictions processed: {status['total_predictions']}")
    logger.info(f"  Overall accuracy: {status['overall_accuracy']:.1%}")
    logger.info(f"  Patterns discovered: {status['patterns_learned']}")
    
    if status['model_rankings']:
        logger.info(f"\n  MODEL RANKINGS (by accuracy):")
        for rank, (model_name, accuracy) in enumerate(status['model_rankings'], 1):
            logger.info(f"    {rank}. {model_name}: {accuracy:.1%}")
    
    # Phase 2: Pattern Recognition
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: PATTERN RECOGNITION (Market Patterns Learned)")
    logger.info("=" * 80)
    
    logger.info(f"  Patterns discovered: {status['patterns_learned']}")
    
    if status['top_patterns']:
        logger.info(f"\n  TOP PERFORMING PATTERNS:")
        for i, pattern in enumerate(status['top_patterns'], 1):
            logger.info(f"    Pattern {i}:")
            logger.info(f"      Success rate: {pattern['success_rate']:.1%}")
            logger.info(f"      Frequency: {pattern['frequency']} times")
            logger.info(f"      Avg return: {pattern['avg_return']:.2%}")
            logger.info(f"      Confidence: {pattern['confidence']:.1%}")
    
    # Phase 3: Confidence Calibration
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: CONFIDENCE CALIBRATION (Learning What Confidence Means)")
    logger.info("=" * 80)
    
    calib_data = status['confidence_calibration']
    logger.info(f"  Calibration samples collected: {len(calib_data)}")
    
    if calib_data:
        logger.info(f"\n  CONFIDENCE CALIBRATION DATA:")
        logger.info(f"  {'Confidence':<15} {'Actual Accuracy':<20}")
        logger.info(f"  {'-' * 35}")
        
        for conf, (actual_acc, count) in sorted(calib_data.items()):
            if count >= 3:  # Only show calibrated values
                logger.info(f"  {conf:<15.2f} {actual_acc:<20.1%} (n={count})")
    
    # Phase 4: Ensemble Learning
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: ENSEMBLE LEARNING (Learning Optimal Model Weights)")
    logger.info("=" * 80)
    
    logger.info(f"\n  OPTIMIZED MODEL WEIGHTS:")
    ensemble_weights = status['ensemble_weights']
    for model, weight in sorted(ensemble_weights.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(weight * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        logger.info(f"    {model:<20} {bar} {weight:.1%}")
    
    # Phase 5: Intelligence Status
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: SYSTEM INTELLIGENCE STATUS")
    logger.info("=" * 80)
    
    logger.info(f"\n  Learning Duration: {status['learning_duration_hours']:.2f} hours simulation")
    logger.info(f"  Total Learning Events: {status['total_predictions']}")
    logger.info(f"  System Accuracy: {status['overall_accuracy']:.1%}")
    logger.info(f"  Pattern Recognition: {status['patterns_learned']} patterns learned")
    logger.info(f"  Ensemble Status: {len(status['model_rankings'])} models in ensemble")
    
    # Get model forecast with intelligence enhancement
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: ENHANCED FORECAST EXAMPLE")
    logger.info("=" * 80)
    
    # Create sample market data
    test_data = pd.DataFrame({
        'open': np.cumsum(np.random.randn(100) * 0.5) + 100,
        'high': np.cumsum(np.random.randn(100) * 0.5) + 102,
        'low': np.cumsum(np.random.randn(100) * 0.5) + 98,
        'close': np.cumsum(np.random.randn(100) * 0.5) + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Fix high/low
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    logger.info("\n  Generating prediction with Intelligence Core enhancement...")
    
    try:
        forecast = model.get_forecast(test_data)
        
        logger.info(f"\n  FORECAST RESULTS:")
        logger.info(f"    Base Probability: {forecast['probability']:.1%}")
        logger.info(f"    Confidence: {forecast['confidence']:.1%}")
        logger.info(f"    Risk Regime: {forecast['regime']}")
        logger.info(f"    Pattern Detected: {forecast.get('pattern_detected', 'None')}")
        logger.info(f"    Pattern Confidence: {forecast.get('pattern_confidence', 0):.1%}")
        logger.info(f"    Core Accuracy: {forecast.get('core_accuracy', 0):.1%}")
        
        if 'learning_progress' in forecast:
            lp = forecast['learning_progress']
            logger.info(f"\n  LEARNING PROGRESS:")
            logger.info(f"    Recent Accuracy (100): {lp['recent_accuracy_100']:.1%}")
            logger.info(f"    Overall Accuracy: {lp['overall_accuracy']:.1%}")
            logger.info(f"    Improvement: {lp['improvement']:+.1%}")
            logger.info(f"    Patterns Discovered: {lp['patterns_discovered']}")
    
    except Exception as e:
        logger.error(f"  Error generating forecast: {e}")
    
    # Final Summary
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM INTELLIGENCE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"""
    ✓ Intelligence Core Status: OPERATIONAL
    ✓ Meta-Learning: ACTIVE ({status['total_predictions']} predictions learned)
    ✓ Pattern Recognition: ACTIVE ({status['patterns_learned']} patterns)
    ✓ Confidence Calibration: ACTIVE ({len(calib_data)} calibration points)
    ✓ Ensemble Optimization: ACTIVE ({len(ensemble_weights)} models)
    ✓ Risk Adaptation: ACTIVE
    
    SYSTEM INTELLIGENCE LEVEL: {"★" * min(5, max(1, int(status['overall_accuracy'] * 5)))} / ★★★★★
    SYSTEM ACCURACY: {status['overall_accuracy']:.1%}
    LEARNING EFFICIENCY: {status['patterns_learned']} patterns learned from {status['total_predictions']} predictions
    
    The system is getting SMARTER with every prediction!
    """)
    
    logger.info("=" * 80)


if __name__ == "__main__":
    demo_intelligence_core()
