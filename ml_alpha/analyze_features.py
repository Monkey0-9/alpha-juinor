import argparse
import os
import sys

sys.path.append(os.getcwd())
from ml_alpha.predictive_model import PredictiveModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    print("Loading Predictive Model...")
    model = PredictiveModel()

    importance = model.get_feature_importance()

    if not importance:
        # Fallback if model not trained or no importance
        print("No feature importance available (Model might not be trained).")
        # Print dummy for verification of script logic
        importance = {"price_momentum": 100, "volatility": 80, "volume_delta": 60, "smc_signal": 120}

    print(f"\n--- Top {args.top} Features ---")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(sorted_features[:args.top]):
        print(f"{i+1}. {name}: {score}")

if __name__ == "__main__":
    main()
