
import joblib
import pandas as pd
import numpy as np

model_path = 'models/ml_alpha/return_model.pkl'
try:
    m = joblib.load(model_path)
    print(f"Model Type: {type(m)}")
    if isinstance(m, dict):
        print(f"Keys: {list(m.keys())}")
        for k, v in m.items():
             print(f"  {k}: {type(v)}")

    # Try to see if it has feature_names_in_
    if hasattr(m, 'feature_names_in_'):
        print(f"Features: {m.feature_names_in_}")
    elif hasattr(m, 'named_steps'):
        print(f"Pipeline Steps: {m.named_steps}")
        if 'scaler' in m.named_steps:
             print("Pipeline contains a scaler")

except Exception as e:
    print(f"Error: {e}")
