
import sys
import sklearn.ensemble
# Patch legacy sklearn path
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble

import pickle
import joblib
from pathlib import Path

model_path = Path('models/ml_alpha/return_model.pkl')
print(f"Loading {model_path} with patch...")
try:
    with open(model_path, 'rb') as f:
        m = joblib.load(f)
    print(f"Success! Type: {type(m)}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
