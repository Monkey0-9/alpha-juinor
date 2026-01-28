
import pickle
import joblib
from pathlib import Path

model_path = Path('models/ml_alpha/return_model.pkl')
print(f"Loading {model_path} with pickle...")
try:
    with open(model_path, 'rb') as f:
        # pickle.load might fail if joblib uses custom headers, but let's try
        m = pickle.load(f)
    print(f"Pickle Success! Type: {type(m)}")
except Exception as e:
    print(f"Pickle Failed: {e}")
    print("Trying joblib...")
    try:
        m = joblib.load(model_path)
        print(f"Joblib Success! Type: {type(m)}")
    except Exception as e2:
        print(f"Joblib Failed: {e2}")
