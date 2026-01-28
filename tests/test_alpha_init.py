
from alpha_families.ml_alpha import MLAlpha
import logging
logging.basicConfig(level=logging.INFO)
print("Instantiating MLAlpha...")
alpha = MLAlpha()
print("Success!")
if 'LEGACY_GLOBAL' in alpha._cached_models:
    print("Legacy global model loaded successfully.")
else:
    print("Legacy global model NOT loaded (not found or error).")
