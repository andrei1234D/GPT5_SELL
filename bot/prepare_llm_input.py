# prepare_llm_input.py
import numpy as np
import json
import joblib
import os

BRAIN_DIR = "Brain"
SCALER_PATH = os.path.join(BRAIN_DIR, "scaler_sell.pkl")
FEATURE_PATH = os.path.join(BRAIN_DIR, "feature_cols_sell.json")

# Load once globally
with open(FEATURE_PATH, "r") as f:
    FEATURE_COLS = json.load(f)
SCALER = joblib.load(SCALER_PATH)

def prepare_llm_features(indicators: dict):
    """
    Create standardized feature array from live indicator dictionary.
    Ensures same order and scaling as training dataset.
    """
    row = np.array([indicators.get(col, 0.0) for col in FEATURE_COLS], dtype=float).reshape(1, -1)
    row_scaled = SCALER.transform(row)
    return row_scaled, FEATURE_COLS
