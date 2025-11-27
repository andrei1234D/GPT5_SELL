import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from prepare_llm_input import SCALER, FEATURE_COLS

MODEL_PATH = "Brain/brain_sell.pth"
INPUT_FILE = "bot/LLM_data/input_llm/llm_input_latest.csv"
OUTPUT_FILE = "bot/LLM_data/input_llm/llm_predictions.csv"

class SellBrain:
    """Encapsulates the trained SELL model for inference."""
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load PyTorch model architecture & weights."""
        input_dim = len(FEATURE_COLS)

        class SellMLP(torch.nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(512, 256),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(256, 128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(128, 1)
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)

        self.model = SellMLP(input_dim)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()
        print("🧠 LLM SELL brain loaded successfully.")

    def predict_prob(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute SELL probabilities for all rows."""
        with torch.no_grad():
            tensor = torch.tensor(X_scaled, dtype=torch.float32)
            probs = torch.sigmoid(self.model(tensor)).numpy().flatten()
        return probs


def run_llm_predictions():
    """Run SELL model inference on latest dataset and save results."""
    if not os.path.exists(INPUT_FILE):
        print(f"⚠️ Missing dataset: {INPUT_FILE}")
        return

    # Load dataset
    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        print("⚠️ Dataset is empty.")
        return

    # Filter to feature columns only
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"⚠️ Missing required columns: {missing}")
        return

    X = df[FEATURE_COLS].astype(float)
    X_scaled = SCALER.transform(X)

    # Run predictions
    model = SellBrain()
    df["LLM_Sell_Prob"] = model.predict_prob(X_scaled)
    df["LLM_Signal"] = (df["LLM_Sell_Prob"] > 0.5).astype(int)
    df["Run_Timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save predictions
    output_cols = ["Ticker", "LLM_Sell_Prob", "LLM_Signal", "Run_Timestamp"]
    df[output_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ LLM predictions saved → {OUTPUT_FILE}")
    print(df[output_cols])

    return df[output_cols]


if __name__ == "__main__":
    run_llm_predictions()
