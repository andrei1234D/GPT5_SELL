# llm_predict.py
import torch
import numpy as np
from prepare_llm_input import prepare_llm_features

MODEL_PATH = "Brain/brain_sell.pth"

class SellBrain:
    """Encapsulates the LLM SELL model for clean inference."""
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load PyTorch model structure & weights."""
        # Import late to avoid circular import
        from prepare_llm_input import FEATURE_COLS
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
            def forward(self, x): return self.net(x).squeeze(-1)

        self.model = SellMLP(input_dim)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

    def predict_prob(self, indicators: dict) -> float:
        """Return model probability of SELL (0–1)."""
        X_scaled, _ = prepare_llm_features(indicators)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(torch.tensor(X_scaled, dtype=torch.float32))).item()
        return float(prob)
