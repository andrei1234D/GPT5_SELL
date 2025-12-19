"""
llm_predict.py

This file implements the MT (MarketTrend) "brain" that loads your 3 best models:
  - best_model_MT-1.joblib/.json (bear)
  - best_model_MT0.joblib/.json  (neutral)
  - best_model_MT1.joblib/.json  (bull)

It provides a single production-facing method:

  SellBrain.predict_prob(indicators: dict, market_trend: int) -> dict

Return dict includes:
  - prob: float or None           (probability-like score from MT gate model)
  - prob_threshold: float         (gate threshold, default 0.65)
  - weight: float                 (regime-specific weight: -1=0.40, 0=0.20, 1=0.30)
  - regime: int                   (-1/0/1 actually used)
  - model_type: str               ("ET" or "XGB")
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from joblib import load


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _infer_prob_from_regressor(pred: float, sell_threshold: float) -> float:
    """
    Your MT models are regressors predicting SellScore (0..1-ish, not guaranteed).
    We convert that into a probability-like gate score with a logistic around the regime threshold.

    This is a monotonic confidence score suitable for gating, not a perfectly calibrated probability.
    """
    pred = float(pred)
    sell_threshold = float(sell_threshold)

    # Decisiveness of the gate (higher => more "yes/no")
    k = 12.0
    x = (pred - sell_threshold)
    prob = 1.0 / (1.0 + np.exp(-k * x))
    return float(prob)


class SellBrain:
    def __init__(self, model_dir: Optional[str | Path] = None):
        """
        model_dir:
          - defaults to env SELL_MODEL_DIR if set
          - else tries common locations
        """
        env_dir = os.getenv("SELL_MODEL_DIR", "").strip()
        if model_dir is None and env_dir:
            model_dir = env_dir

        candidates = []
        if model_dir is not None:
            candidates.append(Path(model_dir))

        # common repo/colab locations (safe to try)
        candidates.extend([
            Path("SELL_trainer_agent_outputs"),
            Path("bot/models"),
            Path("models"),
            Path("/content/drive/MyDrive/SELL_trainer_agent_outputs"),
        ])

        self.model_dir = None
        for c in candidates:
            if c.exists():
                self.model_dir = c
                break

        if self.model_dir is None:
            raise FileNotFoundError(
                "Could not locate model directory. Set SELL_MODEL_DIR or place files in one of: "
                "SELL_trainer_agent_outputs / bot/models / models."
            )

        self._models: Dict[int, Any] = {}
        self._meta: Dict[int, Dict[str, Any]] = {}

        for regime in (-1, 0, 1):
            self._load_regime(regime)

        # Regime-specific weights (as agreed)
        self._weight_by_regime = {-1: 0.40, 0: 0.20, 1: 0.30}

        # Default gating probability threshold (json may override)
        self._default_prob_thr = 0.65

        # If MT0 is missing sell_signal, synthesize it as mean of -1 and 1 thresholds
        self._ensure_mt0_sell_signal()

    def _load_regime(self, regime: int):
        job = self.model_dir / (f"best_model_MT{regime}.joblib" if regime != 0 else "best_model_MT0.joblib")
        js = self.model_dir / (f"best_model_MT{regime}.json" if regime != 0 else "best_model_MT0.json")

        if not job.exists():
            raise FileNotFoundError(f"Missing model: {job}")
        if not js.exists():
            raise FileNotFoundError(f"Missing meta json: {js}")

        payload = load(job)
        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._models[regime] = payload
        self._meta[regime] = meta

    def _ensure_mt0_sell_signal(self):
        m0 = self._meta.get(0, {})
        if "sell_signal" in m0 and isinstance(m0["sell_signal"], dict):
            return

        m_neg = self._meta.get(-1, {})
        m_pos = self._meta.get(1, {})

        s_neg = m_neg.get("sell_signal", {})
        s_pos = m_pos.get("sell_signal", {})

        thr_neg = _safe_float(s_neg.get("sell_threshold"))
        thr_pos = _safe_float(s_pos.get("sell_threshold"))

        if thr_neg is None or thr_pos is None:
            return

        mean_thr = 0.5 * (thr_neg + thr_pos)
        prob_thr = _safe_float(s_neg.get("prob_threshold"), self._default_prob_thr)

        self._meta[0]["sell_signal"] = {
            "definition": "good_sell := (SellScore >= sell_threshold)",
            "sell_threshold": float(mean_thr),
            "prob_threshold": float(prob_thr),
            "calibration_diag": {
                "note": "Synthesized as mean of bear/bull thresholds; re-train MT0 calibration when MT0 has stable validation groups."
            }
        }

    def _predict_sellscore(self, payload: dict, X_row: np.ndarray) -> float:
        mtype = payload.get("model_type")

        if mtype == "XGB":
            booster = payload["model"]["booster"]
            imp = payload["model"]["imputer"]
            X_imp = imp.transform(X_row.reshape(1, -1))
            import xgboost as xgb
            return float(booster.predict(xgb.DMatrix(X_imp))[0])

        model = payload["model"]  # ET pipeline
        return float(model.predict(X_row.reshape(1, -1))[0])

    def predict_prob(self, indicators: Dict[str, Any], market_trend: int = 0) -> Dict[str, Any]:
        """
        Predicts a probability-like SELL gate score using the best model for the given regime.

        Requires indicators dict to contain the feature columns in each model's meta json.
        Missing features are filled with NaN.
        """
        regime = int(market_trend)
        if regime not in (-1, 0, 1):
            regime = 0

        payload = self._models[regime]
        meta = self._meta[regime]
        feats = meta.get("feature_columns", [])
        if not feats:
            return {
                "prob": None,
                "prob_threshold": float(self._default_prob_thr),
                "weight": float(self._weight_by_regime.get(regime, 0.20)),
                "regime": regime,
                "model_type": payload.get("model_type"),
            }

        row = np.array([_safe_float(indicators.get(c), np.nan) for c in feats], dtype=float)
        pred_sellscore = self._predict_sellscore(payload, row)

        sell_signal = meta.get("sell_signal", {}) if isinstance(meta.get("sell_signal", {}), dict) else {}
        sell_thr = _safe_float(sell_signal.get("sell_threshold"), 0.5)
        prob_thr = _safe_float(sell_signal.get("prob_threshold"), self._default_prob_thr)

        prob = _infer_prob_from_regressor(pred_sellscore, sell_threshold=sell_thr)

        return {
            "prob": float(prob),
            "prob_threshold": float(prob_thr),
            "weight": float(self._weight_by_regime.get(regime, 0.20)),
            "regime": int(regime),
            "model_type": str(payload.get("model_type")),
            "pred_sellscore": float(pred_sellscore),
            "sell_threshold": float(sell_thr),
        }
