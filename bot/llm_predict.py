"""
llm_predict.py

MT (MarketTrend) "brain" for production. Loads your 3 best models:
  - best_model_MT-1.joblib (bear)
  - best_model_MT0.joblib  (neutral)
  - best_model_MT1.joblib  (bull)

Optional (recommended) sidecar metadata JSONs:
  - best_model_MT-1.json / best_model_MT0.json / best_model_MT1.json

If JSONs are missing, the brain will still run, but gating thresholds may fall back to defaults.

Primary API used by decision_engine.py:

  SellBrain.predict_prob(indicators: dict, market_trend: int) -> dict

Optional CLI utility (used by GitHub Actions if you want):
  python bot/llm_predict.py
    - reads:  bot/LLM_data/input_llm/llm_input_latest.csv
    - writes: bot/LLM_data/input_llm/llm_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from joblib import load


# -------------------------
# Small utilities
# -------------------------
def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _infer_prob_from_regressor(pred: float, sell_threshold: float) -> float:
    """
    Your MT models are regressors predicting SellScore (roughly 0..1, not guaranteed).
    Convert that into a probability-like confidence score via a logistic around sell_threshold.
    """
    pred = float(pred)
    sell_threshold = float(sell_threshold)

    # Higher k => more "decisive" (more step-like)
    k = 12.0
    x = pred - sell_threshold
    prob = 1.0 / (1.0 + np.exp(-k * x))
    return float(prob)


def _gate_from_prob(prob: Optional[float], prob_thr: Optional[float]) -> float:
    """
    Gate behavior:
      - prob <= thr => 0
      - prob > thr  => linearly scaled to 0..1
    """
    if prob is None or prob_thr is None:
        return 0.0
    prob = float(prob)
    prob_thr = float(prob_thr)
    if prob_thr >= 1.0:
        return 0.0
    if prob <= prob_thr:
        return 0.0
    return _clip01((prob - prob_thr) / (1.0 - prob_thr))


def _expected_joblib_name(regime: int) -> str:
    return f"best_model_MT{regime}.joblib" if regime != 0 else "best_model_MT0.joblib"


def _expected_json_name(regime: int) -> str:
    return f"best_model_MT{regime}.json" if regime != 0 else "best_model_MT0.json"


def _looks_like_model_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    # Require at least one expected file so we don't accidentally accept "." or unrelated dirs.
    for r in (-1, 0, 1):
        if (p / _expected_joblib_name(r)).exists():
            return True
    return False


# -------------------------
# SellBrain
# -------------------------
class SellBrain:
    def __init__(self, model_dir: Optional[str | Path] = None):
        """
        model_dir resolution order:
          1) explicit model_dir argument
          2) env SELL_MODEL_DIR
          3) common repo locations
        """
        candidates: list[Path] = []

        if model_dir is not None:
            candidates.append(Path(model_dir))

        env_dir = os.getenv("SELL_MODEL_DIR", "").strip()
        if env_dir:
            candidates.append(Path(env_dir))

        # Common repo locations
        candidates.extend([
            Path("Brain"),
        ])

        self.model_dir: Optional[Path] = None
        for c in candidates:
            # Skip empty strings which become "."
            if str(c).strip() in ("", "."):
                continue
            if _looks_like_model_dir(c):
                self.model_dir = c
                break

        if self.model_dir is None:
            raise FileNotFoundError(
                    "Could not locate model directory. Set SELL_MODEL_DIR or place files in one of: "
                    "Brain / bot/Brain / SELL_trainer_agent_outputs / bot/models / models."
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
        job = self.model_dir / _expected_joblib_name(regime)
        js = self.model_dir / _expected_json_name(regime)

        if not job.exists():
            raise FileNotFoundError(f"Missing model file: {job}")

        payload = load(job)

        # Meta JSON is recommended but optional.
        meta: Dict[str, Any] = {}
        if js.exists():
            try:
                with open(js, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        # If meta doesn't have feature_columns, attempt to recover from payload.
        if not meta.get("feature_columns") and isinstance(payload, dict):
            feats = payload.get("feature_columns")
            if isinstance(feats, list) and feats:
                meta["feature_columns"] = feats

        # Ensure regime/model_type populated
        meta.setdefault("regime", int(regime))
        if isinstance(payload, dict):
            meta.setdefault("model_type", payload.get("model_type"))

        self._models[regime] = payload
        self._meta[regime] = meta

    def _ensure_mt0_sell_signal(self):
        m0 = self._meta.get(0, {})
        if isinstance(m0.get("sell_signal"), dict):
            return

        m_neg = self._meta.get(-1, {})
        m_pos = self._meta.get(1, {})

        thr_neg = _safe_float((m_neg.get("sell_signal") or {}).get("sell_threshold"))
        thr_pos = _safe_float((m_pos.get("sell_signal") or {}).get("sell_threshold"))

        if thr_neg is None or thr_pos is None:
            # If we can't synthesize, leave MT0 without sell_signal (defaults will apply).
            return

        mean_thr = 0.5 * (thr_neg + thr_pos)
        prob_thr = _safe_float((m_neg.get("sell_signal") or {}).get("prob_threshold"), self._default_prob_thr)

        self._meta[0]["sell_signal"] = {
            "definition": "good_sell := (SellScore >= sell_threshold)",
            "sell_threshold": float(mean_thr),
            "prob_threshold": float(prob_thr),
            "calibration_diag": {
                "note": "Synthesized as mean of bear/bull thresholds; re-train MT0 calibration when MT0 has stable validation groups."
            },
        }

    def _predict_sellscore(self, payload: dict, X_row: np.ndarray) -> float:
        """
        Supports:
          - XGB payload: {"model":{"booster":..., "imputer":...}, ...}
          - ET payload: pipeline stored as payload["model"] or payload itself if saved that way
        """
        if not isinstance(payload, dict):
            # If you ever saved bare estimator, support it.
            return float(payload.predict(X_row.reshape(1, -1))[0])

        mtype = payload.get("model_type")
        if mtype == "XGB":
            booster = payload["model"]["booster"]
            imp = payload["model"]["imputer"]
            X_imp = imp.transform(X_row.reshape(1, -1))
            import xgboost as xgb
            return float(booster.predict(xgb.DMatrix(X_imp))[0])

        model = payload.get("model", payload)
        return float(model.predict(X_row.reshape(1, -1))[0])

    def predict_prob(self, indicators: Dict[str, Any], market_trend: int = 0) -> Dict[str, Any]:
        """
        Predicts a probability-like SELL gate score using the best model for the given regime.

        Requires `indicators` dict to contain the feature columns expected by that regime.
        Missing features are filled with NaN (imputer inside ET pipeline handles them).
        """
        regime = int(market_trend)
        if regime not in (-1, 0, 1):
            regime = 0

        payload = self._models[regime]
        meta = self._meta[regime]

        feats = meta.get("feature_columns", [])
        if not isinstance(feats, list) or not feats:
            return {
                "prob": None,
                "prob_threshold": float(self._default_prob_thr),
                "weight": float(self._weight_by_regime.get(regime, 0.20)),
                "regime": int(regime),
                "model_type": str(meta.get("model_type") or (payload.get("model_type") if isinstance(payload, dict) else "unknown")),
            }

        row = np.array([_safe_float(indicators.get(c), np.nan) for c in feats], dtype=float)
        pred_sellscore = self._predict_sellscore(payload, row)

        sell_signal = meta.get("sell_signal") if isinstance(meta.get("sell_signal"), dict) else {}
        sell_thr = _safe_float(sell_signal.get("sell_threshold"), 0.5)
        prob_thr = _safe_float(sell_signal.get("prob_threshold"), self._default_prob_thr)

        prob = _infer_prob_from_regressor(pred_sellscore, sell_threshold=sell_thr)

        return {
            "prob": float(prob),
            "prob_threshold": float(prob_thr),
            "weight": float(self._weight_by_regime.get(regime, 0.20)),
            "regime": int(regime),
            "model_type": str(meta.get("model_type") or (payload.get("model_type") if isinstance(payload, dict) else "unknown")),
            "pred_sellscore": float(pred_sellscore),
            "sell_threshold": float(sell_thr),
        }


# -------------------------
# Optional batch prediction utility
# -------------------------
def run_batch_predictions(
    input_csv: str | Path = "bot/LLM_data/input_llm/llm_input_latest.csv",
    output_csv: str | Path = "bot/LLM_data/input_llm/llm_predictions.csv",
    model_dir: Optional[str | Path] = None,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    brain = SellBrain(model_dir=model_dir)

    out_rows = []
    for _, r in df.iterrows():
        row_dict = r.to_dict()
        mt = int(_safe_float(row_dict.get("MarketTrend"), 0) or 0)
        if mt not in (-1, 0, 1):
            mt = 0

        pred = brain.predict_prob(row_dict, market_trend=mt)
        prob = pred.get("prob")
        prob_thr = pred.get("prob_threshold")
        gate = _gate_from_prob(prob, prob_thr)

        out = {
            "Ticker": row_dict.get("Ticker"),
            "MarketTrend": mt,
            "mt_prob": prob,
            "mt_prob_threshold": prob_thr,
            "mt_gate": gate,
            "mt_weight": pred.get("weight"),
            "pred_sellscore": pred.get("pred_sellscore"),
            "sell_threshold": pred.get("sell_threshold"),
            "model_type": pred.get("model_type"),
            "Timestamp": row_dict.get("Timestamp"),
        }

        # Preserve a few useful context columns if present
        for k in ("current_price", "pnl_pct", "avg_price", "shares", "invested_lei"):
            if k in row_dict:
                out[k] = row_dict.get(k)

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    # deterministic, consistent order
    preferred = [
        "Timestamp", "Ticker", "MarketTrend",
        "current_price", "pnl_pct",
        "mt_prob", "mt_prob_threshold", "mt_gate", "mt_weight",
        "pred_sellscore", "sell_threshold", "model_type",
        "avg_price", "shares", "invested_lei",
    ]
    cols = [c for c in preferred if c in out_df.columns] + [c for c in out_df.columns if c not in preferred]
    out_df = out_df[cols]

    out_df.to_csv(output_csv, index=False)
    return output_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="bot/LLM_data/input_llm/llm_input_latest.csv")
    ap.add_argument("--output", type=str, default="bot/LLM_data/input_llm/llm_predictions.csv")
    ap.add_argument("--model_dir", type=str, default=None)
    args = ap.parse_args()

    out = run_batch_predictions(args.input, args.output, model_dir=args.model_dir)
    print(f"✅ Wrote MT predictions → {out}")


if __name__ == "__main__":
    main()
