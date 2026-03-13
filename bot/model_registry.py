"""
Unified model registry for GPT5_SELL.

Loads:
  - Goodsell models (joblib payloads; one per market trend)
  - Peak ML models (JSON or joblib)
  - Quality ML models (joblib payloads)

Provides:
  - get_feature_columns(model_family, regime)
  - predict_mt / predict_peak / predict_quality

Note:
  The historical interface name predict_mt is kept for compatibility with the
  rest of the live stack, but it now returns the retrained per-market-trend
  goodsell classifier probability.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import load


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _resolve_dir(env_var: str, fallback: str) -> Path:
    p = (os.getenv(env_var, "") or "").strip()
    if p:
        return Path(p)
    return Path(fallback)


def _expected_goodsell_name(regime: int) -> str:
    return f"goodsell_MT{regime}.joblib" if regime != 0 else "goodsell_MT0.joblib"


def _expected_peak_name(regime: int) -> str:
    return f"peak_model_MT{regime}.json" if regime != 0 else "peak_model_MT0.json"


def _expected_quality_name(regime: int) -> str:
    return f"goodsell_quality_MT{regime}.joblib" if regime != 0 else "goodsell_quality_MT0.joblib"


def _load_goodsell_models(sell_dir: Path) -> Dict[int, dict]:
    models: Dict[int, dict] = {}
    for regime in (-1, 0, 1):
        path = sell_dir / _expected_goodsell_name(regime)
        if not path.exists():
            raise FileNotFoundError(f"Missing goodsell model joblib: {path}")
        payload = load(path)
        models[regime] = payload if isinstance(payload, dict) else {"model": payload}
    return models


def _load_peak_models(peak_dir: Path) -> Dict[int, dict]:
    models: Dict[int, dict] = {}
    for regime in (-1, 0, 1):
        path = peak_dir / _expected_peak_name(regime)
        if not path.exists():
            raise FileNotFoundError(f"Missing peak model JSON: {path}")
        models[regime] = json.loads(path.read_text(encoding="utf-8"))
    return models


def _load_quality_models(quality_dir: Path) -> Dict[int, dict]:
    models: Dict[int, dict] = {}
    for regime in (-1, 0, 1):
        path = quality_dir / _expected_quality_name(regime)
        if not path.exists():
            raise FileNotFoundError(f"Missing quality model joblib: {path}")
        payload = load(path)
        models[regime] = payload if isinstance(payload, dict) else {"model": payload}
    return models


def _predict_payload_prob(payload: dict, row: Dict[str, Any]) -> float:
    feats = payload.get("feature_columns", []) or []
    model = payload.get("model", payload)
    X = pd.DataFrame([{c: _safe_float(row.get(c), np.nan) for c in feats}])
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            prob = None
    if prob is None:
        try:
            prob = float(model.predict(X)[0])
        except Exception:
            prob = 0.0
    return float(max(0.0, min(1.0, _safe_float(prob, 0.0))))


class ModelRegistry:
    def __init__(
        self,
        mt_dir: Optional[str | Path] = None,
        peak_dir: Optional[str | Path] = None,
        quality_dir: Optional[str | Path] = None,
        mt_meta_dir: Optional[str | Path] = None,
    ):
        del mt_meta_dir  # kept only for compatibility with the old constructor

        if mt_dir is None:
            mt_dir = _resolve_dir("SELL_MODEL_DIR", "sell_models")
        self.mt_dir = Path(mt_dir)
        self.mt_models = _load_goodsell_models(self.mt_dir)

        if peak_dir is None:
            peak_dir = _resolve_dir("PEAK_MODEL_DIR", "peak_models")
        self.peak_dir = Path(peak_dir)
        self.peak_models = _load_peak_models(self.peak_dir)

        if quality_dir is None:
            quality_dir = _resolve_dir("QUALITY_MODEL_DIR", "quality_models")
        self.quality_dir = Path(quality_dir)
        self.quality_models = _load_quality_models(self.quality_dir)

    def get_feature_columns(self, family: str, regime: int) -> list[str]:
        family = str(family).lower().strip()
        regime = int(regime) if regime in (-1, 0, 1) else 0

        if family == "mt":
            return list(self.mt_models[regime].get("feature_columns", []) or [])
        if family == "peak":
            return list(self.peak_models[regime].get("feature_columns", []) or [])
        if family == "quality":
            return list(self.quality_models[regime].get("feature_columns", []) or [])
        return []

    def predict_mt(self, row: Dict[str, Any], market_trend: int) -> Dict[str, Any]:
        regime = int(market_trend) if market_trend in (-1, 0, 1) else 0
        payload = self.mt_models[regime]
        prob = _predict_payload_prob(payload, row)
        thr = _safe_float(payload.get("threshold"), 0.5)
        denom = max(1.0 - float(thr), 1e-9)
        gate = 0.0 if prob < thr else max(0.0, min(1.0, (prob - float(thr)) / denom))
        sell_signal = bool(prob >= float(thr))
        return {
            "mt_prob": prob,
            "mt_prob_threshold": float(thr),
            "mt_sell_signal": sell_signal,
            "mt_gate": float(gate),
            "mt_weight": float(gate),
            "pred_sellscore": float(prob),
            "sell_threshold": float(thr),
            "model_type": payload.get("model_type", "ETC"),
            "prob_source": "goodsell_classifier",
        }

    def predict_peak(self, row: Dict[str, Any], market_trend: int) -> Dict[str, Any]:
        regime = int(market_trend) if market_trend in (-1, 0, 1) else 0
        payload = self.peak_models[regime]
        feats = payload.get("feature_columns", []) or []

        X = np.array([_safe_float(row.get(c), np.nan) for c in feats], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        prob = None
        if payload.get("model_type") == "LOGREG_NP":
            w = np.asarray(payload.get("weights", []), dtype=float)
            b = float(payload.get("bias", 0.0))
            mu = np.asarray(payload.get("mu", []), dtype=float)
            sd = np.asarray(payload.get("sd", []), dtype=float)
            if mu.size == X.size:
                Xn = (X - mu) / np.where(sd == 0, 1.0, sd)
            else:
                Xn = X
            z = float(Xn @ w + b)
            prob = 1.0 / (1.0 + np.exp(-z))
        else:
            model = payload.get("model")
            if model is not None:
                try:
                    prob = float(model.predict_proba(X.reshape(1, -1))[:, 1][0])
                except Exception:
                    try:
                        prob = float(model.predict(X.reshape(1, -1))[0])
                    except Exception:
                        prob = None

        prob = _safe_float(prob, None)
        if prob is None:
            prob = 0.0
        prob = float(max(0.0, min(1.0, prob)))

        return {
            "peak_prob": prob,
            "peak_prob_threshold": _safe_float(payload.get("threshold"), None),
        }

    def predict_quality(self, row: Dict[str, Any], market_trend: int) -> Dict[str, Any]:
        regime = int(market_trend) if market_trend in (-1, 0, 1) else 0
        payload = self.quality_models[regime]
        prob = _predict_payload_prob(payload, row)
        return {
            "quality_prob": prob,
            "quality_prob_threshold": _safe_float(payload.get("threshold"), None),
        }
