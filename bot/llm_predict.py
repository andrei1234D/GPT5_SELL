"""
llm_predict.py

MT (MarketTrend) "brain" for production. Loads your 3 best models:
  - best_model_MT-1.joblib (bear)
  - best_model_MT0.joblib  (neutral)
  - best_model_MT1.joblib  (bull)

Sidecar metadata JSONs:
  - best_model_MT-1.json / best_model_MT0.json / best_model_MT1.json

This version supports a common CI layout:
  - Joblibs are downloaded into SELL_MODEL_DIR (runner temp)
  - JSONs remain in the checked-out repo under Brain/

Model directory resolution (joblibs):
  1) explicit `model_dir` argument
  2) env var `SELL_MODEL_DIR`
  3) common container path `/brain`
  4) local fallback `Brain`
  5) local fallback `../Brain`

JSON directory resolution (sidecars):
  1) env var `SELL_META_DIR` (optional)
  2) repo-root Brain derived from this file location: <repo>/Brain
  3) `Brain` (cwd-relative)
  4) `../Brain` (cwd-relative)
  5) model_dir (last resort, if jsons were downloaded alongside joblibs)

Primary API used by decision_engine.py:
  SellBrain.predict_prob(indicators: dict, market_trend: int) -> dict

Key fix vs earlier versions:
  - Uses the trained LogisticRegression calibrator stored inside the joblib payload
    to compute P(good_sell | predicted_sellscore). This is the intended production gate.
  - Falls back to a simple logistic around sell_threshold ONLY if calibrator is missing.
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
    Fallback only (when calibrator is missing).

    Converts regressor output into a probability-like confidence score via
    a logistic around sell_threshold.

    NOTE: Your training pipeline *already* fits a calibrator (LogisticRegression)
    on the regressor outputs. When present, production should use that calibrator
    instead of this heuristic.
    """
    pred = float(pred)
    sell_threshold = float(sell_threshold)
    k = 12.0  # higher k => more step-like
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


def _looks_like_joblib_dir(p: Path) -> bool:
    """Require all three joblibs."""
    if not p.exists() or not p.is_dir():
        return False
    required = [_expected_joblib_name(r) for r in (-1, 0, 1)]
    return all((p / f).exists() for f in required)


def _looks_like_meta_dir(p: Path) -> bool:
    """Require all three JSON sidecars."""
    if not p.exists() or not p.is_dir():
        return False
    required = [_expected_json_name(r) for r in (-1, 0, 1)]
    return all((p / f).exists() for f in required)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        s = str(p).strip()
        if s in ("", "."):
            continue
        try:
            key = str(p.expanduser().resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# -------------------------
# SellBrain
# -------------------------
class SellBrain:
    def __init__(self, model_dir: Optional[str | Path] = None):
        """
        model_dir: where joblib files live.
        meta_dir: where JSON sidecars live (auto-resolved to repo Brain by default).
        """
        # -------- resolve joblib dir --------
        joblib_candidates: list[Path] = []

        if model_dir is not None:
            joblib_candidates.append(Path(model_dir))

        env_dir = os.getenv("SELL_MODEL_DIR", "").strip()
        if env_dir:
            joblib_candidates.append(Path(env_dir))

        joblib_candidates.append(Path("/brain"))
        joblib_candidates.append(Path("Brain"))
        joblib_candidates.append(Path("../Brain"))

        joblib_candidates = _dedupe_paths(joblib_candidates)

        self.model_dir: Optional[Path] = None
        for c in joblib_candidates:
            if _looks_like_joblib_dir(c):
                self.model_dir = c
                break

        if self.model_dir is None:
            raise FileNotFoundError(
                "Could not locate joblib model directory. Provide --model_dir or set SELL_MODEL_DIR. "
                "Expected joblibs: best_model_MT-1.joblib, best_model_MT0.joblib, best_model_MT1.joblib"
            )

        # -------- resolve meta dir (JSONs) --------
        meta_candidates: list[Path] = []

        env_meta = os.getenv("SELL_META_DIR", "").strip()
        if env_meta:
            meta_candidates.append(Path(env_meta))

        # Repo-local Brain (derived from file location; works in GitHub Actions)
        try:
            repo_root = Path(__file__).resolve().parents[1]  # bot/llm_predict.py -> repo root
            meta_candidates.append(repo_root / "Brain")
        except Exception:
            pass

        meta_candidates.append(Path("Brain"))
        meta_candidates.append(Path("../Brain"))
        meta_candidates.append(self.model_dir)  # last resort

        meta_candidates = _dedupe_paths(meta_candidates)

        self.meta_dir: Optional[Path] = None
        for c in meta_candidates:
            if _looks_like_meta_dir(c):
                self.meta_dir = c
                break

        if self.meta_dir is None:
            raise FileNotFoundError(
                "Could not locate JSON meta directory (sidecars). "
                "Either download JSONs alongside joblibs, set SELL_META_DIR, or ensure repo has Brain/*.json.\n"
                "Expected jsons: best_model_MT-1.json, best_model_MT0.json, best_model_MT1.json"
            )

        print(f"[SellBrain] Using model_dir (joblibs): {self.model_dir}")
        print(f"[SellBrain] Using meta_dir  (jsons):  {self.meta_dir}")

        self._models: Dict[int, Any] = {}
        self._meta: Dict[int, Dict[str, Any]] = {}

        for regime in (-1, 0, 1):
            self._load_regime(regime)

        self._weight_by_regime = {-1: 0.40, 0: 0.20, 1: 0.30}
        self._default_prob_thr = 0.65

        # If MT0 is missing sell_signal, synthesize it as mean of -1 and 1 thresholds
        self._ensure_mt0_sell_signal()

        # CI visibility: print loaded thresholds
        for r in (-1, 0, 1):
            ss = self._meta.get(r, {}).get("sell_signal") or {}
            print(
                f"[SellBrain] regime={r} sell_threshold={ss.get('sell_threshold')} "
                f"prob_threshold={ss.get('prob_threshold')}"
            )

    def _load_regime(self, regime: int):
        job = self.model_dir / _expected_joblib_name(regime)
        js = self.meta_dir / _expected_json_name(regime)

        if not job.exists():
            raise FileNotFoundError(f"Missing model file: {job}")
        if not js.exists():
            raise FileNotFoundError(f"Missing meta JSON: {js}")

        payload = load(job)

        try:
            with open(js, "r", encoding="utf-8") as f:
                meta: Dict[str, Any] = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON meta: {js} ({type(e).__name__}: {e})")

        # If meta doesn't have feature_columns, attempt to recover from payload.
        if not meta.get("feature_columns") and isinstance(payload, dict):
            feats = payload.get("feature_columns")
            if isinstance(feats, list) and feats:
                meta["feature_columns"] = feats

        # Ensure regime/model_type populated
        meta.setdefault("regime", int(regime))
        if isinstance(payload, dict):
            meta.setdefault("model_type", payload.get("model_type"))

        # Validate sell_signal presence
        ss = meta.get("sell_signal")
        if not isinstance(ss, dict) or _safe_float(ss.get("sell_threshold")) is None:
            raise ValueError(f"JSON missing sell_signal.sell_threshold for regime={regime}: {js}")

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

    def _predict_sellscore(self, payload: Any, X_row: np.ndarray) -> float:
        """
        Supports:
          - XGB payload: {"model":{"booster":..., "imputer":...}, ...}
          - ET payload: pipeline stored as payload["model"] or payload itself if saved that way
        """
        if not isinstance(payload, dict):
            return float(payload.predict(X_row.reshape(1, -1))[0])

        mtype = payload.get("model_type")
        if mtype == "XGB":
            booster = payload["model"]["booster"]
            imp = payload["model"]["imputer"]
            X_imp = imp.transform(X_row.reshape(1, -1))
            import xgboost as xgb  # lazy import
            return float(booster.predict(xgb.DMatrix(X_imp))[0])

        model = payload.get("model", payload)
        return float(model.predict(X_row.reshape(1, -1))[0])

    def _predict_prob(self, payload: Any, pred_sellscore: float) -> Optional[float]:
        """
        Primary: use trained calibrator if present (this is the intended gate).
        Fallback: heuristic logistic around sell_threshold (only when calibrator missing).
        """
        if isinstance(payload, dict):
            cal = payload.get("calibrator", None)
            if cal is not None and hasattr(cal, "predict_proba"):
                try:
                    p = cal.predict_proba(np.asarray([float(pred_sellscore)]).reshape(-1, 1))[:, 1]
                    return float(p[0])
                except Exception:
                    # fall through to heuristic
                    pass

        # If no calibrator, do not fabricate unless we have sell_threshold
        return None

    def predict_prob(self, indicators: Dict[str, Any], market_trend: int = 0) -> Dict[str, Any]:
        regime = int(market_trend)
        if regime not in (-1, 0, 1):
            regime = 0

        payload = self._models[regime]
        meta = self._meta[regime]

        feats = meta.get("feature_columns", [])
        prob_thr = float(_safe_float((meta.get("sell_signal") or {}).get("prob_threshold"), self._default_prob_thr))
        sell_thr = float(_safe_float((meta.get("sell_signal") or {}).get("sell_threshold"), 0.0))

        if not isinstance(feats, list) or not feats:
            return {
                "prob": None,
                "prob_threshold": prob_thr,
                "sell_signal": False,
                "gate": 0.0,
                "weight": float(self._weight_by_regime.get(regime, 0.20)),
                "regime": int(regime),
                "model_type": str(meta.get("model_type") or (payload.get("model_type") if isinstance(payload, dict) else "unknown")),
            }

        row = np.array([_safe_float(indicators.get(c), np.nan) for c in feats], dtype=float)
        pred_sellscore = self._predict_sellscore(payload, row)

        # Preferred: trained calibrator
        prob = self._predict_prob(payload, pred_sellscore)

        # Fallback: heuristic if calibrator missing
        if prob is None:
            prob = _infer_prob_from_regressor(pred_sellscore, sell_threshold=sell_thr)

        sell_signal = bool(prob >= prob_thr)
        gate = _gate_from_prob(prob, prob_thr)

        return {
            "prob": float(prob),
            "prob_threshold": float(prob_thr),
            "sell_signal": sell_signal,
            "gate": float(gate),
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

        out = {
            "Ticker": row_dict.get("Ticker"),
            "MarketTrend": mt,
            "mt_prob": pred.get("prob"),
            "mt_prob_threshold": pred.get("prob_threshold"),
            "mt_sell_signal": pred.get("sell_signal"),
            "mt_gate": pred.get("gate"),
            "mt_weight": pred.get("weight"),
            "pred_sellscore": pred.get("pred_sellscore"),
            "sell_threshold": pred.get("sell_threshold"),
            "model_type": pred.get("model_type"),
            "Timestamp": row_dict.get("Timestamp"),
        }

        for k in ("current_price", "pnl_pct", "avg_price", "shares", "invested_lei"):
            if k in row_dict:
                out[k] = row_dict.get(k)

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    preferred = [
        "Timestamp", "Ticker", "MarketTrend",
        "current_price", "pnl_pct",
        "mt_prob", "mt_prob_threshold", "mt_sell_signal", "mt_gate", "mt_weight",
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
