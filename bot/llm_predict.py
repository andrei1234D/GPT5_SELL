"""
llm_predict.py

MT (MarketTrend) "brain" for production. Loads your 3 best models:
  - best_model_MT-1.joblib (bear)
  - best_model_MT0.joblib  (neutral)
  - best_model_MT1.joblib  (bull)

Sidecar metadata JSONs:
  - best_model_MT-1.json / best_model_MT0.json / best_model_MT1.json

Training-consistent semantics:
  - The regressor outputs pred_sellscore (continuous).
  - Training fits a LogisticRegression calibrator on pred_sellscore to estimate:
        P(good_sell) = P(SellScore >= sell_threshold | pred_sellscore)
  - Production is gated by:
        mt_sell_signal := (P(good_sell) >= prob_threshold)

This file therefore:
  - uses payload["calibrator"].predict_proba([[pred_sellscore]]) when available
  - falls back to a simple logistic around sell_threshold ONLY if calibrator is missing/unusable

Directory resolution (joblibs):
  1) explicit `model_dir` argument
  2) env var `SELL_MODEL_DIR`
  3) common container path `/brain`
  4) local fallback `Brain`
  5) local fallback `../Brain`

Directory resolution (JSON sidecars):
  1) env var `SELL_META_DIR` (optional)
  2) repo-root Brain derived from this file location: <repo>/Brain
  3) `Brain` (cwd-relative)
  4) `../Brain` (cwd-relative)
  5) model_dir (last resort)

Primary API used by decision_engine.py:
  SellBrain.predict(indicators: dict, market_trend: int) -> dict
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

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


def _gate_from_prob(prob: Optional[float], prob_thr: Optional[float]) -> float:
    """
    Soft gate in [0, 1]:
      - prob <= thr => 0
      - prob >  thr => linearly scaled to 0..1
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
    """
    Loads 3 regime-specific models and exposes a unified prediction interface.

    Output fields:
      - pred_sellscore: raw regressor output
      - mt_prob: calibrated probability of "good_sell"
      - mt_prob_threshold: gating threshold (prob_threshold)
      - mt_sell_signal: True iff mt_prob >= mt_prob_threshold
      - mt_gate: soft gate strength in [0,1], scaled above threshold
      - mt_weight: static weight (used by decision_engine if it blends multiple brains)
      - prob_source: "calibrator" or "fallback"
    """

    def __init__(self, model_dir: Optional[str | Path] = None):
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

        # Repo-local Brain (derived from file location)
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

        # Decision-engine blending weights (kept as-is; may be overridden upstream)
        self._weight_by_regime = {-1: 0.40, 0: 0.20, 1: 0.30}
        self._default_prob_thr = 0.65

        # If MT0 is missing sell_signal, synthesize it (now actually reachable)
        self._ensure_mt0_sell_signal()

        # Final validation: ensure all regimes have sell_threshold
        self._validate_sell_signal(-1)
        self._validate_sell_signal(0)
        self._validate_sell_signal(1)

        # CI visibility
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

        meta.setdefault("regime", int(regime))
        if isinstance(payload, dict):
            meta.setdefault("model_type", payload.get("model_type"))

        # NOTE: do not hard-fail on missing sell_signal here; MT0 can be synthesized.
        self._models[regime] = payload
        self._meta[regime] = meta

    def _validate_sell_signal(self, regime: int):
        meta = self._meta.get(regime) or {}
        ss = meta.get("sell_signal")
        if not isinstance(ss, dict):
            raise ValueError(f"Missing sell_signal dict for regime={regime}")
        if _safe_float(ss.get("sell_threshold")) is None:
            raise ValueError(f"Missing sell_signal.sell_threshold for regime={regime}")
        # prob_threshold can default; still validate numeric if present
        pt = ss.get("prob_threshold")
        if pt is not None and _safe_float(pt) is None:
            raise ValueError(f"Invalid sell_signal.prob_threshold for regime={regime}: {pt!r}")

    def _ensure_mt0_sell_signal(self):
        m0 = self._meta.get(0, {})
        ss0 = m0.get("sell_signal")
        if isinstance(ss0, dict) and _safe_float(ss0.get("sell_threshold")) is not None:
            return

        m_neg = self._meta.get(-1, {})
        m_pos = self._meta.get(1, {})

        thr_neg = _safe_float((m_neg.get("sell_signal") or {}).get("sell_threshold"))
        thr_pos = _safe_float((m_pos.get("sell_signal") or {}).get("sell_threshold"))

        if thr_neg is None or thr_pos is None:
            # MT0 cannot be synthesized safely; leave missing so validation raises clearly.
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

    # ---------- model prediction ----------
    def _predict_sellscore(self, payload: Any, X_row: np.ndarray) -> float:
        """
        Supports:
          - dict payload saved by your trainer (XGB or ET pipeline)
          - raw sklearn estimator
        """
        if not isinstance(payload, dict):
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

    def _calibrated_prob(self, payload: Any, pred_sellscore: float, sell_threshold: float) -> tuple[float, str]:
        """
        Returns (prob, source):
          - source="calibrator" if LogisticRegression calibrator succeeded
          - source="fallback" otherwise
        """
        p = float(pred_sellscore)

        if isinstance(payload, dict) and payload.get("calibrator") is not None:
            cal = payload["calibrator"]
            try:
                prob = float(cal.predict_proba(np.array([[p]], dtype=float))[0, 1])
                return _clip01(prob), "calibrator"
            except Exception:
                pass

        # Fallback heuristic: logistic around sell_threshold (not identical to training calibrator).
        k = 12.0
        x = p - float(sell_threshold)
        prob = 1.0 / (1.0 + np.exp(-k * x))
        return _clip01(float(prob)), "fallback"

    # ---------- public API ----------
    def predict(self, indicators: Dict[str, Any], market_trend: int = 0) -> Dict[str, Any]:
        regime = int(market_trend)
        if regime not in (-1, 0, 1):
            regime = 0

        payload = self._models[regime]
        meta = self._meta[regime]

        feats = meta.get("feature_columns", [])
        if not isinstance(feats, list) or not feats:
            return {
                "regime": int(regime),
                "model_type": str(meta.get("model_type") or (payload.get("model_type") if isinstance(payload, dict) else "unknown")),
                "mt_prob": None,
                "mt_prob_threshold": float(self._default_prob_thr),
                "mt_sell_signal": False,
                "mt_gate": 0.0,
                "mt_weight": float(self._weight_by_regime.get(regime, 0.20)),
                "pred_sellscore": None,
                "sell_threshold": None,
                "prob_source": None,
            }

        row = np.array([_safe_float(indicators.get(c), np.nan) for c in feats], dtype=float)
        pred_sellscore = self._predict_sellscore(payload, row)

        sell_signal = meta["sell_signal"]
        sell_thr = float(sell_signal["sell_threshold"])
        prob_thr = float(_safe_float(sell_signal.get("prob_threshold"), self._default_prob_thr))

        prob, prob_source = self._calibrated_prob(payload, pred_sellscore=pred_sellscore, sell_threshold=sell_thr)
        sig = bool(prob >= prob_thr)
        gate = _gate_from_prob(prob, prob_thr)

        return {
            "regime": int(regime),
            "model_type": str(meta.get("model_type") or (payload.get("model_type") if isinstance(payload, dict) else "unknown")),
            "mt_prob": float(prob),
            "mt_prob_threshold": float(prob_thr),
            "mt_sell_signal": sig,
            "mt_gate": float(gate),
            "mt_weight": float(self._weight_by_regime.get(regime, 0.20)),
            "pred_sellscore": float(pred_sellscore),
            "sell_threshold": float(sell_thr),
            "prob_source": prob_source,
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

        pred = brain.predict(row_dict, market_trend=mt)

        out = {
            "Timestamp": row_dict.get("Timestamp"),
            "Ticker": row_dict.get("Ticker"),
            "MarketTrend": mt,
            "mt_prob": pred.get("mt_prob"),
            "mt_prob_threshold": pred.get("mt_prob_threshold"),
            "mt_sell_signal": pred.get("mt_sell_signal"),
            "mt_gate": pred.get("mt_gate"),
            "mt_weight": pred.get("mt_weight"),
            "pred_sellscore": pred.get("pred_sellscore"),
            "sell_threshold": pred.get("sell_threshold"),
            "model_type": pred.get("model_type"),
            "prob_source": pred.get("prob_source"),
        }

        # passthrough useful columns if present
        for k in ("current_price", "pnl_pct", "avg_price", "shares", "invested_lei"):
            if k in row_dict:
                out[k] = row_dict.get(k)

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    preferred = [
        "Timestamp", "Ticker", "MarketTrend",
        "current_price", "pnl_pct",
        "mt_prob", "mt_prob_threshold", "mt_sell_signal", "mt_gate", "mt_weight",
        "pred_sellscore", "sell_threshold", "model_type", "prob_source",
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
    print(f"[OK] Wrote MT predictions -> {out}")


if __name__ == "__main__":
    main()
