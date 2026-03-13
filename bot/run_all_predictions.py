#!/usr/bin/env python3
"""
run_all_predictions.py

Reads live features CSV and produces combined predictions for:
  - goodsell ML
  - Peak ML
  - Quality ML

Outputs:
  - bot/LLM_data/input_llm/llm_predictions_all.csv
  - bot/LLM_data/input_llm/llm_predictions.csv (legacy compatibility)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model_registry import ModelRegistry

INPUT_CSV = "bot/LLM_data/input_llm/llm_input_latest.csv"
OUTPUT_CSV = "bot/LLM_data/input_llm/llm_predictions_all.csv"
LEGACY_OUTPUT_CSV = "bot/LLM_data/input_llm/llm_predictions.csv"


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


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    reg = ModelRegistry()

    out_rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        mt = int(_safe_float(row.get("MarketTrend"), 0) or 0)
        if mt not in (-1, 0, 1):
            mt = 0

        mt_pred = reg.predict_mt(row, market_trend=mt)
        peak_pred = reg.predict_peak(row, market_trend=mt)
        quality_pred = reg.predict_quality(row, market_trend=mt)

        out_rows.append({
            "Timestamp": row.get("Timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Ticker": row.get("Ticker"),
            "MarketTrend": mt,
            "current_price": row.get("current_price"),
            "pnl_pct": row.get("pnl_pct"),
            "mt_prob": mt_pred.get("mt_prob"),
            "mt_prob_threshold": mt_pred.get("mt_prob_threshold"),
            "mt_sell_signal": mt_pred.get("mt_sell_signal"),
            "mt_gate": mt_pred.get("mt_gate"),
            "mt_weight": mt_pred.get("mt_weight"),
            "pred_sellscore": mt_pred.get("pred_sellscore"),
            "sell_threshold": mt_pred.get("sell_threshold"),
            "model_type": mt_pred.get("model_type"),
            "prob_source": mt_pred.get("prob_source"),
            "goodsell_prob": mt_pred.get("mt_prob"),
            "goodsell_prob_threshold": mt_pred.get("mt_prob_threshold"),
            "peak_prob": peak_pred.get("peak_prob"),
            "peak_prob_threshold": peak_pred.get("peak_prob_threshold"),
            "quality_prob": quality_pred.get("quality_prob"),
            "quality_prob_threshold": quality_pred.get("quality_prob_threshold"),
        })

    out_df = pd.DataFrame(out_rows)
    preferred = [
        "Timestamp", "Ticker", "MarketTrend",
        "current_price", "pnl_pct",
        "mt_prob", "mt_prob_threshold", "mt_sell_signal", "mt_gate", "mt_weight",
        "pred_sellscore", "sell_threshold", "model_type", "prob_source",
        "goodsell_prob", "goodsell_prob_threshold",
        "peak_prob", "peak_prob_threshold",
        "quality_prob", "quality_prob_threshold",
    ]
    cols = [c for c in preferred if c in out_df.columns] + [c for c in out_df.columns if c not in preferred]
    out_df = out_df[cols]
    out_df.to_csv(OUTPUT_CSV, index=False)
    out_df.to_csv(LEGACY_OUTPUT_CSV, index=False)
    print(f"[OK] Wrote combined predictions -> {OUTPUT_CSV}")
    print(f"[OK] Wrote legacy predictions -> {LEGACY_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
