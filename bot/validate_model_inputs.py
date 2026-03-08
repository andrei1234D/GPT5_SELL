#!/usr/bin/env python3
"""
validate_model_inputs.py

Validates that all required feature columns exist and are finite
for each model family (MT / Peak / Quality) per ticker.

Output: bot/LLM_data/input_llm/model_input_validation.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model_registry import ModelRegistry

INPUT_CSV = "bot/LLM_data/input_llm/llm_input_latest.csv"
OUTPUT_JSON = "bot/LLM_data/input_llm/model_input_validation.json"


def _is_bad(v) -> bool:
    try:
        if v is None:
            return True
        x = float(v)
        return not np.isfinite(x)
    except Exception:
        return True


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    reg = ModelRegistry()

    issues = []
    summary = {
        "total_rows": int(len(df)),
        "models": {
            "mt": {"rows": 0, "issues": 0},
            "peak": {"rows": 0, "issues": 0},
            "quality": {"rows": 0, "issues": 0},
        },
    }

    for _, r in df.iterrows():
        row = r.to_dict()
        ticker = str(row.get("Ticker", "")).strip()
        mt = int(row.get("MarketTrend", 0) or 0)
        if mt not in (-1, 0, 1):
            mt = 0

        for family in ("mt", "peak", "quality"):
            cols = reg.get_feature_columns(family, mt)
            summary["models"][family]["rows"] += 1
            missing = [c for c in cols if c not in row]
            nan_cols = [c for c in cols if c in row and _is_bad(row.get(c))]
            if missing or nan_cols:
                summary["models"][family]["issues"] += 1
                issues.append({
                    "ticker": ticker,
                    "market_trend": mt,
                    "model": family,
                    "missing_cols": missing,
                    "nan_cols": nan_cols,
                })

    status = "PASS" if len(issues) == 0 else "FAIL"
    out = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "issues": issues,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Wrote validation report -> {OUTPUT_JSON} ({status})")


if __name__ == "__main__":
    main()
