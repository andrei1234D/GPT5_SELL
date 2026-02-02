#!/usr/bin/env python3
import json
import os
from datetime import datetime
from typing import Optional

from knobs import TRACKER_FILE, PNL_GAIN_LOSS_FILE


DATA_FILE = "bot/data.json"


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        return v
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(path: str, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _days_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    da = _parse_dt(a)
    db = _parse_dt(b)
    if not da or not db:
        return None
    return max(0.0, (db - da).total_seconds() / 86400.0)


def _update_summary(summary: dict, ticker: str, pnl_pct: float, cash_gain: float, days_held: Optional[float]):
    tickers = summary.setdefault("tickers", {})
    t = tickers.get(ticker) or {}

    trades = _safe_int(t.get("trades"), 0) or 0
    prev_avg_pnl = _safe_float(t.get("avg_pnl_pct"), 0.0) or 0.0
    prev_avg_days = _safe_float(t.get("avg_days_held"), 0.0) or 0.0
    prev_total_cash = _safe_float(t.get("total_cash_gain"), 0.0) or 0.0

    trades_new = trades + 1
    avg_pnl = (prev_avg_pnl * trades + pnl_pct) / trades_new
    if days_held is not None:
        avg_days = (prev_avg_days * trades + float(days_held)) / trades_new
    else:
        avg_days = prev_avg_days if trades > 0 else 0.0

    t.update(
        {
            "trades": trades_new,
            "avg_pnl_pct": float(avg_pnl),
            "avg_days_held": float(avg_days),
            "total_cash_gain": float(prev_total_cash + cash_gain),
            "last_pnl_pct": float(pnl_pct),
            "last_cash_gain": float(cash_gain),
            "last_days_held": None if days_held is None else float(days_held),
            "last_closed_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    tickers[ticker] = t
    summary["tickers"] = tickers


def main():
    data = _load_json(DATA_FILE, {"stocks": {}})
    tracked = data.get("stocks", {}) or {}
    tracked_tickers = set(t.upper().strip() for t in tracked.keys())

    tracker = _load_json(TRACKER_FILE, {"tickers": {}})
    tracker_tickers = set((tracker.get("tickers", {}) or {}).keys())

    removed = sorted(t for t in tracker_tickers if t not in tracked_tickers)
    if not removed:
        return

    summary = _load_json(PNL_GAIN_LOSS_FILE, {"tickers": {}})

    for t in removed:
        info = (tracker.get("tickers", {}) or {}).get(t, {}) or {}
        invested = _safe_float(info.get("last_invested_lei"), None)
        shares = _safe_float(info.get("last_shares"), None)
        last_price = _safe_float(info.get("last_checked_price"), None)
        fx = _safe_float(info.get("last_fx_to_ron"), 1.0) or 1.0

        if invested is None or invested <= 0 or shares is None or shares <= 0 or last_price is None:
            # Not enough data to compute a meaningful record
            tracker["tickers"].pop(t, None)
            continue

        cash_gain = (last_price * shares * fx) - invested
        pnl_pct = (cash_gain / invested) * 100.0 if invested > 0 else 0.0
        days_held = _days_between(info.get("first_seen_time"), info.get("last_seen_time"))

        _update_summary(summary, t, float(pnl_pct), float(cash_gain), days_held)
        tracker["tickers"].pop(t, None)

    _save_json(TRACKER_FILE, tracker)
    _save_json(PNL_GAIN_LOSS_FILE, summary)


if __name__ == "__main__":
    main()
