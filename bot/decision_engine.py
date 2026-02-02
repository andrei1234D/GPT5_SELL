#!/usr/bin/env python3
"""
decision_engine.py (v9.1 fixed)

Key fixes vs prior iterations:
- Uses NYSE trading-day key (America/New_York) for:
    - daily reset
    - rolling_sell_index update (once per market-day)
    - weak_days update (once per market-day)
  This prevents UTC-midnight from splitting one US market day.
- Aligns strictly with llm_predict.SellBrain.predict() output keys:
    mt_prob, mt_prob_threshold, mt_sell_signal, mt_gate, mt_weight,
    pred_sellscore, sell_threshold, model_type, prob_source
- Stores consistent tracker keys and also keeps a few legacy aliases (same values)
  so older UI readers do not display contradictory/stale fields.
- ML contribution uses your requested ramp:
    at p == thr_used -> contributes ML_RAMP_START (default 0.01)
    at p == 1.0      -> contributes ml_cap_used
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators
from llm_predict import SellBrain, run_batch_predictions
from knobs import (
    TRACKER_FILE,
    PROFIT_ADJ,
    STRONG_SELL_MULT,
    FX_TTL_MINUTES,
    fx_pair_to_ron,
    SPX_TICKER,
    SPX_DAILY_DOWN_PCT,
    SPX_WEEKLY_DOWN_PCT,
    SPX_DAILY_UP_PCT,
    SPX_WEEKLY_UP_PCT,
    MARKET_BIAS_FEAR,
    MARKET_BIAS_GREED,
)

LIVE_RESULTS_CSV = "bot/live_results.csv"
LLM_INPUT_CSV = "bot/LLM_data/input_llm/llm_input_latest.csv"
LLM_PRED_CSV = "bot/LLM_data/input_llm/llm_predictions.csv"

# ---------------------------
# V9 regime params (loaded from agent best_params.json when available)
# ---------------------------

RULE_SCALE_BY_MT = {
    -1: 8.0,
     0: 10.0,
     1: 12.0,
}

_FALLBACK_PARAMS = {
    -1: {
        "base_early": 0.6154248696067058,
        "det_cap": 0.5939142383728888,
        "ml_cap": 0.6994237853601922,
        "ml_prob_thr": 0.7150544609388172,
        "weak_frac": 0.7655782785928555,
        "weak_req": 1,
    },
     0: {
        "base_early": 0.608935633403437,
        "det_cap": 0.4134072394701689,
        "ml_cap": 0.7253594424175182,
        "ml_prob_thr": 0.8311084111657097,
        "weak_frac": 0.5724806156424604,
        "weak_req": 5,
    },
     1: {
        "base_early": 0.5524324445578574,
        "det_cap": 0.8230224748887198,
        "ml_cap": 0.7376868954204823,
        "ml_prob_thr": 0.7126578649111824,
        "weak_frac": 0.8267670621985246,
        "weak_req": 5,
    },
}

SELL_INDEX_ROLL_N = 7

ML_RAMP_START = float(os.getenv("ML_RAMP_START", "0.01") or 0.01)
RAMP_DEBUG = (os.getenv("RAMP_DEBUG", "").strip() == "1")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


def _today_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _market_day_key(now_utc: Optional[datetime] = None) -> str:
    """
    Returns YYYY-MM-DD in America/New_York so 'once per day' aligns with US market day.
    """
    now_utc = now_utc or datetime.utcnow()
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        ny = ZoneInfo("America/New_York")
        dt_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ny)
        return dt_local.strftime("%Y-%m-%d")
    except Exception:
        # Safe fallback (UTC)
        return now_utc.strftime("%Y-%m-%d")


def _load_agent_best_params() -> dict:
    """Load best_params.json relative to this script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    env_path = (os.getenv("AGENT_BEST_PARAMS_PATH") or os.getenv("BEST_PARAMS_PATH") or "").strip()
    env_candidate = None
    if env_path:
        env_candidate = env_path if os.path.isabs(env_path) else os.path.join(script_dir, env_path)

    candidates = [
        env_candidate,
        os.path.join(script_dir, "best_params.json"),
        os.path.join(os.path.dirname(script_dir), "best_params.json"),
    ]

    src = None
    raw = None
    tried = []
    for p in candidates:
        if not p:
            continue
        tried.append(p)
        try:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                src = p
                break
        except Exception:
            raw = None

    if not isinstance(raw, dict):
        if env_candidate is not None:
            print(f"⚠️ Agent best_params not found/invalid at override path: {env_candidate}")
        print("⚠️ Agent best_params not found/invalid in expected locations. Using built-in fallback params.")
        if tried:
            print("    Searched:")
            for t in tried[:10]:
                print(f"      - {t}")
        return {k: dict(v) for k, v in _FALLBACK_PARAMS.items()}

    out = {}
    for mt_key in ("-1", "0", "1"):
        v = raw.get(mt_key)
        if not isinstance(v, dict):
            continue
        try:
            mt = int(mt_key)
        except Exception:
            continue
        if mt not in (-1, 0, 1):
            continue
        try:
            out[mt] = {
                "base_early": float(v["base_early"]),
                "det_cap": float(v["det_cap"]),
                "ml_cap": float(v["ml_cap"]),
                "ml_prob_thr": float(v["ml_prob_thr"]),
                "weak_frac": float(v["weak_frac"]),
                "weak_req": int(v["weak_req"]),
            }
        except Exception:
            continue

    if len(out) != 3:
        print(f"⚠️ Agent best_params partially invalid (src={src}). Using built-in fallback params.")
        return {k: dict(v) for k, v in _FALLBACK_PARAMS.items()}

    print(f"✅ Loaded agent best_params.json from: {src}")
    return out


def compute_sell_threshold(base_early: float, pnl_pct: Optional[float]) -> float:
    adj = 0.0
    if pnl_pct is not None and np.isfinite(float(pnl_pct)):
        p = float(pnl_pct)
        for cut, a in PROFIT_ADJ:
            if p >= cut:
                adj = a
                break
    return float(_clamp(float(base_early) + float(adj), 0.45, 0.90))


def _get_spx_daily_weekly_returns() -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (daily_pct, weekly_pct) for S&P 500 using recent closes.
    daily_pct: last close vs previous close
    weekly_pct: last close vs close 5 trading days ago
    """
    try:
        hist = yf.Ticker(SPX_TICKER).history(period="10d", interval="1d")
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None, None
        closes = [float(x) for x in hist["Close"].dropna().tolist()]
        if len(closes) < 2:
            return None, None
        last = closes[-1]
        prev = closes[-2]
        daily_pct = ((last / prev) - 1.0) * 100.0 if prev > 0 else None
        weekly_pct = None
        if len(closes) >= 6:
            wk_prev = closes[-6]
            weekly_pct = ((last / wk_prev) - 1.0) * 100.0 if wk_prev > 0 else None
        return daily_pct, weekly_pct
    except Exception:
        return None, None


def _market_bias_from_spx(daily_pct: Optional[float], weekly_pct: Optional[float]) -> float:
    """
    Compound market bias:
      - If BOTH daily and weekly are down beyond thresholds -> +MARKET_BIAS_FEAR
      - If BOTH daily and weekly are up beyond thresholds   -> +MARKET_BIAS_GREED (negative)
      - Else -> 0.0
    """
    if daily_pct is None or weekly_pct is None:
        return 0.0
    if daily_pct <= float(SPX_DAILY_DOWN_PCT) and weekly_pct <= float(SPX_WEEKLY_DOWN_PCT):
        return float(MARKET_BIAS_FEAR)
    if daily_pct >= float(SPX_DAILY_UP_PCT) and weekly_pct >= float(SPX_WEEKLY_UP_PCT):
        return float(MARKET_BIAS_GREED)
    return 0.0


def compute_strong_threshold(early_thr: float) -> float:
    strong = float(early_thr) * float(STRONG_SELL_MULT)
    strong = max(strong, float(early_thr) + 0.05)
    return float(_clamp(strong, 0.50, 0.95))


# ---------------------------
# FX helpers (cached)
# ---------------------------
_FX_CACHE = {}  # currency -> (rate, ts_utc)
_TICKER_CCY_CACHE = {}


def get_ticker_currency(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if t in _TICKER_CCY_CACHE:
        return _TICKER_CCY_CACHE[t]
    ccy = "USD"
    try:
        yt = yf.Ticker(t)
        fi = getattr(yt, "fast_info", None) or {}
        ccy = fi.get("currency") or ccy
        if not ccy:
            info = getattr(yt, "info", {}) or {}
            ccy = info.get("currency") or ccy
    except Exception:
        pass
    ccy = (ccy or "USD").upper().strip()
    _TICKER_CCY_CACHE[t] = ccy
    return ccy


def get_fx_to_ron(currency: str, *, fallback: Optional[float] = None, ttl_minutes: int = FX_TTL_MINUTES) -> float:
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return 1.0

    now = datetime.utcnow()
    if c in _FX_CACHE:
        rate, ts = _FX_CACHE[c]
        if isinstance(ts, datetime) and (now - ts) <= timedelta(minutes=ttl_minutes) and float(rate) > 0:
            return float(rate)

    pair = fx_pair_to_ron(c)
    if pair is None:
        # Unknown currency mapping → use fallback if provided
        if fallback is not None and float(fallback) > 0:
            _FX_CACHE[c] = (float(fallback), now)
            return float(fallback)
        _FX_CACHE[c] = (1.0, now)
        return 1.0

    try:
        fx = yf.Ticker(pair).history(period="1d")
        if fx is not None and (not fx.empty) and ("Close" in fx.columns):
            rate = float(fx["Close"].iloc[-1])
            if np.isfinite(rate) and rate > 0:
                _FX_CACHE[c] = (rate, now)
                return float(rate)
    except Exception as e:
        print(f"⚠️ FX fetch failed for {c} ({pair}): {e}")

    # If live fails, prefer explicit fallback (tracker/data.json)
    if fallback is not None and float(fallback) > 0:
        _FX_CACHE[c] = (float(fallback), now)
        return float(fallback)

    # Last resort (keep your old behavior)
    last_resort = 4.6 if c == "USD" else 1.0
    _FX_CACHE[c] = (last_resort, now)
    return float(last_resort)

    now = datetime.utcnow()
    if c in _FX_CACHE:
        rate, ts = _FX_CACHE[c]
        if isinstance(ts, datetime) and (now - ts) <= timedelta(minutes=ttl_minutes) and float(rate) > 0:
            return float(rate)

    pair = _fx_pair_to_ron(c)
    if pair is None:
        _FX_CACHE[c] = (1.0, now)
        return 1.0

    try:
        fx = yf.Ticker(pair).history(period="1d")
        if fx is not None and (not fx.empty) and ("Close" in fx.columns):
            rate = float(fx["Close"].iloc[-1])
            if np.isfinite(rate) and rate > 0:
                _FX_CACHE[c] = (rate, now)
                return float(rate)
    except Exception as e:
        print(f"⚠️ FX fetch failed for {c} ({pair}): {e}")

    fallback = 4.6 if c == "USD" else 1.0
    _FX_CACHE[c] = (fallback, now)
    return float(fallback)


# ---------------------------
# Tracker helpers
# ---------------------------
def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if "tickers" not in data:
                    data["tickers"] = {}
                if "had_alerts" not in data:
                    data["had_alerts"] = False
                if "date" not in data:
                    data["date"] = _market_day_key()
                return data
            except json.JSONDecodeError:
                pass
    return {"date": _market_day_key(), "had_alerts": False, "tickers": {}}


def save_tracker(data):
    d = os.path.dirname(TRACKER_FILE)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# Git commit (truthful)
# ---------------------------
def _run(cmd: list) -> int:
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[GIT][FAIL] {' '.join(cmd)}\n  stdout={r.stdout[-500:]}\n  stderr={r.stderr[-500:]}")
        return int(r.returncode)
    except Exception as e:
        print(f"[GIT][EXC] {' '.join(cmd)} -> {e!r}")
        return 1


def git_commit_tracker():
    print("📝 Committing updated tracker, results, and MT datasets...")
    _run(["git", "config", "--global", "user.email", "bot@github.com"])
    _run(["git", "config", "--global", "user.name", "AutoBot"])

    files_to_commit = [TRACKER_FILE, LIVE_RESULTS_CSV, LLM_INPUT_CSV, LLM_PRED_CSV]
    for file_path in files_to_commit:
        if os.path.exists(file_path):
            _run(["git", "add", file_path])
        else:
            print(f"⚠️ Skipping missing file: {file_path}")

    commit_msg = f"Auto-update tracker + MT data [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
    _run(["git", "commit", "-m", commit_msg])
    _run(["git", "pull", "--rebase"])
    push_rc = _run(["git", "push"])

    if push_rc == 0:
        print("✅ Tracker and MT data committed successfully.")
    else:
        print("⚠️ Git push failed (see logs above).")


# ---------------------------
# Deterministic scoring
# ---------------------------
def check_sell_conditions(
    ticker: str,
    buy_price: float,
    current_price: float,
    pnl_pct=None,
    volume=None,
    momentum=None,
    rsi=None,
    ma50=None,
    ma200=None,
    atr=None,
    macd=None,
    macd_signal=None,
    resistance=None,
    support=None,
    info=None,
    debug: bool = True,
):
    if info is None:
        info = {}

    info.setdefault("weak_streak", 0.0)
    info.setdefault("recent_peak", current_price)
    info.setdefault("rolling_scores", [])
    info.setdefault("last_decay_date", None)
    info.setdefault("was_above_47", False)

    reasons = []

    if pnl_pct is not None and float(pnl_pct) <= -25:
        if rsi is not None and float(rsi) < 35:
            return False, f"Oversold RSI={float(rsi):.1f} → HOLD", current_price, 0.0, 0.0, ["HardStop: oversold_exception"]
        if momentum is not None and float(momentum) >= 0:
            return False, "Momentum stabilizing → HOLD", current_price, 0.0, 0.0, ["HardStop: momentum_exception"]
        return True, "HARD STOP LOSS (-25%)", current_price, 0.0, 10.0, ["HardStop: triggered"]

    score = 0.0

    if momentum is not None:
        m = float(momentum)
        if m < -0.8:
            score += 2.0
            reasons.append("Momentum collapse")
        elif m < -0.3:
            score += 1.0
            reasons.append("Weak momentum")

    if rsi is not None:
        r = float(rsi)
        if r < 35:
            score += 1.5
            reasons.append("RSI oversold")
        elif r < 45:
            score += 1.0
            reasons.append("RSI weak")
        elif r > 70:
            score += 0.5
            reasons.append("RSI overbought")

    if macd is not None and macd_signal is not None:
        try:
            if float(macd) < float(macd_signal):
                score += 1.5
                reasons.append("MACD bearish")
        except Exception:
            pass

    if ma50 is not None:
        try:
            if float(current_price) < float(ma50):
                score += 1.0
                reasons.append("Below MA50")
        except Exception:
            pass

    if ma200 is not None:
        try:
            if float(current_price) < float(ma200):
                score += 2.0
                reasons.append("Below MA200")
        except Exception:
            pass

    if support is not None:
        try:
            if float(current_price) < float(support):
                score += 2.0
                reasons.append("Support broken")
        except Exception:
            pass

    if volume is not None:
        try:
            if float(volume) > 1.3:
                score += 1.0
                reasons.append("High rel volume")
        except Exception:
            pass

    if atr is not None and pnl_pct is not None:
        try:
            if float(atr) > 7 and float(pnl_pct) < 0:
                score += 0.5
                reasons.append("High ATR + loss")
        except Exception:
            pass

    rolling = info.get("rolling_scores", []) or []
    rolling.append(float(score))
    if len(rolling) > 7:
        rolling.pop(0)
    info["rolling_scores"] = rolling
    avg_score = (sum(rolling) / len(rolling)) if rolling else float(score)

    info["recent_peak"] = max(float(info.get("recent_peak", current_price)), float(current_price))

    now_utc = datetime.utcnow()
    market_hour = 13 <= now_utc.hour <= 21

    if pnl_pct is not None:
        try:
            p = float(pnl_pct)
            if p >= 47:
                info["was_above_47"] = True
            elif p < 29:
                info["was_above_47"] = False
        except Exception:
            pass

    if market_hour and info.get("last_decay_date") != now_utc.strftime("%Y-%m-%d"):
        info["weak_streak"] = max(0.0, float(info.get("weak_streak", 0.0)) - 0.5)
        info["last_decay_date"] = now_utc.strftime("%Y-%m-%d")

    quiet_market = (atr is not None and float(atr) < 3) and (volume is not None and float(volume) < 0.7)

    if score >= 6.5:
        info["weak_streak"] = float(info.get("weak_streak", 0.0)) + 2.0
    elif score >= 4.0:
        info["weak_streak"] = float(info.get("weak_streak", 0.0)) + (0.5 if quiet_market else 1.0)
    elif score >= 3.0 and quiet_market:
        info["weak_streak"] = float(info.get("weak_streak", 0.0)) + 0.5
    elif momentum is not None and rsi is not None and float(momentum) > 0.4 and float(rsi) > 50:
        info["weak_streak"] = 0.0
    elif score < 3.0 and float(info.get("weak_streak", 0.0)) > 0:
        info["weak_streak"] = float(info.get("weak_streak", 0.0)) - 0.5

    info["weak_streak"] = max(0.0, round(float(info.get("weak_streak", 0.0)), 1))

    if debug:
        print(f"⏳ {ticker}: DetWeakStreak {info['weak_streak']:.1f} — DetAvgScore={avg_score:.1f}")

    return False, "Holding steady", current_price, float(avg_score), float(score), reasons


def _context_tag(pnl):
    if pnl is None:
        return "⚪ Neutral"
    if pnl > 50:
        return "💎 Massive Gain Softening"
    if pnl > 30:
        return "💰 Big Gain Cooling Off"
    if pnl > 10:
        return "📈 Profit Losing Strength"
    if pnl > 0:
        return "🟡 Minor Gain Under Stress"
    if pnl > -5:
        return "📉 Slight Loss Control"
    return "🩸 Drawdown Risk"


def _load_llm_input_map() -> dict:
    if not os.path.exists(LLM_INPUT_CSV):
        return {}
    try:
        df = pd.read_csv(LLM_INPUT_CSV)
        if "Ticker" not in df.columns:
            return {}
        m = {}
        for _, r in df.iterrows():
            t = str(r.get("Ticker", "")).strip()
            if t:
                row = r.to_dict()
                # sanitize NaNs to None to reduce downstream surprises
                for k, v in list(row.items()):
                    if isinstance(v, float) and np.isnan(v):
                        row[k] = None
                m[t] = row
        return m
    except Exception as e:
        print(f"⚠️ Failed reading {LLM_INPUT_CSV}: {e}")
        return {}


def _ml_ramp_contrib(mt_prob: Optional[float], *, thr_used: float, cap: float, start: float) -> Tuple[float, float]:
    """
    Returns (ml_contrib, gate_norm) where gate_norm in [0,1] for p in [thr_used, 1.0].
    """
    if mt_prob is None or not np.isfinite(float(mt_prob)):
        return 0.0, 0.0
    p = float(_clamp(float(mt_prob), 0.0, 1.0))
    thr = float(_clamp(float(thr_used), 0.0, 0.999999))
    cap = max(0.0, float(cap))
    if cap <= 0.0 or p < thr:
        return 0.0, 0.0
    denom = (1.0 - thr)
    gate = (p - thr) / denom if denom > 0 else 1.0
    gate = float(_clamp(gate, 0.0, 1.0))
    start = float(_clamp(float(start), 0.0, cap))
    contrib = start + (cap - start) * gate
    contrib = float(_clamp(contrib, 0.0, cap))
    return contrib, gate


def run_decision_engine(test_mode: bool = False, end_of_day: bool = False):
    agent_params = _load_agent_best_params()

    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracker = load_tracker()

    now_utc = datetime.utcnow()
    day_key = _market_day_key(now_utc)

    if tracker.get("date") != day_key:
        print(f"🧹 Resetting daily alert flag for market-day={day_key}")
        tracker["date"] = day_key
        tracker["had_alerts"] = False

    tracked = load_data(file_to_load)
    if not tracked or "stocks" not in tracked:
        print(f"⚠️ No tracked stocks found in {file_to_load}")
        return

    llm_input_map = _load_llm_input_map()

    sell_brain = None
    try:
        sell_brain = SellBrain()
        print("🧠 MT SELL brain loaded successfully (bear/neutral/bull).")
        try:
            outp = run_batch_predictions(LLM_INPUT_CSV, LLM_PRED_CSV, model_dir=None)
            print(f"🧾 MT predictions refreshed → {outp}")
        except Exception as e:
            print(f"⚠️ Could not refresh MT predictions CSV: {e}")
    except Exception as e:
        print(f"⚠️ Could not load MT brain: {e}")
        sell_brain = None

    stocks = tracked["stocks"]
    sell_alerts = []
    live_rows = []

    spx_daily_pct, spx_weekly_pct = _get_spx_daily_weekly_returns()
    market_bias = _market_bias_from_spx(spx_daily_pct, spx_weekly_pct)

    for ticker, info in stocks.items():
        avg_price = float(info.get("avg_price", 0) or 0)
        invested_lei = float(info.get("invested_lei", 0) or 0)
        shares = float(info.get("shares", 0) or 0)
        if avg_price <= 0 or invested_lei <= 0 or shares <= 0:
            continue

        indicators = compute_indicators(ticker)
        if not indicators:
            continue

        current_price = float(indicators["current_price"])
        info_state = (tracker.get("tickers", {}) or {}).get(ticker, {}) or {}

        # Reset rolling/weak state if buy price changed materially
        last_buy = _safe_float(info_state.get("last_buy_price"), None)
        if last_buy is not None and last_buy > 0:
            if abs(float(last_buy) - float(avg_price)) / float(last_buy) > 0.02:
                info_state["rolling_sell_index"] = []
                info_state["weak_days"] = 0
                info_state["last_roll_day"] = None
                info_state["last_weak_update_day"] = None

        # Prefer stored currency from data.json (more stable than yfinance)
        ccy = (info.get("currency") or get_ticker_currency(ticker) or "USD").upper().strip()

        # Fallback chain: tracker last FX → data.json fx_rate_buy
        fx_fallback = _safe_float(info_state.get("last_fx_to_ron"), None)
        if fx_fallback is None:
            fx_fallback = _safe_float(info_state.get("last_fx_to_lei"), None)
        if fx_fallback is None:
            fx_fallback = _safe_float(info.get("fx_rate_buy"), None)

        fx_to_ron = get_fx_to_ron(ccy, fallback=fx_fallback)
        pnl_lei = current_price * shares * fx_to_ron - invested_lei

        pnl_pct = (pnl_lei / invested_lei) * 100.0 if invested_lei else 0.0

        # MarketTrend from LLM input map
        ml_row = llm_input_map.get(ticker, {}) or {}
        mt = int(_safe_float(ml_row.get("MarketTrend"), 0) or 0)
        if mt not in (-1, 0, 1):
            mt = 0

        # Reset WeakDays if regime changes
        prev_mt = info_state.get("last_mt")
        try:
            prev_mt = int(prev_mt) if prev_mt is not None else None
        except Exception:
            prev_mt = None
        if prev_mt is not None and prev_mt in (-1, 0, 1) and prev_mt != mt:
            info_state["weak_days"] = 0
            info_state["last_weak_update_day"] = None

        rp = agent_params.get(mt, agent_params.get(0, _FALLBACK_PARAMS[0]))

        # Deterministic score
        rule_sell, rule_msg, _, det_avg_score, det_score, rule_reasons = check_sell_conditions(
            ticker,
            avg_price,
            current_price,
            pnl_pct=pnl_pct,
            volume=indicators.get("volume"),
            momentum=indicators.get("momentum"),
            rsi=indicators.get("rsi"),
            ma50=indicators.get("ma50"),
            ma200=indicators.get("ma200"),
            atr=indicators.get("atr"),
            macd=indicators.get("macd"),
            macd_signal=indicators.get("macd_signal"),
            resistance=indicators.get("resistance"),
            support=indicators.get("support"),
            info=info_state,
            debug=True,
        )

        rule_scale = float(RULE_SCALE_BY_MT.get(mt, 10.0))
        rule_norm = min(1.0, float(det_avg_score) / rule_scale) if rule_scale > 0 else 0.0
        det_contrib = min(float(rp["det_cap"]), float(rule_norm))

        # MT brain
        mt_pred = None
        if sell_brain and ml_row:
            try:
                mt_pred = sell_brain.predict(ml_row, market_trend=mt)
            except Exception as e:
                print(f"⚠️ MT prediction failed for {ticker}: {e}")
                mt_pred = None
        elif sell_brain and not ml_row:
            print(f"⚠️ {ticker}: missing ML input row in {LLM_INPUT_CSV} (cannot run MT model).")

        mt_prob = None if not mt_pred else mt_pred.get("mt_prob")
        mt_prob_thr_model = None if not mt_pred else mt_pred.get("mt_prob_threshold")
        mt_gate_model = 0.0 if not mt_pred else float(mt_pred.get("mt_gate") or 0.0)
        mt_weight = 0.0 if not mt_pred else float(mt_pred.get("mt_weight") or 0.0)
        pred_sellscore = None if not mt_pred else mt_pred.get("pred_sellscore")
        sell_threshold_model = None if not mt_pred else mt_pred.get("sell_threshold")
        mt_sell_signal = False if not mt_pred else bool(mt_pred.get("mt_sell_signal") or False)
        model_type = None if not mt_pred else mt_pred.get("model_type")
        prob_source = None if not mt_pred else mt_pred.get("prob_source")

        # ML ramp uses agent params (thr_used/cap_used)
        ml_prob_thr_used = float(rp["ml_prob_thr"])
        ml_cap_used = float(rp["ml_cap"])

        ml_contrib, ml_gate_used = _ml_ramp_contrib(
            mt_prob,
            thr_used=ml_prob_thr_used,
            cap=ml_cap_used,
            start=ML_RAMP_START,
        )

        if RAMP_DEBUG and mt_prob is not None:
            print(
                f"[RAMP_DEBUG] {ticker} mt={mt} "
                f"p={float(mt_prob):.6f} thr_used={ml_prob_thr_used:.6f} "
                f"gate_norm={ml_gate_used:.6f} cap={ml_cap_used:.6f} ml_contrib={ml_contrib:.6f}"
            )

        sell_index_raw = float(_clamp(det_contrib + ml_contrib, 0.0, 1.0))

        # Rolling avg sell index (every engine run / check)
        roll = info_state.get("rolling_sell_index", []) or []

        roll.append(float(sell_index_raw))

        # Keep only the last N samples (rolling window)
        n = int(SELL_INDEX_ROLL_N)
        if n > 0 and len(roll) > n:
            roll = roll[-n:]

        info_state["rolling_sell_index"] = roll
        avg_sell_index = float(sum(roll) / len(roll)) if roll else float(sell_index_raw)


        # Thresholds
        sell_thr_early = compute_sell_threshold(float(rp["base_early"]), pnl_pct)
        if market_bias != 0.0:
            sell_thr_early = float(_clamp(float(sell_thr_early) + float(market_bias), 0.45, 0.90))
        sell_thr_strong = compute_strong_threshold(sell_thr_early)

        # WeakDays (once per market-day)
        weak_req_base = int(rp["weak_req"])
        weak_frac = float(rp["weak_frac"])
        weak_thr = float(weak_frac) * float(sell_thr_early)

        # Profit-based WeakDays requirement (only lowers, never raises)
        # - PnL >= 50%  -> require 4 weak days
        # - PnL >= 100% -> require 3 weak days
        weak_req_eff = weak_req_base
        if pnl_pct is not None and np.isfinite(float(pnl_pct)):
            p = float(pnl_pct)
            if p >= 100.0:
                weak_req_eff = min(weak_req_eff, 3)
            elif p >= 50.0:
                weak_req_eff = min(weak_req_eff, 4)

        weak_days = int(_safe_float(info_state.get("weak_days"), 0) or 0)
        last_weak_update_day = info_state.get("last_weak_update_day")
        if last_weak_update_day != day_key:
            if float(avg_sell_index) >= float(weak_thr):
                weak_days += 1
            else:
                weak_days = 0
            info_state["weak_days"] = int(weak_days)
            info_state["last_weak_update_day"] = day_key

        # Profit bypass (DISABLED): always require WeakDays
        bypass_weak = False

        decision = False
        label = "HOLD"
        color_tag = "🟢"

        if bool(rule_sell):
            decision = True
            label = "HARD STOP-LOSS SELL"
            color_tag = "🔴"
        else:
            if float(avg_sell_index) >= float(sell_thr_strong):
                decision = True
                label = "STRONG SELL"
                color_tag = "🔴"
            elif float(avg_sell_index) >= float(sell_thr_early) and (weak_days >= weak_req_eff or bypass_weak):

                decision = True
                label = "SELL"
                color_tag = "🟠"

        pnl_context = _context_tag(pnl_pct)
        delta = max(0.0, float(sell_thr_early) - float(avg_sell_index))

        print(
            f"{color_tag} {ticker} | {label} | "
            f"AvgSellIndex {avg_sell_index:.2f} (raw {sell_index_raw:.2f}) / {sell_thr_early:.2f} SELL (Δ{delta:.2f}) | "
            f"WeakDays {weak_days}/{weak_req_eff} (thr {weak_thr:.2f}) | PnL {pnl_pct:+.2f}% | CCY {ccy} | MT {mt:+d}"

        )

        flags_txt = "none" if not rule_reasons else "; ".join(rule_reasons[:6]) + (" ..." if len(rule_reasons) > 6 else "")
        if mt_prob is not None:
            print(
                f"    Mix: Det +{det_contrib:.2f} (DetAvgScore {det_avg_score:.1f}/{rule_scale:.0f}, cap {float(rp['det_cap']):.2f}) | "
                f"ML +{ml_contrib:.2f} (P {float(mt_prob):.2f} thr_used {ml_prob_thr_used:.2f}, cap {float(rp['ml_cap']):.2f})"
            )
        else:
            print(
                f"    Mix: Det +{det_contrib:.2f} (DetAvgScore {det_avg_score:.1f}/{rule_scale:.0f}, cap {float(rp['det_cap']):.2f}) | "
                f"ML +0.00 (n/a)"
            )

        print(f"    RuleFlags: {flags_txt}")

        if mt_prob is not None:
            thr_model_txt = f"{float(mt_prob_thr_model):.2f}" if mt_prob_thr_model is not None else "n/a"
            print(
                f"    ML: {'SELL' if mt_sell_signal else 'HOLD'} | P {float(mt_prob):.3f} (thr_model {thr_model_txt}) | "
                f"GateUsed {ml_gate_used:.2f} | GateModel {mt_gate_model:.2f} | mt_weight {mt_weight:.2f} | "
                f"pred {('%.3f' % float(pred_sellscore)) if pred_sellscore is not None else 'n/a'} vs sell_thr {('%.3f' % float(sell_threshold_model)) if sell_threshold_model is not None else 'n/a'} | "
                f"{(model_type or 'model').strip()} | src={prob_source or 'n/a'}"
            )

        # ---------------------------
        # Persist state (authoritative keys + legacy aliases)
        # ---------------------------
        info_state["last_checked_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        info_state["last_market_day"] = day_key

        info_state["last_signal_label"] = label
        info_state["last_buy_price"] = float(avg_price)

        info_state["last_det_avg_score"] = float(det_avg_score)
        info_state["last_det_score"] = float(det_score)
        info_state["last_det_rule_scale"] = float(rule_scale)

        info_state["last_sell_index_raw"] = float(sell_index_raw)
        info_state["avg_sell_index"] = float(avg_sell_index)

        # Legacy aliases (kept equal to authoritative values)
        info_state["sell_index"] = float(sell_index_raw)
        info_state["last_sell_index"] = float(sell_index_raw)
        info_state["last_score"] = float(det_avg_score)

        info_state["last_sell_thr_early"] = float(sell_thr_early)
        info_state["last_sell_thr_strong"] = float(sell_thr_strong)

        info_state["weak_days"] = int(weak_days)
        info_state["weak_req"] = int(weak_req_eff)

        info_state["weak_frac"] = float(weak_frac)
        info_state["weak_thr"] = float(weak_thr)

        info_state["det_cap"] = float(rp["det_cap"])
        info_state["ml_cap"] = float(rp["ml_cap"])
        info_state["ml_prob_thr_used"] = float(ml_prob_thr_used)

        info_state["last_contrib_det"] = float(det_contrib)
        info_state["last_contrib_ml"] = float(ml_contrib)
        # legacy names
        info_state["last_contrib_rule"] = float(det_contrib)
        info_state["last_contrib_mt"] = float(ml_contrib)

        info_state["last_rule_flags"] = list(rule_reasons) if isinstance(rule_reasons, list) else []

        info_state["last_mt"] = int(mt)
        info_state["last_currency"] = ccy
        info_state["last_fx_to_ron"] = float(fx_to_ron)

        info_state["last_mt_prob"] = None if mt_prob is None else float(mt_prob)
        info_state["last_mt_prob_thr_model"] = None if mt_prob_thr_model is None else float(mt_prob_thr_model)
        info_state["last_mt_gate_used"] = float(ml_gate_used)     # normalized ramp gate
        info_state["last_mt_gate_model"] = float(mt_gate_model)   # model-provided gate
        info_state["last_mt_sell_signal"] = bool(mt_sell_signal)
        info_state["last_mt_weight"] = float(mt_weight)
        info_state["last_mt_pred_sellscore"] = None if pred_sellscore is None else float(pred_sellscore)
        info_state["last_mt_sell_threshold"] = None if sell_threshold_model is None else float(sell_threshold_model)
        info_state["last_mt_model_type"] = model_type
        info_state["last_mt_prob_source"] = prob_source or "model"

        # Also keep older UI fields if they exist
        info_state["last_mt_prob_thr"] = info_state["last_mt_prob_thr_model"]
        info_state["last_mt_gate"] = info_state["last_mt_gate_model"]

        # Reset WeakDays after a decision (prevents repeated triggers)
        if decision:
            tracker["had_alerts"] = True
            info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            info_state["weak_days"] = 0
            info_state["last_weak_update_day"] = day_key

                        # Build alert message (concise; show only agent-used MT threshold + gate)
            mt_line = ""
            if mt_prob is not None and np.isfinite(float(mt_prob)):
                mt_line = (
                    f"🤖 MT: P {float(mt_prob):.3f} (thr_used {ml_prob_thr_used:.2f}) | Gate {ml_gate_used:.2f}\n"
                )

            rule_note = (rule_msg or "").strip()
            rule_line = ""
            if rule_note and rule_note.lower() not in ("holding steady", "holding steady."):
                rule_line = f"🧾 Rule note: {rule_note}\n"

            sell_alerts.append(
                f"📈 **[{ticker}] {label}**\n"
                f"{pnl_context}\n"
                f"💰 **PnL:** {pnl_pct:+.2f}% ({ccy}→RON fx={fx_to_ron:.4f})\n"
                f"🧠 **AvgSellIndex:** {avg_sell_index:.2f} (raw {sell_index_raw:.2f})\n"
                f"🎯 **Thr:** early {sell_thr_early:.2f} / strong {sell_thr_strong:.2f}\n"
                f"⏳ **WeakDays:** {weak_days}/{weak_req_eff} (weak_thr {weak_thr:.2f})\n"
                f"🧩 Mix: Det +{det_contrib:.2f} | ML +{ml_contrib:.2f}\n"
                + mt_line
                + rule_line
                + f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )


        tracker.setdefault("tickers", {})[ticker] = info_state

        live_rows.append(
            {
                "Timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "MarketDay": day_key,
                "Ticker": ticker,
                "decision": bool(decision),
                "label": label,
                "SellIndex_raw": float(sell_index_raw),
                "AvgSellIndex": float(avg_sell_index),
                "SellThrEarly": float(sell_thr_early),
                "SellThrStrong": float(sell_thr_strong),
                "WeakDays": int(weak_days),
                "WeakReq": int(weak_req_eff),
                "WeakThr": float(weak_thr),
                "PnL_pct": float(pnl_pct),
                "Currency": ccy,
                "FX_to_RON": float(fx_to_ron),
                "MarketTrend": int(mt),
                "DetAvgScore": float(det_avg_score),
                "DetRuleScale": float(rule_scale),
                "det_cap": float(rp["det_cap"]),
                "ml_cap": float(rp["ml_cap"]),
                "ml_prob_thr_used": float(ml_prob_thr_used),
                "det_contrib": float(det_contrib),
                "ml_contrib": float(ml_contrib),
                "mt_prob": None if mt_prob is None else float(mt_prob),
                "mt_prob_threshold": None if mt_prob_thr_model is None else float(mt_prob_thr_model),
                "mt_gate_used": float(ml_gate_used),
                "mt_gate_model": float(mt_gate_model),
                "mt_sell_signal": bool(mt_sell_signal),
                "pred_sellscore": None if pred_sellscore is None else float(pred_sellscore),
                "sell_threshold_model": None if sell_threshold_model is None else float(sell_threshold_model),
                "model_type": model_type,
                "prob_source": prob_source,
            }
        )

    tracker["had_alerts"] = bool(tracker.get("had_alerts") or (len(sell_alerts) > 0))
    save_tracker(tracker)

    try:
        d = os.path.dirname(LIVE_RESULTS_CSV)
        if d:
            os.makedirs(d, exist_ok=True)
        pd.DataFrame(live_rows).to_csv(LIVE_RESULTS_CSV, index=False)
        print(f"🧾 Live results saved → {LIVE_RESULTS_CSV}")
    except Exception as e:
        print(f"⚠️ Failed to write {LIVE_RESULTS_CSV}: {e}")

    if not test_mode:
        git_commit_tracker()

    now_utc = datetime.utcnow()
    if sell_alerts:
        msg = "🚨 **SELL SIGNALS TRIGGERED** 🚨\n\n" + "\n\n".join(sell_alerts)
        for chunk in [msg[i : i + 1900] for i in range(0, len(msg), 1900)]:
            try:
                send_discord_alert(chunk)
            except Exception as e:
                print(f"⚠️ Discord send failed: {e}")
    elif end_of_day and not test_mode:
        try:
            send_discord_alert(
                "😎 All systems stable. No sell signals today.\n" f"🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
        except Exception as e:
            print(f"⚠️ Discord send failed: {e}")

    print(f"✅ Decision Engine Run Complete at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--endofday", action="store_true")
    args = parser.parse_args()
    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)
