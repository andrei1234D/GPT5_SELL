import argparse
import json
import os
import subprocess
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators
from llm_predict import SellBrain, run_batch_predictions  # MT Brain integration (bear/neutral/bull)

TRACKER_FILE = "bot/sell_alerts_tracker.json"
LIVE_RESULTS_CSV = "bot/live_results.csv"
LLM_INPUT_CSV = "bot/LLM_data/input_llm/llm_input_latest.csv"
LLM_PRED_CSV = "bot/LLM_data/input_llm/llm_predictions.csv"

# Agent best params (V9) path (source-of-truth for tuned thresholds/caps).
AGENT_BEST_PARAMS_JSON = "/best_params.json"

# ---------------------------
# Dynamic sell thresholds (SellIndex is 0..1)
# ---------------------------
# Fallback baselines by MarketTrend (MT) when Agent params are not available:
_BASE_THR_BY_MT = {
    -1: (0.64, 0.74),  # bear
     0: (0.63, 0.73),  # neutral
     1: (0.61, 0.71),  # bull
}
# Profit-tier adjustments (applied to EARLY threshold).
# Negative => lower threshold => easier to sell.
_PROFIT_ADJ = [
    (50.0, -0.08),
    (30.0, -0.06),
    (10.0, -0.03),
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


# ---------------------------
# Agent best-params loader
# ---------------------------
def load_agent_best_params(path: str = AGENT_BEST_PARAMS_JSON) -> Dict[int, Dict[str, Any]]:
    """Load agent-tuned regime params from JSON. Returns dict keyed by regime int (-1/0/1)."""
    try:
        p = Path(path)
        if not p.exists():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {}

        out: Dict[int, Dict[str, Any]] = {}
        for k in ("-1", "0", "1"):
            v = obj.get(k)
            if not isinstance(v, dict):
                continue

            def _f(name: str, default_val=None):
                val = v.get(name, default_val)
                try:
                    return float(val)
                except Exception:
                    return default_val

            def _i(name: str, default_val=None):
                val = v.get(name, default_val)
                try:
                    return int(val)
                except Exception:
                    return default_val

            out[int(k)] = {
                "base_early": _f("base_early", None),
                "det_cap": _f("det_cap", None),
                "ml_cap": _f("ml_cap", None),
                "ml_prob_thr": _f("ml_prob_thr", None),
                "weak_frac": _f("weak_frac", None),
                "weak_req": _i("weak_req", None),
            }

        return out
    except Exception as e:
        print(f"⚠️ Failed to load Agent best params from {path}: {e}")
        return {}


def get_sell_thresholds(
    pnl_pct: Optional[float],
    mt: int,
    *,
    base_early_override: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns (early_thr, strong_thr) for SellIndex."""
    if base_early_override is not None and math.isfinite(float(base_early_override)):
        base_early = float(base_early_override)
    else:
        base_early, _base_strong = _BASE_THR_BY_MT.get(mt, _BASE_THR_BY_MT[0])

    adj = 0.0
    if pnl_pct is not None:
        try:
            p = float(pnl_pct)
        except Exception:
            p = float("nan")
        if math.isfinite(p):
            for cut, a in _PROFIT_ADJ:
                if p >= float(cut):
                    adj = float(a)
                    break

    early = _clamp(float(base_early) + float(adj), 0.45, 0.90)

    strong = _clamp(float(early) * float(STRONG_SELL_MULT), 0.50, 0.90)
    if strong < early + 0.05:
        strong = _clamp(early + 0.05, 0.50, 0.90)

    return float(early), float(strong)


# Deterministic normalization scale by regime.
# Bigger scale => deterministic contributes LESS to SellIndex.
RULE_SCALE_BY_MT = {
    -1: 8.0,
     0: 10.0,
     1: 12.0,
}

STRONG_SELL_MULT = 1.25
SELL_INDEX_ROLL_N = 5

# Fallback weak-day gate params (used only if Agent params missing)
WEAK_REQ_BY_MT = {-1: 2, 0: 5, 1: 5}
WEAK_FRAC_BY_MT = {-1: 0.765, 0: 0.572, 1: 0.827}


# ---------------------------
# FX helpers (multi-currency, cached)
# ---------------------------
_FX_CACHE = {}
_TICKER_CCY_CACHE = {}

def _fx_pair_to_ron(currency: str):
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return None
    if c == "USD":
        return "USDRON=X"
    if c == "EUR":
        return "EURRON=X"
    if c == "GBP":
        return "GBPRON=X"
    if c == "CHF":
        return "CHFRON=X"
    if c == "CAD":
        return "CADRON=X"
    return None


def get_ticker_currency(ticker: str) -> str:
    if ticker in _TICKER_CCY_CACHE:
        return _TICKER_CCY_CACHE[ticker]
    ccy = "USD"
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None) or {}
        ccy = fi.get("currency") or ccy
        if not ccy:
            info = getattr(t, "info", {}) or {}
            ccy = info.get("currency") or ccy
    except Exception:
        pass
    ccy = (ccy or "USD").upper().strip()
    _TICKER_CCY_CACHE[ticker] = ccy
    return ccy


def get_fx_to_ron(currency: str) -> float:
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return 1.0
    if c in _FX_CACHE:
        return _FX_CACHE[c]

    pair = _fx_pair_to_ron(c)
    if pair is None:
        _FX_CACHE[c] = 1.0
        return 1.0

    try:
        fx = yf.Ticker(pair).history(period="1d")
        if not fx.empty and "Close" in fx.columns:
            rate = float(fx["Close"].iloc[-1])
            if np.isfinite(rate) and rate > 0:
                _FX_CACHE[c] = rate
                return rate
    except Exception as e:
        print(f"⚠️ FX fetch failed for {c} ({pair}): {e}")

    fallback = 4.6 if c == "USD" else 1.0
    _FX_CACHE[c] = fallback
    return fallback


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
                return data
            except json.JSONDecodeError:
                pass
    return {"date": datetime.utcnow().strftime("%Y-%m-%d"), "had_alerts": False, "tickers": {}}


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

    files_to_commit = [
        TRACKER_FILE,
        LIVE_RESULTS_CSV,
        LLM_INPUT_CSV,
        LLM_PRED_CSV,
    ]

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
# Deterministic scoring (points-based)
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
    debug=True,
):
    if info is None:
        info = {}

    info.setdefault("weak_streak", 0.0)
    info.setdefault("recent_peak", current_price)
    info.setdefault("rolling_scores", [])
    info.setdefault("last_decay_date", None)
    info.setdefault("was_above_47", False)

    reasons = []

    # Hard stop-loss
    if pnl_pct is not None and float(pnl_pct) <= -25:
        if rsi is not None and float(rsi) < 35:
            return False, f"Oversold RSI={float(rsi):.1f} → HOLD", current_price, 0.0, 0.0, ["HardStop: oversold_exception"]
        if momentum is not None and float(momentum) >= 0:
            return False, "Momentum stabilizing → HOLD", current_price, 0.0, 0.0, ["HardStop: momentum_exception"]
        return True, "HARD STOP LOSS (-25%)", current_price, 0.0, 10.0, ["HardStop: triggered"]

    score = 0.0

    # Momentum
    if momentum is not None:
        m = float(momentum)
        if m < -0.8:
            score += 2.0
            reasons.append("Momentum collapse")
        elif m < -0.3:
            score += 1.0
            reasons.append("Weak momentum")

    # RSI
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

    # MACD
    if macd is not None and macd_signal is not None:
        try:
            if float(macd) < float(macd_signal):
                score += 1.5
                reasons.append("MACD bearish")
        except Exception:
            pass

    # MA50 / MA200
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

    # Support
    if support is not None:
        try:
            if float(current_price) < float(support):
                score += 2.0
                reasons.append("Support broken")
        except Exception:
            pass

    # Relative volume
    if volume is not None:
        try:
            if float(volume) > 1.3:
                score += 1.0
                reasons.append("High rel volume")
        except Exception:
            pass

    # ATR + loss
    if atr is not None and pnl_pct is not None:
        try:
            if float(atr) > 7 and float(pnl_pct) < 0:
                score += 0.5
                reasons.append("High ATR + loss")
        except Exception:
            pass

    # Rolling average
    rolling = info.get("rolling_scores", []) or []
    rolling.append(float(score))
    if len(rolling) > 7:
        rolling.pop(0)
    info["rolling_scores"] = rolling
    avg_score = (sum(rolling) / len(rolling)) if rolling else float(score)

    # Weak streak (kept as-is)
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
        print(f"⏳ {ticker}: Weak {info['weak_streak']:.1f}/3 — AvgScore={avg_score:.1f}")

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
                m[t] = r.to_dict()
        return m
    except Exception as e:
        print(f"⚠️ Failed reading {LLM_INPUT_CSV}: {e}")
        return {}


def _ml_line(
    *,
    mt: int,
    mt_prob,
    mt_prob_thr_used,
    mt_prob_thr_model,
    mt_gate_used: float,
    mt_gate_model: float,
    mt_weight: float,
    ml_contrib: float,
    pred_sellscore,
    sell_threshold,
    model_type,
    source: str = "unknown",
) -> str:
    if mt_prob is None:
        return f"    ML: n/a | w {mt_weight:.2f} → +{ml_contrib:.2f}"

    p = float(mt_prob)
    thr_u = float(mt_prob_thr_used) if mt_prob_thr_used is not None else float("nan")
    thr_m = float(mt_prob_thr_model) if mt_prob_thr_model is not None else float("nan")
    sig = "SELL" if (mt_prob_thr_used is not None and p >= float(mt_prob_thr_used)) else "HOLD"

    ps = f"{float(pred_sellscore):.3f}" if pred_sellscore is not None else "n/a"
    st = f"{float(sell_threshold):.3f}" if sell_threshold is not None else "n/a"
    mt_type = (model_type or "model").strip()

    return (
        f"    ML: {sig} | P {p:.3f} (thr_used {thr_u:.3f}, thr_model {thr_m:.3f}) | "
        f"Gate used {float(mt_gate_used):.2f} vs model {float(mt_gate_model):.2f} | "
        f"w {mt_weight:.2f} → +{ml_contrib:.2f} | pred {ps} vs sell_thr {st} | {mt_type} | src={source}"
    )


def run_decision_engine(test_mode=False, end_of_day=False):
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracker = load_tracker()

    # Load agent params once per run (source-of-truth).
    agent_params = load_agent_best_params(AGENT_BEST_PARAMS_JSON)
    if agent_params:
        def _fmt(mt):
            rp = agent_params.get(mt) or {}
            return f"base_early={rp.get('base_early')} det_cap={rp.get('det_cap')} ml_cap={rp.get('ml_cap')} ml_prob_thr={rp.get('ml_prob_thr')} weak_frac={rp.get('weak_frac')} weak_req={rp.get('weak_req')}"
        print(f"🧩 Agent best_params loaded from {AGENT_BEST_PARAMS_JSON}:")
        print(f"    MT-1: {_fmt(-1)}")
        print(f"    MT 0: {_fmt(0)}")
        print(f"    MT+1: {_fmt(1)}")
    else:
        print(f"⚠️ Agent best_params not found/invalid at {AGENT_BEST_PARAMS_JSON}. Using engine defaults + model-provided thresholds.")

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if tracker.get("date") != today_str:
        print(f"🧹 Resetting daily alert flag for {today_str}")
        tracker["date"] = today_str
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

    for ticker, info in stocks.items():
        avg_price = float(info.get("avg_price", 0))
        invested_lei = float(info.get("invested_lei", 0))
        shares = float(info.get("shares", 0))
        if avg_price <= 0 or invested_lei <= 0 or shares <= 0:
            continue

        indicators = compute_indicators(ticker)
        if not indicators:
            continue

        current_price = float(indicators["current_price"])
        info_state = tracker["tickers"].get(ticker, {}) or {}
        tstate = info_state  # alias: persistent per-ticker state

        ccy = get_ticker_currency(ticker)
        fx_to_ron = get_fx_to_ron(ccy)
        pnl_lei = current_price * shares * fx_to_ron - invested_lei
        pnl_pct = (pnl_lei / invested_lei) * 100.0 if invested_lei else 0.0

        rule_sell, rule_msg, _, avg_score, rule_score, rule_reasons = check_sell_conditions(
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

        weak_streak = float(info_state.get("weak_streak", 0.0))

        ml_row = llm_input_map.get(ticker, {}) or {}
        mt = int(_safe_float(ml_row.get("MarketTrend"), 0) or 0)
        if mt not in (-1, 0, 1):
            mt = 0

        rp = (agent_params.get(mt) or agent_params.get(0) or {}) if agent_params else {}

        base_early_agent = rp.get("base_early")
        det_cap_agent = rp.get("det_cap")
        ml_cap_agent = rp.get("ml_cap")
        ml_prob_thr_agent = rp.get("ml_prob_thr")
        weak_frac_agent = rp.get("weak_frac")
        weak_req_agent = rp.get("weak_req")

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
        sell_threshold = None if not mt_pred else mt_pred.get("sell_threshold")
        mt_sell_signal_model = False if not mt_pred else bool(mt_pred.get("mt_sell_signal") or False)
        model_type = None if not mt_pred else mt_pred.get("model_type")

        mt_prob_thr_used = None
        if ml_prob_thr_agent is not None and math.isfinite(float(ml_prob_thr_agent)):
            mt_prob_thr_used = float(ml_prob_thr_agent)
        elif mt_prob_thr_model is not None:
            try:
                mt_prob_thr_used = float(mt_prob_thr_model)
            except Exception:
                mt_prob_thr_used = None

        mt_gate_used = 0.0
        mt_sell_signal_used = False
        if mt_prob is not None and mt_prob_thr_used is not None:
            try:
                p = float(mt_prob)
                if math.isfinite(p) and p >= float(mt_prob_thr_used):
                    mt_gate_used = p
                    mt_sell_signal_used = True
            except Exception:
                mt_gate_used = 0.0
                mt_sell_signal_used = False

        rule_scale = float(RULE_SCALE_BY_MT.get(mt, 10.0))
        rule_norm = min(1.0, float(avg_score) / rule_scale) if rule_scale > 0 else 0.0

        w_mt = max(0.0, min(0.85, float(mt_weight or 0.0)))
        w_rule = max(0.0, 1.0 - w_mt)

        w_sum = w_mt + w_rule
        if w_sum <= 0:
            w_mt, w_rule = 0.0, 1.0
            w_sum = 1.0
        w_mt /= w_sum
        w_rule /= w_sum

        det_signal = rule_norm
        if det_cap_agent is not None and math.isfinite(float(det_cap_agent)):
            det_signal = min(det_signal, float(det_cap_agent))

        ml_signal = float(mt_gate_used or 0.0)
        if ml_cap_agent is not None and math.isfinite(float(ml_cap_agent)):
            ml_signal = min(ml_signal, float(ml_cap_agent))

        rule_contrib = w_rule * float(det_signal)
        ml_contrib = w_mt * float(ml_signal)
        sell_index = _clamp(rule_contrib + ml_contrib, 0.0, 1.0)

        mt_softened = False
        if (
            6.0 <= float(avg_score) < 7.0
            and mt_prob is not None
            and mt_prob_thr_used is not None
            and float(mt_prob) < float(mt_prob_thr_used)
        ):
            sell_index = max(0.0, sell_index - 0.15)
            mt_softened = True

        sell_thr_early, sell_thr_strong = get_sell_thresholds(
            pnl_pct,
            mt,
            base_early_override=(float(base_early_agent) if (base_early_agent is not None and math.isfinite(float(base_early_agent))) else None),
        )

        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        roll = tstate.get("rolling_sell_index")
        if not isinstance(roll, list):
            roll = []
        if tstate.get("last_sell_index_date") != today_str:
            roll.append(float(sell_index))
            roll = roll[-int(SELL_INDEX_ROLL_N):]
            tstate["rolling_sell_index"] = roll
            tstate["last_sell_index_date"] = today_str

        if roll:
            avg_sell_index = float(np.nanmean(np.array(roll, dtype=np.float32)))
        else:
            avg_sell_index = float(sell_index)

        tstate["sell_index"] = float(sell_index)
        tstate["avg_sell_index"] = float(avg_sell_index)

        weak_req = int(weak_req_agent) if isinstance(weak_req_agent, int) and weak_req_agent > 0 else int(WEAK_REQ_BY_MT.get(mt, 3))
        weak_frac = float(weak_frac_agent) if (weak_frac_agent is not None and math.isfinite(float(weak_frac_agent))) else float(WEAK_FRAC_BY_MT.get(mt, 0.70))
        weak_thr = float(weak_frac) * float(sell_thr_early)

        if tstate.get("last_weak_eval_date") != today_str:
            if math.isfinite(avg_sell_index) and float(avg_sell_index) >= float(weak_thr):
                tstate["weak_days"] = int(tstate.get("weak_days", 0)) + 1
            else:
                tstate["weak_days"] = 0
            tstate["last_weak_eval_date"] = today_str

        weak_days = int(tstate.get("weak_days", 0))
        tstate["weak_thr"] = float(weak_thr)
        tstate["weak_days"] = weak_days

        bypass_weak = False
        try:
            bypass_weak = (pnl_pct is not None and math.isfinite(float(pnl_pct)) and float(pnl_pct) >= 50.0)
        except Exception:
            bypass_weak = False

        decision = False
        label = "HOLD"
        color_tag = "🟢"

        strong_sell = math.isfinite(float(sell_index)) and float(sell_index) >= float(sell_thr_strong)

        if bool(rule_sell):
            decision = True
            label = "HARD STOP-LOSS SELL"
            color_tag = "🔴"
        else:
            if strong_sell:
                decision = True
                label = f"STRONG SELL (x{STRONG_SELL_MULT:.2f})"
                color_tag = "🔴"
            elif math.isfinite(float(avg_sell_index)) and float(avg_sell_index) >= float(sell_thr_early) and (weak_days >= weak_req or bypass_weak):
                decision = True
                label = "SELL"
                color_tag = "🟠"

        delta = max(0.0, sell_thr_early - sell_index)
        pnl_context = _context_tag(pnl_pct)

        print(
            f"{color_tag} {ticker} | {label} | "
            f"SellIndex {sell_index:.2f} / {sell_thr_early:.2f} SELL (Δ{delta:.2f}) | "
            f"Weak {weak_streak:.1f}/3 | PnL {pnl_pct:+.2f}% | CCY {ccy} | MT {mt:+d}"
        )

        print(
            f"    Mix: Rule +{rule_contrib:.2f} (AvgScore {avg_score:.1f}/{rule_scale:.0f}, w={w_rule:.2f}) | "
            f"ML +{ml_contrib:.2f} (Gate {float(mt_gate_used):.2f}, w={w_mt:.2f})"
        )

        flags_txt = "none" if not rule_reasons else "; ".join(rule_reasons[:6]) + (" ..." if len(rule_reasons) > 6 else "")
        print(f"    RuleFlags: {flags_txt}")

        ml_line = _ml_line(
            mt=mt,
            mt_prob=mt_prob,
            mt_prob_thr_used=mt_prob_thr_used,
            mt_prob_thr_model=mt_prob_thr_model,
            mt_gate_used=mt_gate_used,
            mt_gate_model=mt_gate_model,
            mt_weight=w_mt,
            ml_contrib=ml_contrib,
            pred_sellscore=pred_sellscore,
            sell_threshold=sell_threshold,
            model_type=model_type,
            source=("agent_best_params" if (ml_prob_thr_agent is not None) else "model_json"),
        )
        print(ml_line)

        if mt_softened:
            print("    Note: ML softened borderline deterministic sell.")

        now_utc = datetime.utcnow()
        info_state["last_checked_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        info_state["last_signal_label"] = label
        info_state["last_score"] = float(avg_score)
        info_state["last_sell_index"] = float(sell_index)
        info_state["last_mt"] = int(mt)
        info_state["last_currency"] = ccy
        info_state["last_fx_to_ron"] = float(fx_to_ron)

        info_state["last_sell_thr_early"] = float(sell_thr_early)
        info_state["last_sell_thr_strong"] = float(sell_thr_strong)

        info_state["last_rule_scale"] = float(rule_scale)
        info_state["last_w_rule"] = float(w_rule)
        info_state["last_w_mt"] = float(w_mt)

        info_state["last_contrib_rule"] = float(rule_contrib)
        info_state["last_contrib_ml"] = float(ml_contrib)

        info_state["last_rule_flags"] = list(rule_reasons) if isinstance(rule_reasons, list) else []

        info_state["last_mt_prob"] = None if mt_prob is None else float(mt_prob)
        info_state["last_mt_prob_thr_used"] = None if mt_prob_thr_used is None else float(mt_prob_thr_used)
        info_state["last_mt_prob_thr_model"] = None if mt_prob_thr_model is None else float(mt_prob_thr_model)
        info_state["last_mt_gate_used"] = float(mt_gate_used)
        info_state["last_mt_gate_model"] = float(mt_gate_model)
        info_state["last_mt_sell_signal_used"] = bool(mt_sell_signal_used)
        info_state["last_mt_sell_signal_model"] = bool(mt_sell_signal_model)

        info_state["last_mt_pred_sellscore"] = None if pred_sellscore is None else float(pred_sellscore)
        info_state["last_mt_sell_threshold"] = None if sell_threshold is None else float(sell_threshold)
        info_state["last_mt_model_type"] = model_type
        info_state["last_mt_prob_source"] = "agent_best_params" if (ml_prob_thr_agent is not None) else ("model" if mt_prob is not None else None)

        if decision:
            tracker["had_alerts"] = True
            info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            sell_alerts.append(
                f"📈 **[{ticker}] {label}**\n"
                f"{pnl_context}\n"
                f"💰 **PnL:** {pnl_pct:+.2f}% ({ccy}→RON fx={fx_to_ron:.4f})\n"
                f"📊 **AvgSellIndex:** {avg_sell_index:.2f} | WeakDays: {weak_days}/{weak_req} (weak_thr {weak_thr:.2f})\n"
                f"🧾 **RuleAvgScore:** {avg_score:.2f} | RuleWeakStreak: {weak_streak:.1f}\n"
                f"🧠 **SellIndex:** raw {sell_index:.2f} | avg {avg_sell_index:.2f} (early {sell_thr_early:.2f} / strong {sell_thr_strong:.2f})\n"
                f"🧪 **Contrib:** rule={rule_contrib:.2f} | ml={ml_contrib:.2f}\n"
                f"{ml_line}"
                f"{rule_msg}"
            )

            tstate["weak_days"] = 0
            tstate["rolling_sell_index"] = []
            tstate["last_sell_index_date"] = None
            tstate["last_weak_eval_date"] = None

        tracker["tickers"][ticker] = info_state

        live_rows.append({
            "Timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Ticker": ticker,
            "decision": bool(decision),
            "label": label,
            "SellIndex": float(sell_index),
            "SellThrEarly": float(sell_thr_early),
            "SellThrStrong": float(sell_thr_strong),
            "AvgSellIndex": float(avg_sell_index),
            "AvgRuleScore": float(avg_score),
            "WeakDays": int(weak_days),
            "WeakReq": int(weak_req),
            "WeakThr": float(weak_thr),
            "RuleScale": float(rule_scale),
            "WeakStreak": float(weak_streak),
            "PnL_pct": float(pnl_pct),
            "Currency": ccy,
            "FX_to_RON": float(fx_to_ron),
            "MarketTrend": int(mt),

            "mt_prob": None if mt_prob is None else float(mt_prob),
            "mt_prob_threshold": None if mt_prob_thr_used is None else float(mt_prob_thr_used),
            "mt_prob_threshold_model": None if mt_prob_thr_model is None else float(mt_prob_thr_model),
            "mt_gate": float(mt_gate_used),
            "mt_gate_model": float(mt_gate_model),
            "mt_sell_signal": bool(mt_sell_signal_used),
            "mt_sell_signal_model": bool(mt_sell_signal_model),

            "pred_sellscore": None if pred_sellscore is None else float(pred_sellscore),
            "sell_threshold": None if sell_threshold is None else float(sell_threshold),
            "model_type": model_type,

            "w_rule": float(w_rule),
            "w_mt": float(w_mt),
            "rule_contrib": float(rule_contrib),
            "ml_contrib": float(ml_contrib),

            "agent_base_early": None if base_early_agent is None else float(base_early_agent),
            "agent_det_cap": None if det_cap_agent is None else float(det_cap_agent),
            "agent_ml_cap": None if ml_cap_agent is None else float(ml_cap_agent),
            "agent_ml_prob_thr": None if ml_prob_thr_agent is None else float(ml_prob_thr_agent),
            "agent_weak_frac": None if weak_frac_agent is None else float(weak_frac_agent),
            "agent_weak_req": None if weak_req_agent is None else int(weak_req_agent),
        })

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
        for chunk in [msg[i:i + 1900] for i in range(0, len(msg), 1900)]:
            try:
                send_discord_alert(chunk)
            except Exception as e:
                print(f"⚠️ Discord send failed: {e}")
    elif end_of_day and not test_mode:
        try:
            send_discord_alert(
                "😎 All systems stable. No sell signals today.\n"
                f"🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
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
