import argparse
import json
import os
import subprocess
from datetime import datetime

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


# ---------------------------
# FX helpers (multi-currency, cached)
# ---------------------------
_FX_CACHE: dict[str, float] = {}
_TICKER_CCY_CACHE: dict[str, str] = {}


def _fx_pair_to_ron(currency: str) -> str | None:
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return None
    # yfinance FX symbols
    if c == "USD":
        return "USDRON=X"
    if c == "EUR":
        return "EURRON=X"
    if c == "GBP":
        return "GBPRON=X"
    if c == "CHF":
        return "CHFRON=X"
    # Add more pairs as needed
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
        # Unknown currency → conservative fallback to 1.0 (avoid exploding PnL).
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
    # fallback for common case
    fallback = 4.6 if c == "USD" else 1.0
    _FX_CACHE[c] = fallback
    return fallback


# ---------------------------
# Tracker helpers
# ---------------------------
def load_tracker():
    """Load or initialize persistent tracker."""
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
    """Save persistent state."""
    d = os.path.dirname(TRACKER_FILE)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# Git commit (truthful)
# ---------------------------
def _run(cmd: list[str]) -> int:
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[GIT][FAIL] {' '.join(cmd)}\n  stdout={r.stdout[-500:]}\n  stderr={r.stderr[-500:]}")
        return int(r.returncode)
    except Exception as e:
        print(f"[GIT][EXC] {' '.join(cmd)} -> {e!r}")
        return 1


def git_commit_tracker():
    """Auto-commit tracker, live results, and MT dataset outputs to GitHub repo (truthful)."""
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
# Scoring Engine (returns SELL decision + message)
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
    """Compute score and weak streak; returns (sell_bool, message, current_price, avg_score)."""
    if info is None:
        info = {}

    info.setdefault("weak_streak", 0.0)
    info.setdefault("recent_peak", current_price)
    info.setdefault("rolling_scores", [])
    info.setdefault("last_decay_date", None)
    info.setdefault("was_above_47", False)

    # --- Hard Stop Loss (-25%) ---
    if pnl_pct is not None and pnl_pct <= -25:
        if rsi and rsi < 35:
            return False, f"📈 Oversold (RSI={rsi:.1f}), HOLD.", current_price, 0.0
        if momentum and momentum >= 0:
            return False, "📈 Momentum stabilizing → HOLD.", current_price, 0.0
        return True, "🛑 Hard Stop Loss Triggered (-25%)", current_price, 10.0

    score, reasons = 0.0, []

    # --- Core scoring factors ---
    if momentum is not None:
        if momentum < -0.8:
            score += 2
            reasons.append("📉 Momentum Collapse (< -0.8)")
        elif momentum < -0.3:
            score += 1
            reasons.append("📉 Weak Momentum (< -0.3)")

    if rsi is not None:
        if rsi < 35:
            score += 1.5
            reasons.append("📉 RSI Oversold (<35)")
        elif rsi < 45:
            score += 1
            reasons.append("📉 RSI Weak (<45)")
        elif rsi > 70:
            score += 0.5
            reasons.append("📈 RSI Overbought (>70)")

    if macd is not None and macd_signal is not None and macd < macd_signal:
        score += 1.5
        reasons.append("📉 MACD Bearish Crossover")

    if ma50 and current_price < ma50:
        score += 1
        reasons.append("📉 Below MA50")

    if ma200 and current_price < ma200:
        score += 2
        reasons.append("📉 Below MA200 (Major Breakdown)")

    if support and current_price < support:
        score += 2
        reasons.append("📉 Support Broken")

    if volume and volume > 1.3:
        score += 1
        reasons.append("📉 High Volume Breakdown (>1.3x avg)")

    if atr and pnl_pct is not None and atr > 7 and pnl_pct < 0:
        score += 0.5
        reasons.append("⚡ High Volatility + Loss")

    # --- Rolling scores and weak streak ---
    rolling = info.get("rolling_scores", [])
    rolling.append(float(score))
    if len(rolling) > 7:
        rolling.pop(0)
    info["rolling_scores"] = rolling
    avg_score = sum(rolling) / len(rolling) if rolling else float(score)

    info["recent_peak"] = max(info["recent_peak"], current_price)

    # --- Weak streak logic ---
    now_utc = datetime.utcnow()
    market_hour = 13 <= now_utc.hour <= 21
    if pnl_pct is not None:
        if pnl_pct >= 47:
            info["was_above_47"] = True
        elif pnl_pct < 29:
            info["was_above_47"] = False

    if market_hour and info.get("last_decay_date") != now_utc.strftime("%Y-%m-%d"):
        info["weak_streak"] = max(0.0, info["weak_streak"] - 0.5)
        info["last_decay_date"] = now_utc.strftime("%Y-%m-%d")

    quiet_market = (atr is not None and atr < 3) and (volume is not None and volume < 0.7)
    if score >= 6.5:
        info["weak_streak"] += 2.0
    elif score >= 4.0:
        info["weak_streak"] += 1.0 if not quiet_market else 0.5
    elif score >= 3.0 and quiet_market:
        info["weak_streak"] += 0.5
    elif momentum and momentum > 0.4 and rsi and rsi > 50:
        info["weak_streak"] = 0.0
    elif score < 3.0 and info["weak_streak"] > 0:
        info["weak_streak"] -= 0.5

    info["weak_streak"] = max(0.0, round(info["weak_streak"], 1))

    if debug:
        print(f"⏳ {ticker}: Weak {info['weak_streak']:.1f}/3 — AvgScore={avg_score:.1f}")

    return False, "🟢 Holding steady.", current_price, float(avg_score)


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (part / total) * 100.0


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


def _load_llm_input_map() -> dict[str, dict]:
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


def _ml_line(ticker: str, mt: int, pred: dict | None, w_mt: float, ml_contrib: float, m_pct: float) -> str:
    if not pred:
        return f"🤖 [ML] {ticker}: unavailable | MT={mt:+d} | w={w_mt:.2f} contrib={ml_contrib:.2f} ({m_pct:.0f}%)"

    mt_prob = pred.get("mt_prob")
    mt_thr = pred.get("mt_prob_threshold")
    mt_gate = float(pred.get("mt_gate") or 0.0)
    mt_sell = bool(pred.get("mt_sell_signal") or False)
    pred_score = pred.get("pred_sellscore")
    sell_thr = pred.get("sell_threshold")
    model_type = pred.get("model_type") or "?"
    prob_source = pred.get("prob_source") or "?"

    p_txt = "n/a" if mt_prob is None else f"{float(mt_prob):.3f}"
    thr_txt = "n/a" if mt_thr is None else f"{float(mt_thr):.2f}"
    gate_txt = f"{mt_gate:.2f}"
    ss_txt = "n/a" if pred_score is None else f"{float(pred_score):.3f}"
    st_txt = "n/a" if sell_thr is None else f"{float(sell_thr):.3f}"

    return (
        f"🤖 [ML] {ticker}: sell={mt_sell} | MT={mt:+d} | P={p_txt} thr={thr_txt} gate={gate_txt} "
        f"w={w_mt:.2f} contrib={ml_contrib:.2f} ({m_pct:.0f}%) | score={ss_txt} vs sell_thr={st_txt} "
        f"| {model_type} | src={prob_source}"
    )


# ---------------------------
# Runner with Fusion Logic (Deterministic + MT Brain Gate)
# ---------------------------
def run_decision_engine(test_mode=False, end_of_day=False):
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracker = load_tracker()

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if tracker.get("date") != today_str:
        print(f"🧹 Resetting daily alert flag for {today_str}")
        tracker["date"] = today_str
        tracker["had_alerts"] = False

    tracked = load_data(file_to_load)
    if not tracked or "stocks" not in tracked:
        print(f"⚠️ No tracked stocks found in {file_to_load}")
        return

    # Load ML input rows (feature-consistent with training + JSON feature_columns)
    llm_input_map = _load_llm_input_map()

    # Load brain and (optionally) generate fresh batch predictions for audit
    sell_brain = None
    try:
        sell_brain = SellBrain()
        print("🧠 MT SELL brain loaded successfully (bear/neutral/bull).")
        # Keep predictions file in sync when possible (non-fatal if it fails)
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

        # Rule indicators (you may later swap to DB source-of-truth)
        indicators = compute_indicators(ticker)
        if not indicators:
            continue

        current_price = float(indicators["current_price"])
        info_state = tracker["tickers"].get(ticker, {})

        # Multi-currency PnL conversion (ticker currency -> RON)
        ccy = get_ticker_currency(ticker)
        fx_to_ron = get_fx_to_ron(ccy)
        pnl_lei = current_price * shares * fx_to_ron - invested_lei
        pnl_pct = (pnl_lei / invested_lei) * 100.0 if invested_lei else 0.0

        # Rule decision (including hard stop-loss override)
        rule_sell, rule_msg, _, avg_score = check_sell_conditions(
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

        # MarketTrend + ML features must come from the ML input dataset row (training-consistent)
        ml_row = llm_input_map.get(ticker, {})
        mt = int(_safe_float(ml_row.get("MarketTrend"), 0) or 0)
        if mt not in (-1, 0, 1):
            mt = 0

        # --- MT Brain outputs ---
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
        mt_prob_thr = None if not mt_pred else mt_pred.get("mt_prob_threshold")
        mt_gate = 0.0 if not mt_pred else float(mt_pred.get("mt_gate") or 0.0)
        mt_weight = 0.0 if not mt_pred else float(mt_pred.get("mt_weight") or 0.0)
        pred_sellscore = None if not mt_pred else mt_pred.get("pred_sellscore")
        sell_threshold = None if not mt_pred else mt_pred.get("sell_threshold")
        mt_sell_signal = False if not mt_pred else bool(mt_pred.get("mt_sell_signal") or False)
        model_type = None if not mt_pred else mt_pred.get("model_type")
        prob_source = None if not mt_pred else mt_pred.get("prob_source")

        # --- Fusion logic ---
        rule_norm = min(1.0, float(avg_score) / 10.0)

        # Weights: bias + rule + mt = 1 (clamped)
        w_bias = 0.15
        w_mt = max(0.0, min(0.85, float(mt_weight or 0.0)))
        w_rule = max(0.0, 1.0 - w_bias - w_mt)

        # Renormalize safely
        w_sum = w_bias + w_mt + w_rule
        if w_sum <= 0:
            w_bias, w_mt, w_rule = 0.15, 0.0, 0.85
            w_sum = 1.0
        else:
            w_bias /= w_sum
            w_mt /= w_sum
            w_rule /= w_sum

        rule_contrib = w_rule * rule_norm
        ml_contrib = w_mt # weight portion
        # IMPORTANT: MT contribution is gated confidence ramp (0..1)
        ml_contrib = ml_contrib * float(mt_gate or 0.0)

        bias_contrib = w_bias
        sell_index = rule_contrib + ml_contrib + bias_contrib
        sell_index = max(0.0, min(sell_index, 1.0))

        # Optional softening: deterministic borderline sell gets softened if ML is NOT confident.
        mt_softened = False
        if (
            6.0 <= float(avg_score) < 7.0
            and mt_prob is not None
            and mt_prob_thr is not None
            and float(mt_prob) < float(mt_prob_thr)
        ):
            sell_index = max(0.0, sell_index - 0.15)
            mt_softened = True

        # Contribution percentages relative to final sell_index
        r_pct = _pct(rule_contrib, sell_index)
        m_pct = _pct(ml_contrib, sell_index)
        b_pct = _pct(bias_contrib, sell_index)

        # === Determine decision zone ===
        decision = False
        signal_label = "🟢 HOLD / WATCH MODE"
        color_tag = "🟢"

        # Hard override: never ignore rule SELL (hard stop-loss)
        if bool(rule_sell):
            decision = True
            signal_label = "🛑 HARD STOP-LOSS SELL"
            color_tag = "🔴"
        else:
            if sell_index >= 0.75 and weak_streak >= 1.0:
                decision = True
                signal_label = "🔥 STRONG SELL SIGNAL"
                color_tag = "🔴"
            elif sell_index >= 0.60:
                decision = True
                signal_label = "⚠️ EARLY SELL ALERT"
                color_tag = "🟠"

        pnl_context = _context_tag(pnl_pct)
        ml_conf_pct = float(mt_gate or 0.0) * 100.0

        ml_line = _ml_line(
            ticker=ticker,
            mt=mt,
            pred=mt_pred,
            w_mt=w_mt,
            ml_contrib=ml_contrib,
            m_pct=m_pct,
        )

        contrib_line = (
            f"🧩 Mix: Rule={rule_contrib:.2f} ({r_pct:.0f}%) | ML={ml_contrib:.2f} ({m_pct:.0f}%) | "
            f"Bias={bias_contrib:.2f} ({b_pct:.0f}%)"
        )
        soft_line = "💤 ML softened borderline SELL → HOLD." if mt_softened else ""

        reasoning = (
            f"{color_tag} **{signal_label}**\n"
            f"{pnl_context}\n\n"
            f"💰 **PnL:** {pnl_pct:+.2f}% ({ccy}→RON fx={fx_to_ron:.4f})\n"
            f"📊 **AvgScore:** {avg_score:.2f} | Weak: {weak_streak:.1f}/3\n"
            f"🧠 **Consensus Index:** {sell_index:.2f}\n"
            f"{contrib_line}\n"
            f"{ml_line}\n"
            f"{soft_line}\n"
            f"🧾 Rule note: {rule_msg}"
        )

        # Console summary
        print(
            f"{color_tag} {ticker}: {signal_label} | SellIndex={sell_index:.2f} | Weak={weak_streak:.1f} "
            f"| PnL={pnl_pct:+.2f}% | CCY={ccy} | MT={mt:+d} | Rule%={r_pct:.0f} ML%={m_pct:.0f} Bias%={b_pct:.0f}"
        )
        print(ml_line)

        # --- Persist debug fields into tracker ---
        info_state["last_score"] = float(avg_score)
        info_state["last_sell_index"] = float(sell_index)
        info_state["last_mt"] = int(mt)

        info_state["last_w_rule"] = float(w_rule)
        info_state["last_w_mt"] = float(w_mt)
        info_state["last_w_bias"] = float(w_bias)

        info_state["last_contrib_rule"] = float(rule_contrib)
        info_state["last_contrib_ml"] = float(ml_contrib)
        info_state["last_contrib_bias"] = float(bias_contrib)

        info_state["last_contrib_rule_pct"] = float(r_pct)
        info_state["last_contrib_ml_pct"] = float(m_pct)
        info_state["last_contrib_bias_pct"] = float(b_pct)

        info_state["last_mt_prob"] = None if mt_prob is None else float(mt_prob)
        info_state["last_mt_gate"] = float(mt_gate)
        info_state["last_mt_prob_thr"] = None if mt_prob_thr is None else float(mt_prob_thr)
        info_state["last_mt_weight"] = float(mt_weight or 0.0)
        info_state["last_mt_pred_sellscore"] = None if pred_sellscore is None else float(pred_sellscore)
        info_state["last_mt_sell_threshold"] = None if sell_threshold is None else float(sell_threshold)
        info_state["last_mt_sell_signal"] = bool(mt_sell_signal)
        info_state["last_mt_model_type"] = model_type
        info_state["last_mt_prob_source"] = prob_source
        info_state["last_currency"] = ccy
        info_state["last_fx_to_ron"] = float(fx_to_ron)

        now_utc = datetime.utcnow()
        if decision:
            info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            sell_alerts.append(
                f"📈 **[{ticker}] {signal_label}**\n"
                f"{reasoning}\n"
                f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

        tracker["tickers"][ticker] = info_state

        live_rows.append({
            "Timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Ticker": ticker,
            "decision": bool(decision),
            "signal_label": signal_label,
            "SellIndex": float(sell_index),
            "AvgScore": float(avg_score),
            "WeakStreak": float(weak_streak),
            "PnL_pct": float(pnl_pct),
            "Currency": ccy,
            "FX_to_RON": float(fx_to_ron),
            "MarketTrend": int(mt),
            "mt_prob": None if mt_prob is None else float(mt_prob),
            "mt_prob_threshold": None if mt_prob_thr is None else float(mt_prob_thr),
            "mt_gate": float(mt_gate),
            "mt_sell_signal": bool(mt_sell_signal),
            "pred_sellscore": None if pred_sellscore is None else float(pred_sellscore),
            "sell_threshold": None if sell_threshold is None else float(sell_threshold),
            "model_type": model_type,
            "prob_source": prob_source,
            "w_rule": float(w_rule),
            "w_mt": float(w_mt),
            "w_bias": float(w_bias),
        })

    # Persist tracker + daily alert flag
    tracker["had_alerts"] = bool(tracker.get("had_alerts") or (len(sell_alerts) > 0))
    save_tracker(tracker)

    # Save live results CSV
    try:
        os.makedirs(os.path.dirname(LIVE_RESULTS_CSV), exist_ok=True) if os.path.dirname(LIVE_RESULTS_CSV) else None
        pd.DataFrame(live_rows).to_csv(LIVE_RESULTS_CSV, index=False)
        print(f"🧾 Live results saved → {LIVE_RESULTS_CSV}")
    except Exception as e:
        print(f"⚠️ Failed to write {LIVE_RESULTS_CSV}: {e}")

    if not test_mode:
        git_commit_tracker()

    # === Send Discord summaries (safe) ===
    now_utc = datetime.utcnow()
    if sell_alerts:
        msg = "🚨 **SELL SIGNALS TRIGGERED** 🚨\n\n" + "\n\n".join(sell_alerts)
        for chunk in [msg[i: i + 1900] for i in range(0, len(msg), 1900)]:
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
