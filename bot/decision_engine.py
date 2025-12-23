import argparse
import json
import os
import subprocess
from datetime import datetime

import yfinance as yf

from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators
from llm_predict import SellBrain  # MT Brain integration (bear/neutral/bull)

TRACKER_FILE = "bot/sell_alerts_tracker.json"


# ---------------------------
# Helpers
# ---------------------------

def get_usd_to_lei():
    """Fetch live USD/RON conversion rate, fallback to 4.6 if fail."""
    try:
        fx = yf.Ticker("USDRON=X").history(period="1d")
        if not fx.empty:
            return float(fx["Close"].iloc[-1])
    except Exception as e:
        print(f"⚠️ FX fetch failed: {e}")
    return 4.6  # fallback


def load_tracker():
    """Load or initialize persistent tracker."""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if "tickers" not in data:
                    data["tickers"] = {}
                return data
            except json.JSONDecodeError:
                pass
    return {"date": datetime.utcnow().strftime("%Y-%m-%d"), "had_alerts": False, "tickers": {}}


def save_tracker(data):
    """Save persistent state."""
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True) if os.path.dirname(TRACKER_FILE) else None
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def git_commit_tracker():
    """Auto-commit tracker, live results, and MT dataset outputs to GitHub repo."""
    try:
        print("📝 Committing updated tracker, results, and MT datasets...")

        subprocess.run(["git", "config", "--global", "user.email", "bot@github.com"], check=False)
        subprocess.run(["git", "config", "--global", "user.name", "AutoBot"], check=False)

        files_to_commit = [
            "bot/sell_alerts_tracker.json",
            "bot/live_results.csv",
            "bot/LLM_data/input_llm/llm_input_latest.csv",
            "bot/LLM_data/input_llm/llm_predictions.csv",
        ]

        for file_path in files_to_commit:
            if os.path.exists(file_path):
                subprocess.run(["git", "add", file_path], check=False)
            else:
                print(f"⚠️ Skipping missing file: {file_path}")

        commit_msg = f"🤖 Auto-update tracker + MT data [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        subprocess.run(["git", "pull", "--rebase"], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✅ Tracker and MT data committed successfully.")
    except Exception as e:
        print(f"⚠️ Git commit failed: {e}")


def infer_market_trend(indicators: dict) -> int:
    """
    Production-safe MarketTrend proxy if you don't already compute MarketTrend upstream.

    Returns:
      -1 = bear, 0 = neutral, 1 = bull

    Logic (simple, robust):
      - bull if MA50 > MA200 by a margin
      - bear if MA50 < MA200 by a margin
      - else neutral

    Replace this with your own MarketTrend computation when ready.
    """
    ma50 = indicators.get("ma50")
    ma200 = indicators.get("ma200")
    if ma50 is None or ma200 is None:
        return 0

    try:
        ma50 = float(ma50)
        ma200 = float(ma200)
    except Exception:
        return 0

    if ma200 <= 0:
        return 0

    # 1% band to avoid flip-flopping
    band = 0.01 * ma200
    if ma50 > ma200 + band:
        return 1
    if ma50 < ma200 - band:
        return -1
    return 0


# ---------------------------
# Scoring Engine (no direct SELL return)
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
    """Compute score and weak streak (no binary SELL)."""
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
            return False, f"📈 Oversold (RSI={rsi:.1f}), HOLD.", current_price, 0
        if momentum and momentum >= 0:
            return False, "📈 Momentum stabilizing → HOLD.", current_price, 0
        return True, "🛑 Hard Stop Loss Triggered (-25%)", current_price, 0

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

    return False, "🟢 **Holding steady.**", current_price, float(avg_score)


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (part / total) * 100.0


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

    try:
        sell_brain = SellBrain()
        print("🧠 MT SELL brain loaded successfully (bear/neutral/bull).")
    except Exception as e:
        print(f"⚠️ Could not load MT brain: {e}")
        sell_brain = None

    stocks = tracked["stocks"]
    usd_to_lei = get_usd_to_lei()
    sell_alerts = []

    # === Context helper for UI ===
    def context_tag(pnl):
        if pnl is None:
            return "⚪ Neutral"
        if pnl > 50:
            return "💎 Massive Gain Softening"
        elif pnl > 30:
            return "💰 Big Gain Cooling Off"
        elif pnl > 10:
            return "📈 Profit Losing Strength"
        elif pnl > 0:
            return "🟡 Minor Gain Under Stress"
        elif pnl > -5:
            return "📉 Slight Loss Control"
        else:
            return "🩸 Drawdown Risk"

    for ticker, info in stocks.items():
        avg_price = float(info.get("avg_price", 0))
        invested_lei = float(info.get("invested_lei", 0))
        shares = float(info.get("shares", 0))
        if avg_price <= 0 or invested_lei <= 0 or shares <= 0:
            continue

        indicators = compute_indicators(ticker)
        if not indicators:
            continue

        current_price = indicators["current_price"]
        pnl_lei = current_price * shares * usd_to_lei - invested_lei
        pnl_pct = (pnl_lei / invested_lei) * 100
        info_state = tracker["tickers"].get(ticker, {})

        _, _, _, avg_score = check_sell_conditions(
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

        # --- MarketTrend (proxy; replace with your own later) ---
        mt = infer_market_trend(indicators)

        # --- MT Brain outputs ---
        mt_prob = None
        mt_prob_thr = None
        mt_gate = 0.0
        mt_weight = 0.0
        pred_sellscore = None
        sell_threshold = None
        mt_sell_signal = False
        model_type = None

        if sell_brain:
            try:
                # IMPORTANT: Use SellBrain.predict(...) (source-of-truth)
                pred = sell_brain.predict(indicators, market_trend=mt)
                if isinstance(pred, dict):
                    mt_prob = pred.get("mt_prob")
                    mt_prob_thr = pred.get("mt_prob_threshold")
                    mt_gate = float(pred.get("mt_gate") or 0.0)
                    mt_weight = float(pred.get("mt_weight") or 0.0)
                    pred_sellscore = pred.get("pred_sellscore")
                    sell_threshold = pred.get("sell_threshold")
                    mt_sell_signal = bool(pred.get("mt_sell_signal") or False)
                    model_type = pred.get("model_type")
            except Exception as e:
                print(f"⚠️ MT prediction failed for {ticker}: {e}")

        # --- Fusion logic ---
        # Deterministic is proportional; MT is gated (confidence ramp) and weighted per regime.
        rule_norm = min(1.0, float(avg_score) / 10.0)

        # Weights: bias + rule + mt = 1 (clamped)
        w_bias = 0.15
        w_mt = max(0.0, min(0.85, float(mt_weight or 0.0)))
        w_rule = max(0.0, 1.0 - w_bias - w_mt)

        # If weights drift due to bad mt_weight, renormalize safely
        w_sum = w_bias + w_mt + w_rule
        if w_sum <= 0:
            w_bias, w_mt, w_rule = 0.15, 0.0, 0.85
            w_sum = 1.0
        else:
            w_bias /= w_sum
            w_mt /= w_sum
            w_rule /= w_sum

        rule_contrib = w_rule * rule_norm
        ml_contrib = w_mt * float(mt_gate or 0.0)
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
        if sell_index >= 0.75 and weak_streak >= 1.0:
            decision = True
            signal_label = "🔥 **STRONG SELL SIGNAL**"
            color_tag = "🔴"
        elif sell_index >= 0.60:
            decision = True
            signal_label = "⚠️ **EARLY SELL ALERT**"
            color_tag = "🟠"
        else:
            decision = False
            signal_label = "🟢 **HOLD / WATCH MODE**"
            color_tag = "🟢"

        pnl_context = context_tag(pnl_pct)

        # ML confidence: reuse gate (0..1) and show as %
        ml_conf_pct = float(mt_gate or 0.0) * 100.0

        # Friendly ML line (handles missing ML gracefully)
        if mt_prob is None:
            ml_line = (
                f"🤖 [ML] {ticker}: unavailable | w={w_mt:.2f} | contrib={ml_contrib:.2f} ({m_pct:.0f}%)"
            )
        else:
            thr_txt = f"{float(mt_prob_thr):.2f}" if mt_prob_thr is not None else "n/a"
            ss_txt = f"{float(pred_sellscore):.3f}" if pred_sellscore is not None else "n/a"
            st_txt = f"{float(sell_threshold):.3f}" if sell_threshold is not None else "n/a"
            p_txt = f"{float(mt_prob):.3f}"
            ml_line = (
                f"🤖 [ML] {ticker}: sell={mt_sell_signal} | P={p_txt} thr={thr_txt} conf={ml_conf_pct:.0f}% "
                f"gate={float(mt_gate or 0.0):.2f} w={w_mt:.2f} | pred_score={ss_txt} vs sell_thr={st_txt}"
            )

        contrib_line = (
            f"🧩 Mix: Rule={rule_contrib:.2f} ({r_pct:.0f}%) | ML={ml_contrib:.2f} ({m_pct:.0f}%) | "
            f"Bias={bias_contrib:.2f} ({b_pct:.0f}%)"
        )

        soft_line = "💤 ML softened borderline SELL → HOLD." if mt_softened else ""

        reasoning = (
            f"{color_tag} **{signal_label}**\n"
            f"{pnl_context}\n\n"
            f"💰 **PnL:** {pnl_pct:+.2f}%\n"
            f"📊 **AvgScore:** {avg_score:.2f} | Weak: {weak_streak:.1f}/3\n"
            f"🧠 **Consensus Index:** {sell_index:.2f}\n"
            f"{contrib_line}\n"
            f"{ml_line}\n"
            f"{soft_line}"
        )

        # Console summary (friendly but information-dense)
        print(
            f"{color_tag} {ticker}: {signal_label} | SellIndex={sell_index:.2f} | Weak={weak_streak:.1f} "
            f"| PnL={pnl_pct:+.2f}% | MT={mt:+d} | Rule%={r_pct:.0f} ML%={m_pct:.0f} Bias%={b_pct:.0f}"
        )
        print(ml_line)

        # --- Persist debug fields into tracker (for Discord !list) ---
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

        if decision:
            now_utc = datetime.utcnow()
            info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            sell_alerts.append(
                f"📈 **[{ticker}] {signal_label}**\n"
                f"{reasoning}\n"
                f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

        tracker["tickers"][ticker] = info_state

    save_tracker(tracker)
    if not test_mode:
        git_commit_tracker()

    # === Send Discord summaries ===
    now_utc = datetime.utcnow()
    if sell_alerts:
        msg = "🚨 **SELL SIGNALS TRIGGERED** 🚨\n\n" + "\n\n".join(sell_alerts)
        for chunk in [msg[i : i + 1900] for i in range(0, len(msg), 1900)]:
            send_discord_alert(chunk)
    elif end_of_day and not test_mode:
        send_discord_alert(
            "😎 All systems stable. No sell signals today.\n"
            f"🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

    print(f"✅ Decision Engine Run Complete at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--endofday", action="store_true")
    args = parser.parse_args()
    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)
