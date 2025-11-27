import argparse
import yfinance as yf
from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators
from llm_predict import SellBrain  # ✅ new brain integration

from datetime import datetime
import os
import json
import subprocess

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
        with open(TRACKER_FILE, "r") as f:
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
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def git_commit_tracker():
    """Auto-commit tracker, live results, and LLM data to GitHub repo."""
    try:
        print("📝 Committing updated tracker, results, and LLM datasets...")

        subprocess.run(["git", "config", "--global", "user.email", "bot@github.com"], check=False)
        subprocess.run(["git", "config", "--global", "user.name", "AutoBot"], check=False)

        files_to_commit = [
            "bot/sell_alerts_tracker.json",
            "bot/live_results.csv",
            "LLM_data/input_llm/llm_input_latest.csv",
            "LLM_data/input_llm/llm_predictions.csv"
        ]

        for file_path in files_to_commit:
            if os.path.exists(file_path):
                subprocess.run(["git", "add", file_path], check=False)
            else:
                print(f"⚠️ Skipping missing file: {file_path}")

        commit_msg = f"🤖 Auto-update tracker + LLM data [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        subprocess.run(["git", "pull", "--rebase"], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✅ Tracker and LLM data committed successfully.")
    except Exception as e:
        print(f"⚠️ Git commit failed: {e}")


# ---------------------------
# Scoring Engine (no direct SELL return)
# ---------------------------
def check_sell_conditions(
    ticker: str, buy_price: float, current_price: float,
    pnl_pct=None,
    volume=None, momentum=None, rsi=None,
    ma50=None, ma200=None, atr=None,
    macd=None, macd_signal=None,
    resistance=None, support=None,
    info=None,
    debug=True
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
            return False, f"📈 Momentum stabilizing → HOLD.", current_price, 0
        return True, "🛑 Hard Stop Loss Triggered (-25%)", current_price, 0

    score, reasons = 0, []

    # --- Core scoring factors ---
    if momentum is not None:
        if momentum < -0.8:
            score += 2; reasons.append("📉 Momentum Collapse (< -0.8)")
        elif momentum < -0.3:
            score += 1; reasons.append("📉 Weak Momentum (< -0.3)")

    if rsi is not None:
        if rsi < 35:
            score += 1.5; reasons.append("📉 RSI Oversold (<35)")
        elif rsi < 45:
            score += 1; reasons.append("📉 RSI Weak (<45)")
        elif rsi > 70:
            score += 0.5; reasons.append("📈 RSI Overbought (>70)")

    if macd is not None and macd_signal is not None and macd < macd_signal:
        score += 1.5; reasons.append("📉 MACD Bearish Crossover")

    if ma50 and current_price < ma50:
        score += 1; reasons.append("📉 Below MA50")

    if ma200 and current_price < ma200:
        score += 2; reasons.append("📉 Below MA200 (Major Breakdown)")

    if support and current_price < support:
        score += 2; reasons.append("📉 Support Broken")

    if volume and volume > 1.3:
        score += 1; reasons.append("📉 High Volume Breakdown (>1.3x avg)")

    if atr and pnl_pct is not None and atr > 7 and pnl_pct < 0:
        score += 0.5; reasons.append("⚡ High Volatility + Loss")

    # --- Rolling scores and weak streak ---
    rolling = info.get("rolling_scores", [])
    rolling.append(score)
    if len(rolling) > 7:
        rolling.pop(0)
    info["rolling_scores"] = rolling
    avg_score = sum(rolling) / len(rolling) if rolling else score

    info["recent_peak"] = max(info["recent_peak"], current_price)
    drop_from_peak = 100 * (1 - current_price / info["recent_peak"])

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

    return False, "🟢 **Holding steady.**", current_price, avg_score


# ---------------------------
# Runner with Fusion Logic
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
        print("🧠 LLM SELL brain loaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not load LLM brain: {e}")
        sell_brain = None

    stocks = tracked["stocks"]
    usd_to_lei = get_usd_to_lei()
    sell_alerts, weak_near = [], []

    # === Context helper for UI ===
    def context_tag(pnl):
        if pnl is None: return "⚪ Neutral"
        if pnl > 50: return "💎 Massive Gain Softening"
        elif pnl > 30: return "💰 Big Gain Cooling Off"
        elif pnl > 10: return "📈 Profit Losing Strength"
        elif pnl > 0: return "🟡 Minor Gain Under Stress"
        elif pnl > -5: return "📉 Slight Loss Control"
        else: return "🩸 Drawdown Risk"

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
            ticker, avg_price, current_price,
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
            debug=True
        )

        weak_streak = info_state.get("weak_streak", 0.0)
        ml_prob = None
        if sell_brain:
            try:
                ml_prob = sell_brain.predict_prob(indicators)
            except Exception as e:
                print(f"⚠️ ML prediction failed for {ticker}: {e}")

        # --- Fusion logic ---
        rule_norm = min(1.0, avg_score / 10.0)
        w_rule, w_ml, w_bias = 0.50, 0.325, 0.175
        distance_to_mid = abs(0.55 - rule_norm)
        dynamic_llm_weight = w_ml * (1 - distance_to_mid * 2)
        sell_index = (w_rule * rule_norm) + (dynamic_llm_weight * ml_prob) + w_bias
        sell_index = max(0, min(sell_index, 1))

        # Pull-down effect (LLM softens borderline)
        llm_softened = False
        if avg_score >= 6.0 and avg_score < 7.0 and ml_prob < 0.45:
            sell_index -= 0.15
            llm_softened = True
        sell_index = max(0, min(sell_index, 1))

        # === Determine decision zone ===
        decision = False
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

        # === Build rich message ===
        pnl_context = context_tag(pnl_pct)
        llm_line = f"🤖 LLM Brain: {ml_prob:.2f} | 🧮 Deterministic: {rule_norm:.2f}"
        soft_line = "💤 LLM softened borderline SELL → HOLD." if llm_softened else ""
        reasoning = (
            f"{color_tag} **{signal_label}**\n"
            f"{pnl_context}\n\n"
            f"💰 **PnL:** {pnl_pct:+.2f}%\n"
            f"📊 **AvgScore:** {avg_score:.2f} | Weak: {weak_streak:.1f}/3\n"
            f"🧠 **Consensus Index:** {sell_index:.2f}\n"
            f"{llm_line}\n"
            f"{soft_line}"
        )

        print(f"{color_tag} {ticker}: {signal_label} | SellIndex={sell_index:.2f} | Weak={weak_streak:.1f} | PnL={pnl_pct:+.2f}%")

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
        for chunk in [msg[i:i + 1900] for i in range(0, len(msg), 1900)]:
            send_discord_alert(chunk)
    elif end_of_day and not test_mode:
        send_discord_alert(f"😎 All systems stable. No sell signals today.\n"
                           f"🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    print(f"✅ Decision Engine Run Complete at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--endofday", action="store_true")
    args = parser.parse_args()
    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)
