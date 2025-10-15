import argparse
import yfinance as yf
from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators

from datetime import datetime, timedelta
import csv
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
    """Auto-commit the tracker & live results to GitHub repo."""
    try:
        print("📝 Committing updated tracker and results...")
        subprocess.run(["git", "config", "--global", "user.email", "bot@github.com"])
        subprocess.run(["git", "config", "--global", "user.name", "AutoBot"])
        subprocess.run(["git", "add", "bot/sell_alerts_tracker.json", "bot/live_results.csv"], check=False)
        commit_msg = f"🤖 Auto-update tracker [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        subprocess.run(["git", "pull", "--rebase"], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✅ Tracker committed successfully.")
    except Exception as e:
        print(f"⚠️ Git commit failed: {e}")


# ---------------------------
# Sell Condition Logic
# ---------------------------
def check_sell_conditions(
    ticker: str, buy_price: float, current_price: float,
    pnl_pct=None,
    volume=None, momentum=None, rsi=None, market_trend=None,
    ma50=None, ma200=None, atr=None,
    macd=None, macd_signal=None,
    bb_upper=None, bb_lower=None,
    resistance=None, support=None,
    info=None,
    debug=True
):
    """Enhanced sell logic with persistence and adaptive trailing."""
    if info is None:
        info = {}

    info.setdefault("weak_streak", 0)
    info.setdefault("recent_peak", current_price)

    # --- HARD STOP LOSS ---
    if pnl_pct <= -25:
        if rsi and rsi < 35:
            return False, f"📈 Oversold (RSI={rsi:.1f}), HOLD.", current_price, 0
        if momentum and momentum >= 0:
            return False, f"📈 Momentum stabilizing → HOLD.", current_price, 0
        if market_trend == "BULLISH":
            return False, f"📈 Bullish market, avoid panic sell.", current_price, 0
        return True, f"🛑 Stop Loss Triggered (-25%)", current_price, 0

    # --- WEAKNESS TRACKER ---
    if (momentum is not None and momentum < 0) or (rsi is not None and rsi < 45) or (ma50 and current_price < ma50):
        info["weak_streak"] += 1
    else:
        info["weak_streak"] = 0

    # --- SCORING ---
    score, reasons = 0, []
    if momentum is not None and momentum < -0.5:
        score += 2
        reasons.append("📉 Strong Negative Momentum (< -0.5)")
    if macd is not None and macd_signal is not None and macd < macd_signal:
        score += 1.5
        reasons.append("📉 MACD Bearish Crossover")
    if rsi and rsi > 70:
        score += 0.5
        reasons.append("📉 RSI Overbought (>70)")
    if ma50 and current_price < ma50:
        score += 1
        reasons.append("📉 Below MA50")
    if ma200 and current_price < ma200:
        score += 2
        reasons.append("📉 Below MA200 (major shift)")
    if atr and atr > 7 and pnl_pct < -10:
        score += 0.5
        reasons.append("⚡ High ATR + Loss")
    if bb_upper and current_price > bb_upper:
        score += 0.5
        reasons.append("📉 Above Upper Bollinger Band")
    if support and current_price < support:
        if (rsi and rsi < 45) or (momentum and momentum < 0):
            score += 2
            reasons.append("📉 Broke Support with Weak RSI/Momentum")
        else:
            reasons.append("⚠️ Touched Support — no confirmation")
    if volume and volume > 1.5:
        if (ma50 and current_price < ma50) or (support and current_price < support):
            score += 1.5
            reasons.append("📉 Breakdown confirmed by High Volume")

    # --- DEBUG OUTPUT ---
    if debug:
        print(f"⏳ {ticker}: Weak {info['weak_streak']}/3 — waiting confirmation")
        print(f"🧮 DEBUG {ticker}: Score={score:.1f}, Reasons={reasons}")

    if info["weak_streak"] < 3:
        return False, f"🕐 Weakness streak {info['weak_streak']}/3 — waiting confirmation", current_price, score

    # --- QUICK REBOUND CANCEL ---
    if info.get("last_sell_trigger_price") and current_price > info["last_sell_trigger_price"] * 1.04:
        info["weak_streak"] = 0
        return False, "📈 Quick rebound (>4%) — cancelling.", current_price, score

    # --- PROFIT LOCK-IN ---
    info["recent_peak"] = max(info["recent_peak"], current_price)
    drop_from_peak = 100 * (1 - current_price / info["recent_peak"])
    if pnl_pct >= 20 and (drop_from_peak >= 7 or (momentum < 0 and macd < macd_signal)):
        return True, f"💰 Trailing Stop +{pnl_pct:.1f}% drop {drop_from_peak:.1f}%", current_price, score

    # --- FINAL EXIT CONDITIONS ---
    if pnl_pct >= 25 and score >= 3:
        return True, f"🎯 Profit +25%, weakening (score {score})", current_price, score
    if score >= 4:
        info["last_sell_trigger_price"] = current_price
        return True, f"🚨 Score {score} (≥4) triggered SELL", current_price, score

    return False, "🟢 HOLD — no sell signal", current_price, score


# ---------------------------
# Runner
# ---------------------------
def run_decision_engine(test_mode=False, end_of_day=False):
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracked = load_data(file_to_load)
    if not tracked or "stocks" not in tracked:
        print(f"⚠️ No tracked stocks found in {file_to_load}")
        return

    stocks = tracked["stocks"]
    tracker = load_tracker()
    usd_to_lei = get_usd_to_lei()
    sell_alerts = []
    weak_near = []

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

        decision, reason, _, score = check_sell_conditions(
            ticker, avg_price, current_price,
            pnl_pct=pnl_pct,
            volume=indicators.get("volume"),
            momentum=indicators.get("momentum"),
            rsi=indicators.get("rsi"),
            market_trend=indicators.get("market_trend"),
            ma50=indicators.get("ma50"),
            ma200=indicators.get("ma200"),
            atr=indicators.get("atr"),
            macd=indicators.get("macd"),
            macd_signal=indicators.get("macd_signal"),
            bb_upper=indicators.get("bb_upper"),
            bb_lower=indicators.get("bb_lower"),
            resistance=indicators.get("resistance"),
            support=indicators.get("support"),
            info=info_state,
            debug=True
        )

        tracker["tickers"][ticker] = info_state

        # Collect “Weak” ones
        if info_state.get("weak_streak", 0) in [1, 2]:
            weak_near.append(f"⚠️ {ticker}: Weak {info_state['weak_streak']}/3")

        # Apply cooldown only to SELL alerts
        if decision:
            now_utc = datetime.utcnow()
            send_alert = True
            last_alert = info_state.get("last_alert_time")
            last_score = info_state.get("last_score", 0)
            if last_alert:
                last_alert_dt = datetime.strptime(last_alert, "%Y-%m-%dT%H:%M:%SZ")
                hours_since = (now_utc - last_alert_dt).total_seconds() / 3600
                if hours_since < 3 and abs(score - last_score) < 0.5:
                    send_alert = False

            if send_alert:
                info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                info_state["last_score"] = score
                sell_alerts.append(f"**{ticker}** — {reason} ({pnl_pct:.2f}%)")

       # --- Discord Output Logic ---
    now_utc = datetime.utcnow()

    # 🚨 Send SELL alerts immediately when they happen
    if sell_alerts:
        msg = "🚨 **SELL ALERTS TRIGGERED** 🚨\n\n" + "\n".join(sell_alerts)
        if weak_near:
            msg += "\n\n📉 **Approaching Weak Signals:**\n" + "\n".join(weak_near)
        if not test_mode:
            send_discord_alert(msg)
            tracker["had_alerts"] = True

    # 🕐 End-of-day summary
    elif end_of_day and not test_mode:
        weak_list = []
        for ticker, state in tracker["tickers"].items():
            if state.get("weak_streak", 0) > 0:
                weak_list.append(f"⚠️ {ticker}: Weak {state['weak_streak']}/3")

        if weak_list:
            summary = (
                "📊 **Daily Monitoring Update**\n\n"
                + "\n".join(weak_list)
                + f"\n\n🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
            send_discord_alert(summary)
        elif not tracker["had_alerts"]:
            send_discord_alert(
                f"😎 No sell signals today. Business doing good, boss!\n🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

    # --- Save and commit tracker ---
    save_tracker(tracker)
    if not test_mode:
        git_commit_tracker()

    print(f"✅ Decision Engine Run Complete at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--endofday", action="store_true")
    args = parser.parse_args()

    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)
