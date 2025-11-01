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
    """Smart sell logic using score-based weakness, profit protection, and adaptive hard stops."""
    from datetime import datetime

    if info is None:
        info = {}

    info.setdefault("weak_streak", 0)
    info.setdefault("recent_peak", current_price)

    # --- HARD STOP LOSS ---
    if pnl_pct is not None and pnl_pct <= -25:
        if rsi and rsi < 35:
            return False, f"📈 Oversold (RSI={rsi:.1f}), HOLD.", current_price, 0
        if momentum and momentum >= 0:
            return False, f"📈 Momentum stabilizing → HOLD.", current_price, 0
        return True, "🛑 Hard Stop Loss Triggered (-25%)", current_price, 0

    # --- SCORE CALCULATION ---
    score, reasons = 0, []

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

    # --- CALCULATE DROP FROM PEAK ---
    info["recent_peak"] = max(info["recent_peak"], current_price)
    drop_from_peak = 100 * (1 - current_price / info["recent_peak"])

        # --- ADAPTIVE HARD STOPS (for high profits) ---
    if pnl_pct is not None:
        # 🏆 Euphoria Top Guard
        if pnl_pct >= 50:
            if drop_from_peak >= 6 or (macd < macd_signal and rsi > 65):
                return True, (
                    f"🏆 **Euphoria top detected** — locking in big winner.\n"
                    f"Gain: +{pnl_pct:.1f}% | Drop: {drop_from_peak:.1f}% | RSI: {rsi:.1f}"
                ), current_price, score

        # 💎 Adaptive Trailing Stop (big profit, soft reversal)
        if pnl_pct >= 35 and info.get('weak_streak', 0) >= 1:
            if drop_from_peak >= 8 or (score >= 3.5 and momentum < 0):
                return True, (
                    f"💎 **Strong profit showing early weakness.**\n"
                    f"Gain: +{pnl_pct:.1f}% | Drop: {drop_from_peak:.1f}% | Momentum: {momentum:.2f}"
                ), current_price, score

        # ⚠️ Mid-Profit Weakness (moderate profit losing strength)
        if 10 <= pnl_pct < 35 and info.get('weak_streak', 0) >= 2 and score >= 5.0 and momentum < -0.2:
            return True, (
                f"⚠️ **Uptrend under pressure.**\n"
                f"Gain: +{pnl_pct:.1f}% | Score: {score:.1f} | Momentum weakening ({momentum:.2f})"
            ), current_price, score


    # --- MARKET-ACTIVITY CHECK (skip overnight increments) ---
    last_price = info.get("last_checked_price")
    last_time = info.get("last_checked_time")
    now_utc = datetime.utcnow()
    market_hour = 13 <= now_utc.hour <= 21

    price_change = None
    if last_price:
        try:
            price_change = abs((current_price - last_price) / last_price) * 100
        except ZeroDivisionError:
            price_change = None
    skip_weak_update = (price_change is not None and price_change < 0.3) or not market_hour
    info["last_checked_price"] = current_price
    info["last_checked_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- UPDATE WEAK STREAK ---
    if not skip_weak_update:
        if score >= 6.5:
            info["weak_streak"] += 2
        elif score >= 4.5:
            info["weak_streak"] += 1
        elif score < 2.5 or (momentum and momentum > 0.3):
            info["weak_streak"] = 0
        else:
            info["weak_streak"] = max(0, info["weak_streak"] - 1)

    # --- SCORE LEVEL TAG ---
    if score < 2.5:
        level = "🟢 Stable"
    elif score < 4.5:
        level = "🟡 Watch"
    elif score < 6.5:
        level = "🟠 Weak"
    else:
        level = "🔴 Critical"

    if debug:
        print(f"⏳ {ticker}: Weak {info['weak_streak']}/3 — Score={score:.1f} ({level})")
        print(f"🧮 DEBUG {ticker}: Reasons={reasons}")

    # --- STANDARD SELL CONDITIONS ---
    if score >= 6.5:
        return True, (
            f"🔴 **Critical breakdown confirmed!**\n"
            f"Score: {score:.1f} ({level}) — trend has reversed."
        ), current_price, score

    if score >= 4.5 and info["weak_streak"] >= 3:
        return True, (
            f"🟠 **Sustained weakness detected.**\n"
            f"Streak: {info['weak_streak']}/3 | Score: {score:.1f} — selling to preserve gains."
        ), current_price, score

    if pnl_pct is not None and pnl_pct >= 20 and drop_from_peak >= 7 and score >= 3.5 and info["weak_streak"] >= 2:
        return True, (
            f"💰 **Protecting profit — trend cooling off.**\n"
            f"Gain: +{pnl_pct:.1f}% | Drop: {drop_from_peak:.1f}% | Score: {score:.1f}"
        ), current_price, score

    if pnl_pct is not None and pnl_pct >= 10 and score >= 5.0 and info["weak_streak"] >= 2:
        return True, (
            f"🏁 **Momentum reversal confirmed.**\n"
            f"Gain: +{pnl_pct:.1f}% | Score: {score:.1f} — exiting early."
        ), current_price, score

    return False, (
        f"🟢 **Holding steady.** Score: {score:.1f}, Weak {info['weak_streak']}/3 ({level})"
    ), current_price, score



# ---------------------------
# Runner
# ---------------------------
def run_decision_engine(test_mode=False, end_of_day=False):
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"

    # ✅ Load tracker first to avoid UnboundLocalError
    tracker = load_tracker()

    # ✅ Reset daily alert flag if date changed
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if tracker.get("date") != today_str:
        print(f"🧹 Resetting daily alert flag for {today_str}")
        tracker["date"] = today_str
        tracker["had_alerts"] = False

    tracked = load_data(file_to_load)
    if not tracked or "stocks" not in tracked:
        print(f"⚠️ No tracked stocks found in {file_to_load}")
        return

    stocks = tracked["stocks"]
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

        # Collect “Weak” ones only for currently tracked stocks
        if info_state.get("weak_streak", 0) in [1, 2]:
            weak_near.append(
                f"⚠️ {ticker}: Weak {info_state['weak_streak']}/3 | Score {score:.1f}"
            )

        # --- SELL ALERT LOGIC ---
        if decision:
            now_utc = datetime.utcnow()
            send_alert = True
            last_alert = info_state.get("last_alert_time")
            last_score = info_state.get("last_score", 0)

            if last_alert:
                last_alert_dt = datetime.strptime(last_alert, "%Y-%m-%dT%H:%M:%SZ")
                hours_since = (now_utc - last_alert_dt).total_seconds() / 3600
                if hours_since < 3 and abs(score - last_score) < 0.5:
                    send_alert = False  # avoid spamming duplicates

            if send_alert:
                info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                info_state["last_score"] = score

                # More readable Discord alert format
                alert_msg = (
                    f"📊 **[{ticker}] SELL SIGNAL TRIGGERED**\n"
                    f"{reason}\n"
                    f"💰 Profit: {pnl_pct:+.2f}% | Score: {score:.1f} | Weak: {info_state['weak_streak']}/3\n"
                    f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )
                sell_alerts.append(alert_msg)

    # --- CLEAN OLD TICKERS (REMOVE weak streaks for sold stocks) ---
    current_tickers = set(stocks.keys())
    tracker["tickers"] = {k: v for k, v in tracker["tickers"].items() if k in current_tickers}

    # --- Discord Output Logic ---
    now_utc = datetime.utcnow()

    # 🚨 SELL ALERTS
    if sell_alerts:
        msg = "🚨 **SELL ALERTS TRIGGERED** 🚨\n\n" + "\n\n".join(sell_alerts)
        if weak_near:
            msg += "\n\n📉 **Approaching Weak Signals:**\n" + "\n".join(weak_near)

        if not test_mode:
            max_len = 1900
            chunks = [msg[i:i + max_len] for i in range(0, len(msg), max_len)]
            for chunk in chunks:
                send_discord_alert(chunk)
            tracker["had_alerts"] = True

    # 🕐 End-of-Day Summary (Always send once daily)
    elif end_of_day and not test_mode:
        weak_list = []
        for ticker, state in tracker["tickers"].items():
            if state.get("weak_streak", 0) > 0:
                weak_list.append(
                    f"⚠️ {ticker}: Weak {state['weak_streak']}/3 | Score {state.get('last_score', 0):.1f}"
                )

        if weak_list:
            summary = (
                "📊 **Daily Monitoring Update**\n\n"
                + "\n".join(weak_list)
                + f"\n\n🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
            max_len = 1900
            chunks = [summary[i:i + max_len] for i in range(0, len(summary), max_len)]
            for chunk in chunks:
                send_discord_alert(chunk)

        else:
            send_discord_alert(
                f"😎 All systems stable. No sell signals today.\n"
                f"🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
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
