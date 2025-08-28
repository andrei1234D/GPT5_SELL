import argparse
import yfinance as yf
from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators

from datetime import datetime
import csv
import os
import json

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
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"date": datetime.utcnow().strftime("%Y-%m-%d"), "had_alerts": False}


def save_tracker(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f)


# ---------------------------
# Core Logic
# ---------------------------
def check_sell_conditions(ticker: str, buy_price: float, current_price: float,
                          volume=None, momentum=None, rsi=None, market_trend=None,
                          ma50=None, ma200=None, atr=None,
                          macd=None, macd_signal=None,
                          bb_upper=None, bb_lower=None,
                          resistance=None, support=None,
                          debug=True):

    pnl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0

    # Hard Stop-Loss
    if pnl_pct <= -25:
        if rsi and rsi < 35:
            return False, f"📈 Oversold at {rsi}, deep loss but HOLD.", current_price
        if momentum and momentum >= 0:
            return False, f"📈 Momentum stabilizing despite loss → HOLD.", current_price
        if market_trend == "BULLISH":
            return False, f"📈 Market bullish, avoid panic selling at deep loss.", current_price
        return True, f"🛑 Smart Stop Loss Triggered (-25%).", current_price

    # Score-based logic
    score = 0
    reasons = []

    if momentum is not None and momentum < -0.5:
        score += 2
        reasons.append("📉 Strong Negative Momentum (< -0.5)")
    if macd is not None and macd_signal is not None and macd < macd_signal:
        score += 1
        reasons.append("📉 MACD Bearish Crossover")
    if rsi is not None and rsi > 70:
        score += 1
        reasons.append("📉 RSI Overbought (>70)")
    if ma50 is not None and current_price < ma50:
        score += 1
        reasons.append("📉 Price below MA50")
    if ma200 is not None and current_price < ma200:
        score += 1
        reasons.append("📉 Price below MA200")
    if atr is not None and atr > 7 and pnl_pct < -10:
        score += 1
        reasons.append("⚡ High ATR + Loss")
    if bb_upper is not None and current_price > bb_upper:
        score += 1
        reasons.append("📉 Price above Upper Bollinger Band")

     # Profit-taking safeguard
    if pnl_pct >= 23 and score >= 3:
        reason_text = " | ".join(reasons)
        return True, f"🎯 Profit target reached (+23%) with weakening signals (score {score}): {reason_text}", current_price

    # Normal score-based exit (≥4 signals)
    if score > 3:
        reason_text = " | ".join(reasons)
        return True, f"🚨 Score {score} (≥4) triggered SELL: {reason_text}", current_price

    return False, "Hold — no sell signal", current_price


def run_decision_engine(test_mode=False, end_of_day=False):
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracked = load_data(file_to_load)

    if not tracked or "stocks" not in tracked:
        print(f"⚠️ No tracked stocks found in {file_to_load}")
        return

    stocks = tracked["stocks"]
    tracker = load_tracker()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    if tracker["date"] != today:
        tracker = {"date": today, "had_alerts": False}

    total = len(stocks)
    correct = 0
    sell_alerts = []

    csv_file = "bot/test_results.csv" if test_mode else "bot/live_results.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    usd_to_lei = get_usd_to_lei()

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ticker", "Avg Price (USD)", "Shares", "Invested (LEI)", "Current Price (USD)",
            "Current Value (LEI)", "PnL %", "PnL (LEI)",
            "Decision", "Reason", "Expected",
            "Volume", "Momentum", "RSI", "Market Trend", "MA50", "MA200", "ATR",
            "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "Resistance", "Support"
        ])

        for ticker, info in stocks.items():
            avg_price = float(info.get("avg_price", 0))   # USD
            invested_lei = float(info.get("invested_lei", 0))  # LEI
            shares = float(info.get("shares", 0))

            if avg_price <= 0 or invested_lei <= 0 or shares <= 0:
                print(f"⚠️ Skipping {ticker}, invalid data")
                continue

            indicators = compute_indicators(ticker)
            if not indicators:
                print(f"⚠️ Skipping {ticker}, no indicators fetched")
                continue

            current_price = indicators["current_price"]  # USD
            current_value_usd = current_price * shares
            current_value_lei = current_value_usd * usd_to_lei

            pnl_lei = current_value_lei - invested_lei
            pnl_pct = (pnl_lei / invested_lei) * 100 if invested_lei > 0 else 0
            info.update(indicators)

            expected = info.get("expected", "HOLD")

            decision, reason, _ = check_sell_conditions(
                ticker, avg_price, current_price,
                volume=info.get("volume"),
                momentum=info.get("momentum"),
                rsi=info.get("rsi"),
                market_trend=info.get("market_trend"),
                ma50=info.get("ma50"),
                ma200=info.get("ma200"),
                atr=info.get("atr"),
                macd=info.get("macd"),
                macd_signal=info.get("macd_signal"),
                bb_upper=info.get("bb_upper"),
                bb_lower=info.get("bb_lower"),
                resistance=info.get("resistance"),
                support=info.get("support"),
                debug=test_mode
            )

            decision_text = "SELL" if decision else "HOLD"

            writer.writerow([
                ticker, f"{avg_price:.2f}", f"{shares:.2f}", f"{invested_lei:.2f}", f"{current_price:.2f}",
                f"{current_value_lei:.2f}", f"{pnl_pct:.2f}%", f"{pnl_lei:.2f}",
                decision_text, reason, expected,
                info.get("volume"), info.get("momentum"), info.get("rsi"), info.get("market_trend"),
                info.get("ma50"), info.get("ma200"), info.get("atr"),
                info.get("macd"), info.get("macd_signal"),
                info.get("bb_upper"), info.get("bb_lower"),
                info.get("resistance"), info.get("support")
            ])

            if decision_text == expected:
                correct += 1

            if decision:
                alert_line = (
                    f"**{ticker}**\n"
                    f"Avg Price: {avg_price:.2f} USD\n"
                    f"Shares: {shares:.2f}\n"
                    f"Invested: {invested_lei:.2f} LEI\n"
                    f"Current Value: {current_value_lei:.2f} LEI\n"
                    f"PnL: {pnl_lei:+.2f} LEI ({pnl_pct:.2f}%)\n"
                    f"Reason: {reason}\n"
                )
                sell_alerts.append(alert_line)

    if sell_alerts and not test_mode and not end_of_day:
        full_message = "🚨 **SELL ALERTS TRIGGERED** 🚨\n\n" + "\n---\n".join(sell_alerts)
        send_discord_alert(message=full_message)
        tracker["had_alerts"] = True

    if end_of_day and not tracker["had_alerts"] and not test_mode:
        send_discord_alert(message="😎 No stocks to sell today. Business doing good, boss!")

    save_tracker(tracker)

    print(f"✅ Decision Engine Run Complete at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"📂 Results saved to {csv_file}")

    if test_mode:
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"🧪 TEST SUMMARY: {accuracy:.1f}% {'✅ PASS' if accuracy >= 85 else '❌ FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--endofday", action="store_true")
    args = parser.parse_args()

    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)
