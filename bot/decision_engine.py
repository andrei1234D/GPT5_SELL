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

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"date": datetime.utcnow().strftime("%Y-%m-%d"), "had_alerts": False}

def save_tracker(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f)

def check_sell_conditions(ticker: str, buy_price: float, current_price: float,
                          volume=None, momentum=None, rsi=None, market_trend=None,
                          ma50=None, ma200=None, atr=None,
                          macd=None, macd_signal=None,
                          bb_upper=None, bb_lower=None,
                          resistance=None, support=None,
                          debug=True):

    pnl_pct = ((current_price - buy_price) / buy_price) * 100

    # --- Smarter Stop Loss ---
    if pnl_pct <= -25:
        if rsi and rsi < 35:
            return False, f"📈 Oversold at {rsi}, deep loss but HOLD (recovery likely).", current_price
        if momentum and momentum >= 0:
            return False, f"📈 Momentum stabilizing despite loss → HOLD.", current_price
        if market_trend == "BULLISH":
            return False, f"📈 Market bullish, avoid panic selling at deep loss.", current_price

        return True, f"🛑 Smart Stop Loss Triggered (-25%) with weakness confirmed.", current_price

    if rsi and rsi > 75:
        return True, f"📉 RSI Extreme Overbought (>75).", current_price
    if rsi and rsi < 30:
        return False, f"📈 RSI Oversold (<30) → HOLD.", current_price
    if momentum and momentum < -0.5:
        return True, f"📉 Strong Negative Momentum (< -0.5).", current_price
    if market_trend == "BEARISH" and pnl_pct < -5:
        return True, f"📉 Bearish Market + Loss → SELL.", current_price
    if ma200 and current_price < ma200 and momentum and momentum < -0.3:
        return True, f"📉 Below MA200 + Weak Momentum.", current_price
    if ma50 and ma200 and ma50 < ma200 and pnl_pct < -5:
        return True, f"📉 Death Cross + Small Loss.", current_price
    if atr and atr > 7 and pnl_pct < -15:
        return True, f"⚡ High Volatility (ATR>7) with Loss.", current_price
    if macd and macd_signal and macd < macd_signal and rsi and rsi > 65:
        return True, f"📉 MACD Bearish Crossover + RSI Overbought.", current_price
    if bb_upper and current_price > bb_upper and rsi and rsi > 70:
        return True, f"📉 Price Above Bollinger Upper Band + RSI Overbought.", current_price
    if bb_lower and current_price < bb_lower and rsi and rsi < 35:
        return False, f"📈 Price Near Bollinger Lower Band + RSI Oversold → HOLD.", current_price
    if resistance and current_price >= resistance * 0.99 and rsi and rsi > 65:
        return True, f"📉 Near Resistance + RSI Overbought.", current_price
    if support and current_price <= support * 1.01 and pnl_pct < -5:
        return True, f"📉 Broke Support ({support}).", current_price

    return False, "Hold — no sell signal", current_price


def run_decision_engine(test_mode=False, end_of_day=False):
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracked = load_data(file_to_load)
    indicators = compute_indicators(ticker)

    if indicators:
        current_price = indicators["current_price"]
        info.update(indicators)  # inject MA, RSI, MACD, ATR, etc.


    if not tracked:
        print(f"⚠️ No tracked stocks found in {file_to_load}")
        return

    tracker = load_tracker()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # reset tracker if new day
    if tracker["date"] != today:
        tracker = {"date": today, "had_alerts": False}

    total = len(tracked)
    correct = 0
    sell_alerts = []

    csv_file = "bot/test_results.csv" if test_mode else "bot/live_results.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ticker", "Buy Price", "Current Price", "PnL %", "Decision", "Reason", "Expected",
            "Volume", "Momentum", "RSI", "Market Trend", "MA50", "MA200", "ATR",
            "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "Resistance", "Support"
        ])

        for ticker, info in tracked.items():
            if not info.get("active", True):
                continue

            buy_price = float(info["buy_price"])
            current_price = float(info.get("current_price", 0) or 0)

            # ✅ Fetch from Yahoo Finance if missing/invalid
            if not current_price or current_price <= 0:
                try:
                    ticker_data = yf.Ticker(ticker)
                    hist = ticker_data.history(period="1d")
                    if not hist.empty:
                        current_price = hist["Close"].iloc[-1]
                        print(f"💹 Fetched live price for {ticker}: {current_price}")
                    else:
                        print(f"⚠️ No price data found for {ticker}, defaulting to buy_price")
                        current_price = buy_price
                except Exception as e:
                    print(f"❌ Error fetching price for {ticker}: {e}")
                    current_price = buy_price
            expected = info.get("expected", "HOLD")

            # 🔎 Debug: Dump all values
            print("\n🔎 DEBUG DATA DUMP ---------------------------")
            print(f"Ticker: {ticker}")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print(f"  current_price (yf): {current_price}")
            print("------------------------------------------------")

            decision, reason, _ = check_sell_conditions(
                ticker, buy_price, current_price,
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

            pnl_pct = ((current_price - buy_price) / buy_price) * 100
            decision_text = "SELL" if decision else "HOLD"

            writer.writerow([
                ticker, f"{buy_price:.2f}", f"{current_price:.2f}", f"{pnl_pct:.2f}",
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
                    f"Buy Price: ${buy_price:.2f}\n"
                    f"Current Price: ${current_price:.2f}\n"
                    f"PnL %: {pnl_pct:.2f}%\n"
                    f"Reason: {reason}\n"
                )
                sell_alerts.append(alert_line)

    # HOURLY: Send alerts if any
    if sell_alerts and not test_mode and not end_of_day:
        full_message = "🚨 **SELL ALERTS TRIGGERED** 🚨\n\n" + "\n---\n".join(sell_alerts)
        send_discord_alert(message=full_message)
        tracker["had_alerts"] = True

    # END OF DAY: send quirky message if no alerts all day
    if end_of_day and not tracker["had_alerts"] and not test_mode:
        send_discord_alert(message="😎 No stocks to sell today. Business doing good, boss!")

    save_tracker(tracker)

    print(f"✅ Decision Engine Run Complete at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"📂 Results saved to {csv_file}")

    if test_mode:
        accuracy = (correct / total) * 100 if total > 0 else 0
        print("🧪 TEST SUMMARY -----------------------------")
        print(f"Total Stocks Tested: {total}")
        print(f"Correct Decisions: {correct}/{total}")
        print(f"Accuracy: {accuracy:.1f}% {'✅ PASS' if accuracy >= 85 else '❌ FAIL'}")
        print("🧪 ------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with test_data.csv (no Discord spam)")
    parser.add_argument("--endofday", action="store_true", help="Send quirky message if no alerts today")
    args = parser.parse_args()

    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)
