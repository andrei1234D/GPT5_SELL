import os
import requests
import json
from datetime import datetime

def send_discord_alert(
    ticker: str = None,
    buy_price: float = None,
    current_price: float = None,
    reason: str = None,
    debug: bool = True,
    test_mode: bool = False,
    message: str = None,
    bundled_alerts: list = None
):
    """
    Sends either:
    - A structured single-stock alert
    - A bundled multi-stock alert (list of dicts)
    - A raw message
    """

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("❌ ERROR: DISCORD_WEBHOOK_URL is not set. Add it as a GitHub Secret.")
        return False

    # --- CASE 1: Raw text message ---
    if message:
        payload = {"content": message}

    # --- CASE 2: Multiple tickers in one embed ---
    elif bundled_alerts:
        fields = []
        for alert in bundled_alerts:
            buy_price = alert.get("buy_price")
            current_price = alert.get("current_price")

            # ✅ Safe profit/loss calculation
            if buy_price and current_price:
                pnl_pct = ((current_price - buy_price) / buy_price) * 100
                pnl_text = f"{pnl_pct:+.2f}%"
            else:
                pnl_text = "N/A"

            fields.append({
                "name": f"{alert.get('ticker', '?')} ({pnl_text})",
                "value": (
                    f"Buy: ${buy_price:.2f}\nCurrent: ${current_price:.2f}\n"
                    f"Reason: {alert.get('reason', '—')}"
                    if (buy_price and current_price)
                    else f"Reason: {alert.get('reason', '—')}"
                ),
                "inline": False
            })

        payload = {
            "embeds": [
                {
                    "title": f"🚨 SELL ALERTS ({len(bundled_alerts)})",
                    "description": "Multiple stocks triggered sell conditions.",
                    "color": 15158332,  # Red
                    "fields": fields,
                    "footer": {
                        "text": f"Bot check at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                    }
                }
            ]
        }

    # --- CASE 3: Default single-stock alert ---
    else:
        # ✅ Safe check for None before computing PnL
        if buy_price and current_price:
            profit_loss_pct = ((current_price - buy_price) / buy_price) * 100
            pnl_text = f"{profit_loss_pct:+.2f}%"
        else:
            pnl_text = "N/A"

        payload = {
            "embeds": [
                {
                    "title": f"🚨 SELL SIGNAL for {ticker}",
                    "color": 15158332,  # Red
                    "fields": [
                        {"name": "Ticker", "value": ticker or "?", "inline": True},
                        {"name": "Buy Price", "value": f"${buy_price or 0:.2f}", "inline": True},
                        {"name": "Current Price", "value": f"${current_price or 0:.2f}", "inline": True},
                        {"name": "PnL %", "value": pnl_text, "inline": True},
                        {"name": "Reason", "value": reason or "—", "inline": False}
                    ],
                    "footer": {
                        "text": f"Bot check at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                    }
                }
            ]
        }

    # --- TEST MODE ---
    if test_mode:
        print("🧪 TEST MODE ALERT (no Discord sent):")
        print(json.dumps(payload, indent=2))
        return True

    # --- Send to Discord ---
    try:
        response = requests.post(webhook_url, json=payload)

        if debug:
            print("📤 Sending Discord Alert...")
            print(f"📡 Response Code: {response.status_code}")
            if response.status_code not in (200, 204):
                print(f"📡 Response Body: {response.text}")

        if response.status_code in (200, 204):
            print("✅ Successfully sent alert to Discord")
            return True
        else:
            print("⚠️ Failed to send alert to Discord")
            return False

    except Exception as e:
        print(f"❌ Exception while sending message: {e}")
        return False


# --- Local test run ---
if __name__ == "__main__":
    send_discord_alert(
        bundled_alerts=[
            {"ticker": "AAPL", "buy_price": 172.50, "current_price": 195.20, "reason": "🎯 Take Profit (+13%)"},
            {"ticker": "MSFT", "buy_price": 315.00, "current_price": 280.00, "reason": "🛑 Stop Loss (-11%)"},
        ],
        test_mode=True
    )
