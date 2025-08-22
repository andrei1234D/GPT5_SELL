import os
import requests
import json
from datetime import datetime

def send_discord_message(message: str, debug: bool = True):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("❌ ERROR: DISCORD_WEBHOOK_URL is not set. Add it as a GitHub Secret.")
        return False

    payload = {
        "content": message
    }

    try:
        response = requests.post(webhook_url, json=payload)

        if debug:
            print("📤 Sending Discord Message...")
            print(f"🔗 Webhook URL: {webhook_url[:50]}... (truncated for security)")
            print(f"📦 Payload: {json.dumps(payload, indent=2)}")
            print(f"📡 Response Code: {response.status_code}")
            print(f"📡 Response Body: {response.text}")

        if response.status_code == 204:
            print("✅ Successfully sent message to Discord")
            return True
        else:
            print("⚠️ Failed to send message to Discord")
            return False

    except Exception as e:
        print(f"❌ Exception while sending message: {e}")
        return False


if __name__ == "__main__":
    # 🔎 Test run
    test_message = f"🚀 Test Notification from bot at {datetime.utcnow().isoformat()} UTC"
    send_discord_message(test_message)
