import os
import requests
import json

def send_discord_alert(message: str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("⚠️ No DISCORD_WEBHOOK_URL found in environment.")
        return

    payload = {"content": message}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload),
                                 headers={"Content-Type": "application/json"})
        if response.status_code in (200, 204):
            print("✅ Discord alert sent successfully.")
        else:
            print(f"⚠️ Discord webhook returned {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Discord alert failed: {e}")
