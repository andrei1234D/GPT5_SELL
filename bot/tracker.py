import json
import os
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data.json')

def load_data():
    if not os.path.exists(DATA_PATH):
        print("📂 No existing data.json found, creating new one")
        return {}
    with open(DATA_PATH, 'r') as f:
        try:
            data = json.load(f)
            print(f"📊 Loaded data: {data}")
            return data
        except json.JSONDecodeError:
            print("⚠️ data.json is corrupted. Resetting.")
            return {}

def save_data(data):
    with open(DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Saved data: {data}")

def add_stock(ticker, buy_price):
    data = load_data()
    data[ticker] = {
        "buy_price": buy_price,
        "active": True,
        "added_at": datetime.utcnow().isoformat()
    }
    save_data(data)
    print(f"📈 Tracking {ticker} at ${buy_price:.2f}")
