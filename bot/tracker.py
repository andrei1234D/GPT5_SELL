import json
import csv

def load_data(file_path="bot/data.json"):
    """
    Load stock data from JSON or CSV file.
    Returns a dictionary with ticker symbols as keys.
    """

    try:
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        elif file_path.endswith(".csv"):
            data = {}
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row["ticker"]
                    data[ticker] = {
                        "buy_price": float(row["buy_price"]),
                        "current_price": float(row.get("current_price", 0) or 0),
                        "expected": row.get("expected", "HOLD").upper(),
                        "volume": int(row.get("volume", 0)),
                        "momentum": float(row.get("momentum", 0)),
                        "rsi": float(row.get("rsi", 50)),
                        "market_trend": row.get("market_trend", "NEUTRAL").upper(),
                        "ma50": float(row.get("ma50", 0)),
                        "ma200": float(row.get("ma200", 0)),
                        "atr": float(row.get("atr", 0)),
                        "macd": float(row.get("macd", 0)),
                        "macd_signal": float(row.get("macd_signal", 0)),
                        "bb_upper": float(row.get("bb_upper", 0)),
                        "bb_lower": float(row.get("bb_lower", 0)),
                        "resistance": float(row.get("resistance", 0)),
                        "support": float(row.get("support", 0)),
                        "active": True
                    }
            return data

        else:
            print(f"⚠️ Unsupported file format: {file_path}")
            return {}

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}
