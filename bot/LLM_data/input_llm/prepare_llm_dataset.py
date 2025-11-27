import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime

OUTPUT_DIR = "bot/LLM_data/input_llm"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llm_input_latest.csv")
INPUT_FILE = "bot/data.json"

# =========================================================
# GIT COMMIT
# =========================================================
def commit_to_repo(files, message):
    try:
        subprocess.run(["git", "add"] + files, check=False)
        subprocess.run(["git", "config", "--global", "user.name", "GitHub Actions Bot"])
        subprocess.run(["git", "config", "--global", "user.email", "bot@github.actions"])
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✅ Committed & pushed successfully.")
    except Exception as e:
        print(f"⚠️ Git commit failed: {e}")

# =========================================================
# TECHNICAL FEATURES
# =========================================================
def compute_technical_features(ticker):
    """Compute the 24 final model features for one ticker."""
    try:
        df = yf.download(ticker, period="200d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            print(f"⚠️ No data for {ticker}.")
            return None

        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Raw base
        df["avg_high_raw"] = df["High"]
        df["avg_low_raw"] = df["Low"]
        df["Year"] = df.index.year

        # Moving averages
        for w in [20, 50, 200]:
            df[f"SMA{w}"] = df["Close"].rolling(w).mean()
            df[f"EMA{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

        # RSI (14)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # ATR
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR%"] = (df["ATR"] / df["Close"]) * 100

        # Volatility & Momentum
        df["Volatility"] = (df["High"] - df["Low"]) / df["Low"]
        df["Momentum"] = df["Close"].pct_change(3)
        df["OBV"] = np.where(df["Close"] > df["Close"].shift(),
                             df["Volume"], -df["Volume"]).cumsum()

        # Derived features
        df["volatility_30"] = (
            (df["High"].rolling(30).mean() - df["Low"].rolling(30).mean())
            / (df["Low"].rolling(30).mean() + 1e-9)
        )

        df["MarketTrend_enc"] = np.where(
            df["Close"] > df["SMA50"], 1,
            np.where(df["Close"] < df["SMA50"], -1, 0)
        )

        df["range_position_30"] = (
            (df["Close"] - df["Low"].rolling(30).min()) /
            (df["High"].rolling(30).max() - df["Low"].rolling(30).min() + 1e-9)
        ).clip(0, 1).fillna(0.5)

        df["momentum_3"] = df["Close"].pct_change(3).replace([np.inf, -np.inf], 0).fillna(0)
        df["vol_regime_ratio"] = (
            df["volatility_30"] / (df["volatility_30"].rolling(90, min_periods=5).mean() + 1e-9)
        ).clip(0, 5).fillna(1)

        df["current_price"] = df["Close"]

        # Take only final row and restrict to expected 24 features
        latest = df.iloc[-1].to_dict()
        keys = [
            "avg_high_raw", "avg_low_raw", "SMA20", "SMA50", "SMA200",
            "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "MACD_signal", "MACD_hist",
            "ATR", "ATR%", "Volatility", "Momentum", "OBV", "Year", "volatility_30",
            "current_price", "MarketTrend_enc", "range_position_30", "momentum_3", "vol_regime_ratio"
        ]
        clean = {k: latest.get(k, 0) for k in keys}
        return clean

    except Exception as e:
        print(f"⚠️ Failed {ticker}: {e}")
        return None

# =========================================================
# MAIN
# =========================================================
def prepare_llm_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"⚠️ Missing {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    stocks = data.get("stocks", {})

    rows = []
    print(f"📊 Building LLM-ready dataset for {len(stocks)} tickers...")

    for ticker, info in stocks.items():
        res = compute_technical_features(ticker)
        if res is None:
            continue

        avg_price = info.get("avg_price", 0)
        shares = info.get("shares", 0)
        invested_lei = info.get("invested_lei", 0)
        current_price = res["current_price"]

        pnl_lei = current_price * shares * 4.6 - invested_lei
        pnl_pct = (pnl_lei / invested_lei * 100) if invested_lei else 0

        res.update({
            "Ticker": ticker,
            "avg_price": avg_price,
            "shares": shares,
            "invested_lei": invested_lei,
            "pnl_pct": pnl_pct,
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        rows.append(res)
        print(f"✅ {ticker}: current_price={current_price:.2f}, pnl={pnl_pct:.2f}%")

    df = pd.DataFrame(rows)
    print("🧾 Final columns in dataset:", df.columns.tolist())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dataset saved → {OUTPUT_FILE}")

    commit_to_repo(
        [OUTPUT_FILE],
        f"🤖 Auto-update LLM dataset [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
    )

if __name__ == "__main__":
    prepare_llm_dataset()
