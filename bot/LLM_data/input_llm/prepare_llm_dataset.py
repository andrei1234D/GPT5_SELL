import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime

OUTPUT_DIR = "LLM_data/input_llm"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llm_input_latest.csv")
INPUT_FILE = "bot/data.json"


# ===============================
# Helper: Safe Git Commit
# ===============================
def commit_to_repo(files, message):
    try:
        subprocess.run(["git", "pull", "--rebase"], check=False)
        subprocess.run(["git", "add"] + files, check=False)
        subprocess.run(["git", "config", "--global", "user.name", "GitHub Actions Bot"])
        subprocess.run(["git", "config", "--global", "user.email", "bot@github.actions"])
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✅ Committed & pushed successfully.")
    except Exception as e:
        print(f"⚠️ Git commit failed: {e}")


# ===============================
# Feature Builders
# ===============================
def compute_technical_features(ticker):
    """Compute modern LLM-ready technical indicators for one ticker."""
    try:
        # ✅ Explicitly set auto_adjust=False to avoid MultiIndex columns
        df = yf.download(ticker, period="200d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            print(f"⚠️ No data for {ticker}.")
            return None

        # ✅ Flatten MultiIndex if present (new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # --- Base columns ---
        for col in ["High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                print(f"⚠️ Missing {col} for {ticker}")
                return None

        df["avg_high_raw"] = df["High"]
        df["avg_low_raw"] = df["Low"]
        df["Year"] = df.index.year

        # --- Moving Averages ---
        for w in [20, 50, 200]:
            df[f"SMA{w}"] = df["Close"].rolling(w).mean()
            df[f"EMA{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

        # --- RSI (14-day) ---
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI14"] = 100 - (100 / (1 + rs))

        # --- MACD ---
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # --- ATR (14-day) ---
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr_series = tr.rolling(14).mean()

        # ✅ Ensure Series (fix for multiple column issue)
        if isinstance(atr_series, pd.DataFrame):
            atr_series = atr_series.iloc[:, 0]

        df["ATR"] = atr_series
        df["ATR%"] = (df["ATR"] / df["Close"]) * 100

        # --- Volatility & Momentum ---
        df["Volatility"] = (df["High"] - df["Low"]) / df["Low"]
        df["Momentum"] = df["Close"].pct_change(3)

        # --- OBV ---
        df["OBV"] = np.where(df["Close"] > df["Close"].shift(),
                             df["Volume"], -df["Volume"]).cumsum()

        # --- Derived ratios ---
        df["volatility_30"] = (
            df["High"].rolling(30).mean() - df["Low"].rolling(30).mean()
        ) / (df["Low"].rolling(30).mean() + 1e-9)

        df["range_position_30"] = (
            (df["Close"] - df["Low"].rolling(30).min())
            / (df["High"].rolling(30).max() - df["Low"].rolling(30).min() + 1e-9)
        )

        df["momentum_3"] = df["Close"].pct_change(3)
        df["vol_regime_ratio"] = df["Volatility"] / (df["Volatility"].rolling(30).mean() + 1e-9)

        # --- Market Trend encoding ---
        df["MarketTrend_enc"] = np.where(
            df["Close"] > df["SMA50"], 1,
            np.where(df["Close"] < df["SMA50"], -1, 0)
        )

        latest = df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0)
        return latest

    except Exception as e:
        print(f"⚠️ Failed {ticker}: {e}")
        return None


# ===============================
# Main builder
# ===============================
def prepare_llm_dataset():
    """Build dataset for all portfolio tickers directly from yfinance."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"⚠️ Missing {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    stocks = data.get("stocks", {})
    if not stocks:
        print("⚠️ No tickers found.")
        return

    rows = []
    print(f"📊 Fetching and computing LLM features for {len(stocks)} tickers...")

    for ticker, info in stocks.items():
        res = compute_technical_features(ticker)
        if res is None:
            continue

        avg_price = info.get("avg_price", 0)
        shares = info.get("shares", 0)
        invested_lei = info.get("invested_lei", 0)
        current_price = res["Close"]

        pnl_lei = current_price * shares * 4.6 - invested_lei
        pnl_pct = (pnl_lei / invested_lei * 100) if invested_lei else None

        row = {
            "Ticker": ticker,
            "avg_price": avg_price,
            "shares": shares,
            "invested_lei": invested_lei,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        feature_cols = [
            "avg_high_raw", "avg_low_raw", "SMA20", "SMA50", "SMA200",
            "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "MACD_signal", "MACD_hist",
            "ATR", "ATR%", "Volatility", "Momentum", "OBV", "Year", "volatility_30",
            "MarketTrend_enc", "range_position_30", "momentum_3", "vol_regime_ratio"
        ]
        for col in feature_cols:
            row[col] = res.get(col, None)

        rows.append(row)
        print(f"✅ {ticker}: current_price={current_price:.2f}, pnl={pnl_pct:.2f}%")

    if not rows:
        print("⚠️ No valid data collected — nothing to save.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dataset saved → {OUTPUT_FILE}")

    commit_to_repo(
        [OUTPUT_FILE],
        f"🤖 Auto-update LLM dataset [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
    )

    return df


if __name__ == "__main__":
    prepare_llm_dataset()
