# bot/build_ml_input_dataset.py
# Builds llm_input_latest.csv in the NEW production format used by MT models.
# NOTE: Date is REMOVED (not in your training schema).

import os
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

OUTPUT_DIR = "bot/LLM_data/input_llm"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llm_input_latest.csv")
INPUT_FILE = "bot/data.json"


# =========================================================
# GIT COMMIT
# =========================================================
def commit_to_repo(files, message):
    try:
        subprocess.run(["git", "add"] + files, check=False)
        subprocess.run(["git", "config", "--global", "user.name", "GitHub Actions Bot"], check=False)
        subprocess.run(["git", "config", "--global", "user.email", "bot@github.actions"], check=False)
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✅ Committed & pushed successfully.")
    except Exception as e:
        print(f"⚠️ Git commit failed: {e}")


# =========================================================
# MATH HELPERS (match training style)
# =========================================================
def safe_div(a, b, eps: float = 1e-12):
    a = a.astype(float)
    b = b.astype(float)
    return a / (b.replace(0.0, np.nan) + eps)


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def cs_rank_series(x: pd.Series) -> pd.Series:
    # percentile rank in [0, 1], higher value => higher rank
    x = pd.to_numeric(x, errors="coerce")
    r = x.rank(method="average", pct=True, ascending=True)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(float)


# =========================================================
# FEATURE ENGINE (time-series, per ticker)
# =========================================================
def _clean_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def compute_features_for_ticker(ticker: str) -> dict | None:
    """
    Produces ONE row (latest) with the schema you requested (WITHOUT Date).
    Cross-sectional columns are NOT computed here (done later across tickers).
    """
    try:
        # Need enough history for 252D and shifts: use multi-year daily data.
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=True,
        )

        if df is None or df.empty:
            print(f"⚠️ No data for {ticker}.")
            return None

        df = _clean_yf_columns(df).copy()

        # Basic sanity
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                print(f"⚠️ Missing {col} for {ticker}.")
                return None

        df = df.dropna(subset=["Close", "High", "Low"]).copy()
        if len(df) < 260:
            print(f"⚠️ Not enough history for {ticker} (rows={len(df)}). Need ~260+.")
            return None

        # --- Core rolling highs/lows ---
        df["High_30D"] = df["High"].rolling(30, min_periods=10).max()
        df["Low_30D"] = df["Low"].rolling(30, min_periods=10).min()
        df["High_52W"] = df["High"].rolling(252, min_periods=50).max()
        df["Low_52W"] = df["Low"].rolling(252, min_periods=50).min()

        # --- SMA20 needed by SMA_Slope_3M ---
        df["SMA20"] = df["Close"].rolling(20, min_periods=10).mean()

        # --- Volatility_30D / Volatility_252D (rolling mean range / rolling mean low) ---
        low30_mean = df["Low"].rolling(30, min_periods=10).mean()
        high30_mean = df["High"].rolling(30, min_periods=10).mean()
        df["Volatility_30D"] = safe_div(high30_mean - low30_mean, low30_mean + 1e-9)

        low252_mean = df["Low"].rolling(252, min_periods=50).mean()
        high252_mean = df["High"].rolling(252, min_periods=50).mean()
        df["Volatility_252D"] = safe_div(high252_mean - low252_mean, low252_mean + 1e-9)

        Close = df["Close"]
        High52 = df["High_52W"]
        Low52 = df["Low_52W"]
        High30 = df["High_30D"]
        Low30 = df["Low_30D"]
        Vol30 = df["Volatility_30D"]
        Vol252 = df["Volatility_252D"]

        # --- Momentum measures ---
        df["Momentum_63D"] = safe_div(Close, Close.shift(63))
        df["Momentum_126D"] = safe_div(Close, Close.shift(126))
        df["Momentum_252D"] = safe_div(Close, Close.shift(252))

        # --- AMT ---
        df["AMT"] = np.where(df["Momentum_126D"] > 1.1, df["Momentum_252D"], 0.0)

        # --- SMC ---
        smc_part1 = safe_div(df["Momentum_252D"], Vol30)
        smc_part2 = safe_div(Close, High52)
        df["SMC"] = smc_part1 * smc_part2

        # --- TSS ---
        df["TSS"] = (df["Momentum_63D"] + df["Momentum_126D"] + df["Momentum_252D"]) / 3.0

        # --- ABS ---
        range_52 = (High52 - Low52)
        abs_ratio = safe_div(Close - Low52, range_52)
        df["ABS"] = abs_ratio * Vol30

        # --- VAM ---
        df["VAM"] = safe_div(df["Momentum_63D"], 1.0 + Vol30)

        # --- RSE ---
        ret = Close.pct_change(fill_method=None)
        roll_mean = ret.rolling(63, min_periods=20).mean()
        roll_std = ret.rolling(63, min_periods=20).std()
        df["RSE"] = safe_div(roll_mean, roll_std + 1e-12)

        # --- CBP ---
        df["CBP"] = safe_div(Vol30, Vol252 + 1e-12)

        # --- SMA slope 3M ---
        sma_past = df["SMA20"].shift(63)
        df["SMA_Slope_3M"] = safe_div(df["SMA20"] - sma_past, sma_past + 1e-12)

        # --- Returns ---
        df["Ret_1D"] = ret
        df["Ret_5D"] = Close.pct_change(5)

        # --- Range position ---
        range_30 = (High30 - Low30)
        df["pos_52w"] = safe_div(Close - Low52, range_52 + 1e-12)
        df["pos_30d"] = safe_div(Close - Low30, range_30 + 1e-12)

        # --- Volume SMA20 ---
        df["Volume_SMA20"] = df["Volume"].rolling(20, min_periods=5).mean()

        # --- Extra aggregates ---
        df["avg_close_past_3_days"] = Close.rolling(3, min_periods=1).mean()
        df["avg_volatility_30D"] = df["Volatility_30D"].rolling(3, min_periods=1).mean()
        df["current_price"] = Close

        # --- MarketTrend regime from Close vs SMA50 with neutral band ---
        sma50 = Close.rolling(50, min_periods=20).mean()
        band = 0.002  # 0.2% neutral band
        mt = np.where(Close > sma50 * (1 + band), 1, np.where(Close < sma50 * (1 - band), -1, 0))
        df["MarketTrend"] = mt.astype(int)

        # Latest row
        last = df.iloc[-1].copy()

        out = {
            "Ticker": ticker,
            "MarketTrend": int(last.get("MarketTrend", 0)),
            "SellScore": np.nan,  # not available in production input
            "High_30D": float(last.get("High_30D", np.nan)),
            "Low_30D": float(last.get("Low_30D", np.nan)),
            "High_52W": float(last.get("High_52W", np.nan)),
            "Low_52W": float(last.get("Low_52W", np.nan)),
            "Volatility_30D": float(last.get("Volatility_30D", np.nan)),
            "Volatility_252D": float(last.get("Volatility_252D", np.nan)),
            "Momentum_63D": float(last.get("Momentum_63D", np.nan)),
            "Momentum_126D": float(last.get("Momentum_126D", np.nan)),
            "Momentum_252D": float(last.get("Momentum_252D", np.nan)),
            "SMC": float(last.get("SMC", np.nan)),
            "ABS": float(last.get("ABS", np.nan)),
            "VAM": float(last.get("VAM", np.nan)),
            "RSE": float(last.get("RSE", np.nan)),
            "SMA_Slope_3M": float(last.get("SMA_Slope_3M", np.nan)),
            "Ret_1D": float(last.get("Ret_1D", np.nan)),
            "Ret_5D": float(last.get("Ret_5D", np.nan)),
            "pos_52w": float(last.get("pos_52w", np.nan)),
            "pos_30d": float(last.get("pos_30d", np.nan)),
            "Volume_SMA20": float(last.get("Volume_SMA20", np.nan)),
            "avg_close_past_3_days": float(last.get("avg_close_past_3_days", np.nan)),
            "avg_volatility_30D": float(last.get("avg_volatility_30D", np.nan)),
            "current_price": float(last.get("current_price", np.nan)),

            # internal for cross-sectional calc
            "_AMT": float(last.get("AMT", np.nan)),
            "_CBP": float(last.get("CBP", np.nan)),
            "_TSS": float(last.get("TSS", np.nan)),
        }

        # Robust fill
        for k, v in list(out.items()):
            if k.startswith("_"):
                continue
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                out[k] = 0.0

        return out

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

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    stocks = data.get("stocks", {}) or {}
    print(f"📊 Building NEW-format dataset (no Date) for {len(stocks)} tickers...")

    rows = []
    for ticker, info in stocks.items():
        feats = compute_features_for_ticker(ticker)
        if feats is None:
            continue

        # Portfolio extras (optional)
        avg_price = float(info.get("avg_price", 0) or 0)
        shares = float(info.get("shares", 0) or 0)
        invested_lei = float(info.get("invested_lei", 0) or 0)

        current_price = float(feats.get("current_price", 0.0) or 0.0)
        pnl_lei = current_price * shares * 4.6 - invested_lei
        pnl_pct = (pnl_lei / invested_lei * 100.0) if invested_lei else 0.0

        feats.update({
            "avg_price": avg_price,
            "shares": shares,
            "invested_lei": invested_lei,
            "pnl_pct": float(pnl_pct),
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

        rows.append(feats)
        print(f"✅ {ticker}: price={current_price:.2f}, pnl={pnl_pct:+.2f}%, MT={feats.get('MarketTrend')}")

    if not rows:
        print("⚠️ No rows produced. Nothing to save.")
        return

    df = pd.DataFrame(rows)

    # =========================================================
    # CROSS-SECTIONAL FEATURES (computed across current tickers)
    # =========================================================
    df["Momentum_63D_cs_z"] = zscore_series(df["Momentum_63D"])
    df["Momentum_252D_cs_z"] = zscore_series(df["Momentum_252D"])
    df["RSE_cs_z"] = zscore_series(df["RSE"])
    df["VAM_cs_z"] = zscore_series(df["VAM"])
    df["ABS_cs_z"] = zscore_series(df["ABS"])
    df["AMT_cs_z"] = zscore_series(df["_AMT"])
    df["pos_52w_cs_z"] = zscore_series(df["pos_52w"])
    df["pos_30d_cs_z"] = zscore_series(df["pos_30d"])
    df["Ret_1D_cs_z"] = zscore_series(df["Ret_1D"])

    df["Momentum_63D_cs_rank"] = cs_rank_series(df["Momentum_63D"])
    df["RSE_cs_rank"] = cs_rank_series(df["RSE"])
    df["SMC_cs_rank"] = cs_rank_series(df["SMC"])
    df["CBP_cs_rank"] = cs_rank_series(df["_CBP"])
    df["TSS_cs_rank"] = cs_rank_series(df["_TSS"])
    df["VAM_cs_rank"] = cs_rank_series(df["VAM"])
    df["ABS_cs_rank"] = cs_rank_series(df["ABS"])
    df["AMT_cs_rank"] = cs_rank_series(df["_AMT"])
    df["pos_52w_cs_rank"] = cs_rank_series(df["pos_52w"])
    df["pos_30d_cs_rank"] = cs_rank_series(df["pos_30d"])
    df["Ret_1D_cs_rank"] = cs_rank_series(df["Ret_1D"])
    df["Ret_5D_cs_rank"] = cs_rank_series(df["Ret_5D"])
    df["Volatility_252D_cs_rank"] = cs_rank_series(df["Volatility_252D"])

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    # =========================================================
    # ENFORCE EXACT COLUMN SET + ORDER (Date REMOVED)
    # =========================================================
    EXPECTED_COLS = [
        "ABS", "ABS_cs_rank", "ABS_cs_z",
        "AMT_cs_rank", "AMT_cs_z",
        "CBP_cs_rank",
        "High_30D", "High_52W", "Low_30D", "Low_52W",
        "MarketTrend",
        "Momentum_126D", "Momentum_252D", "Momentum_252D_cs_z",
        "Momentum_63D", "Momentum_63D_cs_rank", "Momentum_63D_cs_z",
        "RSE", "RSE_cs_rank", "RSE_cs_z",
        "Ret_1D", "Ret_1D_cs_rank", "Ret_1D_cs_z",
        "Ret_5D", "Ret_5D_cs_rank",
        "SMA_Slope_3M",
        "SMC", "SMC_cs_rank",
        "SellScore",
        "TSS_cs_rank",
        "Ticker",
        "VAM", "VAM_cs_rank", "VAM_cs_z",
        "Volatility_252D", "Volatility_252D_cs_rank",
        "Volatility_30D",
        "Volume_SMA20",
        "avg_close_past_3_days", "avg_volatility_30D",
        "current_price",
        "pos_30d", "pos_30d_cs_rank", "pos_30d_cs_z",
        "pos_52w", "pos_52w_cs_rank", "pos_52w_cs_z"
    ]

    # Ensure presence
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Numeric cleanup
    for col in EXPECTED_COLS:
        if col == "Ticker":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Keep extras at end (not part of model columns)
    EXTRA_COLS = [c for c in ["avg_price", "shares", "invested_lei", "pnl_pct", "Timestamp"] if c in df.columns]

    df = df[EXPECTED_COLS + EXTRA_COLS]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dataset saved → {OUTPUT_FILE}")
    print("🧾 Columns:", df.columns.tolist())

    commit_to_repo(
        [OUTPUT_FILE],
        f"🤖 Auto-update ML input dataset [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}]"
    )


if __name__ == "__main__":
    prepare_llm_dataset()
