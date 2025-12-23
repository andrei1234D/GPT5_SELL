# bot/build_ml_input_dataset.py
# Builds llm_input_latest.csv in the NEW production format used by MT models.
# NOTE: Date is NOT required by the MT models (it is excluded from features during training),
# but downstream tooling may still choose to include it separately for logging.

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

# -------------------------
# MarketTrend (macro) config
# -------------------------
# IMPORTANT: training used ^GSPC; keep as default for regime consistency.
# If ^GSPC fails on yfinance in your environment, we fall back to SPY.
MARKET_TICKER_PRIMARY = "^GSPC"
MARKET_TICKER_FALLBACK = "SPY"

SMA_FAST = 50
SMA_SLOW = 200
MT_BAND = 0.01            # 1% band to reduce flip-flops
MT_CONFIRM_DAYS = 3       # require rule to hold for N last days


# =========================================================
# GIT COMMIT (truthful)
# =========================================================
def _run(cmd: list[str]) -> int:
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[GIT][FAIL] {' '.join(cmd)}\n  stdout={r.stdout[-500:]}\n  stderr={r.stderr[-500:]}")
        return int(r.returncode)
    except Exception as e:
        print(f"[GIT][EXC] {' '.join(cmd)} -> {e!r}")
        return 1


def commit_to_repo(files, message):
    # Never claim success unless push succeeded.
    rc = 0
    rc |= _run(["git", "add"] + list(files))
    rc |= _run(["git", "config", "--global", "user.name", "GitHub Actions Bot"])
    rc |= _run(["git", "config", "--global", "user.email", "bot@github.actions"])
    rc |= _run(["git", "commit", "-m", message])
    push_rc = _run(["git", "push"])

    if rc == 0 and push_rc == 0:
        print("[GIT] Commit and push succeeded.")
    else:
        print("[GIT] Commit/push did not fully succeed (see logs above).")


# =========================================================
# MATH HELPERS (match TRAINING STYLE)
# =========================================================
def safe_div(num, denom, eps: float = 1e-6):
    """
    Training-style safe division:
      - returns NaN when denom is too small (|denom| <= eps) or NaN
      - avoids inf blowups
    Works with Series/arrays/scalars.
    """
    if isinstance(denom, pd.Series):
        d = pd.to_numeric(denom, errors="coerce").replace([np.inf, -np.inf], np.nan)
        n = pd.to_numeric(num, errors="coerce")
        return n / d.where(d.abs() > eps, np.nan)

    d = np.asarray(denom, dtype=float)
    n = np.asarray(num, dtype=float)
    d = np.where(np.isfinite(d), d, np.nan)
    d = np.where(np.abs(d) > eps, d, np.nan)
    out = n / d
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def cs_rank_series(x: pd.Series) -> pd.Series:
    # Percentile rank in [0, 1], higher value => higher rank
    x = pd.to_numeric(x, errors="coerce")
    r = x.rank(method="average", pct=True, ascending=True)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(float)


# =========================================================
# YFINANCE HELPERS
# =========================================================
def _clean_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _download_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    IMPORTANT: training used auto_adjust=True.
    Keep it consistent to avoid dividend/split artifacts shifting features.
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=True,
        actions=False,
        threads=True,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = _clean_yf_columns(df).copy()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    return df


# =========================================================
# MarketTrend (macro) builder
# =========================================================
def compute_markettrend_series() -> pd.DataFrame:
    """
    Computes a daily MarketTrend series from MARKET_TICKER using:
      bull: SMA50 > SMA200*(1+band)
      bear: SMA50 < SMA200*(1-band)
      else neutral

    With confirmation: the label on day t is only set to bull/bear if the last
    MT_CONFIRM_DAYS all agree; otherwise 0.

    Returns DataFrame with columns: ["Date", "MarketTrend"]
    """
    # Try primary, fall back if needed
    mkt = _download_ohlcv(MARKET_TICKER_PRIMARY, period="5y")
    used = MARKET_TICKER_PRIMARY
    if mkt.empty or "Close" not in mkt.columns:
        mkt = _download_ohlcv(MARKET_TICKER_FALLBACK, period="5y")
        used = MARKET_TICKER_FALLBACK

    if mkt.empty or "Close" not in mkt.columns:
        raise RuntimeError(f"Market data missing for {MARKET_TICKER_PRIMARY} and fallback {MARKET_TICKER_FALLBACK}")

    close = pd.to_numeric(mkt["Close"], errors="coerce")
    sma_fast = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    sma_slow = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

    bull = (sma_fast > sma_slow * (1.0 + MT_BAND))
    bear = (sma_fast < sma_slow * (1.0 - MT_BAND))
    base = np.where(bull, 1, np.where(bear, -1, 0)).astype(int)

    mt = pd.Series(base, index=mkt.index, name="MarketTrend")

    if MT_CONFIRM_DAYS and MT_CONFIRM_DAYS > 1:
        def confirm_window(s: pd.Series) -> int:
            vals = s.to_numpy()
            if len(vals) == 0:
                return 0
            if np.all(vals == 1):
                return 1
            if np.all(vals == -1):
                return -1
            return 0

        mt_conf = mt.rolling(MT_CONFIRM_DAYS, min_periods=MT_CONFIRM_DAYS).apply(confirm_window, raw=False)
        mt = mt_conf.fillna(0.0).astype(int)

    out = pd.DataFrame({
        "Date": mt.index,
        "MarketTrend": mt.astype("int8").to_numpy(),
    }).dropna(subset=["Date"]).sort_values("Date")

    print(f"[MT] MarketTrend computed from {used} (rows={len(out)}).")
    return out


def markettrend_asof(markettrend_df: pd.DataFrame, dt: pd.Timestamp) -> int:
    """
    Backward alignment: find last market date <= dt and return its MarketTrend.
    """
    if markettrend_df is None or markettrend_df.empty:
        return 0
    dates = markettrend_df["Date"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(dates, np.datetime64(dt), side="right") - 1
    if idx < 0:
        return 0
    return int(markettrend_df["MarketTrend"].iloc[idx])


# =========================================================
# FEATURE ENGINE (time-series, per ticker)
# =========================================================
def compute_features_for_ticker(ticker: str) -> tuple[dict, pd.Timestamp] | None:
    """
    Produces ONE row (latest) with the schema used by MT models.

    IMPORTANT alignment notes:
      - Volatility_* matches TRAINING: rolling std of daily returns (pct_change)
      - auto_adjust=True matches TRAINING
      - Cross-sectional features are computed later across *your tickers* (explicit design)
    """
    try:
        df = _download_ohlcv(ticker, period="5y")
        if df.empty:
            print(f"[WARN] No data for {ticker}.")
            return None

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                print(f"[WARN] Missing {col} for {ticker}.")
                return None

        df = df.dropna(subset=["Close", "High", "Low"]).copy()
        if len(df) < 260:
            print(f"[WARN] Not enough history for {ticker} (rows={len(df)}). Need ~260+.")
            return None

        Close = pd.to_numeric(df["Close"], errors="coerce")
        High = pd.to_numeric(df["High"], errors="coerce")
        Low = pd.to_numeric(df["Low"], errors="coerce")
        Volume = pd.to_numeric(df["Volume"], errors="coerce")

        # Daily returns (training-style)
        ret = Close.pct_change(fill_method=None)

        # Rolling highs/lows (match training min_periods)
        df["High_30D"] = High.rolling(30, min_periods=10).max()
        df["Low_30D"] = Low.rolling(30, min_periods=10).min()
        df["High_52W"] = High.rolling(252, min_periods=60).max()
        df["Low_52W"] = Low.rolling(252, min_periods=60).min()

        df["SMA20"] = Close.rolling(20, min_periods=10).mean()

        # Volatility (match training: return std)
        df["Volatility_30D"] = ret.rolling(30, min_periods=10).std()
        df["Volatility_252D"] = ret.rolling(252, min_periods=60).std()

        High52 = pd.to_numeric(df["High_52W"], errors="coerce")
        Low52 = pd.to_numeric(df["Low_52W"], errors="coerce")
        High30 = pd.to_numeric(df["High_30D"], errors="coerce")
        Low30 = pd.to_numeric(df["Low_30D"], errors="coerce")
        Vol30 = pd.to_numeric(df["Volatility_30D"], errors="coerce")
        Vol252 = pd.to_numeric(df["Volatility_252D"], errors="coerce")
        SMA20 = pd.to_numeric(df["SMA20"], errors="coerce")

        # Momentum
        df["Momentum_63D"] = safe_div(Close, Close.shift(63))
        df["Momentum_126D"] = safe_div(Close, Close.shift(126))
        df["Momentum_252D"] = safe_div(Close, Close.shift(252))

        # AMT
        df["AMT"] = np.where(df["Momentum_126D"] > 1.1, df["Momentum_252D"], 0.0)

        # SMC
        smc_part1 = safe_div(df["Momentum_252D"], Vol30)
        smc_part2 = safe_div(Close, High52)
        df["SMC"] = smc_part1 * smc_part2

        # TSS
        df["TSS"] = (df["Momentum_63D"] + df["Momentum_126D"] + df["Momentum_252D"]) / 3.0

        # ABS
        range_52 = (High52 - Low52)
        abs_ratio = safe_div(Close - Low52, range_52)
        df["ABS"] = abs_ratio * Vol30

        # VAM
        df["VAM"] = safe_div(df["Momentum_63D"], 1.0 + Vol30)

        # RSE
        roll_mean = ret.rolling(63, min_periods=20).mean()
        roll_std = ret.rolling(63, min_periods=20).std()
        df["RSE"] = safe_div(roll_mean, roll_std)

        # CBP
        df["CBP"] = safe_div(Vol30, Vol252)

        # SMA slope 3M
        sma_past = SMA20.shift(63)
        df["SMA_Slope_3M"] = safe_div(SMA20 - sma_past, sma_past)

        # Returns
        df["Ret_1D"] = ret
        df["Ret_5D"] = Close.pct_change(5)

        # Range position
        range_30 = (High30 - Low30)
        df["pos_52w"] = safe_div(Close - Low52, range_52)
        df["pos_30d"] = safe_div(Close - Low30, range_30)

        # Volume SMA20 (as used by your MT feature set)
        df["Volume_SMA20"] = Volume.rolling(20, min_periods=5).mean()

        # Extra aggregates used by your MT feature set
        df["avg_close_past_3_days"] = Close.rolling(3, min_periods=1).mean()
        df["avg_volatility_30D"] = pd.to_numeric(df["Volatility_30D"], errors="coerce").rolling(3, min_periods=1).mean()
        df["current_price"] = Close

        last = df.iloc[-1].copy()
        last_dt = pd.to_datetime(df.index[-1]).tz_localize(None)

        out = {
            "Ticker": ticker,
            "MarketTrend": 0,     # filled later
            "SellScore": np.nan,  # production has no true label; keep NaN (model excludes it anyway)

            "High_30D": last.get("High_30D", np.nan),
            "Low_30D": last.get("Low_30D", np.nan),
            "High_52W": last.get("High_52W", np.nan),
            "Low_52W": last.get("Low_52W", np.nan),

            "Volatility_30D": last.get("Volatility_30D", np.nan),
            "Volatility_252D": last.get("Volatility_252D", np.nan),

            "Momentum_63D": last.get("Momentum_63D", np.nan),
            "Momentum_126D": last.get("Momentum_126D", np.nan),
            "Momentum_252D": last.get("Momentum_252D", np.nan),

            "SMC": last.get("SMC", np.nan),
            "ABS": last.get("ABS", np.nan),
            "VAM": last.get("VAM", np.nan),
            "RSE": last.get("RSE", np.nan),

            "SMA_Slope_3M": last.get("SMA_Slope_3M", np.nan),

            "Ret_1D": last.get("Ret_1D", np.nan),
            "Ret_5D": last.get("Ret_5D", np.nan),

            "pos_52w": last.get("pos_52w", np.nan),
            "pos_30d": last.get("pos_30d", np.nan),

            "Volume_SMA20": last.get("Volume_SMA20", np.nan),

            "avg_close_past_3_days": last.get("avg_close_past_3_days", np.nan),
            "avg_volatility_30D": last.get("avg_volatility_30D", np.nan),
            "current_price": last.get("current_price", np.nan),

            # internal for CS calc (kept local only)
            "_AMT": last.get("AMT", np.nan),
            "_CBP": last.get("CBP", np.nan),
            "_TSS": last.get("TSS", np.nan),
        }

        return out, last_dt

    except Exception as e:
        print(f"[WARN] Failed {ticker}: {e}")
        return None


# =========================================================
# MAIN
# =========================================================
def prepare_llm_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"[WARN] Missing {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    stocks = data.get("stocks", {}) or {}
    print(f"[INFO] Building dataset for {len(stocks)} tickers (Date excluded from model features).")

    # Build macro MarketTrend series once
    try:
        mt_series = compute_markettrend_series()
    except Exception as e:
        mt_series = pd.DataFrame()
        print(f"[WARN] MarketTrend series failed: {e}. Falling back to 0 for all tickers.")

    rows = []
    for ticker, info in stocks.items():
        res = compute_features_for_ticker(ticker)
        if res is None:
            continue

        feats, last_dt = res
        feats["MarketTrend"] = markettrend_asof(mt_series, last_dt) if not mt_series.empty else 0

        # Portfolio extras (optional; not part of model features)
        avg_price = float(info.get("avg_price", 0) or 0)
        shares = float(info.get("shares", 0) or 0)
        invested_lei = float(info.get("invested_lei", 0) or 0)

        current_price = float(pd.to_numeric(feats.get("current_price", np.nan), errors="coerce") or 0.0)

        # Keep your existing heuristic; does not affect MT features
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
        print(f"[OK] {ticker}: price={current_price:.2f}, pnl={pnl_pct:+.2f}%, MT={feats.get('MarketTrend')}")

    if not rows:
        print("[WARN] No rows produced. Nothing to save.")
        return

    df = pd.DataFrame(rows)

    # =========================================================
    # CROSS-SECTIONAL FEATURES (explicitly across *your tickers*)
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
    # ENFORCE COLUMN SET + ORDER
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

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan if col != "Ticker" else ""

    # Keep NaNs; let model imputers handle missingness at inference.
    for col in EXPECTED_COLS:
        if col == "Ticker":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    EXTRA_COLS = [c for c in ["avg_price", "shares", "invested_lei", "pnl_pct", "Timestamp"] if c in df.columns]
    df = df[EXPECTED_COLS + EXTRA_COLS]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Dataset saved -> {OUTPUT_FILE}")
    print("[OK] Columns:", df.columns.tolist())

    commit_to_repo(
        [OUTPUT_FILE],
        f"Auto-update ML input dataset [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}]"
    )


if __name__ == "__main__":
    prepare_llm_dataset()
