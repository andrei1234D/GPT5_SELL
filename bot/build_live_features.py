#!/usr/bin/env python3
"""
build_live_features.py

Builds llm_input_latest.csv with all features required by:
  - MT goodsell models
  - Peak ML models
  - Quality ML models
  - Deterministic logic (decision_engine)

Output: bot/LLM_data/input_llm/llm_input_latest.csv
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from knobs import FX_TTL_MINUTES, fx_pair_to_ron

OUTPUT_DIR = "bot/LLM_data/input_llm"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llm_input_latest.csv")
DATA_FILE = "bot/data.json"
TRACKER_FILE = "bot/sell_alerts_tracker.json"

MARKET_TICKER_PRIMARY = "^GSPC"
MARKET_TICKER_FALLBACK = "SPY"
SMA_FAST = 50
SMA_SLOW = 200
MT_BAND = 0.01
MT_CONFIRM_DAYS = 3

_FX_CACHE = {}  # currency -> (rate_to_ron, ts_utc)
_TICKER_CCY_CACHE = {}  # ticker -> currency


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def safe_div(num, denom, eps: float = 1e-6):
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
    x = pd.to_numeric(x, errors="coerce")
    r = x.rank(method="average", pct=True, ascending=True)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(float)


def _clean_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _download_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
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


def _get_ticker_currency(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if t in _TICKER_CCY_CACHE:
        return _TICKER_CCY_CACHE[t]
    ccy = "USD"
    try:
        yt = yf.Ticker(t)
        fi = getattr(yt, "fast_info", None) or {}
        ccy = fi.get("currency") or ccy
        if not ccy:
            info = getattr(yt, "info", {}) or {}
            ccy = info.get("currency") or ccy
    except Exception:
        pass
    ccy = (ccy or "USD").upper().strip()
    _TICKER_CCY_CACHE[t] = ccy
    return ccy


def _read_fx_close(symbol: str) -> Optional[float]:
    try:
        fx = yf.Ticker(symbol).history(period="1d")
        if fx is not None and (not fx.empty) and ("Close" in fx.columns):
            v = float(fx["Close"].iloc[-1])
            if v > 0:
                return v
    except Exception:
        return None
    return None


def get_fx_to_ron(currency: str, *, fallback: Optional[float] = None, ttl_minutes: int = FX_TTL_MINUTES) -> float:
    """
    Returns: RON per 1 unit of `currency` (e.g., CAD->RON).
    Behavior: live-first; if live fails, use cached; else use provided fallback; else 1.0.
    """
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return 1.0

    now = datetime.now(timezone.utc)

    # Cache (TTL)
    if c in _FX_CACHE:
        rate, ts = _FX_CACHE[c]
        if isinstance(ts, datetime) and (now - ts) <= pd.Timedelta(minutes=ttl_minutes) and rate > 0:
            return float(rate)

    # 1) Direct pair to RON
    pair = fx_pair_to_ron(c)
    if pair:
        v = _read_fx_close(pair)
        if v is not None:
            _FX_CACHE[c] = (float(v), now)
            return float(v)

    # 2) Cross via USD if direct missing/fails
    usdron = _read_fx_close("USDRON=X") or None
    if usdron is not None:
        cross = None
        v1 = _read_fx_close(f"{c}USD=X")
        if v1 is not None:
            cross = float(v1) * float(usdron)  # (USD/CCY)*(RON/USD) = RON/CCY
        else:
            v2 = _read_fx_close(f"USD{c}=X")
            if v2 is not None and float(v2) > 0:
                cross = (1.0 / float(v2)) * float(usdron)

        if cross is not None and cross > 0:
            _FX_CACHE[c] = (float(cross), now)
            return float(cross)

    if fallback is not None and float(fallback) > 0:
        return float(fallback)
    return 1.0


def compute_markettrend_series() -> pd.DataFrame:
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
    if markettrend_df is None or markettrend_df.empty:
        return 0
    dates = markettrend_df["Date"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(dates, np.datetime64(dt), side="right") - 1
    if idx < 0:
        return 0
    return int(markettrend_df["MarketTrend"].iloc[idx])


def _load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        return pd.to_datetime(s, utc=True).tz_convert(None)
    except Exception:
        try:
            return pd.to_datetime(s).tz_localize(None)
        except Exception:
            return None


def _compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    if loss.iloc[-1] == 0:
        return None
    rs = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def _compute_macd(close: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if len(close) < 26:
        return None, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(macd_signal.iloc[-1])


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else None


def compute_features_for_ticker(
    ticker: str,
    avg_price: float,
    buy_time: Optional[pd.Timestamp],
) -> tuple[dict, pd.Timestamp] | None:
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

    ret = Close.pct_change(fill_method=None)

    df["High_30D"] = High.rolling(30, min_periods=10).max()
    df["Low_30D"] = Low.rolling(30, min_periods=10).min()
    df["High_52W"] = High.rolling(252, min_periods=60).max()
    df["Low_52W"] = Low.rolling(252, min_periods=60).min()

    df["SMA20"] = Close.rolling(20, min_periods=10).mean()
    df["SMA50"] = Close.rolling(50, min_periods=50).mean()
    df["SMA200"] = Close.rolling(200, min_periods=200).mean()

    df["Volatility_30D"] = ret.rolling(30, min_periods=10).std()
    df["Volatility_252D"] = ret.rolling(252, min_periods=60).std()

    High52 = pd.to_numeric(df["High_52W"], errors="coerce")
    Low52 = pd.to_numeric(df["Low_52W"], errors="coerce")
    High30 = pd.to_numeric(df["High_30D"], errors="coerce")
    Low30 = pd.to_numeric(df["Low_30D"], errors="coerce")
    Vol30 = pd.to_numeric(df["Volatility_30D"], errors="coerce")
    Vol252 = pd.to_numeric(df["Volatility_252D"], errors="coerce")
    SMA20 = pd.to_numeric(df["SMA20"], errors="coerce")

    df["Momentum_63D"] = safe_div(Close, Close.shift(63))
    df["Momentum_126D"] = safe_div(Close, Close.shift(126))
    df["Momentum_252D"] = safe_div(Close, Close.shift(252))

    df["AMT"] = np.where(df["Momentum_126D"] > 1.1, df["Momentum_252D"], 0.0)

    smc_part1 = safe_div(df["Momentum_252D"], Vol30)
    smc_part2 = safe_div(Close, High52)
    df["SMC"] = smc_part1 * smc_part2

    df["TSS"] = (df["Momentum_63D"] + df["Momentum_126D"] + df["Momentum_252D"]) / 3.0

    range_52 = (High52 - Low52)
    abs_ratio = safe_div(Close - Low52, range_52)
    df["ABS"] = abs_ratio * Vol30

    df["VAM"] = safe_div(df["Momentum_63D"], 1.0 + Vol30)

    roll_mean = ret.rolling(63, min_periods=20).mean()
    roll_std = ret.rolling(63, min_periods=20).std()
    df["RSE"] = safe_div(roll_mean, roll_std)

    df["CBP"] = safe_div(Vol30, Vol252)

    sma_past = SMA20.shift(63)
    df["SMA_Slope_3M"] = safe_div(SMA20 - sma_past, sma_past)

    df["Ret_1D"] = ret
    df["Ret_5D"] = Close.pct_change(5)

    range_30 = (High30 - Low30)
    df["pos_52w"] = safe_div(Close - Low52, range_52)
    df["pos_30d"] = safe_div(Close - Low30, range_30)

    df["Volume_SMA20"] = Volume.rolling(20, min_periods=5).mean()

    df["avg_close_past_3_days"] = Close.rolling(3, min_periods=1).mean()
    df["avg_volatility_30D"] = pd.to_numeric(df["Volatility_30D"], errors="coerce").rolling(3, min_periods=1).mean()
    df["current_price"] = Close

    # EMA ratios (peak features)
    ema20 = Close.ewm(span=20, adjust=False).mean()
    ema20_vol = Volume.ewm(span=20, adjust=False).mean()
    df["Price_EMA20_Ratio"] = safe_div(Close, ema20)
    df["Volume_EMA20_Ratio"] = safe_div(Volume, ema20_vol)

    last = df.iloc[-1].copy()
    last_dt = pd.to_datetime(df.index[-1]).tz_localize(None)

    # Deterministic features
    ma50 = float(last.get("SMA50", np.nan)) if len(df) >= 50 else None
    ma200 = float(last.get("SMA200", np.nan)) if len(df) >= 200 else None
    rsi = _compute_rsi(Close)
    macd, macd_signal = _compute_macd(Close)
    atr = _compute_atr(High, Low, Close)
    momentum = float(Close.diff().iloc[-1]) if len(Close) >= 2 else None
    volume = None
    if len(Volume) >= 20:
        today_vol = Volume.iloc[-1]
        avg_vol20 = Volume.tail(20).mean()
        volume = float(today_vol / avg_vol20) if avg_vol20 > 0 else None
    support = float(Close.tail(20).min()) if len(Close) >= 20 else None
    resistance = float(Close.tail(20).max()) if len(Close) >= 20 else None

    # Peak-related from buy_time
    days_since_buy = None
    past_peak_pnl = None
    dd_from_past_peak = None
    if buy_time is not None:
        days_since_buy = max(0, int((last_dt - buy_time).days))
        df_since = df[df.index >= buy_time]
        if not df_since.empty and avg_price > 0:
            pnl_series = (df_since["Close"] / avg_price - 1.0) * 100.0
            past_peak_pnl = float(np.nanmax(pnl_series))
            current_pnl = float(pnl_series.iloc[-1])
            if past_peak_pnl and past_peak_pnl > 0:
                dd_from_past_peak = float((past_peak_pnl - current_pnl) / max(past_peak_pnl, 1e-6))
            else:
                dd_from_past_peak = 0.0

    # Derived peak features
    macd_hist = None if macd is None or macd_signal is None else float(macd - macd_signal)
    price = float(last.get("current_price", np.nan))
    price_ma50_ratio = safe_div(price, ma50) if ma50 else np.nan
    price_ma200_ratio = safe_div(price, ma200) if ma200 else np.nan
    ma50_ma200_ratio = safe_div(ma50, ma200) if ma50 and ma200 else np.nan
    mom_atr = safe_div(momentum, atr) if momentum is not None and atr else np.nan
    ret1d_vol30 = safe_div(last.get("Ret_1D", np.nan), last.get("Volatility_30D", np.nan))
    rsi_centered = ((rsi - 50.0) / 50.0) if rsi is not None else np.nan
    rsi_overbought = max(rsi - 70.0, 0.0) / 30.0 if rsi is not None else np.nan
    ema_volume_thrust = (_safe_float(last.get("Price_EMA20_Ratio"), np.nan) - 1.0) * np.log1p(_safe_float(last.get("Volume_EMA20_Ratio"), np.nan))
    vol_regime = safe_div(last.get("Volatility_30D", np.nan), last.get("Volatility_252D", np.nan))
    price_vs_support = safe_div(price - support, price) if support is not None and price else np.nan
    macd_hist_vol = (macd_hist * _safe_float(last.get("Volume_EMA20_Ratio"), np.nan)) if macd_hist is not None else np.nan
    trend_strength = (abs(_safe_float(price_ma50_ratio, np.nan) - 1.0) + abs(_safe_float(ma50_ma200_ratio, np.nan) - 1.0))
    if past_peak_pnl is None or dd_from_past_peak is None:
        peak_risk = 0.0
    else:
        # Clamp to avoid log1p on negatives and keep deterministic, non-NaN output
        peak_risk = np.log1p(max(past_peak_pnl, 0.0)) * dd_from_past_peak
    overbought_drawdown = (rsi_overbought * dd_from_past_peak) if rsi_overbought is not None and dd_from_past_peak is not None else np.nan
    macd_overbought = (macd_hist * rsi_overbought) if macd_hist is not None and rsi_overbought is not None else np.nan
    ema20_pullback = (_safe_float(last.get("Price_EMA20_Ratio"), np.nan) - 1.0) * max(-_safe_float(last.get("Ret_1D", 0.0), 0.0), 0.0)
    vol_spike_down = max(-_safe_float(last.get("Ret_1D", 0.0), 0.0), 0.0) * _safe_float(last.get("Volume_EMA20_Ratio"), np.nan)
    trend_exhaust = trend_strength * (1.0 - np.clip(_safe_float(dd_from_past_peak, 0.0), 0.0, 1.0))
    peak_velocity = (past_peak_pnl / (days_since_buy + 1.0)) if past_peak_pnl is not None and days_since_buy is not None else np.nan
    momentum_reversal = (_safe_float(momentum, np.nan) * max(-_safe_float(last.get("Ret_1D", 0.0), 0.0), 0.0))
    rsi_reversal = (_safe_float(rsi_overbought, np.nan) * max(-_safe_float(last.get("Ret_1D", 0.0), 0.0), 0.0))
    squeeze_break = _safe_float(vol_regime, np.nan) * (_safe_float(last.get("Price_EMA20_Ratio"), np.nan) - 1.0)

    out = {
        "Ticker": ticker,
        "MarketTrend": 0,
        "SellScore": np.nan,

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

        # Deterministic inputs
        "rsi": rsi,
        "momentum": momentum,
        "macd": macd,
        "macd_signal": macd_signal,
        "ma50": ma50,
        "ma200": ma200,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "volume": volume,

        # Peak features
        "Price_EMA20_Ratio": last.get("Price_EMA20_Ratio", np.nan),
        "Volume_EMA20_Ratio": last.get("Volume_EMA20_Ratio", np.nan),
        "days_since_buy": days_since_buy,
        "past_peak_pnl": past_peak_pnl,
        "dd_from_past_peak": dd_from_past_peak,
        "macd_hist": macd_hist,
        "price_ma50_ratio": price_ma50_ratio,
        "price_ma200_ratio": price_ma200_ratio,
        "ma50_ma200_ratio": ma50_ma200_ratio,
        "mom_atr": mom_atr,
        "ret1d_vol30": ret1d_vol30,
        "rsi_centered": rsi_centered,
        "rsi_overbought": rsi_overbought,
        "ema_volume_thrust": ema_volume_thrust,
        "vol_regime": vol_regime,
        "price_vs_support": price_vs_support,
        "macd_hist_vol": macd_hist_vol,
        "trend_strength": trend_strength,
        "peak_risk": peak_risk,
        "overbought_drawdown": overbought_drawdown,
        "macd_overbought": macd_overbought,
        "ema20_pullback": ema20_pullback,
        "vol_spike_down": vol_spike_down,
        "trend_exhaust": trend_exhaust,
        "peak_velocity": peak_velocity,
        "momentum_reversal": momentum_reversal,
        "rsi_reversal": rsi_reversal,
        "squeeze_break": squeeze_break,

        # internal for CS calc
        "_AMT": last.get("AMT", np.nan),
        "_CBP": last.get("CBP", np.nan),
        "_TSS": last.get("TSS", np.nan),
    }

    return out, last_dt


def build_live_features():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"[WARN] Missing {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracker = _load_tracker()
    tracker_t = (tracker.get("tickers", {}) or {})

    stocks = data.get("stocks", {}) or {}
    print(f"[INFO] Building dataset for {len(stocks)} tickers.")

    # Build macro MarketTrend series once
    try:
        mt_series = compute_markettrend_series()
    except Exception as e:
        mt_series = pd.DataFrame()
        print(f"[WARN] MarketTrend series failed: {e}. Falling back to 0 for all tickers.")

    rows = []
    for ticker, info in stocks.items():
        buy_time = _parse_ts(info.get("buy_time"))
        if buy_time is None:
            st = tracker_t.get(ticker, {}) if isinstance(tracker_t, dict) else {}
            buy_time = _parse_ts(st.get("first_seen_time"))
        if buy_time is None:
            buy_time = datetime.now(timezone.utc).replace(tzinfo=None)

        avg_price_raw = float(info.get("avg_price", 0) or 0)

        # Standardize buy price to ticker currency using stored FX if available
        shares = float(info.get("shares", 0) or 0)
        invested_lei = float(info.get("invested_lei", 0) or 0)
        fx_buy = _safe_float(info.get("fx_rate_buy"), None)
        avg_price_ccy = avg_price_raw
        if shares > 0 and invested_lei > 0 and fx_buy and fx_buy > 0:
            avg_price_ccy_from_ron = (invested_lei / fx_buy) / shares
            if (not np.isfinite(avg_price_ccy)) or avg_price_ccy <= 0:
                avg_price_ccy = float(avg_price_ccy_from_ron)
            else:
                # If stored avg price looks off vs FX-derived price, trust FX-derived
                rel_diff = abs(avg_price_ccy - avg_price_ccy_from_ron) / max(avg_price_ccy_from_ron, 1e-6)
                if rel_diff > 0.50:
                    avg_price_ccy = float(avg_price_ccy_from_ron)

        res = compute_features_for_ticker(ticker, avg_price=avg_price_ccy, buy_time=buy_time)
        if res is None:
            continue

        feats, last_dt = res
        feats["MarketTrend"] = markettrend_asof(mt_series, last_dt) if not mt_series.empty else 0

        # Portfolio extras (not model features)
        current_price = float(pd.to_numeric(feats.get("current_price", np.nan), errors="coerce") or 0.0)

        st = tracker_t.get(ticker, {}) if isinstance(tracker_t, dict) else {}
        ccy = (info.get("currency") or st.get("last_currency") or _get_ticker_currency(ticker) or "USD").upper().strip()
        fx_fallback = _safe_float(st.get("last_fx_to_ron"), None)
        if fx_fallback is None:
            fx_fallback = _safe_float(st.get("last_fx_to_lei"), None)
        if fx_fallback is None:
            fx_fallback = _safe_float(info.get("fx_rate_buy"), None)

        fx_to_ron = get_fx_to_ron(ccy, fallback=fx_fallback)
        pnl_lei = current_price * shares * float(fx_to_ron) - invested_lei
        pnl_pct = (pnl_lei / invested_lei * 100.0) if invested_lei else 0.0

        avg_price_ron = (invested_lei / shares) if shares > 0 else 0.0

        feats.update({
            "avg_price": float(avg_price_ccy),
            "avg_price_ron": float(avg_price_ron),
            "shares": shares,
            "invested_lei": invested_lei,
            "pnl_pct": float(pnl_pct),
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "buy_time": buy_time.strftime("%Y-%m-%dT%H:%M:%SZ") if buy_time else None,
        })

        rows.append(feats)
        print(f"[OK] {ticker}: price={current_price:.2f}, pnl={pnl_pct:+.2f}%, MT={feats.get('MarketTrend')}")

    if not rows:
        print("[WARN] No rows produced. Nothing to save.")
        return

    df = pd.DataFrame(rows)

    # Cross-sectional features (MT model)
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

    # Enforce MT expected columns, but keep extras (peak/det)
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

    # Keep NaNs for imputers
    for col in EXPECTED_COLS:
        if col == "Ticker":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Order: expected cols first, then all extra cols
    extras = [c for c in df.columns if c not in EXPECTED_COLS]
    df = df[EXPECTED_COLS + extras]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Dataset saved -> {OUTPUT_FILE}")
    print("[OK] Columns:", df.columns.tolist())


if __name__ == "__main__":
    build_live_features()
