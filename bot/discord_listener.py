import os
import json
import base64
import sys
import math
from threading import Thread
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any

import requests
import yfinance as yf
import discord
from discord.ext import commands

from keep_alive import keep_alive
from knobs import (
    TRACKER_FILE,
    PROFIT_ADJ,
    STRONG_SELL_MULT,
    FX_TTL_MINUTES,
    fx_pair_to_ron,
    profit_based_weak_req,
    DISCORD_MSG_LIMIT,
    DEFAULT_WEAK_REQ,
    UI_BASE_THR_BY_MT,
    UI_BASE_THR_DEFAULT,
    THR_EARLY_MIN,
    THR_EARLY_MAX,
    THR_STRONG_MIN,
    THR_STRONG_MAX,
    THR_STRONG_MIN_ADD,
    RISK_STABLE_MIN,
    RISK_STABLE_FRAC,
    RISK_WATCH_FRAC,
)

# ---------------------------
# Version banner (helps confirm the running code)
# ---------------------------
BOT_VERSION = "2026-01-31 decision-engine-v9_1-ui3-weakreq-3-2"

DATA_FILE = "bot/data.json"
TRACKER_PRICE_MAX_AGE_HOURS = 5

# ---------------------------
# Data Management
# ---------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stocks": {}, "realized_pnl": 0.0}


def save_data(data):
    d = os.path.dirname(DATA_FILE)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# GitHub API Push/Pull
# ---------------------------
def push_to_github(file_path, commit_message="Auto-update data.json from Discord bot"):
    try:
        GH_TOKEN = os.getenv("GH_TOKEN")
        if not GH_TOKEN:
            print("⚠️ GitHub token not set in secrets (GH_TOKEN)")
            return
        print("🔐 GH_TOKEN present; attempting GitHub push")

        repo = "andrei1234D/GPT5_SELL"
        branch = "main"
        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        r = requests.get(api_url, headers={"Authorization": f"token {GH_TOKEN}"})
        sha = None
        try:
            sha = (r.json() or {}).get("sha")
        except Exception:
            sha = None

        payload = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        res = requests.put(api_url, json=payload, headers={"Authorization": f"token {GH_TOKEN}"})
        if res.status_code in (200, 201):
            print("✅ Pushed", file_path, "to GitHub")
        else:
            print("❌ GitHub push failed:", res.status_code, res.text[:500])
    except Exception as e:
        print(f"⚠️ Error pushing to GitHub: {e}")


def pull_from_github(file_path):
    """Fetch latest file from GitHub repo and overwrite local copy."""
    try:
        GH_TOKEN = os.getenv("GH_TOKEN")
        if not GH_TOKEN:
            print("⚠️ GitHub token not set in secrets (GH_TOKEN)")
            return

        repo = "andrei1234D/GPT5_SELL"
        branch = "main"
        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"

        res = requests.get(api_url, headers={"Authorization": f"token {GH_TOKEN}"})
        if res.status_code == 200:
            response_json = res.json() or {}
            if "content" not in response_json:
                print("❌ GitHub response missing 'content':", str(response_json)[:500])
                return

            content = base64.b64decode(response_json["content"]).decode()

            d = os.path.dirname(file_path)
            if d:
                os.makedirs(d, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Pulled latest {file_path} from GitHub")
        else:
            # 404 is common on first run when file hasn't been created yet.
            print("⚠️ GitHub pull failed:", res.status_code, res.text[:200])
    except Exception as e:
        print(f"⚠️ Error pulling from GitHub: {e}")


# ---------------------------
# Small helpers
# ---------------------------
def _safe_float(x, default=None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _safe_int(x, default=None) -> Optional[int]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ---------------------------
# FX helpers (multi-currency; live-first, fallback to last-known)
# ---------------------------
_FX_CACHE = {}  # currency -> (rate_to_ron, ts_utc)
_TICKER_CCY_CACHE = {}  # ticker -> currency


def get_ticker_currency(ticker: str, *, yt=None) -> str:
    t = (ticker or "").upper().strip()
    if t in _TICKER_CCY_CACHE:
        return _TICKER_CCY_CACHE[t]
    ccy = "USD"
    try:
        yt = yt or _get_ticker_obj(t)
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


def _get_ticker_obj(ticker: str):
    """Per-run lightweight cache for yfinance Ticker objects."""
    t = (ticker or "").upper().strip()
    if not hasattr(_get_ticker_obj, "_cache"):
        _get_ticker_obj._cache = {}
    cache = _get_ticker_obj._cache
    if t not in cache:
        cache[t] = yf.Ticker(t)
    return cache[t]


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

    now = datetime.utcnow()

    # Cache (TTL)
    if c in _FX_CACHE:
        rate, ts = _FX_CACHE[c]
        if isinstance(ts, datetime) and (now - ts) <= timedelta(minutes=ttl_minutes) and rate > 0:
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

    # 3) Fallbacks
    if c in _FX_CACHE:
        rate, _ = _FX_CACHE[c]
        if rate and rate > 0:
            return float(rate)

    if fallback is not None and fallback > 0:
        _FX_CACHE[c] = (float(fallback), now)
        return float(fallback)

    return 1.0


# ---------------------------
# Price helper (more "live" than daily close)
# ---------------------------
def get_recent_price(ticker: str, fallback: float, *, yt=None) -> float:
    """
    Attempts, in order:
      1) fast_info.last_price (if available)
      2) 1-minute interval last close for today
      3) daily history last close
      4) fallback
    """
    t = (ticker or "").upper().strip()

    try:
        yt = yt or _get_ticker_obj(t)
        fi = getattr(yt, "fast_info", None) or {}
        for k in ("last_price", "lastPrice"):
            v = fi.get(k)
            if v is not None:
                vv = float(v)
                if vv > 0:
                    return vv
    except Exception:
        pass

    try:
        yt = yt or _get_ticker_obj(t)
        intraday = yt.history(period="1d", interval="1m")
        if intraday is not None and (not intraday.empty) and ("Close" in intraday.columns):
            vv = float(intraday["Close"].iloc[-1])
            if vv > 0:
                return vv
    except Exception:
        pass

    try:
        yt = yt or _get_ticker_obj(t)
        px = yt.history(period="1d")
        if px is not None and (not px.empty) and ("Close" in px.columns):
            vv = float(px["Close"].iloc[-1])
            if vv > 0:
                return vv
    except Exception:
        pass

    return float(fallback)


def _parse_utc_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


# ---------------------------
# Threshold fallbacks (UI only; engine values from tracker take precedence)
# ---------------------------
def get_sell_thresholds(pnl_pct: Optional[float], mt: int) -> Tuple[float, float]:
    mt = int(mt) if mt in (-1, 0, 1) else 0
    early = float(UI_BASE_THR_BY_MT.get(mt, UI_BASE_THR_DEFAULT))

    p = _safe_float(pnl_pct, None)
    if p is not None:
        for cutoff, adj in PROFIT_ADJ:
            if p >= float(cutoff):
                early += float(adj)
                break

    early = _clamp(early, THR_EARLY_MIN, THR_EARLY_MAX)

    # Align to engine: strong = max(early*1.25, early+0.05) then clamp to 0.95
    strong = max(float(early) * float(STRONG_SELL_MULT), float(early) + THR_STRONG_MIN_ADD)
    strong = _clamp(strong, THR_STRONG_MIN, THR_STRONG_MAX)

    return float(early), float(strong)


def risk_label_from_sell_index(avg_sell_index: Optional[float], early_thr: float) -> str:
    if avg_sell_index is None:
        return "🟢 Stable"

    stable_cut = max(RISK_STABLE_MIN, RISK_STABLE_FRAC * early_thr)
    watch_cut = max(stable_cut, RISK_WATCH_FRAC * early_thr)

    if avg_sell_index < stable_cut:
        return "🟢 Stable"
    if avg_sell_index < watch_cut:
        return "🟡 Watch"
    if avg_sell_index < early_thr:
        return "🟠 Weak"
    return "🔴 Critical"


# ---------------------------
# Chunk-safe Discord message builder (NEVER splits a ticker block)
# ---------------------------
def _segments_len(segs: list[str]) -> int:
    if not segs:
        return 0
    return sum(len(s) for s in segs) + (len(segs) - 1)


async def send_chunked_blocks(ctx, header: str, blocks: list[str], totals: str, limit: int = DISCORD_MSG_LIMIT):
    chunks: list[list[str]] = [[header]] if header else [[]]

    for b in blocks:
        if _segments_len(chunks[-1] + [b]) <= limit:
            chunks[-1].append(b)
        else:
            chunks.append([b])

    while _segments_len(chunks[-1] + [totals]) > limit:
        if len(chunks[-1]) > 1:
            moved = chunks[-1].pop()
            chunks.append([moved])
        else:
            break

    if _segments_len(chunks[-1] + [totals]) <= limit:
        chunks[-1].append(totals)
    else:
        chunks.append([totals])

    for segs in chunks:
        msg = "\n".join(segs).strip()
        if msg:
            await ctx.send(msg)


# ---------------------------
# Discord Bot Setup
# ---------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    try:
        pull_from_github(DATA_FILE)
        pull_from_github(TRACKER_FILE)
    except Exception as e:
        print(f"⚠️ Skipped GitHub pull at startup: {e}")
    print(f"✅ Logged in as {bot.user} | version={BOT_VERSION}")


@bot.event
async def on_command_error(ctx, error):
    # Surface errors so commands don't fail silently in Discord or crash the bot.
    try:
        await ctx.send(f"?? Command error: {type(error).__name__}: {error}")
    except Exception:
        pass
    try:
        import traceback
        traceback.print_exception(type(error), error, error.__traceback__)
    except Exception:
        pass


# ---------------------------
# Bot Commands
# ---------------------------
@bot.command()
async def buy(ctx, ticker: str, price: float, lei_invested: float):
    t = ticker.upper().strip()
    data = load_data()
    stocks = data.get("stocks", {}) or {}

    ccy = get_ticker_currency(t)
    fx_now = get_fx_to_ron(ccy)

    if fx_now <= 0:
        await ctx.send(f"⚠️ FX unavailable for {ccy}. Try again later.")
        return

    ccy_amount = lei_invested / fx_now
    shares_bought = ccy_amount / price if price > 0 else 0.0

    if shares_bought <= 0:
        await ctx.send("⚠️ Invalid buy. Check price and invested amount.")
        return

    if t in stocks:
        old_price = float(stocks[t].get("avg_price", price))
        old_shares = float(stocks[t].get("shares", 0.0))
        old_invested = float(stocks[t].get("invested_lei", 0.0))
        old_fx = float(stocks[t].get("fx_rate_buy", fx_now))
        old_ccy = (stocks[t].get("currency") or ccy).upper().strip()

        if old_ccy != ccy:
            await ctx.send(f"⚠️ Currency mismatch for {t} (stored {old_ccy}, live {ccy}). Not updating.")
            return

        new_shares = old_shares + shares_bought
        new_invested = old_invested + lei_invested
        avg_price = ((old_price * old_shares) + (price * shares_bought)) / new_shares if new_shares > 0 else price

        weight = (lei_invested / new_invested) if new_invested > 0 else 1.0
        new_fx = old_fx * (1 - weight) + fx_now * weight

        stocks[t]["avg_price"] = float(avg_price)
        stocks[t]["shares"] = float(new_shares)
        stocks[t]["invested_lei"] = float(new_invested)
        stocks[t]["currency"] = ccy
        stocks[t]["fx_rate_buy"] = float(new_fx)
    else:
        stocks[t] = {
            "avg_price": float(price),
            "shares": float(shares_bought),
            "invested_lei": float(lei_invested),
            "currency": ccy,
            "fx_rate_buy": float(fx_now),
        }

    data["stocks"] = stocks
    save_data(data)
    push_to_github(DATA_FILE, f"BUY {lei_invested} RON of {t} @ {price} {ccy} (FX {fx_now:.4f} RON/{ccy})")

    await ctx.send(
        f"✅ BUY **{t}** | Avg {stocks[t]['avg_price']:.2f} {ccy} | Shares {stocks[t]['shares']:.4f} | "
        f"Invested {stocks[t]['invested_lei']:.2f} RON | FX(ref) {stocks[t]['fx_rate_buy']:.4f} RON/{ccy} "
        f"(live {fx_now:.4f})"
    )


@bot.command()
async def sell(ctx, ticker: str, price: float, amount: str):
    t = ticker.upper().strip()
    data = load_data()
    stocks = data.get("stocks", {}) or {}

    if t not in stocks:
        await ctx.send(f"⚠️ {t} is not being tracked.")
        return

    avg_price_ccy = float(stocks[t].get("avg_price", 0))
    invested_lei = float(stocks[t].get("invested_lei", 0))
    total_shares = float(stocks[t].get("shares", 0))
    ccy = (stocks[t].get("currency") or get_ticker_currency(t)).upper().strip()
    fx_ref = float(stocks[t].get("fx_rate_buy", get_fx_to_ron(ccy)))

    fx_sell = get_fx_to_ron(ccy, fallback=fx_ref)
    if fx_sell <= 0:
        await ctx.send(f"⚠️ FX unavailable for {ccy}. Try again later.")
        return

    if amount.lower() == "all":
        shares_sold = total_shares
        proceeds_ccy = shares_sold * price
        proceeds_lei = proceeds_ccy * fx_sell
    else:
        proceeds_lei = _safe_float(amount, None)
        if proceeds_lei is None:
            await ctx.send("⚠️ Invalid amount. Use a number (RON proceeds) or 'all'.")
            return
        if proceeds_lei <= 0:
            await ctx.send("⚠️ Amount must be positive.")
            return

        proceeds_ccy = proceeds_lei / fx_sell
        shares_sold = proceeds_ccy / price if price > 0 else 0.0

        if shares_sold > total_shares + 1e-9:
            await ctx.send(f"⚠️ Not enough shares. You have {total_shares:.4f} shares.")
            return

    share_ratio = (shares_sold / total_shares) if total_shares > 0 else 0.0
    cost_basis_lei = invested_lei * share_ratio

    pnl_lei = proceeds_lei - cost_basis_lei
    data["realized_pnl"] = float(data.get("realized_pnl", 0.0)) + pnl_lei

    remaining_shares = total_shares - shares_sold
    if remaining_shares <= 1e-9:
        del stocks[t]
    else:
        weight = (cost_basis_lei / invested_lei) if invested_lei > 0 else 1.0
        new_fx = fx_ref * (1 - weight) + fx_sell * weight

        stocks[t]["shares"] = float(remaining_shares)
        stocks[t]["invested_lei"] = float(invested_lei - cost_basis_lei)
        stocks[t]["avg_price"] = float(avg_price_ccy)
        stocks[t]["currency"] = ccy
        stocks[t]["fx_rate_buy"] = float(new_fx)

    data["stocks"] = stocks
    save_data(data)
    push_to_github(DATA_FILE, f"SELL {amount.upper()} of {t} @ {price} {ccy} (FX {fx_sell:.4f} RON/{ccy})")

    await ctx.send(
        f"💸 SELL **{t}**\n"
        f"Price: {price:.2f} {ccy} | Proceeds: {proceeds_lei:.2f} RON | FX: {fx_sell:.4f} RON/{ccy}\n"
        f"PnL (realized): {pnl_lei:+.2f} RON\n"
        f"📊 Cumulative Realized PnL: {float(data.get('realized_pnl', 0.0)):.2f} RON"
        + ("" if t not in stocks else f"\n🔁 FX ref after smoothing: {stocks[t]['fx_rate_buy']:.4f} RON/{ccy}")
    )


@bot.command(name="list")
async def list_cmd(ctx):
    pull_from_github(DATA_FILE)
    pull_from_github(TRACKER_FILE)

    data = load_data()
    stocks = data.get("stocks", {}) or {}
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return

    tracker = {"tickers": {}}
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                tracker = json.load(f) or tracker
        except Exception:
            tracker = {"tickers": {}}

    total_unrealized_pnl = 0.0
    stock_list = []
    fx_run_cache = {}

    for ticker, info in stocks.items():
        t = str(ticker).upper().strip()
        yt = _get_ticker_obj(t)
        avg_price = float(info.get("avg_price", 0))
        shares = float(info.get("shares", 0))
        invested_lei = float(info.get("invested_lei", 0))

        state = (tracker.get("tickers", {}) or {}).get(t, {}) or {}
        state_price = _safe_float(state.get("last_checked_price"), None)
        state_ts = _parse_utc_ts(state.get("last_checked_time"))
        use_tracker_price = False
        if state_price is not None and state_price > 0 and state_ts is not None:
            age_sec = (datetime.utcnow() - state_ts).total_seconds()
            use_tracker_price = (age_sec >= 0) and (age_sec <= (TRACKER_PRICE_MAX_AGE_HOURS * 3600))

        if use_tracker_price:
            current_price = float(state_price)
        else:
            current_price = get_recent_price(t, fallback=avg_price, yt=yt)

        ccy = (info.get("currency") or state.get("last_currency") or get_ticker_currency(t, yt=yt) or "USD").upper().strip()

        fx_fallback = _safe_float(state.get("last_fx_to_ron"), None)
        if fx_fallback is None:
            fx_fallback = _safe_float(state.get("last_fx_to_lei"), None)
        if fx_fallback is None:
            fx_fallback = _safe_float(info.get("fx_rate_buy"), None)

        if ccy in fx_run_cache:
            fx_to_ron = fx_run_cache[ccy]
        else:
            fx_to_ron = get_fx_to_ron(ccy, fallback=fx_fallback)
            fx_run_cache[ccy] = fx_to_ron

        current_value_lei = current_price * shares * float(fx_to_ron)
        pnl_lei = current_value_lei - invested_lei
        total_unrealized_pnl += pnl_lei
        pnl_pct = (pnl_lei / invested_lei) * 100.0 if invested_lei > 0 else None

        # Engine weak-days
        weak_days = _safe_int(state.get("weak_days"), 0) or 0

        # Prefer engine-provided effective weak_req from tracker (source of truth)
        weak_req_base = _safe_int(state.get("weak_req"), None)
        if weak_req_base is None:
            weak_req_base = _safe_int(state.get("last_weak_req"), None)

        # If still missing (first run / old tracker), use a conservative default,
        # then apply profit-based lowering to match engine policy.
        if weak_req_base is None:
            weak_req_base = DEFAULT_WEAK_REQ

        weak_req = profit_based_weak_req(pnl_pct, int(weak_req_base))


        # Regime
        mt_regime_int = _safe_int(state.get("last_mt"), 0) or 0
        if mt_regime_int not in (-1, 0, 1):
            mt_regime_int = 0

        # Prefer engine thresholds if present
        early_thr = _safe_float(state.get("last_sell_thr_early"), None)
        strong_thr = _safe_float(state.get("last_sell_thr_strong"), None)
        if early_thr is None or strong_thr is None:
            early_f, strong_f = get_sell_thresholds(pnl_pct, mt_regime_int)
            early_thr = early_thr if early_thr is not None else early_f
            strong_thr = strong_thr if strong_thr is not None else strong_f

        # Sell indices: prefer avg_sell_index (engine uses that for decisions)
        sell_index_raw = _safe_float(state.get("last_sell_index_raw"), None)
        avg_sell_index = _safe_float(state.get("avg_sell_index"), None)
        if avg_sell_index is None:
            # legacy fallback
            avg_sell_index = _safe_float(state.get("last_avg_sell_index"), None)
        if avg_sell_index is None:
            # last resort: rolling list
            rolling = state.get("rolling_sell_index", []) or []
            if isinstance(rolling, list) and rolling:
                try:
                    vals = [float(x) for x in rolling if x is not None and math.isfinite(float(x))]
                    avg_sell_index = (sum(vals) / len(vals)) if vals else None
                except Exception:
                    avg_sell_index = None
        if avg_sell_index is None:
            avg_sell_index = sell_index_raw if sell_index_raw is not None else 0.0

        risk_emoji = risk_label_from_sell_index(avg_sell_index, float(early_thr))

        # ML diagnostics (engine keys)
        mt_prob = _safe_float(state.get("last_mt_prob"), None)
        thr_used = _safe_float(state.get("ml_prob_thr_used"), None)
        gate_used = _safe_float(state.get("last_mt_gate_used"), None)  # normalized ramp gate
        mt_sell = state.get("last_mt_sell_signal", None)
        mt_score = _safe_float(state.get("last_mt_pred_sellscore"), None)

        det_contrib = _safe_float(state.get("last_contrib_det"), None)
        ml_contrib = _safe_float(state.get("last_contrib_ml"), None)

        last_alert = state.get("last_alert_time", None)

        stock_list.append({
            "ticker": t,
            "weak_days": weak_days,
            "weak_req": weak_req,
            "avg_sell_index": float(avg_sell_index),
            "sell_index_raw": sell_index_raw,
            "emoji": risk_emoji,
            "pnl_lei": pnl_lei,
            "pnl_pct": pnl_pct,
            "ccy": ccy,
            "fx": float(fx_to_ron),
            "early_thr": float(early_thr),
            "strong_thr": float(strong_thr),
            "mt_regime": mt_regime_int,
            "mt_prob": mt_prob,
            "thr_used": thr_used,
            "gate_used": gate_used,
            "mt_sell": mt_sell,
            "mt_score": mt_score,
            "det_contrib": det_contrib,
            "ml_contrib": ml_contrib,
            "last_alert": last_alert,
            "sort_key": float(avg_sell_index),
        })

    stock_list.sort(key=lambda x: x["sort_key"], reverse=True)

    ticker_blocks = []
    for s in stock_list:
        pnl_sign = "+" if s["pnl_lei"] >= 0 else ""
        pct_txt = f" | PnL {float(s['pnl_pct']):+.2f}%" if s["pnl_pct"] is not None else ""

        lines = [
            f"**{s['ticker']}** — {s['emoji']}",
            f"    WeakDays {int(s['weak_days'])}/{int(s['weak_req'])} | AvgSellIndex {float(s['avg_sell_index']):.2f}{pct_txt}",
            f"    💰 Unrealized PnL: {pnl_sign}{s['pnl_lei']:.2f} RON ({s['ccy']}→RON fx={s['fx']:.4f})",
            "",
            f"    🧠 Thr: early {s['early_thr']:.2f} | strong {s['strong_thr']:.2f} | MT {int(s['mt_regime']):+d}",
        ]

        if s["sell_index_raw"] is not None:
            lines.append(f"    🧾 SellIndex_raw: {float(s['sell_index_raw']):.2f}")
        lines.append("")

        if s["det_contrib"] is not None or s["ml_contrib"] is not None:
            det_txt = f"{float(s['det_contrib']):.2f}" if s["det_contrib"] is not None else "n/a"
            ml_txt = f"{float(s['ml_contrib']):.2f}" if s["ml_contrib"] is not None else "n/a"
            lines.append(f"    🧩 Mix: Det {det_txt} | ML {ml_txt}")
            lines.append("")
        if s["mt_prob"] is not None:
            thr_used_txt = f"{float(s['thr_used']):.2f}" if s['thr_used'] is not None else "n/a"
            gate_used_txt = f"{float(s['gate_used']):.2f}" if s['gate_used'] is not None else "n/a"

            sell_txt = "SELL" if bool(s["mt_sell"]) else "HOLD"
            lines.append(
                f"    🤖 MT | {sell_txt} | P {float(s['mt_prob']):.3f} | thr_used {thr_used_txt}"
            )
            lines.append(f"    🔧 Gate {gate_used_txt}")
            lines.append("")

        ticker_blocks.append("\n".join(lines).rstrip())

    realized_pnl = float(data.get("realized_pnl", 0.0))
    pnl_sign_unrealized = "+" if total_unrealized_pnl >= 0 else ""
    pnl_sign_realized = "+" if realized_pnl >= 0 else ""
    totals = (
        f"📈 **Unrealized PnL (open positions):** {pnl_sign_unrealized}{total_unrealized_pnl:.2f} RON\n"
        f"💰 **Cumulative Realized PnL:** {pnl_sign_realized}{realized_pnl:.2f} RON"
    )

    header = f"**📊 Currently Tracked Stocks:** (v={BOT_VERSION})\n"
    await send_chunked_blocks(ctx, header, ticker_blocks, totals, limit=DISCORD_MSG_LIMIT)


@bot.command()
async def remove(ctx, ticker: str):
    t = ticker.upper().strip()
    data = load_data()
    stocks = data.get("stocks", {}) or {}

    if t not in stocks:
        await ctx.send(f"⚠️ {t} is not being tracked.")
        return

    del stocks[t]
    data["stocks"] = stocks
    save_data(data)
    push_to_github(DATA_FILE, f"REMOVE {t} without PnL tracking")

    await ctx.send(f"✅ Removed **{t}** from tracking (no PnL recorded)")


@bot.command()
async def pnl(ctx):
    data = load_data()
    await ctx.send(f"💰 **Cumulative Realized PnL:** {float(data.get('realized_pnl', 0.0)):.2f} RON")


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    if TOKEN:
        # Defensive trim in case env var has trailing whitespace/newlines (e.g., from file-based injection).
        TOKEN = TOKEN.strip()
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN is not set")
        raise SystemExit(1)

    print("✅ DISCORD_BOT_TOKEN found")

    # Run keep_alive only once, in a background thread.
    Thread(target=keep_alive, daemon=True).start()

    bot.run(TOKEN)
