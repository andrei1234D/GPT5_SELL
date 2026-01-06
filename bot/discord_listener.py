
import os
import json
import base64
import subprocess
import sys
import math
from threading import Thread
from datetime import datetime, timedelta
from typing import Optional, Tuple

import requests
import yfinance as yf
import discord
from discord.ext import commands

from keep_alive import keep_alive

# ---------------------------
# Version banner (helps confirm the running code)
# ---------------------------
BOT_VERSION = "2026-01-06 decision-engine-v9_1-ui"

# ✅ Auto-install required packages if missing
required = ["discord.py", "requests", "yfinance"]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

DATA_FILE = "bot/data.json"
TRACKER_FILE = "bot/sell_alerts_tracker.json"

keep_alive()

# ---------------------------
# Data Management
# ---------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stocks": {}, "realized_pnl": 0.0}


def save_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
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

        repo = "andrei1234D/GPT5_SELL"
        branch = "main"
        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        r = requests.get(api_url, headers={"Authorization": f"token {GH_TOKEN}"})
        sha = (r.json() or {}).get("sha")

        data = {"message": commit_message, "content": base64.b64encode(content.encode()).decode(), "branch": branch}
        if sha:
            data["sha"] = sha

        res = requests.put(api_url, json=data, headers={"Authorization": f"token {GH_TOKEN}"})
        if res.status_code in [200, 201]:
            print("✅ Pushed", file_path, "to GitHub")
        else:
            print("❌ GitHub push failed:", res.text)
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
            response_json = res.json()
            if "content" not in response_json:
                print("❌ GitHub response missing 'content':", response_json)
                return
            content = base64.b64decode(response_json["content"]).decode()

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Pulled latest {file_path} from GitHub")
        else:
            print("❌ GitHub pull failed:", res.text)
    except Exception as e:
        print(f"⚠️ Error pulling from GitHub: {e}")


# ---------------------------
# FX helpers (multi-currency; live-first, fallback to last-known)
# ---------------------------
_FX_CACHE = {}  # currency -> (rate_to_ron, ts_utc)
_TICKER_CCY_CACHE = {}  # ticker -> currency


def _fx_pair_to_ron(currency: str):
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return None
    if c == "USD":
        return "USDRON=X"
    if c == "EUR":
        return "EURRON=X"
    if c == "GBP":
        return "GBPRON=X"
    if c == "CHF":
        return "CHFRON=X"
    if c == "CAD":
        return "CADRON=X"
    return None


def get_ticker_currency(ticker: str) -> str:
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


def get_fx_to_ron(currency: str, *, fallback: Optional[float] = None, ttl_minutes: int = 20) -> float:
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
    pair = _fx_pair_to_ron(c)
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
def get_recent_price(ticker: str, fallback: float) -> float:
    """
    Attempts, in order:
      1) fast_info.last_price (if available)
      2) 1-minute interval last close for today
      3) daily history last close
      4) fallback
    """
    t = (ticker or "").upper().strip()
    try:
        yt = yf.Ticker(t)
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
        intraday = yf.Ticker(t).history(period="1d", interval="1m")
        if intraday is not None and (not intraday.empty) and ("Close" in intraday.columns):
            vv = float(intraday["Close"].iloc[-1])
            if vv > 0:
                return vv
    except Exception:
        pass

    try:
        px = yf.Ticker(t).history(period="1d")
        if px is not None and (not px.empty) and ("Close" in px.columns):
            vv = float(px["Close"].iloc[-1])
            if vv > 0:
                return vv
    except Exception:
        pass

    return float(fallback)


# ---------------------------
# Decision-engine-aligned thresholds (fallback only; prefer tracker fields when present)
# ---------------------------
# These mirror decision_engine_v9_1 defaults. If you override regime params in the engine,
# the tracker values (last_sell_thr_early/strong) will still be used first.
_BASE_THR_BY_MT = {-1: 0.64, 0: 0.63, 1: 0.61}
_STRONG_SELL_MULT = 1.25

# Profit-based easing (lower threshold => easier to trigger sell)
# V9_1 change: slightly reduced easing; removed the +0% tier.
_PROFIT_ADJ = [
    (50.0, -0.08),
    (30.0, -0.06),
    (10.0, -0.03),
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_sell_thresholds(pnl_pct: Optional[float], mt: int) -> Tuple[float, float]:
    """
    Fallback thresholds for UI, aligned to decision_engine_v9_1:
      - Profit adjustment applies to EARLY threshold only.
      - STRONG threshold = max(early*1.25, early+0.05), clamped.
    """
    mt = int(mt) if mt in (-1, 0, 1) else 0
    early = float(_BASE_THR_BY_MT.get(mt, 0.63))

    p = None
    try:
        if pnl_pct is not None and math.isfinite(float(pnl_pct)):
            p = float(pnl_pct)
    except Exception:
        p = None

    if p is not None:
        for cutoff, adj in _PROFIT_ADJ:
            if p >= float(cutoff):
                early += float(adj)
                break

    early = _clamp(early, 0.45, 0.90)

    strong = max(float(early) * float(_STRONG_SELL_MULT), float(early) + 0.05)
    strong = _clamp(strong, 0.50, 0.90)

    return float(early), float(strong)


def risk_label_from_sell_index(sell_index: Optional[float], early_thr: float) -> str:
    if sell_index is None:
        return "🟢 Stable"

    stable_cut = max(0.20, 0.40 * early_thr)
    watch_cut = max(stable_cut, 0.70 * early_thr)

    if sell_index < stable_cut:
        return "🟢 Stable"
    if sell_index < watch_cut:
        return "🟡 Watch"
    if sell_index < early_thr:
        return "🟠 Weak"
    return "🔴 Critical"


# ---------------------------
# Chunk-safe Discord message builder (NEVER splits a ticker block)
# ---------------------------
def _segments_len(segs: list[str]) -> int:
    if not segs:
        return 0
    return sum(len(s) for s in segs) + (len(segs) - 1)


async def send_chunked_blocks(ctx, header: str, blocks: list[str], totals: str, limit: int = 1900):
    """
    Build Discord messages from:
      - header (only in the first message)
      - blocks (ticker samples; never split)
      - totals (always appended to the final message)

    If totals don't fit at the end, move the last ticker block to a new message and append totals there.
    Repeat until it fits.
    """
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
        try:
            proceeds_lei = float(amount)
        except ValueError:
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


@bot.command()
async def list(ctx):
    pull_from_github(DATA_FILE)
    pull_from_github(TRACKER_FILE)

    data = load_data()
    stocks = data.get("stocks", {}) or {}
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return

    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            try:
                tracker = json.load(f)
            except json.JSONDecodeError:
                tracker = {"tickers": {}}
    else:
        tracker = {"tickers": {}}

    total_unrealized_pnl = 0.0
    stock_list = []

    for ticker, info in stocks.items():
        t = str(ticker).upper().strip()
        avg_price = float(info.get("avg_price", 0))
        shares = float(info.get("shares", 0))
        invested_lei = float(info.get("invested_lei", 0))

        current_price = get_recent_price(t, fallback=avg_price)

        state = (tracker.get("tickers", {}) or {}).get(t, {}) or {}

        ccy = (info.get("currency") or state.get("last_currency") or get_ticker_currency(t) or "USD").upper().strip()

        fx_fallback = None
        if state.get("last_fx_to_ron") is not None:
            fx_fallback = float(state.get("last_fx_to_ron"))
        elif state.get("last_fx_to_lei") is not None:
            fx_fallback = float(state.get("last_fx_to_lei"))
        elif info.get("fx_rate_buy") is not None:
            fx_fallback = float(info.get("fx_rate_buy"))

        fx_to_ron = get_fx_to_ron(ccy, fallback=fx_fallback)

        current_value_lei = current_price * shares * float(fx_to_ron)
        pnl_lei = current_value_lei - invested_lei
        total_unrealized_pnl += pnl_lei
        pnl_pct = (pnl_lei / invested_lei) * 100.0 if invested_lei > 0 else None

        # NEW (decision-engine aligned): show weak-days and avg sell-index when available.
        weak_days = None
        for k in ("weak_days", "last_weak_days", "weak_streak"):
            if state.get(k) is not None:
                try:
                    weak_days = float(state.get(k))
                    break
                except Exception:
                    pass
        if weak_days is None:
            weak_days = 0.0

        weak_req = None
        for k in ("weak_req", "last_weak_req", "WeakReq"):
            if state.get(k) is not None:
                try:
                    weak_req = int(state.get(k))
                    break
                except Exception:
                    pass
        if weak_req is None:
            weak_req = 3  # UI default; engine should ideally provide this

        sell_index_raw = state.get("last_sell_index", None)
        sell_index = None
        if sell_index_raw is not None:
            try:
                sell_index = float(sell_index_raw)
            except Exception:
                sell_index = None

        # Avg sell-index (rolling awareness). If missing, fall back to last sell-index.
        avg_sell_index = None
        for k in ("avg_sell_index", "last_avg_sell_index", "avgSellIndex"):
            if state.get(k) is not None:
                try:
                    avg_sell_index = float(state.get(k))
                    break
                except Exception:
                    pass
        if avg_sell_index is None:
            rolling = state.get("rolling_sell_index", []) or []
            if isinstance(rolling, list) and rolling:
                try:
                    vals = [float(x) for x in rolling if x is not None]
                    avg_sell_index = (sum(vals) / len(vals)) if vals else None
                except Exception:
                    avg_sell_index = None
        if avg_sell_index is None:
            avg_sell_index = sell_index if sell_index is not None else 0.0

        mt_regime = state.get("last_mt", state.get("last_mt_regime", 0))
        try:
            mt_regime_int = int(mt_regime)
        except Exception:
            mt_regime_int = 0
        if mt_regime_int not in (-1, 0, 1):
            mt_regime_int = 0

        # Prefer decision engine computed thresholds (if present in tracker).
        early_thr = None
        strong_thr = None
        if state.get("last_sell_thr_early") is not None:
            try:
                early_thr = float(state.get("last_sell_thr_early"))
            except Exception:
                early_thr = None
        if state.get("last_sell_thr_strong") is not None:
            try:
                strong_thr = float(state.get("last_sell_thr_strong"))
            except Exception:
                strong_thr = None

        if early_thr is None or strong_thr is None:
            early_thr_f, strong_thr_f = get_sell_thresholds(pnl_pct, mt_regime_int)
            early_thr = early_thr if early_thr is not None else early_thr_f
            strong_thr = strong_thr if strong_thr is not None else strong_thr_f

        risk_emoji = risk_label_from_sell_index(sell_index, float(early_thr))

        mt_prob = state.get("last_mt_prob", None)
        mt_thr = state.get("last_mt_prob_thr", None)
        mt_gate = state.get("last_mt_gate", None)
        mt_sell = state.get("last_mt_sell_signal", None)
        mt_model = state.get("last_mt_model_type", None)
        mt_src = state.get("last_mt_prob_source", None)
        last_alert = state.get("last_alert_time", None)

        stock_list.append({
            "ticker": t,
            "weak_days": weak_days,
            "weak_req": weak_req,
            "avg_sell_index": float(avg_sell_index),
            "emoji": risk_emoji,
            "pnl_lei": pnl_lei,
            "pnl_pct": pnl_pct,
            "ccy": ccy,
            "fx": float(fx_to_ron),
            "sell_index": sell_index,
            "sell_index_sort": sell_index if sell_index is not None else -1.0,
            "early_thr": float(early_thr),
            "strong_thr": float(strong_thr),
            "mt_regime": mt_regime_int,
            "mt_prob": mt_prob,
            "mt_thr": mt_thr,
            "mt_gate": mt_gate,
            "mt_sell": mt_sell,
            "mt_model": mt_model,
            "mt_src": mt_src,
            "last_alert": last_alert,
        })

    stock_list.sort(key=lambda x: x["sell_index_sort"], reverse=True)

    ticker_blocks = []
    for s in stock_list:
        pnl_sign = "+" if s["pnl_lei"] >= 0 else ""
        pct_txt = f" | PnL {float(s['pnl_pct']):+.2f}%" if s["pnl_pct"] is not None else ""

        lines = [
            f"**{s['ticker']}** — {s['emoji']}",
            f"    WeakDays {float(s['weak_days']):.1f}/{int(s['weak_req'])} | AvgSellIndex {float(s['avg_sell_index']):.2f}{pct_txt}",
            f"    💰 Unrealized PnL: {pnl_sign}{s['pnl_lei']:.2f} RON ({s['ccy']}→RON fx={s['fx']:.4f})",
            "",
        ]

        if s["sell_index"] is not None:
            lines.append(
                f"    🧠 SellIndex: {float(s['sell_index']):.2f} / {s['early_thr']:.2f} (strong {s['strong_thr']:.2f}) | MT {int(s['mt_regime']):+d}"
            )
            lines.append("")
        else:
            lines.append(f"    🧠 SellIndex: n/a / {s['early_thr']:.2f} (strong {s['strong_thr']:.2f}) | MT {int(s['mt_regime']):+d}")
            lines.append("")

        if s["mt_prob"] is not None:
            thr_txt = f"{float(s['mt_thr']):.2f}" if s["mt_thr"] is not None else "n/a"
            gate_txt = f"{float(s['mt_gate']):.2f}" if s["mt_gate"] is not None else "n/a"
            sell_txt = "SELL" if bool(s["mt_sell"]) else "NO-SELL"
            model_txt = s["mt_model"] or "?"
            src_txt = s["mt_src"] or "?"
            lines.append(
                f"    🤖 MT-brain | {sell_txt} | P {float(s['mt_prob']):.2f} (thr {thr_txt}) | Gate {gate_txt} | {model_txt} | src={src_txt}"
            )
            lines.append("")

        if s["last_alert"]:
            lines.append(f"    🚨 Last alert: {s['last_alert']}")
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
    await send_chunked_blocks(ctx, header, ticker_blocks, totals, limit=1900)


@bot.command()
async def pnl(ctx):
    data = load_data()
    await ctx.send(f"💰 **Cumulative Realized PnL:** {float(data.get('realized_pnl', 0.0)):.2f} RON")


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN is not set")
        raise SystemExit(1)

    print("✅ DISCORD_BOT_TOKEN found, length:", len(TOKEN))
    print("Preview:", TOKEN[:8], "...")
    Thread(target=keep_alive).start()
    bot.run(TOKEN)
