import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive
import subprocess, sys
import base64
import requests
import yfinance as yf
from threading import Thread

# ✅ Auto-install required packages if missing
required = ["discord.py", "requests"]
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
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"stocks": {}, "realized_pnl": 0.0}


def save_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# GitHub API Push
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

        with open(file_path, "r") as f:
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


# ---------------------------
# FX helpers (multi-currency; cached)
# ---------------------------
_FX_CACHE = {}
_TICKER_CCY_CACHE = {}

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
    if ticker in _TICKER_CCY_CACHE:
        return _TICKER_CCY_CACHE[ticker]
    ccy = "USD"
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None) or {}
        ccy = fi.get("currency") or ccy
        if not ccy:
            info = getattr(t, "info", {}) or {}
            ccy = info.get("currency") or ccy
    except Exception:
        pass
    ccy = (ccy or "USD").upper().strip()
    _TICKER_CCY_CACHE[ticker] = ccy
    return ccy


def get_fx_to_ron(currency: str, default_usdron=4.6) -> float:
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return 1.0
    if c in _FX_CACHE:
        return _FX_CACHE[c]
    pair = _fx_pair_to_ron(c)
    if pair is None:
        _FX_CACHE[c] = 1.0
        return 1.0
    try:
        fx = yf.Ticker(pair).history(period="1d")
        if not fx.empty:
            v = float(fx["Close"].iloc[-1])
            if v > 0:
                _FX_CACHE[c] = v
                return v
    except Exception:
        pass
    fallback = default_usdron if c == "USD" else 1.0
    _FX_CACHE[c] = fallback
    return fallback


def pull_from_github(file_path=DATA_FILE):
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
            with open(file_path, "w") as f:
                f.write(content)
            print(f"✅ Pulled latest {file_path} from GitHub")
        else:
            print("❌ GitHub pull failed:", res.text)
    except Exception as e:
        print(f"⚠️ Error pulling from GitHub: {e}")


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
    print(f"✅ Logged in as {bot.user}")


# ---------------------------
# Bot Commands
# ---------------------------
@bot.command()
async def list(ctx):
    """
    Show tracked stocks with Weak Streak, deterministic Score, SellIndex, and ML diagnostics.
    Uses tracker fields written by decision_engine.py.
    """
    pull_from_github(DATA_FILE)
    pull_from_github(TRACKER_FILE)

    data = load_data()
    stocks = data.get("stocks", {}) or {}
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return

    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            try:
                tracker = json.load(f)
            except json.JSONDecodeError:
                tracker = {"tickers": {}}
    else:
        tracker = {"tickers": {}}

    msg_lines = ["**📊 Currently Tracked Stocks:**\n"]
    total_unrealized_pnl = 0.0
    stock_list = []

    for ticker, info in stocks.items():
        avg_price = float(info.get("avg_price", 0))
        shares = float(info.get("shares", 0))
        invested_lei = float(info.get("invested_lei", 0))

        try:
            px = yf.Ticker(ticker).history(period="1d")
            current_price = float(px["Close"].iloc[-1]) if not px.empty else avg_price
        except Exception:
            current_price = avg_price

        state = (tracker.get("tickers", {}) or {}).get(ticker, {}) or {}
        ccy = state.get("last_currency") or get_ticker_currency(ticker)
        fx_to_ron = state.get("last_fx_to_ron")
        if fx_to_ron is None:
            fx_to_ron = get_fx_to_ron(ccy)

        current_value_lei = current_price * shares * float(fx_to_ron)
        pnl_lei = current_value_lei - invested_lei
        total_unrealized_pnl += pnl_lei

        weak_streak = float(state.get("weak_streak", 0.0))
        score = state.get("last_score", None)
        if score is None:
            rolling = state.get("rolling_scores", []) or []
            score = (sum(rolling) / len(rolling)) if rolling else 0.0
        score = float(score)

        sell_index = state.get("last_sell_index", None)
        mt_regime = state.get("last_mt", None)
        mt_prob = state.get("last_mt_prob", None)
        mt_thr = state.get("last_mt_prob_thr", None)
        mt_score = state.get("last_mt_gate", None)   # show as "ML score"
        mt_sell = state.get("last_mt_sell_signal", None)
        mt_model = state.get("last_mt_model_type", None)
        mt_src = state.get("last_mt_prob_source", None)
        last_alert = state.get("last_alert_time", None)

        if score < 2.5:
            risk_emoji = "🟢 Stable"
        elif score < 4.5:
            risk_emoji = "🟡 Watch"
        elif score < 6.5:
            risk_emoji = "🟠 Weak"
        else:
            risk_emoji = "🔴 Critical"

        stock_list.append({
            "ticker": ticker,
            "weak": weak_streak,
            "score": score,
            "emoji": risk_emoji,
            "pnl_lei": pnl_lei,
            "ccy": ccy,
            "fx": float(fx_to_ron),
            "sell_index": sell_index,
            "mt_regime": mt_regime,
            "mt_prob": mt_prob,
            "mt_thr": mt_thr,
            "mt_score": mt_score,
            "mt_sell": mt_sell,
            "mt_model": mt_model,
            "mt_src": mt_src,
            "last_alert": last_alert,
        })

    stock_list.sort(key=lambda x: x["score"], reverse=True)

    for s in stock_list:
        pnl_sign = "+" if s["pnl_lei"] >= 0 else ""
        msg_lines.append(
            f"**{s['ticker']}** — {s['emoji']}\n"
            f"    Weak {s['weak']:.1f}/3 | Score {s['score']:.1f}\n"
            f"    💰 Unrealized PnL: {pnl_sign}{s['pnl_lei']:.2f} LEI ({s['ccy']}→RON fx={s['fx']:.4f})\n"
        )

        if s["sell_index"] is not None:
            try:
                msg_lines.append(f"    🧠 SellIndex: {float(s['sell_index']):.2f}\n")
            except Exception:
                pass

        if s["mt_regime"] is not None and s["mt_prob"] is not None:
            thr_txt = f"{float(s['mt_thr']):.2f}" if s["mt_thr"] is not None else "n/a"
            score_txt = f"{float(s['mt_score']):.2f}" if s["mt_score"] is not None else "n/a"
            sell_txt = "SELL" if bool(s["mt_sell"]) else "NO-SELL"
            model_txt = s["mt_model"] or "?"
            src_txt = s["mt_src"] or "?"
            msg_lines.append(
                f"    🤖 MT {int(s['mt_regime']):+d} | {sell_txt} | P {float(s['mt_prob']):.2f} (thr {thr_txt}) | "
                f"ML score {score_txt} | {model_txt} | src={src_txt}\n"
            )

        if s["last_alert"]:
            msg_lines.append(f"    🚨 Last alert: {s['last_alert']}\n")

    realized_pnl = float(data.get("realized_pnl", 0.0))
    pnl_sign_unrealized = "+" if total_unrealized_pnl >= 0 else ""
    pnl_sign_realized = "+" if realized_pnl >= 0 else ""

    msg_lines.append(f"📈 **Unrealized PnL (open positions):** {pnl_sign_unrealized}{total_unrealized_pnl:.2f} LEI")
    msg_lines.append(f"💰 **Cumulative Realized PnL:** {pnl_sign_realized}{realized_pnl:.2f} LEI")

    msg = "\n".join(msg_lines)
    if len(msg) > 1900:
        for chunk in [msg[i:i + 1900] for i in range(0, len(msg), 1900)]:
            await ctx.send(chunk)
    else:
        await ctx.send(msg)


@bot.command()
async def pnl(ctx):
    data = load_data()
    await ctx.send(f"💰 **Cumulative Realized PnL:** {float(data.get('realized_pnl', 0.0)):.2f} LEI")


# ---------------------------
# Run Bot
# ---------------------------
if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN is not set")
    else:
        print("✅ DISCORD_BOT_TOKEN found, length:", len(TOKEN))
        print("Preview:", TOKEN[:8], "...")
        Thread(target=keep_alive).start()
        bot.run(TOKEN)
