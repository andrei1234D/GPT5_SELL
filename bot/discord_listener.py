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
        sha = r.json().get("sha")

        data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }
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
# Helpers
# ---------------------------
def get_fx_usdron(default=4.6):
    try:
        fx = yf.Ticker("USDRON=X").history(period="1d")
        return float(fx["Close"].iloc[-1]) if not fx.empty else default
    except Exception:
        return default


def smooth_fx_toward(old_fx: float, new_fx: float, weight: float) -> float:
    """
    Move old_fx toward new_fx by 'weight'. Weight should be in [0,1].
    Example: weight=0.1 -> new = 90% old + 10% new.
    """
    if weight <= 0:
        return old_fx
    if weight >= 1:
        return new_fx
    return old_fx * (1 - weight) + new_fx * weight


# ---------------------------
# Discord Bot Setup
# ---------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


def pull_from_github(file_path=DATA_FILE):
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
            with open(file_path, "w") as f:
                f.write(content)
            print(f"✅ Pulled latest {file_path} from GitHub")
        else:
            print("❌ GitHub pull failed:", res.text)
    except Exception as e:
        print(f"⚠️ Error pulling from GitHub: {e}")


@bot.event
async def on_ready():
    try:
        pull_from_github(DATA_FILE)
        pull_from_github(TRACKER_FILE)  # ✅ also sync tracker from GitHub
    except Exception as e:
        print(f"⚠️ Skipped GitHub pull at startup: {e}")
    print(f"✅ Logged in as {bot.user}")


# ---------------------------
# Bot Commands
# ---------------------------
@bot.command()
async def buy(ctx, ticker: str, price: float, lei_invested: float):
    """
    Buy in LEI at a given USD price. Shares are computed via USD amount using current USDRON FX.
    FX tracking (fx_rate_buy) is updated by smoothing toward today's FX based on the size of the buy
    relative to the total invested LEI after the transaction.
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    fx_rate = get_fx_usdron()
    usd_invested = lei_invested / fx_rate
    shares_bought = usd_invested / price

    if ticker in stocks:
        old_price = float(stocks[ticker]["avg_price"])
        old_shares = float(stocks[ticker]["shares"])
        old_invested = float(stocks[ticker]["invested_lei"])
        old_fx = float(stocks[ticker].get("fx_rate_buy", fx_rate))

        new_shares = old_shares + shares_bought
        new_invested = old_invested + lei_invested

        avg_price = ((old_price * old_shares) + (price * shares_bought)) / new_shares if new_shares > 0 else price

        weight = (lei_invested / new_invested) if new_invested > 0 else 1.0
        new_fx_smoothed = smooth_fx_toward(old_fx, fx_rate, max(0.0, min(1.0, weight)))

        stocks[ticker]["avg_price"] = float(avg_price)
        stocks[ticker]["shares"] = float(new_shares)
        stocks[ticker]["invested_lei"] = float(new_invested)
        stocks[ticker]["fx_rate_buy"] = float(new_fx_smoothed)
    else:
        stocks[ticker] = {
            "avg_price": float(price),
            "shares": float(shares_bought),
            "invested_lei": float(lei_invested),
            "fx_rate_buy": float(fx_rate)
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {price} (FX {fx_rate})")
    await ctx.send(
        f"✅ Now tracking **{ticker}** | Avg Buy Price: {stocks[ticker]['avg_price']:.2f} USD | "
        f"Shares: {stocks[ticker]['shares']:.4f} | Invested: {stocks[ticker]['invested_lei']:.2f} LEI "
        f"(FX ref: {stocks[ticker]['fx_rate_buy']:.4f}, latest: {fx_rate:.4f})"
    )


@bot.command()
async def sell(ctx, ticker: str, price: float, amount: str):
    """
    Sell at a given USD price.
    - 'amount' can be 'all' or a number in LEI (proceeds target).
    - Realized PnL is computed in LEI (includes FX).
    - FX reference is smoothed toward today's FX based on the proportion of invested LEI being sold.
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks:
        await ctx.send(f"⚠️ {ticker} is not being tracked.")
        return

    avg_price_usd = float(stocks[ticker]["avg_price"])
    invested_lei = float(stocks[ticker]["invested_lei"])
    total_shares = float(stocks[ticker]["shares"])
    fx_ref = float(stocks[ticker].get("fx_rate_buy", get_fx_usdron()))

    fx_sell = get_fx_usdron()

    if amount.lower() == "all":
        shares_sold = total_shares
        usd_proceeds = shares_sold * price
        lei_proceeds = usd_proceeds * fx_sell
    else:
        try:
            lei_proceeds = float(amount)
        except ValueError:
            await ctx.send("⚠️ Invalid amount. Use a number (LEI proceeds) or 'all'.")
            return
        if lei_proceeds <= 0:
            await ctx.send("⚠️ Amount must be positive.")
            return

        usd_proceeds = lei_proceeds / fx_sell
        shares_sold = usd_proceeds / price

        if shares_sold > total_shares + 1e-9:
            await ctx.send(f"⚠️ Not enough shares. You have {total_shares:.4f} shares.")
            return

    share_ratio = shares_sold / total_shares if total_shares > 0 else 0.0
    cost_basis_lei = invested_lei * share_ratio

    pnl_lei = lei_proceeds - cost_basis_lei
    data["realized_pnl"] += pnl_lei

    remaining_shares = total_shares - shares_sold
    if remaining_shares <= 1e-9:
        del stocks[ticker]
    else:
        weight = (cost_basis_lei / invested_lei) if invested_lei > 0 else 1.0
        new_fx_smoothed = smooth_fx_toward(fx_ref, fx_sell, max(0.0, min(1.0, weight)))

        stocks[ticker]["shares"] = float(remaining_shares)
        stocks[ticker]["invested_lei"] = float(invested_lei - cost_basis_lei)
        stocks[ticker]["avg_price"] = float(avg_price_usd)
        stocks[ticker]["fx_rate_buy"] = float(new_fx_smoothed)

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {amount.upper()} of {ticker} at {price} (FX {fx_sell})")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Price: {price:.2f} USD | Proceeds: {lei_proceeds:.2f} LEI | FX now: {fx_sell:.4f}\n"
        f"PnL (realized): {pnl_lei:+.2f} LEI\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI"
        + ("" if ticker not in stocks else f"\n🔁 FX ref after smoothing: {stocks[ticker]['fx_rate_buy']:.4f}")
    )


@bot.command()
async def list(ctx):
    """Show all currently tracked stocks with Weak Streak, Score, and optional MT+SellIndex diagnostics."""
    pull_from_github(DATA_FILE)
    pull_from_github(TRACKER_FILE)

    data = load_data()
    stocks = data.get("stocks", {})

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

    fx_live = get_fx_usdron()

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

        current_value_lei = current_price * shares * fx_live
        pnl_lei = current_value_lei - invested_lei
        total_unrealized_pnl += pnl_lei

        state = tracker.get("tickers", {}).get(ticker, {})
        weak_streak = float(state.get("weak_streak", 0.0))

        # ✅ Score: prefer persisted last_score, else average rolling_scores
        score = state.get("last_score", None)
        if score is None:
            rolling = state.get("rolling_scores", []) or []
            score = (sum(rolling) / len(rolling)) if rolling else 0.0
        score = float(score)

        # Optional newer fields from updated decision_engine
        sell_index = state.get("last_sell_index", None)
        mt_regime = state.get("last_mt_regime", None)
        mt_prob = state.get("last_mt_prob", None)
        mt_thr = state.get("last_mt_prob_thr", None)
        mt_gate = state.get("last_mt_gate", None)
        label = state.get("last_signal_label", None)
        checked = state.get("last_checked_time", state.get("last_checked_time", None))

        # Risk emoji based on deterministic score (keeps your existing mental model)
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
            "sell_index": sell_index,
            "mt_regime": mt_regime,
            "mt_prob": mt_prob,
            "mt_thr": mt_thr,
            "mt_gate": mt_gate,
            "label": label,
            "checked": checked
        })

    stock_list.sort(key=lambda x: x["score"], reverse=True)

    for s in stock_list:
        pnl_sign = "+" if s["pnl_lei"] >= 0 else ""
        msg_lines.append(
            f"**{s['ticker']}** — {s['emoji']}\n"
            f"    Weak {s['weak']:.1f}/3 | Score {s['score']:.1f}\n"
            f"    💰 Unrealized PnL: {pnl_sign}{s['pnl_lei']:.2f} LEI\n"
        )

        # ✅ Extra diagnostics (only if present)
        if s["sell_index"] is not None:
            try:
                msg_lines.append(f"    🧠 SellIndex: {float(s['sell_index']):.2f}\n")
            except Exception:
                pass

        if s["mt_regime"] is not None and s["mt_prob"] is not None:
            thr_txt = f"{float(s['mt_thr']):.2f}" if s["mt_thr"] is not None else "n/a"
            gate_txt = f"{float(s['mt_gate']):.2f}" if s["mt_gate"] is not None else "n/a"
            msg_lines.append(
                f"    📌 MT {int(s['mt_regime']):+d} | prob {float(s['mt_prob']):.2f} (thr {thr_txt}) | gate {gate_txt}\n"
            )

        if s["label"]:
            msg_lines.append(f"    🏷️ {s['label']}\n")

        if s["checked"]:
            msg_lines.append(f"    🕒 {s['checked']}\n")

    realized_pnl = float(data.get("realized_pnl", 0.0))
    pnl_sign_unrealized = "+" if total_unrealized_pnl >= 0 else ""
    pnl_sign_realized = "+" if realized_pnl >= 0 else ""

    msg_lines.append(f"📈 **Unrealized PnL (open positions):** {pnl_sign_unrealized}{total_unrealized_pnl:.2f} LEI")
    msg_lines.append(f"💰 **Cumulative Realized PnL:** {pnl_sign_realized}{realized_pnl:.2f} LEI")

    msg = "\n".join(msg_lines)

    if len(msg) > 1900:
        chunks = [msg[i:i+1900] for i in range(0, len(msg), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(msg)


@bot.command()
async def pnl(ctx):
    data = load_data()
    await ctx.send(f"💰 **Cumulative Realized PnL:** {data['realized_pnl']:.2f} LEI")


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
