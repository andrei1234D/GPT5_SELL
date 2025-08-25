import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive
import subprocess, sys
import base64
import requests
from threading import Thread




# ✅ Auto-install required packages if missing
required = ["discord.py", "requests"]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

keep_alive()


DATA_FILE = "bot/data.json"



# ---------------------------
# Currency Conversion
# ---------------------------
def get_usd_to_ron():
    try:
        fx = yf.Ticker("USDRON=X")
        rate = fx.history(period="1d")["Close"].iloc[-1]
        return float(rate)
    except Exception as e:
        print(f"⚠️ FX fetch failed, defaulting to 1. Error: {e}")
        return 1.0


# ---------------------------
# Data Management
# ---------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        if "stocks" not in data:
            data = {"stocks": {}, "realized_pnl": 0.0}
            save_data(data)
        if "realized_pnl" not in data:
            data["realized_pnl"] = 0.0
            save_data(data)

        return data
    return {"stocks": {}, "realized_pnl": 0.0}


def save_data(data):
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

        # Get current file SHA
        r = requests.get(api_url, headers={"Authorization": f"token {GH_TOKEN}"})
        sha = r.json().get("sha")

        # Prepare payload
        data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }
        if sha:
            data["sha"] = sha

        res = requests.put(api_url, json=data, headers={"Authorization": f"token {GH_TOKEN}"})
        if res.status_code in [200, 201]:
            print("✅ Pushed data.json to GitHub")
        else:
            print("❌ GitHub push failed:", res.text)
    except Exception as e:
        print(f"⚠️ Error pushing to GitHub: {e}")

# ---------------------------
# Discord Bot Setup
# ---------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

# ---------------------------
# Bot Commands
# ---------------------------
@bot.command()
async def buy(ctx, ticker: str, usd_price: float, lei_invested: float):
    """
    Buy stock in LEI (USD price auto-converted).
    Example: !buy AAPL 200 1000
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    usd_to_ron = get_usd_to_ron()
    lei_price = usd_price / usd_to_ron   # convert to LEI

    shares_bought = lei_invested / lei_price

    if ticker in stocks:
        old_shares = stocks[ticker]["shares"]
        old_price = stocks[ticker]["avg_price"]

        new_total_shares = old_shares + shares_bought
        if new_total_shares != 0:
            new_avg_price = ((old_price * old_shares) + (lei_price * shares_bought)) / new_total_shares
        else:
            new_avg_price = 0

        stocks[ticker]["shares"] = new_total_shares
        stocks[ticker]["avg_price"] = new_avg_price
        stocks[ticker]["invested_lei"] = new_total_shares * new_avg_price
    else:
        stocks[ticker] = {
            "avg_price": lei_price,
            "shares": shares_bought,
            "invested_lei": shares_bought * lei_price
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {usd_price}$ ({lei_price:.2f} LEI)")
    await ctx.send(
        f"✅ Bought **{ticker}**\n"
        f"Price: {usd_price}$ → {lei_price:.2f} LEI\n"
        f"Shares: {shares_bought:.4f} | Avg Price: {stocks[ticker]['avg_price']:.2f} LEI | "
        f"Total Shares: {stocks[ticker]['shares']:.4f}"
    )


@bot.command()
async def sell(ctx, ticker: str, usd_price: float, lei_sold: float):
    """
    Sell stock (USD price auto-converted).
    Example: !sell AAPL 220 500
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    usd_to_ron = get_usd_to_ron()
    lei_price = usd_price / usd_to_ron
    shares_to_sell = lei_sold / lei_price

    if ticker not in stocks:
        stocks[ticker] = {
            "avg_price": lei_price,
            "shares": -shares_to_sell,
            "invested_lei": -shares_to_sell * lei_price
        }
        save_data(data)
        push_to_github(DATA_FILE, f"Shorted {lei_sold} LEI of {ticker} at {usd_price}$")
        await ctx.send(f"📉 Opened short on **{ticker}** | Sold {shares_to_sell:.4f} shares at {lei_price:.2f} LEI")
        return

    old_shares = stocks[ticker]["shares"]
    avg_price = stocks[ticker]["avg_price"]

    pnl = shares_to_sell * (lei_price - avg_price) if old_shares > 0 else shares_to_sell * (avg_price - lei_price)
    data["realized_pnl"] += pnl

    new_shares = old_shares - shares_to_sell

    if new_shares == 0:
        del stocks[ticker]
    else:
        stocks[ticker]["shares"] = new_shares
        stocks[ticker]["invested_lei"] = new_shares * avg_price

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {lei_sold} LEI of {ticker} at {usd_price}$")
    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Price: {usd_price}$ → {lei_price:.2f} LEI | Shares Sold: {shares_to_sell:.4f}\n"
        f"PnL: {pnl:+.2f} LEI | 📊 Realized PnL: {data['realized_pnl']:.2f} LEI\n"
        f"Remaining Shares: {new_shares:.4f}"
    )


@bot.command()
async def list(ctx):
    """ List all tracked stocks (long or short). """
    data = load_data()
    stocks = data["stocks"]
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return
    msg = "**📊 Currently Tracked Stocks:**\n"
    for t, info in stocks.items():
        position = "LONG" if info["shares"] > 0 else "SHORT"
        msg += f"- {t}: {position} | Shares: {info['shares']:.2f} | Avg Price: {info['buy_price']:.2f} | Invested: {info['invested_lei']:.2f} LEI\n"
    msg += f"\n💰 **Cumulative Realized PnL:** {data['realized_pnl']:.2f} LEI"
    await ctx.send(msg)


@bot.command()
async def pnl(ctx):
    """ Show cumulative realized PnL """
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
        Thread(target=keep_alive).start()
        bot.run(TOKEN)
