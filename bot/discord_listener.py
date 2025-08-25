import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive
import subprocess, sys
import base64
import requests
from threading import Thread
import yfinance as yf

# ✅ Auto-install required packages if missing
required = ["discord.py", "requests"]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

DATA_FILE = "bot/data.json"
keep_alive()



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
        return 4.5


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
    Buy stock with LEI → converted to USD to compute shares.
    Example: !buy AAPL 200 1000  (1000 LEI at $200/share)
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    usd_to_ron = get_usd_to_ron()
    usd_invested = lei_invested / usd_to_ron   # convert LEI → USD
    shares_bought = usd_invested / usd_price   # shares bought in USD market

    if ticker in stocks:
        old_shares = stocks[ticker]["shares"]
        old_price = stocks[ticker]["avg_price"]

        new_total_shares = old_shares + shares_bought
        new_avg_price = ((old_price * old_shares) + (usd_price * shares_bought)) / new_total_shares

        stocks[ticker]["shares"] = new_total_shares
        stocks[ticker]["avg_price"] = new_avg_price
    else:
        stocks[ticker] = {
            "avg_price": usd_price,
            "shares": shares_bought
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {usd_price}$")
    await ctx.send(
        f"✅ Bought **{ticker}**\n"
        f"Price: {usd_price}$ | FX: 1$={usd_to_ron:.2f} LEI\n"
        f"Shares Bought: {shares_bought:.4f} | Total Shares: {stocks[ticker]['shares']:.4f}"
    )


@bot.command()
async def sell(ctx, ticker: str, usd_price: float, lei_amount: float):
    """
    Sell stock using LEI → converted to USD for share calculation.
    Example: !sell AAPL 220 500  (Sell 500 LEI worth at $220)
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    usd_to_ron = get_usd_to_ron()
    usd_amount = lei_amount / usd_to_ron
    shares_to_sell = usd_amount / usd_price

    if ticker not in stocks:
        # Short position
        stocks[ticker] = {
            "avg_price": usd_price,
            "shares": -shares_to_sell
        }
        save_data(data)
        push_to_github(DATA_FILE, f"Shorted {lei_amount} LEI of {ticker} at {usd_price}$")
        await ctx.send(f"📉 Opened short on **{ticker}** | Shares: -{shares_to_sell:.4f} at {usd_price}$")
        return

    old_shares = stocks[ticker]["shares"]
    avg_price = stocks[ticker]["avg_price"]

    pnl = shares_to_sell * (usd_price - avg_price) if old_shares > 0 else shares_to_sell * (avg_price - usd_price)
    data["realized_pnl"] += pnl

    new_shares = old_shares - shares_to_sell
    if new_shares == 0:
        del stocks[ticker]
    else:
        stocks[ticker]["shares"] = new_shares

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {lei_amount} LEI of {ticker} at {usd_price}$")
    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Price: {usd_price}$ | FX: 1$={usd_to_ron:.2f} LEI\n"
        f"Shares Sold: {shares_to_sell:.4f} | Remaining: {new_shares:.4f}\n"
        f"PnL: {pnl:+.2f}$ | 📊 Total Realized PnL: {data['realized_pnl']:.2f}$"
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
