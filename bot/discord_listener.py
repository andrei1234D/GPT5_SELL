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

DATA_FILE = "bot/data.json"
keep_alive()

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
async def buy(ctx, ticker: str, price: float, lei_invested: float):
    """
    Buy stock (adds shares).
    Example: !buy AAPL 125 200  -> Buys 200 LEI worth of AAPL at 125
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    shares_bought = lei_invested / price

    if ticker in stocks:
        old_shares = stocks[ticker]["shares"]
        old_invested = stocks[ticker]["invested_lei"]

        new_shares = old_shares + shares_bought
        new_invested = old_invested + lei_invested

        # Weighted average price
        if new_shares != 0:
            avg_price = ((old_shares * stocks[ticker]["avg_price"]) + (shares_bought * price)) / new_shares
        else:
            avg_price = 0  

        stocks[ticker] = {
            "avg_price": avg_price,
            "shares": new_shares,
            "invested_lei": new_invested
        }
    else:
        stocks[ticker] = {
            "avg_price": price,
            "shares": shares_bought,
            "invested_lei": lei_invested
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {price}")
    await ctx.send(f"✅ Bought **{shares_bought:.2f} shares** of {ticker} @ {price:.2f} | Total Shares: {stocks[ticker]['shares']:.2f}")


@bot.command()
async def sell(ctx, ticker: str, price: float, lei_sold: float):
    """
    Sell stock (subtracts shares, supports shorting).
    Example: !sell AAPL 140 100 -> Sell 100 LEI worth of AAPL at 140
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    shares_sold = lei_sold / price

    if ticker not in stocks:
        # Initialize as short position
        stocks[ticker] = {
            "avg_price": price,
            "shares": -shares_sold,
            "invested_lei": -lei_sold
        }
        save_data(data)
        push_to_github(DATA_FILE, f"Opened short of {lei_sold} LEI ({shares_sold:.2f} shares) on {ticker} at {price}")
        await ctx.send(f"📉 Opened short on **{ticker}**: {shares_sold:.2f} shares @ {price:.2f}")
        return

    old_shares = stocks[ticker]["shares"]
    avg_price = stocks[ticker]["avg_price"]

    # PnL from this sale
    pnl = (price - avg_price) * shares_sold
    data["realized_pnl"] += pnl

    # Update shares
    new_shares = old_shares - shares_sold

    stocks[ticker]["shares"] = new_shares
    stocks[ticker]["invested_lei"] = new_shares * avg_price  # keep consistent

    if new_shares == 0:
        del stocks[ticker]  # fully closed position

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {lei_sold} LEI ({shares_sold:.2f} shares) of {ticker} at {price}")

    await ctx.send(
        f"💸 Sold **{shares_sold:.2f} shares** of {ticker} @ {price:.2f}\n"
        f"📊 Realized PnL: {pnl:+.2f} LEI\n"
        f"🔢 Remaining Shares: {new_shares:.2f}\n"
        f"💰 Total Realized PnL: {data['realized_pnl']:.2f} LEI"
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
