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
    """ Buy stock (supports reducing short positions if negative). """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    shares_bought = lei_invested / price

    if ticker in stocks:
        stock = stocks[ticker]
        prev_shares = stock.get("shares", 0)

        # If currently short and buying → reduce short or flip to long
        if prev_shares < 0:
            new_shares = prev_shares + shares_bought
            stock["shares"] = new_shares
            stock["invested_lei"] = new_shares * price
            if new_shares > 0:
                stock["buy_price"] = price
        else:
            # Normal averaging when long
            old_price = stock["buy_price"]
            old_shares = stock["shares"]

            new_shares = old_shares + shares_bought
            avg_price = ((old_price * old_shares) + (price * shares_bought)) / new_shares

            stock["buy_price"] = avg_price
            stock["shares"] = new_shares
            stock["invested_lei"] = new_shares * avg_price
    else:
        # New long position
        stocks[ticker] = {
            "buy_price": price,
            "shares": shares_bought,
            "invested_lei": lei_invested
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {price}")
    await ctx.send(
        f"✅ Now tracking **{ticker}** | Avg Buy Price: {stocks[ticker]['buy_price']:.2f} | "
        f"Shares: {stocks[ticker]['shares']:.2f} | Invested: {stocks[ticker]['invested_lei']:.2f} LEI"
    )


@bot.command()
async def sell(ctx, ticker: str, price: float, lei_sold: float):
    """ Sell stock (supports shorting if selling more than you own). """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    shares_sold = lei_sold / price

    if ticker not in stocks:
        # New short position
        stocks[ticker] = {
            "buy_price": price,
            "shares": -shares_sold,
            "invested_lei": -lei_sold
        }
        save_data(data)
        push_to_github(DATA_FILE, f"Opened short: Sold {lei_sold} LEI of {ticker} at {price}")
        await ctx.send(f"📉 Opened short position in **{ticker}** | Sold {lei_sold:.2f} LEI at {price:.2f}")
        return

    stock = stocks[ticker]
    prev_shares = stock.get("shares", 0)

    # Calculate PnL for this sale
    pnl_per_share = price - stock["buy_price"]
    total_pnl = pnl_per_share * shares_sold
    data["realized_pnl"] += total_pnl

    # Update shares
    new_shares = prev_shares - shares_sold
    stock["shares"] = new_shares
    stock["invested_lei"] = new_shares * price
    if new_shares < 0:  # went short
        stock["buy_price"] = price

    if abs(new_shares) < 1e-6:
        del stocks[ticker]  # position fully closed

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {lei_sold} LEI ({shares_sold:.2f} shares) of {ticker} at {price}")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Price: {price:.2f} | Amount Sold: {lei_sold:.2f} LEI ({shares_sold:.2f} shares)\n"
        f"PnL: {total_pnl:+.2f} LEI\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI\n"
        f"Remaining Shares: {new_shares:.2f}"
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
