import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive
import subprocess, sys

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
            return json.load(f)
    return {"stocks": {}, "realized_pnl": 0.0}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------------
# Git Auto Commit/Push
# ---------------------------
def git_commit_and_push(message="Auto-update data.json from Discord bot"):
    try:
        subprocess.run(["git", "config", "--global", "user.email", "bot@replit.com"])
        subprocess.run(["git", "config", "--global", "user.name", "Replit Bot"])

        subprocess.run(["git", "add", "bot/data.json"], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ Auto-committed and pushed data.json")
    except Exception as e:
        print(f"⚠️ Git push failed: {e}")

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
    """Buy stock with given LEI amount, auto-averaging buy price if ticker exists"""
    ticker = ticker.upper()
    data = load_data()

    if ticker in data["stocks"]:
        # Weighted average calculation
        old_price = data["stocks"][ticker]["buy_price"]
        old_invested = data["stocks"][ticker]["lei_invested"]

        new_avg_price = (
            (old_price * old_invested) + (price * lei_invested)
        ) / (old_invested + lei_invested)

        data["stocks"][ticker]["buy_price"] = new_avg_price
        data["stocks"][ticker]["lei_invested"] += lei_invested
        data["stocks"][ticker]["active"] = True
        msg = f"📈 Averaged buy for **{ticker}** | New Avg: {new_avg_price:.2f} LEI | Total Invested: {data['stocks'][ticker]['lei_invested']:.2f} LEI"
    else:
        data["stocks"][ticker] = {
            "buy_price": price,
            "lei_invested": lei_invested,
            "active": True
        }
        msg = f"✅ Bought **{ticker}** @ {price:.2f} | Invested: {lei_invested:.2f} LEI"

    save_data(data)
    git_commit_and_push(f"Bought {lei_invested} LEI of {ticker} at {price}")
    await ctx.send(msg)


@bot.command()
async def sell(ctx, ticker: str, price: float, lei_to_sell: float):
    """Sell stock by specifying LEI amount to sell"""
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks or not stocks[ticker].get("active", True):
        await ctx.send(f"⚠️ {ticker} is not being tracked or already sold.")
        return

    buy_price = stocks[ticker]["buy_price"]
    invested = stocks[ticker]["lei_invested"]

    if lei_to_sell > invested:
        await ctx.send(f"⚠️ Cannot sell {lei_to_sell}, only {invested:.2f} LEI invested.")
        return

    # Calculate PnL (proportional to LEI sold)
    pnl_pct = (price - buy_price) / buy_price
    total_pnl = pnl_pct * lei_to_sell
    data["realized_pnl"] += total_pnl

    # Reduce or remove ticker
    if lei_to_sell == invested:
        del data["stocks"][ticker]
        msg = f"💸 Fully sold **{ticker}** @ {price:.2f} | Sold {lei_to_sell:.2f} LEI | PnL: {total_pnl:.2f} LEI\n📊 Realized PnL: {data['realized_pnl']:.2f} LEI"
    else:
        stocks[ticker]["lei_invested"] -= lei_to_sell
        msg = f"💸 Partially sold **{ticker}** @ {price:.2f} | Sold {lei_to_sell:.2f} LEI | Remaining: {stocks[ticker]['lei_invested']:.2f} LEI | PnL: {total_pnl:.2f} LEI\n📊 Realized PnL: {data['realized_pnl']:.2f} LEI"

    save_data(data)
    git_commit_and_push(f"Sold {lei_to_sell} LEI of {ticker} at {price}")
    await ctx.send(msg)


@bot.command()
async def list(ctx):
    """List all tracked stocks"""
    data = load_data()
    stocks = data["stocks"]
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return

    msg = "**📊 Currently Tracked Stocks:**\n"
    for t, info in stocks.items():
        status = "✅ ACTIVE" if info.get("active", True) else "❌ INACTIVE"
        msg += f"- {t}: Avg Buy @ {info['buy_price']:.2f} | Invested: {info['lei_invested']:.2f} LEI ({status})\n"
    msg += f"\n💰 **Cumulative Realized PnL:** {data['realized_pnl']:.2f} LEI"
    await ctx.send(msg)


@bot.command()
async def pnl(ctx):
    """Show cumulative realized PnL"""
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
        keep_alive()
        bot.run(TOKEN)
