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
    """
    Buy a stock with LEI invested at a price.
    Example: !buy AAPL 125 200  (→ 200 LEI invested at 125)
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker in stocks and stocks[ticker]["active"]:
        old_price = stocks[ticker]["buy_price"]
        old_invested = stocks[ticker]["lei_invested"]

        # Weighted average new price
        new_invested = old_invested + lei_invested
        avg_price = ((old_price * (old_invested / old_price)) + (price * (lei_invested / price))) / ((old_invested / old_price) + (lei_invested / price))

        stocks[ticker]["buy_price"] = avg_price
        stocks[ticker]["lei_invested"] = new_invested
    else:
        stocks[ticker] = {
            "buy_price": price,
            "lei_invested": lei_invested,
            "active": True
        }

    save_data(data)
    git_commit_and_push(f"Bought {ticker} {lei_invested} LEI at {price}")
    await ctx.send(f"✅ Now tracking **{ticker}** | Avg Buy Price: {stocks[ticker]['buy_price']:.2f} | Invested: {stocks[ticker]['lei_invested']:.2f} LEI")


@bot.command()
async def sell(ctx, ticker: str, price: float, lei_sold: float):
    """
    Sell a stock for LEI.
    Example: !sell AAPL 140 100  (→ Sell 100 LEI worth at 140)
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks or not stocks[ticker]["active"]:
        await ctx.send(f"⚠️ {ticker} is not being tracked or already sold.")
        return

    buy_price = stocks[ticker]["buy_price"]
    invested = stocks[ticker]["lei_invested"]

    if lei_sold > invested:
        await ctx.send(f"⚠️ Cannot sell {lei_sold}, only {invested:.2f} LEI invested.")
        return

    # Calculate shares from LEI
    qty_sold = lei_sold / buy_price
    pnl_per_share = price - buy_price
    total_pnl = pnl_per_share * qty_sold
    data["realized_pnl"] += total_pnl

    # Update investment
    stocks[ticker]["lei_invested"] -= lei_sold
    if stocks[ticker]["lei_invested"] <= 0:
        del stocks[ticker]  # remove stock if fully sold

    save_data(data)
    git_commit_and_push(f"Sold {lei_sold} LEI of {ticker} at {price}")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Price: {price:.2f} | Amount Sold: {lei_sold:.2f} LEI\n"
        f"PnL: {total_pnl:+.2f} LEI\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI"
    )


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
        msg += f"- {t}: Avg Buy Price: {info['buy_price']:.2f} | Invested: {info['lei_invested']:.2f} LEI ({status})\n"
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
