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

        # 🔄 Auto-migrate old flat format to new structured format
        if "stocks" not in data:
            print("⚠️ Migrating old data.json format to new schema...")
            data = {"stocks": data, "realized_pnl": 0.0}

            for t, info in data["stocks"].items():
                if "lei_invested" not in info:
                    info["lei_invested"] = info.get("qty", 0)
                if "shares" not in info:
                    info["shares"] = round(info["lei_invested"] / info["buy_price"], 4)
                if "active" not in info:
                    info["active"] = True
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
async def buy(ctx, ticker: str, price: float, lei: float):
    """Buy stock: price per share, lei invested"""
    ticker = ticker.upper()
    data = load_data()

    shares = round(lei / price, 4)

    data["stocks"][ticker] = {
        "buy_price": price,
        "lei_invested": lei,
        "shares": shares,
        "active": True
    }
    save_data(data)
    git_commit_and_push(f"Bought {ticker} with {lei} LEI at {price} ({shares} shares)")

    await ctx.send(
        f"✅ Now tracking **{ticker}** | Invested: {lei:.2f} LEI | "
        f"Buy Price: {price:.2f} | Shares: {shares}"
    )


@bot.command()
async def sell(ctx, ticker: str, price: float, lei: float):
    """Sell stock: price per share, lei amount to sell"""
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks or not stocks[ticker].get("active", True):
        await ctx.send(f"⚠️ {ticker} is not being tracked or already sold.")
        return

    buy_price = stocks[ticker]["buy_price"]
    lei_invested = stocks[ticker]["lei_invested"]

    if lei > lei_invested:
        await ctx.send(f"⚠️ Cannot sell {lei} LEI, only {lei_invested:.2f} LEI invested.")
        return

    shares_sold = round(lei / buy_price, 4)
    pnl_per_share = price - buy_price
    total_pnl = pnl_per_share * shares_sold

    # Deduct from holdings
    stocks[ticker]["lei_invested"] -= lei
    stocks[ticker]["shares"] -= shares_sold

    # If fully sold → delete ticker
    fully_sold = stocks[ticker]["lei_invested"] <= 0.01 or stocks[ticker]["shares"] <= 0.0001
    if fully_sold:
        del stocks[ticker]  # 👈 remove completely

    data["realized_pnl"] += total_pnl

    save_data(data)
    git_commit_and_push(f"Sold {ticker} for {lei} LEI at {price}")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Amount: {lei:.2f} LEI ({shares_sold:.4f} shares)\n"
        f"Buy Price: {buy_price:.2f} | Sell Price: {price:.2f}\n"
        f"PnL/share: {pnl_per_share:.2f} | Total PnL: {total_pnl:.2f} LEI\n"
        f"{'🗑️ Removed from tracking (fully sold)' if fully_sold else '📊 Still tracking remaining position'}\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI"
    )
    """Sell stock: price per share, lei amount to sell"""
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks or not stocks[ticker].get("active", True):
        await ctx.send(f"⚠️ {ticker} is not being tracked or already sold.")
        return

    buy_price = stocks[ticker]["buy_price"]
    lei_invested = stocks[ticker]["lei_invested"]
    shares_total = stocks[ticker]["shares"]

    if lei > lei_invested:
        await ctx.send(f"⚠️ Cannot sell {lei} LEI, only {lei_invested} LEI invested.")
        return

    shares_sold = round(lei / buy_price, 4)
    pnl_per_share = price - buy_price
    total_pnl = pnl_per_share * shares_sold

    # Deduct from holdings
    stocks[ticker]["lei_invested"] -= lei
    stocks[ticker]["shares"] -= shares_sold
    if stocks[ticker]["lei_invested"] <= 0.01:  # fully sold
        stocks[ticker]["active"] = False

    data["realized_pnl"] += total_pnl

    save_data(data)
    git_commit_and_push(f"Sold {ticker} for {lei} LEI at {price}")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Amount: {lei:.2f} LEI ({shares_sold} shares)\n"
        f"Buy Price: {buy_price:.2f} | Sell Price: {price:.2f}\n"
        f"PnL/share: {pnl_per_share:.2f} | Total PnL: {total_pnl:.2f} LEI\n"
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
        msg += (
            f"- {t}: Invested {info['lei_invested']:.2f} LEI | "
            f"Buy Price: {info['buy_price']} | Shares: {info['shares']:.4f} ({status})\n"
        )
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
