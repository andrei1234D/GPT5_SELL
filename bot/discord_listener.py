import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive
import subprocess, sys



required = ["discord.py", "requests"]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])




DATA_FILE = "bot/data.json"
keep_alive()
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"stocks": {}, "realized_pnl": 0.0}  # ✅ root has stocks + pnl



def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Discord bot
intents = discord.Intents.default()
intents.message_content = True  # required
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.command()
async def buy(ctx, ticker: str, price: float, qty: int):
    """Add a stock with buy price and qty"""
    ticker = ticker.upper()
    data = load_data()
    data["stocks"][ticker] = {
        "buy_price": price,
        "qty": qty,
        "active": True
    }
    save_data(data)
    await ctx.send(f"✅ Now tracking **{ticker}** | Bought @ {price:.2f} LEI | Qty: {qty}")

@bot.command()
async def sell(ctx, ticker: str, price: float, qty: int):
    """Sell a stock, record sell price, qty, and PnL"""
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks or not stocks[ticker].get("active", True):
        await ctx.send(f"⚠️ {ticker} is not being tracked or already sold.")
        return

    buy_price = stocks[ticker]["buy_price"]
    buy_qty = stocks[ticker]["qty"]

    # Prevent oversell
    if qty > buy_qty:
        await ctx.send(f"⚠️ Cannot sell {qty}, only {buy_qty} available.")
        return

    stocks[ticker]["sell_price"] = price
    stocks[ticker]["sell_qty"] = qty
    stocks[ticker]["active"] = False if qty == buy_qty else True
    if qty < buy_qty:  # partial sell
        stocks[ticker]["qty"] -= qty

    # Calculate PnL
    pnl_per_share = price - buy_price
    total_pnl = pnl_per_share * qty

    # Update cumulative PnL
    data["realized_pnl"] += total_pnl

    save_data(data)

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Buy Price: {buy_price:.2f} LEI | Qty Bought: {buy_qty}\n"
        f"Sell Price: {price:.2f} LEI | Qty Sold: {qty}\n"
        f"PnL/share: {pnl_per_share:.2f} LEI\n"
        f"Total PnL: {total_pnl:.2f} LEI\n"
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
        msg += f"- {t}: Buy @ {info['buy_price']} LEI | Qty: {info.get('qty', '?')} ({status})\n"
    msg += f"\n💰 **Cumulative Realized PnL:** {data['realized_pnl']:.2f} LEI"
    await ctx.send(msg)

@bot.command()
async def pnl(ctx):
    """Show cumulative realized PnL"""
    data = load_data()
    await ctx.send(f"💰 **Cumulative Realized PnL:** {data['realized_pnl']:.2f} LEI")

if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN is not set")
    else:
        keep_alive()  # 👈 keeps the bot running on Replit
        bot.run(TOKEN)