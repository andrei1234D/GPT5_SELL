import os
import json
import discord
from discord.ext import commands

DATA_FILE = "bot/data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

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
async def buy(ctx, ticker: str, price: float):
    ticker = ticker.upper()
    data = load_data()
    data[ticker] = {"buy_price": price, "active": True}
    save_data(data)
    await ctx.send(f"✅ Now tracking **{ticker}** at ${price:.2f}")

@bot.command()
async def sell(ctx, ticker: str):
    ticker = ticker.upper()
    data = load_data()
    if ticker in data:
        data[ticker]["active"] = False
        save_data(data)
        await ctx.send(f"❌ Stopped tracking **{ticker}**")
    else:
        await ctx.send(f"⚠️ {ticker} not found in tracked list.")

@bot.command()
async def list(ctx):
    data = load_data()
    if not data:
        await ctx.send("📭 No stocks currently tracked.")
        return
    msg = "**📊 Currently Tracked Stocks:**\n"
    for t, info in data.items():
        status = "✅ ACTIVE" if info.get("active", True) else "❌ INACTIVE"
        msg += f"- {t}: Buy @ ${info['buy_price']} ({status})\n"
    await ctx.send(msg)

if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN is not set")
    else:
        bot.run(TOKEN)
