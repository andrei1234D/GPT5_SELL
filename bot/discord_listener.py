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
            return json.load(f)
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
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    # ✅ Fetch FX rate at purchase
    try:
        fx = yf.Ticker("USDRON=X").history(period="1d")
        fx_rate = float(fx["Close"].iloc[-1]) if not fx.empty else 4.6
    except:
        fx_rate = 4.6

    usd_invested = lei_invested / fx_rate
    shares_bought = usd_invested / price

    if ticker in stocks:
        old_price = float(stocks[ticker]["avg_price"])
        old_shares = float(stocks[ticker]["shares"])
        old_invested = float(stocks[ticker]["invested_lei"])

        new_shares = old_shares + shares_bought
        avg_price = ((old_price * old_shares) + (price * shares_bought)) / new_shares

        stocks[ticker]["avg_price"] = float(avg_price)
        stocks[ticker]["shares"] = float(new_shares)
        stocks[ticker]["invested_lei"] = float(old_invested + lei_invested)
        stocks[ticker]["fx_rate_buy"] = fx_rate
    else:
        stocks[ticker] = {
            "avg_price": float(price),       # in USD
            "shares": float(shares_bought),
            "invested_lei": float(lei_invested),
            "fx_rate_buy": fx_rate           # store FX at buy
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {price} (FX {fx_rate})")
    await ctx.send(
        f"✅ Now tracking **{ticker}** | Avg Buy Price: {stocks[ticker]['avg_price']:.2f} USD | "
        f"Shares: {stocks[ticker]['shares']:.2f} | Invested: {stocks[ticker]['invested_lei']:.2f} LEI "
        f"(FX at buy: {fx_rate:.2f})"
    )




@bot.command()
async def sell(ctx, ticker: str, price: float, lei_sold: float):
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks:
        await ctx.send(f"⚠️ {ticker} is not being tracked.")
        return

    avg_price = float(stocks[ticker]["avg_price"])
    invested = float(stocks[ticker]["invested_lei"])
    shares = float(stocks[ticker]["shares"])

    if lei_sold > invested:
        await ctx.send(f"⚠️ Cannot sell {lei_sold}, only {invested:.2f} LEI invested.")
        return

    shares_sold = lei_sold / avg_price
    pnl_per_share = price - avg_price
    total_pnl = pnl_per_share * shares_sold
    data["realized_pnl"] += total_pnl

    stocks[ticker]["shares"] -= shares_sold
    stocks[ticker]["invested_lei"] -= lei_sold

    if stocks[ticker]["shares"] <= 0:
        del stocks[ticker]

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {lei_sold} LEI of {ticker} at {price}")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Price: {price:.2f} | Amount Sold: {lei_sold:.2f} LEI\n"
        f"PnL: {total_pnl:+.2f} LEI\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI"
    )


@bot.command()
async def list(ctx):
    data = load_data()
    stocks = data["stocks"]
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return
    msg = "**📊 Currently Tracked Stocks:**\n"
    for t, info in stocks.items():
        msg += f"- {t}: Avg Price: {info['avg_price']:.2f} | Shares: {info['shares']:.2f} | Invested: {info['invested_lei']:.2f} LEI\n"
    msg += f"\n💰 **Cumulative Realized PnL:** {data['realized_pnl']:.2f} LEI"
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
        Thread(target=keep_alive).start()
        bot.run(TOKEN)
