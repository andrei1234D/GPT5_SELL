import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive
import subprocess, sys
import base64
import requests
import yfinance as yf
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
        old_fx = float(stocks[ticker]["fx_rate_buy"])

        new_shares = old_shares + shares_bought
        new_invested = old_invested + lei_invested

        # ✅ Weighted average stock price (USD)
        avg_price = ((old_price * old_shares) + (price * shares_bought)) / new_shares

        # ✅ Weighted average FX
        weighted_fx = ((old_fx * old_invested) + (fx_rate * lei_invested)) / new_invested

        stocks[ticker]["avg_price"] = float(avg_price)
        stocks[ticker]["shares"] = float(new_shares)
        stocks[ticker]["invested_lei"] = float(new_invested)
        stocks[ticker]["fx_rate_buy"] = float(weighted_fx)
    else:
        stocks[ticker] = {
            "avg_price": float(price),       # in USD
            "shares": float(shares_bought),
            "invested_lei": float(lei_invested),
            "fx_rate_buy": float(fx_rate)    # store FX at buy
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {price} (FX {fx_rate})")
    await ctx.send(
        f"✅ Now tracking **{ticker}** | Avg Buy Price: {stocks[ticker]['avg_price']:.2f} USD | "
        f"Shares: {stocks[ticker]['shares']:.2f} | Invested: {stocks[ticker]['invested_lei']:.2f} LEI "
        f"(FX at buy: {fx_rate:.2f})"
    )




@bot.command()
async def sell(ctx, ticker: str, price: float, amount: str):
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks:
        await ctx.send(f"⚠️ {ticker} is not being tracked.")
        return

    avg_price = float(stocks[ticker]["avg_price"])      # in USD
    invested = float(stocks[ticker]["invested_lei"])    # in LEI
    shares = float(stocks[ticker]["shares"])
    fx_buy = float(stocks[ticker].get("fx_rate_buy", 4.6))  # fallback

    # ✅ Fetch FX rate at sell
    try:
        fx = yf.Ticker("USDRON=X").history(period="1d")
        fx_sell = float(fx["Close"].iloc[-1]) if not fx.empty else 4.6
    except:
        fx_sell = 4.6

    # Handle "all" → sell everything at market
    if amount.lower() == "all":
        shares_sold = shares
        usd_sold = shares_sold * price
        lei_sold = usd_sold * fx_sell  # proceeds in LEI
    else:
        try:
            lei_sold = float(amount)
        except ValueError:
            await ctx.send("⚠️ Invalid amount. Use a number or 'all'.")
            return

        if lei_sold > invested:
            await ctx.send(f"⚠️ Cannot sell {lei_sold}, only {invested:.2f} LEI invested.")
            return

        # Convert LEI sold → USD (at SELL FX)
        usd_sold = lei_sold / fx_sell
        shares_sold = usd_sold / price

    # ✅ Realized PnL (USD → LEI)
    pnl_per_share_usd = price - avg_price
    total_pnl_usd = pnl_per_share_usd * shares_sold
    total_pnl_lei = total_pnl_usd * fx_sell  # realized in LEI

    data["realized_pnl"] += total_pnl_lei

    # Update stock holdings
    if amount.lower() == "all":
        del stocks[ticker]  # sold everything
    else:
        stocks[ticker]["shares"] -= shares_sold
        stocks[ticker]["invested_lei"] *= stocks[ticker]["shares"] / (stocks[ticker]["shares"] + shares_sold)

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {amount.upper()} of {ticker} at {price} (FX {fx_sell})")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Price: {price:.2f} USD | Amount Sold: {amount.upper()} ({lei_sold:.2f} LEI)\n"
        f"PnL: {total_pnl_lei:+.2f} LEI\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI"
    )


@bot.command()
async def list(ctx):
    data = load_data()
    stocks = data["stocks"]
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return

    # ✅ Fetch current FX
    try:
        fx = yf.Ticker("USDRON=X").history(period="1d")
        fx_rate = float(fx["Close"].iloc[-1]) if not fx.empty else 4.6
    except:
        fx_rate = 4.6

    msg = "**📊 Currently Tracked Stocks:**\n"
    for t, info in stocks.items():
        avg_price = float(info["avg_price"])
        shares = float(info["shares"])
        invested = float(info["invested_lei"])

        # ✅ Fetch latest price
        try:
            px = yf.Ticker(t).history(period="1d")
            current_price = float(px["Close"].iloc[-1]) if not px.empty else avg_price
        except:
            current_price = avg_price

        current_value_usd = current_price * shares
        current_value_lei = current_value_usd * fx_rate

        pnl_lei = current_value_lei - invested
        pnl_pct = (pnl_lei / invested * 100) if invested > 0 else 0

        # 🔹 Slimmed-down output
        msg += (
            f"{t}: Avg Buy: {avg_price:.2f} USD | Current: {current_value_lei:.2f} LEI "
            f"(PnL: {pnl_lei:+.2f} LEI / {pnl_pct:+.2f}%)\n"
        )

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
