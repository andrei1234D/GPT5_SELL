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
# Helpers
# ---------------------------
def get_fx_usdron(default=4.6):
    try:
        fx = yf.Ticker("USDRON=X").history(period="1d")
        return float(fx["Close"].iloc[-1]) if not fx.empty else default
    except Exception:
        return default


def smooth_fx_toward(old_fx: float, new_fx: float, weight: float) -> float:
    """
    Move old_fx toward new_fx by 'weight'. Weight should be in [0,1].
    Example: weight=0.1 -> new = 90% old + 10% new.
    """
    if weight <= 0:
        return old_fx
    if weight >= 1:
        return new_fx
    return old_fx * (1 - weight) + new_fx * weight


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
    Buy in LEI at a given USD price. Shares are computed via USD amount using current USDRON FX.
    FX tracking (fx_rate_buy) is updated by smoothing toward today's FX based on the size of the buy
    relative to the total invested LEI after the transaction.
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    # ✅ Fetch FX rate at purchase
    fx_rate = get_fx_usdron()

    usd_invested = lei_invested / fx_rate
    shares_bought = usd_invested / price

    if ticker in stocks:
        old_price = float(stocks[ticker]["avg_price"])
        old_shares = float(stocks[ticker]["shares"])
        old_invested = float(stocks[ticker]["invested_lei"])
        old_fx = float(stocks[ticker].get("fx_rate_buy", fx_rate))

        new_shares = old_shares + shares_bought
        new_invested = old_invested + lei_invested

        # ✅ Weighted average stock price (USD) by shares
        avg_price = ((old_price * old_shares) + (price * shares_bought)) / new_shares if new_shares > 0 else price

        # ✅ FX smoothing based on new buy vs total post-buy invested
        weight = (lei_invested / new_invested) if new_invested > 0 else 1.0
        new_fx_smoothed = smooth_fx_toward(old_fx, fx_rate, max(0.0, min(1.0, weight)))

        stocks[ticker]["avg_price"] = float(avg_price)
        stocks[ticker]["shares"] = float(new_shares)
        stocks[ticker]["invested_lei"] = float(new_invested)
        stocks[ticker]["fx_rate_buy"] = float(new_fx_smoothed)
    else:
        stocks[ticker] = {
            "avg_price": float(price),       # in USD
            "shares": float(shares_bought),
            "invested_lei": float(lei_invested),
            "fx_rate_buy": float(fx_rate)    # initial FX is today's buy FX
        }

    save_data(data)
    push_to_github(DATA_FILE, f"Bought {lei_invested} LEI of {ticker} at {price} (FX {fx_rate})")
    await ctx.send(
        f"✅ Now tracking **{ticker}** | Avg Buy Price: {stocks[ticker]['avg_price']:.2f} USD | "
        f"Shares: {stocks[ticker]['shares']:.4f} | Invested: {stocks[ticker]['invested_lei']:.2f} LEI "
        f"(FX ref: {stocks[ticker]['fx_rate_buy']:.4f}, latest: {fx_rate:.4f})"
    )


@bot.command()
async def sell(ctx, ticker: str, price: float, amount: str):
    """
    Sell at a given USD price.
    - 'amount' can be 'all' or a number in LEI (proceeds target).
    - Realized PnL is computed in LEI (includes FX).
    - FX reference is smoothed toward today's FX based on the proportion of invested LEI being sold.
    """
    ticker = ticker.upper()
    data = load_data()
    stocks = data["stocks"]

    if ticker not in stocks:
        await ctx.send(f"⚠️ {ticker} is not being tracked.")
        return

    avg_price_usd = float(stocks[ticker]["avg_price"])     # kept for info
    invested_lei = float(stocks[ticker]["invested_lei"])   # total LEI cost basis (pre-sell)
    total_shares = float(stocks[ticker]["shares"])
    fx_ref = float(stocks[ticker].get("fx_rate_buy", get_fx_usdron()))

    # ✅ Fetch FX at sell
    fx_sell = get_fx_usdron()

    # Determine shares_sold and lei_proceeds
    if amount.lower() == "all":
        shares_sold = total_shares
        usd_proceeds = shares_sold * price
        lei_proceeds = usd_proceeds * fx_sell
    else:
        try:
            lei_proceeds = float(amount)
        except ValueError:
            await ctx.send("⚠️ Invalid amount. Use a number (LEI proceeds) or 'all'.")
            return
        if lei_proceeds <= 0:
            await ctx.send("⚠️ Amount must be positive.")
            return
        # Convert LEI proceeds to USD, then to shares to sell (using SELL fx)
        usd_proceeds = lei_proceeds / fx_sell
        shares_sold = usd_proceeds / price

        if shares_sold > total_shares + 1e-9:
            await ctx.send(f"⚠️ Not enough shares. You have {total_shares:.4f} shares.")
            return

    # 🔢 Proportional LEI cost basis (captures FX at buys)
    share_ratio = shares_sold / total_shares if total_shares > 0 else 0.0
    cost_basis_lei = invested_lei * share_ratio

    # ✅ Realized PnL in LEI (includes FX effect)
    pnl_lei = lei_proceeds - cost_basis_lei
    data["realized_pnl"] += pnl_lei

    # 🧾 Update holdings
    remaining_shares = total_shares - shares_sold
    if remaining_shares <= 1e-9:
        # sold everything
        del stocks[ticker]
    else:
        # Smooth FX toward today's sell FX by ratio of cost_basis_lei vs total invested_lei
        weight = (cost_basis_lei / invested_lei) if invested_lei > 0 else 1.0
        new_fx_smoothed = smooth_fx_toward(fx_ref, fx_sell, max(0.0, min(1.0, weight)))

        stocks[ticker]["shares"] = float(remaining_shares)
        stocks[ticker]["invested_lei"] = float(invested_lei - cost_basis_lei)
        stocks[ticker]["avg_price"] = float(avg_price_usd)  # USD avg stays
        stocks[ticker]["fx_rate_buy"] = float(new_fx_smoothed)

    save_data(data)
    push_to_github(DATA_FILE, f"Sold {amount.upper()} of {ticker} at {price} (FX {fx_sell})")

    await ctx.send(
        f"💸 Sold **{ticker}**\n"
        f"Sell Price: {price:.2f} USD | Proceeds: {lei_proceeds:.2f} LEI | FX now: {fx_sell:.4f}\n"
        f"PnL (realized): {pnl_lei:+.2f} LEI\n"
        f"📊 Cumulative Realized PnL: {data['realized_pnl']:.2f} LEI"
        + ("" if ticker not in stocks else f"\n🔁 FX ref after smoothing: {stocks[ticker]['fx_rate_buy']:.4f}")
    )


@bot.command()
async def list(ctx):
    data = load_data()
    stocks = data["stocks"]
    if not stocks:
        await ctx.send("📭 No stocks currently tracked.")
        return

    # ✅ Fetch current FX
    fx_rate = get_fx_usdron()

    msg = "**📊 Currently Tracked Stocks:**\n"
    for t, info in stocks.items():
        avg_price = float(info["avg_price"])
        shares = float(info["shares"])
        invested = float(info["invested_lei"])

        # ✅ Fetch latest price
        try:
            px = yf.Ticker(t).history(period="1d")
            current_price = float(px["Close"].iloc[-1]) if not px.empty else avg_price
        except Exception:
            current_price = avg_price

        current_value_usd = current_price * shares
        current_value_lei = current_value_usd * fx_rate

        pnl_lei = current_value_lei - invested
        pnl_pct = (pnl_lei / invested * 100) if invested > 0 else 0

        msg += (
            f"{t}: Avg Buy: {avg_price:.2f} USD | Current: {current_value_lei:.2f} LEI "
            f"(PnL: {pnl_lei:+.2f} LEI / {pnl_pct:+.2f}%) | FX ref: {float(info.get('fx_rate_buy', fx_rate)):.4f}\n"
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
