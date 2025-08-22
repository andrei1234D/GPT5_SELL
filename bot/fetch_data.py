import yfinance as yf
import pandas as pd


def compute_indicators(ticker: str):
    """
    Fetch price history for a stock and compute technical indicators.
    Returns dict with current price, RSI, MA50, MA200, MACD, ATR, Bollinger Bands, etc.
    """

    try:
        ticker_data = yf.Ticker(ticker)

        # Get last 6 months daily data
        hist = ticker_data.history(period="6mo", interval="1d")

        if hist.empty:
            print(f"⚠️ No data for {ticker}")
            return None

        # Current price (latest close)
        current_price = hist["Close"].iloc[-1]

        # Moving Averages
        ma50 = hist["Close"].rolling(window=50).mean().iloc[-1]
        ma200 = hist["Close"].rolling(window=200).mean().iloc[-1]

        # RSI (14-day)
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        # MACD (12,26) + Signal (9)
        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        macd_val = macd.iloc[-1]
        macd_signal_val = macd_signal.iloc[-1]

        # ATR (14-day Average True Range)
        high_low = hist["High"] - hist["Low"]
        high_close = (hist["High"] - hist["Close"].shift()).abs()
        low_close = (hist["Low"] - hist["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        # Bollinger Bands (20-day)
        sma20 = hist["Close"].rolling(window=20).mean()
        stddev = hist["Close"].rolling(window=20).std()
        bb_upper = (sma20 + 2 * stddev).iloc[-1]
        bb_lower = (sma20 - 2 * stddev).iloc[-1]

        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "ma50": float(ma50),
            "ma200": float(ma200),
            "rsi": float(rsi),
            "macd": float(macd_val),
            "macd_signal": float(macd_signal_val),
            "atr": float(atr),
            "bb_upper": float(bb_upper),
            "bb_lower": float(bb_lower),
        }

    except Exception as e:
        print(f"❌ Error fetching indicators for {ticker}: {e}")
        return None


if __name__ == "__main__":
    # 🔎 Test with AAPL
    data = compute_indicators("AAPL")
    print(data)
