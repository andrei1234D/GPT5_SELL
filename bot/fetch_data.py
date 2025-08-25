import yfinance as yf
import pandas as pd

def compute_indicators(ticker: str):
    """
    Fetch price history for a stock and compute technical indicators.
    Robust version with fallbacks and error handling.
    """
    try:
        ticker_data = yf.Ticker(ticker)

        # Try multiple fallbacks for history
        hist = None
        for period in ["6mo", "1mo", "5d"]:
            try:
                hist = ticker_data.history(period=period, interval="1d")
                if not hist.empty:
                    break
            except Exception:
                continue

        if hist is None or hist.empty:
            print(f"⚠️ No data for {ticker}")
            return None

        current_price = float(hist["Close"].iloc[-1])

        # Moving Averages
        ma50 = hist["Close"].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
        ma200 = hist["Close"].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None

        # RSI (14-day)
        rsi = None
        if len(hist) >= 14:
            delta = hist["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            if loss.iloc[-1] != 0:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))

        # MACD
        macd = macd_signal = None
        if len(hist) >= 26:
            ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
            ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd = macd_line.iloc[-1]
            macd_signal = macd_line.ewm(span=9, adjust=False).mean().iloc[-1]

        # ATR (14-day)
        atr = None
        if len(hist) >= 14:
            high_low = hist["High"] - hist["Low"]
            high_close = (hist["High"] - hist["Close"].shift()).abs()
            low_close = (hist["Low"] - hist["Close"].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

        # Bollinger Bands (20-day)
        bb_upper = bb_lower = None
        if len(hist) >= 20:
            sma20 = hist["Close"].rolling(window=20).mean()
            stddev = hist["Close"].rolling(window=20).std()
            bb_upper = (sma20 + 2 * stddev).iloc[-1]
            bb_lower = (sma20 - 2 * stddev).iloc[-1]

        # Momentum (latest price change)
        momentum = hist["Close"].diff().iloc[-1] if len(hist) >= 2 else None

        # Volume
        volume = hist["Volume"].iloc[-1] if "Volume" in hist.columns else None

        # Market Trend
        market_trend = None
        if ma50 and ma200:
            market_trend = "BULLISH" if ma50 > ma200 else "BEARISH"

        return {
            "ticker": ticker,
            "current_price": current_price,
            "ma50": float(ma50) if ma50 is not None else None,
            "ma200": float(ma200) if ma200 is not None else None,
            "rsi": float(rsi) if rsi is not None else None,
            "macd": float(macd) if macd is not None else None,
            "macd_signal": float(macd_signal) if macd_signal is not None else None,
            "atr": float(atr) if atr is not None else None,
            "bb_upper": float(bb_upper) if bb_upper is not None else None,
            "bb_lower": float(bb_lower) if bb_lower is not None else None,
            "momentum": float(momentum) if momentum is not None else None,
            "volume": int(volume) if volume is not None else None,
            "market_trend": market_trend
        }

    except Exception as e:
        print(f"❌ Error fetching indicators for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # 🔎 Debug Test
    for t in ["AAPL", "MSFT", "NVDA", "ONDS"]:
        data = compute_indicators(t)
        print(f"{t}: {data}")
