import yfinance as yf
import pandas as pd
import json

def compute_indicators(ticker: str):
    """
    Fetch price history for a stock and compute technical indicators.
    Robust version with detailed logging.
    """
    print(f"\n🔍 [START] Computing indicators for {ticker}")

    try:
        ticker_data = yf.Ticker(ticker)

        # Try multiple fallbacks for history
        hist = None
        for period in ["2y", "6mo", "1mo", "5d"]:
            try:
                print(f"🔄 Trying {period} history for {ticker}")
                hist = ticker_data.history(period=period, interval="1d")
                if not hist.empty:
                    print(f"✅ Data found for {ticker} ({len(hist)} rows) using period={period}")
                    break
            except Exception as e:
                print(f"❌ Error fetching {period} history for {ticker}: {e}")
        
        if hist is None or hist.empty:
            print(f"⚠️ No data available for {ticker}")
            return None

        current_price = float(hist["Close"].iloc[-1])
        print(f"💰 Current Price: {current_price}")

        # Moving Averages
        ma50 = hist["Close"].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
        ma200 = hist["Close"].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None

        # RSI
        rsi = None
        if len(hist) >= 14:
            delta = hist["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            if loss.iloc[-1] != 0:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))
        print(f"📈 RSI: {rsi}")

        # MACD
        macd = macd_signal = None
        if len(hist) >= 26:
            ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
            ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd = macd_line.iloc[-1]
            macd_signal = macd_line.ewm(span=9, adjust=False).mean().iloc[-1]
        print(f"📊 MACD: {macd}, Signal: {macd_signal}")

        # ATR
        atr = None
        if len(hist) >= 14:
            high_low = hist["High"] - hist["Low"]
            high_close = (hist["High"] - hist["Close"].shift()).abs()
            low_close = (hist["Low"] - hist["Close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
        print(f"⚡ ATR: {atr}")

        # Bollinger Bands
        bb_upper = bb_lower = None
        if len(hist) >= 20:
            sma20 = hist["Close"].rolling(window=20).mean()
            stddev = hist["Close"].rolling(window=20).std()
            bb_upper = (sma20 + 2 * stddev).iloc[-1]
            bb_lower = (sma20 - 2 * stddev).iloc[-1]
        print(f"📉 Bollinger Bands → Upper: {bb_upper}, Lower: {bb_lower}")

        # Momentum
        momentum = hist["Close"].diff().iloc[-1] if len(hist) >= 2 else None
        print(f"🚀 Momentum: {momentum}")

        # Volume → relative vs 20-day avg
        volume = None
        if "Volume" in hist.columns and len(hist) >= 20:
            today_vol = hist["Volume"].iloc[-1]
            avg_vol20 = hist["Volume"].tail(20).mean()
            volume = today_vol / avg_vol20 if avg_vol20 > 0 else None
        print(f"🔊 Relative Volume: {volume}")

        # Support & Resistance (20-day min/max)
        support = hist["Close"].tail(20).min() if len(hist) >= 20 else None
        resistance = hist["Close"].tail(20).max() if len(hist) >= 20 else None
        print(f"🛑 Support: {support}, 📈 Resistance: {resistance}")

        # Market Trend
        market_trend = None
        if ma50 and ma200:
            if ma50 > ma200: 
                market_trend = "BULLISH"
            elif ma50 < ma200:
                market_trend = "BEARISH"
            else:
                market_trend = "NEUTRAL"
        print(f"🌍 Market Trend: {market_trend}")

        result = {
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
            "volume": float(volume) if volume is not None else None,
            "support": float(support) if support is not None else None,
            "resistance": float(resistance) if resistance is not None else None,
            "market_trend": market_trend
        }

        print(f"✅ Final Indicators for {ticker}:\n{json.dumps(result, indent=2)}")
        return result

    except Exception as e:
        print(f"❌ Error computing indicators for {ticker}: {e}")
        return None
