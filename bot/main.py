import sys
import json
import os
from tracker import add_stock

def main():
    if len(sys.argv) < 3:
        print("❌ Usage: python main.py <TICKER> <BUY_PRICE>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    try:
        buy_price = float(sys.argv[2])
    except ValueError:
        print("❌ Error: BUY_PRICE must be a number")
        sys.exit(1)

    print(f"📝 Adding stock: {ticker} at ${buy_price:.2f}")
    add_stock(ticker, buy_price)
    print("✅ Stock added successfully")

if __name__ == "__main__":
    main()
