# scripts/make_stage1_min.py
import pandas as pd

IN_PATH  = "live_results.csv"
OUT_PATH = "cristi.csv"

df = pd.read_csv(IN_PATH)

# Keep only these columns (all rows)
out = df[["Ticker","AvgSellIndex","SellThrEarly","SellThrStrong"]].copy()

out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(out)} rows -> {OUT_PATH}")
