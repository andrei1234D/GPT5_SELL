"""
Shared knobs between decision_engine and discord_listener.
Keep this file small and only include values used by both.
"""

# File paths
TRACKER_FILE = "bot/sell_alerts_tracker.json"

# Threshold tuning
PROFIT_ADJ = [
    (50.0, -0.08),
    (30.0, -0.06),
    (10.0, -0.03),
]

STRONG_SELL_MULT = 1.25

# FX cache TTL (minutes)
FX_TTL_MINUTES = 30

# Discord output/rendering knobs (UI-only)
DISCORD_MSG_LIMIT = 1900
DEFAULT_WEAK_REQ = 5
UI_BASE_THR_BY_MT = {-1: 0.64, 0: 0.63, 1: 0.61}
UI_BASE_THR_DEFAULT = 0.63
UI_THR_EARLY_MIN = 0.45
UI_THR_EARLY_MAX = 0.90
UI_THR_STRONG_MIN = 0.50
UI_THR_STRONG_MAX = 0.95
UI_THR_STRONG_MIN_ADD = 0.05
RISK_STABLE_MIN = 0.20
RISK_STABLE_FRAC = 0.40
RISK_WATCH_FRAC = 0.70

# Market regime bias (S&P 500)
SPX_TICKER = "^GSPC"
SPX_DAILY_DOWN_PCT = -1.5
SPX_WEEKLY_DOWN_PCT = -3.5
SPX_DAILY_UP_PCT = 1.0
SPX_WEEKLY_UP_PCT = 2.5
MARKET_BIAS_FEAR = 0.05
MARKET_BIAS_GREED = -0.07


def fx_pair_to_ron(currency: str):
    c = (currency or "").upper().strip()
    if c in ("RON", "LEI"):
        return None
    if c == "USD":
        return "USDRON=X"
    if c == "EUR":
        return "EURRON=X"
    if c == "GBP":
        return "GBPRON=X"
    if c == "CHF":
        return "CHFRON=X"
    if c == "CAD":
        return "CADRON=X"
    return None


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def profit_based_weak_req(pnl_pct, base_req: int) -> int:
    """
    Match decision_engine logic:
      - PnL >= 50%  -> require 4 weak days
      - PnL >= 100% -> require 3 weak days
    Only lowers, never raises.
    """
    req = int(base_req)
    p = _safe_float(pnl_pct, None)
    if p is None:
        return req
    if p >= 100.0:
        return max(0, req - 2)
    if p >= 50.0:
        return max(0, req - 1)
    return req
