import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import yfinance as yf

from tracker import load_data
from notify import send_discord_alert
from fetch_data import compute_indicators
from llm_predict import SellBrain  # MT Brain integration (bear/neutral/bull)

TRACKER_FILE = "bot/sell_alerts_tracker.json"



# ML debug controls
ML_DEBUG = os.getenv("ML_DEBUG", "0").strip() in ("1", "true", "True", "yes", "YES")
ML_DUMP_DIR = os.getenv("ML_DUMP_DIR", "bot/ml_debug")

def ensure_dir(path: str):
    """Create directory if missing (idempotent)."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Could not create directory '{path}': {e}")

# ---------------------------
# Helpers
# ---------------------------
_FX_CACHE: Dict[str, float] = {}
_TICKER_CCY_CACHE: Dict[str, str] = {}


def _ensure_parent_dir(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def get_fx_to_lei(currency: str) -> float:
    """
    Fetch live FX conversion rate for <currency>/RON using yfinance FX tickers, with caching.

    Notes:
      - For USD => "USDRON=X"
      - For EUR => "EURRON=X"
      - For GBP => "GBPRON=X"
      - For CAD => "CADRON=X" (may not exist for all providers)
    If FX is unavailable, falls back to:
      - 1.0 for RON
      - 4.6 for unknown (logs a warning)
    """
    if not currency:
        currency = "USD"

    ccy = currency.upper().strip()
    if ccy == "RON":
        return 1.0

    if ccy in _FX_CACHE:
        return _FX_CACHE[ccy]

    fx_ticker = f"{ccy}RON=X"
    try:
        fx = yf.Ticker(fx_ticker).history(period="5d", interval="1d")
        if fx is not None and not fx.empty and "Close" in fx.columns:
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 0:
                _FX_CACHE[ccy] = rate
                return rate
    except Exception as e:
        print(f"[WARN] FX fetch failed for {fx_ticker}: {e}")

    # Fallback
    fallback = 4.6 if ccy == "USD" else 4.6
    print(f"[WARN] Using fallback FX rate for {ccy}->RON: {fallback} (FX ticker '{fx_ticker}' unavailable)")
    _FX_CACHE[ccy] = fallback
    return fallback


def get_ticker_currency(ticker: str) -> str:
    """
    Best-effort currency detection via yfinance with caching.
    Returns a 3-letter currency code (e.g., 'USD', 'CAD', 'EUR'), defaulting to 'USD'.
    """
    t = ticker.strip().upper()
    if t in _TICKER_CCY_CACHE:
        return _TICKER_CCY_CACHE[t]

    ccy = "USD"
    try:
        yt = yf.Ticker(t)
        # Prefer fast_info when available
        fi = getattr(yt, "fast_info", None)
        if fi and isinstance(fi, dict):
            ccy = str(fi.get("currency") or ccy).upper()
        else:
            info = getattr(yt, "info", None) or {}
            ccy = str(info.get("currency") or ccy).upper()
    except Exception as e:
        print(f"[WARN] Currency lookup failed for {t}: {e}")

    if not ccy or len(ccy) < 3:
        ccy = "USD"

    _TICKER_CCY_CACHE[t] = ccy
    return ccy


def load_tracker() -> Dict[str, Any]:
    """Load or initialize persistent tracker."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "tickers" not in data or not isinstance(data["tickers"], dict):
                data["tickers"] = {}
            if "date" not in data:
                data["date"] = datetime.utcnow().strftime("%Y-%m-%d")
            if "had_alerts" not in data:
                data["had_alerts"] = False
            return data
        except Exception as e:
            print(f"[WARN] Failed to read tracker JSON, reinitializing: {e}")

    return {"date": datetime.utcnow().strftime("%Y-%m-%d"), "had_alerts": False, "tickers": {}}


def save_tracker(data: Dict[str, Any]) -> None:
    """Save persistent state (ensures parent directory exists)."""
    _ensure_parent_dir(TRACKER_FILE)
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _run(cmd, check: bool = False) -> int:
    """
    Run a shell command and return its return code.
    Uses check=False by default because git may legitimately fail (e.g. nothing to commit).
    """
    try:
        r = subprocess.run(cmd, check=check)
        return int(r.returncode)
    except Exception as e:
        print(f"[WARN] Command failed: {cmd} | {e}")
        return 1


def git_commit_tracker() -> bool:
    """
    Auto-commit tracker + outputs to GitHub repo.
    Returns True if push succeeded, False otherwise.
    """
    print("[GIT] Committing updated tracker, results, and MT datasets...")

    _run(["git", "config", "--global", "user.email", "bot@github.com"])
    _run(["git", "config", "--global", "user.name", "AutoBot"])

    files_to_commit = [
        "bot/sell_alerts_tracker.json",
        "bot/live_results.csv",
        "bot/LLM_data/input_llm/llm_input_latest.csv",
        "bot/LLM_data/input_llm/llm_predictions.csv",
    ]

    for file_path in files_to_commit:
        if os.path.exists(file_path):
            _run(["git", "add", file_path])
        else:
            print(f"[GIT] Skipping missing file: {file_path}")

    commit_msg = f"Auto-update tracker + MT data [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}]"
    rc_commit = _run(["git", "commit", "-m", commit_msg])

    # "nothing to commit" is not an error for our purposes
    if rc_commit != 0:
        print("[GIT] Commit step returned non-zero (often means nothing to commit). Continuing...")

    rc_pull = _run(["git", "pull", "--rebase"])
    if rc_pull != 0:
        print("[GIT] WARNING: pull --rebase failed (continuing to push may fail).")

    rc_push = _run(["git", "push"])
    if rc_push == 0:
        print("[GIT] Push succeeded.")
        return True

    print("[GIT] Push failed.")
    return False


def infer_market_trend(indicators: dict) -> int:
    """
    Production-safe MarketTrend proxy if you don't already compute MarketTrend upstream.

    Returns:
      -1 = bear, 0 = neutral, 1 = bull
    """
    ma50 = indicators.get("ma50")
    ma200 = indicators.get("ma200")
    if ma50 is None or ma200 is None:
        return 0

    try:
        ma50 = float(ma50)
        ma200 = float(ma200)
    except Exception:
        return 0

    if ma200 <= 0:
        return 0

    # 1% band to avoid flip-flopping
    band = 0.01 * ma200
    if ma50 > ma200 + band:
        return 1
    if ma50 < ma200 - band:
        return -1
    return 0


def _safe_float(x):
    """Convert to float if possible; otherwise None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def append_ml_trace(trace: dict):
    """Append one ML trace row to a daily CSV file for post-run verification."""
    ensure_dir(ML_DUMP_DIR)
    # Partition by date to keep files small
    day = datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = os.path.join(ML_DUMP_DIR, f"ml_trace_{day}.csv")

    # Stable columns (keep order)
    cols = [
        "ts_utc", "ticker", "market_trend",
        "current_price", "pnl_pct",
        "mt_prob", "mt_prob_threshold", "mt_gate", "mt_weight",
        "pred_sellscore", "sell_threshold", "model_type",
        "signal_from_mt", "decision", "sell_index",
    ]
    row = {c: trace.get(c) for c in cols}
    # Add any extras (e.g., model ids)
    for k, v in trace.items():
        if k not in row:
            row[k] = v

    # Write header only if new file
    write_header = not os.path.exists(csv_path)
    try:
        import csv
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[WARN] Could not append ML trace to {csv_path}: {e}")

# ---------------------------
# Scoring Engine (no direct SELL return)
# ---------------------------

def load_latest_ml_predictions(pred_csv_path: str) -> dict:
    """Load latest per-ticker ML predictions from a CSV produced by the MT pipeline.

    Returns:
        dict[ticker] -> row dict with typed fields.

    Notes:
      - If multiple rows exist for a ticker, we keep the most recent Timestamp (lexicographic ISO-8601).
      - This is used to ensure the on-run logs and decisions match the precomputed `llm_predictions.csv`.
    """
    if not pred_csv_path or not os.path.exists(pred_csv_path):
        return {}

    out: dict = {}
    try:
        with open(pred_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = (row.get("Ticker") or "").strip()
                if not t:
                    continue

                ts = (row.get("Timestamp") or "").strip()
                prev = out.get(t)
                if prev is not None:
                    prev_ts = (prev.get("Timestamp") or "").strip()
                    if prev_ts and ts and ts <= prev_ts:
                        continue  # keep newest

                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                def _to_int(x):
                    try:
                        return int(float(x))
                    except Exception:
                        return None

                def _to_bool(x):
                    s = str(x).strip().lower()
                    if s in ("true", "1", "yes", "y", "t"):
                        return True
                    if s in ("false", "0", "no", "n", "f"):
                        return False
                    return None

                out[t] = {
                    "Timestamp": ts,
                    "MarketTrend": _to_int(row.get("MarketTrend")),
                    "mt_prob": _to_float(row.get("mt_prob")),
                    "mt_prob_threshold": _to_float(row.get("mt_prob_threshold")),
                    "mt_sell_signal": _to_bool(row.get("mt_sell_signal")),
                    "mt_gate": _to_float(row.get("mt_gate")),
                    "mt_weight": _to_float(row.get("mt_weight")),
                    "pred_sellscore": _to_float(row.get("pred_sellscore")),
                    "sell_threshold": _to_float(row.get("sell_threshold")),
                    "model_type": (row.get("model_type") or "").strip() or None,
                }
    except Exception as e:
        print(f"⚠️ [ML] Failed to load predictions CSV '{pred_csv_path}': {e}")
        return {}

    return out

def check_sell_conditions(
    ticker: str,
    buy_price: float,
    current_price: float,
    pnl_pct: Optional[float] = None,
    volume: Optional[float] = None,
    momentum: Optional[float] = None,
    rsi: Optional[float] = None,
    ma50: Optional[float] = None,
    ma200: Optional[float] = None,
    atr: Optional[float] = None,
    macd: Optional[float] = None,
    macd_signal: Optional[float] = None,
    resistance: Optional[float] = None,
    support: Optional[float] = None,
    info: Optional[Dict[str, Any]] = None,
    debug: bool = True,
) -> Tuple[bool, str, float, float]:
    """
    Returns:
      hard_sell (bool): True if hard stop-loss should force a sell
      message (str): diagnostic text
      current_price (float)
      avg_score (float): rolling average rule score used for fusion
    """
    if info is None:
        info = {}

    info.setdefault("weak_streak", 0.0)
    info.setdefault("recent_peak", current_price)
    info.setdefault("rolling_scores", [])
    info.setdefault("last_decay_date", None)
    info.setdefault("was_above_47", False)

    # --- Hard Stop Loss (-25%) ---
    if pnl_pct is not None and pnl_pct <= -25:
        if (rsi is not None) and (rsi < 35):
            return False, f"Oversold (RSI={rsi:.1f}), HOLD.", current_price, 0.0
        if (momentum is not None) and (momentum >= 0):
            return False, "Momentum stabilizing → HOLD.", current_price, 0.0
        return True, "Hard Stop Loss Triggered (-25%).", current_price, 10.0  # force high score

    score, reasons = 0.0, []

    # --- Core scoring factors ---
    if momentum is not None:
        if momentum < -0.8:
            score += 2.0; reasons.append("Momentum Collapse (< -0.8)")
        elif momentum < -0.3:
            score += 1.0; reasons.append("Weak Momentum (< -0.3)")

    if rsi is not None:
        if rsi < 35:
            score += 1.5; reasons.append("RSI Oversold (<35)")
        elif rsi < 45:
            score += 1.0; reasons.append("RSI Weak (<45)")
        elif rsi > 70:
            score += 0.5; reasons.append("RSI Overbought (>70)")

    if (macd is not None) and (macd_signal is not None) and (macd < macd_signal):
        score += 1.5; reasons.append("MACD Bearish Crossover")

    if (ma50 is not None) and (current_price < ma50):
        score += 1.0; reasons.append("Below MA50")

    if (ma200 is not None) and (current_price < ma200):
        score += 2.0; reasons.append("Below MA200 (Major Breakdown)")

    if (support is not None) and (current_price < support):
        score += 2.0; reasons.append("Support Broken")

    if (volume is not None) and (volume > 1.3):
        score += 1.0; reasons.append("High Volume Breakdown (>1.3x avg)")

    if (atr is not None) and (pnl_pct is not None) and (atr > 7) and (pnl_pct < 0):
        score += 0.5; reasons.append("High Volatility + Loss")

    # --- Rolling scores and weak streak ---
    rolling = info.get("rolling_scores", [])
    rolling.append(float(score))
    if len(rolling) > 7:
        rolling.pop(0)
    info["rolling_scores"] = rolling
    avg_score = sum(rolling) / len(rolling) if rolling else float(score)

    info["recent_peak"] = max(float(info.get("recent_peak", current_price)), float(current_price))

    # --- Weak streak logic ---
    now_utc = datetime.utcnow()
    market_hour = 13 <= now_utc.hour <= 21  # simple US session proxy

    if pnl_pct is not None:
        if pnl_pct >= 47:
            info["was_above_47"] = True
        elif pnl_pct < 29:
            info["was_above_47"] = False

    if market_hour and info.get("last_decay_date") != now_utc.strftime("%Y-%m-%d"):
        info["weak_streak"] = max(0.0, float(info.get("weak_streak", 0.0)) - 0.5)
        info["last_decay_date"] = now_utc.strftime("%Y-%m-%d")

    quiet_market = (atr is not None and atr < 3) and (volume is not None and volume < 0.7)
    ws = float(info.get("weak_streak", 0.0))

    if score >= 6.5:
        ws += 2.0
    elif score >= 4.0:
        ws += 1.0 if not quiet_market else 0.5
    elif score >= 3.0 and quiet_market:
        ws += 0.5
    elif (momentum is not None) and (momentum > 0.4) and (rsi is not None) and (rsi > 50):
        ws = 0.0
    elif score < 3.0 and ws > 0:
        ws -= 0.5

    info["weak_streak"] = max(0.0, round(ws, 1))

    if debug:
        print(f"[RULE] {ticker}: weak_streak={info['weak_streak']:.1f}/3 | score={score:.1f} | avg_score={avg_score:.1f}")

    return False, "Holding steady.", float(current_price), float(avg_score)


# ---------------------------
# Runner with Fusion Logic (Deterministic + MT Brain Gate)
# ---------------------------
def run_decision_engine(test_mode: bool = False, end_of_day: bool = False) -> None:
    file_to_load = "bot/test_data.csv" if test_mode else "bot/data.json"
    tracker = load_tracker()

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if tracker.get("date") != today_str:
        print(f"[INFO] Resetting daily alert flag for {today_str}")
        tracker["date"] = today_str
        tracker["had_alerts"] = False

    tracked = load_data(file_to_load)
    if not tracked or "stocks" not in tracked:
        print(f"[WARN] No tracked stocks found in {file_to_load}")
        return

    try:
        sell_brain = SellBrain()
        print("[INFO] MT SELL brain loaded successfully (bear/neutral/bull).")
    except Exception as e:
        print(f"[WARN] Could not load MT brain: {e}")
        sell_brain = None

    stocks = tracked["stocks"]
    sell_alerts = []

    # --- Optional: load precomputed MT predictions (keeps logs/decisions consistent with `llm_predictions.csv`) ---
    ml_pred_csv = "bot/LLM_data/input_llm/llm_predictions.csv"
    ml_preds = load_latest_ml_predictions(ml_pred_csv)
    if ml_preds:
        print(f"🤖 [ML] Using precomputed predictions from {ml_pred_csv} (tickers={len(ml_preds)})")
    else:
        print(f"🤖 [ML] No precomputed predictions found at {ml_pred_csv} (will infer live if possible)")

    def context_tag(pnl: Optional[float]) -> str:
        if pnl is None: return "Neutral"
        if pnl > 50: return "Massive Gain Softening"
        if pnl > 30: return "Big Gain Cooling Off"
        if pnl > 10: return "Profit Losing Strength"
        if pnl > 0: return "Minor Gain Under Stress"
        if pnl > -5: return "Slight Loss Control"
        return "Drawdown Risk"

    for ticker, info in stocks.items():
        avg_price = float(info.get("avg_price", 0) or 0)
        invested_lei = float(info.get("invested_lei", 0) or 0)
        shares = float(info.get("shares", 0) or 0)
        if avg_price <= 0 or invested_lei <= 0 or shares <= 0:
            continue

        indicators = compute_indicators(ticker)
        if not indicators:
            continue

        current_price = float(indicators.get("current_price") or 0.0)
        if current_price <= 0:
            continue

        # Currency-aware PnL
        ccy = str(indicators.get("currency") or get_ticker_currency(ticker)).upper()
        fx_to_lei = get_fx_to_lei(ccy)
        current_value_lei = current_price * shares * fx_to_lei
        pnl_lei = current_value_lei - invested_lei
        pnl_pct = (pnl_lei / invested_lei) * 100.0

        info_state = tracker["tickers"].get(ticker, {})

        hard_sell, hard_msg, _, avg_score = check_sell_conditions(
            ticker, avg_price, current_price,
            pnl_pct=pnl_pct,
            volume=indicators.get("volume"),
            momentum=indicators.get("momentum"),
            rsi=indicators.get("rsi"),
            ma50=indicators.get("ma50"),
            ma200=indicators.get("ma200"),
            atr=indicators.get("atr"),
            macd=indicators.get("macd"),
            macd_signal=indicators.get("macd_signal"),
            resistance=indicators.get("resistance"),
            support=indicators.get("support"),
            info=info_state,
            debug=True,
        )

        weak_streak = float(info_state.get("weak_streak", 0.0))

        # --- MarketTrend (proxy; replace with your own later) ---
        mt = infer_market_trend(indicators)

        # --- MT Brain probability (single source of truth) ---
        # Prefer the precomputed CSV (same artifact you inspect), otherwise fall back to live inference.
        ml_source = "none"

        mt_prob = None
        mt_prob_thr = None
        mt_sell_signal = False
        mt_gate = 0.0
        mt_weight = 0.0
        pred_sellscore = None
        sell_threshold = None
        model_type = None

        # 1) Use precomputed predictions if available (keeps logs consistent with `llm_predictions.csv`)
        row_pred = ml_preds.get(ticker) if isinstance(ml_preds, dict) else None
        if row_pred:
            ml_source = "csv"
            mt_csv = row_pred.get("MarketTrend")
            if mt_csv in (-1, 0, 1) and int(mt_csv) != int(mt):
                print(f"⚠️ [ML] {ticker}: MarketTrend mismatch (live={mt:+d} vs csv={int(mt_csv):+d}) → using csv regime.")
                mt = int(mt_csv)

            mt_prob = row_pred.get("mt_prob")
            mt_prob_thr = row_pred.get("mt_prob_threshold")
            mt_sell_signal = bool(row_pred.get("mt_sell_signal")) if row_pred.get("mt_sell_signal") is not None else False
            mt_gate = row_pred.get("mt_gate") or 0.0
            mt_weight = row_pred.get("mt_weight") or 0.0
            pred_sellscore = row_pred.get("pred_sellscore")
            sell_threshold = row_pred.get("sell_threshold")
            model_type = row_pred.get("model_type")

        # 2) Live inference fallback (only if we are confident the payload matches training feature space)
        if (ml_source != "csv") and sell_brain:
            try:
                # Defensive: verify feature coverage against the trained pipeline feature list (meta json).
                required = []
                try:
                    required = list((sell_brain._meta.get(int(mt), {}) or {}).get("feature_columns") or [])
                except Exception:
                    required = []

                missing = [c for c in required if c not in indicators] if required else []
                missing_rate = (len(missing) / max(len(required), 1)) if required else 0.0

                if required and missing_rate > 0.20:
                    # If we pass a heavily-missing payload, the imputer collapses to the training mean → near-constant preds.
                    # Better to disable ML than to emit misleading "confident" numbers.
                    print(
                        f"⚠️ [ML] {ticker}: payload missing {len(missing)}/{len(required)} features ({missing_rate:.0%}). "
                        f"Disabling live ML for this ticker. Example missing: {missing[:8]}"
                    )
                else:
                    ml_source = "live"
                    pred = sell_brain.predict(indicators, market_trend=mt)
                    if isinstance(pred, dict):
                        mt_prob = pred.get("mt_prob")
                        mt_prob_thr = pred.get("mt_prob_threshold")
                        mt_sell_signal = bool(pred.get("mt_sell_signal")) if pred.get("mt_sell_signal") is not None else False
                        mt_gate = pred.get("mt_gate") if pred.get("mt_gate") is not None else 0.0
                        mt_weight = pred.get("mt_weight") or 0.0
                        pred_sellscore = pred.get("pred_sellscore")
                        sell_threshold = pred.get("sell_threshold")
                        model_type = pred.get("model_type")

                        # If the upstream did not compute gate / sell_signal, compute deterministically here.
                        if (mt_prob is not None) and (mt_prob_thr is not None):
                            mt_sell_signal = bool(mt_prob >= mt_prob_thr) if pred.get("mt_sell_signal") is None else mt_sell_signal
                            if pred.get("mt_gate") is None:
                                if mt_prob <= mt_prob_thr:
                                    mt_gate = 0.0
                                else:
                                    mt_gate = (mt_prob - mt_prob_thr) / max(1e-9, (1.0 - mt_prob_thr))
                    else:
                        print(f"⚠️ [ML] {ticker}: unexpected SellBrain output type: {type(pred)}")
            except Exception as e:
                print(f"⚠️ [ML] Live prediction failed for {ticker}: {e}")

        # Final clamps
        try:
            mt_gate = float(mt_gate or 0.0)
        except Exception:
            mt_gate = 0.0
        mt_gate = max(0.0, min(mt_gate, 1.0))

        try:
            mt_weight = float(mt_weight or 0.0)
        except Exception:
            mt_weight = 0.0
        mt_weight = max(0.0, min(mt_weight, 0.85))

        # --- Fusion logic ---
        rule_norm = min(1.0, float(avg_score) / 10.0)

        # Weights: deterministic vs ML vs fixed bias.
        # Keep weights stable and bounded; ensures w_rule+w_mt+w_bias == 1.0.
        w_bias = 0.15
        w_mt = float(mt_weight or 0.0)
        w_mt = max(0.0, min(w_mt, 1.0 - w_bias))
        w_rule = 1.0 - w_bias - w_mt

        sell_index = (w_rule * rule_norm) + (w_mt * float(mt_gate or 0.0)) + w_bias
        sell_index = max(0.0, min(sell_index, 1.0))

        # Softening: borderline deterministic SELL is softened when ML is present but NOT confident.
        mt_softened = False
        if 6.0 <= float(avg_score) < 7.0 and (mt_prob is not None) and (mt_prob_thr is not None) and (not bool(mt_sell_signal)):
            sell_index = max(0.0, sell_index - 0.15)
            mt_softened = True

        # Hard stop-loss should override everything
        if hard_sell:
            sell_index = 1.0

        # === Determine decision zone ===
        if hard_sell:
            decision = True
            signal_label = "HARD STOP LOSS"
            color_tag = "🛑"
        elif sell_index >= 0.75 and weak_streak >= 1.0:
            decision = True
            signal_label = "STRONG SELL SIGNAL"
            color_tag = "🔴"
        elif sell_index >= 0.60:
            decision = True
            signal_label = "EARLY SELL ALERT"
            color_tag = "🟠"
        else:
            decision = False
            signal_label = "HOLD / WATCH MODE"
            color_tag = "🟢"

        pnl_context = context_tag(pnl_pct)

        # Friendly MT line (handles missing MT gracefully)
        if mt_prob is None:
            mt_line = f"🤖 [ML/{ml_source}] MT={mt:+d}: unavailable | Rule={rule_norm:.2f}"
        else:
            thr_txt = f"{mt_prob_thr:.2f}" if mt_prob_thr is not None else "n/a"
            ss_txt = f"{pred_sellscore:.3f}" if pred_sellscore is not None else "n/a"
            st_txt = f"{sell_threshold:.3f}" if sell_threshold is not None else "n/a"
            mt_line = (
                f"🤖 [ML/{ml_source}] MT={mt:+d} model={model_type or 'n/a'} w={w_mt:.2f} | "
                f"P={float(mt_prob):.3f} thr={thr_txt} sell={bool(mt_sell_signal)} gate={mt_gate:.2f} | "
                f"pred={ss_txt} vs sell_thr={st_txt} | Rule={rule_norm:.2f}"
            )

        soft_line = "MT softened borderline SELL." if mt_softened else ""
        hard_line = f"Hard reason: {hard_msg}" if hard_sell else ""

        print(
            f"{color_tag} {ticker}: {signal_label} | SellIndex={sell_index:.2f} | Weak={weak_streak:.1f} | "
            f"PnL={pnl_pct:+.2f}% | MT={mt:+d} | ccy={ccy} fx={fx_to_lei:.3f}"
        )

        # --- Persist debug fields into tracker ---
        info_state["last_score"] = float(avg_score)
        info_state["last_sell_index"] = float(sell_index)
        info_state["last_mt"] = int(mt)
        info_state["last_mt_prob"] = None if mt_prob is None else float(mt_prob)
        info_state["last_mt_gate"] = float(mt_gate)
        info_state["last_mt_prob_thr"] = None if mt_prob_thr is None else float(mt_prob_thr)
        info_state["last_mt_weight"] = float(mt_weight or 0.0)
        info_state["last_mt_pred_sellscore"] = None if pred_sellscore is None else float(pred_sellscore)
        info_state["last_mt_sell_threshold"] = None if sell_threshold is None else float(sell_threshold)
        info_state["last_currency"] = ccy
        info_state["last_fx_to_lei"] = float(fx_to_lei)

        if decision:
            now_utc = datetime.utcnow()
            info_state["last_alert_time"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            tracker["had_alerts"] = True

            reasoning = (
                f"{color_tag} {signal_label}\n"
                f"{pnl_context}\n\n"
                f"PnL: {pnl_pct:+.2f}% | {pnl_lei:+.2f} RON\n"
                f"AvgScore: {avg_score:.2f} | Weak: {weak_streak:.1f}/3\n"
                f"Consensus Index: {sell_index:.2f}\n"
                f"{mt_line}\n"
                f"{soft_line}\n"
                f"{hard_line}\n"
                f"UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            sell_alerts.append(f"[{ticker}] {reasoning}")

        tracker["tickers"][ticker] = info_state

    save_tracker(tracker)

    if not test_mode:
        git_commit_tracker()


    # === Send Discord summaries ===
    now_utc = datetime.utcnow()
    if sell_alerts:
        msg = "🚨 **SELL SIGNALS TRIGGERED** 🚨\n\n" + "\n\n".join(sell_alerts)
        for chunk in [msg[i:i + 1900] for i in range(0, len(msg), 1900)]:
            try:
                send_discord_alert(chunk)
            except Exception as e:
                print(f"[WARN] Discord send failed: {e}")
    elif end_of_day and not test_mode:
        try:
            send_discord_alert(
                f"😎 All systems stable. No sell signals today.\n"
                f"🕐 Checked at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
        except Exception as e:
            print(f"[WARN] Discord send failed: {e}")

    print(f"[DONE] Decision Engine completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run using bot/test_data.csv instead of bot/data.json")
    parser.add_argument("--endofday", action="store_true", help="Send end-of-day no-signal message (non-test only)")
    parser.add_argument("--ml_debug", action="store_true", help="Print MLTRACE JSON lines for each ticker")
    parser.add_argument("--ml_dump_dir", type=str, default=None, help="Directory for ML trace CSV (default: env ML_DUMP_DIR or bot/ml_debug)")
    args = parser.parse_args()

    if args.ml_dump_dir:
        ML_DUMP_DIR = args.ml_dump_dir  # noqa: PLW0603 (module-level override)
    if args.ml_debug:
        ML_DEBUG = True  # noqa: PLW0603

    run_decision_engine(test_mode=args.test, end_of_day=args.endofday)