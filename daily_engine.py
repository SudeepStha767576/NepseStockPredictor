"""
NEPSE Per-Stock Daily Prediction Engine
----------------------------------------
Predicts the next 5 NEPSE trading days for a single stock using:
  1. V8 weekly engine score as directional bias (read-only, no changes)
  2. Per-stock historical base rate (actual avg return on BULL weeks in history)
  3. Daily ATR + RSI + 5-day momentum for magnitude and range estimates

Usage:
    from daily_engine import predict_daily
    result = predict_daily("NABIL", stock_meta={"eps": 30, "sector": "Bank", ...})
"""

import json
import math
import statistics
from datetime import datetime, timedelta
from dataclasses import replace
from pathlib import Path
from typing import Optional

from engine import score_v5, WeekData, assign_grade, generate_reason
from scraper import HIST_CACHE_DIR, _aggregate_weekly

# ─── CONFIG ────────────────────────────────────────────────────────
MAX_WEEKLY   = 52   # weekly candles to pass to V8 engine
MIN_WEEKLY   = 20   # minimum weeks before we run predictions
DAILY_WINDOW = 60   # daily candles for indicator computation


# ─── DAILY RSI (Wilder's smoothing) ────────────────────────────────

def _compute_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


# ─── EMA ───────────────────────────────────────────────────────────

def _compute_ema(values: list, period: int) -> float:
    if len(values) < period:
        return values[-1] if values else 0.0
    k   = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return round(ema, 2)


# ─── DAILY ATR ─────────────────────────────────────────────────────

def _compute_daily_atr(rows: list, period: int = 14) -> float:
    """Average True Range over the last `period` daily candles."""
    recent = rows[-(period + 1):]
    trs = []
    for i in range(1, len(recent)):
        h  = recent[i]["high"]
        l  = recent[i]["low"]
        pc = recent[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    subset = trs[-period:] if len(trs) >= period else trs
    return round(statistics.mean(subset), 2) if subset else 5.0


# ─── RENUMBER (same logic as backtest) ─────────────────────────────

def _renumber(weeks: list) -> list:
    n = len(weeks)
    return [replace(w, week_num=n - i) for i, w in enumerate(weeks)]


# ─── NEXT NEPSE TRADING DAYS ───────────────────────────────────────

def _next_trading_days(n: int = 5) -> list:
    """
    Return the next N NEPSE trading days starting from tomorrow.
    NEPSE trades Sunday–Thursday (Python weekday: Sun=6, Mon=0…Thu=3).
    """
    trading = {0, 1, 2, 3, 6}   # Mon, Tue, Wed, Thu, Sun
    days, candidate = [], datetime.now() + timedelta(days=1)
    while len(days) < n:
        if candidate.weekday() in trading:
            days.append(candidate)
        candidate += timedelta(days=1)
    return days


# ─── HISTORICAL BASE RATE ──────────────────────────────────────────

def _compute_base_rate(symbol: str, all_weekly: list,
                        eps: float, hi52: float, lo52: float,
                        sector: str) -> dict:
    """
    Walk through the stock's own weekly history.
    For every week where V8 scored BULL, record the ACTUAL next-week return.
    Returns base_rate_pct (median BULL return), bull_hit_rate, n_bull_weeks.
    """
    bull_returns = []

    # Need at least min_idx weeks before we can score
    min_idx = 21   # EMA-21 needs 21 candles to warm up

    for i in range(min_idx, len(all_weekly) - 1):
        train = _renumber(all_weekly[: i + 1])
        try:
            pred, score, _sigs, _plan = score_v5(
                symbol              = symbol,
                weeks               = train,
                current_idx         = len(train) - 1,
                eps                 = eps,
                high52              = hi52,
                low52               = lo52,
                sector              = sector,
                sector_peer_avg     = None,
                nepse_market_return = None,
            )
        except Exception:
            continue

        if pred != "BULL":
            continue

        nw = all_weekly[i + 1]
        if nw.open and nw.close and nw.open != 0:
            ret = (nw.close - nw.open) / nw.open * 100
            bull_returns.append(ret)

    if bull_returns:
        srt  = sorted(bull_returns)
        n    = len(srt)
        med  = (srt[n // 2 - 1] + srt[n // 2]) / 2 if n % 2 == 0 else srt[n // 2]
        hits = sum(1 for r in bull_returns if r > 0)
        return {
            "base_rate_pct":         round(med, 3),
            "bull_hit_rate":         round(hits / n, 3),
            "bull_weeks_in_history": n,
        }

    # Fallback when not enough BULL history
    return {"base_rate_pct": 0.8, "bull_hit_rate": 0.50, "bull_weeks_in_history": 0}


# ─── MAIN FUNCTION ─────────────────────────────────────────────────

def predict_daily(symbol: str, stock_meta: Optional[dict] = None) -> dict:
    """
    Predict next 5 NEPSE trading days for `symbol`.

    stock_meta keys: eps, hi52, lo52, name, sector, current_price
    All are optional — will be estimated from cached data if absent.
    """
    sym = symbol.upper()

    # ── 1. Load & sort daily rows ───────────────────────────────────
    hist_path = HIST_CACHE_DIR / f"{sym}.json"
    if not hist_path.exists():
        raise ValueError(f"No history cache for {sym}. "
                         f"Run a scrape first or check the symbol spelling.")

    hist     = json.loads(hist_path.read_text())
    all_rows = sorted(hist.get("rows", []), key=lambda r: r["date"])

    if len(all_rows) < 25:
        raise ValueError(f"Only {len(all_rows)} daily rows for {sym} — need ≥ 25")

    # ── 2. Metadata ─────────────────────────────────────────────────
    sm = stock_meta or {}
    eps           = float(sm.get("eps") or 1.0)
    sector        = sm.get("sector") or "Others"
    name          = sm.get("name")   or sym
    current_price = sm.get("current_price") or float(all_rows[-1]["close"])

    # ── 3. Weekly candles → V8 weekly engine ───────────────────────
    all_weekly = _aggregate_weekly(all_rows, MAX_WEEKLY)

    if len(all_weekly) < MIN_WEEKLY:
        raise ValueError(
            f"Only {len(all_weekly)} weekly candles for {sym} — "
            f"need ≥ {MIN_WEEKLY} for a reliable prediction."
        )

    # Derive 52W H/L from actual weekly data if not supplied
    hi52 = float(sm.get("hi52") or 0) or max(
        (w.high for w in all_weekly[-52:] if w.high), default=current_price
    )
    lo52 = float(sm.get("lo52") or 0) or min(
        (w.low  for w in all_weekly[-52:] if w.low),  default=current_price
    )

    weekly_rn = _renumber(all_weekly)
    pred, score, signals, weekly_plan = score_v5(
        symbol              = sym,
        weeks               = weekly_rn,
        current_idx         = len(weekly_rn) - 1,
        eps                 = eps,
        high52              = hi52,
        low52               = lo52,
        sector              = sector,
        sector_peer_avg     = None,
        nepse_market_return = None,
    )
    grade  = assign_grade(score, pred)
    reason = generate_reason(signals, pred, eps)

    # ── 4. Daily indicators ─────────────────────────────────────────
    recent = all_rows[-DAILY_WINDOW:]
    closes  = [float(r["close"])  for r in recent]
    volumes = [int(r.get("volume", 0)) for r in recent]

    last_close  = current_price
    daily_rsi   = _compute_rsi(closes, 14)
    daily_ema5  = _compute_ema(closes, 5)
    daily_ema20 = _compute_ema(closes, 20)
    daily_atr   = _compute_daily_atr(recent, 14)

    # 5-day momentum (avg daily % change)
    if len(closes) >= 6:
        mom_rets    = [(closes[i] - closes[i - 1]) / closes[i - 1] * 100
                       for i in range(-5, 0)]
        momentum_5d = round(statistics.mean(mom_rets), 3)
    else:
        momentum_5d = 0.0

    vol_med   = statistics.median(volumes[-21:]) if len(volumes) >= 5 else 1
    vol_ratio = round(volumes[-1] / vol_med, 2) if vol_med else 1.0

    # ── 5. Per-stock historical base rate ──────────────────────────
    base_info  = _compute_base_rate(sym, all_weekly, eps, hi52, lo52, sector)
    base_rate  = base_info["base_rate_pct"]
    hit_rate   = base_info["bull_hit_rate"]
    n_bull     = base_info["bull_weeks_in_history"]

    # ── 6. Expected weekly return ──────────────────────────────────
    if pred == "BULL":
        confidence = min((score - 115) / max(185 - 115, 1), 1.0)
        weekly_exp = base_rate * (1 + confidence * 0.5)   # e.g. 0.8% → 1.2%
        direction  = 1
    else:
        # NEUTRAL: small RSI-driven mean-reversion bias
        rsi_bias   = (50 - daily_rsi) / 50 * 0.25
        weekly_exp = rsi_bias
        direction  = 1 if rsi_bias > 0 else (-1 if rsi_bias < 0 else 0)

    # Blend in 5-day momentum (20 % weight)
    weekly_exp = weekly_exp * 0.80 + momentum_5d * 5 * 0.20

    # ── 7. Per-day projection ──────────────────────────────────────
    # Intraday distribution: Mon/Fri smaller moves, Tue–Thu larger
    day_weights  = [0.16, 0.22, 0.24, 0.22, 0.16]
    trading_days = _next_trading_days(5)

    days_out   = []
    prev_close = last_close

    for dt, wt in zip(trading_days, day_weights):
        day_exp_ret = weekly_exp * wt        # % for this specific day

        exp_close = round(prev_close * (1 + day_exp_ret / 100), 2)
        exp_open  = round(prev_close * (1 + day_exp_ret * 0.3 / 100), 2)
        exp_high  = round(exp_close + 0.65 * daily_atr, 2)
        exp_low   = round(exp_close - 0.65 * daily_atr, 2)

        if direction >= 0:
            # BULL / neutral-bull: enter on dip
            entry   = round(exp_low + 0.20 * daily_atr, 2)
            stop    = round(entry   - 0.50 * daily_atr, 2)
            target1 = round(entry   + 1.00 * daily_atr, 2)
            target2 = round(entry   + 1.80 * daily_atr, 2)
        else:
            # Bearish neutral: enter on bounce-short (informational)
            entry   = round(exp_high - 0.20 * daily_atr, 2)
            stop    = round(entry    + 0.50 * daily_atr, 2)
            target1 = round(entry    - 1.00 * daily_atr, 2)
            target2 = round(entry    - 1.80 * daily_atr, 2)

        days_out.append({
            "day":            dt.strftime("%A"),
            "date":           dt.strftime("%Y-%m-%d"),
            "expected_open":  exp_open,
            "expected_high":  exp_high,
            "expected_low":   exp_low,
            "expected_close": exp_close,
            "entry":          entry,
            "stop_loss":      stop,
            "target_1":       target1,
            "target_2":       target2,
            "daily_exp_ret":  round(day_exp_ret, 3),
        })

        prev_close = exp_close

    # ── 8. Chart history (last 30 daily candles) ───────────────────
    history = []
    for r in all_rows[-30:]:
        d = r["date"]
        history.append({
            "date":   d if isinstance(d, str) else d.strftime("%Y-%m-%d"),
            "open":   round(float(r["open"]),  2),
            "high":   round(float(r["high"]),  2),
            "low":    round(float(r["low"]),   2),
            "close":  round(float(r["close"]), 2),
            "volume": int(r.get("volume", 0)),
        })

    # ── 9. Assemble response ────────────────────────────────────────
    return {
        "symbol":        sym,
        "name":          name,
        "current_price": round(float(last_close), 2),
        "weekly": {
            "prediction": pred,
            "score":      score,
            "grade":      grade,
            "reason":     reason,
            "signals": {
                "s1_momentum":     signals.s1_momentum,
                "s2_streak":       signals.s2_streak,
                "s3_volume":       signals.s3_volume,
                "s4_position":     signals.s4_position,
                "s5_sector":       signals.s5_sector,
                "s6_rsi":          signals.s6_rsi,
                "s7_ema":          signals.s7_ema,
                "s8_rel_strength": signals.s8_rel_strength,
                "s9_week52":       signals.s9_week52,
                "s10_monthly":     signals.s10_monthly,
                "s11_market":      signals.s11_market,
                "total":           signals.total,
                "rsi_value":       round(signals.rsi_value, 1),
                "atr_flag":        signals.atr_flag,
                "atr_value":       round(signals.atr_value, 2),
            },
            "plan": {
                "entry":       weekly_plan.entry,
                "stop_loss":   weekly_plan.stop_loss,
                "target_1":    weekly_plan.target_1,
                "target_2":    weekly_plan.target_2,
                "target_3":    weekly_plan.target_3,
                "risk_reward": weekly_plan.risk_reward,
            },
        },
        "model": {
            "base_rate_pct":          round(base_rate, 3),
            "bull_hit_rate":          round(hit_rate, 3),
            "bull_weeks_in_history":  n_bull,
            "weekly_expected_return": round(weekly_exp, 3),
            "daily_atr":              daily_atr,
            "daily_rsi":              daily_rsi,
            "daily_ema5":             daily_ema5,
            "daily_ema20":            daily_ema20,
            "momentum_5d":            momentum_5d,
            "vol_ratio":              vol_ratio,
            "total_weekly_candles":   len(all_weekly),
            "total_daily_rows":       len(all_rows),
        },
        "days":    days_out,
        "history": history,
    }


# ─── STOCK LIST HELPER ─────────────────────────────────────────────

def list_available_symbols() -> list:
    """Return sorted list of symbols that have a history cache file."""
    return sorted(p.stem for p in HIST_CACHE_DIR.glob("*.json"))
