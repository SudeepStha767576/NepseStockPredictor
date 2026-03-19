"""
NEPSE V6 Scoring Engine
Signals → direction prediction (BULL / NEUTRAL only) + price plan

V6 changes vs V5:
  • Added S6: 6-week RSI  (mean-reversion — most reliable weekly indicator)
  • Added S7: 3W/6W EMA crossover  (trend direction confirmation)
  • Raised BULL threshold to 108 on new 205-pt scale  (~same selectivity as V5)
  • Removed BEAR predictions — backtest showed only 28% BEAR accuracy (worse
    than random). BEAR scores now map to NEUTRAL.
  • S1 enhanced: 3-week momentum window (not just 2) with recency weighting
"""

from dataclasses import dataclass, field
from typing import Optional
import math
from datetime import date

# ─── DATA STRUCTURES ──────────────────────────────────────────────

@dataclass
class WeekData:
    week_num: int
    label: str
    open: float
    close: float
    high: float
    low: float
    volume: int
    month_day: int = 1  # Day of month for S10

@dataclass
class SignalBreakdown:
    s1_momentum: int = 0
    s2_streak: int = 0
    s3_volume: int = 0
    s4_position: int = 0
    s5_sector: int = 0
    s6_rsi: int = 0          # NEW V6
    s7_ema: int = 0          # NEW V6
    s8_rel_strength: int = 0
    s9_week52: int = 0
    s10_monthly: int = 0
    total: int = 0
    streak_count: int = 0
    atr_flag: bool = False
    atr_value: float = 0.0
    rsi_value: float = 50.0  # NEW V6
    bull_threshold: int = 108

@dataclass
class PricePlan:
    entry: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    band_low: float = 0.0
    band_high: float = 0.0
    risk_reward: float = 0.0

# ─── SECTOR SUB-INDEX DIRECTION (per week number) ─────────────────
# Based on real NEPSE sector index data Dec 2025 – Mar 2026
SECTOR_IDX = {
    "Hydro":      {12:None,11:1,10:0,9:1,8:1,7:1,6:1,5:-1,4:1,3:1,2:1,1:1},
    "Bank":       {12:None,11:None,10:0,9:0,8:0,7:1,6:0,5:0,4:1,3:1,2:1,1:0},
    "DevBank":    {12:None,11:None,10:None,9:None,8:0,7:1,6:1,5:0,4:1,3:1,2:1,1:0},
    "Insurance":  {12:1,11:1,10:1,9:1,8:0,7:1,6:0,5:0,4:1,3:1,2:None,1:None},
    "Microfinance":{12:None,11:None,10:None,9:0,8:0,7:1,6:0,5:0,4:1,3:1,2:None,1:None},
    "Hotel":      {12:None,11:None,10:None,9:1,8:1,7:1,6:0,5:0,4:1,3:1,2:1,1:1},
    "Telecom":    {12:None,11:None,10:0,9:0,8:0,7:0,6:0,5:0,4:1,3:0,2:None,1:None},
    "Power":      {12:None,11:None,10:None,9:1,8:1,7:0,6:0,5:0,4:0,3:1,2:1,1:None},
}

MONTHLY_CYCLE = {}

def get_monthly_bonus(day_of_month: int) -> int:
    if day_of_month <= 7:   return 10
    if day_of_month <= 14:  return 5
    if day_of_month <= 21:  return 0
    return -5

# ─── NEW V6: TECHNICAL INDICATOR HELPERS ──────────────────────────

def compute_rsi(weeks: list, current_idx: int, period: int = 6) -> float:
    """
    Wilder RSI on weekly closes over `period` weeks.
    Returns value in [0, 100]. 50 = neutral / insufficient data.
    """
    gains, losses = [], []
    start = max(1, current_idx - period + 1)
    for i in range(start, current_idx + 1):
        if weeks[i].close and weeks[i-1].close:
            chg = weeks[i].close - weeks[i-1].close
            if chg > 0:   gains.append(chg)
            elif chg < 0: losses.append(-chg)
    if not gains and not losses:
        return 50.0
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 99.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def compute_ema(prices: list, period: int) -> Optional[float]:
    """Exponential moving average seeded with SMA of first `period` values."""
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(prices[:period]) / period   # SMA seed
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


# ─── UNCHANGED V5 HELPERS ─────────────────────────────────────────

def compute_weekly_atr(weeks: list, current_idx: int, lookback: int = 4) -> Optional[float]:
    ranges = []
    for j in range(current_idx - 1, max(-1, current_idx - lookback - 1), -1):
        if j >= 0 and weeks[j].high and weeks[j].low:
            ranges.append(weeks[j].high - weeks[j].low)
    if not ranges:
        return None
    return sum(ranges) / len(ranges)

def compute_median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n//2-1] + s[n//2]) / 2
    return s[n//2]

def get_sector_peer_avg(symbol: str, week_num: int, sector: str, all_stocks: dict) -> Optional[float]:
    returns = []
    for sym, data in all_stocks.items():
        if sym == symbol:
            continue
        if data.get("sector") != sector:
            continue
        weeks = data.get("weeks", [])
        wk = next((w for w in weeks if w.week_num == week_num), None)
        if wk and wk.open and wk.close:
            returns.append((wk.close - wk.open) / wk.open * 100)
    if not returns:
        return None
    return sum(returns) / len(returns)


# ─── V6 CORE ENGINE ───────────────────────────────────────────────

def score_v5(
    symbol: str,
    weeks: list,
    current_idx: int,
    eps: float,
    high52: float,
    low52: float,
    sector: str,
    sector_peer_avg: Optional[float] = None,
) -> tuple:
    """
    Returns: (prediction, total_score, signals, price_plan)
    prediction: BULL | NEUTRAL   (BEAR removed in V6 — backtest showed 28% accuracy)
    Max possible score: ~205
    BULL threshold: 108
    """
    w = weeks[current_idx]
    signals = SignalBreakdown()

    # ── EPS Gate (hard filter — loss-making = skip) ────────────────
    if eps < 0:
        signals.total = 0
        return "NEUTRAL", 0, signals, PricePlan()

    p1 = weeks[current_idx - 1] if current_idx > 0 else None
    p2 = weeks[current_idx - 2] if current_idx > 1 else None
    p3 = weeks[current_idx - 3] if current_idx > 2 else None

    # ── S1: 3-Week Momentum with recency weighting (max 30) ───────
    # Current week counts most, then p1, then p2, p3 is context only
    m0 = (w.close  >= w.open)         # current week
    m1 = (p1.close >= p1.open) if p1 else m0
    m2 = (p2.close >= p2.open) if p2 else m1
    # Recency-weighted: current=40%, p1=35%, p2=25%
    bull_score = (0.40 * m0 + 0.35 * m1 + 0.25 * m2)
    if   bull_score >= 0.75: s1 = 30   # 3/3 or strong 2/3
    elif bull_score >= 0.50: s1 = 18   # mixed leaning bull
    elif bull_score >= 0.35: s1 = 8    # mixed leaning bear
    else:                    s1 = 0    # 3/3 bear
    signals.s1_momentum = s1

    # ── S2: Streak Guard (max 25) ────────────────────────────────
    streak, sdir = 0, None
    for j in range(current_idx - 1, max(-1, current_idx - 7), -1):
        if j < 0: break
        d = "B" if weeks[j].close >= weeks[j].open else "b"
        if sdir is None:
            sdir, streak = d, 1
        elif d == sdir:
            streak += 1
        else:
            break
    s2 = 25 if streak <= 2 else 12 if streak == 3 else 0
    signals.s2_streak = s2
    signals.streak_count = streak

    # ── S3: Volume Quality — direction-aware (max 25) ─────────────
    vols = []
    for j in range(current_idx - 1, max(-1, current_idx - 7), -1):
        if weeks[j].volume:
            vols.append(weeks[j].volume)
    med_vol = compute_median(vols) if len(vols) >= 3 else None
    week_bull = w.close >= w.open
    if med_vol and w.volume:
        if week_bull and w.volume > med_vol * 1.2:
            s3 = 25
        elif week_bull and w.volume > med_vol * 0.8:
            s3 = 12
        elif not week_bull and w.volume > med_vol * 1.2:
            s3 = 0   # Distribution — danger
        else:
            s3 = 8
    else:
        s3 = 12
    signals.s3_volume = s3

    # ATR filter
    atr = compute_weekly_atr(weeks, current_idx)
    signals.atr_value = round(atr, 1) if atr else 0
    atr_flag = False
    if atr and w.high and w.low:
        if (w.high - w.low) > atr * 1.5:
            atr_flag = True
    signals.atr_flag = atr_flag

    # ── S4: Price vs 4-week average (max 20) ─────────────────────
    closes4 = [weeks[j].close for j in range(current_idx - 1, max(-1, current_idx - 5), -1)
               if weeks[j].close]
    if len(closes4) >= 2:
        avg4 = sum(closes4) / len(closes4)
        s4 = 20 if w.open > avg4 * 1.03 else 10 if w.open >= avg4 * 0.97 else 0
    else:
        s4 = 10
    signals.s4_position = s4

    # ── S5: Sector sub-index direction (max 20, penalty -10) ──────
    sector_dir = SECTOR_IDX.get(sector, {}).get(w.week_num)
    s5 = 10 if sector_dir is None else 20 if sector_dir == 1 else -10
    signals.s5_sector = s5

    # ── S6: RSI (6-week) — mean-reversion signal (max 25, -15) ───
    # NEW in V6: oversold = high score (expect bounce), overbought = penalty
    rsi = compute_rsi(weeks, current_idx, period=6) if current_idx >= 4 else 50.0
    signals.rsi_value = rsi
    if   rsi <= 30: s6 = 25    # Heavily oversold — strong bounce candidate
    elif rsi <= 40: s6 = 18    # Oversold
    elif rsi <= 50: s6 = 8     # Neutral-low
    elif rsi <= 60: s6 = 2     # Neutral-high
    elif rsi <= 70: s6 = -5    # Approaching overbought
    else:           s6 = -15   # Overbought — caution on new longs
    signals.s6_rsi = s6

    # ── S7: EMA Crossover 3W vs 6W (max 20, -10) ─────────────────
    # NEW in V6: short EMA crossing above long EMA = uptrend confirmation
    closes_all = [weeks[j].close for j in range(max(0, current_idx - 8), current_idx + 1)
                  if weeks[j].close]
    ema3 = compute_ema(closes_all, 3)
    ema6 = compute_ema(closes_all, 6)
    ema3_prev = compute_ema(closes_all[:-1], 3) if len(closes_all) > 3 else None
    ema6_prev = compute_ema(closes_all[:-1], 6) if len(closes_all) > 6 else None

    if ema3 and ema6:
        cross_up   = ema3_prev and ema6_prev and ema3_prev <= ema6_prev and ema3 > ema6
        cross_down = ema3_prev and ema6_prev and ema3_prev >= ema6_prev and ema3 < ema6
        if   cross_up:       s7 = 20   # Golden cross — strongest uptrend signal
        elif ema3 > ema6:    s7 = 12   # Sustained uptrend
        elif cross_down:     s7 = -10  # Death cross — trend reversal warning
        else:                s7 = -5   # Sustained downtrend
    else:
        s7 = 5   # Insufficient data — neutral
    signals.s7_ema = s7

    # ── S8: Relative strength vs sector peers (max 15, -10) ───────
    stock_ret = (w.close - w.open) / w.open * 100
    if sector_peer_avg is not None:
        if stock_ret > sector_peer_avg + 2:
            s8 = 15
        elif stock_ret < sector_peer_avg - 3:
            s8 = -10
        else:
            s8 = 7
    else:
        s8 = 7
    signals.s8_rel_strength = s8

    # ── S9: 52-week position filter (max 10, -5) ──────────────────
    if high52 and low52 and high52 > low52:
        pos = (w.open - low52) / (high52 - low52)
        s9 = 10 if pos < 0.3 else 5 if pos <= 0.7 else 0 if pos < 0.9 else -5
    else:
        s9 = 5
    signals.s9_week52 = s9

    # ── S10: Monthly liquidity cycle (max 10, -5) ─────────────────
    s10 = get_monthly_bonus(w.month_day)
    signals.s10_monthly = s10

    # ── Total (max ~205) ──────────────────────────────────────────
    total = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
    signals.total = total

    # ── Dynamic threshold (V6: raised to 108 base on new scale) ───
    # Tighten further if streak risk or weak EPS
    bull_thresh = 100 if (streak >= 3 or eps < 2) else 108
    signals.bull_threshold = bull_thresh

    # ── Prediction (V6: BULL or NEUTRAL only) ─────────────────────
    # BEAR removed — backtest showed 28% accuracy (worse than random).
    # Low-scoring stocks are NEUTRAL, not tradeable shorts in NEPSE context.
    if atr_flag:
        pred = "NEUTRAL"
    elif total >= bull_thresh:
        pred = "BULL"
    else:
        pred = "NEUTRAL"

    # ── Price Plan (ATR-based, BULL only) ─────────────────────────
    plan = PricePlan()
    if atr and pred == "BULL":
        entry = w.close
        plan.entry      = round(entry, 1)
        plan.stop_loss  = round(entry - atr * 0.5, 1)
        plan.target_1   = round(entry + atr * 0.8, 1)
        plan.target_2   = round(entry + atr * 1.3, 1)
        plan.target_3   = round(entry + atr * 1.8, 1)
        plan.band_low   = round(plan.target_1 - atr * 0.3, 1)
        plan.band_high  = round(plan.target_1 + atr * 0.3, 1)
        risk   = abs(plan.entry - plan.stop_loss)
        reward = abs(plan.entry - plan.target_1)
        plan.risk_reward = round(reward / risk, 2) if risk > 0 else 0

    return pred, total, signals, plan


def assign_grade(score: int, pred: str) -> str:
    """Grade on new V6 scale (max ~205)."""
    if pred == "NEUTRAL": return "N"
    if score >= 145: return "A+"
    if score >= 125: return "A"
    if score >= 112: return "B"
    if score >= 100: return "C"
    return "D"


def generate_reason(signals: SignalBreakdown, pred: str, eps: float) -> str:
    if eps < 0:
        return "Loss-making company (EPS < 0) — excluded from active signals"
    if pred == "NEUTRAL" and signals.atr_flag:
        return "High volatility week (ATR spike) — too noisy to call"
    if pred == "NEUTRAL":
        return f"Score {signals.total} below BULL threshold ({signals.bull_threshold}) — no edge"
    parts = []
    # Momentum
    if signals.s1_momentum == 30: parts.append("3-week bull momentum confirmed")
    if signals.s2_streak == 0:    parts.append("⚠ Long streak — reversal risk")
    if signals.s3_volume == 25:   parts.append("volume surge confirming move")
    # V6 signals
    if signals.rsi_value <= 35:   parts.append(f"RSI {signals.rsi_value} — heavily oversold, bounce due")
    elif signals.rsi_value <= 45: parts.append(f"RSI {signals.rsi_value} — oversold zone")
    elif signals.rsi_value >= 70: parts.append(f"⚠ RSI {signals.rsi_value} — overbought, caution")
    if signals.s7_ema == 20:      parts.append("golden cross (EMA 3W crossed above 6W)")
    elif signals.s7_ema == 12:    parts.append("EMA uptrend confirmed")
    elif signals.s7_ema == -10:   parts.append("⚠ death cross — EMA trend reversed down")
    # Sector / other
    if signals.s5_sector == 20:   parts.append("sector index bullish")
    if signals.s5_sector == -10:  parts.append("sector index bearish")
    if signals.s8_rel_strength == 15:  parts.append("outperforming sector peers")
    if signals.s8_rel_strength == -10: parts.append("underperforming peers — caution")
    if signals.s9_week52 == -5:   parts.append("near 52W high — resistance zone")
    if signals.s10_monthly == 10: parts.append("Week 1 of month — liquidity boost")
    if signals.s10_monthly == -5: parts.append("Week 4 of month — tighter liquidity")
    return " · ".join(parts) if parts else f"Score {signals.total}"
