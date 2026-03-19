"""
NEPSE V5 Scoring Engine
10 signals → direction prediction + price range + entry/exit plan
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
    s8_rel_strength: int = 0
    s9_week52: int = 0
    s10_monthly: int = 0
    total: int = 0
    streak_count: int = 0
    atr_flag: bool = False
    atr_value: float = 0.0
    bull_threshold: int = 90

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

@dataclass
class StockResult:
    symbol: str
    name: str
    sector: str
    eps: float
    current_price: float
    week52_high: float
    week52_low: float
    prediction: str           # BULL / BEAR / NEUTRAL
    score: int
    signals: SignalBreakdown
    plan: PricePlan
    reason: str = ""
    grade: str = "C"
    percent_change_last_week: float = 0.0

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

MONTHLY_CYCLE = {
    # Week of month → score adjustment
    # Week1 (days 1-7): +10, Week2 (8-14): +5, Week3 (15-21): 0, Week4+ (22+): -5
}

def get_monthly_bonus(day_of_month: int) -> int:
    if day_of_month <= 7:   return 10
    if day_of_month <= 14:  return 5
    if day_of_month <= 21:  return 0
    return -5

def compute_weekly_atr(weeks: list[WeekData], current_idx: int, lookback: int = 4) -> Optional[float]:
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

# ─── V5 CORE ENGINE ──────────────────────────────────────────────

def score_v5(
    symbol: str,
    weeks: list[WeekData],
    current_idx: int,
    eps: float,
    high52: float,
    low52: float,
    sector: str,
    sector_peer_avg: Optional[float] = None,
) -> tuple[str, int, SignalBreakdown, PricePlan]:
    """
    Returns: (prediction, total_score, signals, price_plan)
    prediction: BULL | BEAR | NEUTRAL
    """
    w = weeks[current_idx]
    signals = SignalBreakdown()

    # ── S6: EPS Gate (hard filter) ────────────────────────────────
    if eps < 0:
        signals.total = 0
        return "NEUTRAL", 0, signals, PricePlan()

    p1 = weeks[current_idx - 1] if current_idx > 0 else None
    p2 = weeks[current_idx - 2] if current_idx > 1 else None

    # ── S1: 2-Week Momentum (max 30) ─────────────────────────────
    m1 = (p1.close >= p1.open) if p1 else None
    m2 = (p2.close >= p2.open) if p2 else None
    if m1 is not None and m2 is not None:
        s1 = 30 if (m1 and m2) else 15 if (m1 or m2) else 0
    elif m1 is not None:
        s1 = 15 if m1 else 0
    else:
        s1 = 15
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

    # ── Total ─────────────────────────────────────────────────────
    total = s1 + s2 + s3 + s4 + s5 + s8 + s9 + s10
    signals.total = total

    # ── Dynamic threshold ────────────────────────────────────────
    bull_thresh = 85 if (streak >= 3 or eps < 2) else 90
    bear_thresh = 54
    signals.bull_threshold = bull_thresh

    # ── Prediction ───────────────────────────────────────────────
    if atr_flag:
        pred = "NEUTRAL"
    elif total >= bull_thresh:
        pred = "BULL"
    elif total <= bear_thresh:
        pred = "BEAR"
    else:
        pred = "NEUTRAL"

    # ── Price Plan (ATR-based) ────────────────────────────────────
    plan = PricePlan()
    if atr and pred != "NEUTRAL":
        entry = w.close  # Entry = current week's close (Sunday open proxy)
        if pred == "BULL":
            plan.entry      = round(entry, 1)
            plan.stop_loss  = round(entry - atr * 0.5, 1)
            plan.target_1   = round(entry + atr * 0.8, 1)
            plan.target_2   = round(entry + atr * 1.3, 1)
            plan.target_3   = round(entry + atr * 1.8, 1)
            plan.band_low   = round(plan.target_1 - atr * 0.3, 1)
            plan.band_high  = round(plan.target_1 + atr * 0.3, 1)
        else:  # BEAR
            plan.entry      = round(entry, 1)
            plan.stop_loss  = round(entry + atr * 0.5, 1)
            plan.target_1   = round(entry - atr * 0.8, 1)
            plan.target_2   = round(entry - atr * 1.3, 1)
            plan.target_3   = round(entry - atr * 1.8, 1)
            plan.band_low   = round(plan.target_1 - atr * 0.3, 1)
            plan.band_high  = round(plan.target_1 + atr * 0.3, 1)

        risk = abs(plan.entry - plan.stop_loss)
        reward = abs(plan.entry - plan.target_1)
        plan.risk_reward = round(reward / risk, 2) if risk > 0 else 0

    return pred, total, signals, plan

def assign_grade(score: int, pred: str) -> str:
    if pred == "NEUTRAL": return "N"
    if score >= 95: return "A+"
    if score >= 85: return "A"
    if score >= 75: return "B"
    if score >= 65: return "C"
    return "D"

def generate_reason(signals: SignalBreakdown, pred: str, eps: float) -> str:
    if eps < 0:
        return "Loss-making company (EPS < 0) — excluded from active signals"
    if pred == "NEUTRAL" and signals.atr_flag:
        return "High volatility week (ATR spike) — too noisy to call"
    if pred == "NEUTRAL":
        return f"Mixed signals — score {signals.total} in neutral zone ({signals.bull_threshold-1}–55)"
    parts = []
    if signals.s1_momentum == 30: parts.append("2-week bull momentum confirmed")
    if signals.s2_streak == 0:    parts.append("⚠ Long streak — reversal risk")
    if signals.s3_volume == 25:   parts.append("volume surge confirming move")
    if signals.s5_sector == 20:   parts.append("sector index bullish")
    if signals.s5_sector == -10:  parts.append("sector index bearish")
    if signals.s8_rel_strength == 15: parts.append("outperforming sector peers")
    if signals.s8_rel_strength == -10: parts.append("underperforming peers — caution")
    if signals.s9_week52 == -5:   parts.append("near 52W high — resistance zone")
    if signals.s10_monthly == 10: parts.append("Week 1 of month — liquidity boost")
    if signals.s10_monthly == -5: parts.append("Week 4 of month — tighter liquidity")
    return " · ".join(parts) if parts else f"Score {signals.total}"
