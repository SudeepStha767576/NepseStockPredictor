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
    s6_rsi: int = 0          # V6: RSI mean-reversion
    s7_ema: int = 0          # V6: EMA crossover
    s8_rel_strength: int = 0
    s9_week52: int = 0
    s10_monthly: int = 0
    s11_market: int = 0      # V7: NEPSE market breadth gate
    total: int = 0
    streak_count: int = 0
    streak_dir: str = ""     # V7: "bull" or "bear"
    atr_flag: bool = False
    atr_value: float = 0.0
    rsi_value: float = 50.0
    bull_threshold: int = 115

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

def compute_weekly_atr(weeks: list, current_idx: int, lookback: int = 8) -> Optional[float]:
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
    nepse_market_return: Optional[float] = None,   # V7: cross-sectional median
) -> tuple:
    """
    Returns: (prediction, total_score, signals, price_plan)
    prediction: BULL | NEUTRAL

    V8 signal recalibration (based on 306-prediction ablation study):
      S6 RSI  : +11.75 correlation with hits  → BOOSTED (max 35)
      S9 52W  : +1.0   → slightly boosted
      S1      : +1.0   → kept
      S4 price: -3.5   → INVERTED (below avg = bullish; was above avg = bullish)
      S3 vol  : -4.0   → REDUCED  (high vol on green = distribution risk)
      S7 EMA  : -4.8   → REDUCED  (trend maturity ≠ continuation)
    Max possible score ~185.  BULL threshold 95.
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
    # Count consecutive same-direction weeks immediately before current week.
    # Long streaks = exhaustion / reversal risk → lower score.
    streak, sdir = 0, None
    for j in range(current_idx - 1, max(-1, current_idx - 7), -1):
        if j < 0: break
        d = "bull" if weeks[j].close >= weeks[j].open else "bear"
        if sdir is None:
            sdir, streak = d, 1
        elif d == sdir:
            streak += 1
        else:
            break
    s2 = 25 if streak <= 2 else 12 if streak == 3 else 0
    signals.s2_streak   = s2
    signals.streak_count = streak
    signals.streak_dir  = sdir or "bull"

    # ── S3: Volume Quality (max 15) — 12-week baseline ────────────
    # 12-week volume baseline is far more stable than 4-week.
    # Reduces noise from single anomalous weeks distorting the average.
    vols = []
    for j in range(current_idx - 1, max(-1, current_idx - 13), -1):
        if weeks[j].volume:
            vols.append(weeks[j].volume)
    med_vol = compute_median(vols) if len(vols) >= 3 else None
    week_bull = w.close >= w.open
    if med_vol and w.volume:
        if not week_bull and w.volume > med_vol * 1.2:
            s3 = 0    # Distribution — red week on high volume = danger
        elif week_bull and w.volume > med_vol * 1.2:
            s3 = 10   # Reduced from 25: possible distribution, not confirmed
        elif week_bull and w.volume > med_vol * 0.8:
            s3 = 8
        else:
            s3 = 5
    else:
        s3 = 8
    signals.s3_volume = s3

    # ATR filter
    atr = compute_weekly_atr(weeks, current_idx)
    signals.atr_value = round(atr, 1) if atr else 0
    atr_flag = False
    if atr and w.high and w.low:
        if (w.high - w.low) > atr * 1.5:
            atr_flag = True
    signals.atr_flag = atr_flag

    # ── S4: Price vs 8-week average — INVERTED (max 15) ──────────
    # 8-week baseline more stable than 4-week with 52 weeks available.
    # Price ABOVE average correlated with MISSES (mean-reversion bias).
    closes4 = [weeks[j].close for j in range(current_idx - 1, max(-1, current_idx - 9), -1)
               if weeks[j].close]
    if len(closes4) >= 2:
        avg4 = sum(closes4) / len(closes4)
        if   w.open < avg4 * 0.95:  s4 = 15   # Depressed below avg → bounce candidate
        elif w.open < avg4 * 1.00:  s4 = 8    # Slightly below → mild bullish
        elif w.open < avg4 * 1.05:  s4 = 3    # Near avg → neutral
        else:                       s4 = -5   # Extended above avg → reversion risk
    else:
        s4 = 5
    signals.s4_position = s4

    # ── S5: Sector sub-index direction (max 20, penalty -10) ──────
    sector_dir = SECTOR_IDX.get(sector, {}).get(w.week_num)
    s5 = 10 if sector_dir is None else 20 if sector_dir == 1 else -10
    signals.s5_sector = s5

    # ── S6: RSI (14-week standard) — STRONGEST signal (max 35,-20) ─
    # Now using proper 14-period RSI (industry standard) enabled by 52W data.
    # Falls back to 6-period for stocks with limited history.
    # Ablation: +11.75 correlation — by far the most predictive signal.
    if   current_idx >= 14: rsi = compute_rsi(weeks, current_idx, period=14)
    elif current_idx >= 6:  rsi = compute_rsi(weeks, current_idx, period=6)
    else:                   rsi = 50.0
    signals.rsi_value = rsi
    if   rsi <= 25: s6 = 35    # Extremely oversold — very strong bounce signal
    elif rsi <= 35: s6 = 28    # Heavily oversold
    elif rsi <= 45: s6 = 18    # Oversold zone
    elif rsi <= 55: s6 = 8     # Neutral
    elif rsi <= 65: s6 = -2    # Approaching overbought
    elif rsi <= 72: s6 = -10   # Overbought
    else:           s6 = -20   # Very overbought — hard gate effectively
    signals.s6_rsi = s6

    # ── S7: EMA Crossover 8W vs 21W (max 10, -10) ────────────────
    # Now using 8/21 EMA (industry standard) enabled by 52W history.
    # Falls back to 3/6 EMA for stocks with limited history.
    # Fresh crossovers most valuable; mature trends minimal signal.
    closes_all = [weeks[j].close for j in range(max(0, current_idx - 24), current_idx + 1)
                  if weeks[j].close]
    if len(closes_all) >= 21:
        ema_fast = compute_ema(closes_all, 8)
        ema_slow = compute_ema(closes_all, 21)
        ema_fast_prev = compute_ema(closes_all[:-1], 8)  if len(closes_all) > 8  else None
        ema_slow_prev = compute_ema(closes_all[:-1], 21) if len(closes_all) > 21 else None
    else:
        ema_fast = compute_ema(closes_all, 3)
        ema_slow = compute_ema(closes_all, 6)
        ema_fast_prev = compute_ema(closes_all[:-1], 3) if len(closes_all) > 3 else None
        ema_slow_prev = compute_ema(closes_all[:-1], 6) if len(closes_all) > 6 else None

    if ema_fast and ema_slow:
        cross_up   = ema_fast_prev and ema_slow_prev and ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow
        cross_down = ema_fast_prev and ema_slow_prev and ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow
        if   cross_up:          s7 = 10   # Fresh golden cross
        elif ema_fast > ema_slow: s7 = 2   # Mature uptrend — minimal
        elif cross_down:         s7 = -10  # Fresh death cross
        else:                    s7 = -5   # Sustained downtrend
    else:
        s7 = 2
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

    # ── S9: 52-week position (max 15, -8) — slightly boosted ──────
    # Ablation: +1.0 correlation. Near 52W low = bounce candidate.
    # Strengthened to complement S6 RSI oversold signal.
    if high52 and low52 and high52 > low52:
        pos = (w.open - low52) / (high52 - low52)
        if   pos < 0.20: s9 = 15   # Deep in bottom 20% — strong mean reversion
        elif pos < 0.40: s9 = 10   # Lower half
        elif pos < 0.65: s9 = 5    # Mid range
        elif pos < 0.85: s9 = 0    # Upper range
        else:            s9 = -8   # Near 52W high — resistance
    else:
        s9 = 5
    signals.s9_week52 = s9

    # ── S10: Monthly liquidity cycle (max 10, -5) ─────────────────
    s10 = get_monthly_bonus(w.month_day)
    signals.s10_monthly = s10

    # ── S11: NEPSE market breadth gate (0 or negative only) ──────
    # Suppression-only signal: penalise BULL calls when whole market was down.
    # We do NOT give a positive bonus when market was up — that signal reverses
    # too often (W-16: prior week +3%, then market fell the next week).
    # Only suppress when market is clearly bearish.
    if nepse_market_return is None:
        s11 = 0    # No data — neutral
    elif nepse_market_return > -1.0:
        s11 = 0    # Market flat or up → no change to score
    elif nepse_market_return > -2.0:
        s11 = -12  # Market moderately down → reduce confidence
    else:
        s11 = -22  # Market clearly down (≤ -2%) → strong suppression
    signals.s11_market = s11

    # ── Total (max ~185) ──────────────────────────────────────────
    total = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11
    signals.total = total

    # ── Threshold ────────────────────────────────────────────────
    # V8.1: Raised from 95→115 based on 516-prediction ablation.
    # Score gap analysis: hit avg=119 vs miss avg=110.
    # At thresh=115: 76.6% accuracy (was 63.6% at thresh=95), 43% coverage.
    # Eliminates ~57% of false BULL signals while retaining ~70% of true hits.
    bull_thresh = 115
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
    """Grade on V8.1 scale (threshold=115, max ~185).
    All BULL calls already have score >= 115.
    A+ = top-tier conviction, A = strong, B = good, C = borderline.
    """
    if pred == "NEUTRAL": return "N"
    if score >= 150: return "A+"
    if score >= 135: return "A"
    if score >= 125: return "B"
    if score >= 115: return "C"
    return "D"


def generate_reason(signals: SignalBreakdown, pred: str, eps: float) -> str:
    if eps < 0:
        return "Loss-making company (EPS < 0) — excluded from active signals"
    if pred == "NEUTRAL" and signals.atr_flag:
        return "High volatility week (ATR spike) — too noisy to call"
    if pred == "NEUTRAL":
        mkt_note = f" · market breadth {signals.s11_market:+d}" if signals.s11_market < 0 else ""
        return f"Score {signals.total} below BULL threshold ({signals.bull_threshold}) — no edge{mkt_note}"
    parts = []
    # Momentum
    if signals.s1_momentum >= 25: parts.append("3-week bull momentum confirmed")
    if signals.s2_streak == 0:    parts.append("⚠ Long streak — reversal risk")
    if signals.s3_volume >= 10:   parts.append("volume confirming move")
    elif signals.s3_volume == 0 and signals.s1_momentum > 0: parts.append("⚠ distribution volume on green week")
    # Price position (V8 inverted — depressed price = bounce candidate)
    if signals.s4_position >= 15: parts.append("price depressed vs 4W avg — bounce setup")
    elif signals.s4_position <= -5: parts.append("⚠ price extended above 4W avg — reversion risk")
    # V8 RSI signal (primary driver)
    if signals.rsi_value <= 25:   parts.append(f"RSI {signals.rsi_value} — extremely oversold, strong bounce")
    elif signals.rsi_value <= 35: parts.append(f"RSI {signals.rsi_value} — heavily oversold, bounce due")
    elif signals.rsi_value <= 45: parts.append(f"RSI {signals.rsi_value} — oversold zone")
    elif signals.rsi_value >= 70: parts.append(f"⚠ RSI {signals.rsi_value} — overbought, caution")
    if signals.s7_ema >= 10:      parts.append("golden cross (EMA 8W crossed above 21W)")
    elif signals.s7_ema == 2:     parts.append("EMA 8W/21W uptrend active")
    elif signals.s7_ema <= -10:   parts.append("⚠ death cross — EMA 8W dropped below 21W")
    # 52W position (V8 — bottom favoured)
    if signals.s9_week52 >= 15:   parts.append("near 52W low — deep value zone")
    elif signals.s9_week52 >= 10: parts.append("lower 52W range — value area")
    elif signals.s9_week52 <= -8: parts.append("⚠ near 52W high — resistance zone")
    # Sector / other
    if signals.s5_sector == 20:   parts.append("sector index bullish")
    if signals.s5_sector == -10:  parts.append("sector index bearish")
    if signals.s8_rel_strength == 15:  parts.append("outperforming sector peers")
    if signals.s8_rel_strength == -10: parts.append("underperforming peers — caution")
    if signals.s10_monthly == 10: parts.append("Week 1 of month — liquidity boost")
    if signals.s10_monthly == -5: parts.append("Week 4 of month — tighter liquidity")
    if signals.s11_market <= -15: parts.append("⚠ NEPSE market down last week — caution")
    return " · ".join(parts) if parts else f"Score {signals.total}"
