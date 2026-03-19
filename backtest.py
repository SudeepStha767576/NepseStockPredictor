"""
NEPSE V5 — Backtesting Engine
Simulates weekly predictions for the past N weeks and measures direction accuracy.

How it works:
  For each week W (1 = last week, 12 = 12 weeks ago):
    - Slice history to data available BEFORE week W (training data)
    - Run score_v5 on the slice → get BULL/BEAR/NEUTRAL prediction
    - Compare predicted direction to actual close vs open of week W
    - Record hit/miss

Stock selection: 1 stock per sector from cached history (no new web requests).
"""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from engine import score_v5, WeekData
from scraper import HIST_CACHE_DIR, _aggregate_weekly, get_all_stock_meta

# ─── CONFIG ──────────────────────────────────────────────────────
BACKTEST_WEEKS  = 12   # How many past weeks to evaluate
BACKTEST_STOCKS = 10   # One per sector
BT_MAX_HISTORY  = 52   # Use up to 52 weekly candles for richer context


# ─── HELPERS ─────────────────────────────────────────────────────

def _load_daily_rows(symbol: str) -> list:
    """Load raw daily OHLCV rows from disk cache. No web fetch."""
    path = HIST_CACHE_DIR / f"{symbol.upper()}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("rows", [])
    except Exception:
        return []


def _get_full_weekly(symbol: str) -> list:
    """Return up to 52 weekly candles for a symbol from cached daily data."""
    rows = _load_daily_rows(symbol)
    return _aggregate_weekly(rows, BT_MAX_HISTORY) if rows else []


def _renumber(weeks: list) -> list:
    """
    Re-assign week_num so the most-recent candle = 1, oldest = N.
    score_v5 uses week_num for the SECTOR_IDX lookup (S5 signal).
    """
    n = len(weeks)
    return [replace(w, week_num=n - i) for i, w in enumerate(weeks)]


# ─── STOCK SELECTION ─────────────────────────────────────────────

def select_backtest_stocks(meta: dict, n: int = BACKTEST_STOCKS) -> list:
    """
    Pick 1 stock per sector (EPS ≥ 0, cache exists, enough history).
    Returns list of symbols up to n.
    """
    seen: dict = {}  # sector → symbol

    for symbol, info in meta.items():
        sector = info.get("sector", "Others")
        if sector in seen:
            continue
        if info.get("eps", 0) < 0:
            continue
        cache_path = HIST_CACHE_DIR / f"{symbol}.json"
        if not cache_path.exists():
            continue
        # Require at least 30 daily rows (~6 weeks) to be useful
        rows = _load_daily_rows(symbol)
        if len(rows) < 30:
            continue
        seen[sector] = symbol
        if len(seen) >= n:
            break

    return list(seen.values())


# ─── SINGLE-STOCK BACKTEST ────────────────────────────────────────

def _backtest_one(symbol: str, meta_info: dict, n_weeks: int) -> Optional[dict]:
    """
    Backtest a single stock over n_weeks.
    Returns result dict or None if insufficient data.
    """
    all_weeks = _get_full_weekly(symbol)

    # Need enough history: n_weeks actual + at least 3 training weeks
    if len(all_weeks) < n_weeks + 3:
        return None

    eps    = meta_info.get("eps",    0.0)
    hi52   = meta_info.get("hi52",   0.0)
    lo52   = meta_info.get("lo52",   0.0)
    sector = meta_info.get("sector", "Others")
    name   = meta_info.get("name",   symbol)

    weekly_results = []

    for week_back in range(1, n_weeks + 1):
        # Training slice: everything BEFORE the target week
        train_end   = len(all_weeks) - week_back   # exclusive upper bound
        if train_end < 3:
            continue

        train_weeks  = _renumber(all_weeks[:train_end])
        actual_week  = all_weeks[-week_back]        # the week we're evaluating
        current_idx  = len(train_weeks) - 1

        try:
            pred, score, signals, plan = score_v5(
                symbol          = symbol,
                weeks           = train_weeks,
                current_idx     = current_idx,
                eps             = eps,
                high52          = hi52,
                low52           = lo52,
                sector          = sector,
                sector_peer_avg = None,   # excluded for speed & isolation
            )
        except Exception:
            continue

        # Actual direction
        actual_ret = 0.0
        actual_dir = "FLAT"
        if actual_week.open and actual_week.close:
            actual_ret = (actual_week.close - actual_week.open) / actual_week.open * 100
            if   actual_ret >  0.5:  actual_dir = "UP"
            elif actual_ret < -0.5:  actual_dir = "DOWN"

        pred_dir = "UP" if pred == "BULL" else "DOWN" if pred == "BEAR" else "NEUTRAL"

        # A hit requires: directional prediction + non-flat actual
        hit = None
        if pred_dir != "NEUTRAL" and actual_dir != "FLAT":
            hit = (pred_dir == actual_dir)

        weekly_results.append({
            "week_back":          week_back,
            "week_label":         actual_week.label,        # e.g. "Mar14"
            "predicted":          pred,                     # BULL / BEAR / NEUTRAL
            "predicted_direction": pred_dir,                # UP / DOWN / NEUTRAL
            "score":              score,
            "entry_price":        round(train_weeks[-1].close, 1),  # price at prediction time
            "actual_open":        actual_week.open,
            "actual_close":       actual_week.close,
            "actual_return_pct":  round(actual_ret, 2),
            "actual_direction":   actual_dir,
            "hit":                hit,
        })

    if not weekly_results:
        return None

    decided  = [r for r in weekly_results if r["hit"] is not None]
    accuracy = round(sum(1 for r in decided if r["hit"]) / len(decided) * 100, 1) if decided else None

    return {
        "symbol":   symbol,
        "name":     name,
        "sector":   sector,
        "accuracy": accuracy,
        "weeks":    weekly_results,
    }


# ─── MAIN BACKTEST RUNNER ─────────────────────────────────────────

def run_backtest(symbols: list = None, n_weeks: int = BACKTEST_WEEKS) -> dict:
    """
    Run direction-accuracy backtest for the past n_weeks.
    If symbols is None, picks 1 stock per sector automatically.

    Performance: ~1-3 seconds (pure calculation from cache, no HTTP).
    """
    # Load metadata from cache (no web fetch)
    cached_symbols = [p.stem for p in HIST_CACHE_DIR.glob("*.json")]
    meta = get_all_stock_meta(cached_symbols)

    if not symbols:
        symbols = select_backtest_stocks(meta, n=BACKTEST_STOCKS)

    print(f"[backtest] {n_weeks}-week backtest | {len(symbols)} stocks: {symbols}")

    stock_results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(_backtest_one, sym, meta.get(sym, {}), n_weeks): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                stock_results.append(result)

    # Stable sort: sector then symbol
    stock_results.sort(key=lambda x: (x["sector"], x["symbol"]))

    # Aggregate stats
    all_hits = [
        r["hit"]
        for s in stock_results
        for r in s["weeks"]
        if r["hit"] is not None
    ]
    overall         = round(sum(all_hits) / len(all_hits) * 100, 1) if all_hits else 0
    total_correct   = sum(all_hits)
    total_wrong     = len(all_hits) - total_correct

    # Per-week accuracy (week_back=1 = last week … n_weeks = oldest)
    per_week = {}
    for wb in range(1, n_weeks + 1):
        hits = []
        for s in stock_results:
            row = next((r for r in s["weeks"] if r["week_back"] == wb), None)
            if row and row["hit"] is not None:
                hits.append(row["hit"])
        per_week[str(wb)] = round(sum(hits) / len(hits) * 100, 1) if hits else None

    print(f"[backtest] Done: {overall}% accuracy ({total_correct}/{len(all_hits)} predictions)")

    return {
        "generated_at":      datetime.now().isoformat(),
        "n_weeks":           n_weeks,
        "n_stocks":          len(stock_results),
        "overall_accuracy":  overall,
        "total_predictions": len(all_hits),
        "total_correct":     total_correct,
        "total_wrong":       total_wrong,
        "per_week_accuracy": per_week,
        "stocks":            stock_results,
    }
