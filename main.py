"""
NEPSE V5 Weekly Report — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json, os
from datetime import datetime
from pathlib import Path

from engine import (
    score_v5, assign_grade, generate_reason,
    SignalBreakdown, PricePlan
)
from scraper import STOCK_META, get_all_stock_data, check_event_week

app = FastAPI(title="NEPSE V5 Weekly Engine", version="1.0.0")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_FILE = Path("cache/weekly_report.json")
ACCURACY_FILE = Path("cache/accuracy_log.json")

def ensure_cache():
    CACHE_FILE.parent.mkdir(exist_ok=True)

# ─── MAIN REPORT GENERATOR ───────────────────────────────────────

def build_weekly_report(force_refresh: bool = False) -> dict:
    ensure_cache()

    # Return cached if same day and not forced
    if CACHE_FILE.exists() and not force_refresh:
        data = json.loads(CACHE_FILE.read_text())
        cached_date = data.get("generated_date", "")
        if cached_date == datetime.now().strftime("%Y-%m-%d"):
            return data

    print("[engine] Building weekly report...")

    # Fetch all stock data
    all_data = get_all_stock_data()
    is_event, event_desc = check_event_week()

    # ── V7: Compute NEPSE market breadth (S11) ─────────────────────
    # Median cross-sectional return of all stocks last completed week
    market_breadth = None
    try:
        all_rets = []
        for _sym, _d in all_data.items():
            _wks = _d.get("weeks", [])
            if len(_wks) >= 2:
                lw = _wks[-1]
                if lw.open and lw.close:
                    all_rets.append((lw.close - lw.open) / lw.open * 100)
        if len(all_rets) >= 10:
            all_rets.sort()
            n = len(all_rets)
            market_breadth = round((all_rets[n//2-1] + all_rets[n//2]) / 2 if n % 2 == 0 else all_rets[n//2], 2)
            print(f"[engine] Market breadth (median return): {market_breadth:+.2f}%")
    except Exception as e:
        print(f"[engine] Market breadth compute failed: {e}")

    results = []

    for symbol, data in all_data.items():
        weeks = data["weeks"]
        if len(weeks) < 3:
            continue  # Need at least 3 weeks of data

        current_idx = len(weeks) - 1  # Most recent week
        w = weeks[current_idx]
        eps = data["eps"]

        # Compute sector peer average for S8
        sector_peer_avg = None
        try:
            from scraper import get_sector_peer_avg_from_data
            sector_peer_avg = get_sector_peer_avg_from_data(symbol, w, data["sector"], all_data)
        except Exception:
            pass

        # Run V7 engine
        pred, score, signals, plan = score_v5(
            symbol               = symbol,
            weeks                = weeks,
            current_idx          = current_idx,
            eps                  = eps,
            high52               = data["hi52"],
            low52                = data["lo52"],
            sector               = data["sector"],
            sector_peer_avg      = sector_peer_avg,
            nepse_market_return  = market_breadth,
        )

        # If event week → force NEUTRAL
        if is_event and pred != "NEUTRAL":
            pred = "NEUTRAL"
            reason_extra = f" [EVENT WEEK: {event_desc}]"
        else:
            reason_extra = ""

        pct_change = round((w.close - w.open) / w.open * 100, 2) if w.open else 0

        result = {
            "symbol":         symbol,
            "name":           data["name"],
            "sector":         data["sector"],
            "eps":            eps,
            "current_price":  data["current_price"],
            "last_week_close": w.close,
            "last_week_open":  w.open,
            "pct_change":     pct_change,
            "week52_high":    data["hi52"],
            "week52_low":     data["lo52"],
            "prediction":     pred,
            "score":          score,
            "grade":          assign_grade(score, pred),
            "reason":         generate_reason(signals, pred, eps) + reason_extra,
            "signals": {
                "s1_momentum":    signals.s1_momentum,
                "s2_streak":      signals.s2_streak,
                "s3_volume":      signals.s3_volume,
                "s4_position":    signals.s4_position,
                "s5_sector":      signals.s5_sector,
                "s6_rsi":         signals.s6_rsi,
                "s7_ema":         signals.s7_ema,
                "s8_rel_strength":signals.s8_rel_strength,
                "s9_week52":      signals.s9_week52,
                "s10_monthly":    signals.s10_monthly,
                "s11_market":     signals.s11_market,
                "total":          signals.total,
                "streak":         signals.streak_count,
                "streak_dir":     signals.streak_dir,
                "atr_flag":       signals.atr_flag,
                "atr_value":      signals.atr_value,
                "rsi_value":      signals.rsi_value,
                "bull_threshold": signals.bull_threshold,
            },
            "plan": {
                "entry":       plan.entry,
                "stop_loss":   plan.stop_loss,
                "target_1":    plan.target_1,
                "target_2":    plan.target_2,
                "target_3":    plan.target_3,
                "band_low":    plan.band_low,
                "band_high":   plan.band_high,
                "risk_reward": plan.risk_reward,
            }
        }
        results.append(result)

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Mark top 4 BULL picks
    bull_count = 0
    for r in results:
        if r["prediction"] == "BULL" and bull_count < 4:
            r["top_pick"] = True
            bull_count += 1
        else:
            r["top_pick"] = False

    report = {
        "generated_date":   datetime.now().strftime("%Y-%m-%d"),
        "generated_time":   datetime.now().strftime("%H:%M NPT"),
        "week_label":       get_week_label(),
        "nepse_index_dir":  1,  # Update from live NEPSE
        "is_event_week":    is_event,
        "event_description": event_desc,
        "total_stocks":     len(results),
        "bull_count":       sum(1 for r in results if r["prediction"] == "BULL"),
        "neutral_count":    sum(1 for r in results if r["prediction"] == "NEUTRAL"),
        "bear_count":       sum(1 for r in results if r["prediction"] == "BEAR"),
        "model_accuracy":   None,  # populated by /backtest endpoint
        "model_version":    "V6",
        "stocks":           results,
    }

    CACHE_FILE.write_text(json.dumps(report, indent=2))
    print(f"[engine] Report built: {len(results)} stocks, {bull_count} BULL picks")
    return report

def get_week_label() -> str:
    now = datetime.now()
    # Next Friday
    days_to_friday = (4 - now.weekday()) % 7
    if days_to_friday == 0: days_to_friday = 7
    from datetime import timedelta
    friday = now + timedelta(days=days_to_friday)
    return f"Week of {now.strftime('%b %d')} → {friday.strftime('%b %d, %Y')}"

# ─── API ENDPOINTS ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "NEPSE V5 Weekly Engine — call /report for this week's picks"}

@app.get("/report")
def get_report(refresh: bool = False):
    """Get this week's full report, sorted by score descending"""
    try:
        report = build_weekly_report(force_refresh=refresh)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/top4")
def get_top4():
    """Get only the top 4 BULL picks"""
    report = build_weekly_report()
    top4 = [s for s in report["stocks"] if s.get("top_pick")]
    return {
        "week_label":    report["week_label"],
        "generated_date": report["generated_date"],
        "is_event_week": report["is_event_week"],
        "event_description": report["event_description"],
        "top_picks":     top4
    }

@app.get("/stock/{symbol}")
def get_stock(symbol: str):
    """Get detailed analysis for a single stock"""
    report = build_weekly_report()
    stock = next((s for s in report["stocks"] if s["symbol"] == symbol.upper()), None)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    return stock

@app.get("/accuracy")
def get_accuracy():
    """Get running accuracy log from past predictions"""
    ensure_cache()
    if ACCURACY_FILE.exists():
        return json.loads(ACCURACY_FILE.read_text())
    return {
        "model_version": "V5",
        "backtest_accuracy": 84,
        "live_weeks_tracked": 0,
        "live_correct": 0,
        "live_accuracy": None,
        "history": []
    }

@app.post("/accuracy/log")
def log_outcome(symbol: str, predicted: str, actual: str, week: str):
    """Log a prediction outcome for accuracy tracking"""
    ensure_cache()
    acc = json.loads(ACCURACY_FILE.read_text()) if ACCURACY_FILE.exists() else {
        "model_version": "V5",
        "backtest_accuracy": 84,
        "live_weeks_tracked": 0,
        "live_correct": 0,
        "live_accuracy": None,
        "history": []
    }
    correct = predicted.upper() == actual.upper()
    acc["history"].append({
        "week": week, "symbol": symbol,
        "predicted": predicted, "actual": actual,
        "correct": correct,
        "logged_at": datetime.now().isoformat()
    })
    acc["live_weeks_tracked"] = len(acc["history"])
    acc["live_correct"] = sum(1 for h in acc["history"] if h["correct"])
    if acc["live_weeks_tracked"] > 0:
        acc["live_accuracy"] = round(acc["live_correct"] / acc["live_weeks_tracked"] * 100, 1)
    ACCURACY_FILE.write_text(json.dumps(acc, indent=2))
    return acc

@app.get("/backtest")
def get_backtest(weeks: int = 12, symbols: str = None):
    """
    Run walk-forward accuracy backtest for the past N weeks.
    Picks 1 stock per sector automatically, or pass ?symbols=UPPER,NABIL,...
    Only uses cached history — no new web requests needed.
    """
    try:
        from backtest import run_backtest
        sym_list = (
            [s.strip().upper() for s in symbols.split(",")]
            if symbols else None
        )
        return run_backtest(symbols=sym_list, n_weeks=min(int(weeks), 24))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat(), "version": "V5"}
