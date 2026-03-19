"""
NEPSE V5 — Friday Outcome Logger
Runs every Friday at 5:00 PM NPT
Fetches actual closing prices, compares to Sunday predictions,
logs accuracy to cache/accuracy_log.json

Run manually:    python logger.py
Run on schedule: python logger.py --schedule
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

CACHE_DIR    = Path("cache")
REPORT_FILE  = CACHE_DIR / "weekly_report.json"
ACCURACY_FILE = CACHE_DIR / "accuracy_log.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ─── FETCH FRIDAY CLOSING PRICE ──────────────────────────────────

def fetch_closing_price(symbol: str) -> float | None:
    """Fetch latest closing price from financialnotices.com"""
    url = f"https://www.financialnotices.com/stock-nepse.php?symbol={symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try h2 current price first
        for h2 in soup.find_all("h2"):
            txt = h2.text.strip().replace("Rs.", "").replace(",", "").strip()
            try:
                return float(txt)
            except ValueError:
                continue

        # Fallback: first row of price history table
        table = soup.find("table")
        if table:
            first_row = table.find_all("tr")
            if len(first_row) > 1:
                tds = first_row[1].find_all("td")
                if len(tds) >= 2:
                    try:
                        return float(tds[1].text.strip().replace(",", ""))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  [logger] Failed to fetch {symbol}: {e}")
    return None

# ─── MAIN LOGGING FUNCTION ────────────────────────────────────────

def log_outcomes():
    """
    Compare this week's predictions against actual Friday closes.
    Updates accuracy_log.json with results.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    now = datetime.now()
    week_str = now.strftime("%Y-W%U")
    date_str = now.strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  NEPSE V5 — Friday Outcome Logger")
    print(f"  {now.strftime('%A, %d %B %Y  %I:%M %p NPT')}")
    print(f"{'='*55}\n")

    # Load this week's report
    if not REPORT_FILE.exists():
        print("  [logger] No weekly report found. Run the backend first on Sunday.")
        return

    report = json.loads(REPORT_FILE.read_text())
    report_date = report.get("generated_date", "")

    # Only process BULL and BEAR predictions (skip NEUTRAL)
    active_calls = [
        s for s in report.get("stocks", [])
        if s.get("prediction") in ("BULL", "BEAR")
    ]

    if not active_calls:
        print("  [logger] No active calls this week (all NEUTRAL). Nothing to log.")
        return

    print(f"  Week report from: {report_date}")
    print(f"  Active calls to verify: {len(active_calls)}\n")

    # Load existing accuracy log
    if ACCURACY_FILE.exists():
        acc_log = json.loads(ACCURACY_FILE.read_text())
    else:
        acc_log = {
            "model_version":       "V5",
            "backtest_accuracy":   84,
            "live_weeks_tracked":  0,
            "live_correct":        0,
            "live_accuracy":       None,
            "weekly_summaries":    [],
            "call_history":        []
        }

    # Check if this week already logged
    already_logged = any(
        w.get("week") == week_str
        for w in acc_log.get("weekly_summaries", [])
    )
    if already_logged:
        print(f"  [logger] Week {week_str} already logged. Use --force to re-log.")
        print_current_accuracy(acc_log)
        return

    # Fetch actual closing prices and compare
    week_correct  = 0
    week_total    = 0
    week_calls    = []

    print(f"  {'Symbol':<10} {'Predicted':<10} {'Entry':<8} {'Close':<8} {'Result':<10} {'P&L %'}")
    print(f"  {'-'*58}")

    for stock in active_calls:
        symbol    = stock["symbol"]
        predicted = stock["prediction"]
        entry     = stock["plan"].get("entry", 0)
        sl        = stock["plan"].get("stop_loss", 0)
        t1        = stock["plan"].get("target_1", 0)

        # Fetch actual Friday close
        actual_close = fetch_closing_price(symbol)
        time.sleep(0.4)  # polite delay

        if actual_close is None:
            print(f"  {symbol:<10} {predicted:<10} {entry:<8} {'N/A':<8} {'SKIP':<10}")
            continue

        # Determine actual direction
        actual_dir = "BULL" if actual_close >= entry else "BEAR"
        correct    = actual_dir == predicted

        # Calculate P&L %
        if predicted == "BULL":
            pnl = round((actual_close - entry) / entry * 100, 2) if entry else 0
        else:
            pnl = round((entry - actual_close) / entry * 100, 2) if entry else 0

        result_str = "✓ CORRECT" if correct else "✗ WRONG"
        pnl_str    = f"+{pnl}%" if pnl >= 0 else f"{pnl}%"

        print(f"  {symbol:<10} {predicted:<10} {entry:<8.1f} {actual_close:<8.1f} {result_str:<10} {pnl_str}")

        week_total  += 1
        if correct:
            week_correct += 1

        week_calls.append({
            "symbol":       symbol,
            "sector":       stock.get("sector", ""),
            "predicted":    predicted,
            "entry_price":  entry,
            "stop_loss":    sl,
            "target_1":     t1,
            "actual_close": actual_close,
            "actual_dir":   actual_dir,
            "correct":      correct,
            "pnl_pct":      pnl,
            "logged_at":    now.isoformat()
        })

    if week_total == 0:
        print("\n  [logger] Could not fetch any closing prices. Try again.")
        return

    # Weekly summary
    week_acc = round(week_correct / week_total * 100, 1)
    print(f"\n  {'-'*58}")
    print(f"  This week: {week_correct}/{week_total} correct = {week_acc}%")

    # Update cumulative log
    acc_log["call_history"].extend(week_calls)
    acc_log["weekly_summaries"].append({
        "week":         week_str,
        "date":         date_str,
        "calls":        week_total,
        "correct":      week_correct,
        "accuracy_pct": week_acc,
    })

    all_calls   = acc_log["call_history"]
    total_calls = len(all_calls)
    total_ok    = sum(1 for c in all_calls if c["correct"])

    acc_log["live_weeks_tracked"] = len(acc_log["weekly_summaries"])
    acc_log["live_correct"]       = total_ok
    acc_log["live_accuracy"]      = round(total_ok / total_calls * 100, 1) if total_calls else None

    ACCURACY_FILE.write_text(json.dumps(acc_log, indent=2))
    print_current_accuracy(acc_log)
    print(f"\n  Saved to {ACCURACY_FILE}\n")

def print_current_accuracy(acc_log: dict):
    live = acc_log.get("live_accuracy")
    bt   = acc_log.get("backtest_accuracy", 84)
    wks  = acc_log.get("live_weeks_tracked", 0)
    print(f"\n  ┌─ Cumulative Live Accuracy ──────────────────┐")
    print(f"  │  Backtest (V5):    {bt}%                        │")
    print(f"  │  Live accuracy:    {str(live)+'%' if live else 'Not enough data yet':<22}   │")
    print(f"  │  Weeks tracked:    {wks:<22}   │")
    print(f"  └────────────────────────────────────────────┘")

# ─── SCHEDULER ────────────────────────────────────────────────────

def run_scheduler():
    """
    Keeps running and triggers the logger every Friday at 5:00 PM NPT.
    Run this once: python logger.py --schedule
    """
    print("  NEPSE V5 Logger — Scheduler started")
    print("  Waiting for Friday 5:00 PM NPT...\n")
    print("  (Press Ctrl+C to stop)\n")

    while True:
        now = datetime.now()
        is_friday = now.weekday() == 4          # Friday = 4
        is_5pm    = now.hour == 17 and now.minute == 0

        if is_friday and is_5pm:
            log_outcomes()
            # Sleep 61 seconds to avoid double-triggering
            time.sleep(61)
        else:
            # Show next trigger time
            days_to_friday = (4 - now.weekday()) % 7 or 7
            next_friday = now.replace(hour=17, minute=0, second=0) + timedelta(days=days_to_friday)
            secs_left = int((next_friday - now).total_seconds())
            hrs  = secs_left // 3600
            mins = (secs_left % 3600) // 60
            print(f"\r  Next run: Friday 5:00 PM NPT  ({hrs}h {mins}m away)   ", end="", flush=True)
            time.sleep(30)

# ─── ENTRY POINT ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPSE V5 Friday Outcome Logger")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule (every Friday 5pm)")
    parser.add_argument("--force",    action="store_true", help="Force re-log even if already logged this week")
    args = parser.parse_args()

    if args.schedule:
        run_scheduler()
    else:
        log_outcomes()
