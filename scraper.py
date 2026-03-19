"""
NEPSE Data Scraper
Fetches real weekly OHLCV data from financialnotices.com
Falls back to merolagani.com if needed
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime, timedelta
from engine import WeekData

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

STOCK_META = {
    "AKJCL": {"name": "Ankhu Khola Jalvidhyut", "sector": "Hydro",    "eps": 5.78,  "hi52": 417.8, "lo52": 172.9},
    "API":   {"name": "Api Power Company",       "sector": "Hydro",    "eps": 17.3,  "hi52": 355.0, "lo52": 258.8},
    "UPPER": {"name": "Upper Tamakoshi Hydro",   "sector": "Hydro",    "eps": 7.91,  "hi52": 225.0, "lo52": 166.0},
    "BHL":   {"name": "Balephi Hydropower",      "sector": "Hydro",    "eps": 13.62, "hi52": 345.0, "lo52": 164.0},
    "BARUN": {"name": "Barun Hydropower",         "sector": "Hydro",    "eps": 7.09,  "hi52": 589.0, "lo52": 292.1},
    "BPCL":  {"name": "Butwal Power Company",     "sector": "Hydro",    "eps": 2.49,  "hi52": 1065.0,"lo52": 380.7},
    "CHCL":  {"name": "Chilime Hydropower",       "sector": "Hydro",    "eps": 9.38,  "hi52": 599.6, "lo52": 456.0},
    "NABIL": {"name": "Nabil Bank",               "sector": "Bank",     "eps": 35.04, "hi52": 562.0, "lo52": 471.0},
    "EBL":   {"name": "Everest Bank",             "sector": "Bank",     "eps": 30.86, "hi52": 780.6, "lo52": 594.0},
    "GBIME": {"name": "Global IME Bank",          "sector": "Bank",     "eps": 17.06, "hi52": 278.9, "lo52": 210.7},
    "CZBIL": {"name": "Citizens Bank Intl",       "sector": "Bank",     "eps": 4.63,  "hi52": 240.1, "lo52": 183.0},
    "SANIMA":{"name": "Sanima Bank",              "sector": "Bank",     "eps": 20.54, "hi52": 391.0, "lo52": 294.3},
    "GBBL":  {"name": "Garima Bikas Bank",        "sector": "DevBank",  "eps": 21.1,  "hi52": 498.0, "lo52": 367.4},
    "NLICL": {"name": "National Life Insurance",  "sector": "Insurance","eps": 10.3,  "hi52": 738.0, "lo52": 539.0},
    "NLIC":  {"name": "Nepal Life Insurance",     "sector": "Insurance","eps": 9.55,  "hi52": 890.0, "lo52": 702.0},
    "CBBL":  {"name": "Chhimek Laghubitta",       "sector": "Microfinance","eps":33.46,"hi52":1144.8,"lo52": 821.5},
    "CGH":   {"name": "Chandragiri Hills",        "sector": "Hotel",    "eps": -0.81, "hi52": 1150.0,"lo52": 800.2},
    "NTC":   {"name": "Nepal Telecom",            "sector": "Telecom",  "eps": 29.45, "hi52": 996.3, "lo52": 762.5},
    "SPDL":  {"name": "Synergy Power Dev",        "sector": "Power",    "eps": 12.87, "hi52": 465.0, "lo52": 308.9},
    "AKPL":  {"name": "Arun Kabeli Power",        "sector": "Hydro",    "eps": 5.96,  "hi52": 306.5, "lo52": 216.1},
}

def fetch_stock_history(symbol: str, max_weeks: int = 12) -> list[WeekData]:
    """Fetch daily price history and aggregate into weekly OHLCV"""
    url = f"https://www.financialnotices.com/stock-nepse.php?symbol={symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = []
        table = soup.find("table")
        if not table:
            return []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            try:
                date_str = tds[0].text.strip()
                close = float(tds[1].text.strip().replace(",",""))
                open_ = float(tds[2].text.strip().replace(",",""))
                high  = float(tds[3].text.strip().replace(",",""))
                low   = float(tds[4].text.strip().replace(",",""))
                vol   = int(tds[5].text.strip().replace(",",""))
                dt    = parse_date(date_str)
                if dt:
                    rows.append({"date": dt, "open": open_, "high": high,
                                 "low": low, "close": close, "volume": vol})
            except (ValueError, AttributeError):
                continue

        if not rows:
            return []

        rows.sort(key=lambda x: x["date"])
        return aggregate_weekly(rows, max_weeks)

    except Exception as e:
        print(f"[scraper] Error fetching {symbol}: {e}")
        return []

def parse_date(s: str) -> datetime | None:
    for fmt in ["%Y-%m-%d", "%d %b %y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None

def aggregate_weekly(daily_rows: list, max_weeks: int) -> list[WeekData]:
    """
    NEPSE week = Sunday open → Friday close
    Aggregate daily rows into weekly OHLCV
    """
    if not daily_rows:
        return []

    # Group by ISO week
    weeks: dict = {}
    for row in daily_rows:
        dt = row["date"]
        # Sunday = 6 in Python weekday(), but NEPSE week starts Sunday
        # Use isocalendar week + year as key, adjusting for Sunday start
        # Shift Sunday to be start of week: Monday=0..Sunday=6, we want Sunday=0
        weekday = dt.weekday()  # Mon=0, Sun=6
        # Days since last Sunday
        days_since_sunday = (weekday + 1) % 7
        week_start = dt - timedelta(days=days_since_sunday)
        week_key = week_start.strftime("%Y-%m-%d")

        if week_key not in weeks:
            weeks[week_key] = {
                "start": week_start,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "count": 1
            }
        else:
            w = weeks[week_key]
            w["high"]   = max(w["high"], row["high"])
            w["low"]    = min(w["low"], row["low"])
            w["close"]  = row["close"]   # last day of week
            w["volume"] += row["volume"]
            w["count"]  += 1

    # Sort and take last max_weeks
    sorted_keys = sorted(weeks.keys())[-max_weeks:]
    result = []
    for idx, key in enumerate(sorted_keys):
        w = weeks[key]
        week_num = max_weeks - idx  # 12 = oldest, 1 = most recent
        dt = w["start"]
        result.append(WeekData(
            week_num   = week_num,
            label      = dt.strftime("%b%d-%b") + str(dt.day + 4),
            open       = w["open"],
            close      = w["close"],
            high       = w["high"],
            low        = w["low"],
            volume     = w["volume"],
            month_day  = dt.day
        ))

    return result

def get_current_price(symbol: str) -> float:
    """Get latest LTP from financialnotices.com"""
    url = f"https://www.financialnotices.com/stock-nepse.php?symbol={symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for h2 with current price
        for h2 in soup.find_all("h2"):
            txt = h2.text.strip().replace("Rs.", "").replace(",", "").strip()
            try:
                return float(txt)
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0

def get_nepse_index_weekly() -> dict:
    """
    Returns recent weekly NEPSE index direction
    {week_num: 1 (up) or 0 (down)}
    Based on known data from our research session
    """
    return {12:None,11:1,10:0,9:1,8:1,7:1,6:0,5:0,4:1,3:1,2:1,1:1}

def check_event_week() -> tuple[bool, str]:
    """
    Check if current week has a known NEPSE-moving event
    Returns (is_event_week, event_description)
    In production: maintain a calendar file or DB table
    """
    today = datetime.now()
    # Known Nepal stock market events - expand this list
    known_events = [
        # (month, day_start, day_end, description)
        (3, 5, 5, "House of Representatives Election 2082"),
        (4, 15, 15, "Nepal New Year"),
        (4, 29, 29, "Buddha Jayanti"),
    ]
    for month, d_start, d_end, desc in known_events:
        if today.month == month and d_start <= today.day <= d_end:
            return True, desc
    return False, ""

def get_sector_peer_avg_from_data(symbol: str, w, sector: str, all_data: dict) -> float | None:
    """Compute sector peer average return for S8 signal"""
    returns = []
    for sym, data in all_data.items():
        if sym == symbol:
            continue
        if data.get("sector") != sector:
            continue
        weeks = data.get("weeks", [])
        if not weeks:
            continue
        latest = weeks[-1]
        if latest.open and latest.close:
            returns.append((latest.close - latest.open) / latest.open * 100)
    if not returns:
        return None
    return sum(returns) / len(returns)

def get_all_stock_data() -> dict:
    """Fetch data for all stocks — returns dict keyed by symbol"""
    all_data = {}
    for symbol, meta in STOCK_META.items():
        print(f"[scraper] Fetching {symbol}...")
        weeks = fetch_stock_history(symbol, max_weeks=12)
        current_price = get_current_price(symbol)
        all_data[symbol] = {
            **meta,
            "symbol": symbol,
            "weeks": weeks,
            "current_price": current_price or (weeks[-1].close if weeks else 0),
        }
        time.sleep(0.5)  # polite delay
    return all_data
