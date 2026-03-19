"""
NEPSE Data Scraper — Optimized
- Live prices for all 342 stocks: 1 request (merolagani LatestMarket)
- Metadata (sector, EPS, 52W H/L): cached weekly (merolagani CompanyDetail)
- OHLCV history: incremental cache per stock — only re-fetches when new week arrives
- Parallel fetching: 12 concurrent workers (ThreadPoolExecutor)

Performance:
  First run : ~4–5 min  (cold start: fetch all 342 histories + metadata)
  Same-week : ~5–10 sec (LTP update only, history served from cache)
  New week  : ~30–60 sec (parallel fetch of latest week for all stocks)
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from engine import WeekData

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

META_CACHE_FILE  = Path("cache/stock_meta.json")
HIST_CACHE_DIR   = Path("cache/history")
META_CACHE_DAYS  = 7    # Refresh metadata weekly
MAX_WORKERS      = 12   # Parallel fetch workers
MAX_WEEKS        = 12   # Weeks of history to keep

SECTOR_MAP = {
    "hydro power":                  "Hydro",
    "commercial banks":             "Bank",
    "development banks":            "DevBank",
    "finance":                      "Finance",
    "microfinance":                 "Microfinance",
    "life insurance":               "Insurance",
    "non life insurance":           "Insurance",
    "hotels and tourism":           "Hotel",
    "telecom":                      "Telecom",
    "manufacturing and processing": "Manufacturing",
    "tradings":                     "Trading",
    "others":                       "Others",
    "investment":                   "Investment",
    "mutual fund":                  "MutualFund",
    "hydropower":                   "Hydro",
    "power":                        "Power",
}

# Kept for backward compatibility
STOCK_META = {}


# ─── NEPSE WEEK HELPERS ───────────────────────────────────────────

def current_nepse_week_start() -> str:
    """ISO date of the Sunday that starts the current NEPSE trading week."""
    today = datetime.now()
    days_since_sunday = (today.weekday() + 1) % 7   # Mon=1 … Sun=0
    return (today - timedelta(days=days_since_sunday)).strftime("%Y-%m-%d")


# ─── LIVE PRICES (1 REQUEST FOR ALL 342 STOCKS) ───────────────────

def fetch_all_live_prices() -> dict:
    """Returns { symbol: ltp_float } for every listed stock."""
    url = "https://merolagani.com/LatestMarket.aspx"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            return {}
        prices = {}
        for row in table.find_all("tr")[1:]:
            cols = [td.text.strip() for td in row.find_all("td")]
            if len(cols) >= 2:
                try:
                    prices[cols[0]] = float(cols[1].replace(",", ""))
                except ValueError:
                    pass
        print(f"[scraper] Live prices fetched: {len(prices)} stocks")
        return prices
    except Exception as e:
        print(f"[scraper] Error fetching live prices: {e}")
        return {}


# ─── METADATA (CACHED WEEKLY) ─────────────────────────────────────

def fetch_stock_meta(symbol: str) -> dict:
    """Fetch sector, EPS, 52W H/L, name from merolagani CompanyDetail."""
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    result = {"name": symbol, "sector": "Others", "eps": 0.0, "hi52": 0.0, "lo52": 0.0}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")

        for t in tables:
            for row in t.find_all("tr"):
                cols = [td.text.strip() for td in row.find_all(["td", "th"])]
                if len(cols) < 2:
                    continue
                label, val = cols[0], cols[1]
                if label == "Company Name":
                    result["name"] = val
                elif label == "Sector":
                    result["sector"] = _map_sector(val)

        if tables:
            for row in tables[0].find_all("tr"):
                cols = [td.text.strip() for td in row.find_all(["td", "th"])]
                if len(cols) < 2:
                    continue
                label, val = cols[0], cols[1]
                if label == "52 Weeks High - Low":
                    parts = val.replace(",", "").split("-")
                    if len(parts) == 2:
                        try:
                            result["hi52"] = float(parts[0])
                            result["lo52"] = float(parts[1])
                        except ValueError:
                            pass
                elif label == "EPS":
                    eps_str = re.sub(r"\(.*?\)", "", val).strip()
                    try:
                        result["eps"] = float(eps_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[scraper] Meta error {symbol}: {e}")
    return result


def _map_sector(raw: str) -> str:
    return SECTOR_MAP.get(raw.strip().lower(), "Others")


def _load_meta_cache() -> dict:
    if not META_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(META_CACHE_FILE.read_text())
        age = (datetime.now() - datetime.fromisoformat(data.get("_cached_on", "2000-01-01"))).days
        if age < META_CACHE_DAYS:
            return data.get("stocks", {})
    except Exception:
        pass
    return {}


def _save_meta_cache(meta: dict):
    META_CACHE_FILE.parent.mkdir(exist_ok=True)
    META_CACHE_FILE.write_text(json.dumps(
        {"_cached_on": datetime.now().isoformat(), "stocks": meta}, indent=2
    ))


def get_all_stock_meta(symbols: list) -> dict:
    cached = _load_meta_cache()
    missing = [s for s in symbols if s not in cached]
    if missing:
        print(f"[scraper] Fetching metadata for {len(missing)} stocks (parallel)...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(fetch_stock_meta, s): s for s in missing}
            done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    cached[sym] = fut.result()
                except Exception as e:
                    print(f"[scraper] Meta failed {sym}: {e}")
                done += 1
                if done % 30 == 0:
                    print(f"[scraper]   metadata {done}/{len(missing)}")
        _save_meta_cache(cached)
        print(f"[scraper] Metadata cached for {len(cached)} stocks")
    return cached


# ─── HISTORY CACHE (INCREMENTAL, PER STOCK) ──────────────────────

def _hist_cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return HIST_CACHE_DIR / f"{safe}.json"


def _load_hist_cache(symbol: str) -> dict | None:
    """Returns {"week_start": str, "rows": [...daily rows with date as str]}"""
    p = _hist_cache_path(symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_hist_cache(symbol: str, week_start: str, daily_rows: list):
    HIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Serialise dates to strings for JSON
    serialisable = [
        {**r, "date": r["date"].strftime("%Y-%m-%d") if isinstance(r["date"], datetime) else r["date"]}
        for r in daily_rows
    ]
    _hist_cache_path(symbol).write_text(json.dumps(
        {"week_start": week_start, "rows": serialisable}, indent=2
    ))


# ─── HISTORY FETCH FROM FINANCIALNOTICES ─────────────────────────

def _fetch_daily_rows(symbol: str) -> list:
    """Fetch raw daily OHLCV rows from financialnotices.com."""
    url = f"https://www.financialnotices.com/stock-nepse.php?symbol={symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        table = tables[1] if len(tables) >= 2 else (tables[0] if tables else None)
        if not table:
            return []
        rows = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            try:
                dt = _parse_date(tds[0].text.strip())
                if not dt:
                    continue
                rows.append({
                    "date":   dt,
                    "open":   float(tds[2].text.strip().replace(",", "")),
                    "high":   float(tds[3].text.strip().replace(",", "")),
                    "low":    float(tds[4].text.strip().replace(",", "")),
                    "close":  float(tds[1].text.strip().replace(",", "")),
                    "volume": int(tds[5].text.strip().replace(",", "")),
                })
            except (ValueError, AttributeError):
                continue
        return sorted(rows, key=lambda x: x["date"])
    except Exception as e:
        print(f"[scraper] History error {symbol}: {e}")
        return []


def _parse_date(s: str):
    for fmt in ["%Y-%m-%d", "%d %b %y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _aggregate_weekly(daily_rows: list, max_weeks: int) -> list:
    """Aggregate daily OHLCV into weekly candles (NEPSE week = Sun→Thu)."""
    weeks: dict = {}
    for row in daily_rows:
        dt = row["date"] if isinstance(row["date"], datetime) else datetime.fromisoformat(row["date"])
        days_since_sunday = (dt.weekday() + 1) % 7
        week_key = (dt - timedelta(days=days_since_sunday)).strftime("%Y-%m-%d")
        if week_key not in weeks:
            weeks[week_key] = {
                "start":  dt - timedelta(days=days_since_sunday),
                "open":   row["open"], "high": row["high"],
                "low":    row["low"],  "close": row["close"],
                "volume": row["volume"],
            }
        else:
            w = weeks[week_key]
            w["high"]    = max(w["high"], row["high"])
            w["low"]     = min(w["low"],  row["low"])
            w["close"]   = row["close"]
            w["volume"] += row["volume"]

    sorted_keys = sorted(weeks.keys())[-max_weeks:]
    result = []
    for idx, key in enumerate(sorted_keys):
        w = weeks[key]
        week_num = max_weeks - idx
        dt = w["start"]
        result.append(WeekData(
            week_num=week_num, label=dt.strftime("%b%d"),
            open=w["open"], close=w["close"],
            high=w["high"], low=w["low"],
            volume=w["volume"], month_day=dt.day,
        ))
    return result


def get_stock_history(symbol: str) -> list:
    """
    Return weekly OHLCV for a stock.
    Uses incremental cache: only fetches from web if current week is not cached.
    Empty results are also cached so failures aren't retried every run.
    """
    now_week = current_nepse_week_start()
    cached = _load_hist_cache(symbol)

    if cached and cached.get("week_start") == now_week:
        # Cache is current (including cached empty result) — no web fetch needed
        return _aggregate_weekly(cached["rows"], MAX_WEEKS)

    # Need to refresh — fetch from financialnotices
    daily_rows = _fetch_daily_rows(symbol)

    # Always cache result (even empty) so we don't retry failed stocks this week
    _save_hist_cache(symbol, now_week, daily_rows if daily_rows else [])

    if daily_rows:
        return _aggregate_weekly(daily_rows, MAX_WEEKS)

    # Fetch failed — fall back to stale cache rows if available
    if cached and cached.get("rows"):
        print(f"[scraper] Using stale cache for {symbol}")
        return _aggregate_weekly(cached["rows"], MAX_WEEKS)

    return []


# ─── PARALLEL STOCK DATA LOADER ──────────────────────────────────

def _fetch_one_stock(symbol: str, meta: dict, live_prices: dict) -> tuple:
    """Worker: returns (symbol, stock_data_dict). Called in parallel."""
    weeks = get_stock_history(symbol)
    ltp = live_prices.get(symbol, 0.0)
    current_price = ltp if ltp else (weeks[-1].close if weeks else 0.0)
    m = meta.get(symbol, {})
    return symbol, {
        "symbol":        symbol,
        "name":          m.get("name", symbol),
        "sector":        m.get("sector", "Others"),
        "eps":           m.get("eps", 0.0),
        "hi52":          m.get("hi52", 0.0),
        "lo52":          m.get("lo52", 0.0),
        "weeks":         weeks,
        "current_price": current_price,
    }


def get_all_stock_data() -> dict:
    """
    Fetch data for ALL NEPSE stocks with caching + parallelism.

    Cold start  : ~4–5 min (builds history + metadata cache)
    Same week   : ~5–10 sec (only LTP update, history from cache)
    New week    : ~30–60 sec (parallel history refresh)
    """
    t0 = datetime.now()

    print("[scraper] Fetching live prices...")
    live_prices = fetch_all_live_prices()
    if not live_prices:
        print("[scraper] ERROR: No live prices")
        return {}

    symbols = sorted(live_prices.keys())
    print(f"[scraper] {len(symbols)} stocks | week={current_nepse_week_start()}")

    # Count how many need a web fetch vs can use cache
    now_week = current_nepse_week_start()
    cached_count  = sum(1 for s in symbols if _load_hist_cache(s) and
                        _load_hist_cache(s).get("week_start") == now_week)
    fetch_count   = len(symbols) - cached_count
    print(f"[scraper] History: {cached_count} cached, {fetch_count} need fetch")

    # Metadata
    meta = get_all_stock_meta(symbols)

    # Parallel history fetch
    all_data = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one_stock, s, meta, live_prices): s for s in symbols}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                symbol, data = fut.result()
                all_data[symbol] = data
            except Exception as e:
                print(f"[scraper] Failed {sym}: {e}")
            done += 1
            if done % 50 == 0:
                print(f"[scraper]   {done}/{len(symbols)} stocks done")

    elapsed = (datetime.now() - t0).seconds
    print(f"[scraper] Done: {len(all_data)} stocks in {elapsed}s")
    return all_data


# ─── HELPERS ─────────────────────────────────────────────────────

def check_event_week() -> tuple:
    today = datetime.now()
    known_events = [
        (3, 5,  5,  "House of Representatives Election 2082"),
        (4, 15, 15, "Nepal New Year"),
        (4, 29, 29, "Buddha Jayanti"),
    ]
    for month, d_start, d_end, desc in known_events:
        if today.month == month and d_start <= today.day <= d_end:
            return True, desc
    return False, ""


def get_sector_peer_avg_from_data(symbol: str, w, sector: str, all_data: dict):
    returns = []
    for sym, data in all_data.items():
        if sym == symbol or data.get("sector") != sector:
            continue
        weeks = data.get("weeks", [])
        if weeks and weeks[-1].open and weeks[-1].close:
            returns.append((weeks[-1].close - weeks[-1].open) / weeks[-1].open * 100)
    return (sum(returns) / len(returns)) if returns else None
