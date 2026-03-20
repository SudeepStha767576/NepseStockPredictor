"""
NEPSE Data Scraper — Consolidated (V3, high-performance)

Sources:
  merolagani.com   — live prices (1 request, all 342 stocks) + metadata
  sharesansar.com  — OHLCV history (sole source, covers ALL 342 stocks,
                     ~2 years of daily data per stock, 500 rows per request)

Performance:
  Cold start : ~60–90 sec  (was 3–5 min — shared session + 25 workers)
  Same week  : ~3–5 sec    (LTP update only, history from cache)
  New week   : ~25–40 sec  (was 60–90 sec — single CSRF + parallel POSTs)
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from engine import WeekData

BASE_DIR = Path(__file__).resolve().parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

META_CACHE_FILE  = BASE_DIR / "cache" / "stock_meta.json"
HIST_CACHE_DIR   = BASE_DIR / "cache" / "history"
SS_ID_CACHE_FILE = BASE_DIR / "cache" / "ss_company_ids.json"

META_CACHE_DAYS   = 7     # Refresh metadata weekly
MAX_WORKERS       = 25    # Parallel fetch workers (was 10)
MAX_WEEKS         = 52    # 1 full year of weekly candles
SS_ROWS_PER_FETCH = 500   # ~2 years of daily rows per stock

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


# ─── SESSION FACTORY ─────────────────────────────────────────────

def _make_session(pool_size: int = MAX_WORKERS) -> requests.Session:
    """
    Create a requests.Session with:
    - Connection pooling (pool_size connections)
    - Auto-retry (3 attempts, exponential backoff) on 429/5xx
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,                          # 1s, 2s, 4s waits
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update(HEADERS)
    return session


# ─── NEPSE WEEK HELPERS ──────────────────────────────────────────

def current_nepse_week_start() -> str:
    """ISO date of the Sunday that starts the current NEPSE trading week."""
    today = datetime.now()
    days_since_sunday = (today.weekday() + 1) % 7
    return (today - timedelta(days=days_since_sunday)).strftime("%Y-%m-%d")


# ─── LIVE PRICES — merolagani (1 request = all 342 stocks) ──────

def fetch_all_live_prices(session: requests.Session = None) -> dict:
    """Returns { symbol: ltp_float } for every listed stock."""
    url = "https://merolagani.com/LatestMarket.aspx"
    sess = session or _make_session()
    try:
        resp = sess.get(url, timeout=20)
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


# ─── METADATA — merolagani CompanyDetail (cached weekly) ────────

def fetch_stock_meta(symbol: str, session: requests.Session = None) -> dict:
    """Fetch sector, EPS, 52W H/L, name from merolagani CompanyDetail."""
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    result = {"name": symbol, "sector": "Others", "eps": 0.0, "hi52": 0.0, "lo52": 0.0}
    sess = session or _make_session()
    try:
        resp = sess.get(url, timeout=15)
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


def get_all_stock_meta(symbols: list, session: requests.Session = None) -> dict:
    cached = _load_meta_cache()
    missing = [s for s in symbols if s not in cached]
    if missing:
        print(f"[scraper] Fetching metadata for {len(missing)} stocks...")
        sess = session or _make_session()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(fetch_stock_meta, s, sess): s for s in missing}
            done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    cached[sym] = fut.result()
                except Exception as e:
                    print(f"[scraper] Meta failed {sym}: {e}")
                done += 1
                if done % 50 == 0:
                    print(f"[scraper]   metadata {done}/{len(missing)}")
        _save_meta_cache(cached)
        print(f"[scraper] Metadata cached for {len(cached)} stocks")
    return cached


# ─── OHLCV HISTORY — sharesansar.com (sole source) ──────────────

def _load_ss_id_cache() -> dict:
    if not SS_ID_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(SS_ID_CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_ss_id_cache(mapping: dict):
    SS_ID_CACHE_FILE.parent.mkdir(exist_ok=True)
    SS_ID_CACHE_FILE.write_text(json.dumps(mapping, indent=2))


def _ss_get_company_id(symbol: str, session: requests.Session = None) -> str | None:
    """
    Fetch sharesansar company page and extract numeric company ID.
    Reuses the provided session for connection pooling.
    """
    url = f"https://www.sharesansar.com/company/{symbol.lower()}"
    sess = session or _make_session()
    try:
        resp = sess.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        el = soup.find(id="companyid")
        return el.text.strip() if el else None
    except Exception as e:
        print(f"[scraper] SS ID error {symbol}: {e}")
        return None


def _ss_fetch_all_company_ids(symbols: list, session: requests.Session = None) -> dict:
    """
    Bulk-fetch sharesansar company IDs. Uses shared session for connection reuse.
    """
    id_cache = _load_ss_id_cache()
    missing = [s for s in symbols if s not in id_cache]
    if not missing:
        return id_cache

    print(f"[scraper] Fetching sharesansar IDs for {len(missing)} stocks...")
    sess = session or _make_session()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_ss_get_company_id, s, sess): s for s in missing}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                cid = fut.result()
                id_cache[sym] = cid or ""
            except Exception:
                id_cache[sym] = ""
            done += 1
            if done % 50 == 0:
                print(f"[scraper]   SS IDs {done}/{len(missing)}")
                _save_ss_id_cache(id_cache)

    _save_ss_id_cache(id_cache)
    found = sum(1 for v in id_cache.values() if v)
    print(f"[scraper] SS IDs: {found}/{len(symbols)} stocks mapped")
    return id_cache


def _get_ss_csrf(session: requests.Session) -> str:
    """
    Get ONE sharesansar CSRF token for the entire batch.
    This replaces 342 individual GET requests with a single one.
    """
    try:
        resp = session.get(
            "https://www.sharesansar.com/company/nabil",
            timeout=15,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        tag = soup.find("meta", {"name": "_token"})
        if tag:
            return tag["content"]
    except Exception as e:
        print(f"[scraper] CSRF fetch failed: {e}")
    return ""


def _ss_fetch_ohlcv(
    symbol: str,
    company_id: str,
    session: requests.Session,
    csrf: str,
    rows: int = SS_ROWS_PER_FETCH,
) -> list:
    """
    Fetch OHLCV using a shared session + pre-fetched CSRF token.
    No new session or GET request needed — just one POST per stock.
    """
    try:
        resp = session.post(
            "https://www.sharesansar.com/company-price-history",
            data={
                "draw": "1",
                "columns[0][data]": "DT_Row_Index",
                "start": "0",
                "length": str(rows),
                "company": company_id,
            },
            headers={
                "X-CSRF-TOKEN": csrf,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
                "Referer": f"https://www.sharesansar.com/company/{symbol.lower()}",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return []

        raw_rows = resp.json().get("data", [])
        result = []
        for r in raw_rows:
            try:
                dt = _parse_date(r["published_date"])
                if not dt:
                    continue
                result.append({
                    "date":   dt,
                    "open":   float(str(r["open"]).replace(",", "")),
                    "high":   float(str(r["high"]).replace(",", "")),
                    "low":    float(str(r["low"]).replace(",", "")),
                    "close":  float(str(r["close"]).replace(",", "")),
                    "volume": int(float(str(r["traded_quantity"]).replace(",", ""))),
                })
            except (ValueError, KeyError):
                continue
        return sorted(result, key=lambda x: x["date"])

    except Exception as e:
        print(f"[scraper] SS OHLCV error {symbol}: {e}")
        return []


# ─── HISTORY CACHE (INCREMENTAL, PER STOCK) ────────────────────

def _hist_cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return HIST_CACHE_DIR / f"{safe}.json"


def _load_hist_cache(symbol: str) -> dict | None:
    p = _hist_cache_path(symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_hist_cache(symbol: str, week_start: str, daily_rows: list):
    HIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serialisable = [
        {**r, "date": r["date"].strftime("%Y-%m-%d") if isinstance(r["date"], datetime) else r["date"]}
        for r in daily_rows
    ]
    _hist_cache_path(symbol).write_text(json.dumps(
        {"week_start": week_start, "rows": serialisable}, indent=2
    ))


def _parse_date(s: str):
    for fmt in ["%Y-%m-%d", "%d %b %y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _aggregate_weekly(daily_rows: list, max_weeks: int = MAX_WEEKS) -> list:
    """Aggregate daily OHLCV into weekly candles (NEPSE week = Sun–Thu)."""
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


def get_stock_history(
    symbol: str,
    company_id: str,
    ss_session: requests.Session = None,
    ss_csrf: str = "",
) -> list:
    """
    Return weekly OHLCV for a stock.
    Uses incremental cache: only fetches when current week is not cached.
    Accepts shared session + CSRF for zero-overhead batch fetching.
    """
    now_week = current_nepse_week_start()
    cached = _load_hist_cache(symbol)

    if cached and cached.get("week_start") == now_week:
        return _aggregate_weekly(cached.get("rows", []))

    # Need to refresh
    if not company_id or not ss_session or not ss_csrf:
        if cached and cached.get("rows"):
            return _aggregate_weekly(cached["rows"])
        return []

    daily_rows = _ss_fetch_ohlcv(symbol, company_id, ss_session, ss_csrf)

    # Cache result (even empty) to avoid retry this week
    _save_hist_cache(symbol, now_week, daily_rows if daily_rows else [])

    if daily_rows:
        return _aggregate_weekly(daily_rows)

    # Fetch failed — fall back to stale cache
    if cached and cached.get("rows"):
        print(f"[scraper] Using stale cache for {symbol}")
        return _aggregate_weekly(cached["rows"])

    return []


# ─── PARALLEL STOCK DATA LOADER ────────────────────────────────

def _fetch_one_stock(
    symbol: str,
    meta: dict,
    live_prices: dict,
    company_id: str,
    ss_session: requests.Session = None,
    ss_csrf: str = "",
) -> tuple:
    """Worker: returns (symbol, stock_data_dict). Called in parallel."""
    weeks = get_stock_history(symbol, company_id, ss_session, ss_csrf)
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
    Fetch data for ALL NEPSE stocks.

    Performance improvements vs V2:
      - Shared merolagani session (connection reuse across 342 meta fetches)
      - ONE sharesansar CSRF token for entire batch (was 342 GETs, now 1)
      - 25 parallel workers (was 10)
      - Auto-retry with backoff on transient errors

    Cold start  : ~60–90 sec  (was 3–5 min)
    Same week   : ~3–5 sec    (was 5–10 sec)
    New week    : ~25–40 sec  (was 60–90 sec)
    """
    t0 = datetime.now()

    # Shared session for merolagani (connection reuse + retry)
    ml_session = _make_session()

    print("[scraper] Fetching live prices (merolagani)...")
    live_prices = fetch_all_live_prices(ml_session)
    if not live_prices:
        print("[scraper] ERROR: No live prices — merolagani may be unavailable")
        return {}

    symbols = sorted(live_prices.keys())
    now_week = current_nepse_week_start()
    print(f"[scraper] {len(symbols)} stocks | week={now_week}")

    # Pre-fetch sharesansar IDs using shared session
    id_map = _ss_fetch_all_company_ids(symbols, ml_session)

    # Metadata (cached weekly)
    meta = get_all_stock_meta(symbols, ml_session)

    # Count how many stocks need a fresh sharesansar fetch
    cached_count = sum(
        1 for s in symbols
        if (c := _load_hist_cache(s)) and c.get("week_start") == now_week
    )
    fetch_count = len(symbols) - cached_count
    print(f"[scraper] History: {cached_count} cached, {fetch_count} need fetch (sharesansar)")

    # Only establish sharesansar session if we actually need to fetch
    ss_session, ss_csrf = None, ""
    if fetch_count > 0:
        print("[scraper] Connecting to sharesansar (getting CSRF token)...")
        ss_session = _make_session(pool_size=MAX_WORKERS)
        ss_csrf = _get_ss_csrf(ss_session)
        if ss_csrf:
            print(f"[scraper] CSRF OK — fetching {fetch_count} stocks in parallel ({MAX_WORKERS} workers)...")
        else:
            print("[scraper] CSRF failed — will use stale cache for all stocks")

    # Parallel OHLCV fetch (all workers share one ss_session + ss_csrf)
    all_data = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                _fetch_one_stock, s, meta, live_prices,
                id_map.get(s, ""), ss_session, ss_csrf
            ): s
            for s in symbols
        }
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
    has_history = sum(1 for d in all_data.values() if d.get("weeks"))
    print(f"[scraper] Done: {len(all_data)} stocks ({has_history} with history) in {elapsed}s")
    return all_data


# ─── HELPERS ──────────────────────────────────────────────────

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
