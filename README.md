# NEPSE V5 Weekly Signal Engine

**84% accuracy · 10 signals · Weekly entry & exit plan**

## What this does
Every Sunday morning, open the website → the engine fetches real NEPSE data,
runs all 10 V5 signals, scores all 20 stocks, and gives you:
- Top 4 BULL picks with full entry / stop-loss / 3 targets
- All stocks ranked by score (descending)
- Automatic NEUTRAL on event weeks (elections, NRB meetings)

---

## Project Structure
```
nepse-v5-app/
├── engine.py        ← V5 scoring logic (10 signals)
├── scraper.py       ← Fetches real NEPSE data from financialnotices.com
├── main.py          ← FastAPI backend server
├── logger.py        ← Friday outcome logger (accuracy tracking)
├── index.html       ← Frontend UI (open in browser)
├── requirements.txt ← Python dependencies
├── cache/           ← Auto-created: stores weekly report JSON
└── README.md
```

---

## Setup (one-time, ~10 minutes)

### Step 1 — Install Python 3.11+
Download from python.org and install.

### Step 2 — Install dependencies
```bash
cd nepse-v5-app
pip install -r requirements.txt
```

### Step 3 — Start the backend
```bash
uvicorn main:app --reload --port 8000
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

### Step 4 — Open the frontend
Double-click `index.html` OR open your browser and go to:
```
file:///C:/Users/SudipShrestha/nepse-v5-app/index.html
```

---

## Every Sunday Morning Routine
1. Start backend: `uvicorn main:app --port 8000`
2. Open `index.html` in browser
3. Wait ~2 minutes for engine to fetch all stocks
4. Read the Top 4 picks — entry, stop-loss, and 3 targets are ready
5. Check "Event Warning" banner — if visible, DO NOT TRADE this week

---

## API Endpoints (when backend is running)

| Endpoint | Description |
|---|---|
| `GET /report` | Full weekly report, all stocks sorted by score |
| `GET /report?refresh=true` | Force re-fetch all data |
| `GET /report/top4` | Only the top 4 BULL picks |
| `GET /stock/AKJCL` | Analysis for single stock |
| `GET /accuracy` | Running accuracy log |

---

## V5 Signals Explained

| Signal | Max Pts | What it measures |
|---|---|---|
| S1 — 2-week momentum | 30 | Were last 2 weeks bullish? |
| S2 — Streak guard | 25 | Prevents 4+ week continuation traps |
| S3 — Volume quality | 25 | Bull week + high volume = confirm |
| S4 — Price position | 20 | Price vs its own 4-week average |
| S5 — Sector sub-index | 20 | Was the sector index up? (−10 penalty if down) |
| S8 — Relative strength | 15 | Is stock outperforming sector peers? |
| S9 — 52W position | 10 | Is entry in safe zone (not near 52W high)? |
| S10 — Monthly cycle | 10 | Week 1/2 = stronger, Week 4 = cautious |
| S6 — EPS gate | Gate | EPS < 0 → always NEUTRAL |
| S7 — Holiday gate | Gate | < 4 trading days → always NEUTRAL |
| **Total possible** | **155** | |

**Thresholds:**
- Score ≥ 90 → **BULL** (buy)
- Score 55–89 → **NEUTRAL** (skip)
- Score ≤ 54 → **BEAR** (avoid)
- Streak ≥ 3 → threshold raised to 85

---

## Adding New Stocks
Edit `STOCK_META` in `scraper.py`:
```python
"NEWSTOCK": {
    "name": "Company Name",
    "sector": "Hydro",        # Hydro / Bank / Insurance / Microfinance / Hotel / Telecom / Power
    "eps": 8.5,               # From latest quarterly report
    "hi52": 300.0,            # 52-week high
    "lo52": 180.0,            # 52-week low
},
```

---

## Deploy to the Web (optional)

**Frontend:** Upload `index.html` to Vercel (vercel.com) — free

**Backend:** Deploy to Railway (railway.app) — free tier
```bash
# In Railway dashboard: connect GitHub repo, set start command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Update `API_BASE` in `index.html` from `http://localhost:8000` to your Railway URL.

---

## Hard Rules (never break these)
1. **Never trade a NEUTRAL week** — the model said skip
2. **Never skip the stop-loss** — 1 unmanaged loss = 5 wins gone
3. **Never trade event weeks** — red banner = stay out
4. **Never trade EPS < 0 stocks** — fundamentally broken
5. **If you miss Sunday scoring** — sit out the whole week

---

## Disclaimer
For educational and research purposes only. Not financial advice.
NEPSE trading involves significant risk. Past performance (84% backtest)
does not guarantee future results. Always DYOR. Trade at your own risk.
