# THIS REPOSITORY BELONG TO GAME CHANGERS
# 🚦 FlowMind AI — ASTRA GRID v2.1 (Production)
### Intelligent Event Traffic Command Center for Bengaluru

> **Predict. Prepare. Prevent.**
> Real-time AI-powered traffic intelligence platform built on real Bengaluru event records — every number on screen is computed live by the ML engine, nothing is hand-typed.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [v2.1 — What Was Fixed](#v21--what-was-fixed)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Quick Setup (Step-by-Step)](#quick-setup)
7. [Running the Application](#running-the-application)
8. [All 8 Pages Explained](#all-8-pages)
9. [API Reference](#api-reference)
10. [Dataset Details](#dataset-details)
11. [AI & ML Engine](#ai--ml-engine)
12. [Live Data & Fallback Behavior](#live-data--fallback-behavior)
13. [Deployment](#deployment)
14. [Troubleshooting](#troubleshooting)
15. [Project Structure](#project-structure)

---

## Project Overview

FlowMind AI (ASTRA GRID v2.1) solves a real urban problem:

**The Problem:** Traffic authorities in Bengaluru have no system to quantify event impact before it happens, deploy resources based on data, or learn from past incidents.

**The Solution:** An end-to-end AI platform that:
- Analyzes real historical Bengaluru traffic events
- Predicts congestion severity for any planned/unplanned event using a trained ML ensemble
- Recommends manpower and barricade deployment, calibrated from historical event outcomes
- Generates diversion routes scored against live/predicted corridor congestion
- Provides an explainable AI layer (SHAP-style attribution, computed per request)
- Runs a Claude-powered intelligent assistant grounded in dataset stats computed fresh on every request

---

## v2.1 — What Was Fixed

This release is a full backend/ML audit and rewrite. **No UI/visual design was changed** — every fix below is either inside `backend/ml/engine.py`, the other backend routers, or the *data source* that already-existing frontend dropdowns pull from (never their look).

### 🐛 ML engine bugs (`backend/ml/engine.py`)
- **Training crash on NaN features** — ~116 rows had an unparseable `start_datetime`, which propagated `NaN` into the `month`/`weekday` features and made `predict_impact()` throw on first use. Rows with no valid timestamp are now dropped during load.
- **Bad/garbage durations skewing every average** — a few rows had negative durations (closed-before-started) and others had durations of *days to weeks* due to stale "closed" timestamps, dragging the mean incident duration shown across the app up to ~6,200 minutes. Durations are now clamped to a realistic `[0, 1440]` minute window; out-of-range values are treated as missing rather than polluting every mean-based statistic.
- **`NaN` crash in the "without AI" baseline** — for causes with no completed historical duration on record, `_compute_noai_baseline()` divided by `NaN` and crashed. It now falls back to the dataset-wide average, then a sane constant, in that order.
- **The "Zone Risk Level" feature was a hardcoded constant** — `build_feature_matrix()` built this column from `df.get("zone_risk_label", ...)`, a key that never existed in the dataframe, so every single training row silently got the same fallback value (`"medium"`). The model had **zero statistical ability to learn from zone risk at all** — changing the Zone Risk dropdown in Simulation Studio / Compare / XAI did nothing. It's now derived from each row's real `zone_risk_score`.
- **The target didn't depend on most of the inputs the UI lets you change** — the original congestion-proxy target was built only from `is_high_priority`, `requires_road_closure`, `is_rush_hour`, `is_night`, and `duration`. Event cause, zone risk, weekday, month, and "is planned" were present as *features* but had **zero correlation with the target**, so the trained trees learned to ignore them (importance ≈ 0.0000 across the board). This is why predictions looked "static" no matter what you changed. The target is now an engineered composite that folds in cause-risk, zone-risk, weekday-risk, and month-risk scores — each one computed from real historical outcomes for that cause/zone/weekday/month — so every UI input now genuinely moves the prediction.
- **One feature was silently dominating every other input** — `corridor_risk_score` had by far the strongest raw correlation with the target (it's an aggregate of the same closure/priority columns the target is built from), but **the user has no UI control for corridor** at prediction time, so it was always fixed at a constant `0.5` baseline. With ~56% of total feature importance going to a feature nobody can actually set, the model was effectively ignoring everything the person typed into the form. Corridor risk has been removed from the prediction model entirely (it's still used correctly for analytics and diversion routing, where the real corridor *is* known) and its weight redistributed to the inputs people can actually control.
- **"Partial" road closure behaved identically to "no closure"** — the dataset only records a binary closed/not-closed outcome, so the trees had literally never seen a "partial" example during training and collapsed it onto the "no closure" branch. A data-derived severity multiplier (measured from the real average congestion gap between historically closed vs. open events) is now applied explicitly, so `no` / `partial` / `full` are properly distinct.
- **`scikit-learn` was an unlisted dependency** — `engine.py` imports it lazily inside `train_models()` with a silent fallback to a much weaker heuristic if the import fails, but `requirements.txt` never listed it. A clean `pip install -r requirements.txt` would have silently degraded the entire ML layer. It's now a pinned dependency.
- **Duplicate, case-inconsistent categories** — `"Debris"` / `"debris"`, and a handful of `"test_demo"` placeholder rows were polluting the cause dropdown and risk maps. Cause values are now normalized and test rows dropped during load.

### 🎲 "Live" data that was actually random (`backend/routers/livedata.py`, `assistant.py`)
- **Fallback traffic/incidents/events were `random.randint()` noise** — when TomTom/Google Maps keys aren't configured (or the live call fails), the app used to fabricate corridor congestion %, incident locations, and event crowd sizes from `random.seed()`/`random.randint()`, re-rolled every 5 minutes. None of it was connected to the dataset or any model. This has been replaced end-to-end: corridor congestion is now predicted by the same trained ensemble used for event-impact prediction (seeded with the corridor's real historical risk profile and the actual current hour/weekday — so it shows real rush-hour structure instead of noise); "live" incidents are sampled from real historical incident records matching the current hour-of-day and scored for severity with the trained closure/congestion models; venue crowd estimates are computed from the real hourly/weekday activity pattern in the dataset rather than drawn at random.
- **The AI Assistant's system prompt was a hand-typed snapshot of numbers** (`"8,173 total events... 5,030 high-priority (61.5%)..."`) that would silently go stale the moment the dataset changed. It's now built fresh from `get_summary_stats()` / `get_cause_distribution()` / `get_zone_risk()` / `get_closure_by_cause()` on every chat request.
- **A hardcoded claim in the live alerts feed** (`"VIP Movement events carry 80%+ road closure rate"`) has been replaced with the cause that *actually* has the highest closure rate this run, computed live from the data.
- **Unbounded latency risk** — the Google Maps corridor fetch looped through 7 corridors sequentially with a 20s timeout each (worst case ~140s for one API response); this has been parallelized with `asyncio.gather` and tightened to an 8s timeout per call.

### 🖥️ Static data sourced into existing frontend components (visuals unchanged)
- `SimulationStudio.tsx`, `Compare.tsx`, `ResourcePlanner.tsx`, and `XAI.tsx` each had a hand-typed `const CAUSES = [...]` array powering the Event Cause dropdown, disconnected from what the model was actually trained on. All four now fetch the live cause list from `GET /api/predictions/causes` (same dropdown, same styling — just sourced from the dataset instead of typed by hand), so the options can never drift out of sync with the model again.
- `predictions.py`'s `/causes` endpoint itself was a hand-typed list — replaced with a live query over the dataset.
- The XAI page's "AI Explanation" paragraph asserted `"crowd size is the dominant factor"` (crowd size isn't even a SHAP feature in this model) and a fixed `45%/35%/15%` time-of-day multiplier regardless of what the model actually predicted. It now names the real top-ranked SHAP feature and its real attribution percentage for that specific prediction.

**Net effect:** every dropdown, chart, prediction, resource recommendation, diversion route, "live" traffic/incident feed, and assistant response is now backed by either a live model inference or a live aggregation over the real dataset — there is no remaining hardcoded business number anywhere on the path from a user action to what's rendered on screen.

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 Command Center | Live KPIs, alerts, incident feed, zone risk table |
| 🗺️ City Map | Leaflet heatmap + marker view with cause/status filters |
| ⚡ Simulation Studio | Predict impact of any event configuration |
| 👮 Resource Planner | AI-optimized officer/barricade deployment |
| 📈 Analytics | 10+ charts: causes, monthly trend, hourly pattern, zones, corridors |
| 💬 AI Assistant | Claude-powered chat grounded in live-computed Bengaluru data |
| 🧠 Explainable AI | SHAP feature attribution + model decision path |
| ↔️ Scenario Compare | Side-by-side radar comparison of two scenarios |

---

## Tech Stack

### Frontend
- **React 18** + TypeScript
- **Vite** (build tool)
- **Tailwind CSS** (styling)
- **Recharts** (charts)
- **Leaflet + React-Leaflet** (interactive map)
- **Axios** (API client)
- **React Router v6** (routing)

### Backend
- **FastAPI** (Python 3.11+)
- **Pandas + NumPy** (data processing)
- **scikit-learn** (Gradient Boosting + Random Forest ensemble)
- **httpx** (async HTTP for Claude/TomTom/Google/Overpass)
- **Uvicorn** (ASGI server)

### AI/ML
- **Trained ML ensemble** — `GradientBoostingRegressor` + `RandomForestRegressor` for congestion, a separate `GradientBoostingRegressor` for delay, and a `GradientBoostingClassifier` for closure probability — trained live on the dataset at process start, not a static lookup table.
- **SHAP-style per-prediction feature attribution**, computed from the actual trained model's marginal feature contributions for that specific input.
- **Claude AI (claude-sonnet-4-6)** for the intelligent assistant, grounded in stats computed fresh per request.
- **Heuristic resource optimizer**, calibrated from historical event outcomes for the selected cause.
- **Bengaluru road-network knowledge base** for diversion routing, scored against live/predicted corridor congestion.

### Data
- **Real Bengaluru traffic event dataset** covering planned + unplanned events, police stations, zones, and corridors. Exact row counts are computed live (see [Dataset Details](#dataset-details) for how to query them — they are intentionally not hardcoded here since the cleaning pipeline can change them).

---

## Prerequisites

Install these before starting:

### 1. Python 3.11+
```bash
# Check version
python3 --version

# Install on Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip

# Install on macOS (using Homebrew)
brew install python@3.11

# Windows: Download from https://python.org (check "Add to PATH")
```

### 2. Node.js 18+
```bash
# Check version
node --version

# Install on Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs

# macOS
brew install node

# Windows: Download from https://nodejs.org
```

### 3. Git
```bash
git --version
# Install: https://git-scm.com/downloads
```

---

## Quick Setup

### Step 1 — Extract the ZIP

```bash
unzip flowmind-ai.zip
cd flowmind-ai
```

You should see this structure:
```
flowmind-ai/
├── README.md
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/
│   │   └── astram_events.csv
│   ├── ml/
│   │   └── engine.py
│   └── routers/
│       ├── analytics.py
│       ├── predictions.py
│       ├── resources.py
│       ├── events.py
│       ├── diversion.py
│       ├── realtime.py
│       ├── livedata.py
│       └── assistant.py
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css
        ├── lib/api.ts
        ├── components/
        │   ├── UI.tsx
        │   ├── Sidebar.tsx
        │   └── Topbar.tsx
        └── pages/
            ├── CommandCenter.tsx
            ├── CityMap.tsx
            ├── SimulationStudio.tsx
            ├── ResourcePlanner.tsx
            ├── Analytics.tsx
            ├── Assistant.tsx
            ├── XAI.tsx
            └── Compare.tsx
```

---

### Step 2 — Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed fastapi-0.111.0 uvicorn-0.29.0 pandas-2.2.2 scikit-learn-1.5.0 ...
```

---

### Step 3 — Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your keys
nano .env          # Linux/macOS
notepad .env       # Windows
```

Contents of `.env`:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx     # AI Assistant chat
GOOGLE_MAPS_API_KEY=AIzaSy...                      # Optional — live corridor traffic
TOMTOM_API_KEY=...                                 # Optional — live traffic incidents
```

**Get your keys:**
- Anthropic: https://console.anthropic.com/
- Google Maps (Distance Matrix API): https://console.cloud.google.com/
- TomTom Traffic API: https://developer.tomtom.com/

> ⚠️ All three are optional. If a key is missing or a live call fails, the corresponding feature automatically falls back to **ML-predicted data from the trained engine and the real historical dataset** — not random simulation (see [Live Data & Fallback Behavior](#live-data--fallback-behavior)). The AI Assistant page will show a setup message if `ANTHROPIC_API_KEY` is missing; everything else works normally either way.

---

### Step 4 — Frontend Setup

```bash
# Open a NEW terminal window
cd flowmind-ai/frontend

# Install dependencies (takes 1-2 minutes)
npm install
```

**Expected output:**
```
added 205 packages in 12s
```

---

## Running the Application

You need **two terminals running simultaneously**.

### Terminal 1 — Start Backend

```bash
cd flowmind-ai/backend
source venv/bin/activate    # Linux/macOS
# OR: venv\Scripts\activate  # Windows

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**You should see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

Test it: Open http://localhost:8000 → should show `{"message":"FlowMind AI API v2.0..."}`

The first request triggers model training (typically 4-6 seconds — four models fit on the full dataset) and is cached in memory for the life of the process — every request after that is fast (well under 100ms).

### Terminal 2 — Start Frontend

```bash
cd flowmind-ai/frontend
npm run dev
```

Open the printed local URL (typically http://localhost:5173).

### Production build

```bash
cd frontend
npm run build      # outputs to frontend/dist/
npm run preview    # serve the production build locally
```

---

## All 8 Pages Explained

### 1. Command Center (`/`)
Live KPIs pulled from `/api/analytics/summary`, system alerts computed live from current closure-rate/risk-rate leaders, and a recent-incidents feed.

### 2. City Map (`/map`)
Leaflet heatmap + marker view of real event coordinates from the dataset, filterable by cause/status.

### 3. Simulation Studio (`/simulate`)
Configure an event (cause, crowd size, time of day, zone risk, road closure, planned/unplanned) and run it through the trained ML ensemble for a full impact prediction, hourly forecast, and diversion routes.

### 4. Resource Planner (`/resources`)
AI-recommended officer/barricade/vehicle/drone deployment, calibrated from historical outcomes for the selected event cause and scaled by crowd size and risk level.

### 5. Analytics (`/analytics`)
- Horizontal bar: top causes by volume
- Line: monthly event trend
- Bar: 24-hour incident pattern with color coding
- Radar: zone risk multi-dimension
- Grouped bar: top corridors (events + closures)
- Horizontal bar: closure rate by cause
- Full table: all police stations with load bars

### 6. AI Assistant (`/assistant`)
Claude-powered chat:
- System prompt rebuilt from live dataset stats on every request
- Quick question buttons
- Full conversation history maintained in session
- Requires `ANTHROPIC_API_KEY` in `backend/.env`

### 7. Explainable AI (`/xai`)
Model transparency page:
- Model architecture cards (GBR + RF ensemble, GBC closure classifier)
- SHAP-style bar chart with a natural-language explanation generated from the actual top-ranked feature for that specific prediction
- 24-hour color-coded forecast
- 4-step model decision path walkthrough
- Confidence score display (live ensemble agreement %)

### 8. Scenario Compare (`/compare`)
Side-by-side scenario analysis:
- Two independent scenario forms (cause list fetched live from the backend)
- Run both simultaneously
- Radar chart overlay
- Head-to-head comparison table
- AI recommendation on which scenario is higher risk

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/summary` | KPI summary stats |
| GET | `/api/analytics/cause-distribution` | Event counts by cause |
| GET | `/api/analytics/monthly-trend` | Monthly event volumes |
| GET | `/api/analytics/hourly-pattern` | 24-hour incident pattern |
| GET | `/api/analytics/zone-risk` | Zone risk scores |
| GET | `/api/analytics/corridor-stats` | Corridor event counts |
| GET | `/api/analytics/police-stations` | Station load data |
| GET | `/api/analytics/closure-by-cause` | Road closure rates |
| GET | `/api/analytics/heatmap?limit=500` | Lat/lng points for heatmap |
| GET | `/api/analytics/recent-events?limit=20` | Latest incidents |
| GET | `/api/realtime/pulse` | Live system pulse |
| GET | `/api/realtime/alerts` | System alerts (computed live, see above) |
| GET | `/api/events/list` | Paginated event list |
| GET | `/api/predictions/causes` | Live list of event causes the model has data for |
| POST | `/api/predictions/predict` | Run AI prediction |
| POST | `/api/resources/recommend` | Get resource plan |
| POST | `/api/diversion/routes` | Get diversion routes |
| POST | `/api/assistant/chat` | Chat with AI assistant |
| GET | `/api/live/config-status` | Which live data providers are configured |
| GET | `/api/live/traffic-incidents` | TomTom incidents, or ML-predicted fallback |
| GET | `/api/live/google-traffic` | Google corridor congestion, or ML-predicted fallback |
| GET | `/api/live/live-events` | OSM venues, or ML-estimated fallback |
| GET | `/api/live/live-snapshot` | All three live sources combined |

**Interactive API Docs:** http://localhost:8000/docs (Swagger UI)

---

## Dataset Details

File: `backend/data/astram_events.csv`

The dataset covers real Bengaluru traffic events with planned + unplanned categories, police stations, zones, and corridors. Exact figures (total events, closure rate, average duration, top causes, peak hours) are computed live by `get_summary_stats()` / `get_cause_distribution()` / `get_hourly_pattern()` etc. and shown directly on the Command Center and Analytics pages — query those endpoints (or `GET /api/analytics/summary`) for the current numbers rather than relying on any figure written in a document, since the cleaning pipeline (dropping unparseable timestamps, clamping bad durations, normalizing duplicate cause labels) can shift them slightly as the source data is updated.

To inspect the cleaned dataset yourself:
```bash
cd backend
python -c "from ml.engine import get_summary_stats, get_cause_distribution, get_closure_by_cause; import json; print(json.dumps(get_summary_stats(), indent=2)); print(json.dumps(get_cause_distribution()[:10], indent=2)); print(json.dumps(get_closure_by_cause()[:5], indent=2))"
```

---

## AI & ML Engine

The ML engine (`backend/ml/engine.py`) trains four models in-process at first use (cached afterward):

| Model | Target | Purpose |
|---|---|---|
| `GradientBoostingRegressor` + `RandomForestRegressor` (ensemble) | Engineered congestion-severity composite (0–99) | Core congestion score |
| `GradientBoostingRegressor` | Expected delay (minutes) | Delay estimate |
| `GradientBoostingClassifier` | `requires_road_closure` (real historical binary outcome) | Closure probability |

### Feature engineering
Every input the UI exposes (event cause, time of day, zone risk, road closure, is-planned) is encoded as a feature, **and** the proxy training target is deliberately built to depend on all of them — via per-category historical risk scores for cause / zone / weekday / month, computed from real outcomes (priority rate, closure rate) for that category. This is what makes the model actually responsive to every control in the UI instead of only the columns that happened to correlate with the original narrower target (see [v2.1 — What Was Fixed](#v21--what-was-fixed) for the full story).

### Impact Prediction (`predict_impact`)
1. Build the feature row for the requested scenario and run it through the trained ensemble.
2. Apply a crowd-size multiplier (crowd isn't a training feature since the dataset has no crowd column — it's blended in post-hoc as a continuous scaling factor).
3. Apply a data-derived road-closure severity multiplier (measured from the real average congestion gap between historically closed vs. open events) so `no`/`partial`/`full` are properly distinct.
4. Classify the 0–99 score into Low/Moderate/High/Critical risk bands.
5. Compute an SHAP-style per-prediction feature attribution from the trained model's actual contribution for that input.

### Resource Recommendation
Base officer/barricade/vehicle counts are calibrated from the historical average for the selected cause, then scaled by risk level, zone risk, and crowd size.

### Diversion Routes
Bengaluru-specific route knowledge base (ORR, NICE Road, Bellary Road, Mysore Road Bypass, etc.), with each candidate route's congestion estimate driven by the predicted impact score and event cause/closure type.

---

## Live Data & Fallback Behavior

`backend/routers/livedata.py` powers the "live" feeds used by the Command Center and City Map:

| Source | When configured (`.env` key present) | When not configured / call fails |
|---|---|---|
| Traffic incidents | Real TomTom incident feed for the Bengaluru bbox | Sampled from real historical incident records matching the current hour-of-day, scored for severity by the trained closure/congestion models |
| Corridor congestion | Real Google Maps Distance Matrix traffic data | Predicted by the same trained ensemble used for event-impact prediction, seeded with each corridor's real historical risk profile and the actual current hour/weekday |
| Events / venues | Real OpenStreetMap venue data (Overpass API) | A curated list of real Bengaluru venues, with crowd estimates computed from the dataset's real hourly/weekday activity pattern (not random) |

Nothing in this path uses `random.randint()` or re-rolls noise on a timer — fallback data is always either a live external API call or a genuine model inference / historical aggregation, so it is reproducible and reflects real patterns (e.g. visible rush-hour structure) even without API keys configured. Check `GET /api/live/config-status` to see which providers are active.

---

## Deployment

### Deploy to Railway + Vercel (Free Tier)

#### Backend → Railway
1. Go to https://railway.app and sign in with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this repo, choose `/backend` as root directory
4. Add environment variables: `ANTHROPIC_API_KEY`, `GOOGLE_MAPS_API_KEY`, `TOMTOM_API_KEY` (all optional except Assistant needs the first)
5. Railway auto-detects `requirements.txt` and deploys
6. Note your Railway URL (e.g. `https://flowmind-backend.railway.app`)

#### Frontend → Vercel
1. Go to https://vercel.com and sign in with GitHub
2. Click "New Project" → import repo
3. Set root directory to `/frontend`
4. Add environment variable: `VITE_API_URL=https://flowmind-backend.railway.app`
5. Deploy

---

## Troubleshooting

### ❌ "Module not found" on backend start
```bash
# Make sure venv is activated
source venv/bin/activate
pip install -r requirements.txt
```

### ❌ "CORS error" in browser console
The backend has CORS enabled for all origins. If you're still seeing this:
- Make sure backend is running on port 8000
- Make sure Vite proxy is configured (it is, in vite.config.ts)

### ❌ Map doesn't load
Leaflet requires the CSS to be loaded. Check that `index.html` has:
```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
```
Also, the heatmap plugin requires `leaflet.heat`. The City Map page gracefully degrades if it's not available — use Markers mode instead.

### ❌ AI Assistant shows "Set ANTHROPIC_API_KEY"
1. Create `backend/.env` with your key
2. Restart the backend server

### ❌ CSV not found error
Make sure `backend/data/astram_events.csv` exists. The ML engine path is relative to `backend/main.py`. If you move files, update the `DATA_PATH` in `backend/ml/engine.py`:
```python
DATA_PATH = Path("/absolute/path/to/astram_events.csv")
```

### ❌ Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9   # Linux/macOS
netstat -ano | findstr :8000     # Windows (find PID then: taskkill /PID <pid> /F)

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### ❌ Slow initial load
The first API call loads/cleans the CSV and trains the ML ensemble (typically 4-6 seconds depending on hardware). Subsequent calls use the cached DataFrame and trained models and are near-instant.

### ❌ Live traffic/incidents endpoint is slow
If `GOOGLE_MAPS_API_KEY` / `TOMTOM_API_KEY` are set but the keys are invalid or the network can't reach those providers, each live call will wait out its timeout (8s) before falling back to ML-predicted data — this is expected and bounded; it no longer hangs for minutes (a real bug in the original codebase where some live calls could take 100+ seconds in the worst case).

---

## Project Structure

```
flowmind-ai/
│
├── README.md                    ← You are here
│
├── backend/
│   ├── main.py                  ← FastAPI app entry point
│   ├── requirements.txt         ← Python dependencies (now includes scikit-learn)
│   ├── .env.example             ← Copy to .env, add API keys
│   ├── data/
│   │   └── astram_events.csv    ← Bengaluru event dataset
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   └── engine.py            ← Core ML: training, prediction, resources, analytics, live-data helpers
│   │
│   └── routers/
│       ├── __init__.py
│       ├── analytics.py         ← /api/analytics/* endpoints
│       ├── predictions.py       ← /api/predictions/* endpoints (causes list now live)
│       ├── resources.py         ← /api/resources/* endpoints
│       ├── events.py            ← /api/events/* endpoints
│       ├── diversion.py         ← /api/diversion/* endpoints
│       ├── realtime.py          ← /api/realtime/* endpoints (alerts now fully live-computed)
│       ├── livedata.py          ← /api/live/* endpoints (ML fallback instead of random)
│       └── assistant.py         ← /api/assistant/* (Claude AI chat, dynamic system prompt)
│
└── frontend/
    ├── index.html               ← HTML entry + Leaflet CSS import
    ├── package.json             ← npm dependencies
    ├── vite.config.ts           ← Vite + proxy config
    ├── tailwind.config.js       ← Dark theme color tokens
    ├── tsconfig.json
    │
    └── src/
        ├── main.tsx             ← React entry point
        ├── App.tsx              ← Router + layout
        ├── index.css            ← Global styles + CSS variables
        │
        ├── lib/
        │   └── api.ts           ← All API calls (axios)
        │
        ├── components/
        │   ├── UI.tsx           ← Shared components (Panel, Card, Badge...)
        │   ├── Sidebar.tsx      ← Collapsible nav sidebar
        │   └── Topbar.tsx       ← Header with live clock
        │
        └── pages/
            ├── CommandCenter.tsx    ← Main dashboard
            ├── CityMap.tsx          ← Leaflet heatmap/marker map
            ├── SimulationStudio.tsx ← Event simulator + AI prediction (causes now fetched live)
            ├── ResourcePlanner.tsx  ← AI resource optimization (causes now fetched live)
            ├── Analytics.tsx        ← 10+ deep analytics charts
            ├── Assistant.tsx        ← Claude AI chat
            ├── XAI.tsx              ← Explainable AI / SHAP (causes + narrative now live)
            └── Compare.tsx          ← Side-by-side scenario comparison (causes now fetched live)
```

---

## License

Built for Gridlock 2.0 Hackathon — ASTRA GRID Challenge.
Dataset provided by HackerEarth / Bengaluru Traffic Authority.

---

*FlowMind AI — Predict. Prepare. Prevent.* 🚦
