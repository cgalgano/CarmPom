# CarmPom Roadmap
> A better KenPom — NCAA Men's Basketball analytics, ML-powered predictions, and bracket simulation.

---

## Vision

Build a full-stack analytics platform for NCAA Men's Basketball that:
- Replicates and **improves upon** KenPom's adjusted efficiency ratings
- Uses **machine learning** to predict game outcomes and bracket results
- Exposes a **web application** for exploring team ratings, head-to-head predictions, and tournament brackets
- Incorporates **AI** for automated scouting reports, game previews, and what-if scenarios

The project grows incrementally. Each phase produces something runnable and understandable before the next begins.

---

## Doing It Better Than KenPom

| KenPom Limitation | CarmPom Improvement |
|---|---|
| Flat 3.75 pt home-court adjustment for all teams | Venue-specific home-court values derived from historical data |
| No uncertainty intervals on ratings | Bayesian model outputs full probability distributions, not just point estimates |
| Team-level metrics only | Player-level impact ratings (on/off splits, lineup data) |
| Static ratings after each game day | Real-time in-game win probability using play-by-play |
| No injury adjustments | Injury-adjusted projections when key players are out |
| No public API | REST API with full data access |

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Package manager | `uv` | Fast, modern, replaces pip/poetry/venv in one tool |
| Language | Python 3.12+ | Best ecosystem for data science + web |
| Data fetching | `cbbpy`, `httpx`, ESPN unofficial API | `cbbpy` is actively maintained; ESPN API is free |
| Data processing | `pandas`, `polars` | pandas for familiarity; polars for speed later |
| Storage (dev) | SQLite via `SQLAlchemy` | Zero infrastructure — just a file |
| Storage (prod) | PostgreSQL + Redis | Concurrency, caching |
| ML models | `scikit-learn`, `xgboost`, `lightgbm`, `pymc` | Compare all; pick the winner |
| Experiment tracking | `mlflow` | Free, self-hosted, records every model run |
| Backend API | `FastAPI` | Async, auto-generates OpenAPI docs |
| Dashboard | `Streamlit` | Pure Python UI — build fast, iterate fast |
| AI/LLM | OpenAI API or local `ollama` | Scouting reports, game previews |
| Linting | `ruff` | Fast, opinionated, replaces flake8/black/isort |

### UV Cheat Sheet
```bash
uv init              # Create new project (pyproject.toml + .venv)
uv add <package>     # Install package and add to pyproject.toml
uv add --dev <pkg>   # Dev-only dependency (ruff, pytest, etc.)
uv run <script>      # Run a script inside the managed venv
uv sync              # Sync all dependencies from pyproject.toml
uv lock              # Regenerate uv.lock file
```

---

## Key Data Sources

| Source | What It Provides | Access |
|---|---|---|
| `cbbpy` (Python package) | Box scores, team stats, ESPN game data | `uv add cbbpy` |
| ESPN Unofficial API | Live scores, rosters, BPI ratings | Free, no key — see endpoints below |
| Kaggle March Mania CSVs | All D-I game results 2003–present, seeds, rankings | Free download (requires account) |
| barttorvik.com | Downloadable efficiency CSVs similar to KenPom | Free CSV export |
| sports-reference.com | Deep historical box scores | Scrape with rate limiting |

### ESPN API Endpoint Examples
```
# All games on a date
https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates=20260312

# Team list
https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams

# Game summary (replace EVENT_ID)
https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event=401654321

# Rankings
https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/rankings
```

---

## The Math: KenPom Core Formulas

### Possessions
```
Possessions = FGA - OffReb + Turnovers + (0.475 × FTA)
```
The `0.475` multiplier accounts for and-ones and technical free throws where no possession ends.

### Raw Offensive Efficiency
```
RawO = (Points Scored / Possessions) × 100
```

### Adjusted Efficiency (Iterative)
Solve iteratively until ratings converge:
```
AdjO(team) = RawO(team) + (national_avg - sum of opponents' AdjD weighted by game)
AdjD(team) = RawD(team) + (national_avg - sum of opponents' AdjO weighted by game)
```
Using the **additive model** (not multiplicative) — performs better at extremes.

### Dean Oliver's Four Factors
These are the most predictive features for ML models:
```
1. eFG%  = (FGM + 0.5 × 3PM) / FGA                  # Effective field goal %
2. TOV%  = Turnovers / (FGA + 0.475×FTA + Turnovers) # Turnover rate
3. ORB%  = OffReb / (OffReb + Opp DefReb)             # Offensive rebounding rate
4. FTR   = FTA / FGA                                  # Free throw rate
```

### Luck Rating
```
Luck = actual_win_pct - expected_win_pct_from_efficiency
```

---

## Phase 0 — Project Setup ✦ START HERE

**Goal:** A working Python project using UV that connects to a SQLite database. Nothing more.

**Completion check:** `uv run python main.py` prints "CarmPom ready. DB connected." with no errors.

### Steps
- [x] `uv init carmpom` — scaffolds `pyproject.toml`, `.venv`, `main.py`
- [x] `uv add sqlalchemy pandas httpx` — core runtime deps
- [x] `uv add --dev ruff pytest` — dev tools
- [x] Create `db/database.py` — SQLAlchemy engine + session factory
- [x] Create `db/models.py` — ORM table definitions for `teams`, `games`, `box_scores`
- [x] Update `main.py` — connect to DB, create tables, print confirmation
- [x] `uv run python main.py` — verify it works

### Notes
- `db/` and `pipeline/` registered as packages via `pyproject.toml` + hatchling
- `tool.uv.link-mode = "copy"` in pyproject.toml required for OneDrive compatibility
- ESPN JSON scoreboard API used directly (not cbbpy's HTML scraper) for speed

### Folder Structure (Phase 0)
```
carmpom/
├── main.py              ← entry point
├── pyproject.toml       ← UV-managed dependencies
├── uv.lock              ← locked dependency versions
├── .venv/               ← virtual environment (git-ignored)
├── db/
│   ├── __init__.py
│   ├── database.py      ← SQLAlchemy engine & session
│   └── models.py        ← ORM table definitions
└── data/
    └── carmpom.db       ← SQLite database file (git-ignored)
```

---

## Phase 1 — Data Pipeline

**Goal:** Fetch real NCAA Men's Basketball game data and store it in the database. Run it manually with a single command.

**Completion check:** `uv run python pipeline/fetch_games.py --season 2026` populates the `games` table with real data.

### Steps
- [ ] `uv add cbbpy` — primary data fetching library
- [ ] Create `pipeline/` folder
- [ ] `pipeline/fetch_teams.py` — fetch and store all D-I teams
- [ ] `pipeline/fetch_games.py` — fetch game results for a given season
- [ ] `pipeline/fetch_box_scores.py` — fetch box score stats per game
- [ ] Validate: query DB and print row counts
- [ ] Bootstrap historical data using Kaggle March Mania CSV import script

### Key Tables After Phase 1
```
teams       (team_id, name, conference, espn_id)
games       (game_id, date, home_team_id, away_team_id, home_score, away_score, neutral_site)
box_scores  (game_id, team_id, fga, fgm, fg3a, fg3m, fta, ftm, oreb, dreb, tov, pts, possessions)
```

---

## Phase 2 — Core Rating Engine

**Goal:** Compute CarmPom adjusted efficiency ratings from the game data. Understand every number.

**Completion check:** `uv run python ratings/engine.py` produces a printed rankings table sorted by AdjEM.

### Steps
- [ ] `ratings/engine.py` — possession calculation from box scores
- [ ] Compute raw O/D efficiency for each team per game
- [ ] Implement iterative adjustment loop (starts from raw ratings, converges in ~20 iterations)
- [ ] Calculate AdjO, AdjD, AdjEM, AdjT per team
- [ ] Calculate Luck rating
- [ ] Calculate Strength of Schedule (WIN50 method)
- [ ] Write results to `carm_pom_ratings` table
- [ ] `ratings/report.py` — print a formatted rankings table to terminal
- [ ] Compare output against current KenPom top-25 — validate sanity of numbers

### New Table After Phase 2
```
carm_pom_ratings  (team_id, season, adjo, adjd, adjem, adjt, luck, sos, rank, updated_at)
```

---

## Phase 3 — ML Model Exploration & Selection

**Goal:** Train multiple models to predict game outcomes. Compare them objectively. Promote the best one.

**Completion check:** `uv run python ml/train.py` outputs a comparison table of all models with log-loss and AUC scores. Best model saved to `models/best_model.pkl`.

### Steps
- [ ] `uv add scikit-learn xgboost lightgbm pymc mlflow`
- [ ] `ml/features.py` — build feature matrix from ratings + box scores (Four Factors, efficiency differential, tempo, recent form, seed when available)
- [ ] `ml/train.py` — train and evaluate all models with 5-fold cross-validation

### Models Compared

| Model | Why Include It |
|---|---|
| Logistic Regression | Interpretable baseline — see which features matter |
| Ridge Regression | Handles multicollinearity in efficiency metrics |
| Random Forest | Captures non-linear interactions without tuning |
| LightGBM | Top performer in most Kaggle sports competitions |
| XGBoost | Close competitor to LightGBM; often wins on smaller datasets |
| PyMC Bayesian | Outputs uncertainty intervals, not just probabilities |

- [ ] `ml/evaluate.py` — log-loss, AUC, Brier score, calibration plot for each model
- [ ] Track all experiments in `mlflow` (auto-logged)
- [ ] `ml/select.py` — load best model by validation log-loss, save to `models/`
- [ ] `ml/predict.py` — given two teams, return win probability

---

## Phase 4 — Web Application

**Goal:** A local web dashboard showing live CarmPom ratings, team profiles, and win probability predictions.

**Completion check:** `uv run streamlit run app/dashboard.py` opens a browser with the ratings table.

### Steps
- [ ] `uv add fastapi uvicorn streamlit plotly`
- [ ] `api/main.py` — FastAPI app with routes: `/ratings`, `/teams/{id}`, `/predictions`, `/games`
- [ ] `app/dashboard.py` — Streamlit pages:
  - Rankings table (sortable by AdjO, AdjD, AdjEM)
  - Team profile page (efficiency trends, schedule, Four Factors)
  - Head-to-head win probability calculator
  - Recent game results with predicted vs. actual outcomes

---

## Phase 5 — Bracket Engine

**Goal:** Simulate the NCAA Tournament bracket 10,000+ times using CarmPom win probabilities.

**Completion check:** `uv run python bracket/simulate.py` prints each team's probability of reaching each round.

### Steps
- [ ] `uv add numpy scipy`
- [ ] `bracket/seed.py` — seed 68 teams using CarmPom ratings + at-large selection heuristics
- [ ] `bracket/simulate.py` — Monte Carlo simulation engine (10k+ iterations)
- [ ] `bracket/optimize.py` — find bracket maximizing expected ESPN Challenge points
- [ ] `bracket/pool.py` — differentiation strategy for winning pools (strategic upset picks)
- [ ] Add `/bracket/simulate` and `/bracket/optimal` API endpoints
- [ ] Add bracket visualization to Streamlit dashboard

---

## Phase 6 — AI Enhancements

**Goal:** Use LLMs to generate natural-language insights from the data.

**Completion check:** `uv run python ai/preview.py --game "Duke vs Carolina"` returns a 200-word game preview.

### Steps
- [ ] `uv add openai ollama`
- [ ] `ai/preview.py` — LLM-generated game previews using CarmPom ratings as context
- [ ] `ai/scouting.py` — automated team scouting report (strengths, weaknesses, key stats)
- [ ] `ai/whatif.py` — "what if Team A plays faster?" scenario tool: adjust AdjT, re-run prediction
- [ ] `ai/anomaly.py` — detect unusual rating swings, flag for review
- [ ] `ai/postgame.py` — automated post-game insight ("Duke won despite lower efficiency — why?")
- [ ] Surface AI outputs in Streamlit dashboard

---

## Phase 7 — Scale & Polish

**Goal:** Production-ready deployment with a public URL.

### Steps
- [ ] Migrate SQLite → PostgreSQL (`uv add psycopg2-binary alembic`)
- [ ] Add Redis caching for computed ratings (`uv add redis`)
- [ ] Add `APScheduler` for nightly data refresh (`uv add apscheduler`)
- [ ] Write `Dockerfile` and `docker-compose.yml`
- [ ] Deploy to Railway or Render (free tier to start)
- [ ] Add user authentication (bracket saving, custom alerts)
- [ ] Conference summary pages
- [ ] Historical leaderboards ("CarmPom Top-25 in 2019")
- [ ] Public bracket sharing links

---

## Progress Tracker

| Phase | Status | Completion Check |
|---|---|---|
| 0 — Project Setup | ✅ Complete | `uv run python main.py` → "DB connected" |
| 1 — Data Pipeline | ✅ Complete | 1,064 games, 383 teams for 2025-26 season in DB |
| 2 — Rating Engine | ⬜ Not Started | `uv run python ratings/engine.py` prints rankings |
| 3 — ML Models | ⬜ Not Started | `uv run python ml/train.py` prints model comparison |
| 4 — Web App | ⬜ Not Started | Streamlit dashboard loads in browser |
| 5 — Bracket Engine | ⬜ Not Started | Monte Carlo simulation prints bracket probabilities |
| 6 — AI Features | ⬜ Not Started | Game preview generated from CLI |
| 7 — Scale & Deploy | ⬜ Not Started | Public URL live |
