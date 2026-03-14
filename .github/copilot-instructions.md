# CarmPom — Copilot Instructions

These instructions apply to every Copilot chat message in this workspace.
Edit any section to change how Copilot behaves for this project.

---

## Project Overview

CarmPom is a KenPom-style NCAA Men's Basketball analytics platform built in Python.
It computes adjusted efficiency ratings, predicts game outcomes using ML, and simulates tournament brackets.
See ROADMAP.md for the full plan.

---

## My Preferences

<!-- Edit this section to tell Copilot your preferences. Examples are pre-filled based on our setup. -->

### Packages & Environment
- Always use `uv` for package management — never suggest `pip install` or `poetry`
- Installing a package: `uv add <package>` (runtime) or `uv add --dev <package>` (dev only)
- Running scripts: `uv run python <script>`

### Python Style
- Python 3.12+
- Use type hints on all function signatures
- Use `pathlib.Path` instead of `os.path`
- Prefer f-strings over `.format()` or `%`
- Use `ruff` for linting — do not suggest `flake8`, `black`, or `isort`

### Data & Storage
- Database is SQLite (via SQLAlchemy) in Phase 0–6; PostgreSQL in Phase 7
- ORM is SQLAlchemy 2.0 — use the modern `Mapped` / `mapped_column` syntax
- Use `pandas` for data processing unless a specific reason to use `polars`

### Machine Learning
- Do not default to XGBoost — the goal is to compare multiple models and select the best
- Models to evaluate: Logistic Regression, Ridge, Random Forest, LightGBM, XGBoost, PyMC Bayesian
- Track all experiments with `mlflow`
- Evaluate models using: log-loss (primary), AUC, Brier score

### Code Comments
- Add docstrings to every module (file-level) and every function
- Comment non-obvious math/logic inline — this is an educational project, clarity matters
- Do not add comments that just describe what the code obviously does (e.g., `# increment i`)

### Testing
- Use `pytest` for all tests
- Test files go in `tests/` mirroring the source structure (e.g., `tests/db/test_models.py`)

---

## Phase We're Currently On

<!-- Update this as the project progresses so Copilot knows the current context -->

**Current phase: Phase 2 — Core Rating Engine**

Active tasks:
- Computing possession estimates from box score data in `box_scores` table
- Building the iterative adjusted efficiency loop (AdjO, AdjD, AdjEM, AdjT)
- Writing results to the `carm_pom_ratings` table

---

## What NOT to Do

- Do not suggest creating new markdown files to summarize changes unless I explicitly ask
- Do not suggest migrating to PostgreSQL until Phase 7
- Do not add a scheduler (APScheduler/Airflow) until Phase 7 — keep it manual CLI scripts for now
- Do not build a frontend (Streamlit/React) until Phase 4
- Do not use `pip`, `conda`, or `poetry`
- Do not write extra comments that just restate what the code does without adding insight
- Do not put extra emojis and formatting in comments that is not particularly helpful — keep it clear and concise
