"""
pipeline/fetch_odds.py
----------------------
Fetches live NCAA tournament moneyline + spread odds from The Odds API and
caches the consensus line to data/odds_cache.json.

The free tier of The Odds API gives 500 credits/month — one call for h2h+spreads
costs 2 credits, so you can refresh ~250 times/month without a bill.

Usage:
    uv run python pipeline/fetch_odds.py

Add your free API key to .streamlit/secrets.toml:
    ODDS_API_KEY = "your_key_here"

Or set it as an environment variable: ODDS_API_KEY=your_key
"""

import json
import os
import sys
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "odds_cache.json"
SPORT = "basketball_ncaab"
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"


def _get_api_key() -> str:
    """Resolve API key from env, .env file, or .streamlit/secrets.toml."""
    # 1. Environment variable
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key

    # 2. .streamlit/secrets.toml (parse manually to avoid streamlit import)
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ODDS_API_KEY"):
                _, _, val = line.partition("=")
                key = val.strip().strip('"').strip("'")
                if key:
                    return key

    return ""


def american_to_prob(ml: float) -> float:
    """Convert American moneyline to implied decimal probability (no vig removal)."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Strip bookmaker vig from two implied probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def fetch_odds(api_key: str) -> list[dict]:
    """Call The Odds API and return raw JSON events list."""
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }
    resp = httpx.get(ODDS_URL, params=params, timeout=15.0)
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  Odds API credits remaining: {remaining}")
    resp.raise_for_status()
    return resp.json()


def build_consensus_line(outcomes: list[dict]) -> dict[str, Any]:
    """Average probabilities / spreads across bookmakers for one market."""
    # outcomes = list of {"team": str, "ml": float, "spread": float|None}
    # Group by team name
    from collections import defaultdict

    ml_by_team: dict[str, list[float]] = defaultdict(list)
    spread_by_team: dict[str, list[float]] = defaultdict(list)
    for o in outcomes:
        ml_by_team[o["team"]].append(o["ml"])
        if o.get("spread") is not None:
            spread_by_team[o["team"]].append(o["spread"])

    result: dict[str, dict] = {}
    teams = list(ml_by_team.keys())
    if len(teams) != 2:
        return {}

    for team in teams:
        avg_ml = sum(ml_by_team[team]) / len(ml_by_team[team])
        avg_spread = (
            sum(spread_by_team[team]) / len(spread_by_team[team])
            if spread_by_team[team] else None
        )
        result[team] = {"ml": round(avg_ml, 1), "spread": avg_spread}

    # Remove vig from the two moneylines
    probs_raw = [american_to_prob(result[t]["ml"]) for t in teams]
    p0, p1 = remove_vig(probs_raw[0], probs_raw[1])
    result[teams[0]]["impl_prob"] = round(p0, 4)
    result[teams[1]]["impl_prob"] = round(p1, 4)

    return result


def normalize_events(events: list[dict]) -> dict:
    """
    Parse Odds API events into a flat lookup:
        lu[team_a_name][team_b_name] = {
            "ml": float,   "spread": float|None,
            "impl_prob": float,  "kickoff": str, "book_count": int
        }
    Both directions (a→b and b→a) are stored.
    """
    lu: dict = {}

    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        commence = ev.get("commence_time", "")

        # Collect all market outcomes across bookmakers
        h2h_outcomes: list[dict] = []
        spread_outcomes: list[dict] = []

        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        h2h_outcomes.append({"team": o["name"], "ml": o["price"]})
                elif mkt["key"] == "spreads":
                    for o in mkt["outcomes"]:
                        spread_outcomes.append({"team": o["name"], "ml": o.get("price", -110), "spread": o.get("point")})

        if not h2h_outcomes:
            continue

        consensus = build_consensus_line(h2h_outcomes)
        spread_consensus = build_consensus_line(spread_outcomes) if spread_outcomes else {}

        teams = list(consensus.keys())
        if len(teams) != 2:
            continue

        book_count = len(ev.get("bookmakers", []))

        for i, ta in enumerate(teams):
            tb = teams[1 - i]
            line = {
                "ml": consensus[ta]["ml"],
                "spread": spread_consensus.get(ta, {}).get("spread"),
                "impl_prob": consensus[ta]["impl_prob"],
                "kickoff": commence,
                "book_count": book_count,
            }
            lu.setdefault(ta, {})[tb] = line

    return lu


def fuzzy_match_teams(
    odds_lu: dict, db_names: list[str]
) -> dict:
    """
    Re-key the odds lookup using DB team names instead of Odds API display names.
    Uses difflib close matches (0.55 threshold) to bridge naming differences.
    e.g. "Saint John's Red Storm" → "St. John's Red Storm"
    """
    all_odds_names = list(odds_lu.keys())
    matched_lu: dict = {}

    for odds_name, opponents in odds_lu.items():
        # Find the DB name closest to this odds name
        hits = get_close_matches(odds_name, db_names, n=1, cutoff=0.55)
        db_name = hits[0] if hits else odds_name

        matched_opponents: dict = {}
        for opp_odds_name, line in opponents.items():
            opp_hits = get_close_matches(opp_odds_name, db_names, n=1, cutoff=0.55)
            opp_db_name = opp_hits[0] if opp_hits else opp_odds_name
            matched_opponents[opp_db_name] = line

        matched_lu[db_name] = matched_opponents

    return matched_lu


def run(db_team_names: list[str] | None = None) -> dict:
    """Fetch, normalize, fuzzy-match, and cache odds. Returns the lookup dict."""
    key = _get_api_key()
    if not key:
        print(
            "ERROR: No ODDS_API_KEY found.\n"
            "  Add it to .streamlit/secrets.toml or set it as an environment variable.\n"
            "  Get a free key (500 credits/month, no credit card) at https://the-odds-api.com"
        )
        sys.exit(1)

    print(f"Fetching NCAA tournament odds from The Odds API…")
    events = fetch_odds(key)
    print(f"  {len(events)} events returned.")

    lu = normalize_events(events)
    print(f"  {len(lu)} teams parsed from odds response.")

    if db_team_names:
        lu = fuzzy_match_teams(lu, db_team_names)
        print(f"  Fuzzy-matched to DB team names.")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sport": SPORT,
        "odds": lu,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved → {CACHE_PATH}")
    return lu


def load_odds_cache() -> dict:
    """Load the most recently fetched odds from disk. Returns {} if unavailable."""
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("odds", {})
    except Exception:
        return {}


def load_odds_meta() -> dict:
    """Return metadata (fetched_at, book_count) from the cache file."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    # Optionally load DB names for better fuzzy matching
    try:
        import sys as _sys

        _sys.path.insert(0, str(ROOT))
        from db.database import SessionLocal
        from db.models import Team

        with SessionLocal() as _sess:
            _db_names = [r[0] for r in _sess.query(Team.name).all()]
    except Exception:
        _db_names = None

    run(_db_names)
