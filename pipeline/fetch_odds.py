"""
pipeline/fetch_odds.py
----------------------
Fetches live NCAA tournament moneyline + spread odds from ESPN's free public
scoreboard API and caches them to data/odds_cache.json.

No API key or account required.

Usage:
    uv run python pipeline/fetch_odds.py

Run once a day (or before each session) to refresh lines.
"""

import csv
import json
from datetime import date, datetime, timedelta, timezone
from difflib import get_close_matches
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "odds_cache.json"
BRACKET_CSV = ROOT / "data" / "bracket_2026.csv"

_ESPN_BASE = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# NCAA tournament group ID in ESPN's system
_NCAAT_GROUP = "50"

# Tournament window: First Four through Championship
_TOURNEY_START = date(2026, 3, 17)
_TOURNEY_END   = date(2026, 4, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ml(ml_str: str | None) -> float | None:
    """Convert ESPN American moneyline string to float. 'EVEN' → +100."""
    if not ml_str:
        return None
    s = str(ml_str).strip()
    if s.upper() in ("EVEN", "PK"):
        return 100.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _impl_prob(ml: float) -> float:
    """Raw (with-vig) implied probability from American moneyline."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100.0 / (ml + 100)


def _remove_vig(p_a: float, p_b: float) -> tuple[float, float]:
    """Normalise two implied probabilities to sum to 1.0."""
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_day(day: date) -> list[dict]:
    """Fetch ESPN scoreboard for one calendar day. Returns raw events list."""
    params = {"groups": _NCAAT_GROUP, "dates": day.strftime("%Y%m%d")}
    resp = httpx.get(_ESPN_BASE, params=params, headers=_HEADERS, timeout=15.0)
    resp.raise_for_status()
    return resp.json().get("events", [])


def fetch_all_tournament_events() -> list[dict]:
    """Scrape ESPN scoreboard for every day in the tournament window."""
    all_events: list[dict] = []
    day = _TOURNEY_START
    while day <= _TOURNEY_END:
        try:
            events = fetch_day(day)
            if events:
                print(f"  {day}  →  {len(events)} events")
            all_events.extend(events)
        except Exception as exc:
            print(f"  {day}  →  error: {exc}")
        day += timedelta(days=1)
    return all_events


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_event(ev: dict) -> list[tuple[str, str, dict]]:
    """
    Parse one ESPN event into a list of (team_a, team_b, line_dict) tuples.
    Returns both directions (a→b and b→a) so the lookup works either way.

    The line dict contains: ml, spread, impl_prob, kickoff, book_count.
    """
    comp = ev.get("competitions", [{}])[0]
    odds_list: list[dict] = comp.get("odds", [])
    if not odds_list:
        return []

    # Use the highest-priority bookmaker (odds_list is sorted by provider.priority)
    odds = odds_list[0]

    # Identify away/home team names
    away_name = ""
    home_name = ""
    for c in comp.get("competitors", []):
        if c.get("homeAway") == "away":
            away_name = c.get("team", {}).get("displayName", "")
        elif c.get("homeAway") == "home":
            home_name = c.get("team", {}).get("displayName", "")

    if not away_name or not home_name or "TBD" in (away_name, home_name):
        return []

    # Who is the favourite?  awayTeamOdds.favorite: bool
    away_odds_obj = odds.get("awayTeamOdds", {})
    home_odds_obj = odds.get("homeTeamOdds", {})
    away_is_fav = bool(away_odds_obj.get("favorite"))

    # Spread — ESPN's `spread` field is already signed from the HOME team's
    # perspective (negative = home is favourite, positive = home is underdog).
    home_spread: float | None = odds.get("spread")   # e.g. -5.5 for Louisville at home
    away_spread: float | None = (-home_spread) if home_spread is not None else None

    # Moneyline strings → floats
    ml_obj = odds.get("moneyline", {})
    away_ml = _parse_ml(ml_obj.get("away", {}).get("close", {}).get("odds"))
    home_ml = _parse_ml(ml_obj.get("home", {}).get("close", {}).get("odds"))

    # Fallback: synthesise ML from spread if moneyline parsing failed
    if away_ml is None or home_ml is None:
        # rough approximation: -110 for both sides of a short spread
        away_ml = away_ml or (-110.0 if away_is_fav else 100.0)
        home_ml = home_ml or (-110.0 if not away_is_fav else 100.0)

    # No-vig implied probabilities
    p_away, p_home = _remove_vig(_impl_prob(away_ml), _impl_prob(home_ml))

    kickoff = comp.get("date", ev.get("date", ""))
    book_count = len(odds_list)

    away_line = {
        "ml": round(away_ml, 1),
        "spread": away_spread,
        "impl_prob": round(p_away, 4),
        "kickoff": kickoff,
        "book_count": book_count,
    }
    home_line = {
        "ml": round(home_ml, 1),
        "spread": home_spread,
        "impl_prob": round(p_home, 4),
        "kickoff": kickoff,
        "book_count": book_count,
    }

    return [
        (away_name, home_name, away_line),
        (home_name, away_name, home_line),
    ]


def build_lookup(events: list[dict]) -> dict:
    """Build nested odds lookup from raw ESPN events.

    Returns lu[team_a][team_b] = {ml, spread, impl_prob, kickoff, book_count}.
    """
    lu: dict = {}
    for ev in events:
        for ta, tb, line in _parse_event(ev):
            lu.setdefault(ta, {})[tb] = line
    return lu


# ---------------------------------------------------------------------------
# Fuzzy matching to bracket/DB names
# ---------------------------------------------------------------------------

def _load_bracket_names() -> list[str]:
    """Return team names from the bracket CSV for fuzzy re-keying."""
    names: list[str] = []
    try:
        with open(BRACKET_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("team", "").strip()
                if t and t.lower() not in ("tbd", ""):
                    names.append(t)
    except FileNotFoundError:
        pass
    return names


def fuzzy_match_teams(lu: dict, bracket_names: list[str]) -> dict:
    """Re-key the lookup to match bracket team name spellings.

    ESPN uses full school names ("Duke Blue Devils") that often already match
    the bracket CSV.  Fuzzy matching ≥0.55 handles edge cases like
    'Saint John's' vs 'St. John's'.
    """
    if not bracket_names:
        return lu

    matched: dict = {}
    for espn_name, opponents in lu.items():
        hits = get_close_matches(espn_name, bracket_names, n=1, cutoff=0.55)
        db_name = hits[0] if hits else espn_name

        matched_opp: dict = {}
        for opp_espn, line in opponents.items():
            opp_hits = get_close_matches(opp_espn, bracket_names, n=1, cutoff=0.55)
            opp_db = opp_hits[0] if opp_hits else opp_espn
            matched_opp[opp_db] = line

        matched.setdefault(db_name, {}).update(matched_opp)

    return matched


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(db_team_names: list[str] | None = None) -> dict:
    """Fetch all tournament odds from ESPN and save cache. Returns lookup dict."""
    print("Fetching NCAA tournament odds from ESPN scoreboard API…")
    events = fetch_all_tournament_events()
    print(f"  {len(events)} total events fetched across tournament window")

    lu = build_lookup(events)
    print(f"  {len(lu)} teams with betting lines")

    bracket_names = db_team_names or _load_bracket_names()
    if bracket_names:
        lu = fuzzy_match_teams(lu, bracket_names)
        print(f"  Fuzzy-matched to {len(bracket_names)} bracket team names")

    # Report coverage
    covered = sum(1 for t in (bracket_names or []) if t in lu)
    if bracket_names:
        print(f"  Bracket coverage: {covered}/{len(bracket_names)} teams have lines")

    for ta, opp in sorted(lu.items(), key=lambda x: x[0]):
        for tb, line in opp.items():
            spread = f"{line['spread']:+.1f}" if line.get("spread") is not None else "N/A"
            print(f"    {ta}  vs  {tb}  |  ML {line['ml']:+.0f}  spread {spread}  impl {line['impl_prob']:.1%}")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "espn-scoreboard",
        "sport": "basketball_ncaab",
        "odds": lu,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved → {CACHE_PATH}")
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
    """Return metadata dict (includes fetched_at, source)."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    # Optionally load bracket/DB names for better fuzzy matching
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
