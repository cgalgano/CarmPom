"""
pipeline/fetch_injuries.py
--------------------------
Scrapes ESPN team news for each of the 64 NCAA tournament teams and extracts
injury/availability intel, saving results to data/injuries_cache.json.

Run this periodically during the tournament (once or twice a day) to keep
the injury feed fresh:
    uv run python pipeline/fetch_injuries.py

The app reads from the cache file — no live HTTP calls at render time.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "injuries_cache.json"
BRACKET_CSV = ROOT / "data" / "bracket_2026.csv"

# Keywords that indicate injury/availability news
_INJURY_KEYWORDS = {
    "injur", "out", "doubtful", "questionable", "day-to-day", "unavailable",
    "sidelined", "missed", "miss", "ankle", "knee", "shoulder", "wrist",
    "foot", "back", "concussion", "illness", "suspend", "eligib",
    "trainers", "medical", "surgery", "torn", "sprain", "strain",
}

ESPN_TEAM_NEWS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{espn_id}/news?limit=20"
)
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams?groups=50&limit=600"
)


def _is_injury_article(headline: str, description: str = "") -> bool:
    """Return True if the text likely describes an injury or availability issue."""
    text = f"{headline} {description}".lower()
    return any(kw in text for kw in _INJURY_KEYWORDS)


def _extract_players(text: str) -> list[str]:
    """Very basic heuristic: find Capitalized-Word pairs that look like player names."""
    # Match "FirstName LastName" patterns
    return re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)


def fetch_espn_team_ids() -> dict[str, int]:
    """Return {display_name: espn_id} for all NCAAB teams from ESPN."""
    resp = httpx.get(ESPN_TEAMS_URL, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    mapping: dict[str, int] = {}
    for item in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = item.get("team", {})
        name = team.get("displayName") or team.get("name")
        eid = team.get("id")
        if name and eid:
            mapping[name] = int(eid)
    return mapping


def fetch_team_news(espn_id: int, client: httpx.Client) -> list[dict]:
    """Return raw ESPN news articles for a single team ID."""
    url = ESPN_TEAM_NEWS_URL.format(espn_id=espn_id)
    try:
        resp = client.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception:
        return []


def extract_injury_notes(articles: list[dict]) -> list[dict]:
    """Filter articles for injury content and return structured note dicts."""
    notes: list[dict] = []
    for art in articles:
        headline = art.get("headline", "")
        description = art.get("description", "") or ""
        published = art.get("published", "")

        if not _is_injury_article(headline, description):
            continue

        # Guess a severity label from keywords
        combined = f"{headline} {description}".lower()
        if "out" in combined or "sidelined" in combined or "surgery" in combined:
            status = "Out"
        elif "doubtful" in combined:
            status = "Doubtful"
        elif "questionable" in combined or "day-to-day" in combined:
            status = "Questionable"
        else:
            status = "Monitor"

        notes.append({
            "headline": headline,
            "description": description[:200] if description else "",
            "status": status,
            "published": published,
        })

    return notes[:3]  # cap at 3 most recent items per team


def run() -> dict:
    """Fetch injury news for all 64 bracket teams. Returns the results dict."""
    # Load bracket team names
    try:
        import csv

        bracket_teams: list[str] = []
        with open(BRACKET_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("team", "").strip()
                if t and t.lower() not in ("tbd", ""):
                    bracket_teams.append(t)
    except FileNotFoundError:
        print(f"ERROR: {BRACKET_CSV} not found. Run the bracket data step first.")
        sys.exit(1)

    # Load ESPN IDs from DB
    try:
        sys.path.insert(0, str(ROOT))
        from db.database import SessionLocal
        from db.models import CarmPomRating, Team

        with SessionLocal() as sess:
            rows = (
                sess.query(Team.name, Team.espn_id)
                .join(CarmPomRating, CarmPomRating.team_id == Team.id)
                .filter(CarmPomRating.season == 2026)
                .all()
            )
        db_espn: dict[str, int | None] = {name: eid for name, eid in rows}
    except Exception as exc:
        print(f"WARNING: Could not load ESPN IDs from DB ({exc}). Proceeding without.")
        db_espn = {}

    results: dict[str, list[dict]] = {}
    found_injury = 0

    with httpx.Client(timeout=12.0) as client:
        for team_name in bracket_teams:
            espn_id = db_espn.get(team_name)
            if not espn_id:
                results[team_name] = []
                continue

            articles = fetch_team_news(int(espn_id), client)
            notes = extract_injury_notes(articles)
            results[team_name] = notes
            if notes:
                found_injury += 1
                print(f"  🚑 {team_name}: {len(notes)} note(s) — {notes[0]['status']}: {notes[0]['headline'][:60]}")
            # Polite rate limiting
            time.sleep(0.15)

    print(f"\nFound injury news for {found_injury}/{len(bracket_teams)} teams.")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "teams": results,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved → {CACHE_PATH}")
    return results


def load_injuries_cache() -> dict[str, list[dict]]:
    """Load the most recently fetched injury notes from disk. Returns {} if unavailable."""
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("teams", {})
    except Exception:
        return {}


def load_injuries_meta() -> dict:
    """Return full cache metadata dict (includes fetched_at timestamp)."""
    if not CACHE_PATH.exists():
        return {}
    try:
        d = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"fetched_at": d.get("fetched_at", "")}
    except Exception:
        return {}


if __name__ == "__main__":
    run()
