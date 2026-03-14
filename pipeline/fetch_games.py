"""
pipeline/fetch_games.py
-----------------------
Fetches game metadata day-by-day from the ESPN scoreboard JSON API and stores
results in the `games` and `teams` tables.

Uses ESPN's public scoreboard endpoint directly (one request per day) instead of
cbbpy's get_game_info, which scrapes HTML page-by-page and is too slow.

Usage:
    uv run python pipeline/fetch_games.py                         # Nov 3 2025 to today
    uv run python pipeline/fetch_games.py --start 2026-03-10 --end 2026-03-12
    uv run python pipeline/fetch_games.py --start 2026-03-12 --end 2026-03-12   # today only

Safe to re-run: already-stored games are skipped automatically.
"""

import argparse
import time
from datetime import date, timedelta

import httpx
from sqlalchemy.orm import Session

from db.database import SessionLocal, engine
from db.models import Base, Game, Team

# ESPN public scoreboard endpoint -- no API key required.
# `dates` param format: YYYYMMDD. `limit` covers all games on a single day.
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)

# Days between requests -- be polite to ESPN's servers.
REQUEST_DELAY = 0.5


# ---------------------------------------------------------------------------
# ESPN API parsing
# ---------------------------------------------------------------------------

def _fetch_day(date_str_espn: str, client: httpx.Client) -> list[dict]:
    """
    Fetch the scoreboard for one day from ESPN.
    Returns a list of raw event dicts from the JSON response.
    date_str_espn format: YYYYMMDD (e.g. "20260312")
    """
    # groups=50 = NCAA Division I Men's Basketball -- required to get ALL D1 games.
    # Without it, ESPN returns only 4-8 featured games per day instead of the full slate.
    resp = client.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": date_str_espn, "limit": 300, "groups": "50"},
    )
    resp.raise_for_status()
    return resp.json().get("events", [])


def _parse_event(event: dict) -> dict | None:
    """
    Extract the fields we care about from a single ESPN scoreboard event dict.
    Returns None if the game is not yet complete.

    ESPN status.type.id values: "1"=pre, "2"=in-progress, "3"=final.
    """
    try:
        comp = event["competitions"][0]
        status_id = event["status"]["type"]["id"]
        if status_id != "3":
            return None  # Not final yet

        competitors = {c["homeAway"]: c for c in comp["competitors"]}
        home = competitors.get("home")
        away = competitors.get("away")
        if not home or not away:
            return None

        def _conf(competitor: dict) -> str | None:
            # conference is sometimes present under team.conferenceId's label;
            # ESPN scoreboard doesn't always include the name, so treat as optional.
            conf = competitor["team"].get("conferenceId")
            return str(conf) if conf else None

        return {
            "espn_game_id": event["id"],
            "game_date": event["date"][:10],  # "2026-03-12T00:00Z"  "2026-03-12"
            "home_id": home["team"]["id"],
            "home_name": home["team"]["displayName"],
            "home_conf": _conf(home),
            "home_score": int(home["score"]) if home.get("score") else None,
            "away_id": away["team"]["id"],
            "away_name": away["team"]["displayName"],
            "away_conf": _conf(away),
            "away_score": int(away["score"]) if away.get("score") else None,
            "neutral_site": comp.get("neutralSite", False),
            # notes[0].headline is the tournament name when present
            "tournament": (comp.get("notes") or [{}])[0].get("headline"),
        }
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _upsert_team(session: Session, espn_id: str, name: str, conference: str | None = None) -> int:
    """Insert a team keyed on espn_id, or return the existing row's id.

    Updates the conference field if it was previously NULL and we now have it.
    """
    existing = session.query(Team).filter_by(espn_id=espn_id).first()
    if existing:
        if conference and not existing.conference:
            existing.conference = conference
        return existing.id
    team = Team(name=name, espn_id=espn_id, conference=conference)
    session.add(team)
    session.flush()
    return team.id


def _store_event(parsed: dict, season: int, session: Session) -> bool:
    """
    Persist a parsed game event to the DB.
    Returns True if inserted, False if already stored.
    """
    if session.query(Game).filter_by(espn_game_id=parsed["espn_game_id"]).first():
        return False

    home_id = _upsert_team(session, parsed["home_id"], parsed["home_name"], parsed.get("home_conf"))
    away_id = _upsert_team(session, parsed["away_id"], parsed["away_name"], parsed.get("away_conf"))

    game = Game(
        espn_game_id=parsed["espn_game_id"],
        season=season,
        game_date=date.fromisoformat(parsed["game_date"]),
        home_team_id=home_id,
        away_team_id=away_id,
        home_score=parsed["home_score"],
        away_score=parsed["away_score"],
        neutral_site=parsed["neutral_site"],
        tournament=parsed["tournament"],
    )
    session.add(game)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_games(start: date, end: date) -> None:
    """
    Walk day-by-day from start to end, fetching the ESPN scoreboard for each day.
    Each day is one HTTP request; games are committed to the DB immediately.
    """
    season = end.year
    current = start
    total_inserted = 0

    print(f"Fetching games  {start} -> {end}  (season {season})")

    with httpx.Client(timeout=15) as client:
        while current <= end:
            date_espn = current.strftime("%Y%m%d")

            try:
                events = _fetch_day(date_espn, client)
            except Exception as e:
                print(f"  {current}: {e} (skipping)")
                current += timedelta(days=1)
                time.sleep(REQUEST_DELAY)
                continue

            day_inserted = 0
            with SessionLocal() as session:
                for event in events:
                    parsed = _parse_event(event)
                    if parsed and _store_event(parsed, season, session):
                        day_inserted += 1
                session.commit()

            if day_inserted > 0:
                print(f"  {current}: +{day_inserted} games")
                total_inserted += day_inserted

            current += timedelta(days=1)
            time.sleep(REQUEST_DELAY)

    with SessionLocal() as session:
        db_games = session.query(Game).filter_by(season=season).count()
        db_teams = session.query(Team).count()

    print(f"\nDone. Inserted {total_inserted} new games.")
    print(f"DB total: {db_games} games for season {season} | {db_teams} teams.")


if __name__ == "__main__":
    SEASON_START = date(2025, 11, 3)  # 2025-26 tip-off

    parser = argparse.ArgumentParser(description="Fetch NCAA game metadata from ESPN.")
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=SEASON_START,
        help=f"Start date YYYY-MM-DD. Default: {SEASON_START}",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="End date inclusive YYYY-MM-DD. Default: today",
    )
    args = parser.parse_args()

    Base.metadata.create_all(engine, checkfirst=True)
    fetch_games(args.start, args.end)
