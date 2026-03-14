"""
pipeline/fetch_box_scores.py
-----------------------------
Fetches team-level box score totals for every game in the games table using
the ESPN game summary JSON API, and stores results in box_scores.

Uses the ESPN summary endpoint (one JSON request per game) rather than cbbpy
HTML scraping -- faster and more reliable.

Usage:
    uv run python pipeline/fetch_box_scores.py                  # all missing games
    uv run python pipeline/fetch_box_scores.py --limit 20       # first 20 (testing)
    uv run python pipeline/fetch_box_scores.py --season 2026

Safe to re-run: games that already have box scores are skipped.
"""

import argparse
import time

import httpx
from sqlalchemy.orm import Session

from db.database import SessionLocal, engine
from db.models import Base, BoxScore, Game, Team

ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/summary"
)

REQUEST_DELAY = 0.4  # seconds between requests to avoid rate-limiting


# ---------------------------------------------------------------------------
# ESPN API parsing
# ---------------------------------------------------------------------------

def _split_made_att(display_value: str) -> tuple[int | None, int | None]:
    """Parse an ESPN 'made-attempted' string like '27-59' into (27, 59)."""
    try:
        made, att = display_value.split("-")
        return int(made), int(att)
    except (ValueError, AttributeError):
        return None, None


def _int_or_none(val: str | None) -> int | None:
    """Convert a string stat value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_team_stats(team_block: dict) -> dict:
    """
    Extract box score fields from one team block in the ESPN summary response.

    The ESPN summary endpoint returns stats as a list of {name, displayValue}
    dicts. Points are not provided directly -- we derive them as:
        pts = 2*fgm + fg3m + ftm
    (each FGM is worth 2 points; each 3PM adds one extra; FTM adds one each)
    """
    stats = {s["name"]: s.get("displayValue") for s in team_block.get("statistics", [])}

    fgm, fga = _split_made_att(stats.get("fieldGoalsMade-fieldGoalsAttempted", ""))
    fg3m, fg3a = _split_made_att(
        stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "")
    )
    ftm, fta = _split_made_att(stats.get("freeThrowsMade-freeThrowsAttempted", ""))
    oreb = _int_or_none(stats.get("offensiveRebounds"))
    dreb = _int_or_none(stats.get("defensiveRebounds"))
    tov  = _int_or_none(stats.get("totalTurnovers"))
    ast  = _int_or_none(stats.get("assists"))
    stl  = _int_or_none(stats.get("steals"))
    blk  = _int_or_none(stats.get("blocks"))
    pf   = _int_or_none(stats.get("fouls"))

    pts: int | None = None
    if fgm is not None and fg3m is not None and ftm is not None:
        pts = 2 * fgm + fg3m + ftm

    return {
        "espn_team_id": team_block["team"]["id"],
        "pts": pts, "fga": fga, "fgm": fgm,
        "fg3a": fg3a, "fg3m": fg3m,
        "fta": fta, "ftm": ftm,
        "oreb": oreb, "dreb": dreb,
        "tov": tov, "ast": ast,
        "stl": stl, "blk": blk,
        "pf": pf,
    }


def _compute_possessions(row: dict) -> float | None:
    """
    Estimate possessions: FGA - OREB + TOV + (0.475 * FTA)

    The 0.475 factor accounts for free throw trips that do not end the
    possession (and-ones, technical fouls, etc.).
    """
    if any(row.get(k) is None for k in ["fga", "oreb", "tov", "fta"]):
        return None
    return row["fga"] - row["oreb"] + row["tov"] + (0.475 * row["fta"])


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _store_box_scores(session: Session, game: Game, team_stats: list[dict]) -> int:
    """
    Persist two team box score rows for a single game.
    Looks up each team by espn_id -- skips any not found in the DB.
    Returns the number of rows inserted.
    """
    inserted = 0
    for stats in team_stats:
        team = session.query(Team).filter_by(espn_id=stats["espn_team_id"]).first()
        if not team:
            continue
        bs = BoxScore(
            game_id=game.id,
            team_id=team.id,
            pts=stats["pts"],
            fga=stats["fga"], fgm=stats["fgm"],
            fg3a=stats["fg3a"], fg3m=stats["fg3m"],
            fta=stats["fta"], ftm=stats["ftm"],
            oreb=stats["oreb"], dreb=stats["dreb"],
            tov=stats["tov"], ast=stats["ast"],
            stl=stats["stl"], blk=stats["blk"],
            pf=stats["pf"],
            possessions=_compute_possessions(stats),
        )
        session.add(bs)
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_box_scores(season: int, limit: int | None = None) -> None:
    """
    For every completed game in the DB without box score data, fetch team stats
    from ESPN summary API and store them.
    """
    with SessionLocal() as session:
        all_games = session.query(Game).filter_by(season=season).all()
        games_needed = [
            g for g in all_games
            if not session.query(BoxScore).filter_by(game_id=g.id).first()
        ]

    if limit:
        games_needed = games_needed[:limit]

    total = len(games_needed)
    print(f"Fetching box scores for {total} games (season {season})...")
    if total == 0:
        print("All games already have box scores.")
        return

    inserted_total = 0
    errors = 0

    with httpx.Client(timeout=15) as client:
        for i, game in enumerate(games_needed, 1):
            try:
                resp = client.get(ESPN_SUMMARY_URL, params={"event": game.espn_game_id})
                resp.raise_for_status()
                team_blocks = resp.json().get("boxscore", {}).get("teams", [])

                if not team_blocks:
                    errors += 1
                    continue

                team_stats = [_parse_team_stats(t) for t in team_blocks]

                with SessionLocal() as session:
                    db_game = session.query(Game).filter_by(
                        espn_game_id=game.espn_game_id
                    ).first()
                    if db_game is None:
                        errors += 1
                        continue
                    n = _store_box_scores(session, db_game, team_stats)
                    session.commit()
                inserted_total += n

                if i % 50 == 0 or i == total:
                    print(f"  [{i}/{total}]  {inserted_total} rows stored")

            except Exception as e:
                print(f"  game {game.espn_game_id}: {e}")
                errors += 1

            time.sleep(REQUEST_DELAY)

    with SessionLocal() as session:
        total_bs = session.query(BoxScore).count()

    print(f"\nDone. Inserted {inserted_total} box score rows. Errors: {errors}.")
    print(f"DB total: {total_bs} box score rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch team box scores from ESPN.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max games (for testing)"
    )
    args = parser.parse_args()
    Base.metadata.create_all(engine, checkfirst=True)
    fetch_box_scores(args.season, args.limit)
