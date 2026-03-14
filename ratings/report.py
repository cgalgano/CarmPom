"""
ratings/report.py
-----------------
Prints a formatted CarmPom rankings table to the terminal.

Usage:
    uv run python ratings/report.py --season 2026           # all teams
    uv run python ratings/report.py --season 2026 --top 25  # top N only
    uv run python ratings/report.py --season 2026 --team "Duke Blue Devils"
"""

import argparse

import pandas as pd

from db.database import SessionLocal
from db.models import CarmPomRating, Team


def load_ratings(season: int) -> pd.DataFrame:
    """
    Pull ratings from DB and join to team names.
    Returns a DataFrame sorted by rank.
    """
    with SessionLocal() as session:
        rows = (
            session.query(
                CarmPomRating.rank,
                Team.name.label("team"),
                Team.conference,
                CarmPomRating.wins,
                CarmPomRating.losses,
                CarmPomRating.adjem,
                CarmPomRating.adjo,
                CarmPomRating.adjd,
                CarmPomRating.adjt,
                CarmPomRating.luck,
                CarmPomRating.sos,
            )
            .join(Team, Team.id == CarmPomRating.team_id)
            .filter(CarmPomRating.season == season)
            .order_by(CarmPomRating.rank)
            .all()
        )

    if not rows:
        print(f"No ratings found for season {season}. Run ratings/engine.py first.")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "rank", "team", "conf", "w", "l",
        "adjem", "adjo", "adjd", "adjt", "luck", "sos",
    ])
    return df


def print_table(df: pd.DataFrame, top: int | None = None, team_filter: str | None = None) -> None:
    """
    Print a formatted rankings table.
    Optionally filter to top N teams or a specific team name substring.
    """
    if df.empty:
        return

    if team_filter:
        df = df[df["team"].str.contains(team_filter, case=False)]
        if df.empty:
            print(f"No team matching '{team_filter}' found.")
            return

    if top:
        df = df.head(top)

    # Format record as "W-L"
    df = df.copy()
    df["record"] = df["w"].astype(str) + "-" + df["l"].astype(str)

    display = df[[
        "rank", "team", "conf", "record",
        "adjem", "adjo", "adjd", "adjt", "luck", "sos",
    ]]

    header = (
        f"{'#':>4}  {'Team':<30}  {'Conf':<10}  {'W-L':<7}  "
        f"{'AdjEM':>7}  {'AdjO':>7}  {'AdjD':>7}  {'AdjT':>6}  "
        f"{'Luck':>7}  {'SOS':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for _, row in display.iterrows():
        print(
            f"{int(row['rank']):>4}  {row['team']:<30}  {row['conf'] or '':.<10}  "
            f"{row['record']:<7}  "
            f"{row['adjem']:>7.2f}  {row['adjo']:>7.2f}  {row['adjd']:>7.2f}  "
            f"{row['adjt']:>6.1f}  {row['luck']:>7.3f}  {row['sos']:>7.2f}"
        )

    print(sep)
    print(f"{len(display)} teams shown.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print CarmPom rankings table.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--top", type=int, default=None, help="Show only top N teams")
    parser.add_argument("--team", type=str, default=None, help="Filter by team name substring")
    args = parser.parse_args()

    df = load_ratings(args.season)
    print(f"\nCarmPom Rankings — {args.season - 1}-{str(args.season)[2:]} Season\n")
    print_table(df, top=args.top, team_filter=args.team)
