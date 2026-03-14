"""
ratings/engine.py
-----------------
Computes CarmPom adjusted efficiency ratings for a given season.

Algorithm (KenPom-style iterative multiplicative adjustment):

  1. Load box score + game data for the season.
  2. Pair each team's row with its opponent's row per game.
  3. Compute average game possessions: poss = (poss_A + poss_B) / 2
  4. Compute raw offensive/defensive efficiency per game:
       raw_o = 100 * pts_scored / poss
       raw_d = 100 * pts_allowed / poss
  5. Initialize AdjO[t] and AdjD[t] as each team's average raw efficiencies.
  6. Iterate N_ITER times:
       AdjO[t] = mean over t's games of: raw_o_game * (nat_avg / AdjD[opp])
       AdjD[t] = mean over t's games of: raw_d_game * (nat_avg / AdjO[opp])
     The multiplicative form divides by the opponent's quality instead of
     subtracting it, which better handles extreme values.
  7. AdjEM  = AdjO - AdjD   (the headline ranking metric)
     AdjT   = adjusted tempo (possessions per game, same iterative method)
  8. Luck   = actual win% - Pythagorean win%
               Pythagorean win% = pts_for^10.25 / (pts_for^10.25 + pts_against^10.25)
  9. SOS    = average AdjEM of all opponents faced.
 10. Write final ratings to the carm_pom_ratings table.

Usage:
    uv run python ratings/engine.py --season 2026
"""

import argparse

import pandas as pd

from db.database import SessionLocal, engine
from db.models import Base, BoxScore, CarmPomRating, Game, Team

# Number of iterations for the adjustment loop.
# Convergence is typically achieved well before 20, but 20 is cheap.
N_ITER = 20

# Pythagorean exponent for college basketball (Dean Oliver's value).
PYTH_EXP = 10.25

# Teams with fewer than this many games are excluded from ratings.
# By March, every real D1 program has 28-35 games. Non-D1 exhibition opponents,
# cancelled schedules, or ESPN data gaps will have far fewer.
MIN_GAMES = 15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_matchup_df(season: int) -> pd.DataFrame:
    """
    Load all box score rows for the season and join each team's row with its
    opponent's row, producing one row per (team, game) pair with both sides'
    stats visible.

    Returns a DataFrame with columns:
        game_id, team_id, opp_id,
        pts_for, pts_against,
        poss_self, poss_opp,
        game_poss, raw_o, raw_d
    """
    with SessionLocal() as session:
        # Pull all box score rows joined to game metadata
        rows = (
            session.query(
                BoxScore.game_id,
                BoxScore.team_id,
                BoxScore.pts,
                BoxScore.possessions,
            )
            .join(Game, Game.id == BoxScore.game_id)
            .filter(Game.season == season)
            .all()
        )

    if not rows:
        raise ValueError(f"No box score data found for season {season}.")

    df = pd.DataFrame(rows, columns=["game_id", "team_id", "pts", "possessions"])

    # Self-join to pair each team with its opponent in the same game
    left  = df.rename(columns={"team_id": "team_id",  "pts": "pts_for",     "possessions": "poss_self"})
    right = df.rename(columns={"team_id": "opp_id",   "pts": "pts_against", "possessions": "poss_opp"})
    pairs = left.merge(right, on="game_id")
    pairs = pairs[pairs["team_id"] != pairs["opp_id"]].copy()

    # Use the average of both teams' possession estimates for the game total.
    # Both estimates should be close; averaging reduces noise from the formula.
    pairs["game_poss"] = (pairs["poss_self"] + pairs["poss_opp"]) / 2

    # Raw efficiency: points per 100 possessions
    pairs["raw_o"] = 100.0 * pairs["pts_for"]     / pairs["game_poss"]
    pairs["raw_d"] = 100.0 * pairs["pts_against"] / pairs["game_poss"]

    # Drop teams (and their opponents' games against them) with too few games.
    # This removes non-D1 schools, exhibition opponents, and data gaps.
    game_counts = pairs.groupby("team_id")["game_id"].count()
    qualified = set(game_counts[game_counts >= MIN_GAMES].index)
    # Both team and opponent must qualify -- keeps the iteration matrix clean.
    pairs = pairs[
        pairs["team_id"].isin(qualified) & pairs["opp_id"].isin(qualified)
    ].copy()

    return pairs


def _load_team_names(season: int) -> dict[int, str]:
    """Return {team_id: team_name} for all teams with games in this season."""
    with SessionLocal() as session:
        rows = (
            session.query(Team.id, Team.name)
            .join(BoxScore, BoxScore.team_id == Team.id)
            .join(Game, Game.id == BoxScore.game_id)
            .filter(Game.season == season)
            .distinct()
            .all()
        )
    return {r.id: r.name for r in rows}


# ---------------------------------------------------------------------------
# Rating computation
# ---------------------------------------------------------------------------

def _iterate_ratings(
    pairs: pd.DataFrame,
    n_iter: int = N_ITER,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Run the iterative multiplicative efficiency adjustment.

    Returns:
        adj_o: {team_id: AdjO}
        adj_d: {team_id: AdjD}
        adj_t: {team_id: AdjT (adjusted tempo)}
    """
    # National averages to anchor the scale at ~100 pts/100 poss.
    national_avg_o = pairs["raw_o"].mean()
    national_avg_t = pairs["game_poss"].mean()

    # Initialize with each team's unweighted average raw efficiency.
    adj_o: dict = pairs.groupby("team_id")["raw_o"].mean().to_dict()
    adj_d: dict = pairs.groupby("team_id")["raw_d"].mean().to_dict()
    adj_t: dict = pairs.groupby("team_id")["game_poss"].mean().to_dict()

    for iteration in range(n_iter):
        # Map each row's opponent to their current adjusted values.
        pairs["opp_adj_o"] = pairs["opp_id"].map(adj_o)
        pairs["opp_adj_d"] = pairs["opp_id"].map(adj_d)
        pairs["opp_adj_t"] = pairs["opp_id"].map(adj_t)

        # Multiplicative adjustment per game:
        #   If your raw_o was 110 against a defense rated 105 when average is 100,
        #   your adjusted contribution is 110 * (100 / 105) = 104.8 -- slightly
        #   deflated because you faced a slightly better-than-average defense.
        pairs["adj_o_game"] = pairs["raw_o"] * (national_avg_o / pairs["opp_adj_d"])
        pairs["adj_d_game"] = pairs["raw_d"] * (national_avg_o / pairs["opp_adj_o"])
        pairs["adj_t_game"] = pairs["game_poss"] * (national_avg_t / pairs["opp_adj_t"])

        adj_o = pairs.groupby("team_id")["adj_o_game"].mean().to_dict()
        adj_d = pairs.groupby("team_id")["adj_d_game"].mean().to_dict()
        adj_t = pairs.groupby("team_id")["adj_t_game"].mean().to_dict()

    return adj_o, adj_d, adj_t


def _compute_luck(pairs: pd.DataFrame) -> dict:
    """
    Luck = actual win% - Pythagorean win%.

    Pythagorean win%: pts_for^k / (pts_for^k + pts_against^k)
    where k = 10.25 (Dean Oliver's constant for college basketball).

    Teams with positive luck won more close games than expected -- expect
    regression. Teams with negative luck were unlucky -- expect improvement.
    """
    # Add a boolean win column before aggregating to avoid outer-reference issues.
    pairs = pairs.copy()
    pairs["won"] = pairs["pts_for"] > pairs["pts_against"]
    totals = pairs.groupby("team_id").agg(
        pts_for=("pts_for", "sum"),
        pts_against=("pts_against", "sum"),
        wins=("won", "sum"),
        games=("won", "count"),
    )
    totals["actual_wp"]  = totals["wins"] / totals["games"]
    totals["pyth_wp"]    = (
        totals["pts_for"] ** PYTH_EXP
        / (totals["pts_for"] ** PYTH_EXP + totals["pts_against"] ** PYTH_EXP)
    )
    totals["luck"] = totals["actual_wp"] - totals["pyth_wp"]
    return totals["luck"].to_dict()  # type: ignore[return-value]


def _compute_record(pairs: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Return ({team_id: wins}, {team_id: losses})."""
    wins_series   = pairs[pairs["pts_for"] > pairs["pts_against"]].groupby("team_id").size()
    losses_series = pairs[pairs["pts_for"] < pairs["pts_against"]].groupby("team_id").size()
    all_teams = pairs["team_id"].unique()
    wins   = {t: int(wins_series.get(t, 0))   for t in all_teams}
    losses = {t: int(losses_series.get(t, 0)) for t in all_teams}
    return wins, losses


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _write_ratings(
    season: int,
    adj_o: dict,
    adj_d: dict,
    adj_t: dict,
    luck: dict,
    sos: dict,
    wins: dict[int, int],
    losses: dict[int, int],
) -> None:
    """Upsert final adjusted ratings into the carm_pom_ratings table."""
    adj_em = {t: adj_o[t] - adj_d[t] for t in adj_o}

    # Rank by AdjEM descending (best team = rank 1)
    sorted_teams = sorted(adj_em, key=lambda t: adj_em[t], reverse=True)
    rank_map = {t: i + 1 for i, t in enumerate(sorted_teams)}

    with SessionLocal() as session:
        # Delete any existing ratings for this season before re-inserting.
        session.query(CarmPomRating).filter_by(season=season).delete()

        for team_id in sorted_teams:
            rating = CarmPomRating(
                team_id=team_id,
                season=season,
                adjo=round(adj_o[team_id], 4),
                adjd=round(adj_d[team_id], 4),
                adjem=round(adj_em[team_id], 4),
                adjt=round(adj_t.get(team_id, 0.0), 4),
                luck=round(luck.get(team_id, 0.0), 4),
                sos=round(sos.get(team_id, 0.0), 4),
                rank=rank_map[team_id],
                wins=wins.get(team_id, 0),
                losses=losses.get(team_id, 0),
            )
            session.add(rating)

        session.commit()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ratings(season: int) -> pd.DataFrame:
    """
    Compute and store CarmPom ratings for the given season.

    Returns a DataFrame of the final ratings sorted by rank, suitable
    for display or downstream use (ML, bracket prediction).
    """
    print(f"Loading box score data for season {season}...")
    pairs = _load_matchup_df(season)
    print(f"  {len(pairs) // 2} games, {pairs['team_id'].nunique()} teams loaded.")

    print("Running iterative rating adjustment...")
    adj_o, adj_d, adj_t = _iterate_ratings(pairs)

    # SOS: average AdjEM of opponents faced
    adj_em_map = {t: adj_o[t] - adj_d[t] for t in adj_o}
    pairs["opp_adj_em"] = pairs["opp_id"].map(adj_em_map)
    sos = pairs.groupby("team_id")["opp_adj_em"].mean().to_dict()

    luck          = _compute_luck(pairs)
    wins, losses  = _compute_record(pairs)

    print("Writing ratings to database...")
    _write_ratings(season, adj_o, adj_d, adj_t, luck, sos, wins, losses)

    # Build a return DataFrame for immediate use
    team_names = _load_team_names(season)
    records = []
    for team_id in sorted(adj_em_map, key=lambda t: adj_em_map[t], reverse=True):
        records.append({
            "rank":    len(records) + 1,
            "team":    team_names.get(team_id, str(team_id)),
            "adjem":   round(adj_em_map[team_id], 2),
            "adjo":    round(adj_o[team_id], 2),
            "adjd":    round(adj_d[team_id], 2),
            "adjt":    round(adj_t.get(team_id, 0.0), 1),
            "luck":    round(luck.get(team_id, 0.0), 3),
            "sos":     round(sos.get(team_id, 0.0), 2),
            "w":       wins.get(team_id, 0),
            "l":       losses.get(team_id, 0),
        })

    df = pd.DataFrame(records)
    print(f"Done. {len(df)} teams rated.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CarmPom rating engine.")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    Base.metadata.create_all(engine, checkfirst=True)

    ratings_df = run_ratings(args.season)

    # Quick preview of top 25
    print("\n--- CarmPom Top 25 ---")
    print(ratings_df.head(25).to_string(index=False))
