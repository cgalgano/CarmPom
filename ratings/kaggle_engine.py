"""
ratings/kaggle_engine.py
------------------------
Computes CarmPom adjusted efficiency ratings for historical seasons using
Kaggle's MRegularSeasonDetailedResults.csv — no ESPN API or DB required.

This uses the identical iterative multiplicative algorithm as ratings/engine.py,
but reads directly from the Kaggle CSV. The output is cached to a CSV so
ml/train.py can include opponent-adjusted AdjEM as a training feature for
every season, not just 2026.

Usage:
    uv run python ratings/kaggle_engine.py                  # 2003-2025
    uv run python ratings/kaggle_engine.py --min 2012 --max 2025
"""

import argparse
from pathlib import Path

import pandas as pd

KAGGLE_DIR   = Path(__file__).parent.parent / "data" / "kaggle"
OUTPUT_PATH  = KAGGLE_DIR / "historical_adjem.csv"

N_ITER    = 20     # iterations before convergence — same as engine.py
PYTH_EXP  = 10.25  # Dean Oliver's Pythagorean constant for college basketball
MIN_GAMES = 15     # exclude teams with too few games (non-D1, data gaps)


# ---------------------------------------------------------------------------
# Build per-(team, game) matchup DataFrame from Kaggle detailed results
# ---------------------------------------------------------------------------

def _build_matchup_df(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one season of Kaggle detailed results into a matchup DataFrame
    with the same schema used by ratings/engine.py.

    Kaggle detailed results have one row per game with winner/loser stats
    side-by-side. We expand to two rows per game (one per team perspective).

    Possessions estimate (Dean Oliver formula):
        poss = FGA - OR + TO + 0.44 * FTA

    Returns columns:
        game_idx, team_id, opp_id,
        pts_for, pts_against,
        game_poss, raw_o, raw_d
    """
    df = season_df.reset_index(drop=True)
    df["game_idx"] = df.index  # stable row ID for self-join

    # Possession estimates for each side
    df["poss_W"] = df["WFGA"] - df["WOR"] + df["WTO"] + 0.44 * df["WFTA"]
    df["poss_L"] = df["LFGA"] - df["LOR"] + df["LTO"] + 0.44 * df["LFTA"]
    # Average both estimates — reduces noise from the imperfect formula
    df["game_poss"] = (df["poss_W"] + df["poss_L"]) / 2

    # Winner's perspective
    w = pd.DataFrame({
        "game_idx":    df["game_idx"],
        "team_id":     df["WTeamID"],
        "opp_id":      df["LTeamID"],
        "pts_for":     df["WScore"].astype(float),
        "pts_against": df["LScore"].astype(float),
        "game_poss":   df["game_poss"],
    })

    # Loser's perspective
    l = pd.DataFrame({
        "game_idx":    df["game_idx"],
        "team_id":     df["LTeamID"],
        "opp_id":      df["WTeamID"],
        "pts_for":     df["LScore"].astype(float),
        "pts_against": df["WScore"].astype(float),
        "game_poss":   df["game_poss"],
    })

    pairs = pd.concat([w, l], ignore_index=True)

    # Drop teams below the minimum game threshold (non-D1, exhibition opponents)
    game_counts = pairs.groupby("team_id")["game_idx"].count()
    qualified   = set(game_counts[game_counts >= MIN_GAMES].index)
    pairs = pairs[
        pairs["team_id"].isin(qualified) & pairs["opp_id"].isin(qualified)
    ].copy()

    # Raw efficiency: points per 100 possessions
    pairs["raw_o"] = 100.0 * pairs["pts_for"]     / pairs["game_poss"]
    pairs["raw_d"] = 100.0 * pairs["pts_against"] / pairs["game_poss"]

    return pairs


# ---------------------------------------------------------------------------
# Iterative rating adjustment — identical logic to engine.py
# ---------------------------------------------------------------------------

def _iterate_ratings(
    pairs: pd.DataFrame,
    n_iter: int = N_ITER,
) -> tuple[dict, dict, dict]:
    """
    Run the iterative multiplicative efficiency adjustment.

    Returns: adj_o, adj_d, adj_t  (all {team_id: float})
    """
    national_avg_o = pairs["raw_o"].mean()
    national_avg_t = pairs["game_poss"].mean()

    adj_o: dict = pairs.groupby("team_id")["raw_o"].mean().to_dict()
    adj_d: dict = pairs.groupby("team_id")["raw_d"].mean().to_dict()
    adj_t: dict = pairs.groupby("team_id")["game_poss"].mean().to_dict()

    for _ in range(n_iter):
        pairs["opp_adj_o"] = pairs["opp_id"].map(adj_o)
        pairs["opp_adj_d"] = pairs["opp_id"].map(adj_d)
        pairs["opp_adj_t"] = pairs["opp_id"].map(adj_t)

        pairs["adj_o_game"] = pairs["raw_o"]     * (national_avg_o / pairs["opp_adj_d"])
        pairs["adj_d_game"] = pairs["raw_d"]     * (national_avg_o / pairs["opp_adj_o"])
        pairs["adj_t_game"] = pairs["game_poss"] * (national_avg_t / pairs["opp_adj_t"])

        adj_o = pairs.groupby("team_id")["adj_o_game"].mean().to_dict()
        adj_d = pairs.groupby("team_id")["adj_d_game"].mean().to_dict()
        adj_t = pairs.groupby("team_id")["adj_t_game"].mean().to_dict()

    return adj_o, adj_d, adj_t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_season_ratings(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AdjEM/AdjO/AdjD/AdjT for all teams in one season.

    Args:
        season_df: rows from MRegularSeasonDetailedResults for a single season

    Returns:
        DataFrame with columns: TeamID, AdjEM, AdjO, AdjD, AdjT
    """
    pairs = _build_matchup_df(season_df)
    adj_o, adj_d, adj_t = _iterate_ratings(pairs)

    records = []
    for team_id in adj_o:
        records.append({
            "TeamID": int(team_id),
            "AdjO":   round(adj_o[team_id], 4),
            "AdjD":   round(adj_d[team_id], 4),
            "AdjEM":  round(adj_o[team_id] - adj_d[team_id], 4),
            "AdjT":   round(adj_t.get(team_id, 0.0), 4),
        })

    return pd.DataFrame(records)


def build_all_seasons(
    min_season: int = 2003,
    max_season: int = 2025,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Compute ratings for all seasons in [min_season, max_season] and cache
    results to a CSV at output_path.

    Returns the full multi-season DataFrame.
    """
    detailed_path = KAGGLE_DIR / "MRegularSeasonDetailedResults.csv"
    if not detailed_path.exists():
        raise FileNotFoundError(
            f"Required file missing: {detailed_path}\n"
            "Download from https://www.kaggle.com/competitions/march-machine-learning-mania-2026"
        )

    print(f"Loading {detailed_path.name}...")
    all_detailed = pd.read_csv(detailed_path)
    all_detailed = all_detailed[
        (all_detailed["Season"] >= min_season) & (all_detailed["Season"] <= max_season)
    ]

    seasons = sorted(all_detailed["Season"].unique())
    print(f"Computing ratings for {len(seasons)} seasons ({min_season}–{max_season})...")

    all_results: list[pd.DataFrame] = []
    for season in seasons:
        season_df   = all_detailed[all_detailed["Season"] == season].copy()
        ratings_df  = compute_season_ratings(season_df)
        ratings_df.insert(0, "Season", season)
        all_results.append(ratings_df)
        n_teams = len(ratings_df)
        top_em  = ratings_df["AdjEM"].max()
        print(f"  {season}: {n_teams} teams  |  top AdjEM = {top_em:.2f}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined):,} rows → {output_path}")
    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute historical CarmPom ratings from Kaggle detailed results."
    )
    parser.add_argument("--min", type=int, default=2003, dest="min_season")
    parser.add_argument("--max", type=int, default=2025, dest="max_season")
    args = parser.parse_args()

    build_all_seasons(min_season=args.min_season, max_season=args.max_season)
