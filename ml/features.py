"""
ml/features.py
--------------
Builds the feature matrix used to train and run bracket prediction models.

Each row represents one tournament game framed from the perspective of
"Team A" (better seed) vs "Team B" (worse seed):

  label = 1 if Team A (better seed) wins (favorite wins)
  label = 0 if Team B (worse seed) wins (upset)

Feature philosophy:
  - All features are expressed as (A - B) differences so positive values
    consistently mean "Team A is better on this metric."
  - This symmetric framing avoids the model learning any spurious
    home/away or bracket-region bias.

Features included:
  seed_diff          — seed_a - seed_b (negative = A is a better seed)
  kenpom_rank_diff   — KenPom ordinal rank diff (lower rank = better team)
  win_pct_diff       — regular season win% diff
  pyth_wp_diff       — Pythagorean win% diff (removes luck from actual W-L)
  adjem_diff         — CarmPom AdjEM diff (opponent-adjusted efficiency margin)
  efg_diff           — effective FG% diff (3-pt adjusted shooting)
  to_rate_diff       — turnover rate diff
  or_pct_diff        — offensive rebound % diff
  ft_rate_diff       — FTA/FGA diff (drawing fouls)
"""

import pandas as pd

from ml.kaggle_loader import (
    compute_four_factors,
    compute_season_stats,
    load_historical_adjem,
    load_massey_kenpom,
    load_regular_season,
    load_regular_season_detailed,
    load_tourney_results,
    load_tourney_seeds,
)

# Columns used as model inputs
FEATURE_COLS = [
    "seed_diff",
    "kenpom_rank_diff",
    "win_pct_diff",
    "pyth_wp_diff",
    "adjem_diff",     # CarmPom opponent-adjusted efficiency margin
    # Dean Oliver's Four Factors (from MRegularSeasonDetailedResults.csv)
    "efg_diff",       # effective FG% diff — 3-pt adjusted shooting efficiency
    "to_rate_diff",   # turnover rate diff — lower is better
    "or_pct_diff",    # offensive rebound % diff
    "ft_rate_diff",   # FTA/FGA diff — drawing fouls
]

# Alias kept for backwards compatibility with predict.py
FEATURE_COLS_2026 = FEATURE_COLS


def _merge_team_stats(
    games: pd.DataFrame,
    seeds: pd.DataFrame,
    massey: pd.DataFrame,
    season_stats: pd.DataFrame,
    four_factors: pd.DataFrame | None = None,
    adjem_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join all per-team per-season features onto each tournament game.

    The output has one row per game with columns for both the winner and loser,
    before we reframe as team_a / team_b in build_features().
    """
    # Attach seeds to winners and losers
    seed_map = seeds.set_index(["Season", "TeamID"])["SeedNum"]

    games = games.copy()

    # Attach seeds — merge on (Season, WTeamID/LTeamID)
    seed_df = seeds[["Season", "TeamID", "SeedNum"]]
    games = games.merge(
        seed_df.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}),
        on=["Season", "WTeamID"], how="left",
    )
    games = games.merge(
        seed_df.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}),
        on=["Season", "LTeamID"], how="left",
    )

    # Attach KenPom ranks
    kp_df = massey[["Season", "TeamID", "KenPomRank"]]
    games = games.merge(
        kp_df.rename(columns={"TeamID": "WTeamID", "KenPomRank": "WKenPom"}),
        on=["Season", "WTeamID"], how="left",
    )
    games = games.merge(
        kp_df.rename(columns={"TeamID": "LTeamID", "KenPomRank": "LKenPom"}),
        on=["Season", "LTeamID"], how="left",
    )

    # Attach regular season stats
    stat_cols = ["WinPct", "PythWinPct"]
    stat_df = season_stats[["Season", "TeamID"] + stat_cols]
    for col in stat_cols:
        games = games.merge(
            stat_df[["Season", "TeamID", col]].rename(
                columns={"TeamID": "WTeamID", col: f"W{col}"}
            ),
            on=["Season", "WTeamID"], how="left",
        )
        games = games.merge(
            stat_df[["Season", "TeamID", col]].rename(
                columns={"TeamID": "LTeamID", col: f"L{col}"}
            ),
            on=["Season", "LTeamID"], how="left",
        )

    # Attach Four Factors if provided
    if four_factors is not None:
        ff_cols = ["eFG_pct", "to_rate", "or_pct", "ft_rate"]
        ff_df = four_factors[["Season", "TeamID"] + ff_cols]
        for col in ff_cols:
            games = games.merge(
                ff_df[["Season", "TeamID", col]].rename(
                    columns={"TeamID": "WTeamID", col: f"W{col}"}
                ),
                on=["Season", "WTeamID"], how="left",
            )
            games = games.merge(
                ff_df[["Season", "TeamID", col]].rename(
                    columns={"TeamID": "LTeamID", col: f"L{col}"}
                ),
                on=["Season", "LTeamID"], how="left",
            )

    # Attach CarmPom AdjEM if provided
    if adjem_df is not None:
        adjem_slim = adjem_df[["Season", "TeamID", "AdjEM"]]
        games = games.merge(
            adjem_slim.rename(columns={"TeamID": "WTeamID", "AdjEM": "WAdjEM"}),
            on=["Season", "WTeamID"], how="left",
        )
        games = games.merge(
            adjem_slim.rename(columns={"TeamID": "LTeamID", "AdjEM": "LAdjEM"}),
            on=["Season", "LTeamID"], how="left",
        )

    return games


def build_training_features(min_season: int = 2003, max_season: int = 2025) -> pd.DataFrame:
    """
    Build the full historical training feature matrix from Kaggle data.

    Rows: one per tournament game for seasons [min_season, max_season]
    Columns: FEATURE_COLS + 'label' + 'Season' (kept for cross-val splitting)

    Team A is always the lower seed number (better seed). Label = 1 if A wins.

    Training extends back to 2003 because:
      - MRegularSeasonDetailedResults.csv covers 2003+
      - MMasseyOrdinals (KenPom) covers 2003+
      - historical_adjem.csv covers 2003+ (generated by ratings/kaggle_engine.py)
    """
    games        = load_tourney_results(min_season, max_season)
    seeds        = load_tourney_seeds(min_season, max_season)
    massey       = load_massey_kenpom(min_season, max_season)
    reg_season   = load_regular_season(min_season, max_season)
    season_stats = compute_season_stats(reg_season)
    detailed     = load_regular_season_detailed(min_season, max_season)
    four_factors = compute_four_factors(detailed)
    adjem_df     = load_historical_adjem(min_season, max_season)

    games = _merge_team_stats(games, seeds, massey, season_stats, four_factors, adjem_df)

    # Fill any missing KenPom ranks (pre-2012 seasons or teams not in Massey)
    # with an AdjEM-derived rank — same concept, just computed from our own ratings.
    # This unlocks training all the way back to 2003 without losing kenpom_rank_diff.
    adjem_rank = (
        adjem_df.copy()
        .assign(KenPomRank_est=lambda d:
            d.groupby("Season")["AdjEM"].rank(ascending=False).astype(int)
        )[["Season", "TeamID", "KenPomRank_est"]]
    )
    games = games.merge(
        adjem_rank.rename(columns={"TeamID": "WTeamID", "KenPomRank_est": "WKenPom_est"}),
        on=["Season", "WTeamID"], how="left",
    )
    games = games.merge(
        adjem_rank.rename(columns={"TeamID": "LTeamID", "KenPomRank_est": "LKenPom_est"}),
        on=["Season", "LTeamID"], how="left",
    )
    games["WKenPom"] = games["WKenPom"].fillna(games["WKenPom_est"])
    games["LKenPom"] = games["LKenPom"].fillna(games["LKenPom_est"])

    # Drop rows with missing seeds (play-in games sometimes have issues)
    games = games.dropna(subset=["WSeed", "LSeed"])

    # Assign team_a = better seed (lower seed number), team_b = worse seed
    # When seeds are equal (rare but possible in finals), use TeamID order.
    a_is_winner = (games["WSeed"] < games["LSeed"]) | (
        (games["WSeed"] == games["LSeed"]) & (games["WTeamID"] < games["LTeamID"])
    )

    def _diff(w_col: str, l_col: str) -> pd.Series:
        """Compute (team_a - team_b) signed difference."""
        return (
            games[w_col].where(a_is_winner, games[l_col])
            - games[l_col].where(a_is_winner, games[w_col])
        )

    features = pd.DataFrame({
        "Season":          games["Season"].values,
        "seed_diff":       _diff("WSeed", "LSeed"),
        "kenpom_rank_diff":_diff("WKenPom", "LKenPom"),
        "win_pct_diff":    _diff("WWinPct", "LWinPct"),
        "pyth_wp_diff":    _diff("WPythWinPct", "LPythWinPct"),
        "adjem_diff":      _diff("WAdjEM", "LAdjEM"),
        "efg_diff":        _diff("WeFG_pct", "LeFG_pct"),
        "to_rate_diff":    _diff("Wto_rate", "Lto_rate"),
        "or_pct_diff":     _diff("Wor_pct", "Lor_pct"),
        "ft_rate_diff":    _diff("Wft_rate", "Lft_rate"),
        "label":           a_is_winner.astype(int),
    })

    return features.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def build_prediction_features(
    matchups: pd.DataFrame,
    seeds_2026: dict[int, int],
    adjem_2026: dict[int, float],
    win_pct_2026: dict[int, float],
    pyth_wp_2026: dict[int, float],
    kenpom_rank_2026: dict[int, int] | None = None,
    four_factors_2026: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """
    Build features for 2026 tournament matchups to generate win probabilities.

    Args:
        matchups: DataFrame with columns TeamID_A, TeamID_B (Kaggle IDs).
                  Team A should be the better seed.
        seeds_2026:         {team_id: seed_number}
        adjem_2026:         {team_id: CarmPom AdjEM}
        win_pct_2026:       {team_id: regular season win%}
        pyth_wp_2026:       {team_id: Pythagorean win%}
        kenpom_rank_2026:   {team_id: KenPom ordinal rank} (optional)
        four_factors_2026:  {team_id: {efg, to_rate, or_pct, ft_rate}} (optional)

    Returns a DataFrame with FEATURE_COLS_2026 or FEATURE_COLS if no KenPom.
    """
    rows = []
    for _, row in matchups.iterrows():
        a, b = int(row["TeamID_A"]), int(row["TeamID_B"])
        ff_a = four_factors_2026.get(a, {}) if four_factors_2026 else {}
        ff_b = four_factors_2026.get(b, {}) if four_factors_2026 else {}
        feat = {
            "TeamID_A":       a,
            "TeamID_B":       b,
            "seed_diff":      seeds_2026.get(a, 8) - seeds_2026.get(b, 8),
            "win_pct_diff":   win_pct_2026.get(a, 0.5) - win_pct_2026.get(b, 0.5),
            "pyth_wp_diff":   pyth_wp_2026.get(a, 0.5) - pyth_wp_2026.get(b, 0.5),
            "kenpom_rank_diff": (
                (kenpom_rank_2026.get(a, 150) - kenpom_rank_2026.get(b, 150))
                if kenpom_rank_2026 else None
            ),
            "adjem_diff":     adjem_2026.get(a, 0.0) - adjem_2026.get(b, 0.0),
            # Four Factors — fall back to 0 diff if not available
            "efg_diff":       ff_a.get("eFG_pct", 0.0) - ff_b.get("eFG_pct", 0.0),
            "to_rate_diff":   ff_a.get("to_rate",  0.0) - ff_b.get("to_rate",  0.0),
            "or_pct_diff":    ff_a.get("or_pct",   0.0) - ff_b.get("or_pct",   0.0),
            "ft_rate_diff":   ff_a.get("ft_rate",  0.0) - ff_b.get("ft_rate",  0.0),
        }
        rows.append(feat)

    df = pd.DataFrame(rows)
    return df
