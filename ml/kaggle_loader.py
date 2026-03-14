"""
ml/kaggle_loader.py
--------------------
Loads NCAA tournament data from Kaggle March Madness competition CSV files.

Download from: https://www.kaggle.com/competitions/march-machine-learning-mania-2026
Place all CSV files in:  data/kaggle/

Key files and their contents:
  MTeams.csv                          — TeamID, TeamName (Kaggle IDs, not ESPN IDs)
  MNCAATourneyCompactResults.csv      — Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc
  MNCAATourneySeeds.csv               — Season, Seed (e.g. 'W01'), TeamID
  MMasseyOrdinals.csv                 — Season, RankingDayNum, SystemName, TeamID, OrdinalRank
  MRegularSeasonCompactResults.csv    — same format as tourney results but regular season
  MRegularSeasonDetailedResults.csv   — compact results + full team box score stats per game

Note on team IDs: Kaggle uses its own integer TeamIDs that differ from ESPN IDs.
  When predicting 2026, we must map ESPN IDs to Kaggle IDs (via team name matching).
"""

from pathlib import Path

import pandas as pd

KAGGLE_DIR = Path(__file__).parent.parent / "data" / "kaggle"

# KenPom's system code in the Massey Ordinals file
KENPOM_SYSTEM = "POM"

# Last day before the tournament typically starts (used to get pre-tourney rankings).
# Day 133 = ~mid-March (Selection Sunday area). Adjust if needed.
MASSEY_CUTOFF_DAY = 133


def _require(filename: str) -> Path:
    """Return path to a Kaggle data file, raising a clear error if missing."""
    path = KAGGLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle data file not found: {path}\n"
            f"Download from https://www.kaggle.com/competitions/march-machine-learning-mania-2026"
            f"\nand place in {KAGGLE_DIR}"
        )
    return path


def load_teams() -> pd.DataFrame:
    """
    Load team metadata.
    Returns: TeamID, TeamName
    """
    return pd.read_csv(_require("MTeams.csv"))


def load_tourney_results(min_season: int = 2012, max_season: int = 2025) -> pd.DataFrame:
    """
    Load historical tournament game results.
    Each row is one game: WTeamID beat LTeamID by (WScore - LScore).

    min_season: first season to include (KenPom data reliable from ~2012)
    """
    df = pd.read_csv(_require("MNCAATourneyCompactResults.csv"))
    return df[(df["Season"] >= min_season) & (df["Season"] <= max_season)].copy()


def load_tourney_seeds(min_season: int = 2012, max_season: int = 2025) -> pd.DataFrame:
    """
    Load tournament seeds.
    Seed format: 'W01' = Region W, seed 1. 'Y16a' = Region Y, seed 16 (play-in).

    Adds numeric 'SeedNum' column (1-16) parsed from the raw seed string.
    """
    df = pd.read_csv(_require("MNCAATourneySeeds.csv"))
    df = df[(df["Season"] >= min_season) & (df["Season"] <= max_season)].copy()
    # Extract numeric seed (strip region letter and optional a/b play-in suffix)
    df["SeedNum"] = df["Seed"].str[1:3].astype(int)
    return df


def load_massey_kenpom(min_season: int = 2012, max_season: int = 2025) -> pd.DataFrame:
    """
    Load KenPom pre-tournament ordinal rankings from the Massey Ordinals file.

    Filters to:
      - SystemName == 'POM' (KenPom)
      - RankingDayNum <= MASSEY_CUTOFF_DAY (latest pre-tournament ranking)
      - The most recent ranking day per team per season

    Returns columns: Season, TeamID, KenPomRank (lower = better)
    """
    df = pd.read_csv(_require("MMasseyOrdinals.csv"))
    df = df[
        (df["SystemName"] == KENPOM_SYSTEM)
        & (df["Season"] >= min_season)
        & (df["Season"] <= max_season)
        & (df["RankingDayNum"] <= MASSEY_CUTOFF_DAY)
    ].copy()

    # Keep only the most recent pre-tourney ranking per team per season
    df = df.sort_values("RankingDayNum")
    df = df.groupby(["Season", "TeamID"]).last().reset_index()
    df = df.rename(columns={"OrdinalRank": "KenPomRank"})[["Season", "TeamID", "KenPomRank"]]
    return df


def load_regular_season(min_season: int = 2012, max_season: int = 2025) -> pd.DataFrame:
    """
    Load regular season compact results for computing win% and Pythagorean win%.

    Returns one row per game per team (winner and loser both get a row):
      Season, TeamID, PtsFor, PtsAgainst, Win (1/0)
    """
    df = pd.read_csv(_require("MRegularSeasonCompactResults.csv"))
    df = df[(df["Season"] >= min_season) & (df["Season"] <= max_season)].copy()

    # Build winner perspective
    w = df[["Season", "WTeamID", "WScore", "LScore"]].copy()
    w.columns = ["Season", "TeamID", "PtsFor", "PtsAgainst"]
    w["Win"] = 1

    # Build loser perspective
    l = df[["Season", "LTeamID", "LScore", "WScore"]].copy()
    l.columns = ["Season", "TeamID", "PtsFor", "PtsAgainst"]
    l["Win"] = 0

    return pd.concat([w, l], ignore_index=True)


def compute_season_stats(reg_season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate regular season game-level data into per-team per-season summaries.

    Returns columns:
      Season, TeamID, Games, Wins, WinPct,
      PtsForTotal, PtsAgainstTotal, PythWinPct
    """
    PYTH_EXP = 10.25

    agg = reg_season_df.groupby(["Season", "TeamID"]).agg(
        Games=("Win", "count"),
        Wins=("Win", "sum"),
        PtsForTotal=("PtsFor", "sum"),
        PtsAgainstTotal=("PtsAgainst", "sum"),
    ).reset_index()

    agg["WinPct"] = agg["Wins"] / agg["Games"]

    # Pythagorean win%: better predictor of future performance than actual win%
    pf = agg["PtsForTotal"] ** PYTH_EXP
    pa = agg["PtsAgainstTotal"] ** PYTH_EXP
    agg["PythWinPct"] = pf / (pf + pa)

    return agg


def load_regular_season_detailed(
    min_season: int = 2012, max_season: int = 2026
) -> pd.DataFrame:
    """
    Load detailed regular season results (box score stats per game, per team).

    Available since the 2003 season. Each row is one game with full shot/rebound/
    turnover/foul stats for both the winning and losing team.
    """
    df = pd.read_csv(_require("MRegularSeasonDetailedResults.csv"))
    return df[(df["Season"] >= min_season) & (df["Season"] <= max_season)].copy()


def compute_four_factors(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Dean Oliver's Four Factors per team per season from detailed results.

    The Four Factors explain ~90% of a team's success:
      eFG_pct  = (FGM + 0.5 * FGM3) / FGA         — shooting efficiency (3-pt adjusted)
      to_rate  = TO / (FGA + 0.44 * FTA + TO)       — turnover rate per possession
      or_pct   = OR / (OR + Opp DR)                 — offensive rebounding dominance
      ft_rate  = FTA / FGA                           — rate of getting to the free throw line

    Returns columns: Season, TeamID, eFG_pct, to_rate, or_pct, ft_rate
    """
    # Winning team perspective
    w = detailed_df[[
        "Season", "WTeamID",
        "WFGM", "WFGA", "WFGM3", "WOR", "WTO", "WFTA",
        "LDR",  # opponent's defensive rebounds (for or_pct denominator)
    ]].rename(columns={
        "WTeamID": "TeamID",
        "WFGM": "FGM", "WFGA": "FGA", "WFGM3": "FGM3",
        "WOR": "OR", "WTO": "TO", "WFTA": "FTA",
        "LDR": "OppDR",
    })

    # Losing team perspective
    l = detailed_df[[
        "Season", "LTeamID",
        "LFGM", "LFGA", "LFGM3", "LOR", "LTO", "LFTA",
        "WDR",  # opponent's defensive rebounds
    ]].rename(columns={
        "LTeamID": "TeamID",
        "LFGM": "FGM", "LFGA": "FGA", "LFGM3": "FGM3",
        "LOR": "OR", "LTO": "TO", "LFTA": "FTA",
        "WDR": "OppDR",
    })

    combined = pd.concat([w, l], ignore_index=True)
    agg = combined.groupby(["Season", "TeamID"]).sum(numeric_only=True).reset_index()

    agg["eFG_pct"] = (agg["FGM"] + 0.5 * agg["FGM3"]) / agg["FGA"]
    poss_est = agg["FGA"] + 0.44 * agg["FTA"] + agg["TO"]
    agg["to_rate"] = agg["TO"] / poss_est
    agg["or_pct"] = agg["OR"] / (agg["OR"] + agg["OppDR"])
    agg["ft_rate"] = agg["FTA"] / agg["FGA"]

    return agg[["Season", "TeamID", "eFG_pct", "to_rate", "or_pct", "ft_rate"]]


def load_historical_adjem(
    min_season: int = 2003, max_season: int = 2025
) -> pd.DataFrame:
    """
    Load pre-computed CarmPom AdjEM ratings from historical_adjem.csv.

    Generated by: uv run python ratings/kaggle_engine.py
    Returns columns: Season, TeamID, AdjEM, AdjO, AdjD, AdjT
    """
    path = KAGGLE_DIR / "historical_adjem.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Historical AdjEM file not found: {path}\n"
            "Run: uv run python ratings/kaggle_engine.py"
        )
    df = pd.read_csv(path)
    return df[(df["Season"] >= min_season) & (df["Season"] <= max_season)].copy()
