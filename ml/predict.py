"""
ml/predict.py
-------------
Generates 2026 March Madness bracket win probabilities using the trained model
and fresh CarmPom ratings.

Workflow:
  1. Load best trained model (from train.py).
  2. Load 2026 tournament seeds (from Kaggle MNCAATourneySeeds.csv after bracket
     is announced on Selection Sunday, March 15 2026).
  3. Map Kaggle team IDs to our ESPN DB IDs via ml/team_map.py.
  4. Pull 2026 CarmPom ratings (AdjEM, win%, Pythagorean win%) from our DB.
  5. Simulate the bracket via Monte Carlo:
     - For each round, match teams, compute win probabilities, sample winner.
     - Repeat N_SIMS times to get P(team reaches each round).
  6. Print results.

Usage:
    # After bracket is announced and Kaggle seeds file is updated:
    uv run python ml/predict.py --model lgb  # use best LightGBM model
    uv run python ml/predict.py --n-sims 50000
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from db.database import SessionLocal
from db.models import BoxScore, CarmPomRating, Game, Team
from ml.features import FEATURE_COLS, build_prediction_features
from ml.kaggle_loader import (
    compute_four_factors,
    load_regular_season_detailed,
    load_tourney_seeds,
)
from ml.team_map import load_mapping

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
N_SIMS_DEFAULT = 10_000

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

# Standard bracket: each seed in each region faces another seed in a fixed order.
# (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15) — first round pairings.
FIRST_ROUND_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGIONS = ["W", "X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_carmpom_2026() -> dict[int, dict]:
    """
    Load 2026 CarmPom ratings keyed by DB team_id.
    Returns {db_team_id: {adjem, win_pct, pyth_wp}}
    """
    # Compute win% and pyth_wp directly from box scores + games
    with SessionLocal() as session:
        ratings = {
            r.team_id: {"adjem": r.adjem, "wins": r.wins, "losses": r.losses}
            for r in session.query(CarmPomRating).filter_by(season=2026).all()
        }
        # Compute pyth_wp from box score totals
        pyth_exp = 10.25
        pts_data = (
            session.query(BoxScore.team_id, BoxScore.pts)
            .join(Game, Game.id == BoxScore.game_id)
            .filter(Game.season == 2026)
            .all()
        )

    # We need pts_for and pts_against per team. BoxScore has the team's pts_for.
    # To get pts_against we need the opponent's pts. Use the same approach as engine:
    # load both sides and join.
    with SessionLocal() as session:
        rows = (
            session.query(
                BoxScore.game_id,
                BoxScore.team_id,
                BoxScore.pts,
            )
            .join(Game, Game.id == BoxScore.game_id)
            .filter(Game.season == 2026)
            .all()
        )
    df = pd.DataFrame(rows, columns=["game_id", "team_id", "pts"])
    left  = df.rename(columns={"team_id": "team_id", "pts": "pts_for"})
    right = df.rename(columns={"team_id": "opp_id",  "pts": "pts_against"})
    pairs = left.merge(right, on="game_id")
    pairs = pairs[pairs["team_id"] != pairs["opp_id"]]

    totals = pairs.groupby("team_id").agg(
        pts_for_sum=("pts_for", "sum"),
        pts_against_sum=("pts_against", "sum"),
        games=("pts_for", "count"),
        wins=("pts_for", lambda x: (x > pairs.loc[x.index, "pts_against"]).sum()),
    )

    result = {}
    for team_id, row in totals.iterrows():
        pf = row["pts_for_sum"] ** pyth_exp
        pa = row["pts_against_sum"] ** pyth_exp
        tid = int(team_id)  # type: ignore[arg-type]
        result[tid] = {
            "adjem":    ratings.get(tid, {}).get("adjem", 0.0),
            "win_pct":  row["wins"] / row["games"] if row["games"] > 0 else 0.5,
            "pyth_wp":  pf / (pf + pa) if (pf + pa) > 0 else 0.5,
        }

    # Compute CarmPom rank (1 = best) from AdjEM as a KenPom rank proxy.
    # Sorted descending by adjem so rank 1 is the strongest team.
    sorted_ids = sorted(result.keys(), key=lambda tid: result[tid]["adjem"], reverse=True)
    for rank, tid in enumerate(sorted_ids, start=1):
        result[tid]["carmpom_rank"] = rank

    # Add Four Factors from Kaggle 2026 detailed results.
    # These are computed from Kaggle TeamIDs, then mapped to db_team_ids.
    kaggle_to_db, _ = load_mapping()
    try:
        detailed_2026 = load_regular_season_detailed(min_season=2026, max_season=2026)
        ff_2026 = compute_four_factors(detailed_2026)
        for _, row in ff_2026.iterrows():
            db_id = kaggle_to_db.get(int(row["TeamID"]), -1)
            if db_id in result:
                result[db_id]["eFG_pct"]  = float(row["eFG_pct"])
                result[db_id]["to_rate"]  = float(row["to_rate"])
                result[db_id]["or_pct"]   = float(row["or_pct"])
                result[db_id]["ft_rate"]  = float(row["ft_rate"])
    except FileNotFoundError:
        pass  # Detailed results not available — Four Factors will default to 0

    return result


# ---------------------------------------------------------------------------
# Kaggle submission generator
# ---------------------------------------------------------------------------

def generate_kaggle_submission(
    model: Any,
    carmpom: dict[int, dict],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate a Kaggle-format submission CSV predicting ALL possible 2026 matchups.

    Reads SampleSubmissionStage2.csv for the list of required predictions.
    Each ID has format 2026_XXXX_YYYY where XXXX < YYYY (Kaggle team IDs).
    Output: ID, Pred  (Pred = P(XXXX beats YYYY))
    """
    kaggle_data = Path(__file__).parent.parent / "data" / "kaggle"
    sub_path = kaggle_data / "SampleSubmissionStage2.csv"
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission template not found: {sub_path}")

    sub = pd.read_csv(sub_path)
    kaggle_to_db, _ = load_mapping()

    # Pre-build Four Factors lookup keyed by db_team_id
    four_factors_2026 = {
        tid: {
            "eFG_pct": stats.get("eFG_pct", 0.0),
            "to_rate": stats.get("to_rate", 0.0),
            "or_pct":  stats.get("or_pct",  0.0),
            "ft_rate": stats.get("ft_rate", 0.0),
        }
        for tid, stats in carmpom.items()
    }

    # Pull regular-season win stats from carmpom (computed inside _load_carmpom_2026)
    # We need seeds for comparison; use a flat dict mapping db_id -> seed=8 (unknown pre-bracket)
    seeds_map: dict[int, int] = {}  # filled in if seeds are loaded

    results = []
    for _, row in sub.iterrows():
        parts = str(row["ID"]).split("_")
        if len(parts) != 3:
            results.append((row["ID"], 0.5))
            continue
        season, id_a, id_b = int(parts[0]), int(parts[1]), int(parts[2])
        if season != 2026:
            results.append((row["ID"], 0.5))
            continue

        db_a = kaggle_to_db.get(id_a, -1)
        db_b = kaggle_to_db.get(id_b, -1)
        if db_a < 0 or db_b < 0:
            results.append((row["ID"], 0.5))
            continue

        matchup = pd.DataFrame([{"TeamID_A": db_a, "TeamID_B": db_b}])
        feats = build_prediction_features(
            matchup,
            seeds_2026={db_a: seeds_map.get(db_a, 8), db_b: seeds_map.get(db_b, 8)},
            adjem_2026={db_a: carmpom.get(db_a, {}).get("adjem", 0.0),
                        db_b: carmpom.get(db_b, {}).get("adjem", 0.0)},
            win_pct_2026={db_a: carmpom.get(db_a, {}).get("win_pct", 0.5),
                          db_b: carmpom.get(db_b, {}).get("win_pct", 0.5)},
            pyth_wp_2026={db_a: carmpom.get(db_a, {}).get("pyth_wp", 0.5),
                          db_b: carmpom.get(db_b, {}).get("pyth_wp", 0.5)},
            kenpom_rank_2026={db_a: carmpom.get(db_a, {}).get("carmpom_rank", 150),
                              db_b: carmpom.get(db_b, {}).get("carmpom_rank", 150)},
            four_factors_2026=four_factors_2026,
        )
        X = feats[FEATURE_COLS].fillna(0)
        pred = float(model.predict_proba(X)[0, 1])
        results.append((row["ID"], pred))

    output_df = pd.DataFrame(results, columns=["ID", "Pred"])

    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "submission_2026.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path} ({len(output_df):,} rows)")
    return output_df


# ---------------------------------------------------------------------------
# Bracket simulation
# ---------------------------------------------------------------------------

class BracketSlot:
    """Represents one slot in the bracket (one team at a moment in time)."""
    def __init__(self, kaggle_id: int, db_team_id: int, seed: int, name: str):
        self.kaggle_id   = kaggle_id
        self.db_team_id  = db_team_id
        self.seed        = seed
        self.name        = name


def win_prob(
    slot_a: BracketSlot,
    slot_b: BracketSlot,
    model: Any,
    carmpom: dict[int, dict],
    seeds: dict[int, int],
    win_pcts: dict[int, float],
    pyth_wps: dict[int, float],
) -> float:
    """
    Predict win probability for slot_a against slot_b using the trained model.
    Returns P(slot_a wins).

    kenpom_rank_diff uses our CarmPom AdjEM rank as a 2026 proxy since Kaggle
    does not yet publish KenPom ordinals for the current season.
    """
    a_id, b_id = slot_a.db_team_id, slot_b.db_team_id
    matchup = pd.DataFrame([{"TeamID_A": a_id, "TeamID_B": b_id}])
    # Use CarmPom rank as KenPom rank proxy (both measure overall team quality)
    kenpom_proxy = {
        a_id: carmpom.get(a_id, {}).get("carmpom_rank", 150),
        b_id: carmpom.get(b_id, {}).get("carmpom_rank", 150),
    }
    # Extract Four Factors stored in carmpom dict
    four_factors_2026 = {
        tid: {
            "eFG_pct": carmpom.get(tid, {}).get("eFG_pct", 0.0),
            "to_rate": carmpom.get(tid, {}).get("to_rate", 0.0),
            "or_pct":  carmpom.get(tid, {}).get("or_pct",  0.0),
            "ft_rate": carmpom.get(tid, {}).get("ft_rate", 0.0),
        }
        for tid in (a_id, b_id)
    }
    features = build_prediction_features(
        matchup,
        seeds_2026={a_id: slot_a.seed, b_id: slot_b.seed},
        adjem_2026={a_id: carmpom.get(a_id, {}).get("adjem", 0.0),
                    b_id: carmpom.get(b_id, {}).get("adjem", 0.0)},
        win_pct_2026={a_id: win_pcts.get(a_id, 0.5), b_id: win_pcts.get(b_id, 0.5)},
        pyth_wp_2026={a_id: pyth_wps.get(a_id, 0.5), b_id: pyth_wps.get(b_id, 0.5)},
        kenpom_rank_2026=kenpom_proxy,
        four_factors_2026=four_factors_2026,
    )
    X = features[FEATURE_COLS].fillna(0)
    return float(model.predict_proba(X)[0, 1])


def simulate_bracket(
    bracket: dict[str, list[BracketSlot]],  # region → [slot1, ..., slot16] ordered by seed
    model: Any,
    carmpom: dict[int, dict],
    n_sims: int = N_SIMS_DEFAULT,
) -> dict[int, dict[int, float]]:
    """
    Monte Carlo bracket simulation.

    Returns {kaggle_team_id: {round_num: P(reach round)}}
    for round_num in 1..6 (1=Round of 64, 6=Championship win).
    """
    rng = np.random.default_rng(42)
    all_teams = [slot for slots in bracket.values() for slot in slots]
    seeds_map  = {s.db_team_id: s.seed for s in all_teams}
    win_pcts   = {s.db_team_id: carmpom.get(s.db_team_id, {}).get("win_pct", 0.5) for s in all_teams}
    pyth_wps   = {s.db_team_id: carmpom.get(s.db_team_id, {}).get("pyth_wp", 0.5) for s in all_teams}

    reach_counts: dict[int, dict[int, int]] = {
        t.kaggle_id: {r: 0 for r in range(1, 7)} for t in all_teams
    }

    for _ in range(n_sims):
        # Each simulation: run 6 rounds
        # Region winners compete in Final Four
        regional_winners: list[BracketSlot] = []

        for _, slots in bracket.items():
            # Sort by seed to set up round 1 matchups: 1v16, 8v9, etc.
            region_teams = sorted(slots, key=lambda s: s.seed)

            # Round 1: pairs (1v16), (8v9), (5v12), (4v13), (6v11), (3v14), (7v10), (2v15)
            current_round = [
                (region_teams[i], region_teams[15 - i]) for i in range(8)
            ]

            for rnd in range(1, 5):  # rounds 1–4 within region
                next_round = []
                for a, b in current_round:
                    p_a = win_prob(a, b, model, carmpom, seeds_map, win_pcts, pyth_wps)
                    winner = a if rng.random() < p_a else b
                    reach_counts[winner.kaggle_id][rnd] += 1
                    next_round.append(winner)

                # Re-pair winners for next round (bracket order preserved)
                current_round = [
                    (next_round[i], next_round[i + 1])
                    for i in range(0, len(next_round), 2)
                ]

            regional_winners.append(current_round[0][0])

        # Final Four: regional winner 0 vs 1, regional winner 2 vs 3
        ff_pairs = [(regional_winners[0], regional_winners[1]),
                    (regional_winners[2], regional_winners[3])]
        finalists = []
        for a, b in ff_pairs:
            p_a = win_prob(a, b, model, carmpom, seeds_map, win_pcts, pyth_wps)
            winner = a if rng.random() < p_a else b
            reach_counts[winner.kaggle_id][5] += 1
            finalists.append(winner)

        # Championship
        a, b = finalists[0], finalists[1]
        p_a = win_prob(a, b, model, carmpom, seeds_map, win_pcts, pyth_wps)
        champion = a if rng.random() < p_a else b
        reach_counts[champion.kaggle_id][6] += 1

    # Convert counts to probabilities
    probs = {
        tid: {rnd: count / n_sims for rnd, count in rounds.items()}
        for tid, rounds in reach_counts.items()
    }
    return probs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run 2026 bracket predictions (requires Selection Sunday data)."""
    parser = argparse.ArgumentParser(description="Predict 2026 March Madness bracket.")
    parser.add_argument("--n-sims", type=int, default=N_SIMS_DEFAULT)
    parser.add_argument("--model", type=str, default="best",
                        help="Model name from data/models/ or 'best'")
    parser.add_argument("--submission", action="store_true",
                        help="Also write a Kaggle submission CSV to data/submission_2026.csv")
    args = parser.parse_args()

    print("Loading 2026 CarmPom ratings...")
    carmpom = _load_carmpom_2026()

    model_path = MODEL_DIR / f"{args.model}.pkl"
    if not model_path.exists():
        print(
            f"No trained model found at {model_path}.\n"
            "Run ml/train.py first, then save the best model:\n"
            "  from ml.train import run_training; run_training()\n"
            "  import pickle; pickle.dump(model, open('data/models/best.pkl', 'wb'))"
        )
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Loading 2026 tournament bracket (Selection Sunday seeds)...")
    seeds_df = load_tourney_seeds(min_season=2026, max_season=2026)
    if seeds_df.empty:
        print("2026 seeds not yet available — waiting for Selection Sunday (March 15).")
        print("Re-download MNCAATourneySeeds.csv from Kaggle after the bracket drops,")
        print("then run: uv run python ml/predict.py")
        return

    kaggle_to_db, _ = load_mapping()
    with SessionLocal() as session:
        teams = {t.id: t.name for t in session.query(Team).all()}

    # Build bracket structure
    bracket: dict[str, list[BracketSlot]] = {r: [] for r in REGIONS}
    for _, row in seeds_df.iterrows():
        region = row["Seed"][0]
        seed_num = int(row["Seed"][1:3])
        kaggle_id = int(row["TeamID"])
        db_id = kaggle_to_db.get(kaggle_id, -1)
        if region in bracket and db_id > 0:
            bracket[region].append(BracketSlot(
                kaggle_id=kaggle_id,
                db_team_id=db_id,
                seed=seed_num,
                name=teams.get(db_id, str(kaggle_id)),
            ))

    print(f"\nSimulating {args.n_sims:,} brackets...")
    probs = simulate_bracket(bracket, model, carmpom, n_sims=args.n_sims)

    # Display championship probabilities sorted
    all_slots = [slot for slots in bracket.values() for slot in slots]
    rows = []
    for slot in all_slots:
        p = probs.get(slot.kaggle_id, {})
        rows.append({
            "Seed":         slot.seed,
            "Team":         slot.name,
            "Sweet 16":     f"{p.get(3, 0):.1%}",
            "Elite 8":      f"{p.get(4, 0):.1%}",
            "Final Four":   f"{p.get(5, 0):.1%}",
            "Champion":     f"{p.get(6, 0):.1%}",
            "_champ_raw":   p.get(6, 0),
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("_champ_raw", ascending=False)
        .drop(columns=["_champ_raw"])
    )
    print("\n" + "="*70)
    print("  2026 CarmPom Bracket Probabilities")
    print("="*70)
    print(results.to_string(index=False))

    if args.submission:
        print("\nGenerating Kaggle submission CSV...")
        generate_kaggle_submission(model, carmpom)


if __name__ == "__main__":
    main()
