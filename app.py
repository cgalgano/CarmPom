"""
app.py
------
CarmPom Streamlit web app.

Run with:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import or_

from db.database import SessionLocal
from db.models import BoxScore, CarmPomRating, Game, Team

sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CarmPom",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 CarmPom")
st.markdown("#### NCAA Basketball rankings and tournament predictions — like KenPom, but open.")

st.divider()

hero_left, hero_right = st.columns([3, 2], gap="large")

with hero_left:
    st.markdown("""
    Win-loss records don't tell the full story. A 20-10 team that beat bad opponents by 30
    looks identical on paper to one that went 20-10 grinding out close games against a tough
    schedule. They are not the same team.

    CarmPom ranks every D1 team by **Adjusted Efficiency Margin (AdjEM)** — the difference
    between Offensive and Defensive Adjusted Efficiency, measured in points per 100 possessions,
    after adjusting for who you played. It's the most honest single-number summary of how good
    a team actually is.

    On top of the ratings, an ML model trained on 20+ years of tournament data converts
    efficiency numbers into win probabilities — so you can see not just who's ranked higher,
    but how much it matters when they play.
    """)

with hero_right:
    st.markdown("#### CarmPom vs KenPom at a glance")
    st.markdown("""
    | | KenPom | CarmPom |
    |---|---|---|
    | **Ratings method** | Opponent-adjusted efficiency | Same |
    | **Data source** | Premium proprietary feeds | Public ESPN box scores |
    | **Updated** | Daily (paid) | On demand |
    | **Tournament predictions** | Ratings only | ML model + bracket sim |
    | **Cost** | $9.99/year | Free |
    """)
    st.caption("Ratings will closely track KenPom — differences come from data source and methodology details.")

st.divider()

# --- Feature importance section ---
st.markdown("### What actually wins in March?")
st.markdown(
    "We looked at every NCAA Tournament game from 2003–2025 and asked: which stats actually "
    "predicted who won? The bars below show what the model learned to lean on. "
    "Longer bar = more predictive."
)

# Fan-friendly labels and plain-English explanations for each feature
FEATURE_EXPLAINERS = {
    "adjem_diff": (
        "Overall efficiency edge  *(AdjEM)*",
        "The single biggest predictor — and the number CarmPom is built around. "
        "It measures how much better one team is at scoring *and* defending, adjusted for the quality of "
        "opponents all season. Think of it as the most honest one-number summary of a team.",
    ),
    "seed_diff": (
        "Seed gap",
        "The selection committee's verdict. A 1 vs 16 gap is enormous; a 4 vs 5 is basically a coin flip. "
        "Seeds matter, but mostly because the committee is reading the same efficiency numbers everyone else is.",
    ),
    "kenpom_rank_diff": (
        "KenPom ranking gap",
        "KenPom's version of the same adjusted efficiency idea. The model uses both CarmPom and KenPom "
        "ratings and figures out which one is more reliable for each situation.",
    ),
    "efg_diff": (
        "Shooting edge  *(effective FG%)*",
        "A 3-pointer is worth more than a 2, so this weights them accordingly. "
        "Teams that shoot the ball well all season don't suddenly forget how in March.",
    ),
    "pyth_wp_diff": (
        "\"Deserved\" record gap  *(Pythagorean W%)*",
        "Calculated purely from points scored and allowed — not actual wins. "
        "A team that went 25-5 by winning 10 games by 1 point is probably not as good as their record says. "
        "This catches that.",
    ),
    "win_pct_diff": (
        "Record gap",
        "Plain old win-loss percentage. Less telling than efficiency numbers, but the model still "
        "finds a small edge here — winning ugly is still winning.",
    ),
    "or_pct_diff": (
        "Second-chance opportunities  *(offensive rebounding)*",
        "When your miss turns into another shot, that's a free possession. "
        "In a one-and-done tournament game, a few extra chances can be the difference.",
    ),
    "ft_rate_diff": (
        "Getting to the free-throw line",
        "Free throws are the most efficient way to score — no defense allowed. "
        "Teams that draw fouls all season keep doing it against tournament defenses too.",
    ),
    "to_rate_diff": (
        "Ball security  *(turnover rate)*",
        "Every turnover is a gift to the other team. In tight tournament games, one sloppy stretch "
        "against a defense you've never seen before can end your season.",
    ),
}

@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load the trained model and return feature importances as a sorted DataFrame."""
    import pickle
    from ml.features import FEATURE_COLS
    with open(ROOT / "data" / "models" / "best.pkl", "rb") as f:
        model = pickle.load(f)
    imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
    imp["pct"] = imp["importance"] / imp["importance"].sum() * 100
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)

feat_df = load_feature_importance()

for _, row in feat_df.iterrows():
    feature = row["feature"]
    pct = row["pct"]
    label, explanation = FEATURE_EXPLAINERS.get(feature, (feature, ""))
    is_top = pct == feat_df["pct"].max()

    col_label, col_bar = st.columns([3, 2])
    with col_label:
        if is_top:
            st.markdown(f"**🏆 {label}**")
        else:
            st.markdown(f"**{label}**")
        st.caption(explanation)
    with col_bar:
        st.progress(int(pct), text=f"{pct:.1f}% of model weight")

st.divider()

# ---------------------------------------------------------------------------
# Data loader (cached so it doesn't re-query on every interaction)
# ---------------------------------------------------------------------------

@st.cache_data
def _build_espn_id_map() -> dict[int, str]:
    """Return {db_team_id: espn_stats_url} for all teams that have an espn_id."""
    with SessionLocal() as session:
        rows = session.query(Team.id, Team.espn_id).filter(Team.espn_id.isnot(None)).all()
    return {
        tid: f"https://www.espn.com/mens-college-basketball/team/stats/_/id/{eid}"
        for tid, eid in rows
    }


# Build once at startup (not user-session cached — it's static data)
_ESPN_ID_MAP: dict[int, str] = _build_espn_id_map()


@st.cache_data(ttl=3600)
def load_rankings(season: int) -> pd.DataFrame:
    """Pull ratings + team info for a given season from the database."""
    with SessionLocal() as session:
        rows = (
            session.query(
                CarmPomRating.team_id,
                CarmPomRating.rank,
                Team.name,
                Team.conference,
                Team.espn_id,
                CarmPomRating.adjem,
                CarmPomRating.adjo,
                CarmPomRating.adjd,
                CarmPomRating.adjt,
                CarmPomRating.luck,
                CarmPomRating.sos,
                CarmPomRating.wins,
                CarmPomRating.losses,
            )
            .join(Team, Team.id == CarmPomRating.team_id)
            .filter(CarmPomRating.season == season)
            .order_by(CarmPomRating.rank)
            .all()
        )

    df = pd.DataFrame(
        rows,
        columns=["team_id", "Rank", "Team", "Conf", "espn_id", "AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS", "W", "L"],
    )
    # Build ESPN team stats page URL — works without the slug, just the numeric ID
    df["ESPN"] = df["team_id"].map(
        lambda tid: _ESPN_ID_MAP.get(int(tid), None)
    )
    # Logo URL from ESPN CDN
    df["logo_url"] = df["espn_id"].apply(
        lambda eid: f"https://a.espncdn.com/i/teamlogos/ncaa/500/{eid}.png" if pd.notna(eid) else None
    )
    df["Record"] = df["W"].astype(str) + "-" + df["L"].astype(str)
    df["Conf"] = df["Conf"].str.removesuffix(" Conference")

    # Compute national rank for each metric (rank 1 = best for that stat).
    # AdjD is ascending because lower points allowed is better.
    rank_cfg = {"AdjEM": False, "AdjO": False, "AdjD": True, "AdjT": False, "Luck": False, "SOS": False}
    for col, asc in rank_cfg.items():
        df[f"{col}_nr"] = df[col].rank(ascending=asc, method="min").astype(int)

    return df


@st.cache_data(ttl=3600)
def load_per_game_stats(season: int) -> pd.DataFrame:
    """Aggregate per-game counting stats from box_scores for all teams in a season."""
    with SessionLocal() as session:
        rows = (
            session.query(
                BoxScore.team_id,
                BoxScore.game_id,
                BoxScore.pts,
                BoxScore.fgm,
                BoxScore.fga,
                BoxScore.fg3m,
                BoxScore.fg3a,
                BoxScore.ftm,
                BoxScore.fta,
                BoxScore.oreb,
                BoxScore.dreb,
                BoxScore.ast,
                BoxScore.tov,
            )
            .join(Game, Game.id == BoxScore.game_id)
            .filter(Game.season == season)
            .all()
        )

    bs = pd.DataFrame(rows, columns=[
        "team_id", "game_id", "pts", "fgm", "fga",
        "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "tov",
    ])

    # Self-join on game_id to get the opponent's pts for each game.
    opp = bs[["game_id", "team_id", "pts"]].rename(columns={"team_id": "opp_id", "pts": "opp_pts"})
    bs = bs.merge(opp, on="game_id", how="left")
    bs = bs[bs["team_id"] != bs["opp_id"]]  # drop the self-row from the join

    # Sum counting stats per team then divide by games played.
    agg = bs.groupby("team_id").agg(
        games=("pts", "count"),
        pts_tot=("pts", "sum"),
        opp_pts_tot=("opp_pts", "sum"),
        fgm=("fgm", "sum"),
        fga=("fga", "sum"),
        fg3m=("fg3m", "sum"),
        fg3a=("fg3a", "sum"),
        ftm=("ftm", "sum"),
        fta=("fta", "sum"),
        oreb=("oreb", "sum"),
        dreb=("dreb", "sum"),
        ast=("ast", "sum"),
        tov=("tov", "sum"),
    ).reset_index()

    g = agg["games"]
    s = pd.DataFrame({"team_id": agg["team_id"]})
    s["PPG"]    = (agg["pts_tot"] / g).round(1)
    s["OppPPG"] = (agg["opp_pts_tot"] / g).round(1)
    s["RebPG"]  = ((agg["oreb"] + agg["dreb"]) / g).round(1)
    s["AstPG"]  = (agg["ast"] / g).round(1)
    s["OrebPG"] = (agg["oreb"] / g).round(1)
    s["TOPG"]   = (agg["tov"] / g).round(1)
    s["FG%"]    = (agg["fgm"] / agg["fga"] * 100).round(1)
    s["3P%"]    = (agg["fg3m"] / agg["fg3a"] * 100).round(1)
    s["3PaPG"]  = (agg["fg3a"] / g).round(1)
    s["3PmPG"]  = (agg["fg3m"] / g).round(1)
    s["FT%"]    = (agg["ftm"] / agg["fta"] * 100).round(1)
    s["FTmPG"]  = (agg["ftm"] / g).round(1)

    # National rank for each stat (rank 1 = best); lower is better for OppPPG and TOPG.
    # Filter to rated D1 teams only before ranking so max rank = number of rated teams.
    with SessionLocal() as session:
        rated_ids = {
            row[0] for row in
            session.query(CarmPomRating.team_id).filter(CarmPomRating.season == season).all()
        }
    s = s[s["team_id"].isin(rated_ids)].copy()

    stat_rank_cfg = {
        "PPG": False, "OppPPG": True, "RebPG": False, "AstPG": False,
        "OrebPG": False, "TOPG": True, "FG%": False, "3P%": False,
        "3PaPG": False, "3PmPG": False, "FT%": False, "FTmPG": False,
    }
    for col, asc in stat_rank_cfg.items():
        s[f"{col}_nr"] = s[col].rank(ascending=asc, method="min").astype(int)

    return s


# ---------------------------------------------------------------------------
# KenPom comparison loader (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_kenpom_comparison(season: int) -> pd.DataFrame:
    """Join CarmPom ratings with KenPom 2026 ranks and return disagreement data."""
    kp_ranks = pd.read_csv(ROOT / "data" / "kenpom2026_ranks.csv")

    with SessionLocal() as session:
        cp_rows = (
            session.query(
                CarmPomRating.team_id,
                CarmPomRating.rank,
                Team.name,
                Team.conference,
                CarmPomRating.adjem,
            )
            .join(Team, Team.id == CarmPomRating.team_id)
            .filter(CarmPomRating.season == season)
            .order_by(CarmPomRating.rank)
            .all()
        )

    cp = pd.DataFrame(cp_rows, columns=["team_id", "cp_rank", "Team", "Conf", "cp_adjem"])
    cp["team_id"] = cp["team_id"].astype(int)
    cp["Conf"] = cp["Conf"].str.removesuffix(" Conference")

    compare = (
        cp
        .merge(kp_ranks[["db_team_id", "kp_rank", "kp_name"]], left_on="team_id", right_on="db_team_id", how="inner")
        .drop_duplicates(subset="team_id")
    )
    compare["rank_diff"] = compare["kp_rank"] - compare["cp_rank"]
    return compare


# ---------------------------------------------------------------------------
# Bracket simulation helpers
# ---------------------------------------------------------------------------

import math

# S-curve region assignment for 64 teams.
# Seeds 1-4 → one #1 seed per region; odd groups go E→W→S→MW, even reverse.
_REGIONS = ["East", "West", "South", "Midwest"]
_SCURVE_REGIONS: list[int] = []
for _g in range(16):
    if _g % 2 == 0:
        _SCURVE_REGIONS.extend([0, 1, 2, 3])
    else:
        _SCURVE_REGIONS.extend([3, 2, 1, 0])


def build_projected_bracket(df: pd.DataFrame, n_teams: int = 64) -> pd.DataFrame:
    """Assign CarmPom top-n teams to a 4-region S-curve bracket.

    Returns a copy of the top-n rows with 'cp_seed', 'region', and 'seed' columns added.
    """
    top = df.head(n_teams).reset_index(drop=True).copy()
    top["cp_seed"] = range(1, n_teams + 1)
    top["region"]  = [_REGIONS[_SCURVE_REGIONS[i]] for i in range(n_teams)]
    top["seed"]    = [(i // 4) + 1 for i in range(n_teams)]  # 1-16 within region
    return top


_BRACKET_CSV = ROOT / "data" / "bracket_2026.csv"


def load_real_bracket(season: int = 2026) -> pd.DataFrame | None:
    """Read data/bracket_2026.csv and merge in CarmPom ratings.

    Returns a bracket DataFrame (same shape as build_projected_bracket output)
    when the CSV is fully filled out (all 64 rows have a team name), otherwise
    returns None so the caller can fall back to the projected bracket.
    """
    import difflib

    if not _BRACKET_CSV.exists():
        return None

    raw = pd.read_csv(_BRACKET_CSV)
    filled = raw[raw["team"].notna() & (raw["team"].str.strip() != "")].copy()
    if len(filled) < 64:
        return None  # bracket not fully entered yet

    # Pull all team names + IDs from DB
    with SessionLocal() as session:
        db_teams = pd.DataFrame(
            session.query(Team.id, Team.name).all(),
            columns=["team_id", "db_name"],
        )
        ratings = pd.DataFrame(
            session.query(
                CarmPomRating.team_id,
                CarmPomRating.rank,
                CarmPomRating.adjem,
                CarmPomRating.adjo,
                CarmPomRating.adjd,
                CarmPomRating.wins,
                CarmPomRating.losses,
            )
            .filter(CarmPomRating.season == season)
            .all(),
            columns=["team_id", "Rank", "AdjEM", "AdjO", "AdjD", "W", "L"],
        )

    # Build a lookup: normalized db name → team_id
    db_teams["norm"] = db_teams["db_name"].str.lower().str.strip()
    norm_to_id = dict(zip(db_teams["norm"], db_teams["team_id"]))
    norm_to_name = dict(zip(db_teams["norm"], db_teams["db_name"]))
    all_norms = list(norm_to_id.keys())

    def _resolve(csv_name: str) -> int | None:
        """Map a CSV team name to a DB team_id."""
        key = csv_name.lower().strip()
        if key in norm_to_id:
            return norm_to_id[key]
        # Try substring: DB name contains the CSV name or vice versa
        for norm in all_norms:
            if key in norm or norm in key:
                return norm_to_id[norm]
        # Difflib close match as last resort
        matches = difflib.get_close_matches(key, all_norms, n=1, cutoff=0.7)
        return norm_to_id[matches[0]] if matches else None

    filled["team_id"] = filled["team"].apply(_resolve)
    unmatched = filled[filled["team_id"].isna()]["team"].tolist()
    if unmatched:
        # Surface unmatched names so the user can fix the CSV
        raise ValueError(f"bracket_2026.csv: could not match these team names — {unmatched}")

    filled["team_id"] = filled["team_id"].astype(int)

    # Merge in ratings + metadata
    bracket = (
        filled
        .merge(db_teams[["team_id", "db_name"]].rename(columns={"db_name": "Team"}), on="team_id")
        .merge(ratings, on="team_id", how="left")
    )
    bracket["seed"] = bracket["seed"].astype(int)
    bracket["region"] = bracket["region"].str.strip()
    bracket["Record"] = bracket["W"].fillna(0).astype(int).astype(str) + "-" + bracket["L"].fillna(0).astype(int).astype(str)
    bracket["Conf"] = ""
    # Attach conference from DB
    with SessionLocal() as session:
        conf_map = {r.id: (r.conference or "").removesuffix(" Conference")
                    for r in session.query(Team.id, Team.conference).all()}
    bracket["Conf"] = bracket["team_id"].map(conf_map).fillna("")
    bracket["cp_seed"] = bracket["Rank"].fillna(999).astype(int)
    bracket["AdjEM"] = bracket["AdjEM"].fillna(0.0)
    bracket["Rank"] = bracket["Rank"].fillna(999).astype(int)
    return bracket.reset_index(drop=True)


def _win_prob(adjem_a: float, adjem_b: float) -> float:
    """Logistic win probability from AdjEM differential.

    Coefficient 0.11 reflects single-elimination variance — March Madness is one-and-done,
    so even a significant efficiency edge doesn't guarantee a win. At +10 AdjEM: ~75% win prob.
    """
    return 1.0 / (1.0 + math.exp(-0.11 * (adjem_a - adjem_b)))


def simulate_bracket(
    bracket: pd.DataFrame,
    n_sims: int = 25_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Monte-Carlo simulate the bracket n_sims times.

    Returns a DataFrame indexed by team with advance-probability columns:
    R64_pct, R32_pct, S16_pct, E8_pct, F4_pct, Champ_pct.
    """
    import random
    rng = random.Random(rng_seed)

    round_labels = ["R64", "R32", "S16", "E8", "F4", "Champ"]
    # advance_counts[team_id][round_idx] = number of sims they reached that round
    team_ids = list(bracket["team_id"])
    advance: dict[int, list[int]] = {tid: [0] * 6 for tid in team_ids}

    # Build per-region seeding: {region: [(seed, team_id, adjem), ...]}
    regions_dict: dict[str, list[tuple[int, int, float]]] = {r: [] for r in _REGIONS}
    for _, row in bracket.iterrows():
        regions_dict[row["region"]].append((int(row["seed"]), int(row["team_id"]), float(row["AdjEM"])))
    for r in regions_dict:
        regions_dict[r].sort(key=lambda x: x[0])

    # NCAA bracket matchup order within a 16-team region:
    # (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
    _SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    _seed_pos = {s: i for i, s in enumerate(_SEED_ORDER)}

    for _ in range(n_sims):
        # Simulate each region to produce 4 Final Four teams
        f4_contestants: list[tuple[int, float]] = []  # (team_id, adjem)
        for region_teams in regions_dict.values():
            # Reorder by bracket position
            ordered = sorted(region_teams, key=lambda x: _seed_pos.get(x[0], x[0]))
            # Round of 64 — 8 matchups
            survivors: list[tuple[int, float]] = []
            for i in range(0, 16, 2):
                a_tid, a_em = ordered[i][1], ordered[i][2]
                b_tid, b_em = ordered[i + 1][1], ordered[i + 1][2]
                advance[a_tid][0] += 1
                advance[b_tid][0] += 1
                winner = a_tid if rng.random() < _win_prob(a_em, b_em) else b_tid
                winner_em = a_em if winner == a_tid else b_em
                survivors.append((winner, winner_em))
            adjem_map = {t[1]: t[2] for t in region_teams}
            # Rounds 32 → E8 (round_idx 1,2,3)
            for rnd_idx in range(1, 4):
                next_survivors: list[tuple[int, float]] = []
                for i in range(0, len(survivors), 2):
                    a_tid, a_em = survivors[i]
                    b_tid, b_em = survivors[i + 1]
                    advance[a_tid][rnd_idx] += 1
                    advance[b_tid][rnd_idx] += 1
                    winner = a_tid if rng.random() < _win_prob(a_em, b_em) else b_tid
                    winner_em = a_em if winner == a_tid else b_em
                    next_survivors.append((winner, winner_em))
                survivors = next_survivors
            f4_contestants.extend(survivors)

        # Final Four (round_idx 4)
        semi_winners: list[tuple[int, float]] = []
        for i in range(0, 4, 2):
            a_tid, a_em = f4_contestants[i]
            b_tid, b_em = f4_contestants[i + 1]
            advance[a_tid][4] += 1
            advance[b_tid][4] += 1
            winner = a_tid if rng.random() < _win_prob(a_em, b_em) else b_tid
            winner_em = a_em if winner == a_tid else b_em
            semi_winners.append((winner, winner_em))

        # Championship (round_idx 5)
        a_tid, a_em = semi_winners[0]
        b_tid, b_em = semi_winners[1]
        advance[a_tid][5] += 1
        advance[b_tid][5] += 1
        winner = a_tid if rng.random() < _win_prob(a_em, b_em) else b_tid
        # Mark champion as having won the championship round
        # (we count reaching the game, not winning — add an extra column for actual wins)
        advance[winner][5] += 1  # counted twice → subtract later for actual champ %
        # Actually let's track champ wins separately with a bonus index

    # Build result df
    results = []
    for _, row in bracket.iterrows():
        tid = int(row["team_id"])
        cnts = advance[tid]
        results.append({
            "team_id": tid,
            "Team": row["Team"],
            "Conf": row["Conf"],
            "Record": row["Record"],
            "Region": row["region"],
            "Seed": int(row["seed"]),
            "CarmPomRk": int(row["Rank"]),
            "AdjEM": float(row["AdjEM"]),
            "R64%":   round(cnts[0] / n_sims * 100, 1),
            "R32%":   round(cnts[1] / n_sims * 100, 1),
            "S16%":   round(cnts[2] / n_sims * 100, 1),
            "E8%":    round(cnts[3] / n_sims * 100, 1),
            "F4%":    round(cnts[4] / n_sims * 100, 1),
            "Champ%": round(cnts[5] / (n_sims * 2) * 100, 1),  # divide by 2 (counted twice above)
        })
    return pd.DataFrame(results).sort_values("Champ%", ascending=False).reset_index(drop=True)


def generate_matchup_analysis(
    ta: pd.Series, tb: pd.Series, wp: float, n_teams: int
) -> list[str]:
    """Return bullet-point matchup insights for team A vs team B.

    wp = CarmPom win probability for team A (0–1).
    n_teams = total D1 rated teams (used for pct thresholds).
    """
    bullets: list[str] = []
    a_name = ta["Team"]
    b_name = tb["Team"]
    adjem_diff = float(ta["AdjEM"]) - float(tb["AdjEM"])

    # --- Overall outlook ---
    if abs(adjem_diff) < 2:
        bullets.append(
            f"🎲 **True toss-up** — CarmPom rates these teams within {abs(adjem_diff):.1f} pts/100 of each other. "
            "Either team wins this on any given night."
        )
    elif abs(adjem_diff) < 6:
        fav = a_name if adjem_diff > 0 else b_name
        bullets.append(
            f"📊 **Slight edge for {fav}** — a {abs(adjem_diff):.1f} pt/100 efficiency gap is meaningful "
            "but absolutely beatable in a single game."
        )
    elif abs(adjem_diff) < 15:
        fav = a_name if adjem_diff > 0 else b_name
        dog = b_name if adjem_diff > 0 else a_name
        bullets.append(
            f"📈 **{fav} is the clear favorite** ({abs(adjem_diff):.1f} pts/100 better). "
            f"{dog} needs to neutralize that edge early or it could get away from them."
        )
    else:
        fav = a_name if adjem_diff > 0 else b_name
        dog = b_name if adjem_diff > 0 else a_name
        bullets.append(
            f"⚡ **Big mismatch** — {fav} is {abs(adjem_diff):.1f} pts/100 more efficient. "
            f"{dog} would need an historically great performance to pull this off."
        )

    # --- Offensive battle ---
    adjo_a = float(ta.get("AdjO", 100))
    adjo_b = float(tb.get("AdjO", 100))
    adjo_diff = adjo_a - adjo_b
    adjd_a = float(ta.get("AdjD", 100))
    adjd_b = float(tb.get("AdjD", 100))
    adjd_diff = adjd_a - adjd_b  # negative = ta has better (lower) defense

    if abs(adjo_diff) >= 4:
        better_off = a_name if adjo_diff > 0 else b_name
        faces_def = b_name if adjo_diff > 0 else a_name
        def_rating = adjd_b if adjo_diff > 0 else adjd_a
        bullets.append(
            f"⚔️ **Offensive mismatch** — {better_off}'s offense ({max(adjo_a, adjo_b):.1f} AdjO) "
            f"is the best attack {faces_def} has seen all season. Their defense gives up {def_rating:.1f} pts/100."
        )

    # --- Defensive battle ---
    if abs(adjd_diff) >= 4:
        better_def = a_name if adjd_diff < 0 else b_name  # lower AdjD = better defense
        opp_def = b_name if adjd_diff < 0 else a_name
        best_def = min(adjd_a, adjd_b)
        bullets.append(
            f"🛡️ **Defensive anchor** — {better_def} ({best_def:.1f} AdjD) is one of the stingiest "
            f"defenses in the country. {opp_def} will need to shoot well to keep up."
        )

    # --- Tempo clash ---
    adjt_a = float(ta.get("AdjT", 68))
    adjt_b = float(tb.get("AdjT", 68))
    adjt_diff = abs(adjt_a - adjt_b)
    if adjt_diff >= 3:
        faster = a_name if adjt_a > adjt_b else b_name
        slower = b_name if adjt_a > adjt_b else a_name
        bullets.append(
            f"🏃 **Pace battle** — {faster} wants to run, {slower} wants to grind. "
            "Whoever imposes their preferred tempo gains a structural edge — watch the early possessions."
        )

    # --- Seed / upset flag ---
    seed_a = int(ta.get("seed", 8))
    seed_b = int(tb.get("seed", 9))
    high_seed = max(seed_a, seed_b)
    model_wp = wp if adjem_diff >= 0 else (1 - wp)
    if high_seed >= 10 and model_wp < 0.80:
        underdog = a_name if seed_a > seed_b else b_name
        bullets.append(
            f"👀 **Upset potential** — the #{high_seed} seed ({underdog}) is closer in efficiency "
            "than the seed gap suggests. Don't sleep on this one."
        )

    return bullets


def generate_style_profile(t: pd.Series, n: int) -> str:
    """2-3 sentence style-focused summary of a single team, for use in matchup panels."""
    name = t["Team"]
    adjt = float(t.get("AdjT", 68.0))
    adjt_nr = int(t.get("AdjT_nr", n // 2))
    adjo = float(t.get("AdjO", 100.0))
    adjo_nr = int(t.get("AdjO_nr", n // 2))
    adjd = float(t.get("AdjD", 100.0))
    adjd_nr = int(t.get("AdjD_nr", n // 2))
    luck = float(t.get("Luck", 0.0))

    # Tempo sentence
    if adjt_nr <= 50:
        tempo = f"pushes the pace as aggressively as almost anyone in D1 (#{adjt_nr} nationally, {adjt:.1f} adj. poss/40 min)"
    elif adjt_nr <= 130:
        tempo = f"operates at an above-average pace ({adjt:.1f} adj. poss/40 min)"
    elif adjt_nr <= 250:
        tempo = f"prefers a deliberate, controlled pace ({adjt:.1f} adj. poss/40 min)"
    else:
        tempo = f"plays one of the slowest tempos in college basketball (#{adjt_nr} nationally, {adjt:.1f} adj. poss/40 min)"

    # Offense tier
    if adjo_nr <= 15:
        off_str = f"one of the most efficient offenses in D1 (#{adjo_nr}, {adjo:.1f} adj. pts/100)"
    elif adjo_nr <= 50:
        off_str = f"an elite offense (#{adjo_nr}, {adjo:.1f} pts/100)"
    elif adjo_nr <= 120:
        off_str = f"an above-average offense (#{adjo_nr})"
    elif adjo_nr <= 230:
        off_str = f"a serviceable but unspectacular offense (#{adjo_nr})"
    else:
        off_str = f"an offense that has struggled to generate efficient looks (#{adjo_nr})"

    # Defense sentence
    if adjd_nr <= 15:
        def_str = f"Elite defensively (#{adjd_nr}, {adjd:.1f} adj. pts/100 allowed) — one of the hardest teams to score against in the country."
    elif adjd_nr <= 50:
        def_str = f"Defensively they're among the best in the country (#{adjd_nr}), making every possession a grind for opponents."
    elif adjd_nr <= 120:
        def_str = f"Defensively they're solid (#{adjd_nr}) — above average, but not a shutdown unit."
    elif adjd_nr <= 230:
        def_str = f"Their defense sits around the national median (#{adjd_nr}) — not a liability, but not a separator either."
    else:
        def_str = f"Defense has been a liability all season (#{adjd_nr}), something opponents will look to exploit."

    # Luck context
    luck_str = ""
    if luck < -0.04:
        luck_str = " Their record actually understates how well they've played — they've been genuinely unlucky in close games."
    elif luck > 0.05:
        luck_str = " Worth noting: they've been fortunate in close games, meaning their record may slightly flatter their real quality."

    return f"{name} {tempo}, featuring {off_str}. {def_str}{luck_str}"


def generate_clash_narrative(ta: pd.Series, tb: pd.Series, wp_a: float, n: int) -> str:
    """2-3 sentence prose about how two teams' styles, schedules, and form will collide."""
    name_a, name_b = ta["Team"], tb["Team"]
    adjt_nr_a = int(ta.get("AdjT_nr", n // 2))
    adjt_nr_b = int(tb.get("AdjT_nr", n // 2))
    adjo_nr_a = int(ta.get("AdjO_nr", n // 2))
    adjo_nr_b = int(tb.get("AdjO_nr", n // 2))
    adjd_nr_a = int(ta.get("AdjD_nr", n // 2))
    adjd_nr_b = int(tb.get("AdjD_nr", n // 2))
    em_a = float(ta.get("AdjEM", 0.0))
    em_b = float(tb.get("AdjEM", 0.0))
    sos_nr_a = int(ta.get("SOS_nr", n // 2))
    sos_nr_b = int(tb.get("SOS_nr", n // 2))
    luck_a = float(ta.get("Luck", 0.0))
    luck_b = float(tb.get("Luck", 0.0))
    seed_a = int(ta.get("seed", 8))
    seed_b = int(tb.get("seed", 9))

    parts: list[str] = []

    # 1. Pace tension
    tempo_gap = abs(adjt_nr_a - adjt_nr_b)
    if tempo_gap >= 120:
        faster = name_a if adjt_nr_a < adjt_nr_b else name_b
        slower = name_b if adjt_nr_a < adjt_nr_b else name_a
        parts.append(
            f"**Pace is the central battle** — {faster} wants to get out and run while {slower} thrives "
            "in a half-court grind. Whoever controls tempo in the opening possessions sets the tone for everything that follows."
        )
    elif tempo_gap >= 50:
        faster = name_a if adjt_nr_a < adjt_nr_b else name_b
        slower = name_b if adjt_nr_a < adjt_nr_b else name_a
        parts.append(
            f"There's a meaningful pace differential: {faster} prefers to push while {slower} likes to slow things down — "
            "expect both teams jostling for tempo control from the opening tip."
        )
    else:
        parts.append(
            "Both teams operate at a similar tempo, so this will come down to half-court execution — "
            "shot quality, ball security, and who can impose their offensive system."
        )

    # 2. Key offensive vs. defensive matchup
    if adjo_nr_a <= 60 and adjd_nr_b >= 200:
        parts.append(
            f"{name_a}'s efficient offense (#{adjo_nr_a}) runs into one of the weaker defenses in the field (#{adjd_nr_b}) — "
            "the favorite should find comfortable looks throughout and won't need a perfect game to win."
        )
    elif adjo_nr_b <= 60 and adjd_nr_a >= 200:
        parts.append(
            f"{name_b}'s offense (#{adjo_nr_b}) against {name_a}'s defense (#{adjd_nr_a}) is a genuine mismatch that "
            "could fuel an upset — if the underdog gets rolling early, this game can flip quickly."
        )
    elif adjd_nr_a <= 40 and adjo_nr_b >= 180:
        parts.append(
            f"{name_a}'s defense (#{adjd_nr_a}) matches up well against {name_b}'s offense (#{adjo_nr_b}), "
            "which could keep the game low-scoring and well within the favorite's comfort zone."
        )
    elif adjd_nr_b <= 40 and adjo_nr_a >= 180:
        parts.append(
            f"{name_b}'s defense (#{adjd_nr_b}) is the key stress test for {name_a}'s offense (#{adjo_nr_a}) — "
            "can the favorite generate quality looks against a unit that has been shutting teams down all season?"
        )
    else:
        em_gap = abs(em_a - em_b)
        if em_gap <= 3.0:
            parts.append(
                f"The efficiency gap is razor-thin ({em_a:+.1f} vs {em_b:+.1f}) — both teams are nearly identical in overall quality. "
                "A single hot-shooting stretch, a turnover run, or one player stepping up can decide the whole thing."
            )
        elif em_gap <= 8.0:
            fav = name_a if em_a > em_b else name_b
            dog = name_b if em_a > em_b else name_a
            parts.append(
                f"{fav} holds a real efficiency edge ({em_gap:.1f} pts/100 AdjEM), but {dog} is capable enough "
                "that a strong game plan and a few favorable bounces could absolutely swing this."
            )
        else:
            fav = name_a if em_a > em_b else name_b
            parts.append(
                f"The efficiency gap ({em_gap:.1f} pts/100) heavily favors {fav} — the underdog's best path is controlling "
                "pace, protecting the ball, and staying close enough that a hot second half becomes possible."
            )

    # 3. Schedule / luck context — pick the most interesting angle
    sos_gap = abs(sos_nr_a - sos_nr_b)
    if sos_gap >= 120:
        harder = name_a if sos_nr_a < sos_nr_b else name_b
        easier = name_b if sos_nr_a < sos_nr_b else name_a
        hr = min(sos_nr_a, sos_nr_b)
        er = max(sos_nr_a, sos_nr_b)
        parts.append(
            f"Schedule context adds a wrinkle: {harder}'s numbers were built against a top-{hr} strength of schedule, "
            f"while {easier} compiled theirs against softer competition (#{er} SOS) — tournament basketball is a different animal."
        )
    elif luck_b < -0.04 and seed_b > seed_a:
        parts.append(
            f"One wildcard: {name_b} has been genuinely unlucky in close games all season — their record undersells "
            "how competitive they've been, and a neutral-court tournament game is exactly when that tends to even out."
        )
    elif luck_a > 0.05 and seed_a < seed_b:
        parts.append(
            f"One caveat for {name_a}: they've been fortunate in clutch situations throughout the year, "
            "meaning their record is a touch rosier than their underlying efficiency alone would suggest."
        )

    return " ".join(parts[:3])


# ---------------------------------------------------------------------------
# Odds + Injury cache loaders (read from pipeline output files)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def _load_odds_data() -> dict:
    """Load betting lines from data/odds_cache.json (written by pipeline/fetch_odds.py).

    Returns a nested dict: odds_lu[team_a][team_b] = {ml, spread, impl_prob, ...}
    Returns {} if cache doesn't exist yet.
    """
    import json as _json
    _p = ROOT / "data" / "odds_cache.json"
    if not _p.exists():
        return {}
    try:
        return _json.loads(_p.read_text(encoding="utf-8")).get("odds", {})
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def _load_injuries_data() -> dict:
    """Load ESPN-scraped injury notes from data/injuries_cache.json.

    Returns a dict: inj_lu[team_name] = [{headline, status, description, published}, ...]
    """
    import json as _json
    _p = ROOT / "data" / "injuries_cache.json"
    if not _p.exists():
        return {}
    try:
        data = _json.loads(_p.read_text(encoding="utf-8"))
        return data.get("teams", {})
    except Exception:
        return {}


def _ml_to_prob(ml: float) -> float:
    """Convert American moneyline to no-vig-adjusted implied probability."""
    if ml < 0:
        raw = abs(ml) / (abs(ml) + 100)
    else:
        raw = 100 / (ml + 100)
    return raw   # caller removes vig by normalising with the other side


def _odds_for_matchup(ta_name: str, tb_name: str, odds_lu: dict) -> tuple[dict, dict] | tuple[None, None]:
    """Look up betting lines for a matchup from the odds cache (both directions).

    Returns (line_a, line_b) or (None, None) if not found.
    """
    from difflib import get_close_matches as _gcm
    all_keys = list(odds_lu.keys())

    def _find(name: str) -> str | None:
        if name in odds_lu:
            return name
        hits = _gcm(name, all_keys, n=1, cutoff=0.55)
        return hits[0] if hits else None

    ka = _find(ta_name)
    if ka is None:
        return None, None
    kb = _find(tb_name) if tb_name in odds_lu.get(ka, {}) else None
    if kb is None:
        # Try looking in kb's inner dict
        for inner_key in odds_lu.get(ka, {}):
            hits = _gcm(tb_name, [inner_key], n=1, cutoff=0.55)
            if hits:
                kb = inner_key
                break
    if kb is None:
        return None, None

    line_a = odds_lu[ka].get(kb)
    line_b = odds_lu.get(kb, {}).get(ka)
    return line_a, line_b


def _generate_single_game_bullets(
    ta: "pd.Series",
    tb: "pd.Series",
    pg_a: dict,
    pg_b: dict,
    wp_a: float,
    odds_lu: dict,
    n: int,
) -> list[str]:
    """Return single-game reality-check bullets for the matchup analyzer.

    Unlike the simulation analysis (which averages 25K games), these bullets
    focus on what can actually change the outcome of ONE game.
    """
    bullets: list[str] = []
    name_a, name_b = ta["Team"], tb["Team"]
    adjo_a = float(ta.get("AdjO", 100))
    adjo_b = float(tb.get("AdjO", 100))
    adjd_a = float(ta.get("AdjD", 100))
    adjd_b = float(tb.get("AdjD", 100))
    adjt_a = float(ta.get("AdjT", 68))
    adjt_b = float(tb.get("AdjT", 68))
    avg_tempo = (adjt_a + adjt_b) / 2.0

    # ── Expected score estimate ──────────────────────────────────────────
    # Uses AdjO * AdjD cross formula (comparable to KenPom pythagorean)
    _NAT_AVG = 100.0
    eff_a = adjo_a * (adjd_b / _NAT_AVG)    # pts/100 for team A vs this defense
    eff_b = adjo_b * (adjd_a / _NAT_AVG)
    pts_a = round(eff_a * avg_tempo / 100)
    pts_b = round(eff_b * avg_tempo / 100)
    margin = pts_a - pts_b
    fav_name  = name_a if pts_a > pts_b else name_b
    dog_name  = name_b if pts_a > pts_b else name_a
    fav_pts   = max(pts_a, pts_b)
    dog_pts   = min(pts_a, pts_b)
    dog_deficit = abs(margin)

    bullets.append(
        f"📐 **Projected score** (AdjEM/AdjT model): {name_a} **{pts_a}** – {name_b} **{pts_b}**. "
        f"{fav_name} is expected to win by ~{dog_deficit} points — but that's one game, and margins like these flip routinely."
    )

    # ── Path to upset ─────────────────────────────────────────────────────
    if dog_deficit >= 3:
        # How much suppression does the underdog need?
        # Dog needs: fav_new_score ≤ dog_pts → fav efficiency needs to drop by X%
        suppress_pct = round((dog_deficit / fav_pts) * 100)
        # OR: dog needs to score X% more than projected
        boost_pct = round((dog_deficit / dog_pts) * 100)
        bullets.append(
            f"🔑 **Path to upset for {dog_name}**: They need to either hold {fav_name} to ≤ "
            f"**{dog_pts} pts** (a {suppress_pct}% efficiency reduction vs projection), "
            f"or score ≥ **{fav_pts} pts** themselves (a {boost_pct}% offensive boost). "
            "In a single game, both are possible — that's the whole point."
        )
    else:
        bullets.append(
            f"🎲 **Dead heat**: Expected margin is just {dog_deficit} pts — this game is genuinely decided by "
            "execution, not talent gap. Any advantage (hot shooter, key foul call, transition bucket) wins it."
        )

    # ── Three-point variance ───────────────────────────────────────────────
    for tname, pg in [(name_a, pg_a), (name_b, pg_b)]:
        three_pa = float(pg.get("3PaPG", 0) or 0) if pg else 0
        three_pct = float(pg.get("3P%", 35) or 35) / 100 if pg else 0.35
        if three_pa >= 18:
            # ±8 percentage point swing in one game is normal (1 standard deviation ≈ 7-8%)
            swing_pts = round(0.08 * three_pa * 3, 1)
            hot_pts = round(three_pct * three_pa * 3 + swing_pts, 1)
            cold_pts = round(max(0, three_pct * three_pa * 3 - swing_pts), 1)
            tier = "elite" if three_pa >= 25 else "heavy"
            bullets.append(
                f"🌐 **Three-point variance — {tname}**: They attempt **{three_pa:.0f} threes/game** ({tier} volume). "
                f"A normal cold shooting night ({round((three_pct - 0.08)*100)}% from three) costs them ~{swing_pts:.0f} pts. "
                f"A hot night ({round((three_pct + 0.08)*100)}%) adds ~{swing_pts:.0f} pts. "
                "One-game swings like this are decisive."
            )
            break  # only show the more three-heavy team to keep it concise

    # ── Market vs Model ───────────────────────────────────────────────────
    line_a, line_b = _odds_for_matchup(name_a, name_b, odds_lu)
    if line_a and line_a.get("impl_prob") is not None:
        book_prob_a = float(line_a["impl_prob"])
        book_prob_b = 1.0 - book_prob_a
        model_pct_a = round(wp_a * 100)
        book_pct_a  = round(book_prob_a * 100)
        gap = model_pct_a - book_pct_a
        ml_str = f"{int(line_a['ml']):+d}" if line_a.get("ml") else "N/A"
        spread_str = (
            f"spread: {line_a['spread']:+.1f}" if line_a.get("spread") is not None else "no spread available"
        )
        if abs(gap) >= 5:
            if gap > 0:
                # CarmPom likes name_a more than market
                bullets.append(
                    f"📊 **Model vs Market edge — {name_a}**: CarmPom gives them **{model_pct_a}%** vs Vegas **{book_pct_a}%** "
                    f"(ML {ml_str}, {spread_str}). That's a {abs(gap)}-pt gap — CarmPom sees more here than the market does."
                )
            else:
                # Market likes name_a more than CarmPom
                bullets.append(
                    f"📊 **Model vs Market divergence — {name_b}**: CarmPom gives the underdog **{100 - model_pct_a}%** "
                    f"vs Vegas **{100 - book_pct_a}%** (ML {ml_str}, {spread_str}). "
                    f"The market is giving {name_b} {abs(gap)} more percentage points than CarmPom — "
                    "something the oddsmakers know that pure efficiency doesn't capture."
                )
        else:
            bullets.append(
                f"📊 **Market alignment** (ML {ml_str}, {spread_str}): CarmPom ({model_pct_a}%) and Vegas ({book_pct_a}%) "
                "are within 5 percentage points of each other — both models see roughly the same game."
            )
    else:
        bullets.append(
            "📊 **Betting lines**: Run `uv run python pipeline/fetch_odds.py` to pull live ESPN odds, "
            "then refresh the page — market vs model comparison will appear here automatically."
        )

    # ── Possession randomness ─────────────────────────────────────────────
    bullets.append(
        f"🎯 **The one-game truth**: March Madness upsets happen because single games are decided by ~{int(avg_tempo)} possessions. "
        "That means 5–8 genuinely 50/50 moments (tip-offs, charge/block calls, scramble rebounds, late-clock shots) "
        "can completely reverse a 10-point efficiency gap. The stats set the stage — randomness writes the ending."
    )

    return bullets
# ---------------------------------------------------------------------------

_BP_MU_PAIRS   = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
_BP_REGIONS    = ["East", "West", "South", "Midwest"]
_BP_ROUND_NAMES = ["1st Round", "Round of 32", "Sweet 16", "Elite Eight", "Final Four", "Championship"]
_BP_ROUND_SLOTS = [32, 16, 8, 4, 2, 1]  # total games per round


def _bp_r1_teams(slot: int, brkt: pd.DataFrame) -> tuple[str, str]:
    """(team_a, team_b) for a 1st-round slot index 0–31."""
    region = _BP_REGIONS[slot // 8]
    sa, sb = _BP_MU_PAIRS[slot % 8]
    reg = brkt[brkt["region"] == region]
    seed_lu = {int(r["seed"]): r["Team"] for _, r in reg.iterrows()}
    return seed_lu.get(sa, "TBD"), seed_lu.get(sb, "TBD")


def _bp_candidates(rnd: int, slot: int, picks: dict, brkt: pd.DataFrame) -> tuple[str, str]:
    """The two teams that play in (rnd, slot), determined by prior picks."""
    if rnd == 0:
        return _bp_r1_teams(slot, brkt)
    ta = picks.get((rnd - 1, 2 * slot)) or "TBD"
    tb = picks.get((rnd - 1, 2 * slot + 1)) or "TBD"
    return ta, tb


def _bp_prob_stats(
    picks: dict, brkt: pd.DataFrame, r_lu: dict
) -> tuple[float, int, float]:
    """
    Returns (probability_pct, picks_made, expected_correct_picks).

    probability_pct  — probability that every single pick is right (0–100).
    picks_made       — count of filled slots.
    expected_correct — sum of individual win-probs for each pick.
    """
    prob = 1.0
    made = 0
    expected = 0.0
    for rnd, total in enumerate(_BP_ROUND_SLOTS):
        for slot in range(total):
            pick = picks.get((rnd, slot))
            if pick is None:
                continue
            made += 1
            ta, tb = _bp_candidates(rnd, slot, picks, brkt)
            if ta == "TBD" or tb == "TBD":
                continue
            wp_a = _win_prob(
                float(r_lu.get(ta, {}).get("AdjEM", 0)),
                float(r_lu.get(tb, {}).get("AdjEM", 0)),
            )
            wp_pick = wp_a if pick == ta else (1.0 - wp_a)
            prob *= wp_pick
            expected += wp_pick
    return prob * 100.0, made, expected


def _bp_autofill(
    mode: str, existing: dict, brkt: pd.DataFrame, r_lu: dict
) -> dict:
    """Return a new picks dict auto-filled according to mode.

    mode: 'chalk'    — always pick the higher-AdjEM team.
          'balanced' — pick favorites but allow clear-value upsets (dog WP > 35%, seed gap >= 4).
          'chaos'    — pick the underdog whenever they have > 27% chance.
    """
    new_picks: dict = dict(existing)
    for rnd, total in enumerate(_BP_ROUND_SLOTS):
        for slot in range(total):
            if (rnd, slot) in new_picks:
                continue  # don't overwrite existing manual picks
            ta, tb = _bp_candidates(rnd, slot, new_picks, brkt)
            if ta == "TBD" or tb == "TBD":
                continue
            em_a = float(r_lu.get(ta, {}).get("AdjEM", 0))
            em_b = float(r_lu.get(tb, {}).get("AdjEM", 0))
            wp_a = _win_prob(em_a, em_b)

            if mode == "chalk":
                new_picks[(rnd, slot)] = ta if wp_a >= 0.5 else tb

            elif mode == "balanced":
                # Seed gap only meaningful in R0; later rounds use the actual seeds from bracket
                seed_gap = 0
                if rnd == 0:
                    sa, sb = _BP_MU_PAIRS[slot % 8]
                    seed_gap = sb - sa  # underdog = higher seed number
                underdog_wp = (1 - wp_a) if wp_a >= 0.5 else wp_a
                underdog = tb if wp_a >= 0.5 else ta
                # Pick the underdog if they're a meaningful upset candidate
                if underdog_wp >= 0.35 and (seed_gap >= 4 or rnd >= 2):
                    new_picks[(rnd, slot)] = underdog
                else:
                    new_picks[(rnd, slot)] = ta if wp_a >= 0.5 else tb

            else:  # chaos
                underdog = tb if wp_a >= 0.5 else ta
                underdog_wp = (1 - wp_a) if wp_a >= 0.5 else wp_a
                if underdog_wp >= 0.27:
                    new_picks[(rnd, slot)] = underdog
                else:
                    new_picks[(rnd, slot)] = ta if wp_a >= 0.5 else tb

    return new_picks


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

rankings_tab, team_tab, bracket_tab, pick_tab, about_tab = st.tabs(
    ["📊 Team Rankings", "🏀 Team Profile", "🏆 Bracket Sim", "🗳️ My Bracket", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Rankings tab
# ---------------------------------------------------------------------------

with rankings_tab:
    # Controls row
    col_season, col_search, col_conf, col_n = st.columns([1, 2, 2, 1])

    with col_season:
        season = st.selectbox("Season", [2026], index=0)

    df = load_rankings(season)

    with col_conf:
        confs = ["All conferences"] + sorted(df["Conf"].dropna().unique().tolist())
        selected_conf = st.selectbox("Conference", confs)

    with col_search:
        search = st.text_input("Search team", placeholder="e.g. Duke")

    with col_n:
        top_n = st.selectbox("Show", [25, 50, 100, 364], index=3)

    # Apply filters
    filtered = df.copy()
    if selected_conf != "All conferences":
        filtered = filtered[filtered["Conf"] == selected_conf]
    if search:
        filtered = filtered[filtered["Team"].str.contains(search, case=False, na=False)]
    filtered = filtered.head(top_n)

    # Merge per-game stats on team_id
    stats = load_per_game_stats(season)
    filtered = filtered.merge(stats, on="team_id", how="left")

    st.caption(
        f"{len(filtered)} teams shown  |  Efficiency cells: **value  national\_rank**  |  "
        "**AdjEM** = pts/100 margin  |  **AdjO** = off. eff.  |  **AdjD** = def. eff. (lower=better)  |  "
        "**AdjT** = tempo (red=fast, blue=slow)  |  **Luck** = actual W%−expected W%  |  "
        "**OppPPG** = opponent pts/game  |  **TOPG** = turnovers/game"
    )

    # Build display DataFrame — efficiency cols get inline national rank, stat cols are plain numbers.
    # gmap= feeds raw numeric values to the gradient so colors work on string cells.
    display_df = filtered[["Rank", "Team", "Conf", "Record", "ESPN"]].copy()
    display_df["AdjEM"] = filtered.apply(lambda r: f"{r['AdjEM']:+.2f}  {r['AdjEM_nr']}", axis=1)
    display_df["AdjO"]  = filtered.apply(lambda r: f"{r['AdjO']:.2f}  {r['AdjO_nr']}", axis=1)
    display_df["AdjD"]  = filtered.apply(lambda r: f"{r['AdjD']:.2f}  {r['AdjD_nr']}", axis=1)
    display_df["AdjT"]  = filtered.apply(lambda r: f"{r['AdjT']:.1f}  {r['AdjT_nr']}", axis=1)
    display_df["Luck"]  = filtered.apply(lambda r: f"{r['Luck']:+.3f}  {r['Luck_nr']}", axis=1)
    display_df["SOS"]   = filtered.apply(lambda r: f"{r['SOS']:+.2f}  {r['SOS_nr']}", axis=1)
    # Per-game stats — inline national rank, no gradient
    STAT_COLS = ["PPG", "OppPPG", "RebPG", "AstPG", "OrebPG", "TOPG",
                 "FG%", "3P%", "3PaPG", "3PmPG", "FT%", "FTmPG"]
    for col in STAT_COLS:
        display_df[col] = filtered.apply(
            lambda r, c=col: f"{r[c]:.1f}  {int(r[f'{c}_nr'])}" if pd.notna(r[c]) else "—", axis=1
        )

    _PLAIN_COLS = [
        "Rank", "Team", "Conf", "Record", "ESPN",
        "PPG", "OppPPG", "RebPG", "AstPG", "OrebPG", "TOPG",
        "FG%", "3P%", "3PaPG", "3PmPG", "FT%", "FTmPG",
    ]
    # Only include cols that are actually present (ESPN may be absent if espn_id is missing)
    plain_present = [c for c in _PLAIN_COLS if c in display_df.columns]

    styled = (
        display_df.style
        # Light gray background on identity + stat columns (gradient cols override their own bg)
        .set_properties(subset=plain_present, **{"background-color": "#f4f6f8"})
        .background_gradient(subset=["AdjEM"], cmap="RdYlGn",   gmap=filtered["AdjEM"].values)
        .background_gradient(subset=["AdjO"],  cmap="Greens",   gmap=filtered["AdjO"].values)
        # AdjD: lower pts allowed = better defense = green; higher = red
        .background_gradient(subset=["AdjD"],  cmap="RdYlGn_r", gmap=filtered["AdjD"].values)
        # AdjT: fast (high) = red, mid = white, slow (low) = blue
        .background_gradient(subset=["AdjT"],  cmap="RdBu_r",   gmap=filtered["AdjT"].values)
        # Luck: positive (lucky) = green; negative (unlucky) = red
        .background_gradient(subset=["Luck"],  cmap="RdYlGn",   gmap=filtered["Luck"].values)
        # SOS: higher (tougher schedule) = green; lower (easier schedule) = red
        .background_gradient(subset=["SOS"],   cmap="RdYlGn",   gmap=filtered["SOS"].values)
    )

    import re

    # Convert styled df to HTML. escape=False lets us inject anchor tags for ESPN links.
    _table_html = styled.hide(axis="index").to_html(escape=False)

    # Replace raw ESPN URLs in td cells with clickable anchor tags.
    _table_html = re.sub(
        r'(<td[^>]*>)(https://www\.espn\.com/[^<]+)(</td>)',
        lambda m: (
            m.group(1).rstrip('>') + ' style="background-color:#f4f6f8;padding:7px 14px">'
            f'<a href="{m.group(2)}" target="_blank" '
            'style="color:#1565C0;text-decoration:none;font-size:12px;font-weight:500">ESPN</a></td>'
        ),
        _table_html,
    )

    _sticky_css = """
    <style>
    .crp-wrap {
        overflow-x: auto;
        overflow-y: auto;
        max-height: 700px;
        border-radius: 6px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.09);
        font-family: system-ui, -apple-system, sans-serif;
    }
    .crp-wrap table { border-collapse: collapse; white-space: nowrap; font-size: 13px; }
    .crp-wrap thead th {
        position: sticky;
        top: 0;
        z-index: 3;
        background: #1e2d40 !important;
        color: white !important;
        font-weight: 600;
        padding: 9px 14px;
        font-size: 12px;
        letter-spacing: .3px;
        border-bottom: 2px solid #344d66;
        cursor: pointer;
        user-select: none;
    }
    .crp-wrap thead th:hover { background: #2a3f59 !important; }
    .crp-wrap thead th[data-dir='asc']::after  { content: ' ▲'; font-size: 10px; opacity: .8; }
    .crp-wrap thead th[data-dir='desc']::after { content: ' ▼'; font-size: 10px; opacity: .8; }
    .crp-wrap td {
        padding: 7px 14px;
        border-bottom: 1px solid #e3e7ec;
        border-right: 1px solid #e3e7ec;
    }
    /* Sticky Rank column */
    .crp-wrap th:nth-child(1), .crp-wrap td:nth-child(1) {
        position: sticky; left: 0; z-index: 2;
        min-width: 36px; max-width: 44px;
    }
    /* Sticky Team column */
    .crp-wrap th:nth-child(2), .crp-wrap td:nth-child(2) {
        position: sticky; left: 50px; z-index: 2;
        min-width: 160px;
        border-right: 2px solid #b0bbc7 !important;
    }
    /* Corner cells (sticky top + left) need highest z-index */
    .crp-wrap thead th:nth-child(1),
    .crp-wrap thead th:nth-child(2) { z-index: 4; }
    /* Conf column: wide enough for "Southwestern Athletic" etc. */
    .crp-wrap th:nth-child(3), .crp-wrap td:nth-child(3) { min-width: 140px; }
    .crp-wrap tbody tr:hover td { filter: brightness(0.93); }
    </style>
    <script>
    (function() {
      // Tooltip text for each column header abbreviation
      var COL_TIPS = {
        'Rank':   'CarmPom national rank, sorted by AdjEM',
        'Team':   'Team name',
        'Conf':   'Conference',
        'Record': 'Win-loss record (includes conference tournament)',
        'ESPN':   'Link to ESPN team stats page',
        'AdjEM':  'Adjusted Efficiency Margin — points scored minus allowed per 100 possessions, adjusted for opponent strength. The headline ranking stat.',
        'AdjO':   'Adjusted Offensive Efficiency — points scored per 100 possessions, adjusted for opponent defense. Higher is better.',
        'AdjD':   'Adjusted Defensive Efficiency — points allowed per 100 possessions, adjusted for opponent offense. Lower is better.',
        'AdjT':   'Adjusted Tempo — possessions per game. Higher = faster pace.',
        'Luck':   'Luck — actual win% minus Pythagorean (expected) win%. Positive means the team won more close games than expected.',
        'SOS':    'Strength of Schedule — average AdjEM of all opponents faced. Higher means a tougher schedule.',
      };

      function cellVal(td) {
        // Cells like "+24.53  1" or "82.4  3" — sort by the leading numeric token.
        // Falls back to raw text for string columns (Team, Conf, etc.).
        var t = td.innerText.trim();
        var m = t.match(/^([+\-]?[\d.]+)/);
        return m ? parseFloat(m[1]) : t.toLowerCase();
      }
      document.addEventListener('DOMContentLoaded', function() {
        var table = document.querySelector('.crp-wrap table');
        if (!table) return;
        var ths = table.querySelectorAll('thead th');
        ths.forEach(function(th, colIdx) {
          // Set tooltip from the map, falling back to the header text itself
          var label = th.innerText.trim().replace(/[\s▲▼]+$/, '');
          if (COL_TIPS[label]) th.title = COL_TIPS[label];

          th.addEventListener('click', function() {
            var dir = th.getAttribute('data-dir') === 'asc' ? 'desc' : 'asc';
            ths.forEach(function(h) { h.removeAttribute('data-dir'); });
            th.setAttribute('data-dir', dir);
            var tbody = table.querySelector('tbody');
            var rows = Array.from(tbody.querySelectorAll('tr'));
            rows.sort(function(a, b) {
              var av = cellVal(a.cells[colIdx]);
              var bv = cellVal(b.cells[colIdx]);
              if (typeof av === 'number' && typeof bv === 'number') {
                return dir === 'asc' ? av - bv : bv - av;
              }
              return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
            });
            rows.forEach(function(r) { tbody.appendChild(r); });
          });
        });
      });
    })();
    </script>
    """

    st.components.v1.html(
        f"{_sticky_css}<div class='crp-wrap'>{_table_html}</div>",
        height=730,
        scrolling=True,
    )

# ---------------------------------------------------------------------------
# Team profile helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_team_games(team_id: int, season: int) -> pd.DataFrame:
    """Return every game for team_id in season with opponent name, score, and opponent AdjEM."""
    with SessionLocal() as session:
        rows = (
            session.query(
                Game.game_date,
                Game.home_team_id,
                Game.away_team_id,
                Game.home_score,
                Game.away_score,
                Game.neutral_site,
                Game.tournament,
            )
            .filter(
                Game.season == season,
                or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
            )
            .order_by(Game.game_date)
            .all()
        )
        team_names: dict[int, str] = {
            r.id: r.name for r in session.query(Team.id, Team.name).all()
        }
        opp_adjem: dict[int, float] = {
            r.team_id: r.adjem for r in
            session.query(CarmPomRating.team_id, CarmPomRating.adjem)
            .filter(CarmPomRating.season == season).all()
        }
        opp_rank: dict[int, int] = {
            r.team_id: r.rank for r in
            session.query(CarmPomRating.team_id, CarmPomRating.rank)
            .filter(CarmPomRating.season == season).all()
        }

    records = []
    for row in rows:
        is_home = row.home_team_id == team_id
        opp_id = row.away_team_id if is_home else row.home_team_id
        team_score = row.home_score if is_home else row.away_score
        opp_score = row.away_score if is_home else row.home_score
        have_score = team_score is not None and opp_score is not None
        win = have_score and team_score > opp_score
        if row.neutral_site:
            loc = "N"
        elif is_home:
            loc = "vs"
        else:
            loc = "at"
        records.append({
            "date": row.game_date,
            "opponent": team_names.get(opp_id, f"Team {opp_id}"),
            "opp_id": opp_id,
            "team_score": team_score,
            "opp_score": opp_score,
            "result": "W" if win else ("L" if have_score else "?"),
            "margin": (team_score - opp_score) if have_score else None,
            "loc": loc,
            "opp_adjem": opp_adjem.get(opp_id),
            "opp_rank": opp_rank.get(opp_id),
            "tournament": row.tournament or "Regular Season",
        })
    return pd.DataFrame(records)


def generate_playstyle_name(t: pd.Series, ts: pd.Series | None, n: int) -> tuple[str, str]:
    """Return (playstyle_name, one-line tagline) based on stats."""
    adjt_nr  = int(t["AdjT_nr"])
    adjo_nr  = int(t["AdjO_nr"])
    adjd_nr  = int(t["AdjD_nr"])
    fast     = adjt_nr <= 80
    slow     = adjt_nr >= 270
    elite_off = adjo_nr <= 40
    elite_def = adjd_nr <= 40

    three_heavy = three_light = glass_eater = ball_safe = ft_heavy = pass_first = False
    if ts is not None:
        three_pa_nr = int(ts.get("3PaPG_nr",  n))
        oreb_nr     = int(ts.get("OrebPG_nr", n))
        to_nr       = int(ts.get("TOPG_nr",  n))
        ft_nr       = int(ts.get("FTmPG_nr", n))
        ast_nr      = int(ts.get("AstPG_nr", n))
        three_heavy = three_pa_nr <= 50
        three_light = three_pa_nr >= 280
        glass_eater = oreb_nr     <= 40
        ball_safe   = to_nr       <= 50
        ft_heavy    = ft_nr       <= 50
        pass_first  = ast_nr      <= 50

    if elite_def and slow:
        return "⛩️ Defensive Fortress", "Built on suffocating defense and half-court execution"
    if fast and three_heavy and elite_off:
        return "🚀 Perimeter Blitz", "Fires threes at an elite clip and never lets up"
    if fast and elite_off:
        return "⚡ Run & Gun", "Pushes before defenses can set and creates easy looks"
    if slow and elite_def:
        return "🐢 Grind-It-Out", "Slows every possession and suffocates opposing offenses"
    if glass_eater and not three_heavy:
        return "💪 Glass-Eating Machine", "Dominates the offensive boards and scores near the rim"
    if three_heavy and ball_safe:
        return "🎯 Sharpshooter System", "Disciplined ball movement and perimeter shooting"
    if three_heavy:
        return "🌎 Sniper Squad", "Lives behind the arc — win or lose by the three"
    if slow and three_light:
        return "⚒️ Post-Up Bully", "Methodical half-court team that attacks the paint"
    if fast:
        return "🏃 Up-Tempo Pusher", "Plays at a fast clip and looks to score in transition"
    if elite_def:
        return "🛡️ Lockdown Unit", "Defense-first identity that keeps opponents under wraps"
    if elite_off:
        return "🏀 Offensive Juggernaut", "One of the most efficient scoring teams in the country"
    if pass_first and ball_safe:
        return "🎭 Ball Movement Maestros", "Patient, pass-first offense with minimal turnovers"
    return "⚖️ Balanced Contender", "Well-rounded program with no glaring identity or weakness"


def generate_team_writeup(t: pd.Series, ts: pd.Series | None, n: int) -> str:
    """Build a template-driven narrative paragraph describing the team's identity."""
    name = t["Team"]
    rank = int(t["Rank"])
    adjem = float(t["AdjEM"])
    adjo = float(t["AdjO"])
    adjd = float(t["AdjD"])
    adjt = float(t["AdjT"])
    adjt_nr = int(t["AdjT_nr"])
    adjo_nr = int(t["AdjO_nr"])
    adjd_nr = int(t["AdjD_nr"])
    sos_nr  = int(t["SOS_nr"])
    luck    = float(t["Luck"])
    record  = t["Record"]

    # Overall tier
    if rank <= 5:
        tier = f"one of the five best teams in the country"
    elif rank <= 15:
        tier = f"a legitimate national title contender"
    elif rank <= 30:
        tier = f"a strong tournament team"
    elif rank <= 64:
        tier = f"an NCAA Tournament-caliber program"
    elif rank <= 100:
        tier = f"a bubble-level team"
    else:
        tier = f"a team outside the projected field"

    # Offensive identity
    if adjo_nr <= 10:
        off_tier = "one of the most efficient offenses in college basketball"
    elif adjo_nr <= 30:
        off_tier = "an elite offense"
    elif adjo_nr <= 75:
        off_tier = "an above-average offense"
    elif adjo_nr <= 150:
        off_tier = "a middle-of-the-pack offense"
    else:
        off_tier = "an offense that has struggled to generate efficient looks"

    # Defensive identity
    if adjd_nr <= 10:
        def_tier = "one of the stingiest defenses in the nation"
    elif adjd_nr <= 30:
        def_tier = "an elite defense"
    elif adjd_nr <= 75:
        def_tier = "an above-average defense"
    elif adjd_nr <= 150:
        def_tier = "a serviceable but unremarkable defense"
    else:
        def_tier = "a defense that has been a liability"

    # Tempo flavor
    if adjt_nr <= 30:
        tempo_str = f"push the pace as aggressively as anyone in D1 ({adjt:.1f} adj. possessions/40 min, #{adjt_nr} nationally)"
    elif adjt_nr <= 100:
        tempo_str = f"play at an up-tempo pace ({adjt:.1f} adj. possessions/40 min)"
    elif adjt_nr <= 250:
        tempo_str = f"operate at a controlled, deliberate pace ({adjt:.1f} adj. possessions/40 min)"
    else:
        tempo_str = f"slow the game down as much as almost anyone in college basketball ({adjt:.1f} adj. possessions/40 min, #{adjt_nr} nationally)"

    # Shooting style (from per-game stats)
    style_notes = []
    if ts is not None:
        three_pa_nr  = int(ts.get("3PaPG_nr",  n))
        three_pct_nr = int(ts.get("3P%_nr",    n))
        oreb_nr      = int(ts.get("OrebPG_nr", n))
        ft_nr        = int(ts.get("FT%_nr",    n))
        to_nr        = int(ts.get("TOPG_nr",   n))
        three_pct    = float(ts.get("3P%", 0) or 0)

        if three_pa_nr <= 40:
            style_notes.append(
                f"They live behind the arc, attempting threes at one of the highest rates in the country "
                f"(#{three_pa_nr} in 3PA/game)" +
                (f" and shooting {three_pct:.1f}% from deep" if three_pct_nr <= 60 else " though their efficiency from distance has been mixed")
                + "."
            )
        elif three_pa_nr >= 300:
            style_notes.append(
                "They rarely look for the three, preferring to attack inside and get to the line."
            )

        if oreb_nr <= 30:
            style_notes.append(
                f"Crashing the offensive glass is a defining trait — they rank #{oreb_nr} nationally in offensive rebounding, consistently generating second-chance points."
            )

        if to_nr <= 30:
            style_notes.append(
                f"Ball security is a genuine strength: #{to_nr} nationally in turnover rate."
            )
        elif to_nr >= 280:
            style_notes.append(
                f"Turnover issues have been a concern all season (#{to_nr} nationally)."
            )

        if ft_nr <= 25:
            style_notes.append(
                f"They're also an elite free-throw shooting team (#{ft_nr} nationally), a reliable way to protect leads late."
            )

    # Schedule context
    if sos_nr <= 20:
        sos_str = "Their schedule has been among the toughest in the country — every win came against legitimate competition."
    elif sos_nr <= 60:
        sos_str = "They've played a legitimately challenging schedule, which makes their efficiency numbers all the more meaningful."
    elif sos_nr >= 300:
        sos_str = "Strength of schedule has been light — their metrics may face more scrutiny come Selection Sunday."
    else:
        sos_str = ""

    # Luck flag
    luck_str = ""
    if luck > 0.04:
        luck_str = " They've also been somewhat fortunate in close games — their record slightly flatters their underlying efficiency."
    elif luck < -0.04:
        luck_str = " Notably, they've been unlucky in close games — their record undersells how good they actually are."

    # Assemble — bold key numbers for scannability
    parts = [
        f"{name} is **{tier}** this season, sitting at **#{rank} nationally** with a **{record}** record. "
        f"They feature {off_tier} (**#{adjo_nr}**) and {def_tier} (**#{adjd_nr}**), "
        f"for an overall AdjEM of **{adjem:+.2f}**.",

        f"In terms of style, they **{tempo_str}**.",
    ]
    # Bold the rank numbers inside each style note
    for note in style_notes:
        import re as _re
        bolded = _re.sub(r"(#\d+)", r"**\1**", note)
        parts.append(bolded)
    if sos_str:
        parts.append(sos_str)
    if luck_str:
        parts[-1] = parts[-1].rstrip(".?") + luck_str

    return "  \n\n".join(parts)


# ---------------------------------------------------------------------------
# Team Profile tab
# ---------------------------------------------------------------------------

with team_tab:
    _all_teams = load_rankings(2026)
    team_options = _all_teams["Team"].sort_values().tolist()

    # Search input filters the dropdown to matching teams
    _search_query = st.text_input(
        "Search team",
        placeholder="🔍  Type a team name to filter...",
        label_visibility="collapsed",
        key="team_profile_search",
    )
    _filtered_opts = (
        [t for t in team_options if _search_query.strip().lower() in t.lower()]
        if _search_query.strip() else team_options
    ) or team_options  # fall back to full list if nothing matches

    _default_idx = 0
    if not _search_query.strip() and "Duke" in _filtered_opts:
        _default_idx = _filtered_opts.index("Duke")

    selected_team = st.selectbox(
        "Team",
        _filtered_opts,
        index=min(_default_idx, len(_filtered_opts) - 1),
        label_visibility="collapsed",
    )

    _t    = _all_teams[_all_teams["Team"] == selected_team].iloc[0]
    _stats_df = load_per_game_stats(2026)
    _ts_rows  = _stats_df[_stats_df["team_id"] == _t["team_id"]]
    _ts   = _ts_rows.iloc[0] if len(_ts_rows) > 0 else None
    _n    = len(_all_teams)
    _team_id = int(_t["team_id"])

    # ── Seed lookup ──────────────────────────────────────────────────────────
    # CarmPom projected seed: where the S-curve would place this team (top 64 only)
    _cp_rank = int(_t["Rank"])
    _cp_seed_str: str | None = None
    if _cp_rank <= 64:
        _cp_seed_num = ((_cp_rank - 1) // 4) + 1  # S-curve: ranks 1-4 → seed 1, 5-8 → seed 2, etc.
        _cp_seed_str = f"#{_cp_seed_num} seed (CarmPom)"

    # Real bracket seed: from bracket_2026.csv if loaded
    _real_seed_str: str | None = None
    try:
        _brk_loaded = load_real_bracket(2026)
        if _brk_loaded is not None:
            _brk_row = _brk_loaded[_brk_loaded["Team"] == selected_team]
            if not _brk_row.empty:
                _rs = int(_brk_row.iloc[0]["seed"])
                _rr = _brk_row.iloc[0]["region"]
                _real_seed_str = f"#{_rs} seed · {_rr}"
    except Exception:
        pass

    # ── Header ──────────────────────────────────────────────────────────────
    espn_url  = _t["ESPN"]     if pd.notna(_t.get("ESPN"))     else None
    logo_url  = _t["logo_url"] if pd.notna(_t.get("logo_url")) else None

    logo_col, name_col, h2, h3, h4 = st.columns([0.7, 3.5, 1, 1, 1])
    with logo_col:
        if logo_url:
            st.image(logo_url, width=80)
    with name_col:
        st.markdown(f"## {selected_team}")
        _seed_parts = []
        if _real_seed_str:
            _seed_parts.append(f"🏆 {_real_seed_str}")
        if _cp_seed_str:
            _seed_parts.append(_cp_seed_str)
        _seed_line = "  ·  ".join(_seed_parts)
        _subline = f"**{_t['Conf']}** · {_t['Record']} · CarmPom rank **#{_cp_rank}**"
        if _seed_line:
            _subline += f"  ·  {_seed_line}"
        st.markdown(_subline)
    with h2:
        adjem_pct = round((1 - (_t['AdjEM_nr'] - 1) / _n) * 100)
        st.metric("AdjEM", f"{_t['AdjEM']:+.2f}", delta=f"#{int(_t['AdjEM_nr'])} · top {100-adjem_pct+1}%", delta_color="off")
    with h3:
        st.metric("AdjO", f"{_t['AdjO']:.1f}", delta=f"#{int(_t['AdjO_nr'])} off.", delta_color="off")
    with h4:
        st.metric("AdjD", f"{_t['AdjD']:.1f}", delta=f"#{int(_t['AdjD_nr'])} def.", delta_color="off")
    if espn_url:
        st.markdown(f"[View on ESPN ↗]({espn_url})")

    st.divider()

    # ── AI overview + radar chart ────────────────────────────────────────────
    overview_col, radar_col = st.columns([3, 2], gap="large")

    with overview_col:
        st.markdown("#### Team Overview")
        writeup = generate_team_writeup(_t, _ts, _n)
        st.markdown(writeup)

    with radar_col:
        # _pct: convert national rank to percentile (100 = best)
        def _pct(nr: int | float) -> float:
            """Convert a national rank to a 0-100 percentile (higher = better)."""
            return round((1 - (nr - 1) / _n) * 100)

        # ── Playstyle name badge ────────────────────────────────────────────
        _style_name, _style_tag = generate_playstyle_name(_t, _ts, _n)
        st.markdown(
            f"<div style='background:#1e2d40;color:white;border-radius:8px;padding:10px 16px;"
            f"margin-bottom:10px;font-family:system-ui'>"
            f"<div style='font-size:18px;font-weight:700'>{_style_name}</div>"
            f"<div style='font-size:12px;opacity:0.75;margin-top:2px'>{_style_tag}</div></div>",
            unsafe_allow_html=True,
        )

        # ── 8-spoke playstyle radar (no Offense/Defense spokes) ─────────────
        # All spokes describe HOW the team plays, not efficiency ratings.
        _radar_labels = ["Pace", "3PT Volume", "3PT Accuracy", "Off. Rebounding",
                         "Ball Security", "FT Drawing", "Assists", "Def. Rebounding"]

        if _ts is not None:
            _adjt_pct  = _pct(_t["AdjT_nr"])
            _3pa_pct   = _pct(_ts["3PaPG_nr"])
            _3pct_pct  = _pct(_ts["3P%_nr"])
            _oreb_pct  = _pct(_ts["OrebPG_nr"])
            _to_pct    = _pct(_ts["TOPG_nr"])    # rank 1 = fewest TOs
            _ftm_pct   = _pct(_ts["FTmPG_nr"])   # free throws drawn/made per game
            _ast_pct   = _pct(_ts["AstPG_nr"])
            _dreb_pct  = round((1 - (_ts["RebPG_nr"] - 1) / _n) * 100)
        else:
            _adjt_pct = _3pa_pct = _3pct_pct = _oreb_pct = _to_pct = _ftm_pct = _ast_pct = _dreb_pct = 50

        _radar_vals = [_adjt_pct, _3pa_pct, _3pct_pct, _oreb_pct,
                       _to_pct,  _ftm_pct,  _ast_pct,  _dreb_pct]

        N_spokes = len(_radar_labels)
        angles   = [n_i / N_spokes * 2 * 3.14159 for n_i in range(N_spokes)]
        angles  += angles[:1]
        vals     = _radar_vals + _radar_vals[:1]

        fig_r, ax_r = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        ax_r.set_theta_offset(3.14159 / 2)
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(_radar_labels, size=7.5, color="#333")
        ax_r.set_yticks([20, 40, 60, 80, 100])
        ax_r.set_yticklabels(["20", "40", "60", "80", "100"], size=6, color="#aaa")
        ax_r.set_ylim(0, 100)
        ax_r.plot(angles, vals, color="#1e2d40", linewidth=2)
        ax_r.fill(angles, vals, color="#4a90d9", alpha=0.35)
        ax_r.set_title("Playstyle Profile", size=9, pad=14, color="#555", fontweight="bold")
        ax_r.spines["polar"].set_visible(False)
        ax_r.grid(color="#ccc", linestyle="--", linewidth=0.6)
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)
        st.caption("Each spoke = national percentile for that playstyle dimension. Ball Security = inverted turnover rate.")

    st.divider()

    # ── Strengths & Weaknesses ───────────────────────────────────────────────
    st.markdown("#### Strengths & Weaknesses")
    sw_col_a, sw_col_b = st.columns(2)

    # Build a unified list of (label, percentile, is_stat) for all metrics
    _all_metrics: list[tuple[str, float]] = [
        ("Adjusted Offense",  _pct(_t["AdjO_nr"])),
        ("Adjusted Defense",  _pct(_t["AdjD_nr"])),
        ("Tempo / Pace",      _pct(_t["AdjT_nr"])),
        ("Luck / Clutch",     _pct(_t["Luck_nr"])),
        ("Str. of Schedule",  _pct(_t["SOS_nr"])),
    ]
    if _ts is not None:
        _stat_labels = [
            ("Scoring (PPG)",     "PPG_nr",    False),
            ("Scoring Defense",   "OppPPG_nr", False),
            ("3PT Volume",        "3PaPG_nr",  False),
            ("3PT Accuracy",      "3P%_nr",    False),
            ("FT Shooting",       "FT%_nr",    False),
            ("Rebounding",        "RebPG_nr",  False),
            ("Off. Rebounding",   "OrebPG_nr", False),
            ("Ball Security",     "TOPG_nr",   False),
            ("Assists",           "AstPG_nr",  False),
        ]
        for lbl, col, _ in _stat_labels:
            if pd.notna(_ts.get(col)):
                _all_metrics.append((lbl, _pct(int(_ts[col]))))

    _strengths = sorted([m for m in _all_metrics if m[1] >= 75], key=lambda x: -x[1])[:6]
    _weaknesses = sorted([m for m in _all_metrics if m[1] < 40], key=lambda x: x[1])[:6]

    with sw_col_a:
        st.markdown("**💪 Strengths** *(top 25% nationally)*")
        if _strengths:
            for lbl, pct_val in _strengths:
                st.markdown(f"- **{lbl}** — {pct_val}th percentile")
        else:
            st.caption("No standout strengths identified.")

    with sw_col_b:
        st.markdown("**⚠️ Weaknesses** *(bottom 40% nationally)*")
        if _weaknesses:
            for lbl, pct_val in _weaknesses:
                st.markdown(f"- **{lbl}** — {pct_val}th percentile")
        else:
            st.caption("No glaring weaknesses identified.")

    st.divider()

    # ── Efficiency metrics (compact row) ────────────────────────────────────
    st.markdown("#### Full Efficiency Breakdown")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for _col_ui, _metric, _label, _val_fmt, _help in [
        (m1, "AdjEM", "Adj. Efficiency Margin", f"{_t['AdjEM']:+.2f}",
         "Adjusted Efficiency Margin — pts/100 margin vs an average D1 opponent. The headline ranking number."),
        (m2, "AdjO",  "Adj. Offense",           f"{_t['AdjO']:.2f}",
         "Adjusted Offensive Efficiency — points scored per 100 possessions. Higher is better."),
        (m3, "AdjD",  "Adj. Defense",           f"{_t['AdjD']:.2f}",
         "Adjusted Defensive Efficiency — points allowed per 100 possessions. Lower is better."),
        (m4, "AdjT",  "Adj. Tempo",             f"{_t['AdjT']:.1f}",
         "Adjusted Tempo — possessions per 40 minutes. Higher = faster pace."),
        (m5, "Luck",  "Luck",                   f"{_t['Luck']:+.3f}",
         "Actual win% minus Pythagorean expected win%. Positive = winning more than efficiency predicts."),
        (m6, "SOS",   "Strength of Schedule",   f"{_t['SOS']:+.2f}",
         "Average AdjEM of all opponents faced. Higher = tougher schedule."),
    ]:
        with _col_ui:
            _nr = int(_t[f"{_metric}_nr"])
            st.metric(
                _label, _val_fmt,
                delta=f"#{_nr} · {_pct(_nr)}th pct",
                delta_color="off",
                help=_help,
            )

    if _ts is not None:
        st.divider()
        st.markdown("#### Per-Game Stats")
        _STAT_DISPLAY = [
            ("PPG",    "Points per game",         False),
            ("OppPPG", "Opp. pts per game",       True),
            ("FG%",    "Field goal %",            False),
            ("3P%",    "Three-point %",           False),
            ("FT%",    "Free throw %",            False),
            ("RebPG",  "Rebounds per game",       False),
            ("OrebPG", "Off. rebounds per game",  False),
            ("AstPG",  "Assists per game",        False),
            ("TOPG",   "Turnovers per game",      True),
            ("3PaPG",  "3PA per game",            False),
            ("3PmPG",  "3PM per game",            False),
            ("FTmPG",  "FTM per game",            False),
        ]
        stat_rows = []
        for col, label, _ in _STAT_DISPLAY:
            nr = int(_ts[f"{col}_nr"]) if pd.notna(_ts.get(f"{col}_nr")) else None
            pct_s = round((1 - (nr - 1) / _n) * 100) if nr else None
            stat_rows.append({
                "Stat": label,
                "Value": f"{float(_ts[col]):.1f}" if pd.notna(_ts[col]) else "—",
                "Nat'l Rank": f"#{nr}" if nr else "—",
                "Percentile": pct_s if pct_s else 0,
            })

        def _stat_bar_html(rows: list[dict]) -> str:
            """Render a list of stat rows as color-coded HTML bars.

            Green ≥ 75th pct, amber 40–74, red < 40.
            """
            def _bar_color(pct: int) -> str:
                if pct >= 80: return "#2e7d32"
                if pct >= 60: return "#66bb6a"
                if pct >= 40: return "#ffa726"
                if pct >= 20: return "#ef5350"
                return "#b71c1c"

            parts = ["<div style='font-family:system-ui,-apple-system,sans-serif;font-size:13px'>"]
            for sr in rows:
                pct  = int(sr["Percentile"])
                color = _bar_color(pct)
                _nr_str = sr["Nat'l Rank"]
                parts.append(
                    f"<div style='margin-bottom:10px'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
                    f"<span><b>{sr['Stat']}</b>: {sr['Value']}</span>"
                    f"<span style='color:#666;font-size:12px'>{_nr_str} &nbsp; {pct}th pct</span>"
                    f"</div>"
                    f"<div style='background:#e8e8e8;border-radius:4px;height:7px;overflow:hidden'>"
                    f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width .3s'></div>"
                    f"</div></div>"
                )
            parts.append("</div>")
            return "".join(parts)

        _sc_a, _sc_b = st.columns(2)
        _half = len(stat_rows) // 2
        with _sc_a:
            st.markdown(_stat_bar_html(stat_rows[:_half]), unsafe_allow_html=True)
        with _sc_b:
            st.markdown(_stat_bar_html(stat_rows[_half:]), unsafe_allow_html=True)

    st.divider()

    # ── Game history ────────────────────────────────────────────────────────
    _games_df = load_team_games(_team_id, 2026)

    games_big_col, games_recent_col = st.columns(2, gap="large")

    def _fmt_game_row(row: pd.Series) -> dict:
        """Format a game row for display."""
        score = f"{int(row['team_score'])}-{int(row['opp_score'])}" if pd.notna(row['team_score']) else "—"
        margin = f"+{int(row['margin'])}" if (pd.notna(row['margin']) and row['margin'] > 0) else (f"{int(row['margin'])}" if pd.notna(row['margin']) else "")
        opp_rank_str = f" (#{int(row['opp_rank'])}" + (f", {row['opp_adjem']:+.1f})" if pd.notna(row['opp_adjem']) else ")") if pd.notna(row['opp_rank']) else ""
        loc = row["loc"]
        loc_prefix = f"{loc} " if loc in ("vs", "at") else ""
        loc_suffix = " (N)" if loc == "N" else ""
        return {
            "Date": str(row["date"]),
            "Opponent": f"{loc_prefix}{row['opponent']}{loc_suffix}{opp_rank_str}",
            "Result": f"{row['result']} {score} ({margin})" if margin else f"{row['result']} {score}",
        }

    with games_big_col:
        st.markdown("#### 5 Biggest Games")
        st.caption("By opponent CarmPom rank (toughest competition first)")
        if not _games_df.empty:
            _big_games = (
                _games_df[_games_df["opp_rank"].notna()]
                .sort_values("opp_rank")   # rank 1 = toughest opponent
                .head(5)
            )
            if not _big_games.empty:
                _big_rows = [_fmt_game_row(r) for _, r in _big_games.iterrows()]
                st.dataframe(pd.DataFrame(_big_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No ranked opponents found.")
        else:
            st.caption("No game data available.")

    with games_recent_col:
        st.markdown("#### 5 Most Recent Games")
        st.caption("Latest results")
        if not _games_df.empty:
            _recent = _games_df[_games_df["team_score"].notna()].sort_values("date", ascending=False).head(5)
            if not _recent.empty:
                _recent_rows = [_fmt_game_row(r) for _, r in _recent.iterrows()]
                st.dataframe(pd.DataFrame(_recent_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No completed game data available.")
        else:
            st.caption("No game data available.")

# ---------------------------------------------------------------------------
# Bracket Simulation tab
# ---------------------------------------------------------------------------

with bracket_tab:
    # --- Load bracket ---
    try:
        _real_bracket = load_real_bracket(2026)
    except ValueError as _brk_err:
        _real_bracket = None
        st.error(f"bracket_2026.csv error: {_brk_err}")

    if _real_bracket is not None:
        bracket = _real_bracket
        _bracket_mode = "real"
    else:
        _brk_df = load_rankings(2026)
        bracket = build_projected_bracket(_brk_df, n_teams=64)
        _bracket_mode = "projected"

    # Auto-run simulation on first load (cached per bracket type)
    _brk_key = "sim_2026_real" if _bracket_mode == "real" else "sim_2026_proj"
    if _brk_key not in st.session_state:
        with st.spinner("Running 25,000 tournament simulations…"):
            st.session_state[_brk_key] = simulate_bracket(bracket, n_sims=25_000)
    _sim = st.session_state[_brk_key]

    # Load full rankings for national rank columns and playstyle lookup
    _brk_full_r = load_rankings(2026)
    _n_teams = len(_brk_full_r)
    _r_lookup = _brk_full_r.set_index("Team").to_dict("index")

    # Per-game stats joined to team names for variance analysis
    _pg_df = load_per_game_stats(2026)
    _pg_named = _brk_full_r[["team_id", "Team"]].merge(_pg_df, on="team_id", how="left")
    _pg_lu: dict = _pg_named.set_index("Team").to_dict("index")

    # Odds + injury caches (populated by pipeline scripts; {} if not yet fetched)
    _odds_lu: dict = _load_odds_data()
    _inj_lu: dict = _load_injuries_data()

    # ── Matchup card HTML ─────────────────────────────────────────────────
    def _card_html(ta: pd.Series, tb: pd.Series, wp_a: float) -> str:
        """Return HTML card for a single first-round matchup."""
        sa, sb = int(ta["seed"]), int(tb["seed"])
        fav_a = wp_a >= 0.5
        wp_pct_a = round(wp_a * 100)
        wp_pct_b = 100 - wp_pct_a
        bar_left = "#1e7d32" if fav_a else "#c62828"
        bar_right = "#1e7d32" if not fav_a else "#c62828"
        em_a, em_b = float(ta["AdjEM"]), float(tb["AdjEM"])
        em_col_a = "#1e7d32" if em_a > 0 else "#c62828"
        em_col_b = "#1e7d32" if em_b > 0 else "#c62828"
        style_a = "font-weight:700" if fav_a else "color:#555"
        style_b = "font-weight:700" if not fav_a else "color:#555"
        n_a = ta["Team"]
        n_b = tb["Team"]
        _base = (
            f"<div style='border:1px solid #dde3eb;border-radius:10px;padding:12px 14px;"
            f"margin-bottom:10px;background:#fff;box-shadow:0 1px 5px rgba(0,0,0,0.07)'>"
            # Team A row
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>"
            f"<div><span style='background:#1e2d40;color:white;border-radius:3px;padding:1px 6px;"
            f"font-size:10px;font-weight:700;margin-right:5px'>{sa}</span>"
            f"<span style='font-size:13px;{style_a}'>{n_a}</span></div>"
            f"<span style='font-size:12px;color:{em_col_a};font-weight:600'>{em_a:+.1f}</span></div>"
            # Win-prob bar
            f"<div style='display:flex;height:6px;border-radius:3px;overflow:hidden;margin:4px 0'>"
            f"<div style='width:{wp_pct_a}%;background:{bar_left}'></div>"
            f"<div style='width:{wp_pct_b}%;background:{bar_right}'></div></div>"
            # Team B row
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
            f"<div><span style='background:#78909c;color:white;border-radius:3px;padding:1px 6px;"
            f"font-size:10px;font-weight:700;margin-right:5px'>{sb}</span>"
            f"<span style='font-size:13px;{style_b}'>{n_b}</span></div>"
            f"<span style='font-size:12px;color:{em_col_b};font-weight:600'>{em_b:+.1f}</span></div>"
            # Footer: win% only
            f"<div style='display:flex;justify-content:space-between;font-size:11px;color:#777;"
            f"border-top:1px solid #eee;padding-top:5px'>"
            f"<span><b style='color:#333'>{wp_pct_a}%</b> {n_a.split()[0]} &nbsp;"
            f"<b style='color:#333'>{wp_pct_b}%</b> {n_b.split()[0]}</span>"
            f"</div>"   # close footer div
        )
        # ── Optional alert badges ─────────────────────────────────────────
        # 🚑 if either team has injury notes cached
        _has_inj = bool(_inj_lu.get(n_a) or _inj_lu.get(n_b))
        # 📊 if CarmPom vs market gap > 8 percentage points
        _has_edge = False
        _line_a, _ = _odds_for_matchup(n_a, n_b, _odds_lu)
        if _line_a and _line_a.get("impl_prob") is not None:
            _gap = abs(wp_a - float(_line_a["impl_prob"]))
            _has_edge = _gap > 0.08
        if _has_inj or _has_edge:
            _badges = []
            if _has_inj:
                _badges.append("<span style='background:#fff3e0;color:#e65100;border:1px solid #ffb300;"
                               "border-radius:4px;padding:1px 5px;font-size:10px;font-weight:600'>🚑 Injury Alert</span>")
            if _has_edge:
                _badges.append("<span style='background:#e8f5e9;color:#1b5e20;border:1px solid #4caf50;"
                               "border-radius:4px;padding:1px 5px;font-size:10px;font-weight:600'>📊 Model Edge</span>")
            return (
                _base
                + f"<div style='margin-top:6px;display:flex;gap:4px;flex-wrap:wrap'>{''.join(_badges)}</div>"
                + "</div>"   # close outer wrapper
            )
        return _base + "</div>"   # close outer wrapper div

    # ── Matchup detail panel ──────────────────────────────────────────────
    def _detail_panel(ta: pd.Series, tb: pd.Series, wp_a: float, n: int) -> None:
        """Render full matchup analysis inside a bordered container."""
        name_a, name_b = ta["Team"], tb["Team"]
        em_a, em_b = float(ta["AdjEM"]), float(tb["AdjEM"])
        wp_pct_a = round(wp_a * 100)
        wp_pct_b = 100 - wp_pct_a
        fav = name_a if wp_a >= 0.5 else name_b
        fav_pct = max(wp_pct_a, wp_pct_b)

        # Title
        st.markdown(
            f"<h3 style='text-align:center;margin:4px 0 2px;font-family:system-ui,sans-serif'>"
            f"<span style='color:#1e2d40'>({int(ta['seed'])}) {name_a}</span>"
            f"  <span style='color:#aaa;font-size:16px'>vs</span>  "
            f"<span style='color:#1e2d40'>({int(tb['seed'])}) {name_b}</span></h3>",
            unsafe_allow_html=True,
        )

        # Win probability gauge
        st.markdown(
            f"<div style='text-align:center;margin:8px 0 16px'>"
            f"<div style='display:flex;height:20px;border-radius:10px;overflow:hidden;"
            f"max-width:500px;margin:0 auto 6px'>"
            f"<div style='width:{wp_pct_a}%;background:#1e7d32;display:flex;align-items:center;"
            f"justify-content:center'><span style='color:white;font-size:12px;font-weight:700'>{wp_pct_a}%</span></div>"
            f"<div style='width:{wp_pct_b}%;background:#c62828;display:flex;align-items:center;"
            f"justify-content:center'><span style='color:white;font-size:12px;font-weight:700'>{wp_pct_b}%</span></div>"
            f"</div>"
            f"<div style='font-size:12px;color:#555'>CarmPom gives "
            f"<b>{fav}</b> a <b>{fav_pct}%</b> chance to win</div></div>",
            unsafe_allow_html=True,
        )

        # Side-by-side stat comparison
        _left_col, _mid_col, _right_col = st.columns([4, 3, 4])

        def _nr(tname: str, col: str) -> str:
            nr = _r_lookup.get(tname, {}).get(f"{col}_nr")
            return f"#{int(nr)}" if nr is not None else ""

        stat_defs = [
            ("AdjEM",   f"{em_a:+.2f} {_nr(name_a,'AdjEM')}",  f"{em_b:+.2f} {_nr(name_b,'AdjEM')}",  False),
            ("AdjO",    f"{float(ta.get('AdjO',100)):.1f} {_nr(name_a,'AdjO')}",
                        f"{float(tb.get('AdjO',100)):.1f} {_nr(name_b,'AdjO')}", False),
            ("AdjD",    f"{float(ta.get('AdjD',100)):.1f} {_nr(name_a,'AdjD')}",
                        f"{float(tb.get('AdjD',100)):.1f} {_nr(name_b,'AdjD')}", True),
            ("Tempo",   f"{float(ta.get('AdjT',68)):.1f} {_nr(name_a,'AdjT')}",
                        f"{float(tb.get('AdjT',68)):.1f} {_nr(name_b,'AdjT')}", False),
            ("Record",  ta.get("Record","—"), tb.get("Record","—"), False),
        ]

        lh = "<div style='font-family:system-ui,sans-serif;text-align:right'>"
        mh = "<div style='font-family:system-ui,sans-serif;text-align:center;color:#888'>"
        rh = "<div style='font-family:system-ui,sans-serif;text-align:left'>"
        _row = "padding:6px 0;border-bottom:1px solid #eee;font-size:13px"
        for lbl, va, vb, lwr in stat_defs:
            try:
                diff = float(va.split()[0]) - float(vb.split()[0])
                if lwr:
                    diff = -diff
                arr = "◀" if diff > 0.05 else ("▶" if diff < -0.05 else "—")
                ac = "#1e7d32" if diff > 0.05 else ("#c62828" if diff < -0.05 else "#aaa")
            except Exception:
                arr, ac = "—", "#aaa"
            lh += f"<div style='{_row}'><b>{va}</b></div>"
            mh += f"<div style='{_row}'><span style='color:{ac}'>{arr}</span> <span style='font-size:11px'>{lbl}</span></div>"
            rh += f"<div style='{_row}'><b>{vb}</b></div>"
        lh += "</div>"; mh += "</div>"; rh += "</div>"

        with _left_col:
            st.markdown(
                f"<div style='background:#eaf5ed;border-radius:8px;padding:8px 12px;"
                f"text-align:center;margin-bottom:6px'><b>{name_a}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(lh, unsafe_allow_html=True)
        with _mid_col:
            st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
            st.markdown(mh, unsafe_allow_html=True)
        with _right_col:
            st.markdown(
                f"<div style='background:#fdecea;border-radius:8px;padding:8px 12px;"
                f"text-align:center;margin-bottom:6px'><b>{name_b}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(rh, unsafe_allow_html=True)

        # ── Playstyle Profiles ─────────────────────────────────────────────
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        st.markdown("##### 🎨 Playstyle Profiles")

        _ta_full = pd.Series({**_r_lookup.get(name_a, {}), "seed": ta.get("seed", 8), "Team": name_a})
        _tb_full = pd.Series({**_r_lookup.get(name_b, {}), "seed": tb.get("seed", 9), "Team": name_b})

        def _trait_pills(tdata: pd.Series) -> str:
            """Build an HTML row of trait pills from a team's ratings."""
            pills = []
            tnr = int(tdata.get("AdjT_nr", n // 2))
            onr = int(tdata.get("AdjO_nr", n // 2))
            dnr = int(tdata.get("AdjD_nr", n // 2))
            # Pace
            if tnr <= 80:
                pills.append(("🏃 Fast Pace", "#e3f2fd", "#1565c0"))
            elif tnr >= 270:
                pills.append(("🐢 Slow Pace", "#fff3e0", "#e65100"))
            else:
                pills.append(("⚖️ Mid Tempo", "#f5f5f5", "#555"))
            # Offense
            if onr <= 40:
                pills.append(("🔥 Elite Off", "#e8f5e9", "#1b5e20"))
            elif onr <= 100:
                pills.append(("✅ Good Off", "#e8f5e9", "#2e7d32"))
            elif onr <= 230:
                pills.append(("➖ Avg Off", "#fafafa", "#888"))
            else:
                pills.append(("⚠️ Weak Off", "#ffebee", "#b71c1c"))
            # Defense
            if dnr <= 40:
                pills.append(("🔒 Elite Def", "#e8f5e9", "#1b5e20"))
            elif dnr <= 100:
                pills.append(("✅ Good Def", "#e8f5e9", "#2e7d32"))
            elif dnr <= 230:
                pills.append(("➖ Avg Def", "#fafafa", "#888"))
            else:
                pills.append(("⚠️ Weak Def", "#ffebee", "#b71c1c"))
            html = "<div style='display:flex;flex-wrap:wrap;gap:4px;margin:6px 0 8px'>"
            for label, bg, fg in pills:
                html += (
                    f"<span style='background:{bg};color:{fg};border-radius:12px;"
                    f"padding:2px 9px;font-size:11px;font-weight:600'>{label}</span>"
                )
            html += "</div>"
            return html

        _ps_left, _ps_right = st.columns(2, gap="large")
        for _psc, tdata, side_color in [
            (_ps_left, _ta_full, "#1e7d32"),
            (_ps_right, _tb_full, "#c62828"),
        ]:
            with _psc:
                tname = tdata["Team"]
                _tr = _r_lookup.get(tname)
                if _tr:
                    # _r_lookup is keyed by Team name (set_index), so "Team" is absent — re-add it
                    _tr_s = pd.Series({**_tr, "Team": tname})
                    _sname, _stag = generate_playstyle_name(_tr_s, None, n)
                    # Playstyle badge (team-color coded)
                    st.markdown(
                        f"<div style='background:{side_color};color:white;border-radius:8px;"
                        f"padding:9px 14px;text-align:center;margin-bottom:8px'>"
                        f"<div style='font-size:15px;font-weight:700'>{_sname}</div>"
                        f"<div style='font-size:11px;opacity:0.75;font-style:italic;margin-top:2px'>{_stag}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    # Trait pills
                    st.markdown(_trait_pills(_tr_s), unsafe_allow_html=True)
                    # Per-team style prose (2-3 sentences)
                    _style_text = generate_style_profile(_tr_s, n)
                    st.markdown(
                        f"<div style='font-size:13px;color:#444;line-height:1.55'>{_style_text}</div>",
                        unsafe_allow_html=True,
                    )

        # ── Matchup Narrative ──────────────────────────────────────────────
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        st.markdown("##### ⚔️ How They Match Up")
        _clash = generate_clash_narrative(_ta_full, _tb_full, wp_a, n)
        st.markdown(
            f"<div style='background:#f8f9fa;border-left:3px solid #1e2d40;border-radius:0 8px 8px 0;"
            f"padding:12px 16px;font-size:13px;color:#333;line-height:1.6'>{_clash}</div>",
            unsafe_allow_html=True,
        )

        # ── Key Factors ────────────────────────────────────────────────────
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        with st.expander("📋 Key Factors", expanded=True):
            for bullet in generate_matchup_analysis(_ta_full, _tb_full, wp_a, n):
                st.markdown(f"- {bullet}")

        # ── Injury Intel ───────────────────────────────────────────────────
        _notes_a = _inj_lu.get(name_a, [])
        _notes_b = _inj_lu.get(name_b, [])
        if _notes_a or _notes_b:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            with st.expander("🚑 Injury Intel", expanded=True):
                _STATUS_COLORS = {
                    "Out":         ("#c62828", "#ffebee"),
                    "Doubtful":    ("#e65100", "#fff3e0"),
                    "Questionable":("#f57f17", "#fffde7"),
                    "Monitor":     ("#616161", "#f5f5f5"),
                }
                for _tname, _notes in [(name_a, _notes_a), (name_b, _notes_b)]:
                    if not _notes:
                        continue
                    st.markdown(f"**{_tname}**")
                    for _note in _notes[:4]:  # cap at 4 per team
                        _status = _note.get("status", "Monitor")
                        _fc, _bg = _STATUS_COLORS.get(_status, ("#616161", "#f5f5f5"))
                        _headline = _note.get("headline", "")
                        _desc     = _note.get("description", "")
                        _pub      = _note.get("published", "")[:10] if _note.get("published") else ""
                        st.markdown(
                            f"<div style='background:{_bg};border-left:3px solid {_fc};"
                            f"border-radius:4px;padding:7px 10px;margin:4px 0;font-size:13px'>"
                            f"<span style='color:{_fc};font-weight:700;font-size:11px;"
                            f"text-transform:uppercase;letter-spacing:0.5px'>{_status}</span>"
                            f"{'&nbsp;·&nbsp;<span style=\"color:#777;font-size:11px\">' + _pub + '</span>' if _pub else ''}"
                            f"<div style='margin-top:3px;font-weight:600'>{_headline}</div>"
                            f"{'<div style=\"color:#555;margin-top:2px\">' + _desc[:180] + ('…' if len(_desc) > 180 else '') + '</div>' if _desc else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                st.caption("Source: ESPN team news. Refresh via `uv run python pipeline/fetch_injuries.py`.")
        else:
            with st.expander("🚑 Injury Intel", expanded=False):
                st.info(
                    "No injury data in cache. Run `uv run python pipeline/fetch_injuries.py` "
                    "in a terminal to scrape ESPN news for tournament teams.",
                    icon="ℹ️",
                )

        # ── Market vs Model ────────────────────────────────────────────────
        _line_a, _line_b = _odds_for_matchup(name_a, name_b, _odds_lu)
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        with st.expander("📊 Market vs Model", expanded=bool(_line_a)):
            if _line_a and _line_a.get("impl_prob") is not None:
                _book_prob_a = float(_line_a["impl_prob"])
                _book_pct_a  = round(_book_prob_a * 100)
                _book_pct_b  = 100 - _book_pct_a
                _model_pct_a = round(wp_a * 100)
                _model_pct_b = 100 - _model_pct_a
                _gap = _model_pct_a - _book_pct_a   # +ve = CarmPom likes A more than market
                _ml_str     = f"{int(_line_a['ml']):+d}" if _line_a.get("ml") else "N/A"
                _spread_str = (f"{_line_a['spread']:+.1f}" if _line_a.get("spread") is not None else "N/A")
                _books_str  = f"{_line_a.get('book_count', 1)} bookmakers" if _line_a.get("book_count", 1) > 1 else "1 bookmaker"

                _col_m, _col_b = st.columns(2)
                with _col_m:
                    st.metric(f"CarmPom — {name_a}", f"{_model_pct_a}%")
                with _col_b:
                    st.metric(f"Vegas ({_books_str}) — {name_a}", f"{_book_pct_a}%",
                              delta=f"{_gap:+d}pp vs model", delta_color="off")

                _col_m2, _col_b2 = st.columns(2)
                with _col_m2:
                    st.metric(f"CarmPom — {name_b}", f"{_model_pct_b}%")
                with _col_b2:
                    st.metric(f"Vegas — {name_b}", f"{_book_pct_b}%",
                              delta=f"{-_gap:+d}pp vs model", delta_color="off")

                st.markdown(
                    f"<div style='background:#f8f9fa;border-radius:6px;padding:8px 12px;"
                    f"font-size:13px;margin-top:8px'>"
                    f"<b>ML:</b> {name_a} {_ml_str} &nbsp;&nbsp; <b>Spread:</b> {name_a} {_spread_str}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if abs(_gap) >= 5:
                    _value_team  = name_a if _gap > 0 else name_b
                    _value_pct   = abs(_gap)
                    st.markdown(
                        f"<div style='background:#e8f5e9;border-left:3px solid #4caf50;"
                        f"border-radius:4px;padding:8px 12px;margin-top:8px;font-size:13px'>"
                        f"⚡ <b>CarmPom sees value on {_value_team}</b>: model gives them "
                        f"{_value_pct} percentage points more than the market."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("CarmPom and Vegas agree within 5 percentage points — both models see the same game.")
                _fetched_at = ""
                try:
                    import json as _j
                    _meta = _j.loads((ROOT / "data" / "odds_cache.json").read_text())
                    _fetched_at = _meta.get("fetched_at", "")[:16].replace("T", " ")
                except Exception:
                    pass
                if _fetched_at:
                    st.caption(f"Odds last fetched: {_fetched_at} UTC  ·  Refresh: `uv run python pipeline/fetch_odds.py`")
            else:
                st.info(
                    "No odds data in cache. Run `uv run python pipeline/fetch_odds.py` "
                    "to pull live ESPN tournament lines (no API key needed), then refresh.",
                    icon="ℹ️",
                )

        # ── Single-Game Reality Check ──────────────────────────────────────
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        with st.expander("🎲 Single-Game Reality Check", expanded=True):
            st.markdown(
                "<div style='background:#1e2d40;color:white;border-radius:6px;padding:8px 14px;"
                "font-size:13px;margin-bottom:10px'>"
                "The win probability above averages thousands of simulations. "
                "This section explains what can actually flip <b>one game</b> — "
                "because March is decided by individual possessions, not projections."
                "</div>",
                unsafe_allow_html=True,
            )
            _pg_a_data = _pg_lu.get(name_a, {})
            _pg_b_data = _pg_lu.get(name_b, {})
            _sg_bullets = _generate_single_game_bullets(
                _ta_full, _tb_full, _pg_a_data, _pg_b_data, wp_a, _odds_lu, n
            )
            for _b in _sg_bullets:
                st.markdown(f"- {_b}")


    _hdr_l, _hdr_r = st.columns([3, 2])
    with _hdr_l:
        st.subheader("🏆 2026 NCAA Tournament")
        if _bracket_mode == "real":
            st.success("Real bracket seedings loaded.", icon="✅")
        else:
            st.info("Projected bracket from CarmPom rankings.", icon="📅")
    with _hdr_r:
        _n_sims_val = st.select_slider(
            "Simulations", options=[5_000, 10_000, 25_000, 50_000, 100_000],
            value=25_000,
        )
        if st.button("🔄 Re-run", use_container_width=True):
            with st.spinner(f"Running {_n_sims_val:,} simulations…"):
                st.session_state[_brk_key] = simulate_bracket(bracket, n_sims=_n_sims_val)
            st.rerun()

    # ── Championship odds strip ───────────────────────────────────────────
    st.markdown("##### Top Championship Contenders")
    _strip_cols = st.columns(8)
    for _ci, (_, _crow) in enumerate(_sim.head(8).iterrows()):
        with _strip_cols[_ci]:
            st.markdown(
                f"<div style='background:#1e2d40;color:white;border-radius:8px;padding:8px 6px;"
                f"text-align:center;font-family:system-ui,sans-serif;margin-bottom:8px'>"
                f"<div style='font-size:20px;font-weight:800;color:#f9d71c'>{_crow['Champ%']:.1f}%</div>"
                f"<div style='font-size:10px;font-weight:600;line-height:1.3;margin-top:2px'>"
                f"{_crow['Team']}</div>"
                f"<div style='font-size:9px;opacity:0.55;margin-top:2px'>"
                f"{_crow['Region']} · {int(_crow['Seed'])}-seed</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Region tabs with bracket cards ───────────────────────────────────
    _MU_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    _champ_lu = _sim.set_index("Team")["Champ%"].to_dict()

    _brk_east, _brk_west, _brk_south, _brk_mw, _brk_upset = st.tabs(
        ["🗺️ East", "🗺️ West", "🗺️ South", "🗺️ Midwest", "⚡ Upset Watch"]
    )

    for _rtab, _region in [
        (_brk_east, "East"), (_brk_west, "West"),
        (_brk_south, "South"), (_brk_mw, "Midwest")
    ]:
        with _rtab:
            _reg = bracket[bracket["region"] == _region].copy()
            _seed_lu: dict = {int(r["seed"]): r for _, r in _reg.iterrows()}

            # Build matchup option strings for the analyzer
            _mu_opts: list[str] = []
            for _sa, _sb in _MU_PAIRS:
                if _sa in _seed_lu and _sb in _seed_lu:
                    _ta = _seed_lu[_sa]
                    _tb = _seed_lu[_sb]
                    _mu_opts.append(
                        f"({_sa}) {_ta['Team']}  vs  ({_sb}) {_tb['Team']}"
                    )

            # Cards: top half (1/8/5/4) then bottom half (6/3/7/2)
            for _half in [_MU_PAIRS[:4], _MU_PAIRS[4:]]:
                _card_cols = st.columns(4, gap="small")
                for _ci, (_sa, _sb) in enumerate(_half):
                    if _sa not in _seed_lu or _sb not in _seed_lu:
                        continue
                    _ta = _seed_lu[_sa]
                    _tb = _seed_lu[_sb]
                    _wp = _win_prob(float(_ta["AdjEM"]), float(_tb["AdjEM"]))
                    with _card_cols[_ci]:
                        st.markdown(_card_html(_ta, _tb, _wp), unsafe_allow_html=True)

            st.divider()

            # Matchup Analyzer
            st.markdown("##### 🔍 Matchup Analyzer")
            st.caption("Select any first-round game for a full breakdown.")
            _sel_mu = st.selectbox(
                "Matchup",
                options=_mu_opts,
                index=0,
                key=f"mu_sel_{_region}",
                label_visibility="collapsed",
            )
            if _sel_mu and _sel_mu in _mu_opts:
                _idx = _mu_opts.index(_sel_mu)
                _sa_sel, _sb_sel = _MU_PAIRS[_idx]
                _ta_sel = _seed_lu[_sa_sel]
                _tb_sel = _seed_lu[_sb_sel]
                _wp_sel = _win_prob(float(_ta_sel["AdjEM"]), float(_tb_sel["AdjEM"]))
                with st.container(border=True):
                    _detail_panel(_ta_sel, _tb_sel, _wp_sel, _n_teams)

    # ── Upset Watch tab ─────────────────────────────────────────────────────────
    with _brk_upset:
        st.markdown("#### ⚡ Upset Watch — First Round")
        st.caption(
            "Ranked by CarmPom's upset score: how much the underdog outperforms their seed suggests. "
            "March Madness is one-and-done — variance is high and the best team doesn't always win."
        )

        # Collect all 32 first-round matchups and score upset potential
        _upset_rows: list[dict] = []
        for _ur in ["East", "West", "South", "Midwest"]:
            _ureg = bracket[bracket["region"] == _ur].copy()
            _usl: dict = {int(r["seed"]): r for _, r in _ureg.iterrows()}
            for _usa, _usb in _MU_PAIRS:
                if _usa not in _usl or _usb not in _usl:
                    continue
                _uta = _usl[_usa]  # lower seed (favorite)
                _utb = _usl[_usb]  # higher seed (underdog)
                _uwp_b = 1 - _win_prob(float(_uta["AdjEM"]), float(_utb["AdjEM"]))  # underdog wp
                _uem_gap = float(_uta["AdjEM"]) - float(_utb["AdjEM"])
                # Upset score: underdog's win probability, weighted by seed gap
                # A 12-seed with 35% chance is juicier than a 9-seed with 45%
                _seed_gap = _usb - _usa
                _upset_score = _uwp_b * (1 + _seed_gap / 15.0)
                _upset_rows.append({
                    "region": _ur,
                    "fav_seed": _usa, "dog_seed": _usb,
                    "fav": _uta["Team"], "dog": _utb["Team"],
                    "fav_em": float(_uta["AdjEM"]), "dog_em": float(_utb["AdjEM"]),
                    "em_gap": _uem_gap,
                    "dog_wp": round(_uwp_b * 100),
                    "upset_score": _upset_score,
                    "ta": _uta, "tb": _utb,
                })

        _upset_rows.sort(key=lambda x: x["upset_score"], reverse=True)

        for _ur_row in _upset_rows[:9]:
            _us_col, _ud_col = st.columns([1, 12], gap="small")
            with _us_col:
                # Color by tier: top 3 = fire, next 3 = warm, rest = mild
                _tier_color = "#c62828" if _upset_rows.index(_ur_row) < 3 else (
                    "#e65100" if _upset_rows.index(_ur_row) < 6 else "#f9a825"
                )
                st.markdown(
                    f"<div style='background:{_tier_color};color:white;border-radius:8px;"
                    f"padding:6px 4px;text-align:center;font-family:system-ui;margin-top:4px'>"
                    f"<div style='font-size:16px;font-weight:800'>{_ur_row['dog_wp']}%</div>"
                    f"<div style='font-size:9px;opacity:0.85'>upset</div></div>",
                    unsafe_allow_html=True,
                )
            with _ud_col:
                _bullets = generate_matchup_analysis(
                    pd.Series({**_r_lookup.get(_ur_row["fav"], {}), "seed": _ur_row["fav_seed"], "Team": _ur_row["fav"]}),
                    pd.Series({**_r_lookup.get(_ur_row["dog"], {}), "seed": _ur_row["dog_seed"], "Team": _ur_row["dog"]}),
                    1 - _ur_row["dog_wp"] / 100,
                    _n_teams,
                )
                # Pick the most relevant bullet (upset-angle)
                _reason = next(
                    (b for b in _bullets if any(k in b for k in ["upset", "gap", "toss", "Slight", "edge", "mismatch"])),
                    _bullets[0] if _bullets else "",
                )
                # Strip leading emoji/bold markers for inline display
                import re as _re2
                _reason_clean = _re2.sub(r'[*_]{1,2}', "", _reason).strip()
                _reason_clean = _re2.sub(r'^[⚡🏃📊📈⛏️⚔️🛡️👀⚖️❤️]+\s*', '', _reason_clean)
                st.markdown(
                    f"<div style='border:1px solid #dde3eb;border-radius:8px;padding:10px 14px;"
                    f"background:#fff;font-family:system-ui,sans-serif'>"
                    f"<div style='font-size:13px;font-weight:700'>"
                    f"<span style='color:#78909c'>({_ur_row['dog_seed']}) {_ur_row['dog']}</span>"
                    f" <span style='color:#aaa;font-size:11px'>over</span> "
                    f"<span style='color:#1e2d40'>({_ur_row['fav_seed']}) {_ur_row['fav']}</span>"
                    f" &nbsp;<span style='font-size:11px;color:#888'>{_ur_row['region']}</span></div>"
                    f"<div style='font-size:12px;color:#555;margin-top:4px'>"
                    f"AdjEM gap: <b>{_ur_row['em_gap']:+.1f}</b> &nbsp;·&nbsp; "
                    f"Underdog AdjEM: <b>{_ur_row['dog_em']:+.1f}</b> &nbsp;·&nbsp; "
                    f"{_reason_clean}</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='margin:4px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# My Bracket (Pick'em) tab
# ---------------------------------------------------------------------------

with pick_tab:
    # ── Load data ──────────────────────────────────────────────────────────
    _pk_brkt = load_real_bracket()
    if _pk_brkt is None:
        st.warning("Real bracket data not loaded — pick'em unavailable.", icon="⚠️")
    else:
        _pk_df   = load_rankings(2026)
        _pk_lu   = _pk_df.set_index("Team").to_dict("index")   # AdjEM/AdjO/AdjD/etc.

        # ── Session state ──────────────────────────────────────────────────
        if "bp_picks" not in st.session_state:
            st.session_state["bp_picks"] = {}
        _picks: dict = st.session_state["bp_picks"]

        # ── Compute summary stats ──────────────────────────────────────────
        _prob_pct, _made, _exp_correct = _bp_prob_stats(_picks, _pk_brkt, _pk_lu)
        _total_games = 63
        _remaining   = _total_games - _made

        # ── Header ────────────────────────────────────────────────────────
        st.markdown(
            "<h2 style='font-family:system-ui,sans-serif;margin-bottom:0'>🗳️ My Bracket</h2>"
            "<p style='color:#666;font-size:13px;margin-top:2px'>"
            "Pick your bracket — CarmPom tracks the probability that every call lands.</p>",
            unsafe_allow_html=True,
        )

        # ── Metrics strip ─────────────────────────────────────────────────
        _m1, _m2, _m3, _m4 = st.columns(4)

        # Progress bar visual
        _fill_pct = round(_made / _total_games * 100)
        _prog_bar = (
            f"<div style='height:6px;border-radius:3px;background:#eee;margin-top:4px'>"
            f"<div style='width:{_fill_pct}%;height:100%;background:#1e7d32;border-radius:3px'></div></div>"
        )
        with _m1:
            st.markdown(
                f"<div style='background:#fff;border:1px solid #dde3eb;border-radius:10px;"
                f"padding:14px 16px;text-align:center'>"
                f"<div style='font-size:28px;font-weight:800;color:#1e2d40'>{_made}<span style='font-size:14px;color:#888'>/{_total_games}</span></div>"
                f"<div style='font-size:11px;color:#666;margin-top:2px'>Picks Made</div>"
                f"{_prog_bar}</div>",
                unsafe_allow_html=True,
            )
        with _m2:
            if _made == 0:
                _prob_display = "—"
                _prob_sub = "Fill your bracket to see odds"
            else:
                if _prob_pct > 0.01:
                    _prob_display = f"{_prob_pct:.4f}%"
                else:
                    _odds = round(1 / (_prob_pct / 100)) if _prob_pct > 0 else 0
                    _prob_display = f"1 in {_odds:,}"
                _prob_sub = "chance this bracket is perfect"
            st.markdown(
                f"<div style='background:#1e2d40;border-radius:10px;padding:14px 16px;text-align:center'>"
                f"<div style='font-size:20px;font-weight:800;color:#f9d71c'>{_prob_display}</div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.6);margin-top:4px'>{_prob_sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with _m3:
            _exp_str = f"{_exp_correct:.1f}" if _made > 0 else "—"
            st.markdown(
                f"<div style='background:#fff;border:1px solid #dde3eb;border-radius:10px;"
                f"padding:14px 16px;text-align:center'>"
                f"<div style='font-size:28px;font-weight:800;color:#1565c0'>{_exp_str}</div>"
                f"<div style='font-size:11px;color:#666;margin-top:2px'>Expected Correct Picks</div>"
                f"<div style='font-size:10px;color:#aaa'>out of {_made} picked</div></div>",
                unsafe_allow_html=True,
            )
        with _m4:
            _champ_pick = _picks.get((5, 0), "—")
            _champ_em   = _pk_lu.get(_champ_pick, {}).get("AdjEM")
            _champ_label = _champ_pick if _champ_pick != "—" else "Not yet picked"
            _champ_sub   = f"AdjEM {_champ_em:+.2f}" if _champ_em is not None else "Pick a champion below"
            st.markdown(
                f"<div style='background:#fff;border:1px solid #dde3eb;border-radius:10px;"
                f"padding:14px 16px;text-align:center'>"
                f"<div style='font-size:14px;font-weight:700;color:#1e2d40;line-height:1.3'>"
                f"🏆 {_champ_label}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:4px'>{_champ_sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin:12px 0 6px'></div>", unsafe_allow_html=True)

        # ── Auto-fill controls ─────────────────────────────────────────────
        with st.expander("⚙️ Auto-Fill Bracket", expanded=False):
            st.markdown(
                "<p style='font-size:13px;color:#555;margin:0 0 10px'>"
                "Auto-fill all remaining unpicked games based on a strategy. "
                "Your existing picks will be preserved.</p>",
                unsafe_allow_html=True,
            )
            _af1, _af2, _af3, _af4 = st.columns([2, 2, 2, 1])
            with _af1:
                st.markdown(
                    "<div style='background:#e3f2fd;border-radius:8px;padding:8px 12px;font-size:12px;color:#1565c0'>"
                    "<b>🎯 Chalk</b> — Always pick the analytics favorite. Safe, boring, probably wrong.</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Fill with Chalk", use_container_width=True, key="bp_af_chalk"):
                    st.session_state["bp_picks"] = _bp_autofill("chalk", _picks, _pk_brkt, _pk_lu)
                    st.rerun()
            with _af2:
                st.markdown(
                    "<div style='background:#e8f5e9;border-radius:8px;padding:8px 12px;font-size:12px;color:#1b5e20'>"
                    "<b>⚖️ Balanced</b> — Chalk with select upsets where the model shows real value.</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Fill Balanced", use_container_width=True, key="bp_af_bal"):
                    st.session_state["bp_picks"] = _bp_autofill("balanced", _picks, _pk_brkt, _pk_lu)
                    st.rerun()
            with _af3:
                st.markdown(
                    "<div style='background:#fff3e0;border-radius:8px;padding:8px 12px;font-size:12px;color:#e65100'>"
                    "<b>🌪️ Chaos</b> — Aggressively targets underdogs with ≥ 27% win probability.</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Fill Chaos", use_container_width=True, key="bp_af_chaos"):
                    st.session_state["bp_picks"] = _bp_autofill("chaos", _picks, _pk_brkt, _pk_lu)
                    st.rerun()
            with _af4:
                st.markdown(
                    "<div style='background:#ffebee;border-radius:8px;padding:8px 12px;font-size:12px;color:#b71c1c'>"
                    "<b>🔄 Reset</b><br>Clear all picks and start over.</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Reset All", use_container_width=True, key="bp_reset", type="primary"):
                    st.session_state["bp_picks"] = {}
                    st.rerun()

        st.divider()

        # ── Round tabs ────────────────────────────────────────────────────
        _pk_r1, _pk_r2, _pk_r3, _pk_r4, _pk_r5, _pk_r6 = st.tabs([
            f"1st Round ({sum(1 for s in range(32) if (0,s) in _picks)}/32)",
            f"Round of 32 ({sum(1 for s in range(16) if (1,s) in _picks)}/16)",
            f"Sweet 16 ({sum(1 for s in range(8) if (2,s) in _picks)}/8)",
            f"Elite Eight ({sum(1 for s in range(4) if (3,s) in _picks)}/4)",
            f"Final Four ({sum(1 for s in range(2) if (4,s) in _picks)}/2)",
            f"Championship ({sum(1 for s in range(1) if (5,s) in _picks)}/1)",
        ])

        # ── Helper: render one matchup card with pick buttons ──────────────
        def _pk_render_matchup(rnd: int, slot: int, container) -> None:
            """Render the pick card + two buttons for one matchup slot."""
            ta, tb = _bp_candidates(rnd, slot, _picks, _pk_brkt)
            picked = _picks.get((rnd, slot))

            with container:
                if ta == "TBD" or tb == "TBD":
                    st.markdown(
                        f"<div style='border:1px dashed #ccc;border-radius:8px;padding:14px;"
                        f"text-align:center;color:#aaa;font-size:12px;min-height:110px;"
                        f"display:flex;align-items:center;justify-content:center'>"
                        f"<div>⏳ Awaiting<br>prior picks</div></div>",
                        unsafe_allow_html=True,
                    )
                    return

                em_a = float(_pk_lu.get(ta, {}).get("AdjEM", 0))
                em_b = float(_pk_lu.get(tb, {}).get("AdjEM", 0))
                wp_a = _win_prob(em_a, em_b)
                wp_pct_a = round(wp_a * 100)
                wp_pct_b = 100 - wp_pct_a

                # Look up seeds (only meaningful in R1; later show AdjEM rank)
                _sa_data = _pk_brkt[_pk_brkt["Team"] == ta]
                _sb_data = _pk_brkt[_pk_brkt["Team"] == tb]
                sa = int(_sa_data["seed"].values[0]) if len(_sa_data) else "?"
                sb = int(_sb_data["seed"].values[0]) if len(_sb_data) else "?"

                sel_a = picked == ta
                sel_b = picked == tb
                em_col_a = "#1e7d32" if em_a > 0 else "#c62828"
                em_col_b = "#1e7d32" if em_b > 0 else "#c62828"
                op_a = "1.0" if (sel_a or not picked) else "0.45"
                op_b = "1.0" if (sel_b or not picked) else "0.45"
                border_a = "2px solid #1e7d32" if sel_a else "1px solid #e0e0e0"
                border_b = "2px solid #1e7d32" if sel_b else "1px solid #e0e0e0"
                bg_a = "#e8f5e9" if sel_a else "#fafafa"
                bg_b = "#e8f5e9" if sel_b else "#fafafa"

                ta_short = ta[:17] + "…" if len(ta) > 17 else ta
                tb_short = tb[:17] + "…" if len(tb) > 17 else tb

                fav_bar_a = "#1565c0" if wp_pct_a >= wp_pct_b else "#b0bec5"
                fav_bar_b = "#1565c0" if wp_pct_b > wp_pct_a else "#b0bec5"

                st.markdown(
                    f"<div style='font-family:system-ui,sans-serif;font-size:12px;margin-bottom:6px'>"
                    # Team A
                    f"<div style='border:{border_a};border-radius:7px;padding:6px 8px;"
                    f"background:{bg_a};opacity:{op_a};margin-bottom:3px;"
                    f"display:flex;justify-content:space-between;align-items:center'>"
                    f"<div><span style='background:#1e2d40;color:white;border-radius:3px;"
                    f"padding:1px 5px;font-size:9px;font-weight:700;margin-right:5px'>{sa}</span>"
                    f"<span style='font-weight:{'700' if sel_a else '400'}'>"
                    f"{'✅ ' if sel_a else ''}{ta_short}</span></div>"
                    f"<span style='color:{em_col_a};font-size:11px;font-weight:600'>{em_a:+.1f}</span>"
                    f"</div>"
                    # Probability bar + labels
                    f"<div style='display:flex;height:5px;border-radius:3px;overflow:hidden;margin:2px 2px'>"
                    f"<div style='width:{wp_pct_a}%;background:{fav_bar_a}'></div>"
                    f"<div style='width:{wp_pct_b}%;background:{fav_bar_b}'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:9px;"
                    f"color:#999;padding:1px 2px 3px'>"
                    f"<span><b style='color:#555'>{wp_pct_a}%</b></span>"
                    f"<span><b style='color:#555'>{wp_pct_b}%</b></span></div>"
                    # Team B
                    f"<div style='border:{border_b};border-radius:7px;padding:6px 8px;"
                    f"background:{bg_b};opacity:{op_b};"
                    f"display:flex;justify-content:space-between;align-items:center'>"
                    f"<div><span style='background:#78909c;color:white;border-radius:3px;"
                    f"padding:1px 5px;font-size:9px;font-weight:700;margin-right:5px'>{sb}</span>"
                    f"<span style='font-weight:{'700' if sel_b else '400'}'>"
                    f"{'✅ ' if sel_b else ''}{tb_short}</span></div>"
                    f"<span style='color:{em_col_b};font-size:11px;font-weight:600'>{em_b:+.1f}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                # Pick buttons
                _ba, _bundo, _bb = st.columns([5, 1, 5])
                with _ba:
                    if st.button(
                        ta_short, key=f"bp_{rnd}_{slot}_a",
                        use_container_width=True,
                        type="primary" if sel_a else "secondary",
                    ):
                        st.session_state["bp_picks"][(rnd, slot)] = ta
                        st.rerun()
                with _bundo:
                    if picked and st.button("✕", key=f"bp_{rnd}_{slot}_x", use_container_width=True):
                        del st.session_state["bp_picks"][(rnd, slot)]
                        # Cascade-clear all downstream picks derived from this slot.
                        # In each subsequent round R, the slot that used this result sits at slot//2^(R-rnd).
                        _cur_slot = slot
                        for _dr in range(rnd + 1, 6):
                            _cur_slot = _cur_slot // 2
                            st.session_state["bp_picks"].pop((_dr, _cur_slot), None)
                        st.rerun()
                with _bb:
                    if st.button(
                        tb_short, key=f"bp_{rnd}_{slot}_b",
                        use_container_width=True,
                        type="primary" if sel_b else "secondary",
                    ):
                        st.session_state["bp_picks"][(rnd, slot)] = tb
                        st.rerun()

        # ── Round 1 (32 games, 4 regions × 8) ──────────────────────────────
        with _pk_r1:
            st.caption("Pick every first-round winner. All 32 matchups are live — start anywhere.")
            for _ri, _region in enumerate(_BP_REGIONS):
                st.markdown(f"**{_region} Region**")
                _cols = st.columns(4, gap="small")
                for _mi in range(8):
                    _slot = _ri * 8 + _mi
                    _pk_render_matchup(0, _slot, _cols[_mi % 4])
                    if _mi == 3:
                        _cols = st.columns(4, gap="small")
                st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

        # ── Round 2 (16 games, 4 regions × 4) ─────────────────────────────
        with _pk_r2:
            st.caption("Round of 32 — your Round 1 picks determine the matchups.")
            for _ri, _region in enumerate(_BP_REGIONS):
                st.markdown(f"**{_region} Region**")
                _cols32 = st.columns(4, gap="small")
                for _mi in range(4):
                    _slot = _ri * 4 + _mi
                    _pk_render_matchup(1, _slot, _cols32[_mi])
                st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

        # ── Sweet 16 (8 games, 4 regions × 2) ─────────────────────────────
        with _pk_r3:
            st.caption("Sweet 16 — four regions, two games each.")
            _s16_cols = st.columns(4, gap="medium")
            for _ri, _region in enumerate(_BP_REGIONS):
                with _s16_cols[_ri]:
                    st.markdown(f"**{_region}**")
                    for _mi in range(2):
                        _slot = _ri * 2 + _mi
                        _pk_render_matchup(2, _slot, st.container())

        # ── Elite Eight (4 games, one per region) ─────────────────────────
        with _pk_r4:
            st.caption("Elite Eight — regional champions. Four teams advance to the Final Four.")
            _e8_cols = st.columns(4, gap="medium")
            for _ri, _region in enumerate(_BP_REGIONS):
                with _e8_cols[_ri]:
                    st.markdown(f"**{_region}**")
                    _pk_render_matchup(3, _ri, st.container())

        # ── Final Four (2 games) ───────────────────────────────────────────
        with _pk_r5:
            st.caption("Final Four — East/West and South/Midwest bracket halves meet in Houston.")
            _f4l, _f4_spacer, _f4r = st.columns([5, 1, 5])
            with _f4l:
                st.markdown("**East vs West**")
                _pk_render_matchup(4, 0, st.container())
            with _f4r:
                st.markdown("**South vs Midwest**")
                _pk_render_matchup(4, 1, st.container())

        # ── Championship ───────────────────────────────────────────────────
        with _pk_r6:
            st.caption("National Championship — one pick to seal the bracket.")
            _ncg_l, _ncg_m, _ncg_r = st.columns([2, 4, 2])
            with _ncg_m:
                st.markdown("### 🏆 National Championship")
                _pk_render_matchup(5, 0, st.container())

                # Final probability ribbon if bracket is complete
                if _made == 63:
                    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                    if _prob_pct > 0.01:
                        _ribbon_prob = f"{_prob_pct:.6f}%"
                        _ribbon_odds = ""
                    else:
                        _ribbon_odds_val = round(1 / (_prob_pct / 100)) if _prob_pct > 0 else 0
                        _ribbon_prob = f"1 in {_ribbon_odds_val:,}"
                        _ribbon_odds = ""
                    st.markdown(
                        f"<div style='background:#1e2d40;color:white;border-radius:12px;"
                        f"padding:20px 24px;text-align:center;font-family:system-ui'>"
                        f"<div style='font-size:13px;opacity:0.7;margin-bottom:6px'>Your completed bracket has a</div>"
                        f"<div style='font-size:32px;font-weight:800;color:#f9d71c'>{_ribbon_prob}</div>"
                        f"<div style='font-size:13px;opacity:0.7;margin-top:6px'>probability of being perfect.</div>"
                        f"<div style='font-size:12px;opacity:0.5;margin-top:4px'>"
                        f"Expected correct picks: <b style='color:white'>{_exp_correct:.1f}</b> out of 63</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

with about_tab:
    st.markdown("""
    ### What is CarmPom?

    CarmPom is a KenPom-style NCAA Men's Basketball analytics platform built in Python.

    | Metric | Description |
    |--------|-------------|
    | **AdjEM** | Adjusted Efficiency Margin — pts scored minus pts allowed per 100 possessions, adjusted for strength of schedule. The headline ranking number. |
    | **AdjO** | Adjusted Offensive Efficiency — points scored per 100 possessions vs. an average D1 defense. Higher is better. |
    | **AdjD** | Adjusted Defensive Efficiency — points allowed per 100 possessions vs. an average D1 offense. **Lower is better.** |
    | **AdjT** | Adjusted Tempo — possessions per 40 minutes, adjusted for opponent pace. Higher = faster. |
    | **Luck** | Actual win% minus Pythagorean expected win%. Positive = winning more than efficiency predicts. |
    | **SOS** | Strength of Schedule — average AdjEM of all opponents faced. |

    Ratings are computed using an iterative adjustment loop similar to KenPom's methodology,
    trained on D1 game data from 2003–present.
    """)
