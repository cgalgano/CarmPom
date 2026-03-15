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
    If you've ever looked up a team on KenPom before a big game, you already know the drill:
    win-loss records don't tell the whole story. **Beating bad teams by 30 looks the same as
    barely squeaking past them** — and neither tells you much about March.

    CarmPom uses the same idea as KenPom: rank every team by how efficiently they score
    and defend, adjusted for the strength of who they played. The headline number is **AdjEM**
    — the margin in points per 100 possessions once you account for schedule. The higher, the better.

    Beyond ratings, CarmPom runs a machine learning model trained on 20+ years of NCAA
    Tournament results to predict game outcomes — so you can see not just who's ranked higher,
    but how likely they are to actually win.
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


def _win_prob(adjem_a: float, adjem_b: float) -> float:
    """Logistic win probability from AdjEM differential.

    Coefficient 0.175 calibrated so a +10 AdjEM edge ≈ 82% win probability,
    roughly matching historical NCAA Tournament margins.
    """
    return 1.0 / (1.0 + math.exp(-0.175 * (adjem_a - adjem_b)))


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


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

rankings_tab, team_tab, bracket_tab, kenpom_tab, about_tab = st.tabs(
    ["📊 Team Rankings", "🏀 Team Profile", "🏆 Bracket Sim", "🆚 vs KenPom", "ℹ️ About"]
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

    # ── Header ──────────────────────────────────────────────────────────────
    espn_url  = _t["ESPN"]     if pd.notna(_t.get("ESPN"))     else None
    logo_url  = _t["logo_url"] if pd.notna(_t.get("logo_url")) else None

    logo_col, name_col, h2, h3, h4 = st.columns([0.7, 3.5, 1, 1, 1])
    with logo_col:
        if logo_url:
            st.image(logo_url, width=80)
    with name_col:
        st.markdown(f"## {selected_team}")
        st.markdown(f"**{_t['Conf']}** · {_t['Record']} · CarmPom rank **#{int(_t['Rank'])}**")
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
        return {
            "Date": str(row["date"]),
            "Opponent": f"{row['loc']} {row['opponent']}{opp_rank_str}",
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
    st.subheader("🏆 Bracket Simulation")

    st.info(
        "The 2026 NCAA Tournament bracket is announced on **Selection Sunday, March 16**. "
        "Until then this tab shows a **projected bracket** using CarmPom's current rankings as seeds. "
        "Once the real bracket drops, seedings will be updated.",
        icon="📅",
    )

    _brk_df = load_rankings(2026)
    bracket = build_projected_bracket(_brk_df, n_teams=64)

    n_sims = st.slider(
        "Simulations", min_value=5_000, max_value=100_000, value=25_000, step=5_000,
        help="More simulations = smoother probabilities, slightly slower to compute."
    )

    if st.button("▶  Run simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournament simulations…"):
            sim_results = simulate_bracket(bracket, n_sims=n_sims)
        st.session_state["sim_results"] = sim_results

    if "sim_results" in st.session_state:
        sim_results = st.session_state["sim_results"]

        st.divider()
        st.markdown("#### Championship probability — all 64 teams")
        st.caption(
            "Percentages = how often each team reached that round across all simulations.  "
            "\nProjected seeds based on current CarmPom rankings (S-curve seeding)."
        )

        # Build a styled DataFrame for the simulation results
        _sim_display = sim_results[[
            "Team", "Conf", "Record", "Region", "Seed", "CarmPomRk", "AdjEM",
            "R64%", "R32%", "S16%", "E8%", "F4%", "Champ%"
        ]].copy()
        _sim_display["AdjEM"] = _sim_display["AdjEM"].map(lambda x: f"{x:+.2f}")

        st.dataframe(
            _sim_display.style
                .background_gradient(subset=["R64%", "R32%", "S16%", "E8%", "F4%", "Champ%"],
                                     cmap="YlGn")
                .format({"R64%": "{:.1f}%", "R32%": "{:.1f}%", "S16%": "{:.1f}%",
                         "E8%": "{:.1f}%", "F4%": "{:.1f}%", "Champ%": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
            height=650,
        )

        st.divider()
        st.markdown("#### Projected bracket by region")
        _tab_east, _tab_west, _tab_south, _tab_mw = st.tabs(["East", "West", "South", "Midwest"])
        for _rtab, _rname in [
            (_tab_east, "East"), (_tab_west, "West"),
            (_tab_south, "South"), (_tab_mw, "Midwest")
        ]:
            with _rtab:
                _region_df = bracket[bracket["region"] == _rname].sort_values("seed").copy()
                _region_display = _region_df[["seed", "Team", "Conf", "Record", "Rank", "AdjEM"]].rename(
                    columns={"seed": "Seed", "Rank": "CarmPomRk"}
                )
                _region_display["AdjEM"] = _region_display["AdjEM"].map(lambda x: f"{x:+.2f}")
                # Pull Champ% from sim results for this region
                _champ_lookup = sim_results.set_index("Team")["Champ%"].to_dict()
                _region_display["Champ%"] = _region_display["Team"].map(
                    lambda t: f"{_champ_lookup.get(t, 0):.1f}%"
                )
                st.dataframe(_region_display, use_container_width=True, hide_index=True)

    else:
        st.markdown("Press **▶ Run simulation** to generate championship probabilities.")
        st.divider()
        st.markdown("#### Projected seedings (pre-bracket, S-curve from CarmPom rankings)")
        _seed_display = bracket[["region", "seed", "Team", "Conf", "Record", "Rank", "AdjEM"]].rename(
            columns={"region": "Region", "seed": "Seed", "Rank": "CarmPomRk"}
        ).sort_values(["Region", "Seed"])
        _seed_display["AdjEM"] = _seed_display["AdjEM"].map(lambda x: f"{x:+.2f}")
        st.dataframe(_seed_display, use_container_width=True, hide_index=True, height=500)

# ---------------------------------------------------------------------------
# KenPom comparison tab
# ---------------------------------------------------------------------------

with kenpom_tab:
    st.subheader("CarmPom 2026 vs KenPom 2026 — Rank Disagreements")
    st.caption(
        "Positive (green) = CarmPom ranks the team **higher** than KenPom.  "
        "Negative (red) = CarmPom ranks the team **lower**.  "
        "Filtered to CarmPom's top 100."
    )

    compare_df = load_kenpom_comparison(2026)
    top100_cmp = compare_df[compare_df["cp_rank"] <= 100].copy()

    col_n2, col_conf2 = st.columns([1, 2])
    with col_n2:
        n_bars = st.slider("Teams per side", min_value=10, max_value=25, value=20)
    with col_conf2:
        confs2 = ["All conferences"] + sorted(top100_cmp["Conf"].dropna().unique().tolist())
        selected_conf2 = st.selectbox("Filter conference", confs2, key="kp_conf")

    filtered_cmp = top100_cmp.copy()
    if selected_conf2 != "All conferences":
        filtered_cmp = filtered_cmp[filtered_cmp["Conf"] == selected_conf2]

    plot_disc = pd.concat([
        filtered_cmp.nlargest(n_bars, "rank_diff"),
        filtered_cmp.nsmallest(n_bars, "rank_diff"),
    ]).drop_duplicates(subset="team_id").sort_values("rank_diff")

    bar_colors = ["#4CAF50" if v > 0 else "#F44336" for v in plot_disc["rank_diff"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_disc) * 0.38)))
    bars = ax.barh(plot_disc["Team"], plot_disc["rank_diff"], color=bar_colors, edgecolor="white", height=0.72)

    for bar, (_, row) in zip(bars, plot_disc.iterrows()):
        x = bar.get_width()
        label = f"CP #{int(row['cp_rank'])} ({row['cp_adjem']:+.1f})  /  KP #{int(row['kp_rank'])}"
        ax.text(
            x + (0.4 if x >= 0 else -0.4),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left" if x >= 0 else "right",
            fontsize=7.5, color="#333",
        )

    ax.axvline(0, color="#555", linewidth=1.0)
    ax.set_xlabel("KenPom rank − CarmPom rank  |  Positive = CarmPom ranks higher", fontsize=10)
    ax.set_title(
        "CarmPom 2026 vs KenPom 2026 — Biggest Rank Disagreements (Top 100)\n"
        "Green = CarmPom ranks higher  |  Red = CarmPom ranks lower",
        fontsize=12, fontweight="bold", pad=12,
    )
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.markdown("#### Full disagreement table (top 100)")

    table_cols = ["Team", "Conf", "cp_rank", "cp_adjem", "kp_rank", "rank_diff"]
    display_cmp = (
        top100_cmp[table_cols]
        .rename(columns={"cp_rank": "CarmPom Rk", "cp_adjem": "CarmPom AdjEM",
                         "kp_rank": "KenPom Rk", "rank_diff": "Rank Diff"})
        .sort_values("Rank Diff", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        display_cmp.style
            .background_gradient(subset=["Rank Diff"], cmap="RdYlGn")
            .format({"CarmPom AdjEM": "{:+.2f}", "Rank Diff": "{:+d}"}),
        use_container_width=True,
        hide_index=True,
        height=500,
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
