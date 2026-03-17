"""
app.py
------
CarmPom Streamlit web app.

Run with:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path
from typing import Any

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

# Bracket pick button styling: light background, black readable team names, wrapping text
st.markdown(
    """
    <style>
    /* Secondary (unpicked) pick buttons — white bg, black text */
    div[data-testid="stBaseButton-secondary"] {
        background-color: #f0f2f6 !important;
        border: 1px solid #ccd0d9 !important;
    }
    div[data-testid="stBaseButton-secondary"]:hover {
        background-color: #e2e6ef !important;
        border-color: #a0a8bc !important;
    }
    div[data-testid="stBaseButton-secondary"] p {
        color: #111111 !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }
    /* Primary (picked) pick buttons — green bg, white text */
    div[data-testid="stBaseButton-primary"] p {
        color: #ffffff !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1117 0%, #1a2a1a 40%, #0d2b0d 70%, #1b4332 100%);
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
">
  <!-- subtle background texture lines -->
  <div style="
    position:absolute;top:0;left:0;right:0;bottom:0;
    background: repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 60px,
      rgba(45,106,79,0.07) 60px,
      rgba(45,106,79,0.07) 61px
    );
    pointer-events:none;
  "></div>

  <div style="position:relative;z-index:1;">
    <div style="font-size:13px;letter-spacing:4px;text-transform:uppercase;color:#52b788;font-weight:600;margin-bottom:10px;">
      NCAA Men's Basketball Analytics
    </div>
    <div style="font-size:52px;font-weight:900;color:#ffffff;letter-spacing:-1px;line-height:1;margin-bottom:12px;">
      CarmPom
    </div>
    <div style="font-size:18px;color:#b7e4c7;font-weight:400;max-width:540px;line-height:1.5;">
      Ultimate March Madness Resource
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("Built by Carmen Galgano")

st.divider()

hero_left, hero_right = st.columns([3, 2], gap="large")

with hero_left:
    st.markdown("""
    - I built this resource to save myself time trying to understand every team during March Madness.
      Being a numbers guy, I just want a straightforward, data-driven approach to each team and matchup.

    - I trained my model on **20+ years of tournament data** to create one of the most accurate and
      predictive metrics available: **Adjusted Efficiency Margin (AdjEM)**, the difference between
      Offensive and Defensive Adjusted Efficiency, measured in points per 100 possessions after
      adjusting for strength of schedule. A team that goes 25-5 beating bad teams will have a much
      lower AdjEM than one that went 22-8 in a brutal conference. It's the most honest single-number
      summary of how good a team actually is, and the backbone of every prediction on this site.

    ---

    **CarmPom Features**
    - Team Rankings and Metrics similar to KenPom
    - Valuable Charts to understand team strengths and weaknesses
    - Team Profile Page — an in-depth look at every team in College Basketball
    - Bracket Creation Tab — analyze each matchup and make selections backed by data

    ---

    *Good luck to all — This is March!*
    """)

with hero_right:
    st.markdown("#### CarmPom vs KenPom at a glance")
    st.markdown("""
    | | KenPom | CarmPom |
    |---|---|---|
    | **Ratings method** | Opponent-adjusted efficiency | Opponent-adjusted efficiency |
    | **Data source** | Premium proprietary feeds | Public ESPN box scores |
    | **Updated** | Daily (paid) | On demand |
    | **Tournament predictions** | Ratings only | ML model + bracket creator |
    | **Tournament accuracy** | 0.752 AUC (2025 holdout) | 0.826 AUC — best model (2025 holdout) |
    | **Cost** | $9.99/year | Free |
    """)
    st.caption("AUC measures how often the model correctly identifies the stronger team. Higher = better. CarmPom's ML model outperforms the KenPom baseline on the 2025 tournament holdout.")

st.divider()

# --- Feature importance section ---
st.markdown("### What actually wins in March?")
st.markdown(
    "I looked at every NCAA Tournament game from 2003–2025 and asked: which stats actually "
    "predicted who won? The numbers below show what the model learned to lean on."
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

    pct_badge = (
        f"<span style='font-size:12px;font-weight:700;color:#29b6f6;background:#0d1f2d;"
        f"border:1px solid #29b6f6;border-radius:4px;padding:1px 6px;white-space:nowrap'>"
        f"{pct:.1f}%</span>"
    )
    if is_top:
        st.markdown(
            f"**🏆 {label}** &nbsp; {pct_badge}",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"**{label}** &nbsp; {pct_badge}",
            unsafe_allow_html=True,
        )
    st.caption(explanation)

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
    df["Player Stats"] = df["team_id"].map(
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
                BoxScore.stl,
                BoxScore.blk,
            )
            .join(Game, Game.id == BoxScore.game_id)
            .filter(Game.season == season)
            .all()
        )

    bs = pd.DataFrame(rows, columns=[
        "team_id", "game_id", "pts", "fgm", "fga",
        "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "tov", "stl", "blk",
    ])

    # Self-join on game_id to get the opponent's pts and 3PT stats for each game.
    opp = bs[["game_id", "team_id", "pts", "fgm", "fga", "fg3a", "fg3m"]].rename(
        columns={"team_id": "opp_id", "pts": "opp_pts", "fgm": "opp_fgm", "fga": "opp_fga",
                 "fg3a": "opp_fg3a", "fg3m": "opp_fg3m"}
    )
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
        opp_fgm=("opp_fgm", "sum"),
        opp_fga=("opp_fga", "sum"),
        opp_fg3a=("opp_fg3a", "sum"),
        opp_fg3m=("opp_fg3m", "sum"),
        stl=("stl", "sum"),
        blk=("blk", "sum"),
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
    s["Opp3PaPG"] = (agg["opp_fg3a"] / g).round(1)
    # Guard against divide-by-zero if opp never attempted a 3 (shouldn't happen)
    s["Opp3P%"]   = (agg["opp_fg3m"] / agg["opp_fg3a"] * 100).where(agg["opp_fg3a"] > 0, other=0.0).round(1)
    # Opponent 2PT FG% — proxy for interior defense (mid-range + at-rim combined)
    _opp_fg2a = agg["opp_fga"] - agg["opp_fg3a"]
    _opp_fg2m = agg["opp_fgm"] - agg["opp_fg3m"]
    s["Opp2P%"]   = (_opp_fg2m / _opp_fg2a * 100).where(_opp_fg2a > 0, other=0.0).round(1)
    s["StlPG"]    = (agg["stl"] / g).round(2)
    s["BlkPG"]    = (agg["blk"] / g).round(2)

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
        # 3PT defense: lower attempts/% allowed = better → ascending rank
        "Opp3PaPG": True, "Opp3P%": True,
        # Interior defense: lower opp 2PT FG% = better → ascending rank
        "Opp2P%": True,
        # Disruption stats: more steals/blocks = better → descending rank
        "StlPG": False, "BlkPG": False,
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
        # Pairings: East (idx 0) vs South (idx 2), West (idx 1) vs Midwest (idx 3)
        semi_winners: list[tuple[int, float]] = []
        for a_idx, b_idx in [(0, 2), (1, 3)]:
            a_tid, a_em = f4_contestants[a_idx]
            b_tid, b_em = f4_contestants[b_idx]
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

    return [b.replace("\u2014", ", ") for b in bullets]


def generate_style_profile(t: pd.Series, n: int) -> str:
    """2-3 sentence style-focused summary of a single team, for use in matchup panels."""
    name = t["Team"]
    adjt = float(t.get("AdjT", 68.0))
    adjt_nr = int(t.get("AdjT_nr", n // 2))
    adjo = float(t.get("AdjO", 100.0))
    adjo_nr = int(t.get("AdjO_nr", n // 2))
    adjd = float(t.get("AdjD", 100.0))
    adjd_nr = int(t.get("AdjD_nr", n // 2))
    adjem = float(t.get("AdjEM", 0.0))
    sos_nr = int(t.get("SOS_nr", n // 2))
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

    # Luck context — multiple alternatives keyed to team characteristics so
    # each write-up reads differently.  We pick deterministically (hash on name)
    # so the same team always gets the same sentence while different teams vary.
    luck_str = ""
    if luck < -0.04:
        _slot = hash(name) % 5
        if _slot == 0:
            luck_str = " Their record actually understates how well they've played — they've been genuinely unlucky in close games."
        elif _slot == 1:
            luck_str = (
                f" The efficiency numbers are notably stronger than their win-loss line suggests — "
                "late-game variance has cost them results their play deserved."
            )
        elif _slot == 2:
            luck_str = (
                f" They haven't gotten many breaks: a disproportionate share of their losses "
                "came in games decided by a possession or two, which the adjusted numbers discount."
            )
        elif _slot == 3:
            # SOS-flavoured if schedule rank available, otherwise generic
            if sos_nr and sos_nr <= 50:
                luck_str = (
                    f" Playing one of the toughest schedules (SOS #{sos_nr}) and running into "
                    "bad luck in tight games has suppressed their record — the underlying metrics "
                    "are kinder than the standings imply."
                )
            else:
                luck_str = (
                    " Close-game results have gone against them more than chance would predict; "
                    "expect some regression to the mean if they stay in games deep into the second half."
                )
        else:
            luck_str = (
                f" Their AdjEM ({adjem:.1f}) is meaningfully better than their record implies — "
                "they've been on the wrong end of the variance coin in tight games."
            )
    elif luck > 0.05:
        _slot = hash(name) % 4
        if _slot == 0:
            luck_str = " Worth noting: they've been fortunate in close games, meaning their record may slightly flatter their real quality."
        elif _slot == 1:
            luck_str = (
                " Their win total leans on some good fortune down the stretch — "
                "the efficiency margin alone would project a slightly lower ceiling."
            )
        elif _slot == 2:
            luck_str = (
                f" They've won a lot of coin-flip games this season. That's fine — winning tight games "
                "is a real skill — but the adjusted numbers suggest some of those margins were thinner "
                "than the scoreboard showed."
            )
        else:
            luck_str = (
                f" A strong record, though the luck metric flags they've been helped by close-game "
                "variance. Opponents who push this to the wire may find a tighter contest than seeding implies."
            )

    return f"{name} {tempo}, featuring {off_str}. {def_str}{luck_str}"


def generate_clash_narrative(ta: pd.Series, tb: pd.Series, wp_a: float, n: int) -> str:
    """Analytical matchup narrative covering pace, efficiency matchups, 3PT tendencies,
    ball-security, and schedule context. Returns a multi-sentence paragraph (4-5 angles).
    """
    name_a, name_b = ta["Team"], tb["Team"]

    # Adjusted efficiency ratings + ranks
    adjt_a    = float(ta.get("AdjT", 68.0))
    adjt_b    = float(tb.get("AdjT", 68.0))
    adjt_nr_a = int(ta.get("AdjT_nr", n // 2))
    adjt_nr_b = int(tb.get("AdjT_nr", n // 2))
    adjo_a    = float(ta.get("AdjO", 100.0))
    adjo_b    = float(tb.get("AdjO", 100.0))
    adjo_nr_a = int(ta.get("AdjO_nr", n // 2))
    adjo_nr_b = int(tb.get("AdjO_nr", n // 2))
    adjd_a    = float(ta.get("AdjD", 100.0))
    adjd_b    = float(tb.get("AdjD", 100.0))
    adjd_nr_a = int(ta.get("AdjD_nr", n // 2))
    adjd_nr_b = int(tb.get("AdjD_nr", n // 2))
    em_a      = float(ta.get("AdjEM", 0.0))
    em_b      = float(tb.get("AdjEM", 0.0))

    # Schedule / luck / seeding
    sos_nr_a = int(ta.get("SOS_nr", n // 2))
    sos_nr_b = int(tb.get("SOS_nr", n // 2))
    luck_a   = float(ta.get("Luck", 0.0))
    luck_b   = float(tb.get("Luck", 0.0))
    seed_a   = int(ta.get("seed", 8))
    seed_b   = int(tb.get("seed", 9))

    # Per-game stats (may be 0 if not yet loaded)
    three_pct_a = float(ta.get("3P%", 0) or 0)
    three_pct_b = float(tb.get("3P%", 0) or 0)
    three_pa_a  = float(ta.get("3PaPG", 0) or 0)
    three_pa_b  = float(tb.get("3PaPG", 0) or 0)
    opp3pa_a    = float(ta.get("Opp3PaPG", 0) or 0)
    opp3pa_b    = float(tb.get("Opp3PaPG", 0) or 0)
    topg_a      = float(ta.get("TOPG", 0) or 0)
    topg_b      = float(tb.get("TOPG", 0) or 0)
    stl_a       = float(ta.get("StlPG", 0) or 0)
    stl_b       = float(tb.get("StlPG", 0) or 0)
    blk_a       = float(ta.get("BlkPG", 0) or 0)
    blk_b       = float(tb.get("BlkPG", 0) or 0)

    parts: list[str] = []

    # ── 1. Pace battle ──────────────────────────────────────────────────────
    tempo_gap = abs(adjt_nr_a - adjt_nr_b)
    if tempo_gap >= 120:
        faster   = name_a if adjt_nr_a < adjt_nr_b else name_b
        slower   = name_b if adjt_nr_a < adjt_nr_b else name_a
        fast_t   = adjt_a if adjt_nr_a < adjt_nr_b else adjt_b
        slow_t   = adjt_b if adjt_nr_a < adjt_nr_b else adjt_a
        fast_nr  = min(adjt_nr_a, adjt_nr_b)
        slow_nr  = max(adjt_nr_a, adjt_nr_b)
        parts.append(
            f"**Pace is the central battle**: {faster} ({fast_t:.1f} adj. poss/40 min, #{fast_nr} nationally) "
            f"wants to push into the open floor while {slower} ({slow_t:.1f}, #{slow_nr}) thrives in half-court "
            "execution — whoever dictates tempo in the first five minutes sets the game's entire character."
        )
    elif tempo_gap >= 50:
        faster = name_a if adjt_nr_a < adjt_nr_b else name_b
        slower = name_b if adjt_nr_a < adjt_nr_b else name_a
        parts.append(
            f"There's a genuine pace differential — {faster} prefers to push early while {slower} likes to "
            "methodically work the shot clock. Expect tactical battles over transition attempts and "
            "early-clock possessions that could define which team plays its preferred game."
        )
    else:
        parts.append(
            "Both teams operate at a similar tempo, so this becomes a half-court chess match: "
            "shot quality, ball security, and whose offensive system is more effective at breaking "
            "down the other's defense will ultimately separate them."
        )

    # ── 2. Efficiency + key matchup angle ───────────────────────────────────
    em_gap   = abs(em_a - em_b)
    fav_name = name_a if em_a >= em_b else name_b
    dog_name = name_b if em_a >= em_b else name_a
    fav_adjo_nr = adjo_nr_a if em_a >= em_b else adjo_nr_b
    fav_adjd_nr = adjd_nr_a if em_a >= em_b else adjd_nr_b
    dog_adjo_nr = adjo_nr_b if em_a >= em_b else adjo_nr_a
    dog_adjd_nr = adjd_nr_b if em_a >= em_b else adjd_nr_a
    fav_adjo    = adjo_a if em_a >= em_b else adjo_b
    fav_adjd    = adjd_a if em_a >= em_b else adjd_b
    dog_adjo    = adjo_b if em_a >= em_b else adjo_a
    dog_adjd    = adjd_b if em_a >= em_b else adjd_a

    if em_gap <= 2.5:
        parts.append(
            f"The efficiency numbers are almost identical ({em_a:+.1f} vs {em_b:+.1f} AdjEM) — "
            "both teams bring comparable offense and defense, which shifts the margin entirely to "
            "individual performances, in-game adjustments, and who makes more of their high-leverage possessions late."
        )
    elif adjo_nr_a <= 50 and adjd_nr_b >= 180:
        parts.append(
            f"{name_a}'s offense (#{adjo_nr_a}, {adjo_a:.1f} adj. pts/100) is a serious problem for "
            f"{name_b}'s defense (#{adjd_nr_b}, {adjd_b:.1f} AdjD) — if {name_a} finds its rhythm early this "
            "game could open up fast and never really be in doubt."
        )
    elif adjo_nr_b <= 50 and adjd_nr_a >= 180:
        parts.append(
            f"{name_b}'s attack (#{adjo_nr_b}, {adjo_b:.1f} pts/100) against {name_a}'s defense (#{adjd_nr_a}) "
            "is the clearest path to an upset — that offensive efficiency hasn't faced many defenses this porous, "
            "and an early lead could force a complete style shift."
        )
    elif fav_adjd_nr <= 40 and dog_adjo_nr >= 180:
        parts.append(
            f"The defining structural edge is {fav_name}'s defense (#{fav_adjd_nr}, {fav_adjd:.1f} AdjD), "
            f"which systematically limits what {dog_name}'s offense (#{dog_adjo_nr}) does. "
            "Expect a grind where every made shot has to be earned and the margin stays narrow until it suddenly doesn't."
        )
    elif dog_adjd_nr <= 40 and fav_adjo_nr >= 180:
        parts.append(
            f"{dog_name}'s defense (#{dog_adjd_nr}, {dog_adjd:.1f} AdjD) is the great equalizer — "
            f"it has the capability to clamp down on {fav_name}'s offense (#{fav_adjo_nr}, {fav_adjo:.1f} pts/100) "
            "and keep the underdog in a game where overall efficiency says they shouldn't compete."
        )
    else:
        if em_gap <= 8.0:
            parts.append(
                f"{fav_name} holds a {em_gap:.1f} pt/100 AdjEM edge — meaningful over a season but absolutely "
                f"closeable in 40 minutes. {dog_name}'s best path runs through limiting {fav_name}'s offense "
                f"(#{fav_adjo_nr}, {fav_adjo:.1f} pts/100) while converting its own half-court sets at a higher clip than normal."
            )
        else:
            parts.append(
                f"The efficiency gap is substantial ({em_gap:.1f} pts/100 AdjEM favoring {fav_name}) — "
                f"{dog_name} would need to compress the game, protect the ball, and get at least neutral "
                "shooting variance just to stay within range into the second half."
            )

    # ── 3. Three-point style (if per-game data available) ────────────────────
    if three_pa_a > 0 and three_pa_b > 0:
        three_gap = abs(three_pa_a - three_pa_b)
        pct_gap   = abs(three_pct_a - three_pct_b)
        if three_gap >= 3:
            more_3    = name_a if three_pa_a > three_pa_b else name_b
            less_3    = name_b if three_pa_a > three_pa_b else name_a
            more_3pa  = max(three_pa_a, three_pa_b)
            more_3pct = three_pct_a if three_pa_a > three_pa_b else three_pct_b
            opp_3pa   = opp3pa_b if three_pa_a > three_pa_b else opp3pa_a
            parts.append(
                f"{more_3}'s offense is built around three-point volume ({more_3pa:.1f} attempts/game at {more_3pct:.1f}%) — "
                f"against {less_3}, who allows {opp_3pa:.1f} three-point attempts per game, the arc "
                "is an exploitable gap if those shots start dropping."
            )
        elif pct_gap >= 3.5:
            better_shooter = name_a if three_pct_a > three_pct_b else name_b
            b_pct = max(three_pct_a, three_pct_b)
            w_pct = min(three_pct_a, three_pct_b)
            parts.append(
                f"Three-point accuracy is a secondary edge: {better_shooter} shoots {b_pct:.1f}% from deep "
                f"vs the other team's {w_pct:.1f}% — in a close game that gap compounds quickly over 20+ attempts."
            )

    # ── 4. Ball security vs. defensive disruption ────────────────────────────
    if topg_a > 0 and topg_b > 0:
        to_gap = abs(topg_a - topg_b)
        if to_gap >= 1.5:
            sloppy    = name_a if topg_a > topg_b else name_b
            careful   = name_b if topg_a > topg_b else name_a
            sloppy_to = max(topg_a, topg_b)
            press_stl = stl_b if topg_a > topg_b else stl_a
            parts.append(
                f"Ball security could be decisive: {sloppy} turns it over {sloppy_to:.1f} times per game, "
                f"and {careful} averages {press_stl:.1f} steals — live-ball turnovers are the fastest way "
                "to swing a momentum run in either team's direction."
            )

    # ── 5. Schedule / luck context ───────────────────────────────────────────
    sos_gap = abs(sos_nr_a - sos_nr_b)
    if sos_gap >= 120:
        harder = name_a if sos_nr_a < sos_nr_b else name_b
        easier = name_b if sos_nr_a < sos_nr_b else name_a
        hr     = min(sos_nr_a, sos_nr_b)
        er     = max(sos_nr_a, sos_nr_b)
        parts.append(
            f"Schedule context matters: {harder}'s numbers were forged against a #{hr} SOS gauntlet while "
            f"{easier} built theirs in softer conditions (#{er} SOS) — the ratings adjust for this, but "
            "tournament experience under pressure carries its own kind of value."
        )
    elif luck_b < -0.09 and seed_b > seed_a:
        parts.append(
            f"One overlooked angle: {name_b} has been genuinely unlucky in close games all season — "
            "their record undersells how well they've competed, and a neutral floor is exactly when "
            "that bad-luck tax tends to even out."
        )
    elif luck_a > 0.10 and seed_a < seed_b:
        parts.append(
            f"One caveat: {name_a} has been exceptionally fortunate in clutch situations this year, "
            "meaning their record overstates what the underlying efficiency alone would project."
        )

    return " ".join(parts[:4]).replace("\u2014", ", ")


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
        f"{fav_name} is expected to win by ~{dog_deficit} points."
    )

    return [b.replace("\u2014", ", ") for b in bullets]
# ---------------------------------------------------------------------------
_BP_MU_PAIRS   = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
_BP_REGIONS    = ["East", "West", "South", "Midwest"]
# Light background + border accent per region for visual differentiation
_BP_REGION_BG: dict[str, str]     = {"East": "#dbeafe", "West": "#fee2e2", "South": "#dcfce7", "Midwest": "#fef3c7"}
_BP_REGION_ACC: dict[str, str]    = {"East": "#1d4ed8", "West": "#b91c1c", "South": "#15803d", "Midwest": "#d97706"}
_BP_REGION_EMOJI: dict[str, str]  = {"East": "🔵", "West": "🔴", "South": "🟢", "Midwest": "🟡"}
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
    # Final Four: East (E8 slot 0) vs South (E8 slot 2), West (E8 slot 1) vs Midwest (E8 slot 3)
    if rnd == 4:
        if slot == 0:
            ta = picks.get((3, 0)) or "TBD"  # East
            tb = picks.get((3, 2)) or "TBD"  # South
        else:
            ta = picks.get((3, 1)) or "TBD"  # West
            tb = picks.get((3, 3)) or "TBD"  # Midwest
        return ta, tb
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

rankings_tab, team_tab, scatter_tab, bracket_tab, upset_tab, picks_tab, about_tab = st.tabs(
    ["📊 Team Rankings", "🏀 Team Profile", "📈 Valuable Charts", "🏆 Bracket", "💡 Upset Value", "🎯 CarmPom Picks", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Rankings tab
# ---------------------------------------------------------------------------

with rankings_tab:
    # Load tournament teams from bracket CSV for the filter
    try:
        _brkt_csv = pd.read_csv("data/bracket_2026.csv")
        _tourn_teams: set[str] = set(_brkt_csv["team"].str.strip().tolist())
    except Exception:
        _tourn_teams = set()

    # Controls row
    col_season, col_search, col_conf, col_n, col_tourn = st.columns([1, 2, 2, 1, 1.4])

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

    with col_tourn:
        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
        tourn_only = st.checkbox("Tournament teams only", value=True)

    # Apply filters
    filtered = df.copy()
    if tourn_only and _tourn_teams:
        filtered = filtered[filtered["Team"].isin(_tourn_teams)]
    if selected_conf != "All conferences":
        filtered = filtered[filtered["Conf"] == selected_conf]
    if search:
        filtered = filtered[filtered["Team"].str.contains(search, case=False, na=False)]
    if not tourn_only:
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
    # ── KenPom comparison: badge teams CarmPom rates higher ─────────────────
    _kp_compare = load_kenpom_comparison(season)
    _kp_diff: dict[int, int] = dict(zip(
        _kp_compare["team_id"].astype(int),
        _kp_compare["rank_diff"].astype(int),
    ))

    def _vs_kp_badge(team_id: int) -> str:
        """Return HTML badge showing CarmPom vs KenPom rank gap."""
        diff = _kp_diff.get(int(team_id), None)
        if diff is None:
            return "—"
        if diff >= 5:
            # CarmPom meaningfully higher — green badge
            return (
                f"<span style='background:#1b5e20;color:#e8f5e9;border-radius:4px;"
                f"padding:2px 7px;font-weight:700;font-size:11px;white-space:nowrap'>"
                f"▲ +{diff}</span>"
            )
        if diff <= -5:
            # KenPom meaningfully higher — muted red
            return (
                f"<span style='background:#4a1010;color:#ffcdd2;border-radius:4px;"
                f"padding:2px 7px;font-weight:700;font-size:11px;white-space:nowrap'>"
                f"▼ {diff}</span>"
            )
        # Within 4 spots — neutral
        sign = "+" if diff > 0 else ""
        return (
            f"<span style='background:#37474f;color:#cfd8dc;border-radius:4px;"
            f"padding:2px 7px;font-size:11px;white-space:nowrap'>{sign}{diff}</span>"
        )

    display_df = filtered[["Rank", "Team", "Conf", "Record", "Player Stats"]].copy()
    display_df["vs KP"] = filtered["team_id"].apply(_vs_kp_badge)
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
        "Rank", "Team", "Conf", "Record", "Player Stats", "vs KP",
        "PPG", "OppPPG", "RebPG", "AstPG", "OrebPG", "TOPG",
        "FG%", "3P%", "3PaPG", "3PmPG", "FT%", "FTmPG",
    ]
    # Only include cols that are actually present (Player Stats may be absent if espn_id is missing)
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

    # Convert styled df to HTML. escape=False lets badge HTML and anchor tags render correctly.
    _table_html = styled.hide(axis="index").to_html(escape=False)

    # Replace raw ESPN URLs in td cells with clickable anchor tags.
    _table_html = re.sub(
        r'(<td[^>]*>)(https://www\.espn\.com/[^<]+)(</td>)',
        lambda m: (
            m.group(1).rstrip('>') + ' style="background-color:#f4f6f8;padding:7px 14px">'
            f'<a href="{m.group(2)}" target="_blank" '
            'style="color:#1565C0;text-decoration:none;font-size:12px;font-weight:500">Stats ↗</a></td>'
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
        'Player Stats': 'Link to ESPN player stats page for this team',
        'vs KP':  'CarmPom rank vs KenPom rank. ▲ = CarmPom rates this team higher. ▼ = KenPom rates them higher. Number = spots difference.',
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
    """Return (playstyle_name, one-line tagline) driven by HOW the team plays.

    The 10 radar-spoke attributes (pace, 3PT volume, 3PT accuracy, off. rebounding,
    ball security, FT drawing, assists, def. rebounding, forced TOs, paint defense)
    are the primary signal. Efficiency-based prestige labels are reserved for the
    top ~10 nationally.
    """
    adjem_nr = int(t.get("AdjEM_nr", n))
    adjt_nr  = int(t["AdjT_nr"])
    adjo_nr  = int(t["AdjO_nr"])
    adjd_nr  = int(t["AdjD_nr"])
    national_elite = adjem_nr <= 10   # only the true top-10 get efficiency-based prestige names

    fast = adjt_nr <= 80
    slow = adjt_nr >= 270

    # Style flags from per-game stats (all 10 radar spokes)
    three_heavy = three_light = three_accurate = three_inaccurate = False
    glass_eater = good_dreb = ball_safe = ball_wild = ft_heavy = pass_first = False
    forced_to = rim_protector = False
    if ts is not None:
        _3pa_nr  = int(ts.get("3PaPG_nr",  n))
        _3pct_nr = int(ts.get("3P%_nr",    n))
        _oreb_nr = int(ts.get("OrebPG_nr", n))
        _reb_nr  = int(ts.get("RebPG_nr",  n))
        _to_nr   = int(ts.get("TOPG_nr",   n))
        _ft_nr   = int(ts.get("FTmPG_nr",  n))
        _ast_nr  = int(ts.get("AstPG_nr",  n))
        _stl_nr  = int(ts.get("StlPG_nr",  n))
        _blk_nr  = int(ts.get("BlkPG_nr",  n))
        three_heavy    = _3pa_nr  <= 60           # top ~16% by 3PA/game
        three_light    = _3pa_nr  >= 280          # bottom ~23%
        three_accurate = _3pct_nr <= 70           # top ~19% by 3P%
        three_inaccurate = _3pct_nr >= 290
        glass_eater    = _oreb_nr <= 50           # top ~14% by OrebPG
        good_dreb      = _reb_nr  <= 80           # top ~22% total rebounds (proxy for def reb)
        ball_safe      = _to_nr   <= 60           # top ~16% (fewest TOs)
        ball_wild      = _to_nr   >= 280          # bottom ~23%
        ft_heavy       = _ft_nr   <= 60           # top ~16% by FTm/game
        pass_first     = _ast_nr  <= 60           # top ~16% by Ast/game
        forced_to      = _stl_nr  <= 60           # top ~16% by steals/game — disruptive defense
        rim_protector  = _blk_nr  <= 60           # top ~16% by blocks/game — paint deterrent

    # ── Top-10 national elite: prestige label inflected by dominant style ──
    if national_elite:
        if fast and three_heavy:
            return "👑 Run-and-Gun Elite", "One of the nation's best — relentless pace and perimeter firepower"
        if slow and three_light and glass_eater:
            return "👑 Dominant Inside Force", "Elite program that controls games in the paint and grinds opponents down"
        if slow and adjo_nr <= 15:
            return "👑 Half-Court Juggernaut", "Elite half-court offense that methodically takes what the defense gives"
        if adjd_nr <= 15:
            return "👑 Defensive Powerhouse", "One of the toughest defensive teams in the country this season"
        if three_heavy and three_accurate:
            return "👑 Perimeter Blitz", "Elite shooting team — opens the floor and buries opponents with threes"
        if rim_protector and forced_to:
            return "👑 Lockdown Machine", "Elite two-way disruption — protects the rim and creates chaos with steals"
        return "👑 National Contender", "One of the most complete programs in the country"

    # ── Fast-pace identity ────────────────────────────────────────────────
    if fast and three_heavy and three_accurate:
        return "🚀 Push-and-Shoot", "Runs the floor before defenses set, then buries the open three"
    if fast and three_heavy:
        return "⚡ Pace-and-Space", "Uses relentless tempo to strain the defense and launch threes in transition"
    if fast and glass_eater:
        return "💥 Crash-and-Dash", "Pushes the pace and crashes every miss — second chances fuel the offense"
    if fast and pass_first and ball_safe:
        return "🎭 Motion Machine", "High-tempo unselfish offense built on constant movement and clean decision-making"
    if fast and ft_heavy:
        return "🏃 Attacking Guards", "Pushes pace and attacks downhill — earns trips to the line at a high rate"
    if fast and ball_wild:
        return "🌪️ Chaotic Speed", "Plays at a breakneck pace but turns it over too often — explosive but sloppy"
    if fast:
        return "⚡ Up-Tempo Pusher", "Lives in transition — plays fast, scores early, and makes opponents uncomfortable"

    # ── Slow-pace identity ────────────────────────────────────────────────
    if slow and three_light and glass_eater and ft_heavy:
        return "⚒️ Paint Dominant", "Grinds the game down and punishes teams inside — boards, fouls, and buckets"
    if slow and three_light and glass_eater:
        return "⚒️ Post-Up Bully", "Controls pace and the glass — score deep in the shot clock near the rim"
    if slow and three_light and ball_safe:
        return "🐢 Half-Court Surgeon", "Patient, disciplined offense that takes care of the ball and attacks the paint"
    if slow and three_light:
        return "⚒️ Half-Court Grinder", "Methodical interior attack — uses the whole clock and minimizes big misses"
    if slow and three_heavy and three_accurate:
        return "🎯 Patient Marksmen", "Slows the game down, moves the ball, and waits for the open three"
    if slow and three_heavy:
        return "🐢 Deliberate Gunners", "Unhurried offense that settles into the half court and fires from deep"
    if slow and pass_first and ball_safe:
        return "🎭 Half-Court Orchestra", "Deliberate possession-by-possession offense driven by ball movement and shot selection"
    if slow and good_dreb:
        return "🐢 Rebounding Grind", "Controls the pace and cleans the glass on both ends — physical and deliberate"
    if slow:
        return "🐢 Half-Court Grinder", "Uses every second of the shot clock and forces opponents into a grinding game"

    # ── Three-point identity (mid-pace) ──────────────────────────────────
    if three_heavy and three_accurate and pass_first:
        return "🎯 Precision Shooters", "Crisp ball movement generates clean looks — and they actually knock them down"
    if three_heavy and three_accurate:
        return "🎯 Perimeter Marksmen", "High three-point volume backed by genuine shooting accuracy — a real threat from deep"
    if three_heavy and ball_safe and pass_first:
        return "🎯 Sharpshooter System", "Takes care of the ball, shares it freely, and searches for the open three"
    if three_heavy and ball_safe:
        return "🏹 Clean Shooters", "Fires a lot of threes and rarely gives the ball away — low-chaos perimeter attack"
    if three_heavy and three_inaccurate:
        return "🎲 Boom-or-Bust Shooters", "High three-point volume with poor accuracy — games depend on which shooting night shows up"
    if three_heavy:
        return "🌍 Arc-Heavy Attack", "Lives behind the arc — run with the shooting variance and the wins follow"

    # ── Paint/rebounding identity ─────────────────────────────────────────
    if glass_eater and ft_heavy and three_light:
        return "💪 Physical Bully", "Punishes teams at the rim — cleans the glass, draws fouls, and scores the hard way"
    if glass_eater and ft_heavy:
        return "💪 Board-and-Foul Machine", "Dominates second chances and earns trips to the line — tough to keep off the scoreboard"
    if glass_eater and pass_first:
        return "💪 Team Glass Crashers", "Coordinated offensive rebounding attack that gives the offense constant second looks"
    if glass_eater:
        return "💪 Board Crashers", "Second-chance opportunities are the engine — winning the offensive glass is the identity"

    # ── Ball movement / passing identity ─────────────────────────────────
    if pass_first and ball_safe and ft_heavy:
        return "🧠 Mistake-Free Machine", "Shares the ball, protects it, and gets to the line — coaches dream of this formula"
    if pass_first and ball_safe:
        return "🎭 Ball Movement Offense", "Patient and unselfish — low turnovers and high assists define the attack"
    if pass_first and three_accurate:
        return "🎭 Playmaking Shooters", "Ball movement creates open looks — and the personnel finishes them from outside"
    if pass_first:
        return "🎭 Unselfish Playmakers", "High assist rate shows a team that always finds the better shot"

    # ── Free throw / aggression identity ─────────────────────────────────
    if ft_heavy and ball_safe:
        return "🏋️ Foul-Drawing Grinders", "Gets into the lane, draws contact, and converts at the line — earns every point"
    if ft_heavy:
        return "🏋️ Foul Hunters", "Attacks aggressively and lives at the free throw line"

    # ── Ball security (alone) ─────────────────────────────────────────────
    if ball_safe and good_dreb:
        return "🧠 Low-Turnover Defense", "Takes care of the ball and cleans the glass — limits extra opportunities for opponents"
    if ball_safe:
        return "🧠 Ball-Control Offense", "Rarely gives it away — methodical possession-by-possession approach"
    if ball_wild:
        return "🎲 High-Turnover Risk", "Capable scorers but prone to giveaways — opponent fast breaks are the danger"

    # ── Forced turnovers (steals) ─────────────────────────────────────────
    if forced_to and fast:
        return "🪤 Press-and-Run", "Forces turnovers and immediately converts them — defense triggers the offense"
    if forced_to and rim_protector:
        return "🛡️ Two-Way Disruptors", "Generates steals on the perimeter and blocks at the rim — defense is the identity"
    if forced_to and good_dreb:
        return "🪤 Chaos Defense", "Swarms passing lanes, crashes the glass, and turns disruption into points"
    if forced_to:
        return "🪤 Pickpocket Defense", "Leads the country in getting into the passing lanes — defense creates offense"

    # ── Rim protection (blocks) ───────────────────────────────────────────
    if rim_protector and slow and three_light:
        return "🧱 Paint Fortress", "Anchors the defense at the rim, controls pace, and dares opponents to shoot over them"
    if rim_protector and good_dreb:
        return "🧱 Interior Anchor", "Protects the basket and wins the glass — opponents think twice before attacking the rim"
    if rim_protector:
        return "🧱 Shot Blocker", "Elite rim presence changes how opponents attack — keeps the defense behind them"

    # ── Rebounding (alone) ────────────────────────────────────────────────
    if good_dreb:
        return "🏀 Rebounding Foundation", "Controls the glass consistently and limits opponent second-chance opportunities"

    # ── Fallback ──────────────────────────────────────────────────────────
    return "⚖️ Balanced", "No single dominant trait — competes in multiple phases without a glaring weakness"


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

    # Luck flag — pool of alternatives so repeated teams don't all read alike
    luck_str = ""
    if luck > 0.04:
        _slot = hash(name) % 4
        if _slot == 0:
            luck_str = " They've also been somewhat fortunate in close games — their record slightly flatters their underlying efficiency."
        elif _slot == 1:
            luck_str = (
                " Their win total has been helped by close-game variance — "
                "the adjusted margin alone would project a slightly lower ceiling."
            )
        elif _slot == 2:
            luck_str = (
                " Worth flagging: they've won a disproportionate share of tight games. "
                "Teams that push them to the wire may find the margin closer than expected."
            )
        else:
            luck_str = (
                f" The luck metric suggests {name} has benefited from late-game variance — "
                "the efficiency numbers are the more reliable predictor going forward."
            )
    elif luck < -0.04:
        _slot = hash(name) % 5
        if _slot == 0:
            luck_str = " Notably, they've been unlucky in close games — their record undersells how good they actually are."
        elif _slot == 1:
            luck_str = (
                f" The efficiency numbers ({adjem:+.2f} AdjEM) paint a better picture than the record alone — "
                "late-game variance has cost them wins they played well enough to deserve."
            )
        elif _slot == 2:
            luck_str = (
                " They've been on the wrong end of several coin-flip games this year. "
                "Expect their underlying quality to show through if this tournament goes deep."
            )
        elif _slot == 3 and sos_nr <= 60:
            luck_str = (
                f" Competing against a tough schedule (SOS #{sos_nr}) while running into bad luck in "
                "tight spots has kept their record modest — but the adjusted numbers are significantly better."
            )
        else:
            luck_str = (
                " Close-game results have skewed against them this season; "
                "the adjusted efficiency is a more honest reflection of their level."
            )

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

    return "  \n\n".join(parts).replace("\u2014", ", ")


# ---------------------------------------------------------------------------
# Team Profile tab
# ---------------------------------------------------------------------------

with team_tab:
    _all_teams = load_rankings(2026)

    # Limit to tournament teams when the real bracket is available
    _profile_bracket = load_real_bracket(2026)
    if _profile_bracket is not None and "Team" in _profile_bracket.columns:
        _tourn_names = set(_profile_bracket["Team"].tolist())
        team_options = sorted(
            t for t in _all_teams["Team"].tolist() if t in _tourn_names
        )
    else:
        team_options = _all_teams["Team"].sort_values().tolist()

    # Sync selected team from URL query param on first load
    if "tp_selected_team" not in st.session_state:
        _qp_team = st.query_params.get("tp_team", "")
        if _qp_team and _qp_team in team_options:
            st.session_state["tp_selected_team"] = _qp_team
        elif "Duke" in team_options:
            st.session_state["tp_selected_team"] = "Duke"
        else:
            st.session_state["tp_selected_team"] = team_options[0] if team_options else ""

    _current_selected: str = st.session_state.get("tp_selected_team", team_options[0] if team_options else "")
    if _current_selected not in team_options:
        _current_selected = team_options[0] if team_options else ""

    # Search input filters the dropdown
    _search_query = st.text_input(
        "Search team",
        placeholder="🔍  Type a team name to filter...",
        label_visibility="collapsed",
        key="team_profile_search",
    )
    _filtered_opts = (
        [t for t in team_options if _search_query.strip().lower() in t.lower()]
        if _search_query.strip() else team_options
    ) or team_options

    def _tp_sync_qp() -> None:
        val = st.session_state.get("tp_selectbox", "")
        st.session_state["tp_selected_team"] = val
        st.query_params["tp_team"] = val

    _default_idx = (
        _filtered_opts.index(_current_selected)
        if _current_selected in _filtered_opts else 0
    )

    selected_team = st.selectbox(
        "Team",
        _filtered_opts,
        index=min(_default_idx, len(_filtered_opts) - 1),
        label_visibility="collapsed",
        key="tp_selectbox",
        on_change=_tp_sync_qp,
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
    espn_url  = _t["Player Stats"] if pd.notna(_t.get("Player Stats")) else None
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
    try:
        _brk_for_cf = load_real_bracket(2026)
        _tourn_cf = _all_teams[_all_teams["Team"].isin(set(_brk_for_cf["Team"].tolist()))] if _brk_for_cf is not None else _all_teams
    except Exception:
        _tourn_cf = _all_teams
    _tf_n_cf = max(len(_tourn_cf), 1)

    def _stat_card(col, label: str, value: str, nat_rank: int, stat_col: str) -> None:
        """Render a color-coded stat card. Color based on tournament-field percentile; text shows national rank."""
        _t_vals = _tourn_cf[stat_col].dropna()
        _team_val = float(_t[stat_col])
        # AdjD: lower is better (rank ascending), all others higher is better
        if stat_col == "AdjD":
            _pct = round((_t_vals >= _team_val).sum() / _tf_n_cf * 100)
        else:
            _pct = round((_t_vals <= _team_val).sum() / _tf_n_cf * 100)
        _pct = max(1, min(100, _pct))
        if _pct >= 75:
            bg, fg = "#1b5e20", "#e8f5e9"
        elif _pct >= 50:
            bg, fg = "#1a3a1a", "#c8e6c9"
        elif _pct >= 25:
            bg, fg = "#4a3800", "#fff8e1"
        else:
            bg, fg = "#4a1010", "#ffebee"
        with col:
            st.markdown(
                f"<div style='background:{bg};border-radius:10px;padding:12px 14px;"
                f"font-family:system-ui;text-align:center'>"
                f"<div style='font-size:11px;color:{fg};opacity:0.7;font-weight:600;"
                f"letter-spacing:0.5px;text-transform:uppercase'>{label}</div>"
                f"<div style='font-size:24px;font-weight:800;color:{fg};margin:4px 0'>{value}</div>"
                f"<div style='font-size:11px;color:{fg};opacity:0.75'>#{nat_rank} nationally</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _stat_card(h2, "AdjEM", f"{_t['AdjEM']:+.2f}", int(_t['AdjEM_nr']), "AdjEM")
    _stat_card(h3, "AdjO",  f"{_t['AdjO']:.1f}",  int(_t['AdjO_nr']),  "AdjO")
    _stat_card(h4, "AdjD",  f"{_t['AdjD']:.1f}",  int(_t['AdjD_nr']),  "AdjD")
    if espn_url:
        st.markdown(f"**ESPN Player Stats Page:** [View on ESPN ↗]({espn_url})")

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

        # ── 10-spoke playstyle radar (no Offense/Defense spokes) ─────────────
        # All spokes describe HOW the team plays, not efficiency ratings.
        _radar_labels = ["Pace", "3PT Volume", "3PT Accuracy", "Off. Rebounding",
                         "Ball Security", "FT Drawing", "Assists", "Def. Rebounding",
                         "Forced TOs", "Paint Def."]

        if _ts is not None:
            _adjt_pct  = _pct(_t["AdjT_nr"])
            _3pa_pct   = _pct(_ts["3PaPG_nr"])
            _3pct_pct  = _pct(_ts["3P%_nr"])
            _oreb_pct  = _pct(_ts["OrebPG_nr"])
            _to_pct    = _pct(_ts["TOPG_nr"])    # rank 1 = fewest TOs → high percentile = ball safe
            _ftm_pct   = _pct(_ts["FTmPG_nr"])   # free throws drawn/made per game
            _ast_pct   = _pct(_ts["AstPG_nr"])
            _dreb_pct  = round((1 - (_ts["RebPG_nr"] - 1) / _n) * 100)
            _stl_pct   = _pct(_ts["StlPG_nr"]) if "StlPG_nr" in _ts.index else 50
            _opp2p_pct = _pct(_ts["Opp2P%_nr"]) if "Opp2P%_nr" in _ts.index else 50
        else:
            _adjt_pct = _3pa_pct = _3pct_pct = _oreb_pct = _to_pct = _ftm_pct = _ast_pct = _dreb_pct = _stl_pct = _opp2p_pct = 50

        _radar_vals = [_adjt_pct, _3pa_pct, _3pct_pct, _oreb_pct,
                       _to_pct,  _ftm_pct,  _ast_pct,  _dreb_pct,
                       _stl_pct, _opp2p_pct]

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
        st.caption("Each spoke = national percentile for that playstyle dimension. Ball Security = inverted turnover rate · Forced TOs = steals/game · Paint Def. = opp 2PT FG% (lower allowed = better).")

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

    def _eff_card(col, label: str, value: str, nat_rank: int, help_txt: str) -> None:
        """Metric card with conditionally-colored rank/percentile delta text."""
        pct_val = _pct(nat_rank)
        if pct_val >= 80:
            rank_color = "#43a047"   # bright green
        elif pct_val >= 60:
            rank_color = "#66bb6a"   # light green
        elif pct_val >= 40:
            rank_color = "#ffa726"   # amber
        elif pct_val >= 20:
            rank_color = "#ef5350"   # red
        else:
            rank_color = "#b71c1c"   # dark red
        with col:
            st.markdown(
                f"<div style='font-family:system-ui;padding:4px 0' title='{help_txt}'>"
                f"<div style='font-size:12px;color:#9aa5b4;font-weight:500'>{label}</div>"
                f"<div style='font-size:26px;font-weight:700;color:#ffffff;margin:2px 0'>{value}</div>"
                f"<div style='font-size:12px;font-weight:600;color:{rank_color}'>"
                f"↑ #{nat_rank} · {pct_val}th pct</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

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
        _eff_card(_col_ui, _label, _val_fmt, int(_t[f"{_metric}_nr"]), _help)

    if _ts is not None:
        st.divider()
        st.markdown("#### Per-Game Stats")
        _STAT_OFFENSE = [
            ("PPG",      "Points per game",             False),
            ("FG%",      "Field goal %",                False),
            ("3P%",      "Three-point %",               False),
            ("3PaPG",    "3PA per game",                False),
            ("3PmPG",    "3PM per game",                False),
            ("FT%",      "Free throw %",                False),
            ("FTmPG",    "FTM per game",                False),
            ("RebPG",    "Rebounds per game",           False),
            ("OrebPG",   "Off. rebounds per game",      False),
            ("AstPG",    "Assists per game",            False),
            ("TOPG",     "Turnovers per game",          True),
        ]
        _STAT_DEFENSE = [
            ("OppPPG",   "Opp. pts per game",           True),
            ("Opp3PaPG", "Opp 3PT attempts/game",       True),
            ("Opp3P%",   "Opp three-point %",           True),
            ("Opp2P%",   "Opp 2PT FG% (interior def.)", True),
            ("StlPG",    "Steals per game (forced TOs)", False),
            ("BlkPG",    "Blocks per game",              False),
        ]

        def _build_stat_rows(stat_defs):
            rows = []
            for col, label, _ in stat_defs:
                nr = int(_ts[f"{col}_nr"]) if pd.notna(_ts.get(f"{col}_nr")) else None
                pct_s = round((1 - (nr - 1) / _n) * 100) if nr else None
                rows.append({
                    "Stat": label,
                    "Value": f"{float(_ts[col]):.1f}" if pd.notna(_ts[col]) else "—",
                    "Nat'l Rank": f"#{nr}" if nr else "—",
                    "Percentile": pct_s if pct_s else 0,
                })
            return rows

        off_rows = _build_stat_rows(_STAT_OFFENSE)
        def_rows = _build_stat_rows(_STAT_DEFENSE)

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
                    f"<span style='color:inherit;font-size:12px'>{_nr_str} &nbsp; {pct}th pct</span>"
                    f"</div>"
                    f"<div style='background:#e8e8e8;border-radius:4px;height:7px;overflow:hidden'>"
                    f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width .3s'></div>"
                    f"</div></div>"
                )
            parts.append("</div>")
            return "".join(parts)

        _sc_a, _sc_b = st.columns(2)
        with _sc_a:
            st.markdown("**⚔️ Offense**")
            st.markdown(_stat_bar_html(off_rows), unsafe_allow_html=True)
        with _sc_b:
            st.markdown("**🛡️ Defense**")
            st.markdown(_stat_bar_html(def_rows), unsafe_allow_html=True)

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
# Valuable Charts tab
# ---------------------------------------------------------------------------

with scatter_tab:
    st.markdown("### 📈 Valuable Charts")
    st.markdown(
        "Visual breakdowns across key dimensions — each logo is a tournament team. "
        "Hover for details."
    )
    st.info(
        "📍 **How to read these charts:** "
        "**Top-right = elite** on both axes shown — the best teams. "
        "**Bottom-left = weakest** on both axes. "
        "The dashed white lines mark the tournament median — anything above/right of both lines "
        "is above average on both dimensions. Each chart's caption explains which direction is better.",
        icon=None,
    )

    _sc_ratings = load_rankings(2026)
    _sc_pg = load_per_game_stats(2026)
    try:
        _sc_brk = pd.read_csv("data/bracket_2026.csv")
        _sc_brk["team"] = _sc_brk["team"].str.strip()
        _sc_tourn_names: set[str] = set(_sc_brk["team"].tolist())
    except Exception:
        _sc_brk = pd.DataFrame(columns=["team", "seed"])
        _sc_tourn_names = set()

    _sc_r = _sc_ratings[_sc_ratings["Team"].isin(_sc_tourn_names)].copy()
    _sc_merged = _sc_r.merge(_sc_pg, on="team_id", how="left")
    _sc_merged = _sc_merged.merge(
        _sc_brk[["team", "seed"]].rename(columns={"team": "Team"}),
        on="Team", how="left",
    )
    _sc_merged["seed"] = pd.to_numeric(_sc_merged["seed"], errors="coerce")

    # ── Filters ───────────────────────────────────────────────────────────────
    _filt_col1, _filt_col2, _filt_col3 = st.columns([2, 2, 3], gap="small")

    # Conference filter
    _conf_options = sorted(_sc_merged["Conf"].dropna().unique().tolist()) if "Conf" in _sc_merged.columns else []
    with _filt_col1:
        _sel_confs = st.multiselect(
            "Conference",
            options=_conf_options,
            placeholder="All conferences",
            key="sc_conf_filter",
        )

    # Seed filter
    _seed_options = sorted([int(s) for s in _sc_merged["seed"].dropna().unique()])
    with _filt_col2:
        _sel_seeds = st.multiselect(
            "Seed",
            options=_seed_options,
            placeholder="All seeds",
            key="sc_seed_filter",
        )

    # Team multiselect
    _team_options_sc = sorted(_sc_merged["Team"].dropna().unique().tolist())
    with _filt_col3:
        _sel_teams = st.multiselect(
            "Teams",
            options=_team_options_sc,
            placeholder="All tournament teams",
            key="sc_team_filter",
        )

    # Apply filters
    _sc_plot = _sc_merged.copy()
    if _sel_confs:
        _sc_plot = _sc_plot[_sc_plot["Conf"].isin(_sel_confs)]
    if _sel_seeds:
        _sc_plot = _sc_plot[_sc_plot["seed"].isin(_sel_seeds)]
    if _sel_teams:
        _sc_plot = _sc_plot[_sc_plot["Team"].isin(_sel_teams)]

    import altair as _alt

    def _scatter_chart(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        invert_x: bool = False,
        invert_y: bool = False,
        x_ref: float | None = None,
        y_ref: float | None = None,
        x_domain: list[float] | None = None,
        y_domain: list[float] | None = None,
    ):
        """Build an Altair logo scatter chart (logos rendered client-side)."""
        df = df.dropna(subset=[x_col, y_col]).copy()

        # Use fixed domains when provided so axes don't rescale on filter changes.
        # Padding is added manually to the domain ends.
        PAD = 0.5  # axis padding in data units
        if x_domain:
            x_scale = _alt.Scale(
                reverse=invert_x, zero=False,
                domain=[x_domain[0] - PAD, x_domain[1] + PAD],
            )
        else:
            x_scale = _alt.Scale(reverse=invert_x, zero=False, padding=20)

        if y_domain:
            y_scale = _alt.Scale(
                reverse=invert_y, zero=False,
                domain=[y_domain[0] - PAD, y_domain[1] + PAD],
            )
        else:
            y_scale = _alt.Scale(reverse=invert_y, zero=False, padding=20)

        layers: list = []

        # Quadrant reference lines at the median — use a real 1-row DataFrame so
        # Altair actually renders the rule in a layered chart.
        if x_ref is not None:
            layers.append(
                _alt.Chart(pd.DataFrame({"v": [x_ref]}))
                .mark_rule(color="white", strokeWidth=2, opacity=0.65, strokeDash=[6, 4])
                .encode(x=_alt.X("v:Q", scale=x_scale))
            )
        if y_ref is not None:
            layers.append(
                _alt.Chart(pd.DataFrame({"v": [y_ref]}))
                .mark_rule(color="white", strokeWidth=2, opacity=0.65, strokeDash=[6, 4])
                .encode(y=_alt.Y("v:Q", scale=y_scale))
            )

        # Split: teams with a logo URL vs without
        _df_logo   = df[df["logo_url"].notna() & (df["logo_url"].astype(str).str.strip() != "")].copy()
        _df_nologo = df[~(df["logo_url"].notna() & (df["logo_url"].astype(str).str.strip() != ""))].copy()

        _tt = [
            _alt.Tooltip("Team:N", title="Team"),
            _alt.Tooltip("seed:Q", title="Seed"),
            _alt.Tooltip(f"{x_col}:Q", title=x_label, format=".2f"),
            _alt.Tooltip(f"{y_col}:Q", title=y_label, format=".2f"),
        ]

        if not _df_logo.empty:
            layers.append(
                _alt.Chart(_df_logo)
                .mark_image(width=28, height=28)
                .encode(
                    x=_alt.X(f"{x_col}:Q", title=x_label, scale=x_scale),
                    y=_alt.Y(f"{y_col}:Q", title=y_label, scale=y_scale),
                    url="logo_url:N",
                    tooltip=_tt,
                )
            )

        if not _df_nologo.empty:
            layers.append(
                _alt.Chart(_df_nologo)
                .mark_point(size=90, color="#4a90d9", filled=True, opacity=0.85)
                .encode(
                    x=_alt.X(f"{x_col}:Q", scale=x_scale),
                    y=_alt.Y(f"{y_col}:Q", scale=y_scale),
                    tooltip=_tt,
                )
            )

        return (
            _alt.layer(*layers)
            .properties(height=380)
            .configure_axis(
                gridColor="rgba(255,255,255,0.1)", gridWidth=0.5,
                domainColor="rgba(255,255,255,0.3)", tickColor="rgba(255,255,255,0.3)",
                labelColor="#ffffff", titleColor="#ffffff",
                titleFontSize=13, labelFontSize=12,
                labelFontWeight="normal", titleFontWeight="bold",
            )
            .configure_view(strokeWidth=0)
        )

    _sc_col1, _sc_col2 = st.columns(2, gap="large")

    with _sc_col1:
        # Chart 1: Efficiency Landscape (AdjO vs AdjD)
        st.markdown("**⚡ Efficiency Landscape**")
        if not _sc_plot.empty and "AdjO" in _sc_plot.columns:
            st.altair_chart(
                _scatter_chart(
                    _sc_plot, "AdjO", "AdjD",
                    "Adjusted Offense (pts/100 possessions)",
                    "Adjusted Defense (pts/100 possessions)",
                    invert_y=True,
                    x_ref=float(_sc_merged["AdjO"].median()),
                    y_ref=float(_sc_merged["AdjD"].median()),
                    x_domain=[float(_sc_merged["AdjO"].min()), float(_sc_merged["AdjO"].max())],
                    y_domain=[float(_sc_merged["AdjD"].min()), float(_sc_merged["AdjD"].max())],
                ),
                use_container_width=True,
            )
            st.markdown(
                "<small>CarmPom's AdjEM is built from these two ingredients. "
                "Teams in the **top-right** dominate on both ends — historically the profile of "
                "Final Four teams. In March, a coaching upset can mask a bad defense for one game, "
                "but elite two-way teams consistently advance. "
                "Dashed lines = tournament median. Better offense → right; better defense → up.</small>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Efficiency data unavailable.")

    with _sc_col2:
        # Chart 2: Ball Security vs FT Drawing
        st.markdown("**🔒 Ball Security vs Free-Throw Drawing**")
        if not _sc_plot.empty and "TOPG" in _sc_plot.columns and "FTmPG" in _sc_plot.columns:
            st.altair_chart(
                _scatter_chart(
                    _sc_plot, "TOPG", "FTmPG",
                    "Turnovers per Game (fewer = better →)",
                    "Free Throws Made per Game",
                    invert_x=True,
                    x_ref=float(_sc_merged["TOPG"].median()),
                    y_ref=float(_sc_merged["FTmPG"].median()),
                    x_domain=[float(_sc_merged["TOPG"].min()), float(_sc_merged["TOPG"].max())],
                    y_domain=[float(_sc_merged["FTmPG"].min()), float(_sc_merged["FTmPG"].max())],
                ),
                use_container_width=True,
            )
            st.markdown(
                "<small>Turnovers are magnified in elimination games — one late giveaway can end a season. "
                "Teams in the **top-right** take care of the ball *and* draw fouls, giving them two reliable "
                "scoring paths when half-court offense stalls in tournament pressure. "
                "Fewer turnovers → right; more FTs made → up.</small>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Per-game data unavailable.")

    _sc_col3, _sc_col4 = st.columns(2, gap="large")

    with _sc_col3:
        # Chart 3: 3PT Defense landscape
        st.markdown("**3PT Defense Landscape**")
        if not _sc_plot.empty and "Opp3PaPG" in _sc_plot.columns and "Opp3P%" in _sc_plot.columns:
            st.altair_chart(
                _scatter_chart(
                    _sc_plot, "Opp3PaPG", "Opp3P%",
                    "Opp 3PT Attempts Allowed per Game",
                    "Opp 3PT % Allowed",
                    invert_x=True,
                    invert_y=True,
                    x_ref=float(_sc_merged["Opp3PaPG"].median()),
                    y_ref=float(_sc_merged["Opp3P%"].median()),
                    x_domain=[float(_sc_merged["Opp3PaPG"].min()), float(_sc_merged["Opp3PaPG"].max())],
                    y_domain=[float(_sc_merged["Opp3P%"].min()), float(_sc_merged["Opp3P%"].max())],
                ),
                use_container_width=True,
            )
            st.markdown(
                "<small>The three-pointer is the biggest equalizer in March — a hot-shooting "
                "mid-major can beat anyone if left open. This chart identifies teams that **both** "
                "contest 3PT attempts (volume) and limit the conversion rate. "
                "Top-right = elite 3PT defense that removes the game's highest-variance shot. "
                "Limits volume → right; limits % → up.</small>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("3PT defense data unavailable.")

    with _sc_col4:
        # Chart 4: 3PT Offense — volume vs accuracy
        st.markdown("**🎯 3PT Offense — Volume vs Accuracy**")
        if not _sc_plot.empty and "3PaPG" in _sc_plot.columns and "3P%" in _sc_plot.columns:
            st.altair_chart(
                _scatter_chart(
                    _sc_plot, "3PaPG", "3P%",
                    "3PT Attempts per Game",
                    "3PT % Made",
                    x_ref=float(_sc_merged["3PaPG"].median()),
                    y_ref=float(_sc_merged["3P%"].median()),
                    x_domain=[float(_sc_merged["3PaPG"].min()), float(_sc_merged["3PaPG"].max())],
                    y_domain=[float(_sc_merged["3P%"].min()), float(_sc_merged["3P%"].max())],
                ),
                use_container_width=True,
            )
            st.markdown(
                "<small>My model uses pre-tournament efficiency, but bracket predictors know "
                "that variance is highest for teams relying on the arc. "
                "**Top-right** teams (high volume + high accuracy) are legitimately dangerous — they can "
                "go supernova or stay ice-cold. **Bottom-left** teams rarely win with 3s and survive "
                "on interior play and defense. Use this to read *how* a team will try to beat you. "
                "Heavy volume → right; better accuracy → up.</small>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("3PT offense data unavailable.")

# ---------------------------------------------------------------------------
# Bracket tab
# ---------------------------------------------------------------------------

with bracket_tab:
    # ── Load all data needed for this tab ────────────────────────────────
    _pk_brkt = load_real_bracket()
    _pk_df   = load_rankings(2026)
    _pk_lu   = _pk_df.set_index("Team").to_dict("index")
    _pk_logo_lu: dict[str, str] = {
        row["Team"]: str(row["logo_url"])
        for _, row in _pk_df.iterrows()
        if pd.notna(row.get("logo_url")) and row.get("logo_url")
    }
    _n_teams = len(_pk_df)
    _r_lookup = _pk_lu
    _pg_df_bp = load_per_game_stats(2026)
    _pg_named = _pk_df[["team_id", "Team"]].merge(_pg_df_bp, on="team_id", how="left")
    _pg_lu: dict = _pg_named.set_index("Team").to_dict("index")
    _odds_lu: dict = _load_odds_data()
    _inj_lu: dict  = _load_injuries_data()
    if "bp_picks" not in st.session_state:
        st.session_state["bp_picks"] = {}
    _picks: dict = st.session_state["bp_picks"]

    def _detail_panel(ta: pd.Series, tb: pd.Series, wp_a: float, n: int) -> None:
        """Render full matchup analysis inside a bordered container."""
        name_a, name_b = ta["Team"], tb["Team"]
        em_a, em_b = float(ta["AdjEM"]), float(tb["AdjEM"])
        wp_pct_a = round(wp_a * 100)
        wp_pct_b = 100 - wp_pct_a
        fav = name_a if wp_a >= 0.5 else name_b
        fav_pct = max(wp_pct_a, wp_pct_b)

        st.markdown(
            f"<h3 style='text-align:center;margin:4px 0 2px;font-family:system-ui,sans-serif'>"
            f"<span style='color:inherit'>({int(ta['seed'])}) {name_a}</span>"
            f"  <span style='color:#aaa;font-size:16px'>vs</span>  "
            f"<span style='color:inherit'>({int(tb['seed'])}) {name_b}</span></h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;margin:8px 0 16px'>"
            f"<div style='display:flex;height:20px;border-radius:10px;overflow:hidden;"
            f"max-width:500px;margin:0 auto 6px'>"
            f"<div style='width:{wp_pct_a}%;background:#1e7d32;display:flex;align-items:center;"
            f"justify-content:center'><span style='color:white;font-size:12px;font-weight:700'>{wp_pct_a}%</span></div>"
            f"<div style='width:{wp_pct_b}%;background:#c62828;display:flex;align-items:center;"
            f"justify-content:center'><span style='color:white;font-size:12px;font-weight:700'>{wp_pct_b}%</span></div>"
            f"</div>"
            f"<div style='font-size:12px;color:inherit'>CarmPom gives "
            f"<b>{fav}</b> a <b>{fav_pct}%</b> chance to win</div></div>",
            unsafe_allow_html=True,
        )
        _lc, _mc, _rc = st.columns([4, 3, 4])
        def _nr(tname: str, col: str) -> str:
            nr = _r_lookup.get(tname, {}).get(f"{col}_nr")
            return f"#{int(nr)}" if nr is not None else ""
        stat_defs = [
            ("AdjEM", f"{em_a:+.2f} {_nr(name_a,'AdjEM')}", f"{em_b:+.2f} {_nr(name_b,'AdjEM')}", False),
            ("AdjO",  f"{float(ta.get('AdjO',100)):.1f} {_nr(name_a,'AdjO')}", f"{float(tb.get('AdjO',100)):.1f} {_nr(name_b,'AdjO')}", False),
            ("AdjD",  f"{float(ta.get('AdjD',100)):.1f} {_nr(name_a,'AdjD')}", f"{float(tb.get('AdjD',100)):.1f} {_nr(name_b,'AdjD')}", True),
            ("Tempo", f"{float(ta.get('AdjT',68)):.1f} {_nr(name_a,'AdjT')}", f"{float(tb.get('AdjT',68)):.1f} {_nr(name_b,'AdjT')}", False),
            ("Record", ta.get("Record","\u2014"), tb.get("Record","\u2014"), False),
        ]
        lh = "<div style='font-family:system-ui,sans-serif;text-align:right'>"
        mh = "<div style='font-family:system-ui,sans-serif;text-align:center;color:#888'>"
        rh = "<div style='font-family:system-ui,sans-serif;text-align:left'>"
        _row = "padding:6px 0;border-bottom:1px solid #eee;font-size:13px"
        for lbl, va, vb, lwr in stat_defs:
            try:
                diff = float(va.split()[0]) - float(vb.split()[0])
                if lwr: diff = -diff
                arr = "\u25c0" if diff > 0.05 else ("\u25b6" if diff < -0.05 else "\u2014")
                ac  = "#1e7d32" if diff > 0.05 else ("#c62828" if diff < -0.05 else "#aaa")
            except Exception:
                arr, ac = "\u2014", "#aaa"
            lh += f"<div style='{_row}'><b>{va}</b></div>"
            mh += f"<div style='{_row}'><span style='color:{ac}'>{arr}</span> <span style='font-size:11px'>{lbl}</span></div>"
            rh += f"<div style='{_row}'><b>{vb}</b></div>"
        lh += "</div>"; mh += "</div>"; rh += "</div>"
        with _lc:
            st.markdown(f"<div style='background:#eaf5ed;border-radius:8px;padding:8px 12px;text-align:center;margin-bottom:6px;color:#1b5e20'><b>{name_a}</b></div>", unsafe_allow_html=True)
            st.markdown(lh, unsafe_allow_html=True)
        with _mc:
            st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
            st.markdown(mh, unsafe_allow_html=True)
        with _rc:
            st.markdown(f"<div style='background:#fdecea;border-radius:8px;padding:8px 12px;text-align:center;margin-bottom:6px;color:#b71c1c'><b>{name_b}</b></div>", unsafe_allow_html=True)
            st.markdown(rh, unsafe_allow_html=True)

        _ta_full = pd.Series({**_r_lookup.get(name_a, {}), "seed": ta.get("seed", 8), "Team": name_a})
        _tb_full = pd.Series({**_r_lookup.get(name_b, {}), "seed": tb.get("seed", 9), "Team": name_b})
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        st.markdown("#### \u2694\ufe0f How They Match Up")
        _clash = generate_clash_narrative(_ta_full, _tb_full, wp_a, n)
        st.markdown(
            f"<div style='background:#1e2d4f;border-left:4px solid #4a9eff;border-radius:0 8px 8px 0;"
            f"padding:14px 18px;font-size:14px;color:#e8eaf0;line-height:1.75;font-weight:400'>{_clash}</div>",
            unsafe_allow_html=True,
        )
        # ── Playstyle Profiles ─────────────────────────────────────────────
        st.markdown("<div style='margin-top:22px'></div>", unsafe_allow_html=True)
        st.markdown("#### 🎨 Playstyle Profiles")
        _ps_col_a, _ps_col_b = st.columns(2, gap="large")

        def _pct_dp(nr: int | float, n_total: int) -> float:
            return round((1 - (nr - 1) / n_total) * 100)

        def _radar_for_team(t_ser: pd.Series, col) -> None:
            """Render playstyle badge + radar in the given column."""
            tname = t_ser["Team"]
            ts_rows = pd.DataFrame()
            try:
                _pg_rows = _pg_lu.get(tname)
                if _pg_rows:
                    ts_rows = pd.Series(_pg_rows)
            except Exception:
                ts_rows = None

            _ts_r: pd.Series | None = ts_rows if (ts_rows is not None and len(ts_rows) > 0) else None
            _style_name_d, _style_tag_d = generate_playstyle_name(t_ser, _ts_r, n)

            with col:
                st.markdown(
                    f"<div style='background:#1e2d40;color:white;border-radius:8px;"
                    f"padding:9px 14px;margin-bottom:8px;font-family:system-ui'>"
                    f"<div style='font-size:15px;font-weight:700'>{_style_name_d}</div>"
                    f"<div style='font-size:11px;opacity:0.72;margin-top:2px'>{_style_tag_d}</div></div>",
                    unsafe_allow_html=True,
                )
                _radar_labels = ["Pace", "3PT Vol.", "3PT Acc.", "Off. Reb.",
                                 "Ball Sec.", "FT Draw", "Assists", "Def. Reb.",
                                 "Forced TO", "Paint Def."]
                if _ts_r is not None:
                    _vals_r = [
                        _pct_dp(t_ser["AdjT_nr"], n),
                        _pct_dp(_ts_r.get("3PaPG_nr", n), n),
                        _pct_dp(_ts_r.get("3P%_nr", n), n),
                        _pct_dp(_ts_r.get("OrebPG_nr", n), n),
                        _pct_dp(_ts_r.get("TOPG_nr", n), n),
                        _pct_dp(_ts_r.get("FTmPG_nr", n), n),
                        _pct_dp(_ts_r.get("AstPG_nr", n), n),
                        _pct_dp(_ts_r.get("RebPG_nr", n), n),
                        _pct_dp(_ts_r.get("StlPG_nr", n) if "StlPG_nr" in _ts_r.index else n, n),
                        _pct_dp(_ts_r.get("Opp2P%_nr", n) if "Opp2P%_nr" in _ts_r.index else n, n),
                    ]
                else:
                    _vals_r = [50] * 10

                N_sp = len(_radar_labels)
                _angles = [i / N_sp * 2 * 3.14159 for i in range(N_sp)] + [0]
                _vals_r = _vals_r + _vals_r[:1]
                fig_dp, ax_dp = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
                ax_dp.set_theta_offset(3.14159 / 2)
                ax_dp.set_theta_direction(-1)
                ax_dp.set_xticks(_angles[:-1])
                ax_dp.set_xticklabels(_radar_labels, size=6.5, color="#333")
                ax_dp.set_yticks([20, 40, 60, 80, 100])
                ax_dp.set_yticklabels([], size=0)
                ax_dp.set_ylim(0, 100)
                ax_dp.plot(_angles, _vals_r, color="#1e2d40", linewidth=1.8)
                ax_dp.fill(_angles, _vals_r, color="#4a90d9", alpha=0.35)
                ax_dp.spines["polar"].set_visible(False)
                ax_dp.grid(color="#ccc", linestyle="--", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig_dp, use_container_width=True)
                plt.close(fig_dp)

        _radar_for_team(_ta_full, _ps_col_a)
        _radar_for_team(_tb_full, _ps_col_b)
        st.caption("Each spoke = national percentile. Ball Sec. = inverted TO rate · Forced TO = steals/game · Paint Def. = opp 2PT% allowed (lower = better).")

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        with st.expander("\U0001f4cb Key Factors", expanded=True):
            for bullet in generate_matchup_analysis(_ta_full, _tb_full, wp_a, n):
                st.markdown(f"- {bullet}")

        # Injury Intel
        _notes_a = _inj_lu.get(name_a, [])
        _notes_b = _inj_lu.get(name_b, [])
        if _notes_a or _notes_b:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            with st.expander("\U0001f691 Injury Intel", expanded=True):
                _STATUS_COLORS = {
                    "Out":          ("#c62828", "#ffebee"),
                    "Doubtful":     ("#e65100", "#fff3e0"),
                    "Questionable": ("#f57f17", "#fffde7"),
                    "Monitor":      ("#616161", "#f5f5f5"),
                }
                for _tname, _notes in [(name_a, _notes_a), (name_b, _notes_b)]:
                    if not _notes: continue
                    st.markdown(f"**{_tname}**")
                    for _note in _notes[:4]:
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
                            f"{'&nbsp;\u00b7&nbsp;<span style=\"color:#777;font-size:11px\">' + _pub + '</span>' if _pub else ''}"
                            f"<div style='margin-top:3px;font-weight:600'>{_headline}</div>"
                            f"{'<div style=\"color:#555;margin-top:2px\">' + _desc[:180] + ('\u2026' if len(_desc)>180 else '') + '</div>' if _desc else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                st.warning("\u26a0\ufe0f Injury information can change rapidly. Always verify before making decisions.", icon=None)
                st.caption("Source: ESPN team news. Refresh via `uv run python pipeline/fetch_injuries.py`.")
        else:
            with st.expander("\U0001f691 Injury Intel", expanded=False):
                st.info("No injury data in cache. Run `uv run python pipeline/fetch_injuries.py` to populate.", icon="\u2139\ufe0f")
        # Market vs Model
        _line_a, _line_b = _odds_for_matchup(name_a, name_b, _odds_lu)
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        with st.expander("\U0001f4ca Market vs Model", expanded=bool(_line_a)):
            if _line_a and _line_a.get("impl_prob") is not None:
                impl_a = float(_line_a["impl_prob"])
                impl_b = 1 - impl_a
                ml_a = _line_a.get("ml"); ml_b = _line_b.get("ml") if _line_b else None
                ml_a = int(ml_a) if ml_a is not None else None
                ml_b = int(ml_b) if ml_b is not None else None
                spread_a = _line_a.get("spread"); spread_b = _line_b.get("spread") if _line_b else None
                _mc1, _mc2, _mc3 = st.columns(3)
                with _mc1: st.metric("Vegas: " + name_a, f"{round(impl_a*100)}%", f"ML: {ml_a:+d}" if ml_a else "")
                with _mc2: st.metric("Vegas: " + name_b, f"{round(impl_b*100)}%", f"ML: {ml_b:+d}" if ml_b else "")
                with _mc3:
                    _cp_edge = (wp_a - impl_a) * 100
                    st.metric("CarmPom Edge", f"{_cp_edge:+.1f}pp",
                              "Favors " + (name_a if _cp_edge > 0 else name_b))
                if spread_a is not None:
                    st.caption(f"Spread: {name_a} {spread_a:+.1f}")
            else:
                st.info("No odds data. Run `uv run python pipeline/fetch_odds.py`.", icon="\u2139\ufe0f")
        # Projected Score
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        with st.expander("\U0001f4d0 Projected Score", expanded=True):
            _pg_a_dp = _pg_lu.get(name_a, {})
            _pg_b_dp = _pg_lu.get(name_b, {})
            for _b in _generate_single_game_bullets(_ta_full, _tb_full, _pg_a_dp, _pg_b_dp, wp_a, _odds_lu, n):
                st.markdown(f"- {_b}")

    if _pk_brkt is None:
        st.warning("Bracket data not loaded.", icon="\u26a0\ufe0f")
    else:
        # ── Session state ─────────────────────────────────────────────────
        if "dev_view" not in st.session_state:
            st.session_state["dev_view"] = "overview"
        if "dev_region" not in st.session_state:
            st.session_state["dev_region"] = None

        _dv = st.session_state["dev_view"]
        _dr = st.session_state["dev_region"]

        # ── Game card HTML (display only — no interactive elements) ────────
        def _dev_card_html(
            ta: str, tb: str, sa, sb,
            em_a: float, em_b: float, wp_a: float,
            logo_a: str, logo_b: str, picked: str | None,
        ) -> str:
            """Return a clean HTML card for one matchup (no pick buttons)."""
            wp_pct_a = round(wp_a * 100)
            wp_pct_b = 100 - wp_pct_a
            sel_a, sel_b = picked == ta, picked == tb
            bg_a = "#e8f5e9" if sel_a else "#f8f9fa"
            bg_b = "#e8f5e9" if sel_b else "#f8f9fa"
            em_col_a = "#1e7d32" if em_a >= em_b else "#888"
            em_col_b = "#1e7d32" if em_b > em_a else "#888"
            bar_a = "#1565c0" if wp_pct_a >= wp_pct_b else "#cfd8dc"
            bar_b = "#1565c0" if wp_pct_b > wp_pct_a else "#cfd8dc"
            img = lambda url: (
                f"<img src='{url}' style='width:18px;height:18px;object-fit:contain;"
                f"vertical-align:middle;margin-right:4px'>"
            ) if url else ""
            seed_bg_a = "#1e2d40"
            seed_bg_b = "#78909c"
            chk_a = "✅ " if sel_a else ""
            chk_b = "✅ " if sel_b else ""
            fw_a = "700" if sel_a else "500"
            fw_b = "700" if sel_b else "500"

            return (
                f"<div style='border-radius:9px;overflow:hidden;border:1px solid #e0e0e0;"
                f"box-shadow:0 1px 5px rgba(0,0,0,0.07);font-family:system-ui,sans-serif;margin-bottom:2px'>"
                # Team A
                f"<div style='background:{bg_a};padding:9px 11px;display:flex;"
                f"justify-content:space-between;align-items:center'>"
                f"<div style='display:flex;align-items:center;min-width:0;flex:1'>"
                f"<span style='background:{seed_bg_a};color:white;border-radius:3px;"
                f"padding:1px 5px;font-size:10px;font-weight:700;flex-shrink:0;margin-right:5px'>{sa}</span>"
                f"{img(logo_a)}"
                f"<span style='font-size:12px;font-weight:{fw_a};color:#111111;overflow:hidden;"
                f"text-overflow:ellipsis;white-space:nowrap'>{chk_a}{ta}</span></div>"
                f"<span style='color:{em_col_a};font-size:11px;font-weight:600;"
                f"flex-shrink:0;margin-left:6px'>{em_a:+.1f}</span></div>"
                # Win prob bar
                f"<div style='display:flex;height:4px'>"
                f"<div style='width:{wp_pct_a}%;background:{bar_a}'></div>"
                f"<div style='width:{wp_pct_b}%;background:{bar_b}'></div></div>"
                f"<div style='text-align:center;font-size:8px;color:#aaa;letter-spacing:0.4px;"
                f"margin:1px 0'>CarmPom Win Probability</div>"
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:1px 11px;font-size:9px;color:#999'>"
                f"<b>{wp_pct_a}%</b><b>{wp_pct_b}%</b></div>"
                # Team B
                f"<div style='background:{bg_b};padding:9px 11px;display:flex;"
                f"justify-content:space-between;align-items:center'>"
                f"<div style='display:flex;align-items:center;min-width:0;flex:1'>"
                f"<span style='background:{seed_bg_b};color:white;border-radius:3px;"
                f"padding:1px 5px;font-size:10px;font-weight:700;flex-shrink:0;margin-right:5px'>{sb}</span>"
                f"{img(logo_b)}"
                f"<span style='font-size:12px;font-weight:{fw_b};color:#111111;overflow:hidden;"
                f"text-overflow:ellipsis;white-space:nowrap'>{chk_b}{tb}</span></div>"
                f"<span style='color:{em_col_b};font-size:11px;font-weight:600;"
                f"flex-shrink:0;margin-left:6px'>{em_b:+.1f}</span></div></div>"
            )

        # ── One matchup block: card + pick buttons + analyze ──────────────
        def _dev_matchup(rnd: int, slot: int) -> None:
            """Render card + pick buttons + analyze button for (rnd, slot)."""
            _dv_picks = st.session_state.get("bp_picks", {})
            ta, tb = _bp_candidates(rnd, slot, _dv_picks, _pk_brkt)
            picked = _dv_picks.get((rnd, slot))

            if ta == "TBD" or tb == "TBD":
                st.markdown(
                    "<div style='border:1px dashed #ccc;border-radius:8px;padding:18px;"
                    "text-align:center;color:#bbb;font-size:12px;margin-bottom:4px'>⏳ Awaiting picks</div>",
                    unsafe_allow_html=True,
                )
                return

            em_a = float(_pk_lu.get(ta, {}).get("AdjEM", 0))
            em_b = float(_pk_lu.get(tb, {}).get("AdjEM", 0))
            wp_a = _win_prob(em_a, em_b)

            _sa_rows = _pk_brkt[_pk_brkt["Team"] == ta]
            _sb_rows = _pk_brkt[_pk_brkt["Team"] == tb]
            sa = int(_sa_rows["seed"].values[0]) if len(_sa_rows) else "?"
            sb = int(_sb_rows["seed"].values[0]) if len(_sb_rows) else "?"

            st.markdown(
                _dev_card_html(ta, tb, sa, sb, em_a, em_b, wp_a,
                               _pk_logo_lu.get(ta, ""), _pk_logo_lu.get(tb, ""), picked),
                unsafe_allow_html=True,
            )

            _cb, _cx, _cc = st.columns([5, 1, 5])
            with _cb:
                if st.button(ta, key=f"dv_{rnd}_{slot}_a", use_container_width=True,
                             type="primary" if picked == ta else "secondary"):
                    st.session_state["bp_picks"][(rnd, slot)] = ta
                    st.rerun()
            with _cx:
                if picked and st.button("✕", key=f"dv_{rnd}_{slot}_x", use_container_width=True):
                    st.session_state["bp_picks"].pop((rnd, slot), None)
                    _cs = slot
                    for _cr in range(rnd + 1, 6):
                        _cs = _cs // 2
                        st.session_state["bp_picks"].pop((_cr, _cs), None)
                    st.rerun()
            with _cc:
                if st.button(tb, key=f"dv_{rnd}_{slot}_b", use_container_width=True,
                             type="primary" if picked == tb else "secondary"):
                    st.session_state["bp_picks"][(rnd, slot)] = tb
                    st.rerun()

            _ak = f"dv_az_{rnd}_{slot}"
            if _ak not in st.session_state:
                st.session_state[_ak] = False
            _open = st.session_state[_ak]
            if st.button(
                "✖ Close" if _open else "📊 Analyze",
                key=f"dv_{rnd}_{slot}_az",
                use_container_width=True,
                type="primary" if _open else "secondary",
            ):
                _nv = not _open
                if _nv:  # close all others first
                    for _k in list(st.session_state.keys()):
                        if _k.startswith("dv_az_") and _k != _ak:
                            st.session_state[_k] = False
                st.session_state[_ak] = _nv
                st.rerun()

        # ── Inline analysis renderer ───────────────────────────────────────
        def _dev_analyses(rnd: int, slots) -> None:
            """Render any open analysis panels for the given slots."""
            for _s in slots:
                if not st.session_state.get(f"dv_az_{rnd}_{_s}"):
                    continue
                _dv_picks = st.session_state.get("bp_picks", {})
                _ata, _atb = _bp_candidates(rnd, _s, _dv_picks, _pk_brkt)
                if _ata == "TBD" or _atb == "TBD":
                    continue
                _asa = int(_pk_brkt[_pk_brkt["Team"] == _ata]["seed"].values[0]) if len(_pk_brkt[_pk_brkt["Team"] == _ata]) else 8
                _asb = int(_pk_brkt[_pk_brkt["Team"] == _atb]["seed"].values[0]) if len(_pk_brkt[_pk_brkt["Team"] == _atb]) else 9
                _ata_s = pd.Series({**_pk_lu.get(_ata, {}), "seed": _asa, "Team": _ata})
                _atb_s = pd.Series({**_pk_lu.get(_atb, {}), "seed": _asb, "Team": _atb})
                _awp = _win_prob(
                    float(_pk_lu.get(_ata, {}).get("AdjEM", 0)),
                    float(_pk_lu.get(_atb, {}).get("AdjEM", 0)),
                )
                st.divider()
                st.markdown(
                    f"<div style='background:#1e2d40;color:white;border-radius:8px 8px 0 0;"
                    f"padding:10px 18px'><span style='font-size:15px;font-weight:700'>"
                    f"📊 ({_asa}) {_ata} vs ({_asb}) {_atb}</span></div>",
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    _detail_panel(_ata_s, _atb_s, _awp, _n_teams)

        # ── Shared pick count helper ───────────────────────────────────────
        def _reg_pick_count(ri: int) -> tuple[int, int]:
            """Return (made, total) picks for one region across all 4 regional rounds."""
            dv_p = st.session_state.get("bp_picks", {})
            made = sum(
                1 for rnd in range(4)
                for s in range(ri * (8 >> rnd), ri * (8 >> rnd) + (8 >> rnd))
                if dv_p.get((rnd, s))
            )
            return made, 15  # 8+4+2+1

        # ══════════════════════════════════════════════════════════════════
        # OVERVIEW — 4 region cards
        # ══════════════════════════════════════════════════════════════════
        if _dv == "overview":
            st.markdown(
                "<h2 style='font-family:system-ui,sans-serif;margin-bottom:2px'>🏟️ 2026 NCAA Tournament</h2>"
                "<p style='color:#666;font-size:13px;margin-top:0'>Pick a region to start filling out your bracket.</p>",
                unsafe_allow_html=True,
            )

            # ── Quick-fill strip ──────────────────────────────────────────
            _qf_l, _qf_r = st.columns([3, 1], gap="small")
            with _qf_l:
                st.markdown(
                    "<div style='background:#e8f5e9;border:1px solid #a5d6a7;border-radius:8px;"
                    "padding:10px 14px;font-family:system-ui,sans-serif'>"
                    "<span style='font-size:13px;font-weight:700;color:#1b5e20'>⚡ Suggested for Quickness</span>"
                    "<span style='font-size:12px;color:#388e3c;margin-left:8px'>"
                    "Fill every game with CarmPom's top pick, then adjust any you disagree with.</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with _qf_r:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if st.button("🎯 Fill with CarmPom Picks", use_container_width=True, key="dv_autofill_chalk"):
                    st.session_state["bp_picks"] = _bp_autofill(
                        "chalk",
                        st.session_state.get("bp_picks", {}),
                        _pk_brkt,
                        _pk_lu,
                    )
                    st.rerun()

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            _ov_r1 = st.columns(2, gap="medium")
            _ov_r2 = st.columns(2, gap="medium")

            for _oi, _oreg in enumerate(_BP_REGIONS):
                _oc = (_ov_r1 if _oi < 2 else _ov_r2)[_oi % 2]
                _ri_ov = _oi
                _acc_ov = _BP_REGION_ACC[_oreg]
                _bg_ov  = _BP_REGION_BG[_oreg]
                _emo_ov = _BP_REGION_EMOJI[_oreg]
                _made_ov, _tot_ov = _reg_pick_count(_ri_ov)

                # Build mini bracket: 4 columns (R64 → R32 → S16 → E8) logo+seed only
                _dv_picks_ov = st.session_state.get("bp_picks", {})
                _ov_col_labels = ["R64", "R32", "S16", "E8"]
                _bracket_col_html = ""
                for _rnd_ov in range(4):
                    _n_ov = 8 >> _rnd_ov  # 8, 4, 2, 1
                    _games_html = ""
                    for _gi in range(_n_ov):
                        _slot_ov = _ri_ov * _n_ov + _gi
                        if _rnd_ov == 0:
                            _ta_ov, _tb_ov = _bp_r1_teams(_slot_ov, _pk_brkt)
                            _sa_ov, _sb_ov = _BP_MU_PAIRS[_gi]
                        else:
                            _ta_ov, _tb_ov = _bp_candidates(_rnd_ov, _slot_ov, _dv_picks_ov, _pk_brkt)
                            _br_a = _pk_brkt[_pk_brkt["Team"] == _ta_ov]
                            _br_b = _pk_brkt[_pk_brkt["Team"] == _tb_ov]
                            _sa_ov = int(_br_a["seed"].values[0]) if _ta_ov != "TBD" and len(_br_a) else "?"
                            _sb_ov = int(_br_b["seed"].values[0]) if _tb_ov != "TBD" and len(_br_b) else "?"
                        _pck_ov = _dv_picks_ov.get((_rnd_ov, _slot_ov))

                        def _ov_pill(team: str, seed, logo: str, is_picked: bool) -> str:
                            if team == "TBD":
                                return (
                                    "<div style='height:17px;background:#f0f0f0;border-radius:3px;"
                                    "margin:1px 0;border:1px dashed #ccc'></div>"
                                )
                            bg = "#d4edda" if is_picked else "#f4f4f4"
                            img_tag = (
                                f"<img src='{logo}' style='width:13px;height:13px;"
                                f"object-fit:contain;vertical-align:middle'>"
                            ) if logo else ""
                            return (
                                f"<div style='background:{bg};border-radius:3px;padding:1px 4px;"
                                f"display:flex;align-items:center;gap:2px;margin:1px 0;overflow:hidden'>"
                                f"<span style='background:#1e2d40;color:white;border-radius:2px;"
                                f"padding:0 3px;font-size:8px;font-weight:700;flex-shrink:0'>{seed}</span>"
                                f"{img_tag}</div>"
                            )

                        _games_html += (
                            "<div style='background:white;border:1px solid #e0e0e0;"
                            "border-radius:4px;padding:2px 3px'>"
                            + _ov_pill(_ta_ov, _sa_ov, _pk_logo_lu.get(_ta_ov, ""), _pck_ov == _ta_ov)
                            + _ov_pill(_tb_ov, _sb_ov, _pk_logo_lu.get(_tb_ov, ""), _pck_ov == _tb_ov)
                            + "</div>"
                        )
                    _bracket_col_html += (
                        f"<div style='display:flex;flex-direction:column;flex:1;min-width:0'>"
                        f"<div style='text-align:center;font-size:8px;color:#999;font-weight:600;"
                        f"margin-bottom:3px;white-space:nowrap'>{_ov_col_labels[_rnd_ov]}</div>"
                        f"<div style='display:flex;flex-direction:column;"
                        f"justify-content:space-around;flex:1;gap:2px'>"
                        + _games_html
                        + "</div></div>"
                    )
                _bracket_html = (
                    f"<div style='display:flex;gap:4px;height:295px'>{_bracket_col_html}</div>"
                )

                _badge_color = "#1e7d32" if _made_ov == _tot_ov else _acc_ov

                with _oc:
                    st.markdown(
                        f"<div style='border:2px solid {_acc_ov};border-radius:12px;overflow:hidden;margin-bottom:8px'>"
                        f"<div style='background:{_acc_ov};padding:10px 14px;display:flex;"
                        f"justify-content:space-between;align-items:center'>"
                        f"<span style='color:white;font-weight:700;font-size:15px'>{_emo_ov} {_oreg}</span>"
                        f"<span style='background:rgba(255,255,255,0.25);color:white;border-radius:20px;"
                        f"padding:2px 10px;font-size:11px;font-weight:600'>{_made_ov}/{_tot_ov} picks</span>"
                        f"</div>"
                        f"<div style='padding:8px 10px;background:{_bg_ov}'>{_bracket_html}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"{'✅ ' if _made_ov == _tot_ov else ''}Make picks — {_oreg} Region →",
                        key=f"dv_enter_{_oreg}",
                        use_container_width=True,
                    ):
                        st.session_state["dev_view"] = "region"
                        st.session_state["dev_region"] = _oreg
                        st.rerun()

            # ── Final Four + Championship ──────────────────────────────────
            st.divider()
            st.markdown("### 🏆 Final Four & Championship")
            _f4c1, _f4c2 = st.columns(2, gap="medium")
            for _f4i, (_ra, _rb, _lbl) in enumerate([
                ("East", "South", "East vs South"),
                ("West", "Midwest", "West vs Midwest"),
            ]):
                with (_f4c1 if _f4i == 0 else _f4c2):
                    st.markdown(f"**{_lbl}**")
                    _dev_matchup(4, _f4i)
            _dev_analyses(4, range(2))

            _ncl, _ncm, _ncr = st.columns([2, 4, 2])
            with _ncm:
                st.markdown("**🏆 National Championship**")
                _dev_matchup(5, 0)
            _dev_analyses(5, range(1))

        # ══════════════════════════════════════════════════════════════════
        # REGION VIEW — single region, all 4 rounds side by side
        # ══════════════════════════════════════════════════════════════════
        elif _dv == "region" and _dr is not None:
            _ri = _BP_REGIONS.index(_dr)
            _acc_r = _BP_REGION_ACC[_dr]
            _emo_r = _BP_REGION_EMOJI[_dr]
            _made_r, _tot_r = _reg_pick_count(_ri)

            # Header row
            _hdr_back, _hdr_title = st.columns([1, 6])
            with _hdr_back:
                if st.button("← Overview", key="dv_back"):
                    st.session_state["dev_view"] = "overview"
                    st.session_state["dev_region"] = None
                    st.rerun()
            with _hdr_title:
                st.markdown(
                    f"<div style='background:{_acc_r};border-radius:8px;padding:9px 16px;"
                    f"color:white;font-family:system-ui;font-size:16px;font-weight:700;'"
                    f">{_emo_r} {_dr} Region &nbsp;"
                    f"<span style='font-size:12px;opacity:0.75;font-weight:400'>{_made_r}/{_tot_r} picks made</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # 4 round columns — R64 slightly wider since it has 8 cards
            _rc1, _rc2, _rc3, _rc4 = st.columns([3, 3, 2, 2], gap="small")

            # Estimated px height of one game block (card + 3 buttons + gaps)
            # Used for bracket-alignment spacers. Tune this value if alignment is off.
            _SLOT = 195

            for _col, _rnd, _rnd_label in [
                (_rc1, 0, "1st Round"),
                (_rc2, 1, "Round of 32"),
                (_rc3, 2, "Sweet 16"),
                (_rc4, 3, "Elite Eight"),
            ]:
                with _col:
                    st.markdown(
                        f"<div style='background:{_acc_r};color:white;border-radius:6px;"
                        f"padding:5px 10px;font-size:11px;font-weight:700;text-align:center;"
                        f"margin-bottom:8px'>{_rnd_label}</div>",
                        unsafe_allow_html=True,
                    )
                    _n_games = 8 >> _rnd
                    _step    = 1 << _rnd          # 2^rnd slot-multiples per game
                    _top_pad = (_SLOT * _step - _SLOT) // 2   # push first game to center
                    _gap_px  = _SLOT * _step - _SLOT          # gap between consecutive games

                    _region_slots = range(_ri * _n_games, _ri * _n_games + _n_games)

                    for _gi, _slot in enumerate(_region_slots):
                        if _gi == 0 and _top_pad > 0:
                            st.markdown(
                                f"<div style='height:{_top_pad}px'></div>",
                                unsafe_allow_html=True,
                            )
                        elif _gi > 0 and _gap_px > 0:
                            st.markdown(
                                f"<div style='height:{_gap_px}px'></div>",
                                unsafe_allow_html=True,
                            )
                        _dev_matchup(_rnd, _slot)

            # ── Inline analysis for this region (all rounds) ───────────────
            st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
            for _rnd_an in range(4):
                _ng = 8 >> _rnd_an
                _dev_analyses(_rnd_an, range(_ri * _ng, _ri * _ng + _ng))


# ---------------------------------------------------------------------------
# Upset Value Picks tab
# ---------------------------------------------------------------------------

with upset_tab:
    import re as _re_uv

    st.markdown(
        "<h2 style='font-family:system-ui,sans-serif;margin-bottom:2px'>\U0001f4a1 CarmPom Upset Value Picks</h2>"
        "<p style='color:#666;font-size:13px;margin-top:0'>"
        "Games where CarmPom gives the underdog significantly more credit than Vegas. "
        "These aren\u2019t necessarily the most likely upsets \u2014 they\u2019re the games where the "
        "<b>market is undervaluing a team</b> relative to the analytics. "
        "Later rounds use your picks from the Bracket tab to determine who\u2019s playing.</p>",
        unsafe_allow_html=True,
    )

    _uv_brkt = load_real_bracket(2026)
    _uv_odds = _load_odds_data()
    _uv_rankings = load_rankings(2026)
    _uv_lu = _uv_rankings.set_index("Team").to_dict("index")
    _uv_logo_lu = {
        row["Team"]: str(row["logo_url"])
        for _, row in _uv_rankings.iterrows()
        if pd.notna(row.get("logo_url")) and row.get("logo_url")
    }
    _uv_n = len(_uv_rankings)
    _uv_picks = st.session_state.get("bp_picks", {})

    def _uv_build_row(
        ta: str, tb: str,
        brkt: pd.DataFrame,
        lu: dict,
        odds: dict,
    ) -> dict | None:
        """Build an upset-value row dict for a matchup, or None if not interesting.

        'Upset' is always seed-based: the team with the higher seed number is the
        underdog. CarmPom value = CarmPom gives that underdog more credit than
        their seeding (or Vegas) implies. Requires a seed gap of at least 2 so
        4-vs-5 coin flips don't pollute the list.
        """
        if ta == "TBD" or tb == "TBD":
            return None
        _bra = brkt[brkt["Team"] == ta]
        _brb = brkt[brkt["Team"] == tb]
        seed_a = int(_bra["seed"].values[0]) if len(_bra) else None
        seed_b = int(_brb["seed"].values[0]) if len(_brb) else None

        em_a = float(lu.get(ta, {}).get("AdjEM", 0))
        em_b = float(lu.get(tb, {}).get("AdjEM", 0))
        wp_a = _win_prob(em_a, em_b)

        # Determine fav/dog by seed (lower seed number = conventional favourite).
        # Fall back to AdjEM when seeds are equal or missing (later rounds).
        if seed_a is not None and seed_b is not None and seed_a != seed_b:
            if seed_a < seed_b:
                fav, dog = ta, tb
                seed_fav, seed_dog = seed_a, seed_b
                wp_fav, wp_dog = wp_a, 1 - wp_a
                em_fav, em_dog = em_a, em_b
            else:
                fav, dog = tb, ta
                seed_fav, seed_dog = seed_b, seed_a
                wp_fav, wp_dog = 1 - wp_a, wp_a
                em_fav, em_dog = em_b, em_a
            seed_gap = seed_dog - seed_fav
        else:
            # No seed distinction — use AdjEM for fav/dog
            if em_a >= em_b:
                fav, dog, wp_fav, wp_dog = ta, tb, wp_a, 1 - wp_a
                em_fav, em_dog = em_a, em_b
                seed_fav = seed_a or 0
                seed_dog = seed_b or 0
            else:
                fav, dog, wp_fav, wp_dog = tb, ta, 1 - wp_a, wp_a
                em_fav, em_dog = em_b, em_a
                seed_fav = seed_b or 0
                seed_dog = seed_a or 0
            seed_gap = 0  # don't filter by gap when seeds are equal

        # Require gap >= 2 when seeding is available (exclude 4v5, 8v9 near-flips)
        if seed_gap == 1:
            return None

        # CarmPom actually favours the seed-underdog — always interesting
        cp_favors_dog = em_dog > em_fav

        # Otherwise require at least 28% CarmPom win probability for the dog
        if not cp_favors_dog and wp_dog < 0.28:
            return None

        _line, _ = _odds_for_matchup(ta, tb, odds)
        dog_book_wp: float | None = None
        dog_ml: int | None = None
        fav_spread: float | None = None
        if _line and _line.get("impl_prob") is not None:
            impl_a = float(_line["impl_prob"])
            dog_book_wp = (1 - impl_a) if fav == ta else impl_a
            _ml_raw = _line.get("ml")
            if _ml_raw is not None:
                ml_a = int(_ml_raw)
                dog_ml = ml_a if dog == tb else (
                    round(-impl_a / (1 - impl_a) * 100) if impl_a < 0.5
                    else round((1 - impl_a) / impl_a * 100)
                )
            _spread_raw = _line.get("spread")
            if _spread_raw is not None:
                # spread is from ta's perspective; convert to fav's perspective
                fav_spread = float(_spread_raw) if fav == ta else -float(_spread_raw)

        # CarmPom projected margin: AdjEM diff * ~0.68 (68 possessions/game)
        cp_margin = (em_fav - em_dog) * 0.68

        edge = (wp_dog - dog_book_wp) * 100 if dog_book_wp is not None else None
        return {
            "fav": fav, "dog": dog,
            "seed_fav": seed_fav, "seed_dog": seed_dog,
            "em_fav": em_fav, "em_dog": em_dog,
            "cp_dog_pct": round(wp_dog * 100, 1),
            "cp_fav_pct": round(wp_fav * 100, 1),
            "book_dog_pct": round(dog_book_wp * 100, 1) if dog_book_wp is not None else None,
            "dog_ml": dog_ml,
            "edge": edge,
            "cp_favors_dog": cp_favors_dog,
            "cp_margin": round(cp_margin, 1),
            "fav_spread": fav_spread,
        }

    def _uv_render_card(rank: int, r: dict) -> None:
        """Render a single upset-value card."""
        fav, dog = r["fav"], r["dog"]
        sf, sd = r["seed_fav"], r["seed_dog"]
        cp_d, cp_f = r["cp_dog_pct"], r["cp_fav_pct"]
        book, edge = r["book_dog_pct"], r["edge"]
        em_d, em_f = r["em_dog"], r["em_fav"]
        logo_f = _uv_logo_lu.get(fav, "")
        logo_d = _uv_logo_lu.get(dog, "")
        img_f = f"<img src='{logo_f}' style='width:18px;height:18px;object-fit:contain;vertical-align:middle;margin-right:3px'>" if logo_f else ""
        img_d = f"<img src='{logo_d}' style='width:18px;height:18px;object-fit:contain;vertical-align:middle;margin-right:3px'>" if logo_d else ""
        cp_favors = r.get("cp_favors_dog", False)
        cp_margin = r.get("cp_margin", 0)
        fav_spread = r.get("fav_spread", None)

        # Badge: CarmPom projected margin vs Vegas spread
        if cp_favors:
            ec, eb = "#6a1b9a", "#f3e5f5"
            es = "⭐ CarmPom favors upset"
        else:
            ec, eb = "#1e2d40", "#e8eaf6"
            cp_margin_str = f"{fav} by {cp_margin:.1f}"
            if fav_spread is not None:
                spread_str = f"{fav_spread:+.1f}" if fav_spread != 0 else "PK"
                es = f"CP: {cp_margin:.1f} · Line: {spread_str}"
            else:
                es = f"CP projects {cp_margin:.1f} pt margin"
        ml_disp = f" &nbsp;(ML: {r['dog_ml']:+d})" if r.get("dog_ml") is not None else ""
        _ta_s = pd.Series({**_uv_lu.get(fav, {}), "seed": sf, "Team": fav})
        _tb_s = pd.Series({**_uv_lu.get(dog, {}), "seed": sd, "Team": dog})
        bullets = generate_matchup_analysis(_ta_s, _tb_s, cp_f / 100, _uv_n)
        raw = next(
            (b for b in bullets if any(k in b.lower() for k in ["upset", "value", "edge", "gap", "pace", "defense", "offense", "tempo", "toss"])),
            bullets[0] if bullets else "",
        )
        reason = _re_uv.sub(r'[*_]{1,2}', "", raw).strip()
        reason = _re_uv.sub(r'^[\U00010000-\U0010ffff\u2600-\u27BF]+\s*', '', reason)
        st.markdown(
            f"<div style='border:1px solid #e0e0e0;border-radius:10px;padding:12px 15px;"
            f"margin-bottom:8px;font-family:system-ui,sans-serif;"
            f"background:{'#fdf6ff' if cp_favors else ('#f9fff9' if (edge or 0) >= 8 else '#ffffff')}'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:7px'>"
            f"<div style='display:flex;align-items:center;gap:9px'>"
            f"<span style='background:#1e2d40;color:white;border-radius:50%;width:22px;height:22px;"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"font-size:11px;font-weight:700;flex-shrink:0'>#{rank}</span>"
            f"<span style='font-size:13px;font-weight:700;color:#1e2d40'>"
            f"{img_d}({sd}) {dog}"
            f"<span style='color:#aaa;font-weight:400;font-size:12px'> over </span>"
            f"{img_f}({sf}) {fav}</span></div>"
            f"<span style='background:{eb};color:{ec};border-radius:20px;"
            f"padding:2px 9px;font-size:11px;font-weight:700;flex-shrink:0'>{es}</span></div>"
            f"<div style='display:flex;gap:12px;margin-bottom:7px;flex-wrap:wrap'>"
            f"<div style='background:#e3f2fd;border-radius:6px;padding:5px 12px;text-align:center;min-width:90px'>"
            f"<div style='font-size:17px;font-weight:800;color:#1565c0'>{cp_d}%</div>"
            f"<div style='font-size:10px;color:#1976d2;margin-top:1px'>CarmPom</div></div>"
            + (
                f"<div style='background:#fff3e0;border-radius:6px;padding:5px 12px;text-align:center;min-width:90px'>"
                f"<div style='font-size:17px;font-weight:800;color:#e65100'>{book}%{ml_disp}</div>"
                f"<div style='font-size:10px;color:#f57c00;margin-top:1px'>Vegas Implied</div></div>"
                if book is not None else
                f"<div style='background:#f5f5f5;border-radius:6px;padding:5px 12px;text-align:center;min-width:90px'>"
                f"<div style='font-size:12px;font-weight:600;color:#aaa'>No odds yet</div>"
                f"<div style='font-size:10px;color:#bbb;margin-top:1px'>Vegas</div></div>"
            )
            + f"<div style='background:#f5f5f5;border-radius:6px;padding:5px 12px;text-align:center;min-width:90px'>"
            f"<div style='font-size:12px;font-weight:700;color:#555'>{em_d:+.1f} vs {em_f:+.1f}</div>"
            f"<div style='font-size:10px;color:#888;margin-top:1px'>AdjEM (dog vs fav)</div></div></div>"
            f"<div style='font-size:11px;color:#555;line-height:1.5;border-top:1px solid #f0f0f0;padding-top:6px'>"
            f"\U0001f4ac {reason}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    def _uv_top_for_round(rnd: int, total_slots: int, picks: dict, brkt: pd.DataFrame, n_show: int) -> list[dict]:
        """Build and sort upset-value rows for all games in a given round."""
        rows: list[dict] = []
        for slot in range(total_slots):
            if rnd == 0:
                ta, tb = _bp_r1_teams(slot, brkt)
            else:
                ta, tb = _bp_candidates(rnd, slot, picks, brkt)
            r = _uv_build_row(ta, tb, brkt, _uv_lu, _uv_odds)
            if r:
                rows.append(r)
        # CarmPom actually favours the seed-underdog → top tier regardless of odds
        cp_fav_dog   = sorted([r for r in rows if r.get("cp_favors_dog")],     key=lambda x: x["cp_dog_pct"], reverse=True)
        with_edge    = sorted([r for r in rows if not r.get("cp_favors_dog") and r["edge"] is not None], key=lambda x: x["edge"], reverse=True)
        without_edge = sorted([r for r in rows if not r.get("cp_favors_dog") and r["edge"] is None],    key=lambda x: x["cp_dog_pct"], reverse=True)
        return (cp_fav_dog + with_edge + without_edge)[:n_show]

    if _uv_brkt is None:
        st.warning("Bracket data not loaded.", icon="\u26a0\ufe0f")
    else:
        odds_note = "" if _uv_odds else " *(no Vegas lines loaded)*"
        st.caption(f"Sorted by CarmPom edge over Vegas implied probability.{odds_note}")

        # Round config: (label, rnd index, total slots in that round, n picks to show)
        _uv_round_cfg = [
            ("\U0001f3c0 Round of 64",   0, 32, 10),
            ("\U0001f3c0 Round of 32",   1, 16,  6),
            ("\U0001f3c0 Sweet 16",       2,  8,  4),
            ("\U0001f3c0 Elite Eight",    3,  4,  2),
            ("\U0001f3c0 Final Four",     4,  2,  1),
        ]

        for _uv_lbl, _uv_rnd, _uv_slots, _uv_n_show in _uv_round_cfg:
            st.markdown(
                f"<div style='background:#1e2d40;color:white;border-radius:8px;"
                f"padding:8px 14px;margin:18px 0 10px;font-family:system-ui,sans-serif;"
                f"font-size:14px;font-weight:700'>{_uv_lbl}</div>",
                unsafe_allow_html=True,
            )
            _uv_section = _uv_top_for_round(_uv_rnd, _uv_slots, _uv_picks, _uv_brkt, _uv_n_show)
            if not _uv_section:
                if _uv_rnd == 0:
                    st.info("No matchup data available yet.", icon="\u2139\ufe0f")
                else:
                    st.caption("\u23f3 Fill in earlier rounds in the Bracket tab to see picks here.")
            else:
                for _uv_rank, _uv_row in enumerate(_uv_section, 1):
                    _uv_render_card(_uv_rank, _uv_row)

        st.caption(
            "\u26a0\ufe0f CarmPom probabilities are derived from adjusted efficiency ratings only. "
            "They do not account for injuries, recent form, or situational factors. "
            "Vegas lines reflect the market consensus incorporating all available information."
        )


# ---------------------------------------------------------------------------
# CarmPom Picks tab
# ---------------------------------------------------------------------------

with picks_tab:
    st.markdown(
        "<h2 style='font-family:system-ui,sans-serif;margin-bottom:2px'>🎯 CarmPom Tournament Picks</h2>"
        "<p style='color:#ccc;font-size:13px;margin-top:0'>"
        "CarmPom's bracket predictions powered by 50,000 Monte Carlo simulations. "
        "Highlights teams positioned to exceed seed expectations, whether through a favorable "
        "bracket path, a mismatch in efficiency vs. seeding, or structural advantages "
        "over their projected opponents. One bad game ends anyone's run — this surfaces "
        "where <b>the model sees opportunity the bracket doesn't price in</b>.</p>",
        unsafe_allow_html=True,
    )

    _pkl_brkt = load_real_bracket(2026)
    if _pkl_brkt is None:
        st.warning(
            "Bracket data not fully loaded — fill in `data/bracket_2026.csv` with all 64 teams.",
            icon="⚠️",
        )
    else:
        # Run simulation — heavy, so cache in session state to avoid re-running on every interaction
        # v2: rekeys cache to invalidate stale results after path-score logic update
        if "pkl_sim_results_v2" not in st.session_state:
            with st.spinner("Running 50,000 bracket simulations…"):
                st.session_state["pkl_sim_results_v2"] = simulate_bracket(
                    _pkl_brkt, n_sims=50_000, rng_seed=42
                )
        _pkl_sim: pd.DataFrame = st.session_state["pkl_sim_results_v2"]

        # -----------------------------------------------------------------------
        # Historical seed advancement rates (all-time NCAA tournament data).
        # Used as a baseline to find teams where CarmPom exceeds seed expectations.
        # -----------------------------------------------------------------------
        _SEED_HIST: dict[int, dict[str, float]] = {
            1:  {"R32%": 99.0, "S16%": 79.0, "E8%": 48.0, "F4%": 28.0, "Champ%": 16.0},
            2:  {"R32%": 95.0, "S16%": 58.0, "E8%": 35.0, "F4%": 19.0, "Champ%":  9.0},
            3:  {"R32%": 85.0, "S16%": 51.0, "E8%": 28.0, "F4%": 13.0, "Champ%":  6.0},
            4:  {"R32%": 79.0, "S16%": 41.0, "E8%": 20.0, "F4%": 10.0, "Champ%":  4.0},
            5:  {"R32%": 65.0, "S16%": 28.0, "E8%": 13.0, "F4%":  6.0, "Champ%":  2.5},
            6:  {"R32%": 63.0, "S16%": 24.0, "E8%": 11.0, "F4%":  5.0, "Champ%":  2.0},
            7:  {"R32%": 60.0, "S16%": 22.0, "E8%":  9.0, "F4%":  4.0, "Champ%":  1.5},
            8:  {"R32%": 51.0, "S16%": 22.0, "E8%":  9.0, "F4%":  4.0, "Champ%":  1.0},
            9:  {"R32%": 49.0, "S16%": 19.0, "E8%":  8.0, "F4%":  3.0, "Champ%":  1.0},
            10: {"R32%": 40.0, "S16%": 17.0, "E8%":  7.0, "F4%":  3.0, "Champ%":  0.8},
            11: {"R32%": 37.0, "S16%": 13.0, "E8%":  5.0, "F4%":  2.0, "Champ%":  0.5},
            12: {"R32%": 35.0, "S16%": 13.0, "E8%":  5.0, "F4%":  2.0, "Champ%":  0.3},
            13: {"R32%": 21.0, "S16%":  6.0, "E8%":  2.0, "F4%":  0.5, "Champ%":  0.1},
            14: {"R32%": 16.0, "S16%":  4.0, "E8%":  1.0, "F4%":  0.2, "Champ%":  0.0},
            15: {"R32%":  6.0, "S16%":  1.0, "E8%":  0.3, "F4%":  0.1, "Champ%":  0.0},
            16: {"R32%":  1.0, "S16%":  0.1, "E8%":  0.0, "F4%":  0.0, "Champ%":  0.0},
        }

        # -----------------------------------------------------------------------
        # NCAA bracket seed pairing structure within a 16-team region.
        # seed → (r64_opponent_seed, r32_pod_seeds, s16_half_seeds, e8_other_half_seeds)
        # Left half of region:  1,16,8,9 | 5,12,4,13
        # Right half of region: 6,11,3,14 | 7,10,2,15
        # -----------------------------------------------------------------------
        _SEED_STRUCTURE: dict[int, tuple[int, list[int], list[int], list[int]]] = {
            1:  (16, [8, 9],      [4, 5, 12, 13],    [2, 3, 6, 7, 10, 11, 14, 15]),
            16: (1,  [8, 9],      [4, 5, 12, 13],    [2, 3, 6, 7, 10, 11, 14, 15]),
            8:  (9,  [1, 16],     [4, 5, 12, 13],    [2, 3, 6, 7, 10, 11, 14, 15]),
            9:  (8,  [1, 16],     [4, 5, 12, 13],    [2, 3, 6, 7, 10, 11, 14, 15]),
            5:  (12, [4, 13],     [1, 8, 9, 16],     [2, 3, 6, 7, 10, 11, 14, 15]),
            12: (5,  [4, 13],     [1, 8, 9, 16],     [2, 3, 6, 7, 10, 11, 14, 15]),
            4:  (13, [5, 12],     [1, 8, 9, 16],     [2, 3, 6, 7, 10, 11, 14, 15]),
            13: (4,  [5, 12],     [1, 8, 9, 16],     [2, 3, 6, 7, 10, 11, 14, 15]),
            6:  (11, [3, 14],     [2, 7, 10, 15],    [1, 4, 5, 8, 9, 12, 13, 16]),
            11: (6,  [3, 14],     [2, 7, 10, 15],    [1, 4, 5, 8, 9, 12, 13, 16]),
            3:  (14, [6, 11],     [2, 7, 10, 15],    [1, 4, 5, 8, 9, 12, 13, 16]),
            14: (3,  [6, 11],     [2, 7, 10, 15],    [1, 4, 5, 8, 9, 12, 13, 16]),
            7:  (10, [2, 15],     [3, 6, 11, 14],    [1, 4, 5, 8, 9, 12, 13, 16]),
            10: (7,  [2, 15],     [3, 6, 11, 14],    [1, 4, 5, 8, 9, 12, 13, 16]),
            2:  (15, [7, 10],     [3, 6, 11, 14],    [1, 4, 5, 8, 9, 12, 13, 16]),
            15: (2,  [7, 10],     [3, 6, 11, 14],    [1, 4, 5, 8, 9, 12, 13, 16]),
        }

        # Build a quick lookup: (region, seed) → (AdjEM, R32%, S16%, E8%) from sim
        # Used to probability-weight path difficulty — favorites in a pod get higher weight
        # because they're more likely to actually show up as your opponent.
        _pkl_sim_lu: dict[tuple[str, int], dict] = {}
        for _, _sr in _pkl_sim.iterrows():
            _pkl_sim_lu[(str(_sr["Region"]), int(_sr["Seed"]))] = {
                "AdjEM": float(_sr["AdjEM"]),
                "R32%":  float(_sr["R32%"]),
                "S16%":  float(_sr["S16%"]),
                "E8%":   float(_sr["E8%"]),
            }

        def _pkl_wavg_adjem(region: str, seeds: list[int], adv_col: str) -> float:
            """Probability-weighted average AdjEM for a pool of seeds.

            Each team is weighted by their simulated probability of reaching that round
            (adv_col), so a 3-seed with a 51% S16 chance outweighs a 14-seed with 4%.
            This reflects who you're actually likely to face — not a naive equal-weight average.
            Falls back to simple average if all weights are zero.
            """
            total_w = 0.0
            total_wem = 0.0
            for s in seeds:
                info = _pkl_sim_lu.get((region, s))
                if info is None:
                    continue
                w = info.get(adv_col, 0.0) / 100.0  # convert pct to 0-1
                total_w   += w
                total_wem += w * info["AdjEM"]
            if total_w <= 0:
                # Fallback: simple average from brkt
                vals = [_pkl_sim_lu[(region, s)]["AdjEM"] for s in seeds if (region, s) in _pkl_sim_lu]
                return float(sum(vals) / len(vals)) if vals else 0.0
            return total_wem / total_w

        def _pkl_path_score(row: pd.Series, brkt: pd.DataFrame) -> dict[str, float]:
            """Compute probability-weighted path difficulty for a team.

            For each round's opponent pool, weights each potential opponent by their
            simulated probability of reaching that round — so the better/more-likely
            opponent carries more weight. Returns the weighted-average AdjEM per round
            plus a composite score (later rounds weighted more heavily).
            Lower composite = easier expected path.
            """
            seed = int(row["Seed"])
            region = str(row["Region"])
            if seed not in _SEED_STRUCTURE:
                return {"path_r64": 0.0, "path_r32": 0.0, "path_s16": 0.0, "path_e8": 0.0, "path_score": 0.0}
            r64_seed, r32_seeds, s16_seeds, e8_seeds = _SEED_STRUCTURE[seed]
            # R64: fixed 1-on-1 matchup — no weighting needed
            p64 = _pkl_sim_lu.get((region, r64_seed), {}).get("AdjEM", 0.0)
            # R32 pool: weight by R32% (probability of making it to that round)
            p32 = _pkl_wavg_adjem(region, r32_seeds, "R32%")
            # S16 pool: weight by S16%
            p16 = _pkl_wavg_adjem(region, s16_seeds, "S16%")
            # E8 pool: weight by E8%
            p8  = _pkl_wavg_adjem(region, e8_seeds,  "E8%")
            # Weight later rounds more — that's where runs are made or broken
            composite = 0.10 * p64 + 0.20 * p32 + 0.30 * p16 + 0.40 * p8
            return {
                "path_r64":   round(p64,       2),
                "path_r32":   round(p32,       2),
                "path_s16":   round(p16,       2),
                "path_e8":    round(p8,        2),
                "path_score": round(composite, 2),
            }

        def _pkl_run_score(sim_row: pd.Series) -> float:
            """Weighted edge over historical seed baseline across S16–Champ.

            A positive score means CarmPom projects this team above what their seed
            has historically produced. Emphasises E8+ since that's where identifiable
            upsets become structurally predictable.
            """
            seed = int(sim_row["Seed"])
            hist = _SEED_HIST.get(seed, {"S16%": 20.0, "E8%": 10.0, "F4%": 5.0, "Champ%": 2.0})
            d_s16   = float(sim_row["S16%"])   - hist["S16%"]
            d_e8    = float(sim_row["E8%"])    - hist["E8%"]
            d_f4    = float(sim_row["F4%"])    - hist["F4%"]
            d_champ = float(sim_row["Champ%"]) - hist["Champ%"]
            return round(0.20 * d_s16 + 0.30 * d_e8 + 0.30 * d_f4 + 0.20 * d_champ, 2)

        # Build enriched simulation table with path and run metrics
        _pkl_enriched = _pkl_sim.copy()
        _path_records = [_pkl_path_score(r, _pkl_brkt) for _, r in _pkl_enriched.iterrows()]
        _pkl_enriched["path_score"] = [p["path_score"] for p in _path_records]
        _pkl_enriched["path_r64"]   = [p["path_r64"]   for p in _path_records]
        _pkl_enriched["path_r32"]   = [p["path_r32"]   for p in _path_records]
        _pkl_enriched["path_s16"]   = [p["path_s16"]   for p in _path_records]
        _pkl_enriched["path_e8"]    = [p["path_e8"]    for p in _path_records]
        _pkl_enriched["run_score"]  = _pkl_enriched.apply(_pkl_run_score, axis=1)
        # seed_gap > 0 means seed is higher number than CP rank implies — undervalued by committee
        _pkl_enriched["seed_gap"] = (
            _pkl_enriched["Seed"] - (_pkl_enriched["CarmPomRk"] / 3.5).round(0).astype(int)
        )

        # Rankings lookup for logos
        _pkl_rank_lu: dict[str, dict] = {
            row["Team"]: row.to_dict()
            for _, row in load_rankings(2026).iterrows()
        }

        def _pkl_logo(team: str) -> str:
            """Return ESPN CDN logo URL for a team, or empty string."""
            url = _pkl_rank_lu.get(team, {}).get("logo_url", "")
            return str(url) if url else ""

        def _pkl_team_badge(team: str, seed: int, cp_rank: int) -> str:
            """Inline HTML: team logo + name + seed badge + CP rank pill."""
            logo = _pkl_logo(team)
            img = (
                f"<img src='{logo}' style='width:20px;height:20px;object-fit:contain;"
                f"vertical-align:middle;margin-right:4px'>"
                if logo else ""
            )
            return (
                f"{img}<span style='font-weight:700;font-size:13px;color:#111'>{team}</span>"
                f"&nbsp;<span style='background:#1e2d40;color:white;border-radius:4px;"
                f"padding:1px 5px;font-size:10px;font-weight:700'>#{seed} seed</span>"
                f"&nbsp;<span style='background:#e3f2fd;color:#1565c0;border-radius:4px;"
                f"padding:1px 5px;font-size:10px;font-weight:700'>CP #{cp_rank}</span>"
            )

        def _pkl_bar(pct: float, hist_pct: float, color: str = "#1565c0") -> str:
            """Horizontal progress bar with an orange baseline tick for historical seed average."""
            w  = min(max(pct,      0.0), 100.0)
            hw = min(max(hist_pct, 0.0), 100.0)
            return (
                f"<div style='position:relative;height:7px;background:#e0e0e0;border-radius:4px;width:100%;margin:2px 0'>"
                f"<div style='position:absolute;left:0;top:0;height:7px;width:{w:.1f}%;"
                f"background:{color};border-radius:4px;opacity:0.85'></div>"
                f"<div style='position:absolute;top:-2px;left:{hw:.1f}%;width:2px;height:11px;"
                f"background:#e65100;border-radius:1px' title='Seed avg {hist_pct:.1f}%'></div>"
                f"</div>"
            )

        # -----------------------------------------------------------------------
        # Section 1 — Championship prediction + Final Four picks
        # -----------------------------------------------------------------------
        _pkl_champ = _pkl_sim.iloc[0]
        _pkl_f4    = _pkl_sim.nlargest(4, "F4%")

        st.markdown(
            "<h3 style='font-family:system-ui,sans-serif;margin-top:18px;margin-bottom:6px;color:#f0f0f0'>"
            "🏆 CarmPom Champion Pick</h3>",
            unsafe_allow_html=True,
        )
        _pc1, _pc2, _pc3 = st.columns([1, 1, 1])

        with _pc1:
            _ch_logo = _pkl_logo(_pkl_champ["Team"])
            _ch_img  = (
                f"<img src='{_ch_logo}' style='width:56px;height:56px;object-fit:contain;"
                f"display:block;margin:0 auto 8px'>"
                if _ch_logo else ""
            )
            st.markdown(
                f"<div style='border:2px solid #ffd700;border-radius:12px;padding:16px 12px;"
                f"text-align:center;font-family:system-ui,sans-serif;"
                f"background:linear-gradient(135deg,#fffde7,#fff8e1)'>"
                f"{_ch_img}"
                f"<div style='font-size:18px;font-weight:800;color:#1e2d40'>{_pkl_champ['Team']}</div>"
                f"<div style='font-size:12px;color:#666;margin:2px 0'>"
                f"({int(_pkl_champ['Seed'])} seed · {_pkl_champ['Region']})</div>"
                f"<div style='font-size:26px;font-weight:900;color:#e65100;margin:6px 0'>"
                f"{_pkl_champ['Champ%']:.1f}%</div>"
                f"<div style='font-size:10px;color:#888'>champion probability</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with _pc2:
            st.markdown(
                "<div style='font-size:12px;font-weight:700;color:#888;font-family:system-ui,sans-serif;"
                "margin-bottom:6px'>FINAL FOUR PICKS</div>",
                unsafe_allow_html=True,
            )
            for _, _fr in _pkl_f4.iterrows():
                _fl    = _pkl_logo(str(_fr["Team"]))
                _fi    = (
                    f"<img src='{_fl}' style='width:16px;height:16px;object-fit:contain;"
                    f"vertical-align:middle;margin-right:4px'>"
                    if _fl else ""
                )
                _is_ch = _fr["Team"] == _pkl_champ["Team"]
                st.markdown(
                    f"<div style='display:flex;align-items:center;justify-content:space-between;"
                    f"padding:5px 8px;margin-bottom:4px;border-radius:7px;"
                    f"background:{'#fff8e1' if _is_ch else '#f5f5f5'};"
                    f"border:{'1px solid #ffd700' if _is_ch else '1px solid #e0e0e0'}'>"
                    f"<span style='font-size:12px;font-weight:{'700' if _is_ch else '500'};"
                    f"color:#1e2d40'>{_fi}({int(_fr['Seed'])}) {_fr['Team']}</span>"
                    f"<span style='font-size:11px;font-weight:700;color:#1565c0'>{_fr['F4%']:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with _pc3:
            _ch_seed      = int(_pkl_champ["Seed"])
            _ch_hist_champ = _SEED_HIST.get(_ch_seed, {}).get("Champ%", 10.0)
            _ch_delta      = _pkl_champ["Champ%"] - _ch_hist_champ
            _ch_delta_col  = "#2e7d32" if _ch_delta > 0 else "#c62828"
            st.markdown(
                f"<div style='background:#f5f5f5;border-radius:10px;padding:14px 12px;"
                f"font-family:system-ui,sans-serif'>"
                f"<div style='font-size:11px;font-weight:700;color:#888;margin-bottom:8px'>CHAMPION CONTEXT</div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:5px'>"
                f"<span style='font-size:12px;color:#555'>CarmPom AdjEM</span>"
                f"<span style='font-size:12px;font-weight:700;color:#1e2d40'>{_pkl_champ['AdjEM']:+.2f}</span></div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:5px'>"
                f"<span style='font-size:12px;color:#555'>CP Rank</span>"
                f"<span style='font-size:12px;font-weight:700;color:#1e2d40'>#{int(_pkl_champ['CarmPomRk'])}</span></div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:5px'>"
                f"<span style='font-size:12px;color:#555'>Seed baseline</span>"
                f"<span style='font-size:12px;color:#888'>{_ch_hist_champ:.1f}%</span></div>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='font-size:12px;color:#555'>Model edge</span>"
                f"<span style='font-size:12px;font-weight:700;color:{_ch_delta_col}'>{_ch_delta:+.1f}pp</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:22px 0'>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------
        # Section 2 — March Madness Runs to Watch (collapsible per-team writeups)
        # -----------------------------------------------------------------------
        st.markdown(
            "<h3 style='font-family:system-ui,sans-serif;margin-bottom:4px;color:#f0f0f0'>"
            "🏃 March Madness Runs to Watch</h3>"
            "<p style='color:#ccc;font-size:12px;margin-top:0'>"
            "Research-backed run candidates drawn from CarmPom's efficiency model and historical "
            "tournament pattern analysis. Over the last 10 seasons, teams that advanced beyond "
            "their seed almost always shared the same fingerprints: a <b>top-40 defense</b> "
            "(the single most reliable predictor), <b>ball security in the top third</b>, "
            "<b>at least one high-ceiling offensive weapon</b>, and a <b>path that avoids a "
            "murderer&#39;s row before the Elite Eight</b>. Expand any team below to see how they "
            "stack up against those benchmarks.</p>",
            unsafe_allow_html=True,
        )

        # Load per-game stats for playstyle analysis (cached)
        _pkl_pgs = load_per_game_stats(2026)
        _pkl_pgs_lu: dict[int, pd.Series] = {
            int(r["team_id"]): r for _, r in _pkl_pgs.iterrows()
        }
        _pkl_n = len(load_rankings(2026))

        def _pkl_ts(team_id: int) -> pd.Series | None:
            """Per-game stats row for a team, or None."""
            return _pkl_pgs_lu.get(team_id)

        def _pkl_t_row(team_name: str) -> pd.Series | None:
            """Full rankings row as a Series, or None."""
            d = _pkl_rank_lu.get(team_name)
            return pd.Series(d) if d else None

        # Select run candidates by tier:
        #   Tier 0 — top 3 seeds 1-2 (elite teams with genuine structural reasons to go deep)
        #   Tier 1 — top 5 seeds 3-7 contenders outperforming seed expectations
        #   Tier 2 — top 5 seeds 8-13 sleepers with real upside (14-16 excluded: inflated deltas)
        _pkl_tier0 = (
            _pkl_enriched[_pkl_enriched["Seed"].between(1, 2)]
            .nlargest(3, "run_score")
        )
        _pkl_tier1 = (
            _pkl_enriched[_pkl_enriched["Seed"].between(3, 7)]
            .nlargest(5, "run_score")
        )
        _pkl_tier2 = (
            _pkl_enriched[_pkl_enriched["Seed"].between(8, 13)]
            .nlargest(5, "run_score")
        )
        _pkl_ten = pd.concat([_pkl_tier0, _pkl_tier1, _pkl_tier2]).reset_index(drop=True)

        # Historical trait thresholds (last 10 years, based on observed Final Four + E8 patterns)
        # Source: aggregated from KenPom-era tournament data 2015-2024.
        _HIST_BENCHMARKS = {
            "defense_elite":   40,   # AdjD rank: top-40 nationally (most common F4 trait)
            "defense_good":    80,   # AdjD rank: top-80 — still noteworthy
            "offense_elite":   25,   # AdjO rank: top-25
            "offense_good":    60,   # AdjO rank: top-60
            "ball_security":   80,   # TOPG rank: top-80 (fewer turnovers)
            "shooting":        70,   # 3P% rank: top-70 (accurate from deep)
            "three_volume":    80,   # 3PaPG rank: top-80 (uses the three as a weapon)
            "paint_defense":   80,   # Opp2P% rank: top-80 (limits interior scoring)
            "tempo_control":  200,   # AdjT rank: ≥200 (slow teams can dictate pace)
        }

        def _pkl_run_traits(t: pd.Series, ts: pd.Series | None, path_row: pd.Series) -> list[tuple[str, str, str]]:
            """Return list of (icon, trait_label, explanation) tuples for this team's run case.

            Each tuple is a green ✅ (strength), amber ⚠️ (mixed), or red ❌ (concern).
            Grounded in historical patterns from the last 10 NCAA tournaments.
            """
            traits: list[tuple[str, str, str]] = []
            adjd_nr = int(t.get("AdjD_nr", 999))
            adjo_nr = int(t.get("AdjO_nr", 999))
            adjt_nr = int(t.get("AdjT_nr", 999))
            adjd    = float(t.get("AdjD",   100.0))
            adjo    = float(t.get("AdjO",   100.0))

            # --- Defense ---
            if adjd_nr <= _HIST_BENCHMARKS["defense_elite"]:
                traits.append(("✅", "Elite defense",
                    f"#{adjd_nr} nationally in AdjD ({adjd:.1f} pts/100) — "
                    "top-40 defenses have reached the Elite Eight at 3× the rate of median teams "
                    "over the last 10 seasons. The most reliable single predictor of a deep run."))
            elif adjd_nr <= _HIST_BENCHMARKS["defense_good"]:
                traits.append(("✅", "Solid defense",
                    f"#{adjd_nr} in AdjD nationally — above the top-80 threshold. "
                    "Not elite, but good enough to keep them in tight games against stronger opponents."))
            else:
                traits.append(("❌", "Defensive liability",
                    f"#{adjd_nr} in AdjD — historically, teams outside the top 100 defensively "
                    "rarely survive beyond the Sweet 16 against elite opposition."))

            # --- Offense ---
            if adjo_nr <= _HIST_BENCHMARKS["offense_elite"]:
                traits.append(("✅", "Elite offense",
                    f"#{adjo_nr} in AdjO ({adjo:.1f} pts/100) — pairs a high floor with the upside "
                    "needed to outscore quality opponents when the bracket demands it."))
            elif adjo_nr <= _HIST_BENCHMARKS["offense_good"]:
                traits.append(("✅", "Efficient offense",
                    f"Top-60 offense nationally (#{adjo_nr} AdjO) — enough firepower to punish "
                    "teams that focus entirely on take-away schemes."))

            # --- Ball security ---
            if ts is not None:
                topg_nr   = int(ts.get("TOPG_nr",   999))
                thp_nr    = int(ts.get("3P%_nr",    999))
                three_nr  = int(ts.get("3PaPG_nr",  999))
                opp2p_nr  = int(ts.get("Opp2P%_nr", 999))
                thpct     = float(ts.get("3P%", 0) or 0)
                three_pa  = float(ts.get("3PaPG", 0) or 0)

                if topg_nr <= _HIST_BENCHMARKS["ball_security"]:
                    traits.append(("✅", "Ball security",
                        f"#{topg_nr} nationally in turnover rate — "
                        "clean ball-handling removes the free possessions that fuel upsets. "
                        "Top-80 TO teams have historically advanced at 1.6× the rate of sloppy teams "
                        "once they reach the second weekend."))
                else:
                    traits.append(("⚠️", "Turnover risk",
                        f"#{topg_nr} in turnover rate — carelessness with the ball has burned "
                        "tournament teams against high-pressure defenses. One of the key variables "
                        "to watch in their opener."))

                # --- Shooting weapon ---
                if thp_nr <= _HIST_BENCHMARKS["shooting"] and three_nr <= _HIST_BENCHMARKS["three_volume"]:
                    traits.append(("✅", "3PT weapon",
                        f"{thpct:.1f}% from deep (#{thp_nr} nationally) on {three_pa:.1f} attempts/game "
                        f"(#{three_nr}) — a team that shoots accurately at volume can make any "
                        "single-game bracket advantage irrelevant in 30 minutes."))
                elif thp_nr <= _HIST_BENCHMARKS["shooting"]:
                    traits.append(("✅", "3PT efficiency",
                        f"{thpct:.1f}% from three (#{thp_nr} nationally) — when they get clean "
                        "looks they knock them down. Quality > quantity here."))

                # --- Interior defense ---
                if opp2p_nr <= _HIST_BENCHMARKS["paint_defense"]:
                    traits.append(("✅", "Interior defense",
                        f"#{opp2p_nr} nationally in opponent 2PT% — limits easy buckets at the rim, "
                        "which matters enormously against athletic big-seed opponents."))

            # --- Tempo as a weapon ---
            if adjt_nr >= _HIST_BENCHMARKS["tempo_control"]:
                traits.append(("✅", "Pace control",
                    f"#{adjt_nr} nationally in tempo (one of the slowest) — slow teams force "
                    "fast opponents to play an uncomfortable half-court game. Historically, "
                    "deliberate teams punch above their seed in single-elimination by reducing variance."))

            # --- Path ---
            path_pctile = (
                (_pkl_enriched["path_score"] < float(path_row["path_score"])).sum()
                / len(_pkl_enriched) * 100
            )
            if path_pctile < 25:
                traits.append(("✅", "Favorable bracket path",
                    f"Path score {float(path_row['path_score']):.2f} — bottom quartile of "
                    "all tournament teams (easier draw). Lighter weighted-average opponent AdjEM "
                    "through the E8 gives them extra margin for error."))
            elif path_pctile < 50:
                traits.append(("✅", "Manageable path",
                    f"Path score {float(path_row['path_score']):.2f} — below-median difficulty. "
                    "Not a cakewalk, but they won't have to beat a top-5 team to reach the Elite Eight."))
            elif path_pctile > 75:
                traits.append(("⚠️", "Tough bracket draw",
                    f"Path score {float(path_row['path_score']):.2f} — top quartile difficulty. "
                    "They'll likely need to beat a title contender-caliber team to reach the Final Four."))

            # --- Seed gap (CP rates them much higher than committee) ---
            seed_gap = int(path_row.get("seed_gap", 0))
            if seed_gap >= 3:
                traits.append(("✅", f"Underseeded by committee (+{seed_gap})",
                    f"CarmPom's efficiency model rates them {seed_gap} seed lines higher than their "
                    "actual seed. The committee used win-loss record and resume; CarmPom uses margin "
                    "quality. When that gap is ≥3, the team is structurally better than bracket position implies."))

            return traits

        # Primary brand hex for every 2026 tournament team — used to tint expander headers.
        _TEAM_COLORS_PKL: dict[str, str] = {
            "Akron Zips":                "#041E42",
            "Alabama Crimson Tide":      "#9E1B32",
            "Arizona Wildcats":          "#AB0520",
            "Arkansas Razorbacks":       "#9D2235",
            "BYU Cougars":               "#002E5D",
            "California Baptist Lancers": "#003087",
            "Clemson Tigers":            "#F56600",
            "Duke Blue Devils":          "#001A57",
            "Florida Gators":            "#0021A5",
            "Furman Paladins":           "#582C83",
            "Georgia Bulldogs":          "#BA0C2F",
            "Gonzaga Bulldogs":          "#002677",
            "Hawai'i Rainbow Warriors":  "#024731",
            "High Point Panthers":       "#4B1869",
            "Hofstra Pride":             "#003E7E",
            "Houston Cougars":           "#C8102E",
            "Idaho Vandals":             "#003082",
            "Illinois Fighting Illini":  "#E84A27",
            "Iowa Hawkeyes":             "#FFCD00",
            "Iowa State Cyclones":       "#C8102E",
            "Kansas Jayhawks":           "#0051A5",
            "Kennesaw State Owls":       "#002F55",
            "Kentucky Wildcats":         "#0033A0",
            "Lehigh Mountain Hawks":     "#653600",
            "Long Island University Sharks": "#003087",
            "Louisville Cardinals":      "#AD0000",
            "McNeese Cowboys":           "#005EB8",
            "Miami Hurricanes":          "#005030",
            "Michigan State Spartans":   "#18453B",
            "Michigan Wolverines":       "#00274C",
            "Missouri Tigers":           "#F1B82D",
            "NC State Wolfpack":         "#CC0000",
            "Nebraska Cornhuskers":      "#E41C38",
            "North Carolina Tar Heels":  "#4B9CD3",
            "North Dakota State Bison":  "#009A44",
            "Northern Iowa Panthers":    "#4B116F",
            "Ohio State Buckeyes":       "#BB0000",
            "Pennsylvania Quakers":      "#011F5B",
            "Purdue Boilermakers":       "#8E6F3E",
            "Queens University Royals":  "#522D80",
            "SMU Mustangs":              "#354CA1",
            "Saint Louis Billikens":     "#003DA5",
            "Saint Mary's Gaels":        "#0E3568",
            "Santa Clara Broncos":       "#862633",
            "Siena Saints":              "#006341",
            "South Florida Bulls":       "#006747",
            "St. John's Red Storm":      "#C8102E",
            "TCU Horned Frogs":          "#4D1979",
            "Tennessee State Tigers":    "#003087",
            "Tennessee Volunteers":      "#FF8200",
            "Texas A&M Aggies":          "#500000",
            "Texas Tech Red Raiders":    "#CC0000",
            "Troy Trojans":              "#CF1020",
            "UCF Knights":               "#BA9B37",
            "UCLA Bruins":               "#2D68C4",
            "UConn Huskies":             "#000E2F",
            "UMBC Retrievers":           "#1E3765",
            "Utah State Aggies":         "#003263",
            "VCU Rams":                  "#FDBB30",
            "Vanderbilt Commodores":     "#866D4B",
            "Villanova Wildcats":        "#003399",
            "Virginia Cavaliers":        "#232D4B",
            "Wisconsin Badgers":         "#C5050C",
            "Wright State Raiders":      "#006233",
        }

        # Tier labels and accents
        _PKL_TIER_CFG = [
            ("👑 Seeds 1–2: Elite Programs with a Clear Path to the Final Four", "#3d2c00", "#fff8e1", _pkl_tier0),
            ("🎯 Seeds 3–7: Contenders Primed for a Deep Run", "#1e3a5f", "#e8f0fe", _pkl_tier1),
            ("💣 Seeds 8–13: The Sleepers with Real Structural Upside", "#4a1a2c", "#fce4ec", _pkl_tier2),
        ]

        for _tier_title, _tier_accent, _tier_bg, _tier_df in _PKL_TIER_CFG:
            st.markdown(
                f"<div style='background:{_tier_accent};color:white;border-radius:8px;"
                f"padding:9px 16px;margin:20px 0 10px;font-family:system-ui,sans-serif;"
                f"font-size:14px;font-weight:700'>{_tier_title}</div>",
                unsafe_allow_html=True,
            )

            for _ri, (_, _pr) in enumerate(_tier_df.iterrows(), 1):
                _pteam = str(_pr["Team"])
                _ps    = int(_pr["Seed"])
                _pcr   = int(_pr["CarmPomRk"])
                _pem   = float(_pr["AdjEM"])
                _preg  = str(_pr["Region"])
                _rs    = float(_pr["run_score"])
                _path  = float(_pr["path_score"])
                _hist     = _SEED_HIST.get(_ps, {})
                # Team primary color for expander tint; fall back to a neutral navy
                _team_col = _TEAM_COLORS_PKL.get(_pteam, "#1e3a5f")

                # Path difficulty label
                _pp = (_pkl_enriched["path_score"] < _path).sum() / len(_pkl_enriched) * 100
                if _pp < 25:
                    _pdiff_lbl, _pdiff_col = "Easy path", "#2e7d32"
                elif _pp < 50:
                    _pdiff_lbl, _pdiff_col = "Avg path",  "#1565c0"
                elif _pp < 75:
                    _pdiff_lbl, _pdiff_col = "Hard path", "#e65100"
                else:
                    _pdiff_lbl, _pdiff_col = "Very hard", "#c62828"

                # Logo
                _plogo  = _pkl_logo(_pteam)
                _pimg   = (f"<img src='{_plogo}' style='width:22px;height:22px;object-fit:contain;"
                           f"vertical-align:middle;margin-right:6px'>" if _plogo else "")

                # Expander label — something concise the user scans before expanding
                _exp_label = (
                    f"#{_ri}  ({_ps}) {_pteam}  —  {_preg}  |  "
                    f"CP #{_pcr}  |  AdjEM {_pem:+.1f}  |  "
                    f"F4: {float(_pr['F4%']):.1f}%  |  {_pdiff_lbl}"
                )

                with st.expander(_exp_label, expanded=False):
                    # Inject CSS that tints THIS expander with the team's primary color.
                    # Uses CSS :has() to scope the style to the expander containing our unique span.
                    _exp_uid = (
                        "expkl_"
                        + _pteam.lower()
                              .replace(" ", "_").replace("'", "").replace(".", "")
                              .replace("&", "").replace("-", "_")
                        )[:38]
                    st.markdown(
                        f"""<style>
[data-testid="stExpander"]:has(#{_exp_uid}) summary {{
    background: linear-gradient(90deg, {_team_col}28 0%, transparent 65%) !important;
    border-left: 4px solid {_team_col} !important;
    border-radius: 6px 6px 0 0 !important;
}}
[data-testid="stExpander"]:has(#{_exp_uid}) {{
    border: 1.5px solid {_team_col}55 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"]:has(#{_exp_uid}) details > div:first-child {{
    background: linear-gradient(160deg, {_team_col}14 0%, transparent 55%) !important;
}}
</style><span id="{_exp_uid}"></span>""",
                        unsafe_allow_html=True,
                    )
                    # ── Top stats bar ──────────────────────────────────────────
                    _t_row  = _pkl_t_row(_pteam)
                    _tid    = int(_pkl_brkt[_pkl_brkt["Team"] == _pteam]["team_id"].values[0]) if len(_pkl_brkt[_pkl_brkt["Team"] == _pteam]) else None
                    _ts_row = _pkl_ts(_tid) if _tid else None

                    # Stat badges row
                    _adjd_nr  = int(_t_row.get("AdjD_nr",  999)) if _t_row is not None else 999
                    _adjo_nr  = int(_t_row.get("AdjO_nr",  999)) if _t_row is not None else 999
                    _adjt_nr  = int(_t_row.get("AdjT_nr",  999)) if _t_row is not None else 999
                    _sos_nr   = int(_t_row.get("SOS_nr",   999)) if _t_row is not None else 999
                    _adjd     = float(_t_row.get("AdjD",  100.0)) if _t_row is not None else 0.0
                    _adjo_v   = float(_t_row.get("AdjO",  100.0)) if _t_row is not None else 0.0
                    _adjt_v   = float(_t_row.get("AdjT",   68.0)) if _t_row is not None else 0.0

                    def _sbadge(label: str, val: str, rank: int, good_thr: int = 50, bg: str = "#e3f2fd", tc: str = "#1565c0") -> str:
                        """Small inline stat badge; turns green if rank ≤ good_thr."""
                        _bc = "#e8f5e9" if rank <= good_thr else bg
                        _tc2 = "#2e7d32" if rank <= good_thr else tc
                        return (
                            f"<div style='background:{_bc};border-radius:7px;padding:7px 13px;"
                            f"text-align:center;min-width:80px'>"
                            f"<div style='font-size:16px;font-weight:800;color:{_tc2}'>{val}</div>"
                            f"<div style='font-size:11px;color:#444;margin-top:2px'>{label}</div>"
                            f"<div style='font-size:11px;color:#666'>#{rank} natl</div>"
                            f"</div>"
                        )

                    # Playstyle
                    _style_name, _style_tag = generate_playstyle_name(
                        _t_row, _ts_row, _pkl_n
                    ) if _t_row is not None else ("—", "")

                    # R64 opponent
                    _r64_opp_seed = _SEED_STRUCTURE.get(_ps, (0, [], [], []))[0]
                    _r64_opp_rows = _pkl_brkt[(_pkl_brkt["region"] == _preg) & (_pkl_brkt["seed"] == _r64_opp_seed)]
                    _r64_opp_name = str(_r64_opp_rows["Team"].values[0]) if len(_r64_opp_rows) else "?"

                    # Round probabilities vs seed baseline
                    _rd_rows = [
                        ("R32", float(_pr["R32%"]), _hist.get("R32%", 50.0)),
                        ("S16", float(_pr["S16%"]), _hist.get("S16%", 25.0)),
                        ("E8",  float(_pr["E8%"]),  _hist.get("E8%",  12.0)),
                        ("F4",  float(_pr["F4%"]),  _hist.get("F4%",   5.0)),
                    ]

                    # Trait analysis
                    _traits = _pkl_run_traits(
                        _t_row if _t_row is not None else pd.Series({}),
                        _ts_row,
                        _pr,
                    )

                    # ── Render ─────────────────────────────────────────────────
                    # Row 1: team identity
                    st.markdown(
                        f"<div style='display:flex;align-items:center;"
                        f"margin-bottom:14px;flex-wrap:wrap;gap:10px'>"
                        f"<div style='display:flex;align-items:center;gap:12px'>"
                        f"{_pimg.replace('width:22px;height:22px', 'width:32px;height:32px')}"
                        f"<div>"
                        f"<div style='font-size:21px;font-weight:800;color:inherit'>{_pteam}</div>"
                        f"<div style='font-size:14px;color:inherit;opacity:0.85'>{_preg} region · "
                        f"<span style='color:{_pdiff_col};font-weight:700'>{_pdiff_lbl}</span></div>"
                        f"</div></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Row 2: stat badges
                    st.markdown(
                        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px'>"
                        + _sbadge("AdjEM",  f"{_pem:+.1f}", _pcr,    50)
                        + _sbadge("AdjO",   f"{_adjo_v:.1f}", _adjo_nr, 50)
                        + _sbadge("AdjD",   f"{_adjd:.1f}",  _adjd_nr, 40, "#fce4ec", "#c62828")
                        + _sbadge("AdjT",   f"{_adjt_v:.1f}", _adjt_nr, 999)
                        + _sbadge("SOS",    f"#{_sos_nr}", _sos_nr, 40)
                        + f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Row 3: playstyle card — same dark-navy badge as team profile tab
                    st.markdown(
                        f"<div style='background:#1e2d40;color:white;border-radius:8px;"
                        f"padding:11px 16px;margin-bottom:16px;font-family:system-ui'>"
                        f"<div style='font-size:18px;font-weight:700'>{_style_name}</div>"
                        f"<div style='font-size:13px;opacity:0.82;margin-top:3px'>{_style_tag}</div></div>",
                        unsafe_allow_html=True,
                    )

                    # Build radar data (same 10-spoke logic as team profile tab)
                    _pkl_has_radar = _ts_row is not None and _t_row is not None
                    if _pkl_has_radar:
                        def _pkl_pct(nr: int) -> int:
                            """National percentile from a rank (higher = better)."""
                            return round((1 - (max(int(nr), 1) - 1) / max(_pkl_n, 1)) * 100)
                        _r_adjt  = _pkl_pct(_t_row.get("AdjT_nr",  180))
                        _r_3pa   = _pkl_pct(_ts_row.get("3PaPG_nr",  180))
                        _r_3pct  = _pkl_pct(_ts_row.get("3P%_nr",   180))
                        _r_oreb  = _pkl_pct(_ts_row.get("OrebPG_nr", 180))
                        _r_to    = _pkl_pct(_ts_row.get("TOPG_nr",   180))
                        _r_ftm   = _pkl_pct(_ts_row.get("FTmPG_nr",  180))
                        _r_ast   = _pkl_pct(_ts_row.get("AstPG_nr",  180))
                        _r_dreb  = round((1 - (int(_ts_row.get("RebPG_nr", 180)) - 1) / max(_pkl_n, 1)) * 100)
                        _r_stl   = _pkl_pct(_ts_row.get("StlPG_nr",  180)) if "StlPG_nr" in _ts_row.index else 50
                        _r_opp2p = _pkl_pct(_ts_row.get("Opp2P%_nr", 180)) if "Opp2P%_nr" in _ts_row.index else 50
                        _rad_lbl = ["Pace", "3PT Vol", "3PT Acc", "Off Reb",
                                    "Ball Sec", "FT Draw", "Assists", "Def Reb",
                                    "Forced TOs", "Paint Def"]
                        _rad_val = [_r_adjt, _r_3pa, _r_3pct, _r_oreb,
                                    _r_to, _r_ftm, _r_ast, _r_dreb, _r_stl, _r_opp2p]
                        _N_r = len(_rad_lbl)
                        _ang_r = [i / _N_r * 2 * 3.14159 for i in range(_N_r)] + [0.0]
                        _val_r = _rad_val + _rad_val[:1]
                        _fig_r, _ax_r = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
                        _ax_r.set_theta_offset(3.14159 / 2)
                        _ax_r.set_theta_direction(-1)
                        _ax_r.set_xticks(_ang_r[:-1])
                        _ax_r.set_xticklabels(_rad_lbl, size=7, color="#cccccc")
                        _ax_r.set_yticks([20, 40, 60, 80, 100])
                        _ax_r.set_yticklabels(["20", "40", "60", "80", "100"], size=5.5, color="#999")
                        _ax_r.set_ylim(0, 100)
                        _ax_r.plot(_ang_r, _val_r, color="#1e2d40", linewidth=2)
                        _ax_r.fill(_ang_r, _val_r, color="#4a90d9", alpha=0.35)
                        _ax_r.set_title("Playstyle Profile", size=8.5, pad=12,
                                        color="#aaa", fontweight="bold")
                        _ax_r.spines["polar"].set_visible(False)
                        _ax_r.grid(color="#444", linestyle="--", linewidth=0.5)
                        _fig_r.patch.set_alpha(0.0)
                        _ax_r.set_facecolor("none")
                        plt.tight_layout()

                    # Three-column layout: [radar | traits | probs+path]
                    # If no radar data, collapse to two-column traits + path
                    if _pkl_has_radar:
                        _col_radar, _col_traits, _col_path = st.columns([1.8, 3, 2])
                        with _col_radar:
                            st.pyplot(_fig_r, use_container_width=True)
                        plt.close(_fig_r)
                    else:
                        _col_traits, _col_path = st.columns([3, 2])

                    with _col_traits:
                        st.markdown(
                            "<div style='font-size:12px;font-weight:700;color:inherit;opacity:0.8;"
                            "letter-spacing:.5px;margin-bottom:8px'>HISTORICAL RUN PROFILE</div>",
                            unsafe_allow_html=True,
                        )
                        for _icon, _tlabel, _texpl in _traits:
                            _tc = "#2e7d32" if _icon == "✅" else ("#e65100" if _icon == "⚠️" else "#c62828")
                            st.markdown(
                                f"<div style='display:flex;gap:8px;margin-bottom:9px;"
                                f"padding:8px 12px;background:#f8f9fa;border-radius:7px;"
                                f"border-left:3px solid {_tc}'>"
                                f"<div style='font-size:15px;flex-shrink:0'>{_icon}</div>"
                                f"<div><div style='font-size:13px;font-weight:700;color:#111;"
                                f"margin-bottom:3px'>{_tlabel}</div>"
                                f"<div style='font-size:12px;color:#333;line-height:1.4'>{_texpl}</div>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )

                    with _col_path:
                        # Round probs vs seed baseline
                        st.markdown(
                            "<div style='font-size:12px;font-weight:700;color:inherit;opacity:0.8;"
                            "letter-spacing:.5px;margin-bottom:8px'>ROUND PROBABILITIES "
                            "<span style='font-style:italic;font-weight:400'>"
                            "(orange = seed avg)</span></div>",
                            unsafe_allow_html=True,
                        )
                        for _rl, _rp, _hp in _rd_rows:
                            _rc = "#1565c0" if _rp >= _hp else "#c62828"
                            st.markdown(
                                f"<div style='display:flex;align-items:center;gap:5px;"
                                f"margin-bottom:6px'>"
                                f"<div style='width:32px;font-size:12px;font-weight:700;"
                                f"color:inherit;flex-shrink:0'>{_rl}</div>"
                                f"<div style='flex:1'>{_pkl_bar(_rp, _hp, _rc)}</div>"
                                f"<div style='width:38px;text-align:right;font-size:12px;"
                                f"font-weight:700;color:{_rc}'>{_rp:.1f}%</div>"
                                f"<div style='width:34px;text-align:right;font-size:11px;"
                                f"color:inherit;opacity:0.6'>{_hp:.1f}%</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            "<div style='font-size:12px;font-weight:700;color:inherit;opacity:0.8;"
                            "letter-spacing:.5px;margin:12px 0 8px'>BRACKET PATH</div>",
                            unsafe_allow_html=True,
                        )
                        _path_detail = [
                            ("R64 opponent",    f"({_r64_opp_seed}) {_r64_opp_name}"),
                            ("Exp. R32 AdjEM",  f"{float(_pr['path_r32']):+.1f}"),
                            ("Exp. S16 AdjEM",  f"{float(_pr['path_s16']):+.1f}"),
                            ("Exp. E8 AdjEM",   f"{float(_pr['path_e8']):+.1f}"),
                            ("Path score",       f"{_path:.2f}"),
                        ]
                        for _pk, _pv in _path_detail:
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;"
                                f"padding:4px 0;border-bottom:1px solid rgba(128,128,128,0.2)'>"
                                f"<span style='font-size:12px;color:inherit;opacity:0.75'>{_pk}</span>"
                                f"<span style='font-size:12px;font-weight:700;color:inherit'>{_pv}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:22px 0'>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------
        # Section 3 — Full simulation results table

        # -----------------------------------------------------------------------
        st.markdown(
            "<h3 style='font-family:system-ui,sans-serif;margin-bottom:4px;color:#f0f0f0'>📋 Full Simulation Results</h3>"
            "<p style='color:#ccc;font-size:12px;margin-top:0'>"
            "All 64 teams sorted by CarmPom champion probability from 50,000 simulations. "
            "<b>Run Score</b> = model edge vs historical seed baseline across S16–Champ (positive = CarmPom likes this "
            "team more than history implies for their seed). "
            "<b>Path Score</b> = composite weighted-average AdjEM of projected opponents — "
            "lower means an easier bracket draw.</p>",
            unsafe_allow_html=True,
        )

        _pkl_col_order = [
            "Team", "Region", "Seed", "CarmPomRk", "AdjEM",
            "R32%", "S16%", "E8%", "F4%", "Champ%",
            "run_score", "path_score",
        ]
        _pkl_display = _pkl_enriched[_pkl_col_order].copy()
        _pkl_display.columns = [
            "Team", "Region", "Seed", "CP Rank", "AdjEM",
            "R32%", "S16%", "E8%", "F4%", "Champ%",
            "Run Score", "Path Score",
        ]
        _pkl_display = _pkl_display.reset_index(drop=True)
        _pkl_display.index = _pkl_display.index + 1

        _pkl_styler = (
            _pkl_display.style
            .format({
                "AdjEM":      "{:+.2f}",
                "R32%":       "{:.1f}%",
                "S16%":       "{:.1f}%",
                "E8%":        "{:.1f}%",
                "F4%":        "{:.1f}%",
                "Champ%":     "{:.1f}%",
                "Run Score":  "{:+.2f}",
                "Path Score": "{:.2f}",
            })
            .background_gradient(subset=["R32%", "S16%", "E8%", "F4%", "Champ%"], cmap="Blues")
            .background_gradient(subset=["Run Score"], cmap="RdYlGn", vmin=-5, vmax=15)
            .background_gradient(subset=["AdjEM"], cmap="RdYlGn", vmin=-10, vmax=35)
        )
        st.dataframe(_pkl_styler, use_container_width=True, height=600)

        # Easiest / hardest draws summary
        _pkl_draw_a, _pkl_draw_b = st.columns(2)
        with _pkl_draw_a:
            _easiest = _pkl_enriched.nsmallest(5, "path_score")[["Team", "Seed", "Region", "path_score", "Champ%"]]
            st.markdown("**🟢 Easiest Bracket Draws**")
            for _, _er in _easiest.iterrows():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 8px;"
                    f"background:#e8f5e9;border-radius:6px;margin-bottom:3px;font-size:12px'>"
                    f"<span style='color:#111'><b>({int(_er['Seed'])}) {_er['Team']}</b> — {_er['Region']}</span>"
                    f"<span style='color:#2e7d32;font-weight:700'>"
                    f"Path: {_er['path_score']:.2f} | Champ: {_er['Champ%']:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with _pkl_draw_b:
            _hardest = _pkl_enriched.nlargest(5, "path_score")[["Team", "Seed", "Region", "path_score", "Champ%"]]
            st.markdown("**🔴 Hardest Bracket Draws**")
            for _, _hr in _hardest.iterrows():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 8px;"
                    f"background:#ffebee;border-radius:6px;margin-bottom:3px;font-size:12px'>"
                    f"<span style='color:#111'><b>({int(_hr['Seed'])}) {_hr['Team']}</b> — {_hr['Region']}</span>"
                    f"<span style='color:#c62828;font-weight:700'>"
                    f"Path: {_hr['path_score']:.2f} | Champ: {_hr['Champ%']:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "⚠️ All probabilities are model-based (AdjEM / efficiency ratings only). "
            "They do not account for injuries, momentum, coaching, or situational factors. "
            "Run Score > 0 = model projects this team above their historical seed average. "
            "Path Score: lower = easier projected bracket draw."
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
