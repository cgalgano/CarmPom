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

from db.database import SessionLocal
from db.models import CarmPomRating, Team

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
st.markdown("#### A KenPom-style NCAA Men's Basketball analytics engine — built from scratch in Python")

st.divider()

hero_left, hero_right = st.columns([3, 2], gap="large")

with hero_left:
    st.markdown("""
    **CarmPom** computes opponent-adjusted efficiency ratings for every Division I team — the same
    kind of numbers you see on KenPom, but built entirely from public box-score data with an
    open-source stack.

    The core loop works like this:
    1. **Estimate possessions** from box score stats (FGA, FTA, TOV, OREB)
    2. **Compute raw efficiency** — points scored and allowed per 100 possessions for every game
    3. **Iterate adjustments** — each team's offensive rating is adjusted for the defensive
       strength of its opponents, and vice versa, until the system converges
    4. **Add secondary metrics** — tempo (pace), luck (actual W% vs Pythagorean W%), and
       strength of schedule (avg opponent AdjEM)

    The result: **AdjEM** (adjusted efficiency margin), the headline number every team is ranked by.
    """)

with hero_right:
    st.markdown("#### How it compares to KenPom")
    st.markdown("""
    | | KenPom | CarmPom |
    |---|---|---|
    | **Source** | Proprietary | Open-source Python |
    | **Data** | Synergy / premium feeds | Public box scores |
    | **Ratings** | Closed formula | Fully auditable code |
    | **ML predictor** | None (ratings only) | LightGBM trained on 20+ seasons |
    | **Bracket sim** | Paywalled | Built-in — coming soon |
    """)

st.divider()

# --- Feature importance section ---
st.markdown("### What our model learned about winning in March")
st.markdown(
    "We trained a machine learning model on every NCAA Tournament game from 2003–2025 "
    "to predict upset probabilities. Here's what it found actually matters — and how much. "
    "The bar shows each factor's relative importance to the final prediction."
)

# Fan-friendly labels and plain-English explanations for each feature
FEATURE_EXPLAINERS = {
    "adjem_diff": (
        "Overall efficiency edge  *(CarmPom AdjEM)*",
        "The single biggest predictor. How much better one team is at scoring **and** stopping the other team, "
        "adjusted for who they played all season. It's like a one-number report card for the entire season.",
    ),
    "seed_diff": (
        "Tournament seed gap",
        "The committee's official opinion. A 1-seed vs a 16-seed gap is massive; "
        "a 4 vs a 5 is basically a coin flip. Seeds matter, but they're just humans reading the same stats.",
    ),
    "kenpom_rank_diff": (
        "KenPom ranking gap",
        "KenPom's own adjusted efficiency difference between the two teams. "
        "Similar signal to CarmPom AdjEM — the model uses both and lets them compete.",
    ),
    "efg_diff": (
        "Shooting quality edge  *(eFG%)*",
        "Effective Field Goal % counts 3-pointers as worth 50% more than 2s. "
        "Teams that shoot efficiently all season tend to keep doing it when it counts.",
    ),
    "pyth_wp_diff": (
        "\"Deserved\" win % gap  *(Pythagorean)*",
        "Based purely on points scored vs. points allowed — ignores actual wins. "
        "Teams that win close games all year are often lucky; this strips that out.",
    ),
    "win_pct_diff": (
        "Record advantage",
        "Simple win-loss record gap. Less informative than efficiency but the model "
        "still finds some signal — winning ugly beats losing pretty.",
    ),
    "or_pct_diff": (
        "Second-chance points edge  *(Off. rebounding)*",
        "The team that crashes the offensive glass gets extra possessions. "
        "In a one-and-done tournament game, those extra shots can flip outcomes.",
    ),
    "ft_rate_diff": (
        "Getting to the line  *(FT rate)*",
        "Free throws are the most efficient shot in basketball. "
        "Teams that draw fouls at a high rate stress opposing defenses and stay out of foul trouble themselves.",
    ),
    "to_rate_diff": (
        "Ball security edge  *(turnover rate)*",
        "Turnovers are essentially gifted possessions to the opponent. "
        "In tight tournament games, one bad stretch of careless passing can end a season.",
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
def load_rankings(season: int) -> pd.DataFrame:
    """Pull ratings + team info for a given season from the database."""
    with SessionLocal() as session:
        rows = (
            session.query(
                CarmPomRating.rank,
                Team.name,
                Team.conference,
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
        columns=["Rank", "Team", "Conf", "AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS", "W", "L"],
    )
    df["Record"] = df["W"].astype(str) + "-" + df["L"].astype(str)
    df["Conf"] = df["Conf"].str.removesuffix(" Conference")
    return df


# ---------------------------------------------------------------------------
# KenPom comparison loader (cached)
# ---------------------------------------------------------------------------

@st.cache_data
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
# Tabs
# ---------------------------------------------------------------------------

rankings_tab, kenpom_tab, about_tab = st.tabs(["📊 Team Rankings", "🆚 vs KenPom", "ℹ️ About"])

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
        top_n = st.selectbox("Show", [25, 50, 100, 364], index=1)

    # Apply filters
    filtered = df.copy()
    if selected_conf != "All conferences":
        filtered = filtered[filtered["Conf"] == selected_conf]
    if search:
        filtered = filtered[filtered["Team"].str.contains(search, case=False, na=False)]
    filtered = filtered.head(top_n)

    display_cols = ["Rank", "Team", "Conf", "Record", "AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS"]

    st.caption(
        f"{len(filtered)} teams shown  |  "
        "**AdjEM** = pts/100 poss margin  |  "
        "**AdjO** = off. efficiency (higher = better)  |  "
        "**AdjD** = def. efficiency (lower = better)  |  "
        "**AdjT** = tempo  |  "
        "**Luck** = actual W% − expected W%"
    )

    styled = (
        filtered[display_cols]
        .style
        .background_gradient(subset=["AdjEM"], cmap="RdYlGn")
        .background_gradient(subset=["AdjO"], cmap="Greens")
        .background_gradient(subset=["AdjD"], cmap="Reds_r")
        .background_gradient(subset=["Luck"], cmap="coolwarm")
        .background_gradient(subset=["SOS"], cmap="Purples")
        .format({
            "AdjEM": "{:+.2f}",
            "AdjO":  "{:.2f}",
            "AdjD":  "{:.2f}",
            "AdjT":  "{:.1f}",
            "Luck":  "{:+.3f}",
            "SOS":   "{:+.2f}",
        })
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=700)

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
