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
        "a 4 vs a 5 is basically a coin flip. Seeds matter, but they're just people reading the same stats.",
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
        columns=["team_id", "Rank", "Team", "Conf", "AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS", "W", "L"],
    )
    # Build ESPN team stats page URL — works without the slug, just the numeric ID
    df["ESPN"] = df["team_id"].map(
        lambda tid: _ESPN_ID_MAP.get(int(tid), None)
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

    # National rank for each stat (rank 1 = best); lower is better for OppPPG and TOPG
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
