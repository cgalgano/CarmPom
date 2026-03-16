"""
pipeline/fetch_injuries.py
--------------------------
Scrapes Covers.com NCAAB injury page for all tournament teams and caches
results to data/injuries_cache.json.

Run this periodically during the tournament (once or twice a day):
    uv run python pipeline/fetch_injuries.py

The app reads from the cache file — no live HTTP calls at render time.
"""

import json
import re
import sys
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "injuries_cache.json"
BRACKET_CSV = ROOT / "data" / "bracket_2026.csv"

COVERS_URL = "https://www.covers.com/sport/basketball/ncaab/injuries"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Covers status labels → our canonical labels
_STATUS_MAP = {
    "out": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "probable": "Probable",
    "monitor": "Monitor",
}


def fetch_covers_page() -> str:
    """Fetch the Covers NCAAB injuries page HTML."""
    resp = httpx.get(COVERS_URL, headers=_HEADERS, timeout=25.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _extract_team_name(block) -> str:
    """Pull the school name (not nickname) from a blockContainer div.

    The anchor text looks like:
        NavigableString("Air Force") → <br/> → <span>Falcons</span>
    We collect NavigableString children until we hit a <br/> or <span> tag.
    NavigableString objects have name=None; Tag objects have name = "br"/"span"/etc.
    """
    name_div = block.find("div", class_="covers-CoversMatchups-teamName")
    if not name_div:
        return ""
    a = name_div.find("a")
    if not a:
        return ""
    parts = []
    for child in a.children:
        # Tags (br, span) signal end of the school-name portion
        if child.name is not None:
            break
        # NavigableString — child.name is None
        t = str(child).strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def parse_covers_injuries(html: str) -> dict[str, list[dict]]:
    """Parse the Covers.com NCAAB injury page.

    Returns {team_display_name: [{player, position, status, reason, published, headline, description}]}
    Each team_display_name is the school portion only (e.g. 'Duke', 'Air Force').
    """
    soup = BeautifulSoup(html, "html.parser")
    results: dict[str, list[dict]] = {}

    for block in soup.find_all("div", class_="covers-CoversSeasonInjuries-blockContainer"):
        team_name = _extract_team_name(block)
        if not team_name:
            continue

        notes: list[dict] = []
        tbody = block.find("tbody")
        if not tbody:
            results[team_name] = notes
            continue

        # Rows alternate between main injury rows and optional collapse/description rows.
        # Main row: 4 <td> cells — player link, position, status bold text + date, toggle btn
        # Description row: class="collapse", single <td colspan="4"> with injuryCopy text
        rows = tbody.find_all("tr")
        last_note: dict | None = None
        for row in rows:
            # Description rows (the collapsed detail text)
            if "collapse" in (row.get("class") or []):
                copy_div = row.find("div", class_="covers-CoversMatchups-injuryCopy")
                if copy_div and last_note is not None:
                    last_note["description"] = copy_div.get_text(strip=True)
                continue

            tds = row.find_all("td")
            if len(tds) < 3:
                continue

            # Cell 0: player anchor — abbreviated first + last name
            player_td = tds[0]
            player_a = player_td.find("a", class_="player-link")
            if player_a:
                player = " ".join(player_a.get_text().split())
            else:
                player = player_td.get_text(strip=True)

            if not player or "no injuries" in player.lower():
                last_note = None
                continue

            position = tds[1].get_text(strip=True)

            # Cell 2: <b>Status - Reason</b><br/>(Date)
            status_cell = tds[2]
            bold = status_cell.find("b")
            bold_text = bold.get_text(strip=True) if bold else status_cell.get_text(strip=True)

            # Date is in parentheses after the <br/>
            date_str = ""
            full_text = status_cell.get_text(separator="\n", strip=True)
            date_m = re.search(r"\(([^)]+)\)", full_text)
            if date_m:
                date_str = date_m.group(1).strip()

            # bold_text: "Questionable - Undisclosed" or "Out - Knee"
            m = re.match(r"(?P<status>[^-]+?)\s*-\s*(?P<reason>.+)", bold_text)
            if m:
                status_word = m.group("status").strip().lower()
                reason = m.group("reason").strip()
            else:
                status_word = bold_text.lower()
                reason = ""

            status = _STATUS_MAP.get(status_word, "Monitor")
            headline = f"{player} ({position}) — {status}: {reason}"

            note: dict = {
                "player": player,
                "position": position,
                "status": status,
                "reason": reason,
                "published": date_str,
                "headline": headline,
                "description": "",
            }
            notes.append(note)
            last_note = note

        results[team_name] = notes

    return results


def _load_bracket_teams() -> list[str]:
    """Return list of team names from the bracket CSV."""
    import csv
    teams: list[str] = []
    try:
        with open(BRACKET_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("team", "").strip()
                if t and t.lower() not in ("tbd", ""):
                    teams.append(t)
    except FileNotFoundError:
        print(f"WARNING: {BRACKET_CSV} not found — matching against all teams.")
    return teams


def match_to_bracket(
    covers_lu: dict[str, list[dict]],
    bracket_teams: list[str],
) -> dict[str, list[dict]]:
    """Re-key covers data using exact bracket team names where possible.

    Covers uses short school names ('Duke'), bracket CSV uses the DB name
    ('Duke Blue Devils'). We try exact match first, then fuzzy.
    """
    out: dict[str, list[dict]] = {}
    covers_keys = list(covers_lu.keys())  # e.g. ['Duke', 'Gonzaga', ...]

    for bt in bracket_teams:
        # 1. Exact match (unlikely but possible if names align)
        if bt in covers_lu:
            out[bt] = covers_lu[bt]
            continue

        # 2. Check if the Covers short name is a prefix/substring of the bracket name
        # e.g. Covers "Duke" vs bracket "Duke Blue Devils"
        matched_key = None
        for ck in covers_keys:
            if bt.lower().startswith(ck.lower()) or ck.lower() in bt.lower():
                matched_key = ck
                break

        if matched_key:
            out[bt] = covers_lu[matched_key]
            continue

        # 3. Fuzzy match on first word of bracket name vs covers keys
        bt_first = bt.split()[0]
        hits = get_close_matches(bt_first, covers_keys, n=1, cutoff=0.82)
        if hits:
            out[bt] = covers_lu[hits[0]]
            continue

        # No match — empty list
        out[bt] = []

    return out


def run() -> dict:
    """Scrape Covers.com and save injury cache for all bracket teams."""
    print("Fetching Covers.com NCAAB injuries page…")
    html = fetch_covers_page()
    print(f"  Page fetched ({len(html):,} bytes)")

    covers_all = parse_covers_injuries(html)
    teams_with_data = sum(1 for v in covers_all.values() if v)
    print(f"  Parsed {len(covers_all)} teams, {teams_with_data} with injury notes")

    bracket_teams = _load_bracket_teams()
    if bracket_teams:
        matched = match_to_bracket(covers_all, bracket_teams)
    else:
        # Fallback: use all Covers teams
        matched = covers_all

    found = sum(1 for v in matched.values() if v)
    print(f"\nMatched {found}/{len(matched)} bracket teams with injury data:")
    for t, notes in sorted(matched.items(), key=lambda x: -len(x[1])):
        if notes:
            top = notes[0]
            print(f"  {t}: {len(notes)} player(s) — {top['status']}: {top['headline'][:55]}")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "covers.com",
        "teams": matched,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {CACHE_PATH}")
    return matched


def load_injuries_cache() -> dict[str, list[dict]]:
    """Load the most recently fetched injury notes from disk. Returns {} if unavailable."""
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("teams", {})
    except Exception:
        return {}


def load_injuries_meta() -> dict:
    """Return full cache metadata dict (includes fetched_at timestamp)."""
    if not CACHE_PATH.exists():
        return {}
    try:
        d = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {"fetched_at": d.get("fetched_at", ""), "source": d.get("source", "")}
    except Exception:
        return {}


if __name__ == "__main__":
    run()
