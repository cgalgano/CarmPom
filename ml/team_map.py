"""
ml/team_map.py
--------------
Maps team identities between our ESPN DB and Kaggle's TeamID system.

ESPN uses full names with mascots: "Duke Blue Devils", "Abilene Christian Wildcats"
Kaggle uses abbreviated school names:  "Duke", "Abilene Chr"

Matching strategy:
  1. Normalize ESPN names by stripping mascot words (last 1-3 tokens that are
     in a known mascot set), leaving just the school identifier.
  2. Normalize Kaggle names by expanding common abbreviations ("St"->"state",
     "Univ"->"university", etc.) and removing punctuation.
  3. Exact match on normalized forms.
  4. For remaining, fuzzy match (difflib, cutoff=0.82).
  5. Unmatched stored with db_team_id=-1 for manual review.

The mapping is written to data/team_id_map.csv after the first run.
Re-run manually if teams are added/changed.

Usage:
    uv run python ml/team_map.py
    from ml.team_map import load_mapping
    kaggle_to_db, db_to_kaggle = load_mapping()
"""

import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

from db.database import SessionLocal
from db.models import Team
from ml.kaggle_loader import load_teams

MAP_FILE = Path(__file__).parent.parent / "data" / "team_id_map.csv"

# Mascot words that appear at the END of ESPN team names.
# We strip these to isolate the school identifier portion.
_MASCOT_WORDS = {
    "aggies", "anteaters", "aztecs", "badgers", "bears", "beavers", "bengals",
    "billikens", "bison", "blazers", "blue", "bobcats", "boilermakers", "braves",
    "broncos", "broncs", "bruins", "buccaneers", "buffaloes", "bulldogs", "bulls",
    "camels", "cardinals", "cavaliers", "chanticleers", "colonels", "colonials",
    "cornhuskers", "cougars", "cowboys", "crimson", "crusaders", "cyclones",
    "deacons", "demon", "devils", "dons", "dragons", "dukes", "eagles",
    "falcons", "fighting", "flashes", "flames", "flyers", "friars", "frogs",
    "gators", "gaels", "gophers", "governors", "grizzlies", "gusties",
    "hawkeyes", "hawks", "heels", "hoosiers", "hornets", "huskies", "hurricanes",
    "illini", "jayhawks", "keydets", "knights", "leathernecks", "lions",
    "lobos", "longhorns", "lumberjacks", "mavericks", "midshipmen", "minutemen",
    "monarchs", "mountaineers", "mustangs", "nanooks", "nittany", "noles",
    "oaks", "orange", "owls", "pack", "panthers", "patriots", "penguins",
    "phoenix", "pilots", "pioneers", "pirates", "racers", "rams", "razorbacks",
    "rebels", "retrievers", "runnin", "sailors", "saints", "scarlet",
    "seahawks", "seminoles", "shockers", "sooners", "spartans", "spiders",
    "tigers", "tide", "toreros", "trojans", "utes", "vols", "volunteers",
    "vulcans", "wahoos", "wave", "wildcats", "wolves", "wolverines", "zags", "zips",
    # additional mascots not in original list
    "bearcats", "bluejays", "commodores", "ducks", "musketeers", "paladins",
    "red", "flash", "quakers", "leopards", "engineers", "big", "flying",
    "dutchmen", "retrievers", "catamounts", "peacocks", "gaels", "toreros",
    "patriot", "highlanders", "ospreys", "sycamores", "bears", "ramblers",
    "skyhawks", "explorers", "penguins", "jaguars", "buccaneers", "chiefs",
    "colonels", "minutewomen", "rattlers", "ravens", "tigers", "warhawks",
    "antelopes", "privateers", "sharks", "stags", "lakers", "seahawks",
    # multi-word mascot fragments that follow the school name
    "black", "golden", "green", "terrapins", "terps", "hoyas",
    "tar", "blue", "devilettes",
}

# Kaggle abbreviation expansions — applied BEFORE normalization.
_KAGGLE_EXPANSIONS: list[tuple[str, str]] = [
    # word-boundary replacements (order matters — longer first)
    (r"\bSt\b",        "State"),
    (r"\bUniv\b",      "University"),
    (r"\bChr\b",       "Christian"),
    (r"\bCar\b",       "Carolina"),
    (r"\bSo\b",        "Southern"),
    (r"\bNo\b",        "Northern"),
    (r"\bAf\b",        "Air Force"),
    (r"\bTech\b",      "Tech"),          # keep as-is; ESPN also uses "Tech"
    (r"\bIntl\b",      "International"),
    (r"\bInt'l\b",     "International"),
    (r"\bInst\b",      "Institute"),
    (r"\bMt\b",        "Mount"),
    (r"\bFla\b",       "Florida"),
    (r"\bIll\b",       "Illinois"),
    (r"\bCol\b",       "Colorado"),
    (r"\bIndiana\b",   "Indiana"),       # no-op but harmless
    (r"\bMiss\b",      "Mississippi"),
    (r"\bNeb\b",       "Nebraska"),
    (r"\bTenn\b",      "Tennessee"),
    (r"\bVa\b",        "Virginia"),
    (r"\bW\b",         "West"),
    (r"\bE\b",         "East"),
    (r"\bN\b",         "North"),
    (r"\bS\b",         "South"),
    (r"SUNY Albany",   "Albany"),        # ESPN has "Albany Great Danes"
    (r"SUNY Binghamton", "Binghamton"),
    (r"LIU Brooklyn",  "LIU"),
    (r"Birmingham So", "UAB"),
    (r"Loyola-Chicago","Loyola Chicago"),
    (r"Prairie View",  "Prairie View AM"),
    (r"Texas A&M CC",  "Texas AM Corpus Christi"),
    (r"Tex-Arlington", "UT Arlington"),
    (r"Tex-Pan Amer",  "UT Rio Grande Valley"),
    (r"IPFW",          "Purdue Fort Wayne"),
    (r"IUPUI",         "Indiana University Indianapolis"),
]

# Manual overrides: kaggle_name -> exact ESPN DB team name fragment
_MANUAL: dict[str, str] = {
    "Loyola MD":          "Loyola Maryland",
    "Loyola-Chicago":     "Loyola Chicago",
    "UMKC":               "Kansas City",
    "UMES":               "Maryland Eastern Shore",
    "UMBC":               "UMBC",
    "UNC Asheville":      "UNC Asheville",
    "UNC Greensboro":     "UNC Greensboro",
    "UNC Wilmington":     "UNC Wilmington",
    "UNCW":               "UNC Wilmington",
    "SIU-Edwardsville":   "SIU Edwardsville",
    "S Carolina St":      "South Carolina State",
    "Sou Illinois":       "Southern Illinois",
    "CS Bakersfield":     "Cal State Bakersfield",
    "CS Fullerton":       "Cal State Fullerton",
    "CS Northridge":      "Cal State Northridge",
    "CSUN":               "Cal State Northridge",
    "Long Beach St":      "Long Beach State",
    "SF Austin":          "Stephen F Austin",
    "Tennessee St":       "Tennessee State",
    "Tennessee Tech":     "Tennessee Tech",
    "Indiana St":         "Indiana State",
    "Fort Wayne":         "Purdue Fort Wayne",
    "The Citadel":        "The Citadel",
    "Citadel":            "The Citadel",
    "Maryland-Eastern Shore": "Maryland Eastern Shore",
    "NJIT":               "NJIT",
    "VCU":                "VCU",
    "UCF":                "UCF",
    "USC":                "USC",
    "LSU":                "LSU",
    "SMU":                "SMU",
    "TCU":                "TCU",
    "BYU":                "BYU",
    "UIW":                "Incarnate Word",
    "LIU":                "LIU",
    "Saint Joseph's":     "Saint Joseph's",
    "St Joseph's PA":     "Saint Joseph's",
    "Saint Peter's":      "Saint Peter's",
    "St Peter's":         "Saint Peter's",
    "St John's":          "St John's",
    "Saint Mary's":       "Saint Mary's",
    "St Mary's CA":       "Saint Mary's",
    "Monmouth NJ":        "Monmouth",
    "Col of Charleston":  "College of Charleston",
    "Morehead St":        "Morehead State",
    "Murray St":          "Murray State",
    "McNeese St":         "McNeese State",
    "Kennesaw St":        "Kennesaw State",
    "Jacksonville St":    "Jacksonville State",
    "Southeast Mo St":    "Southeast Missouri State",
    "SE Missouri St":     "Southeast Missouri State",
    "Southern Ill":       "Southern Illinois",
    "Ill-Chicago":        "Illinois Chicago",
    "Wis-Milwaukee":      "Milwaukee",
    "Wis-Green Bay":      "Green Bay",
    "Ark-Pine Bluff":     "Arkansas Pine Bluff",
    "Ark Little Rock":    "Little Rock",
    "Western Ky":         "Western Kentucky",
    "Western Car":        "Western Carolina",
    "Middle Tenn":        "Middle Tennessee",
    "Fla Gulf Coast":     "Florida Gulf Coast",
    "Fla Atlantic":       "Florida Atlantic",
    "Fla International":  "FIU",
    "FIU":                "FIU",
    "FAU":                "Florida Atlantic",
    "FGCU":               "Florida Gulf Coast",
    "New Orleans":        "New Orleans",
    "New Mexico St":      "New Mexico State",
    "Mississippi Val":    "Mississippi Valley State",
    "Miss Valley St":     "Mississippi Valley State",
    "Southern Univ":      "Southern University",
    "N Colorado":         "Northern Colorado",
    "N Iowa":             "Northern Iowa",
    "N Kentucky":         "Northern Kentucky",
    "N Dakota St":        "North Dakota State",
    "S Dakota St":        "South Dakota State",
    "N Dakota":           "North Dakota",
    "S Dakota":           "South Dakota",
    "E Washington":       "Eastern Washington",
    "W Kentucky":         "Western Kentucky",
    "W Virginia":         "West Virginia",
    "Bethune-Cookman":    "Bethune Cookman",
    "UMass Lowell":       "UMass Lowell",
    "UT Martin":          "UT Martin",
    "Miami OH":           "Miami Ohio",
    "Ohio St":            "Ohio State",
    "Michigan St":        "Michigan State",
    "Penn St":            "Penn State",
    "Kansas St":          "Kansas State",
    "Iowa St":            "Iowa State",
    "Florida St":         "Florida State",
    "Arizona St":         "Arizona State",
    "Colorado St":        "Colorado State",
    "Washington St":      "Washington State",
    "Oregon St":          "Oregon State",
    "Oklahoma St":        "Oklahoma State",
    "Mississippi St":     "Mississippi State",
    "Louisiana St":       "LSU",
    "NC State":           "NC State",
    "GA Tech":            "Georgia Tech",
    "Ga Tech":            "Georgia Tech",
    "Cal Poly":           "Cal Poly",
    "Cal Poly SLO":       "Cal Poly",
    "Coastal Car":        "Coastal Carolina",
    "Appalachian St":     "Appalachian State",
    "Boston Univ":        "Boston University",
    "American Univ":      "American University",
    # Acronyms and short forms
    "WKU":               "Western Kentucky",
    "MTSU":              "Middle Tennessee",
    "ETSU":              "East Tennessee State",
    "SIUE":              "SIU Edwardsville",
    "PFW":               "Purdue Fort Wayne",
    "ULM":               "Louisiana Monroe",
    "UTEP":              "UTEP",
    "UTRGV":             "UT Rio Grande Valley",
    "UTSA":              "UTSA",
    "UT San Antonio":    "UTSA",
    "NC A&T":            "North Carolina AT",
    "LIU Brooklyn":      "LIU",
    "SUNY Albany":       "Albany",
    # Teams whose normalized form didn't auto-resolve
    "Cornell":           "Cornell",
    "Colgate":           "Colgate",
    "Canisius":          "Canisius",
    "Cincinnati":        "Cincinnati",
    "Creighton":         "Creighton",
    "DePaul":            "DePaul",
    "Detroit":           "Detroit Mercy",
    "CS Sacramento":     "Sacramento State",
    "Evansville":        "Evansville",
    "Furman":            "Furman",
    "Hartford":          "Hartford",
    "Idaho":             "Idaho",
    "Lafayette":         "Lafayette",
    "Lehigh":            "Lehigh",
    "Lipscomb":          "Lipscomb",
    "Longwood":          "Longwood",
    "Manhattan":         "Manhattan",
    "Marist":            "Marist",
    "Marshall":          "Marshall",
    "Niagara":           "Niagara",
    "Okla City":         "Oklahoma City",
    "Oregon":            "Oregon",
    "Penn":              "Penn",
    "Savannah St":       "Savannah State",
    "Seattle":           "Seattle University",
    "Southern Utah":     "Southern Utah",
    "Stetson":           "Stetson",
    "Stony Brook":       "Stony Brook",
    "Toledo":            "Toledo",
    "Vanderbilt":        "Vanderbilt",
    "Wofford":           "Wofford",
    "Xavier":            "Xavier",
    "Merrimack":         "Merrimack",
    "St Thomas MN":      "St Thomas",
    "Le Moyne":          "Le Moyne",
    "New Haven":         "New Haven",
    "Queens NC":         "Queens Charlotte",
    "La Salle":          "La Salle",
    "MS Valley St":      "Mississippi Valley State",
    "MD E Shore":        "Maryland Eastern Shore",
    "TAM C. Christi":    "Texas AM Corpus Christi",
    "TN Martin":         "UT Martin",
    "Sam Houston St":    "Sam Houston State",
    "UC Santa Barbara":  "UC Santa Barbara",
    "UC Riverside":      "UC Riverside",
    "SC Upstate":        "USC Upstate",
    "Ark Pine Bluff":    "Arkansas Pine Bluff",
    "IL Chicago":        "Illinois Chicago",
    "St Francis NY":     "Saint Francis",
    "St Francis PA":     "Saint Francis PA",
    "W Salem St":        "Winston Salem State",
    "Birmingham So":     "UAB",
    "Augusta":           "Augusta University",
    "Brooklyn":          "LIU",
    "Florida A&M":       "Florida AM",
    "Morris Brown":      "Morris Brown",
    "Hardin-Simmons":    "Hardin Simmons",
    "IUPUI":             "IU Indianapolis",
}


def _normalize_espn(name: str) -> str:
    """
    Strip the mascot from an ESPN team name and return a lowercase school identifier.

    e.g. "Duke Blue Devils" -> "duke"
         "Abilene Christian Wildcats" -> "abilene christian"
         "Air Force Falcons" -> "air force"
    """
    name = re.sub(r"[^a-zA-Z0-9 ]", "", name).strip()
    tokens = name.split()
    # Walk from the right, dropping tokens that are pure mascot words
    while tokens and tokens[-1].lower() in _MASCOT_WORDS:
        tokens.pop()
    return " ".join(tokens).lower().strip()


def _normalize_kaggle(name: str) -> str:
    """
    Expand abbreviations in a Kaggle team name and return a lowercase identifier.

    e.g. "Ball St"       -> "ball state"
         "Abilene Chr"   -> "abilene christian"
         "Appalachian St"-> "appalachian state"
    """
    for pattern, replacement in _KAGGLE_EXPANSIONS:
        name = re.sub(pattern, replacement, name)
    name = re.sub(r"[^a-zA-Z0-9 ]", " ", name)
    return " ".join(name.split()).lower().strip()


def build_mapping() -> pd.DataFrame:
    """
    Build a DataFrame mapping Kaggle TeamID ↔ our DB team_id.

    Strategy:
      1. Apply manual overrides for known problem cases.
      2. Exact match on normalized names (ESPN mascot-stripped, Kaggle abbrev-expanded).
      3. Fuzzy match (cutoff=0.82) on remaining.
      4. Unmatched stored with db_team_id=-1 for review.
    """
    kaggle_teams = load_teams()[["TeamID", "TeamName"]].copy()

    with SessionLocal() as session:
        espn_rows = session.query(Team.id, Team.name).all()
    espn_df = pd.DataFrame(espn_rows, columns=["db_team_id", "name"])
    espn_df["norm"] = espn_df["name"].map(_normalize_espn)

    # Build lookup: normalized ESPN school name -> db_team_id
    # Prefer exact-name matches over fuzzy to avoid duplicate norm keys
    espn_by_norm: dict[str, int] = {}
    for _, row in espn_df.iterrows():
        espn_by_norm[row["norm"]] = row["db_team_id"]

    # Also build a lowercase full-name lookup for manual override resolution
    espn_by_lower: dict[str, int] = {
        row["name"].lower(): row["db_team_id"] for _, row in espn_df.iterrows()
    }
    espn_by_fragment: dict[str, int] = {}
    for _, row in espn_df.iterrows():
        # index by first word(s) of the school name for partial lookups
        espn_by_fragment[row["norm"]] = row["db_team_id"]

    rows = []
    unmatched = []

    for _, row in kaggle_teams.iterrows():
        kname: str = row["TeamName"]
        kid: int = row["TeamID"]

        # --- Step 1: manual override ---
        if kname in _MANUAL:
            target = _MANUAL[kname].lower()
            # match against lowercase full ESPN name OR normalized ESPN name
            db_id = espn_by_lower.get(target) or espn_by_lower.get(target + " ")
            if db_id is None:
                # Try normalized match of the manual target
                tnorm = _normalize_espn(_MANUAL[kname])
                db_id = espn_by_norm.get(tnorm, -1)
            rows.append({
                "kaggle_id":   kid, "kaggle_name": kname,
                "db_team_id":  db_id, "match_type": "manual",
            })
            continue

        knorm = _normalize_kaggle(kname)

        # --- Step 2: exact normalized match ---
        if knorm in espn_by_norm:
            rows.append({
                "kaggle_id":   kid, "kaggle_name": kname,
                "db_team_id":  espn_by_norm[knorm], "match_type": "exact",
            })
            continue

        # --- Step 3: fuzzy match ---
        matches = get_close_matches(knorm, espn_by_norm.keys(), n=1, cutoff=0.82)
        if matches:
            rows.append({
                "kaggle_id":   kid, "kaggle_name": kname,
                "db_team_id":  espn_by_norm[matches[0]], "match_type": "fuzzy",
            })
        else:
            # try lower cutoff for single-word names (e.g. "Akron" -> "akron")
            matches2 = get_close_matches(knorm, espn_by_norm.keys(), n=1, cutoff=0.70)
            if matches2 and len(knorm.split()) <= 2:
                rows.append({
                    "kaggle_id":   kid, "kaggle_name": kname,
                    "db_team_id":  espn_by_norm[matches2[0]], "match_type": "fuzzy_loose",
                })
            else:
                unmatched.append(kname)
                rows.append({
                    "kaggle_id":   kid, "kaggle_name": kname,
                    "db_team_id":  -1, "match_type": "unmatched",
                })

    df = pd.DataFrame(rows)
    matched = (df["db_team_id"] > 0).sum()
    print(f"Matched {matched} / {len(df)} Kaggle teams.")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}): {unmatched}")
    return df


def save_mapping(df: pd.DataFrame) -> None:
    """Save mapping to CSV for inspection and manual correction."""
    MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MAP_FILE, index=False)
    print(f"Saved to {MAP_FILE}")


def load_mapping() -> tuple[dict[int, int], dict[int, int]]:
    """
    Load the cached mapping CSV.

    Returns:
        kaggle_to_db: {kaggle_team_id: db_team_id}
        db_to_kaggle: {db_team_id: kaggle_team_id}
    """
    if not MAP_FILE.exists():
        print("team_id_map.csv not found. Run: uv run python ml/team_map.py")
        return {}, {}

    df = pd.read_csv(MAP_FILE)
    df = df[df["db_team_id"] > 0]  # drop unmatched
    kaggle_to_db = dict(zip(df["kaggle_id"], df["db_team_id"]))
    db_to_kaggle = dict(zip(df["db_team_id"], df["kaggle_id"]))
    return kaggle_to_db, db_to_kaggle


if __name__ == "__main__":
    mapping = build_mapping()
    save_mapping(mapping)
    print(mapping.groupby("match_type").size())
