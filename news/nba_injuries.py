"""Point-in-time NBA injury reports from the league's official injury report PDFs.

Source (data-rights audit 2026-11-17, data-rights review): the NBA publishes a PDF every
15 minutes at ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM.pdf. The report time in the
filename (US Eastern) is the publication timestamp we trust; never the time we downloaded it.
Rules: private copy only (data/ is git-ignored), about 1 request per second, cache everything, never re-pull.

    .venv/bin/python -m news.nba_injuries          # ingest reports for every NBA benchmark game (t-1h and t-24h)
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "news" / "nba_injury"
PDF_DIR = OUT / "pdf"
INDEX = OUT / "index.json"            # report stamp -> HTTP status (200 / 403 / 404), so nothing is fetched twice
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
ET = "America/New_York"
STEP = pd.Timedelta(minutes=15)
MAX_LOOKBACK = pd.Timedelta(hours=12)  # search this far back from t for the latest published report
HEADERS = {"User-Agent": "Mozilla/5.0 (personal research; non-commercial)"}
_index: dict | None = None
_last_request = 0.0


def stamp(t_et: pd.Timestamp) -> str:
    """2026-04-17 17:30 ET -> '2026-04-17_05_30PM' (the filename format)."""
    return t_et.strftime("%Y-%m-%d_%I_%M%p")


def stamp_time_utc(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%Y-%m-%d_%I_%M%p").tz_localize(ET).tz_convert("UTC")


def index() -> dict:
    global _index
    if _index is None:
        _index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    return _index


def save_index() -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index(), indent=0, sort_keys=True))


def fetch(s: str) -> bool:
    """Download one report if not already known. Returns True if the report exists (cached or fetched)."""
    global _last_request
    if s in index():
        return index()[s] == 200
    if (PDF_DIR / f"{s}.pdf").exists():          # downloaded in an earlier, interrupted run
        index()[s] = 200
        return True
    r = None
    for attempt in range(4):                     # the server occasionally times out: back off and retry
        wait = 1.0 - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        try:
            r = requests.get(URL.format(stamp=s), headers=HEADERS, timeout=30)
            break
        except requests.RequestException:
            time.sleep(5 * 2 ** attempt)
    if r is None:
        raise RuntimeError(f"giving up on {s} after retries")
    index()[s] = r.status_code
    if len(index()) % 25 == 0:
        save_index()                             # keep progress if the run stops
    if r.status_code == 200 and r.content[:4] == b"%PDF":
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        (PDF_DIR / f"{s}.pdf").write_bytes(r.content)
        return True
    if r.status_code == 200:            # not actually a PDF: record as missing
        index()[s] = 0
    return False


def latest_before(t_utc: pd.Timestamp) -> str | None:
    """The latest report published at or before t (stepping back in 15-minute slots, at most MAX_LOOKBACK)."""
    slot = t_utc.tz_convert(ET).floor("15min")
    floor = slot - MAX_LOOKBACK
    while slot >= floor:
        if fetch(stamp(slot)):
            return stamp(slot)
        slot -= STEP
    return None


def latest_cached_before(t_utc: pd.Timestamp) -> str | None:
    """Like latest_before, but never touches the network: only reports already downloaded (used at eval time)."""
    slot = t_utc.tz_convert(ET).floor("15min")
    floor = slot - MAX_LOOKBACK
    while slot >= floor:
        st = stamp(slot)
        if index().get(st) == 200 and (PDF_DIR / f"{st}.pdf").exists():
            return st
        slot -= STEP
    return None


NBA_TEAM_CODES = {   # report team name -> Kalshi ticker code
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


# Column left edges in PDF points; identical on every page (checked 2026-11-17), so page 2+ without a header still parses.
COLUMN_X = {"date": 0, "time": 115, "matchup": 195, "team": 260, "player": 420, "status": 580, "reason": 660}


def _lines(pdf_path: Path) -> list[dict]:
    """Every text line of the report as {column: text}, in reading order across pages."""
    import pdfplumber
    edges = list(COLUMN_X.items())
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.0, y_tolerance=2)
            rows: dict[int, list] = {}
            for w in words:
                key = next((k for k in rows if abs(k - w["top"]) <= 3), round(w["top"]))
                rows.setdefault(key, []).append(w)
            for top in sorted(rows):
                cells = {name: [] for name, _ in edges}
                for w in sorted(rows[top], key=lambda w: w["x0"]):
                    col = [name for name, x in edges if w["x0"] >= x - 2][-1]
                    cells[col].append(w["text"])
                out.append({k: " ".join(v) for k, v in cells.items()})
    return out


def parse(s: str) -> pd.DataFrame:
    """Parse one report PDF into rows: report time, game date, matchup (AWAY@HOME), team, player, status, reason."""
    lines = _lines(PDF_DIR / f"{s}.pdf")
    rows, last = [], {"date": None, "time": None, "matchup": None, "team": None}
    for i, c in enumerate(lines):
        joined = " ".join(v for v in c.values() if v).strip()
        if "Player Name" in joined or joined.startswith(("Injury Report:", "Page ")):
            continue
        if re.match(r"\d{2}/\d{2}/\d{4}", c["date"]):
            last["date"] = c["date"]
        if "(ET)" in c["time"]:
            last["time"] = c["time"]
        if "@" in c["matchup"]:
            last["matchup"], last["team"] = c["matchup"].replace(" ", ""), None
        if c["team"]:
            last["team"] = c["team"]
        if "NOT YET SUBMITTED" in joined.upper():
            rows.append({**last, "player": "", "status": "NOT YET SUBMITTED", "reason": ""})
            continue
        if not c["player"] or not c["status"]:
            continue
        reason = c["reason"]
        if not reason:   # long reasons wrap onto the lines just above and below the player row
            near = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
            reason = " ".join(l["reason"] for l in near if l["reason"] and not l["player"]).strip()
        rows.append({**last, "player": c["player"], "status": c["status"], "reason": reason})
    df = pd.DataFrame(rows, columns=["date", "time", "matchup", "team", "player", "status", "reason"])
    df.insert(0, "report", s)
    df.insert(1, "published_at", stamp_time_utc(s))
    df["published_at_precision"] = "15min"
    return df


def ingest(lead_hours=(1, 24)) -> pd.DataFrame:
    """For every NBA game in each benchmark, find the latest report before t, download it, and parse it."""
    from benchmark.build_benchmark import benchmark_path
    manifest = []
    for lead in lead_hours:
        bench = pd.read_parquet(benchmark_path(lead))
        nba = bench[bench.sport == "NBA"]
        for g in nba.itertuples():
            s = latest_before(g.t)
            manifest.append({"event_ticker": g.event_ticker, "lead_hours": lead, "t": g.t, "report": s})
        save_index()
        print(f"lead {lead}h: {len(nba)} games, {sum(m['report'] is not None for m in manifest if m['lead_hours'] == lead)}"
              f" with a report before t", flush=True)
    m = pd.DataFrame(manifest)
    m.to_parquet(OUT / "manifest.parquet")
    parsed = pd.concat([parse(s) for s in sorted(m.report.dropna().unique())], ignore_index=True)
    parsed["fetched_note"] = "report time from filename; fetch time not used"
    parsed.to_parquet(OUT / "rows.parquet")
    print(f"parsed {m.report.nunique()} reports -> {len(parsed)} rows", flush=True)
    return parsed


if __name__ == "__main__":
    ingest()
