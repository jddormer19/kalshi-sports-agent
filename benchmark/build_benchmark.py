"""Build the frozen agent benchmark from data/games/ (see benchmark/SPEC.md).

Output: data/benchmark.parquet, one row per game:
    event_ticker, sport, start_time, t, home_team, away_team, home_ticker, label (1 = home won),
    market_p (mid at t), spread_at_t, split ("dev" / "holdout")

You write the five TODO functions; the loading and saving plumbing is done.
Run:  .venv/bin/python -m benchmark.build_benchmark
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "data" / "games"
DECISION_LEAD = pd.Timedelta(minutes=60)   # rule 2
MAX_SPREAD = 0.10                          # rule 3
HOLDOUT_FRAC = 0.30                        # rule 4
NFL_REGULAR_SEASON_MONTH = 9               # NFL games before September are preseason (spec rule 2)


def load_markets() -> pd.DataFrame:
    df = pd.read_parquet(GAMES / "markets.parquet")
    df["sport"] = df.series_ticker.str.replace("KX", "").str.replace("GAME", "")
    return df


def load_candles(ticker: str) -> pd.DataFrame:
    """Hourly candles: t (unix seconds, END of the hour), price, bid, ask, volume, oi."""
    f = GAMES / "candles" / f"{ticker}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["t", "price", "bid", "ask", "volume", "oi"])


# ---------------------------------------------------------------- TODO 1
def game_start(event_ticker: str, markets: pd.DataFrame) -> pd.Timestamp | None:
    """Scheduled start time (UTC) of the game.

    Rule 4 of SPEC.md: the ticker time when present (KXMLBGAME-26SEP161340NYYMIN = Sep 16, 13:40 *US Eastern*,
    so convert to UTC), otherwise `occurrence_datetime` minus 3 hours (it is a scheduled END time; verified
    on 3,884 MLB markets). NBA/NFL tickers never have a time. `occurrence_datetime` is a column in
    markets.parquet (NaT when Kalshi doesn't provide it).
    Return None if neither exists (the game gets excluded, and you count how many).
    """
    code = event_ticker.split("-")[1]                       # "26SEP161340NYYMIN"
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})(\d{4})", code)  # yy, mon, dd, hhmm
    if m:
        yy, mon, dd, hhmm = m.groups()
        local = pd.to_datetime(f"20{yy}-{mon}-{dd} {hhmm[:2]}:{hhmm[2:]}", format="%Y-%b-%d %H:%M")
        return local.tz_localize("America/New_York").tz_convert("UTC")
    occ = markets.loc[markets.event_ticker == event_ticker, "occurrence_datetime"].dropna()
    if len(occ):
        return occ.iloc[0] - pd.Timedelta(hours=3)
    return None


# ---------------------------------------------------------------- TODO 2
def home_side(event_ticker: str, game_markets: pd.DataFrame) -> tuple[str, str, str] | None:
    """Return (home_team, away_team, home_market_ticker).

    The event code ends with both team codes, e.g. ...NYYMIN or ...SDSF (codes are 2 or 3 letters).
    Hint: each market's ticker ends with ONE team code (…-NYY, …-MIN), so use the siblings to split.
    Check the order convention (away+home vs. home+away) on 3 games you can verify on a schedule.
    Verified 2026-11-13: away+home (KXNFLGAME-25SEP04DALPHI: PHI settled YES, Eagles won at home).
    """
    code = re.sub(r"G?\d+$", "", event_ticker.split("-")[1])   # drop a doubleheader suffix: ...ATHMIL2 or ...COLKCG1
    teams = {t.split("-")[-1]: t for t in game_markets.ticker}  # {"NYY": "...-NYY", "MIN": "...-MIN"}
    if len(teams) != 2:
        return None
    a, b = teams
    for away, home in ((a, b), (b, a)):
        if code.endswith(away + home):                         # event code = away + home
            return home, away, teams[home]
    return None


# ---------------------------------------------------------------- TODO 3
def market_price_at(candles: pd.DataFrame, t: pd.Timestamp) -> tuple[float, float] | None:
    """(mid, spread) from the last candle whose period ENDED at or before t. Never use a later candle.
    Return None if there is no usable candle (missing bid/ask) or the spread exceeds MAX_SPREAD."""
    t_unix = t.timestamp()
    before = candles[candles.t <= t_unix]            # candle "t" = END of its hour, so <= t means fully before t
    if before.empty:
        return None
    last = before.sort_values("t").iloc[-1]
    bid, ask = last.bid, last.ask
    if pd.isna(bid) or pd.isna(ask) or ask < bid:
        return None
    spread = round(ask - bid, 4)                     # round: 0.41 - 0.31 is 0.09999... in floating point
    if spread > MAX_SPREAD:
        return None
    return (bid + ask) / 2, spread


# ---------------------------------------------------------------- TODO 4
def assign_split(bench: pd.DataFrame) -> pd.Series:
    """'dev' for the earliest 70% of games by start_time, 'holdout' for the latest 30%.
    Think: split per sport, or one global date cutoff? Write the choice and the reason in SPEC.md.
    Cut at a DATE boundary (US Eastern) so no single day's games are split across dev and holdout.
    """
    split = pd.Series("dev", index=bench.index)
    for sport, g in bench.groupby("sport"):
        days = g.start_time.dt.tz_convert("America/New_York").dt.date
        ordered = days.sort_values()
        cutoff = ordered.iloc[int(len(ordered) * (1 - HOLDOUT_FRAC))]   # the date at the 70% mark
        split.loc[g.index[days >= cutoff]] = "holdout"
    return split


# ---------------------------------------------------------------- TODO 5
def build() -> pd.DataFrame:
    """Loop over games: start time -> t -> home side -> label -> market price at t.
    Keep a counter of every exclusion reason (no start time, no candle before t, wide spread, void)
    and print it: reviewers will ask how many games you dropped and why."""
    markets = load_markets()
    dropped = Counter()
    rows = []
    for event, g in markets.groupby("event_ticker"):
        sport = g.sport.iloc[0]
        start = game_start(event, g)                                   # TODO 1
        if start is None:
            dropped["no start time"] += 1
            continue
        if sport == "NFL" and start.tz_convert("America/New_York").month < NFL_REGULAR_SEASON_MONTH:
            dropped["NFL preseason"] += 1
            continue
        side = home_side(event, g)                                     # TODO 2
        if side is None:
            dropped["teams not resolved"] += 1
            continue
        home, away, home_ticker = side
        results = g.set_index("ticker").result
        home_result = results[home_ticker]
        if sorted(results) != ["no", "yes"]:                           # exactly one side must settle YES
            dropped["void / tie / bad settlement"] += 1
            continue
        t = start - DECISION_LEAD
        candles = load_candles(home_ticker)
        if candles.empty or (candles.t <= t.timestamp()).sum() == 0:
            dropped["no candle before t"] += 1
            continue
        priced = market_price_at(candles, t)                           # TODO 3
        if priced is None:
            dropped["no usable quote at t (missing or spread > 10c)"] += 1
            continue
        mid, spread = priced
        rows.append(dict(event_ticker=event, sport=sport, start_time=start, t=t, home_team=home, away_team=away,
                         home_ticker=home_ticker, label=int(home_result == "yes"), market_p=float(mid),
                         spread_at_t=float(spread)))
    bench = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    bench["split"] = assign_split(bench)                               # TODO 4
    total = markets.event_ticker.nunique()
    print(f"games: {total} | kept: {len(bench)} | dropped: {sum(dropped.values())}")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:5d}  {reason}")
    return bench


def benchmark_path(lead_hours: float) -> Path:
    """The frozen 1h benchmark keeps its original name; other decision times get their own file."""
    return ROOT / "data" / ("benchmark.parquet" if lead_hours == 1 else f"benchmark_{lead_hours:g}h.parquet")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-hours", type=float, default=1.0, help="decision time t = start - this many hours")
    lead = ap.parse_args().lead_hours
    DECISION_LEAD = pd.Timedelta(hours=lead)
    bench = build()
    if lead != 1:
        # keep the SAME dev/holdout membership as the frozen 1h benchmark, so the holdout is never re-drawn
        base = pd.read_parquet(benchmark_path(1)).set_index("event_ticker").split
        same = bench.event_ticker.map(base)
        print(f"games also in the 1h benchmark: {same.notna().sum()} of {len(bench)} (others dropped)")
        bench = bench[same.notna()].assign(split=same[same.notna()].values).reset_index(drop=True)
    bench.to_parquet(benchmark_path(lead))
    print(f"t = start - {lead:g}h ->", benchmark_path(lead).name)
    print(bench.groupby(["sport", "split"]).size().unstack(fill_value=0))
