"""Build the frozen agent benchmark from data/games/ (see benchmark/SPEC.md).

Output: data/benchmark.parquet, one row per game:
    event_ticker, sport, start_time, t, home_team, away_team, home_ticker, label (1 = home won),
    market_p (mid at t), spread_at_t, split ("dev" / "holdout")

You write the five TODO functions; the loading and saving plumbing is done.
Run:  .venv/bin/python -m benchmark.build_benchmark
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "data" / "games"
DECISION_LEAD = pd.Timedelta(minutes=60)   # rule 2
MAX_SPREAD = 0.10                          # rule 3
HOLDOUT_FRAC = 0.30                        # rule 4


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
    raise NotImplementedError


# ---------------------------------------------------------------- TODO 2
def home_side(event_ticker: str, game_markets: pd.DataFrame) -> tuple[str, str, str]:
    """Return (home_team, away_team, home_market_ticker).

    The event code ends with both team codes, e.g. ...NYYMIN or ...SDSF (codes are 2 or 3 letters).
    Hint: each market's ticker ends with ONE team code (…-NYY, …-MIN), so use the siblings to split.
    Check the order convention (away+home vs. home+away) on 3 games you can verify on a schedule.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- TODO 3
def market_price_at(candles: pd.DataFrame, t: pd.Timestamp) -> tuple[float, float] | None:
    """(mid, spread) from the last candle whose period ENDED at or before t. Never use a later candle.
    Return None if there is no usable candle (missing bid/ask) or the spread exceeds MAX_SPREAD."""
    raise NotImplementedError


# ---------------------------------------------------------------- TODO 4
def assign_split(bench: pd.DataFrame) -> pd.Series:
    """'dev' for the earliest 70% of games by start_time, 'holdout' for the latest 30%.
    Think: split per sport, or one global date cutoff? Write the choice and the reason in SPEC.md."""
    raise NotImplementedError


# ---------------------------------------------------------------- TODO 5
def build() -> pd.DataFrame:
    """Loop over games: start time -> t -> home side -> label -> market price at t.
    Keep a counter of every exclusion reason (no start time, no candle before t, wide spread, void)
    and print it: reviewers will ask how many games you dropped and why."""
    raise NotImplementedError


if __name__ == "__main__":
    bench = build()
    bench.to_parquet(ROOT / "data" / "benchmark.parquet")
    print(bench.groupby(["sport", "split"]).size().unstack(fill_value=0))
