"""Full crawl of settled game-winner markets for the agent benchmark (MLB, NFL, NBA).

Unlike collect.py (a 60-market sample per series for the mispricing model), this keeps EVERY
settled game in the chosen series, with no volume filter or sampling: the benchmark rules
(which side, decision time t, price at t, split) are applied later in build_benchmark.py.

Outputs (separate from the September dataset, which stays untouched):
  data/games/events/<SERIES>.json   raw settled events with nested markets (all fields, for inspection)
  data/games/markets.parquet        one row per settled binary market
  data/games/candles/<ticker>.parquet  hourly candles over each market's life (cached, resumable)
Read-only: uses Kalshi's public market-data API and never places orders.
"""
import json
from pathlib import Path

import pandas as pd

from .api import Kalshi
from .collect import DATA, fetch_candles, market_rows, to_frame

SERIES = ["KXMLBGAME", "KXNFLGAME", "KXNBAGAME"]
OUT = DATA / "games"


def crawl(api: Kalshi, series: str) -> list:
    """Settled events with their markets. Markets settled before Kalshi's archive cutoff are no
    longer nested in /events, so they are filled in from /historical/markets."""
    f = OUT / "events" / f"{series}.json"
    if f.exists():
        ev = json.loads(f.read_text())
    else:
        ev = list(api.paginate("/events", "events", series_ticker=series, status="settled",
                               with_nested_markets="true", limit=200))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(ev))
    hf = OUT / "events" / f"{series}.historical_markets.json"
    if hf.exists():
        hist = json.loads(hf.read_text())
    else:
        hist = list(api.paginate("/historical/markets", "markets", series_ticker=series, limit=1000))
        hf.write_text(json.dumps(hist))
    by_event = {}
    for m in hist:
        by_event.setdefault(m["event_ticker"], []).append(m)
    filled = 0
    for e in ev:
        if not e.get("markets") and e["event_ticker"] in by_event:
            e["markets"] = by_event[e["event_ticker"]]
            filled += 1
    print(f"  {series}: {len(hist)} archived markets, filled {filled} events from the archive", flush=True)
    return ev


def run():
    api = Kalshi()
    events = []
    for s in SERIES:
        ev = crawl(api, s)
        print(f"{s}: {len(ev)} settled events", flush=True)
        events.extend(ev)
    df = to_frame(market_rows(events, series_meta={s: "custom" for s in SERIES}))
    occ = {m["ticker"]: m.get("occurrence_datetime") for e in events for m in (e.get("markets") or [])}
    df["occurrence_datetime"] = pd.to_datetime(df.ticker.map(occ), utc=True, format="ISO8601")  # scheduled END time (start + 3 h)
    df.to_parquet(OUT / "markets.parquet")
    print(f"{len(df)} settled binary markets from {df.event_ticker.nunique()} games:\n"
          f"{df.series_ticker.value_counts().to_string()}", flush=True)
    fetch_candles(api, df, OUT / "candles")
    print("done", flush=True)
