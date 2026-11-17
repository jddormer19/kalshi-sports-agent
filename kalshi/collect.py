"""Series-based crawl of recurring, short-dated markets.

Why this universe: a dataset of *settled* markets is survivorship-biased for long-dated
markets (a "by 2027?" market is only settled if it resolved early, usually YES). Recurring
short-dated series (daily weather, games, weekly mentions/charts, index ranges) close on a
schedule, so their inclusion does not depend on the outcome. Series whose YES markets close
systematically earlier than NO markets are still dropped as a data-driven check.

Outputs: data/markets.parquet and data/candles/<ticker>.parquet (hourly, cached/resumable).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .api import Kalshi

DATA = Path(__file__).resolve().parent.parent / "data"
FREQS = {"daily", "weekly"}
CUSTOM_PATTERN = re.compile(r"GAME|MATCH")  # sports game series are tagged 'custom'
MAX_LIFE_DAYS = 14
EVENTS_PER_SERIES = 200
MARKETS_PER_EVENT = 3      # most-traded markets of each event
MARKETS_PER_SERIES = 60
MAX_YES_EARLY = 0.15       # drop series where YES markets close >15% of lifetime earlier than NO


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def market_rows(events, status_ok=("yes", "no"), series_meta=None):
    rows = []
    for e in events:
        if e["series_ticker"].startswith("KXMVE"):
            continue
        mkts = e.get("markets") or []
        for m in mkts:
            if m.get("mve_collection_ticker") or m.get("market_type") != "binary":
                continue
            if status_ok and m.get("result") not in status_ok:
                continue
            rows.append(dict(
                ticker=m["ticker"], event_ticker=e["event_ticker"], series_ticker=e["series_ticker"],
                category=e.get("category") or "Unknown", mutually_exclusive=bool(e.get("mutually_exclusive")),
                n_siblings=len(mkts), result=m.get("result"),
                open_time=m.get("open_time"), close_time=m.get("close_time"),
                volume=_f(m.get("volume_fp")), open_interest=_f(m.get("open_interest_fp")),
                yes_bid=_f(m.get("yes_bid_dollars")), yes_ask=_f(m.get("yes_ask_dollars")),
                last_price=_f(m.get("last_price_dollars")), title=m.get("title") or e.get("title"),
                frequency=(series_meta or {}).get(e["series_ticker"], ""),
            ))
    return rows


def to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("open_time", "close_time"):
        df[c] = ts(df[c])
    df["life_days"] = (df.close_time - df.open_time).dt.total_seconds() / 86400
    return df


def universe(api: Kalshi) -> dict:
    series = api.get("/series")["series"]
    keep = {s["ticker"]: s.get("frequency") for s in series
            if not s["ticker"].startswith("KXMVE")
            and (s.get("frequency") in FREQS or (s.get("frequency") == "custom" and CUSTOM_PATTERN.search(s["ticker"])))}
    return keep


def crawl_series(api: Kalshi, keep: dict, cache: Path) -> list:
    cache.mkdir(parents=True, exist_ok=True)
    events = []
    for i, s in enumerate(sorted(keep), 1):
        f = cache / f"{s}.json"
        if f.exists():
            ev = json.loads(f.read_text())
        else:
            d = api.get("/events", series_ticker=s, status="settled", with_nested_markets="true",
                        limit=EVENTS_PER_SERIES)
            ev = (d or {}).get("events", [])
            f.write_text(json.dumps(ev))
        events.extend(ev)
        if i % 100 == 0:
            print(f"  series {i}/{len(keep)}, {len(events)} events", flush=True)
    return events


def outcome_dependent_series(df: pd.DataFrame) -> set:
    g = df.groupby("series_ticker")
    life = g.life_days.median().clip(lower=0.05)
    yes = df[df.result == "yes"].groupby("series_ticker").life_days.median()
    no = df[df.result == "no"].groupby("series_ticker").life_days.median()
    yes_early = ((no - yes) / life).dropna()
    return set(yes_early[yes_early > MAX_YES_EARLY].index)


def select(df: pd.DataFrame, min_volume: float, rng) -> pd.DataFrame:
    df = df[(df.volume >= min_volume) & (df.life_days <= MAX_LIFE_DAYS) & (df.life_days > 1 / 24)]
    df = df.sort_values("volume", ascending=False).groupby("event_ticker").head(MARKETS_PER_EVENT)
    # spread the per-series cap evenly across time rather than taking only the latest events
    parts = []
    for _, g in df.groupby("series_ticker"):
        parts.append(g if len(g) <= MARKETS_PER_SERIES else g.iloc[
            np.sort(rng.choice(len(g), MARKETS_PER_SERIES, replace=False))])
    return pd.concat(parts).drop_duplicates("ticker").reset_index(drop=True)


def _get(d, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def candles_to_df(cs) -> pd.DataFrame:
    rows = []
    for c in cs:
        p, b, a = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
        # live candles use *_dollars / *_fp field names; the historical archive uses close / volume / open_interest
        rows.append(dict(t=int(c["end_period_ts"]), price=_f(_get(p, "close_dollars", "close")),
                         bid=_f(_get(b, "close_dollars", "close")), ask=_f(_get(a, "close_dollars", "close")),
                         volume=_f(_get(c, "volume_fp", "volume")), oi=_f(_get(c, "open_interest_fp", "open_interest"))))
    return pd.DataFrame(rows, columns=["t", "price", "bid", "ask", "volume", "oi"])


def fetch_candles(api: Kalshi, markets: pd.DataFrame, out: Path, period=60):
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in markets.itertuples() if not (out / f"{r.ticker}.parquet").exists()]
    for i, r in enumerate(todo, 1):
        start, end = int(r.open_time.timestamp()), int(r.close_time.timestamp())
        candles_to_df(api.candles(r.series_ticker, r.ticker, start, end, period)).to_parquet(out / f"{r.ticker}.parquet")
        if i % 250 == 0 or i == len(todo):
            print(f"  candles {i}/{len(todo)}", flush=True)


def run(max_events=None, min_volume=500.0):
    api, rng = Kalshi(), np.random.default_rng(0)
    DATA.mkdir(exist_ok=True)
    keep = universe(api)
    print(f"{len(keep)} recurring series in universe; crawling settled events...")
    df = to_frame(market_rows(crawl_series(api, keep, DATA / "series_events"), series_meta=keep))
    bad = outcome_dependent_series(df[df.life_days <= MAX_LIFE_DAYS])
    print(f"{len(df)} settled binary markets; dropping {len(bad)} series with outcome-dependent close timing")
    df = select(df[~df.series_ticker.isin(bad)], min_volume, rng)
    df.to_parquet(DATA / "markets.parquet")
    print(f"{len(df)} markets from {df.series_ticker.nunique()} series kept; by category:\n"
          f"{df.category.value_counts().to_string()}")
    fetch_candles(api, df, DATA / "candles")
