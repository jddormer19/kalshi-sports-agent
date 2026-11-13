"""Score currently open markets from the trained series universe.

Read-only: prints a ranked edge table and writes data/live_edges.csv. Places no orders.
"""
import time

import joblib
import numpy as np
import pandas as pd

from .api import Kalshi
from .backtest import taker_fee
from .collect import DATA, MAX_LIFE_DAYS, candles_to_df, market_rows, to_frame
from .features import build


def run(hours=None, top=25, min_volume=500.0):
    api, model = Kalshi(), joblib.load(DATA / "model.joblib")
    series = sorted(pd.read_parquet(DATA / "markets.parquet").series_ticker.unique())
    now = int(time.time())
    print(f"fetching open events for {len(series)} series...")
    events = []
    for s in series:
        events += (api.get("/events", series_ticker=s, status="open", with_nested_markets="true", limit=200)
                   or {}).get("events", [])
    m = to_frame(market_rows(events, status_ok=None))
    m = m[(m.volume >= min_volume) & (m.life_days <= MAX_LIFE_DAYS) & m.yes_bid.notna()
          & m.yes_ask.between(0.01, 0.99) & (m.close_time.map(pd.Timestamp.timestamp) > now + 3600)]
    m = m.reset_index(drop=True)
    print(f"{len(m)} open markets (vol >= {min_volume}, closing in > 1h); fetching candles...")

    cdir = DATA / "live_candles"
    cdir.mkdir(exist_ok=True)
    for i, r in enumerate(m.itertuples(), 1):
        c = candles_to_df(api.candles(r.series_ticker, r.ticker, int(r.open_time.timestamp()), now, 60))
        c = c[c.t <= now]  # completed candles only
        live = dict(t=now, price=r.last_price, bid=r.yes_bid, ask=r.yes_ask, volume=0.0, oi=r.open_interest)
        pd.concat([c, pd.DataFrame([live])], ignore_index=True).to_parquet(cdir / f"{r.ticker}.parquet")
        if i % 200 == 0:
            print(f"  {i}/{len(m)}", flush=True)

    X = build(m, cdir, at=now)
    X["category"] = pd.Categorical(X.category.astype(str), categories=model.categories)
    X["p_model"] = model.predict(X)
    yes_cost = X.ask + X.ask.map(taker_fee)
    no_cost = (1 - X.bid) + (1 - X.bid).map(taker_fee)
    X["edge_yes"], X["edge_no"] = X.p_model - yes_cost, (1 - X.p_model) - no_cost
    X["side"] = np.where(X.edge_yes >= X.edge_no, "YES", "NO")
    X["edge"] = X[["edge_yes", "edge_no"]].max(axis=1)
    X = X.sort_values("edge", ascending=False)
    cols = ["ticker", "category", "title", "bid", "ask", "p_model", "side", "edge", "hours_to_close"]
    X[cols].to_csv(DATA / "live_edges.csv", index=False)
    with pd.option_context("display.max_colwidth", 45, "display.width", 220):
        print(X[cols].head(top).round(3).to_string(index=False))
    print(f"\n{(X.edge > 0).sum()} of {len(X)} markets show positive fee-adjusted edge; full table in data/live_edges.csv")
