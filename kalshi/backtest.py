"""Fee-aware backtest on the held-out test period.

One trade per market: the first test snapshot (chronologically) whose fee-adjusted edge
exceeds `margin`. Buy YES at the ask or NO at (1 - bid), i.e. we pay the spread as a taker,
and hold to settlement.
"""
import math

import numpy as np
import pandas as pd

from .collect import DATA

TAKER_RATE = 0.07


def taker_fee(price, contracts=1):
    """Kalshi taker fee: ceil to the cent of 0.07 * C * P * (1 - P)."""
    return math.ceil(round(100 * TAKER_RATE * contracts * price * (1 - price), 6)) / 100


def pick_trades(df: pd.DataFrame, margin: float) -> pd.DataFrame:
    df = df.sort_values("t0").copy()
    yes_cost = df.ask + df.ask.map(taker_fee)
    no_price = 1 - df.bid
    no_cost = no_price + no_price.map(taker_fee)
    df["edge_yes"] = df.p_model - yes_cost
    df["edge_no"] = (1 - df.p_model) - no_cost
    df["side"] = np.where(df.edge_yes >= df.edge_no, "YES", "NO")
    df["edge"] = df[["edge_yes", "edge_no"]].max(axis=1)
    df["price"] = np.where(df.side == "YES", df.ask, no_price)
    df["cost"] = np.where(df.side == "YES", yes_cost, no_cost)
    df["win"] = np.where(df.side == "YES", df.y == 1, df.y == 0)
    df = df[(df.edge > margin) & (df.price > 0.0) & (df.price < 1.0)]
    t = df.groupby("ticker", sort=False).head(1).copy()
    t["pnl"] = t.win.astype(float) - t.cost
    # quarter-Kelly stake on a fixed (non-compounding) $1,000 bankroll
    p_side = np.where(t.side == "YES", t.p_model, 1 - t.p_model)
    kelly = np.clip((p_side - t.cost) / (1 - t.price), 0, 1) * 0.25
    t["contracts_kelly"] = np.floor(1000 * kelly / t.price)
    fees_k = [taker_fee(p, c) for p, c in zip(t.price, t.contracts_kelly)]
    t["pnl_kelly"] = t.contracts_kelly * (t.win.astype(float) - t.price) - np.array(fees_k)
    return t


def bootstrap_roi(t: pd.DataFrame, n=2000, seed=0):
    """Event-clustered bootstrap CI for flat-stake ROI (outcomes within an event are correlated)."""
    rng = np.random.default_rng(seed)
    g = t.groupby("event_ticker").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
    pnl, cost = g.pnl.to_numpy(), g.cost.to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g)))
    rois = pnl[idx].sum(1) / cost[idx].sum(1)
    return np.percentile(rois, [2.5, 97.5])


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(trades=0)
    lo, hi = bootstrap_roi(t)
    return dict(trades=len(t), events=t.event_ticker.nunique(), hit_rate=t.win.mean(),
                avg_price=t.price.mean(), avg_edge=t.edge.mean(),
                roi_flat=t.pnl.sum() / t.cost.sum(), roi_ci_lo=lo, roi_ci_hi=hi,
                pnl_kelly=t.pnl_kelly.sum())


def run(hours=None, margin=0.02):
    df = pd.read_parquet(DATA / "test_preds.parquet")
    print(f"test set: {len(df)} snapshots, {df.ticker.nunique()} markets\n")
    rows = {m: summarize(pick_trades(df, m)) for m in sorted({0.0, 0.01, margin, 0.05, 0.10})}
    print("Margin sweep (flat 1-contract ROI after fees, 95% event-bootstrap CI; Kelly PnL on $1k):")
    print(pd.DataFrame(rows).T.round(4).to_string())

    t = pick_trades(df, margin)
    if t.empty:
        print("\nno trades at this margin"); return
    print(f"\nAt margin {margin}: by category")
    print(t.groupby("category", observed=True).agg(trades=("pnl", "size"), hit=("win", "mean"),
          pnl=("pnl", "sum"), cost=("cost", "sum")).assign(roi=lambda d: d.pnl / d.cost).round(3).to_string())
    print("\nby side / price")
    t["price_bucket"] = pd.cut(t.price, [0, .1, .3, .5, .7, .9, 1])
    print(t.groupby(["side", "price_bucket"], observed=True).agg(trades=("pnl", "size"), hit=("win", "mean"),
          pnl=("pnl", "sum")).round(3).to_string())
    t.to_csv(DATA / "backtest_trades.csv", index=False)
