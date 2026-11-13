"""Point-in-time features from hourly candles.

A snapshot at time t0 (an hourly candle boundary at least 1h before close) uses only candles
ending at or before t0. Snapshot times are sampled across each market's life, and the
tradeable price is the bid/ask quoted at t0.
"""
import numpy as np
import pandas as pd

from .collect import DATA

MAX_SNAPS_PER_MARKET = 6
MIN_HOURS_BEFORE_CLOSE = 1.0
H = 3600
FEATURES = ["mid", "logit_mid", "spread", "dist_half", "ask", "bid",
            "d_mid_1h", "d_mid_6h", "d_mid_24h", "vol_mid_24h",
            "log_vol_1h", "log_vol_24h", "log_vol_cum", "log_oi",
            "age_h", "hours_to_close", "life_frac", "hours_since_trade", "category"]
# n_siblings / mutually_exclusive / hour / weekday were dropped: they mostly identify the series
# and scored worse on inner validation.


def _logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def prepare_candles(c: pd.DataFrame) -> pd.DataFrame:
    c = c.sort_values("t").reset_index(drop=True)
    has_quote = c.bid.notna() & c.ask.notna() & (c.ask > 0)
    mid = np.where(has_quote, (c.bid.fillna(0) + c.ask) / 2, np.nan)
    c["mid"] = pd.Series(mid).fillna(c.price).ffill()
    c["volume"] = c.volume.fillna(0)
    return c


def snapshot(c: pd.DataFrame, t0: int, m) -> dict | None:
    """Features for market row `m` from prepared candles `c`, using only rows with t <= t0."""
    h = c[c.t <= t0]
    if h.empty:
        return None
    last = h.iloc[-1]
    if pd.isna(last.mid) or pd.isna(last.ask) or pd.isna(last.bid) or not (0 < last.ask < 1):
        return None  # no tradeable quote
    mid = last.mid

    def d_mid(k):
        prev = h.mid[h.t <= t0 - k * H]
        return mid - prev.iloc[-1] if len(prev) else 0.0

    w24 = h[h.t > t0 - 24 * H]
    traded = h.t[h.volume > 0]
    open_ts, close_ts = m.open_time.timestamp(), m.close_time.timestamp()
    t = pd.Timestamp(t0, unit="s", tz="UTC")
    return dict(
        mid=mid, logit_mid=_logit(mid), spread=last.ask - last.bid, dist_half=abs(mid - 0.5),
        ask=last.ask, bid=last.bid,
        d_mid_1h=d_mid(1), d_mid_6h=d_mid(6), d_mid_24h=d_mid(24),
        vol_mid_24h=w24.mid.diff().std() if len(w24) > 2 else 0.0,
        log_vol_1h=np.log1p(last.volume), log_vol_24h=np.log1p(w24.volume.sum()),
        log_vol_cum=np.log1p(h.volume.sum()), log_oi=np.log1p(last.oi if pd.notna(last.oi) else 0),
        age_h=(t0 - open_ts) / H, hours_to_close=max(close_ts - t0, 0) / H,
        life_frac=(t0 - open_ts) / max(close_ts - open_ts, 1),
        hours_since_trade=(t0 - traded.iloc[-1]) / H if len(traded) else 999.0,
        hour_utc=t.hour, dow=t.dayofweek,
    )


def build(markets: pd.DataFrame, candle_dir, rng=None, max_snaps=MAX_SNAPS_PER_MARKET, at=None) -> pd.DataFrame:
    """Training mode (at=None): up to max_snaps random hourly snapshots per market.
    Live mode (at=unix ts): one snapshot per market at `at`."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for m in markets.itertuples():
        f = candle_dir / f"{m.ticker}.parquet"
        if not f.exists():
            continue
        c = pd.read_parquet(f)
        if c.empty:
            continue
        c = prepare_candles(c)
        if at is None:
            cutoff = m.close_time.timestamp() - MIN_HOURS_BEFORE_CLOSE * H
            cand = c.t[(c.t <= cutoff) & c.bid.notna() & c.ask.between(0, 1, inclusive="neither")].to_numpy()
            if len(cand) == 0:
                continue
            times = rng.choice(cand, size=min(max_snaps, len(cand)), replace=False)
        else:
            times = [at]
        for t0 in times:
            s = snapshot(c, int(t0), m)
            if s is None:
                continue
            s.update(ticker=m.ticker, event_ticker=m.event_ticker, series_ticker=m.series_ticker,
                     category=m.category, n_siblings=m.n_siblings, mutually_exclusive=int(m.mutually_exclusive),
                     t0=int(t0), close_time=m.close_time, title=m.title,
                     y=int(m.result == "yes") if m.result in ("yes", "no") else -1)
            rows.append(s)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["category"] = df.category.astype("category")
    return df


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Each market sums to 1; events with several kept markets are down-weighted by sqrt."""
    per_mkt = df.groupby("ticker").ticker.transform("size")
    mkts_in_event = df.groupby("event_ticker").ticker.transform("nunique")
    df["w"] = 1.0 / per_mkt / np.sqrt(mkts_in_event)
    df["w"] *= len(df) / df.w.sum()
    return df


def check_no_leakage(df: pd.DataFrame, candle_dir, n=200, seed=0):
    """Rebuild a sample of snapshots from candles truncated at t0 and confirm identical features."""
    markets = pd.read_parquet(DATA / "markets.parquet").set_index("ticker")
    sample = df.sample(min(n, len(df)), random_state=seed)
    num = [f for f in FEATURES if f not in ("category", "n_siblings", "mutually_exclusive")]
    for r in sample.itertuples():
        c = prepare_candles(pd.read_parquet(candle_dir / f"{r.ticker}.parquet"))
        m = markets.loc[r.ticker]
        m = m.to_frame().T.assign(ticker=r.ticker).itertuples().__next__()
        trunc = snapshot(c[c.t <= r.t0].reset_index(drop=True), r.t0, m)
        full = df.loc[r.Index, num]
        assert trunc is not None and np.allclose([trunc[k] for k in num], full.to_numpy(float), equal_nan=True), r.ticker
        assert r.t0 <= r.close_time.timestamp() - MIN_HOURS_BEFORE_CLOSE * H
    print(f"leakage check passed on {len(sample)} snapshots")


def run(hours=None):
    markets = pd.read_parquet(DATA / "markets.parquet")
    df = add_weights(build(markets, DATA / "candles"))
    df.to_parquet(DATA / "features.parquet")
    print(f"{len(df)} snapshots from {df.ticker.nunique()} markets / {df.event_ticker.nunique()} events "
          f"/ {df.series_ticker.nunique()} series; base YES rate {df.y.mean():.3f}")
    check_no_leakage(df, DATA / "candles")
