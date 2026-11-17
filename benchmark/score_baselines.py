"""Score baseline systems on the frozen benchmark (SPEC rules 8-9).

Systems: the market mid at t, and "my model alone" (walk-forward predictions: each season predicted by a model fit
only on earlier seasons; exported by benchmark/export_mlb_preds.py and benchmark/export_hub_preds.py).
Scores the DEV split only unless --holdout is passed; the holdout is scored once, at the end, with the final
system (SPEC rule 7).
Run: .venv/bin/python -m benchmark.score_baselines [--holdout]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MLB_PREDS = DATA / "model_preds_mlb_2026.csv"
NBA_PREDS = DATA / "model_preds_nba.csv"
NFL_PREDS = DATA / "model_preds_nfl.csv"
# model team code -> Kalshi ticker code
CODE_FIX = {"MLB": {"ARI": "AZ"},
            "NBA": {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"},
            "NFL": {"JAX": "JAC", "LA": "LAR"}}
MATCH_WINDOW = pd.Timedelta(hours=3)    # model and Kalshi start times must agree within this


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_bootstrap(a, b, y, n=10_000, seed=0):
    """95% CI of Brier(a) - Brier(b), resampling games (SPEC rule 9)."""
    d = (a - y) ** 2 - (b - y) ** 2
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def model_preds() -> pd.DataFrame:
    """All model predictions in one shape: sport, home, away, start (UTC), p_model, model_home_win."""
    mlb = pd.read_csv(MLB_PREDS)
    mlb = pd.DataFrame({"sport": "MLB", "home": mlb.home_abbr, "away": mlb.away_abbr,
                        "start": pd.to_datetime(mlb.game_datetime, utc=True), "p_model": mlb.p_model,
                        "model_home_win": mlb.home_win})
    nba = pd.read_csv(NBA_PREDS)
    nba = pd.DataFrame({"sport": "NBA", "home": nba.home, "away": nba.away,
                        "start": pd.to_datetime(nba.start, utc=True), "p_model": nba.pred_p_home,
                        "model_home_win": nba.home_win})
    nfl = pd.read_csv(NFL_PREDS)
    kick = pd.to_datetime(nfl.gameday + " " + nfl.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    nfl = pd.DataFrame({"sport": "NFL", "home": nfl.home_team, "away": nfl.away_team, "start": kick,
                        "p_model": nfl.pred_p_home, "model_home_win": nfl.home_win})
    out = pd.concat([mlb, nba, nfl], ignore_index=True)
    for sport, fix in CODE_FIX.items():
        m = out.sport == sport
        out.loc[m, "home"] = out.loc[m, "home"].replace(fix)
        out.loc[m, "away"] = out.loc[m, "away"].replace(fix)
    return out


def attach_model(bench: pd.DataFrame) -> pd.DataFrame:
    """Add p_model to each benchmark game: same sport, home and away, closest start within MATCH_WINDOW."""
    preds = model_preds()
    b = bench.reset_index()
    m = b.merge(preds, left_on=["sport", "home_team", "away_team"], right_on=["sport", "home", "away"],
                how="inner", suffixes=("", "_model"))
    m["gap"] = (m.start - m.start_time).abs()
    m = m[m.gap <= MATCH_WINDOW].sort_values("gap").drop_duplicates("index")   # doubleheaders: closest game
    out = bench.copy()
    out["p_model"] = np.nan
    out["model_home_win"] = np.nan
    out.loc[m["index"], "p_model"] = m.p_model.values
    out.loc[m["index"], "model_home_win"] = m.model_home_win.values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="score the holdout (only for the final, frozen system)")
    split = "holdout" if ap.parse_args().holdout else "dev"
    bench = attach_model(pd.read_parquet(DATA / "benchmark.parquet"))

    print("coverage (benchmark games with a model prediction):")
    for sport, g in bench.groupby("sport"):
        matched = g.p_model.notna()
        disagree = int((matched & g.model_home_win.notna() & (g.model_home_win != g.label)).sum())
        print(f"  {sport}: {matched.sum()} of {len(g)} matched; result disagreements with Kalshi: {disagree}")

    s = bench[(bench.split == split) & bench.p_model.notna()]
    rows = []
    for sport, g in [("ALL", s)] + list(s.groupby("sport")):
        y, p_market, p_model = g.label.values, g.market_p.values, g.p_model.values
        diff, lo, hi = paired_bootstrap(p_model, p_market, y)
        verdict = "model better" if hi < 0 else "market better" if lo > 0 else "no detectable difference"
        rows.append((sport, len(g), brier(p_market, y), brier(p_model, y), brier(np.full(len(y), y.mean()), y),
                     diff, lo, hi, verdict, np.corrcoef(p_model, p_market)[0, 1]))
    cols = ["sport", "games", "market Brier", "model Brier", "base-rate Brier", "model-market", "CI low", "CI high",
            "verdict", "corr"]
    print(f"\n{split.upper()} split, games with both forecasts:")
    print(pd.DataFrame(rows, columns=cols).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
