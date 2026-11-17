"""Score baseline systems on the frozen benchmark (SPEC rules 8-9).

Systems: market mid at t, and "my model alone" (MLB walk-forward predictions exported from
the MLB model project (a separate private repo); NBA/NFL to be added). Scores the DEV split only unless --holdout is passed;
the holdout is scored once, at the end, with the final system (SPEC rule 7).
Run: .venv/bin/python -m benchmark.score_baselines [--holdout]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MLB_PREDS = ROOT / "data" / "model_preds_mlb_2026.csv"
MLB_CODE_FIX = {"ARI": "AZ"}            # MLB Stats API abbreviation -> Kalshi ticker code
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
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def attach_mlb(bench: pd.DataFrame) -> pd.DataFrame:
    preds = pd.read_csv(MLB_PREDS, parse_dates=["game_datetime"])
    preds["home_abbr"] = preds.home_abbr.replace(MLB_CODE_FIX)
    preds["away_abbr"] = preds.away_abbr.replace(MLB_CODE_FIX)
    mlb = bench[bench.sport == "MLB"].reset_index()
    m = mlb.merge(preds, left_on=["home_team", "away_team"], right_on=["home_abbr", "away_abbr"], how="left")
    m["gap"] = (m.game_datetime - m.start_time).abs()
    m = m[m.gap <= MATCH_WINDOW].sort_values("gap").drop_duplicates("index")   # closest game (doubleheaders)
    out = bench.copy()
    out["p_model"] = np.nan
    out.loc[m["index"], "p_model"] = m.p_model.values
    out.loc[m["index"], "model_label_check"] = m.home_win.values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="score the holdout (only for the final, frozen system)")
    split = "holdout" if ap.parse_args().holdout else "dev"
    bench = attach_mlb(pd.read_parquet(ROOT / "data" / "benchmark.parquet"))
    mlb = bench[bench.sport == "MLB"]
    matched = mlb.p_model.notna()
    print(f"MLB games matched to a model prediction: {matched.sum()} of {len(mlb)}")
    bad = mlb[matched & (mlb.model_label_check != mlb.label)]
    print(f"label disagreements between Kalshi and MLB data: {len(bad)}")
    s = bench[(bench.split == split) & bench.p_model.notna()]
    y, p_market, p_model = s.label.values, s.market_p.values, s.p_model.values
    rows = [("market at t", len(s), brier(p_market, y), logloss(p_market, y)),
            ("my model alone", len(s), brier(p_model, y), logloss(p_model, y)),
            ("always base rate", len(s), brier(np.full(len(y), y.mean()), y), logloss(np.full(len(y), y.mean()), y))]
    print(f"\n{split.upper()} split, MLB games with both forecasts:")
    print(pd.DataFrame(rows, columns=["system", "games", "Brier", "log loss"]).round(4).to_string(index=False))
    diff, lo, hi = paired_bootstrap(p_model, p_market, y)
    print(f"\nmodel - market Brier: {diff:+.4f}  (95% CI {lo:+.4f} to {hi:+.4f}; "
          f"{'model better' if hi < 0 else 'market better' if lo > 0 else 'no detectable difference'})")
    print(f"correlation(model, market): {np.corrcoef(p_model, p_market)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
