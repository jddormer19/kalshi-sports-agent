"""Blend baseline: can ONE logistic regression on the market price and my model match the agent?

    logit(p) = a + b * logit(market) + c * logit(model)

1) Head-to-head on the agent's games: fit on all OTHER dev games, score on the agent's games.
2) Robustness on all of dev: forward-chaining by date (fit on earlier games, predict the next block).
DEV only; the holdout is never read here.
Run: .venv/bin/python -m benchmark.blend_baseline [--agent-run data/agent_runs/dev60_v1.jsonl]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from benchmark.score_baselines import attach_model, brier, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([logit(d.market_p), logit(d.p_model)])


def fit(train: pd.DataFrame) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=1000).fit(features(train), train.label.values)


def fit_recal(train: pd.DataFrame) -> LogisticRegression:
    """Recalibrated market (SPEC rule 8): logistic calibration of the market price alone."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(features(train)[:, :1], train.label.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-run", default=str(ROOT / "data" / "agent_runs" / "dev60_v1.jsonl"))
    a = ap.parse_args()
    bench = attach_model(pd.read_parquet(ROOT / "data" / "benchmark.parquet"))
    dev = bench[(bench.split == "dev") & bench.p_model.notna()].sort_values("start_time").reset_index(drop=True)

    # 1) head-to-head on the agent's games
    rows = [json.loads(l) for l in open(a.agent_run)]
    agent = {r["event_ticker"]: r["forecast"]["p_home"] for r in rows if r.get("forecast")}
    test = dev[dev.event_ticker.isin(agent)].copy()
    model = fit(dev[~dev.event_ticker.isin(agent)])
    b0, (b_mkt, b_mod) = model.intercept_[0], model.coef_[0]
    test["p_blend"] = model.predict_proba(features(test))[:, 1]
    test["p_recal"] = fit_recal(dev[~dev.event_ticker.isin(agent)]).predict_proba(features(test)[:, :1])[:, 1]
    test["p_agent"] = test.event_ticker.map(agent)
    y = test.label.values
    print(f"blend fit on {len(dev) - len(test)} other dev games: "
          f"logit(p) = {b0:+.3f} + {b_mkt:.3f}*logit(market) + {b_mod:.3f}*logit(model)")
    print(f"\nhead-to-head on the agent's {len(test)} games:")
    for name, col in [("market", "market_p"), ("recal mkt", "p_recal"), ("my model", "p_model"), ("blend", "p_blend"),
                      ("agent", "p_agent")]:
        print(f"  {name:9s} Brier {brier(test[col].values, y):.4f}")
    for (an, ac), (bn, bc) in [(("agent", "p_agent"), ("blend", "p_blend")), (("blend", "p_blend"), ("market", "market_p"))]:
        diff, lo, hi = paired_bootstrap(test[ac].values, test[bc].values, y)
        print(f"  {an} - {bn}: {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")

    # 2) forward-chaining over all of dev: 6 date-ordered blocks, predict block k from blocks < k
    blocks = np.array_split(np.arange(len(dev)), 6)
    preds = []
    for k in range(1, len(blocks)):
        tr, te = dev.iloc[np.concatenate(blocks[:k])], dev.iloc[blocks[k]]
        preds.append(te.assign(p_blend=fit(tr).predict_proba(features(te))[:, 1],
                               p_recal=fit_recal(tr).predict_proba(features(te)[:, :1])[:, 1]))
    fc = pd.concat(preds)
    y = fc.label.values
    print(f"\nforward-chaining over dev ({len(fc)} games predicted out of sample, by date):")
    for name, col in [("market", "market_p"), ("recal mkt", "p_recal"), ("my model", "p_model"), ("blend", "p_blend")]:
        print(f"  {name:9s} Brier {brier(fc[col].values, y):.4f}")
    diff, lo, hi = paired_bootstrap(fc.p_blend.values, fc.market_p.values, y)
    print(f"  blend - market: {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    diff, lo, hi = paired_bootstrap(fc.p_recal.values, fc.market_p.values, y)
    print(f"  recalibrated market - market: {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    for sport, g in fc.groupby("sport"):
        d, lo, hi = paired_bootstrap(g.p_blend.values, g.market_p.values, g.label.values)
        print(f"    {sport}: {len(g)} games, blend - market {d:+.4f} (CI {lo:+.4f} to {hi:+.4f})")


if __name__ == "__main__":
    main()
