"""Run the agent on a sample of DEV games and score it against the market and my model on the same games.

    .venv/bin/python -m evals.run_agent --n 3            # smoke test
    .venv/bin/python -m evals.run_agent --n 60 --tag v1  # small dev eval (stratified by sport)
Results append to data/agent_runs/<tag>.jsonl (one line per game; already-run games are skipped).
Never touches the holdout: there is deliberately no option for it here.
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.graph import run_games
from benchmark.score_baselines import attach_model, brier, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent


def sample_dev(n: int, seed: int = 0) -> list[str]:
    """Stratified by sport (proportional, at least 1 per sport), only games that have a model prediction."""
    b = attach_model(pd.read_parquet(ROOT / "data" / "benchmark.parquet"))
    dev = b[(b.split == "dev") & b.p_model.notna()]
    share = dev.sport.value_counts(normalize=True)
    picks = []
    for sport, frac in share.items():
        k = max(1, round(n * frac))
        picks += dev[dev.sport == sport].sample(min(k, (dev.sport == sport).sum()), random_state=seed).event_ticker.tolist()
    return picks[:n]


def score(rows: list[dict]) -> None:
    b = attach_model(pd.read_parquet(ROOT / "data" / "benchmark.parquet")).set_index("event_ticker")
    ok = [r for r in rows if r.get("forecast")]
    if not ok:
        print("no successful runs to score")
        return
    d = b.loc[[r["event_ticker"] for r in ok]].assign(p_agent=[r["forecast"]["p_home"] for r in ok])
    y = d.label.values
    print(f"\n{len(ok)} games scored ({len(rows) - len(ok)} errors)")
    for name, col in [("market", "market_p"), ("my model", "p_model"), ("agent", "p_agent")]:
        print(f"  {name:9s} Brier {brier(d[col].values, y):.4f}")
    diff, lo, hi = paired_bootstrap(d.p_agent.values, d.market_p.values, y)
    print(f"  agent - market: {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"  mean |agent - market|: {np.mean(np.abs(d.p_agent - d.market_p)):.3f}")
    costs = [r["cost_usd"] for r in ok]
    secs = [r["seconds"] for r in ok]
    print(f"  cost: ${sum(costs):.2f} total, ${np.mean(costs):.3f}/game | time: {np.mean(secs):.0f}s/game "
          f"| tool calls: {np.mean([r['tool_calls'] for r in ok]):.1f}/game | leaks: {sum(r['leaks'] for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "data" / "agent_runs" / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["event_ticker"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [e for e in sample_dev(a.n, a.seed) if e not in done]
    print(f"{len(todo)} games to run ({len(done)} already done) -> {out}")
    rows = asyncio.run(run_games(todo))
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    score([json.loads(l) for l in out.read_text().splitlines()])


if __name__ == "__main__":
    main()
