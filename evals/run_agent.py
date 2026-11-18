"""Run the agent on a sample of DEV games and score it against the market (and my model, when used) on the same games.

    .venv/bin/python -m evals.run_agent --n 3                                   # smoke test (1h benchmark)
    .venv/bin/python -m evals.run_agent --n 60 --tag dev60_v1                   # 1h benchmark, all tools
    .venv/bin/python -m evals.run_agent --n 60 --tag nba24_inj --benchmark benchmark_24h.parquet --sport NBA \
        --tools get_game,market_price,team_recent_form,injury_report             # 24h variant, NBA, chosen tools
Results append to data/agent_runs/<tag>.jsonl (one line per game; already-run games are skipped).
Never touches the holdout: there is deliberately no option for it here.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.score_baselines import attach_model, brier, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent


def load_bench(name: str) -> pd.DataFrame:
    return attach_model(pd.read_parquet(ROOT / "data" / name))


NBA_REGULAR_SEASON_START = "2026-10-19"   # first 2026-27 injury report; earlier fall NBA games are preseason


def is_nba_preseason(bench: pd.DataFrame) -> pd.Series:
    """NBA preseason games (no injury reports, rested starters): Sep 1 to the regular-season start."""
    day = bench.start_time.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return (bench.sport == "NBA") & (day >= "2026-09-01") & (day < NBA_REGULAR_SEASON_START)


def sample_dev(n: int, bench: pd.DataFrame, sport: str | None, need_model: bool, seed: int = 0) -> list[str]:
    """Stratified by sport (proportional, at least 1 per sport); same seed -> same games across runs."""
    dev = bench[(bench.split == "dev") & ~is_nba_preseason(bench)]
    if sport:
        dev = dev[dev.sport == sport]
    if need_model:
        dev = dev[dev.p_model.notna()]
    share = dev.sport.value_counts(normalize=True)
    picks = []
    for s, frac in share.items():
        k = max(1, round(n * frac))
        picks += dev[dev.sport == s].sample(min(k, (dev.sport == s).sum()), random_state=seed).event_ticker.tolist()
    return picks[:n]


def score(rows: list[dict], bench: pd.DataFrame) -> None:
    b = bench.set_index("event_ticker")
    ok = [r for r in rows if r.get("forecast")]
    if not ok:
        print("no successful runs to score")
        return
    d = b.loc[[r["event_ticker"] for r in ok]].assign(p_agent=[r["forecast"]["p_home"] for r in ok])
    y = d.label.values
    print(f"\n{len(ok)} games scored ({len(rows) - len(ok)} errors)")
    systems = [("market", "market_p"), ("agent", "p_agent")]
    if d.p_model.notna().all():
        systems.insert(1, ("my model", "p_model"))
    for name, col in systems:
        print(f"  {name:9s} Brier {brier(d[col].values, y):.4f}")
    one_h = pd.read_parquet(ROOT / "data" / "benchmark.parquet").set_index("event_ticker").market_p
    if not d.index.to_series().map(one_h).isna().any() and not np.allclose(d.index.map(one_h), d.market_p):
        print(f"  (market 1h before, for reference: Brier {brier(d.index.map(one_h).values, y):.4f})")
    diff, lo, hi = paired_bootstrap(d.p_agent.values, d.market_p.values, y)
    print(f"  agent - market: {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"  mean |agent - market|: {np.mean(np.abs(d.p_agent - d.market_p)):.3f}")
    costs, secs = [r["cost_usd"] for r in ok], [r["seconds"] for r in ok]
    tools = pd.Series([t["tool"] for r in ok for t in r.get("trace", [])]).value_counts().to_dict()
    print(f"  cost: ${sum(costs):.2f} total, ${np.mean(costs):.3f}/game | time: {np.mean(secs):.0f}s/game "
          f"| tool calls: {np.mean([r['tool_calls'] for r in ok]):.1f}/game {tools} | leaks: {sum(r['leaks'] for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", default="benchmark.parquet", help="file in data/ (e.g. benchmark_24h.parquet)")
    ap.add_argument("--sport", default=None, help="MLB, NBA or NFL (default: all, stratified)")
    ap.add_argument("--tools", default="", help="comma-separated allowlist (default: every server tool)")
    a = ap.parse_args()
    os.environ["BENCHMARK_FILE"] = a.benchmark          # the MCP server subprocess inherits these
    os.environ["AGENT_TOOLS"] = a.tools
    from agent.graph import run_games                    # imported after AGENT_TOOLS is set
    bench = load_bench(a.benchmark)
    out = ROOT / "data" / "agent_runs" / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["event_ticker"] for l in out.read_text().splitlines()} if out.exists() else set()
    need_model = not a.tools or "model_win_prob" in a.tools.split(",")
    todo = [e for e in sample_dev(a.n, bench, a.sport, need_model, a.seed) if e not in done]
    print(f"{len(todo)} games to run ({len(done)} already done) -> {out} | benchmark {a.benchmark} | "
          f"tools {a.tools or 'all'}", flush=True)
    rows = asyncio.run(run_games(todo))
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    score([json.loads(l) for l in out.read_text().splitlines()], bench)


if __name__ == "__main__":
    main()
