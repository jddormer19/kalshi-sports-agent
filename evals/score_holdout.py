"""ONE-TIME holdout scoring, exactly as pre-registered in benchmark/SPEC.md (change log, 2026-11-18).

Systems: market at t (1h), my model alone, market+model blend (fit on ALL dev games), and agent v1 on a 100-game
holdout sample (games with a model prediction, stratified by sport, seed 0).
Refuses to run if results already exist: the holdout is scored once. Results: data/holdout/results.json
Run: .venv/bin/python -m evals.score_holdout
"""
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.blend_baseline import features, fit
from benchmark.score_baselines import attach_model, brier, logloss, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "holdout"
AGENT_TOOLS = "get_game,market_price,model_win_prob,team_recent_form"
AGENT_N, SEED = 100, 0


def agent_sample(hold: pd.DataFrame) -> list[str]:
    h = hold[hold.p_model.notna()]
    share = h.sport.value_counts(normalize=True)
    picks = []
    for sport, frac in share.items():
        k = max(1, round(AGENT_N * frac))
        picks += h[h.sport == sport].sample(min(k, (h.sport == sport).sum()), random_state=SEED).event_ticker.tolist()
    return picks[:AGENT_N]


def summarize(d: pd.DataFrame, cols: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for sport, g in [("ALL", d)] + list(d.groupby("sport")):
        y = g.label.values
        for name, col in cols:
            r = {"sport": sport, "system": name, "games": int(len(g)), "brier": brier(g[col].values, y),
                 "log_loss": logloss(g[col].values, y)}
            if col != "market_p":
                r["minus_market"], r["ci_low"], r["ci_high"] = paired_bootstrap(g[col].values, g.market_p.values, y)
            rows.append(r)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("holdout already scored (data/holdout/results.json exists); it is scored once by design")
    bench = attach_model(pd.read_parquet(ROOT / "data" / "benchmark.parquet"))
    dev = bench[(bench.split == "dev") & bench.p_model.notna()]
    hold = bench[bench.split == "holdout"].copy()

    # 1) baselines on every holdout game (market on all; model and blend where my model has a prediction)
    blend = fit(dev)
    hm = hold[hold.p_model.notna()].copy()
    hm["p_blend"] = blend.predict_proba(features(hm))[:, 1]
    results = {"pre_registration": "benchmark/SPEC.md change log 2026-11-18",
               "blend_coefficients": {"intercept": float(blend.intercept_[0]), "market": float(blend.coef_[0][0]),
                                      "model": float(blend.coef_[0][1])},
               "market_all_holdout": summarize(hold, [("market", "market_p")]),
               "baselines": summarize(hm, [("market", "market_p"), ("my model", "p_model"), ("blend", "p_blend")])}

    # 2) agent v1 on the pre-registered 100-game sample
    os.environ["BENCHMARK_FILE"] = "benchmark.parquet"
    os.environ["AGENT_TOOLS"] = AGENT_TOOLS
    from agent.graph import MODEL, EFFORT, run_games
    games = agent_sample(hm)
    runs = asyncio.run(run_games(games))
    with open(OUT / "agent_runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r, default=str) + "\n")
    ok = {r["event_ticker"]: r for r in runs if r.get("forecast")}
    a = hm.set_index("event_ticker").loc[list(ok)].reset_index()
    a["p_agent"] = [ok[e]["forecast"]["p_home"] for e in a.event_ticker]
    results["agent"] = {"model": MODEL, "effort": EFFORT, "tools": AGENT_TOOLS, "sample": len(games),
                        "scored": len(ok), "errors": len(runs) - len(ok), "leaks": sum(r["leaks"] for r in runs),
                        "cost_usd": round(sum(r["cost_usd"] for r in ok.values()), 2),
                        "seconds_per_game": round(float(np.mean([r["seconds"] for r in ok.values()])), 1),
                        "scores": summarize(a, [("market", "market_p"), ("my model", "p_model"),
                                                ("blend", "p_blend"), ("agent", "p_agent")])}
    (OUT / "results.json").write_text(json.dumps(results, indent=1))

    def show(rows, title):
        print(f"\n{title}")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    show(results["market_all_holdout"], "market, all holdout games")
    show(results["baselines"], "baselines, holdout games with a model prediction")
    show(results["agent"]["scores"], f"agent v1 sample ({len(ok)} games, ${results['agent']['cost_usd']}, "
                                     f"{results['agent']['leaks']} leaks, {results['agent']['errors']} errors)")


if __name__ == "__main__":
    main()
