"""MCP server exposing point-in-time research tools for the Kalshi sports agent (Milestone 3).

Design rule: the SERVER decides what "before t" means, never the agent. Every tool takes an event_ticker; the server
looks up that game's decision time t (SPEC rule 4) and only ever returns data from before t. Every call is logged
with the newest timestamp it returned, so the eval runner can audit for leakage (SPEC rule 6).

Run (stdio):  .venv/bin/python -m agent.mcp_server
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from mcp.server.mcpserver import MCPServer

from benchmark.build_benchmark import load_candles, market_price_at
from benchmark.score_baselines import attach_model

ROOT = Path(__file__).resolve().parent.parent
CALL_LOG = ROOT / "data" / "agent_tool_calls.jsonl"

server = MCPServer("kalshi-research", instructions=(
    "Point-in-time research tools for forecasting sports game winners. Every tool answers as of the game's decision "
    "time t (60 minutes before the scheduled start); nothing after t is ever returned."))


@lru_cache(maxsize=1)
def games() -> pd.DataFrame:
    """The frozen benchmark, indexed by event_ticker."""
    return pd.read_parquet(ROOT / "data" / "benchmark.parquet").set_index("event_ticker")


MODEL_DESCRIPTIONS = {   # verified against the model code on 2026-11-17
    "MLB": "logistic regression on team strength (Pythagorean win %), recent run differential, and each starter's "
           "FIP and K-BB%",
    "NBA": "ridge regression on point margin (home court, efficiency gap, recent margin, rest and back-to-backs, "
           "player availability), turned into a win probability by logistic calibration",
    "NFL": "ridge regression on point margin (home field, EPA overall/pass/rush, success rate, explosive plays, "
           "points, QB quality, rest, missing skill players), turned into a win probability from the predicted margin",
}


@lru_cache(maxsize=1)
def model_probs() -> pd.Series:
    """Walk-forward model P(home wins) per game (each season predicted by a model fit only on earlier seasons)."""
    bench = attach_model(pd.read_parquet(ROOT / "data" / "benchmark.parquet"))
    return bench.set_index("event_ticker").p_model


def game(event_ticker: str) -> pd.Series:
    try:
        return games().loc[event_ticker]
    except KeyError:
        raise ValueError(f"unknown game {event_ticker!r}; use an event_ticker from the benchmark") from None


def log_call(tool: str, event_ticker: str, t: pd.Timestamp, newest_data: pd.Timestamp | None) -> None:
    """Append one audit line. leak = the tool returned something timestamped after t (should never be true)."""
    rec = {"logged_at": datetime.now(timezone.utc).isoformat(), "tool": tool, "event_ticker": event_ticker,
           "t": t.isoformat(), "newest_data": newest_data.isoformat() if newest_data is not None else None,
           "leak": bool(newest_data is not None and newest_data > t)}
    with open(CALL_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- tools
@server.tool()
def get_game(event_ticker: str) -> dict:
    """Basic facts about a game: sport, home and away team, scheduled start, and the decision time t
    (all forecasts are made as of t, 60 minutes before the start)."""
    g = game(event_ticker)
    log_call("get_game", event_ticker, g.t, None)
    return {"event_ticker": event_ticker, "sport": g.sport, "home_team": g.home_team, "away_team": g.away_team,
            "scheduled_start_utc": g.start_time.isoformat(), "decision_time_utc": g.t.isoformat()}


@server.tool()
def market_price(event_ticker: str) -> dict:
    """The Kalshi market's probability that the HOME team wins, as of the decision time t: the bid/ask midpoint from
    the last hourly candle that ended at or before t, plus the spread (a wide spread means a less reliable price)."""
    g = game(event_ticker)
    candles = load_candles(g.home_ticker)
    priced = market_price_at(candles, g.t)
    before = candles[candles.t <= g.t.timestamp()]
    newest = pd.Timestamp(before.t.max(), unit="s", tz="UTC") if len(before) else None
    log_call("market_price", event_ticker, g.t, newest)
    if priced is None:
        return {"event_ticker": event_ticker, "available": False, "reason": "no usable quote before t"}
    mid, spread = priced
    return {"event_ticker": event_ticker, "available": True, "p_home": round(float(mid), 4),
            "spread": round(float(spread), 4), "price_as_of_utc": newest.isoformat()}


@server.tool()
def model_win_prob(event_ticker: str) -> dict:
    """My own model's probability that the HOME team wins. The model for each season was trained only on earlier
    seasons, and its inputs use only data from before the game's date. May be unavailable for some games."""
    g = game(event_ticker)
    p = model_probs().get(event_ticker)
    # features use data from before the game's date: newest possible input = midnight Eastern on game day
    game_day = g.start_time.tz_convert("America/New_York").normalize().tz_convert("UTC")
    log_call("model_win_prob", event_ticker, g.t, game_day)
    if p is None or pd.isna(p):
        return {"event_ticker": event_ticker, "available": False, "reason": "no model prediction for this game"}
    return {"event_ticker": event_ticker, "available": True, "p_home": round(float(p), 4),
            "model": MODEL_DESCRIPTIONS[g.sport], "trained_on": f"seasons before {g.start_time.year}",
            "inputs_as_of_utc": game_day.isoformat()}


if __name__ == "__main__":
    server.run("stdio")
