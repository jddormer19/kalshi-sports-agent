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

ROOT = Path(__file__).resolve().parent.parent
CALL_LOG = ROOT / "data" / "agent_tool_calls.jsonl"

server = MCPServer("kalshi-research", instructions=(
    "Point-in-time research tools for forecasting sports game winners. Every tool answers as of the game's decision "
    "time t (60 minutes before the scheduled start); nothing after t is ever returned."))


@lru_cache(maxsize=1)
def games() -> pd.DataFrame:
    """The frozen benchmark, indexed by event_ticker."""
    return pd.read_parquet(ROOT / "data" / "benchmark.parquet").set_index("event_ticker")


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


if __name__ == "__main__":
    server.run("stdio")
