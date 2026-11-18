"""Tests for the injury_report MCP tool: point-in-time, NBA only, never a leak. Skipped if the ingest hasn't run."""
import asyncio
import json

import pandas as pd
import pytest

from agent import mcp_server as srv
from news import nba_injuries

pytestmark = pytest.mark.skipif(not (nba_injuries.OUT / "rows.parquet").exists(), reason="NBA injury ingest not run")


def call(name, **args):
    result = asyncio.run(srv.server.call_tool(name, args))
    blocks = result[0] if isinstance(result, tuple) else result
    blocks = getattr(blocks, "content", blocks)
    return json.loads(blocks[0].text)


def nba_games(n=25):
    g = srv.games()
    return g[g.sport == "NBA"].index[:n]


def test_report_is_never_published_after_t():
    for ev in nba_games():
        r = call("injury_report", event_ticker=ev)
        if r["available"]:
            assert pd.Timestamp(r["report_published_utc"]) <= srv.game(ev).t


def test_listed_teams_are_this_games_teams():
    for ev in nba_games():
        r = call("injury_report", event_ticker=ev)
        if r.get("game_listed"):
            g = srv.game(ev)
            assert (r["home"]["team"], r["away"]["team"]) == (g.home_team, g.away_team)


def test_non_nba_game_is_unavailable_not_an_error():
    g = srv.games()
    mlb = g[g.sport == "MLB"].index[0]
    r = call("injury_report", event_ticker=mlb)
    assert r["available"] is False


def test_injury_calls_never_leak():
    for ev in nba_games(10):
        call("injury_report", event_ticker=ev)
    lines = [json.loads(l) for l in srv.CALL_LOG.read_text().splitlines()]
    assert not any(l["leak"] for l in lines if l["tool"] == "injury_report")
