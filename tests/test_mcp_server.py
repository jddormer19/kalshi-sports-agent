"""Tests for the MCP research server: tools work through MCP, and nothing after t is ever returned."""
import asyncio
import json

import pandas as pd
import pytest

from agent import mcp_server as srv

GAME = "KXMLBGAME-26SEP161340NYYMIN"     # Sep 16, 2026, 13:40 ET start; t = 16:40 UTC; mid at t = 0.405


def call(name, **args):
    """Call a tool the way an MCP client would (through the server, not the Python function)."""
    result = asyncio.run(srv.server.call_tool(name, args))
    blocks = result[0] if isinstance(result, tuple) else result
    blocks = getattr(blocks, "content", blocks)
    return json.loads(blocks[0].text)


def test_tools_are_registered():
    names = {t.name for t in asyncio.run(srv.server.list_tools())}
    assert {"get_game", "market_price"} <= names


def test_get_game():
    g = call("get_game", event_ticker=GAME)
    assert (g["sport"], g["home_team"], g["away_team"]) == ("MLB", "MIN", "NYY")
    assert g["decision_time_utc"].startswith("2026-09-16T16:40")


def test_market_price_is_point_in_time():
    p = call("market_price", event_ticker=GAME)
    assert p["available"] and p["p_home"] == 0.405 and p["spread"] == 0.01
    assert pd.Timestamp(p["price_as_of_utc"]) <= pd.Timestamp("2026-09-16T16:40Z")   # never after t


def test_unknown_game_is_an_error():
    with pytest.raises(Exception):
        call("get_game", event_ticker="NOT-A-GAME")


def test_calls_are_logged_without_leaks():
    call("market_price", event_ticker=GAME)
    last = json.loads(srv.CALL_LOG.read_text().strip().splitlines()[-1])
    assert last["tool"] == "market_price" and last["leak"] is False
