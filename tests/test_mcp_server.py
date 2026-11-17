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
    assert {"get_game", "market_price", "model_win_prob", "team_recent_form"} <= names


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


def test_model_win_prob_matches_the_exported_prediction():
    m = call("model_win_prob", event_ticker=GAME)
    preds = pd.read_csv(srv.ROOT / "data" / "model_preds_mlb_2026.csv")
    row = preds[(preds.home_abbr == "MIN") & (preds.away_abbr == "NYY") & preds.game_datetime.str.startswith("2026-09-16")]
    assert m["available"] and m["p_home"] == round(float(row.p_model.iloc[0]), 4)
    assert pd.Timestamp(m["inputs_as_of_utc"]) <= pd.Timestamp("2026-09-16T16:40Z")


def test_model_win_prob_unavailable_is_not_an_error():
    missing = srv.model_probs()[srv.model_probs().isna()].index[0]
    m = call("model_win_prob", event_ticker=missing)
    assert m["available"] is False and "reason" in m


def test_no_tool_call_ever_leaked():
    for line in srv.CALL_LOG.read_text().splitlines():
        assert json.loads(line)["leak"] is False


def test_team_recent_form_only_uses_games_final_before_t():
    f = call("team_recent_form", event_ticker=GAME, team="MIN")
    assert f["games"] == 10 and f["wins"] + f["losses"] == 10
    r = srv.results()
    t = srv.game(GAME).t
    for row in f["results_newest_first"]:   # every returned game must have ended (conservatively) by t
        match = r[(r.sport == "MLB") & ((r.home == "MIN") | (r.away == "MIN")) &
                  (r.start.dt.tz_convert("America/New_York").dt.date.astype(str) == row["date"])]
        assert (match.ends_by <= t).any()


def test_team_recent_form_excludes_a_game_still_in_progress(monkeypatch):
    """A game that started 1 hour before t may still be in progress at t: its result must not be returned."""
    g = srv.game(GAME)
    in_progress = pd.DataFrame([{"sport": "MLB", "home": "MIN", "away": "DET", "start": g.t - pd.Timedelta(hours=1),
                                 "p_model": 0.5, "model_home_win": 1.0, "ends_by": g.t + pd.Timedelta(hours=3)}])
    with_fake = pd.concat([srv.results(), in_progress], ignore_index=True)
    monkeypatch.setattr(srv, "results", lambda: with_fake)
    form = srv.team_recent_form(GAME, "MIN")
    assert "DET" not in [row["opponent"] for row in form["results_newest_first"]][:1]
    assert all(row["date"] < "2026-09-16" or row["opponent"] != "DET" for row in form["results_newest_first"])


def test_team_must_be_in_the_game():
    with pytest.raises(Exception):
        call("team_recent_form", event_ticker=GAME, team="LAD")
