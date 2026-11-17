"""LangGraph research agent for the Kalshi sports benchmark (Milestone 4).

Graph: research -> forecast -> END
  research: Claude calls the point-in-time MCP tools (agent/mcp_server.py) in a loop, capped at MAX_TOOL_CALLS.
  forecast: one final call with structured output (JSON schema) -> p_home, confidence, key_factors, reasoning.
Claude is called through the official Anthropic SDK; MCP tools through the official MCP client over stdio.
Every run records tokens, dollars, seconds, the tools used, and whether the server logged a leak.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("AGENT_EFFORT", "medium")          # Opus 5.5 default is medium; set explicitly
MAX_TOOL_CALLS = 8
BETAS = ["server-side-fallback-2026-07-01"]           # refusal fallback: rerun a declined request on another model
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20), "claude-sonnet-5-5": (2.00, 10.00, 0.10),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}     # $/1M tokens: input, output, cache read

SYSTEM = """You forecast whether the HOME team wins a game, as of the decision time t (60 minutes before the \
scheduled start). Your tools only return information that was available at t.

How to work:
- Start with get_game and market_price. The prediction-market price is a strong baseline: markets are hard to beat.
- Then check model_win_prob (my own statistical model) and team_recent_form for both teams.
- On past games (development data), my model has been slightly less accurate than the market on average, but it \
often disagrees with it. Treat both as evidence; move away from the market only for a concrete reason you found.
- Recent form over 10 games is noisy. Don't overreact to streaks.
- Be calibrated: your probability should mean what it says. When evidence is thin, stay close to the market.
- If a tool says something is unavailable, work with what you have."""

FORECAST_PROMPT = ("Give your final forecast now: the probability that the HOME team wins, your confidence "
                   "(low/medium/high), the key factors, and short reasoning.")

FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "p_home": {"type": "number", "description": "Probability the home team wins, between 0 and 1"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["p_home", "confidence", "key_factors", "reasoning"],
    "additionalProperties": False,
}


class AgentState(TypedDict):
    event_ticker: str
    messages: list
    tool_calls: int
    trace: list
    usage: dict
    models_used: list
    forecast: dict | None


def _add_usage(usage: dict, resp) -> dict:
    u = resp.usage
    out = dict(usage)
    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        out[k] = out.get(k, 0) + (getattr(u, k, 0) or 0)
    out["requests"] = out.get("requests", 0) + 1
    return out


def cost_usd(usage: dict, model: str = MODEL) -> float:
    pin, pout, pcache = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (usage.get("input_tokens", 0) * pin + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
            + usage.get("cache_read_input_tokens", 0) * pcache + usage.get("output_tokens", 0) * pout) / 1e6


def build_graph(client: anthropic.AsyncAnthropic, session: ClientSession, tools: list[dict]):
    common = dict(model=MODEL, max_tokens=16000, system=SYSTEM, tools=tools, betas=BETAS, fallbacks="default",
                  cache_control={"type": "ephemeral"})

    async def research(state: AgentState) -> dict:
        msgs, calls, trace = list(state["messages"]), state["tool_calls"], list(state["trace"])
        usage, models = dict(state["usage"]), list(state["models_used"])
        while True:
            resp = await client.beta.messages.create(messages=msgs, output_config={"effort": EFFORT}, **common)
            usage, models = _add_usage(usage, resp), models + [resp.model]
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"refused: {resp.stop_details}")
            msgs.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                break
            results = []
            for b in uses:
                if calls >= MAX_TOOL_CALLS:
                    content, is_err = "Tool budget reached. Give your forecast with what you have.", True
                else:
                    r = await session.call_tool(b.name, b.input)
                    calls += 1
                    content = "\n".join(c.text for c in r.content if getattr(c, "type", "") == "text")
                    is_err = bool(r.isError)
                    trace.append({"tool": b.name, "input": b.input, "error": is_err})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_err})
            msgs.append({"role": "user", "content": results})
            if calls >= MAX_TOOL_CALLS and all(r["is_error"] for r in results):
                break
        return {"messages": msgs, "tool_calls": calls, "trace": trace, "usage": usage, "models_used": models}

    async def forecast(state: AgentState) -> dict:
        msgs = state["messages"] + [{"role": "user", "content": FORECAST_PROMPT}]
        resp = await client.beta.messages.create(
            messages=msgs, tool_choice={"type": "none"},
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": FORECAST_SCHEMA}}, **common)
        usage = _add_usage(state["usage"], resp)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused: {resp.stop_details}")
        fc = json.loads(next(b.text for b in resp.content if b.type == "text"))
        fc["p_home"] = min(max(float(fc["p_home"]), 0.01), 0.99)
        return {"forecast": fc, "usage": usage, "models_used": state["models_used"] + [resp.model]}

    g = StateGraph(AgentState)
    g.add_node("research", research)
    g.add_node("forecast", forecast)
    g.add_edge(START, "research")
    g.add_edge("research", "forecast")
    g.add_edge("forecast", END)
    return g.compile()


def load_env_file(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines into the environment (the project .env is git-ignored and chmod 600)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


async def run_games(event_tickers: list[str]) -> list[dict]:
    """Run the agent on each game with one MCP server process and one API client."""
    load_env_file()
    client = anthropic.AsyncAnthropic()
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent.mcp_server"],
                                   env={**os.environ, "PYTHONPATH": str(ROOT)}, cwd=str(ROOT))
    call_log = ROOT / "data" / "agent_tool_calls.jsonl"
    out = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                     for t in listed.tools]
            graph = build_graph(client, session, tools)
            for ev in event_tickers:
                log_start = len(call_log.read_text().splitlines()) if call_log.exists() else 0
                t0 = time.time()
                row: dict[str, Any] = {"event_ticker": ev, "model": MODEL, "effort": EFFORT}
                try:
                    st = await graph.ainvoke({"event_ticker": ev, "tool_calls": 0, "trace": [], "usage": {},
                                              "models_used": [], "forecast": None,
                                              "messages": [{"role": "user", "content":
                                                            f"Forecast game {ev}. Use the tools, then stop."}]})
                    row.update(forecast=st["forecast"], tool_calls=st["tool_calls"], trace=st["trace"],
                               usage=st["usage"], models_used=sorted(set(st["models_used"])),
                               cost_usd=round(cost_usd(st["usage"]), 5))
                except Exception as e:  # one bad game must not kill a whole eval run
                    row.update(error=f"{type(e).__name__}: {e}")
                row["seconds"] = round(time.time() - t0, 1)
                new = call_log.read_text().splitlines()[log_start:] if call_log.exists() else []
                row["leaks"] = sum(json.loads(l)["leak"] for l in new if json.loads(l)["event_ticker"] == ev)
                out.append(row)
                print(json.dumps({k: row.get(k) for k in ("event_ticker", "seconds", "cost_usd", "tool_calls",
                                                          "leaks", "error")} |
                                 {"p_home": (row.get("forecast") or {}).get("p_home")}), flush=True)
    return out
