"""Small sync/async entry points around the compiled agent graph."""

from __future__ import annotations
from typing import Any,Dict
import asyncio

from app.agent.graph import build_graph

_graph = build_graph()

async def run_agent_async(question:str)->Dict[str,Any]:
    """Run one investigation through the graph asynchronously."""
    out = await _graph.ainvoke({"question":question})
    return out

def run_agent_blocking(question:str)->Dict[str,Any]:
    """Run one investigation from synchronous callers."""
    return asyncio.run(run_agent_async(question))
