from __future__ import annotations
from typing import Any,Dict
import asyncio

from app.agent.graph import build_graph

_graph = build_graph()

# this is the most important method it takes the question and calls entire graph loop 
async def run_agent_async(question:str)->Dict[str,Any]:
    out = await _graph.ainvoke({"question":question})
    return out

def run_agent_blocking(question:str)->Dict[str,Any]:
    return asyncio.run(run_agent_async(question))