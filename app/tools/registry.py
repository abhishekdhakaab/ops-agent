"""Register tools with schemas and enforce their Pydantic boundaries."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Type
from pydantic import BaseModel 

from .contracts import (
    MetricsIn, MetricsOut, LogsIn, LogsOut, DeploymentsIn, DeploymentsOut
)

from .impl import metrics_tool, logs_tool, deployments_tool
from .contracts import TicketIn, TicketOut
from .impl import ticket_tool

class ToolSpec(BaseModel):
    """Public description and JSON schemas exposed by the API."""
    name: str
    description : str
    input_schema : Dict[str, Any]
    output_schema : Dict[str, Any]

class ToolEntry(BaseModel):
    """Internal link between a specification, models, and handler."""
    spec : ToolSpec
    input_model : Type[BaseModel]
    output_model : Type[BaseModel]
    handler : Callable[[BaseModel],BaseModel]

def _schema_for(model:Type[BaseModel]) -> Dict[str,Any]:
    return model.model_json_schema()

TOOLS: Dict[str, ToolEntry] = {
    "metrics": ToolEntry(
        spec=ToolSpec(
            name="metrics",
            description="Fetch latency stats and detect spikes for a service over a time range.",
            input_schema=_schema_for(MetricsIn),
            output_schema=_schema_for(MetricsOut),
        ),
        input_model=MetricsIn,
        output_model=MetricsOut,
        handler=lambda inp: metrics_tool(inp),  # type: ignore[arg-type]
    ),
    "logs": ToolEntry(
        spec=ToolSpec(
            name="logs",
            description="Summarize error counts and common messages; return sample log lines.",
            input_schema=_schema_for(LogsIn),
            output_schema=_schema_for(LogsOut),
        ),
        input_model=LogsIn,
        output_model=LogsOut,
        handler=lambda inp: logs_tool(inp),  
    ),
    "deployments": ToolEntry(
        spec=ToolSpec(
            name="deployments",
            description="Return recent deployment metadata for a service.",
            input_schema=_schema_for(DeploymentsIn),
            output_schema=_schema_for(DeploymentsOut),
        ),
        input_model=DeploymentsIn,
        output_model=DeploymentsOut,
        handler=lambda inp: deployments_tool(inp),
    ),
    "ticket": ToolEntry(
        spec=ToolSpec(
            name="ticket",
            description="Create an incident ticket payload from investigation findings.",
            input_schema=_schema_for(TicketIn),
            output_schema=_schema_for(TicketOut),
        ),
        input_model=TicketIn,
        output_model=TicketOut,
        handler=lambda inp: ticket_tool(inp), 
    ),

}

def list_tool_specs()->List[ToolSpec]:
    """Return schemas for every tool exposed by the service."""
    return [entry.spec for entry in TOOLS.values()]

def invoke_tool(name:str, payload : Dict[str,Any])->Dict[str,Any]:
    """Validate a payload, call its handler, and normalize the response."""
    if name not in TOOLS:
        raise KeyError(f"Unkown tool : {name}")
    entry = TOOLS[name]
    inp = entry.input_model(**payload)
    out = entry.handler(inp)

    # Handlers may return either a model or a plain provider dictionary.
    if isinstance(out, BaseModel):
        return out.model_dump() 
    return entry.output_model(**out).model_dump()
