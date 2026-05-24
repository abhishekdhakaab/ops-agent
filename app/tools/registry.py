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
    name: str
    description : str
    input_schema : Dict[str, Any]
    output_schema : Dict[str, Any]

class ToolEntry(BaseModel):
    spec : ToolSpec
    input_model : Type[BaseModel]
    output_model : Type[BaseModel]
    handler : Callable[[BaseModel],BaseModel] ## basically a function that takes a subclass of BaseModel( not it' instance because we will use this variable to create objects later) and returns a output class which is also subclass of BaseModel

def _schema_for(model:Type[BaseModel]) -> Dict[str,Any]:
    ## general method to get schema for any of the pydantic class
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
        handler=lambda inp: deployments_tool(inp),  # right now it just takes the input and get the output from the mode, later we can use this to do some preprocessing on our output before sending back to the user 
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
    return [entry.spec for entry in TOOLS.values()]

def invoke_tool(name:str, payload : Dict[str,Any])->Dict[str,Any]:
    if name not in TOOLS:
        raise KeyError(f"Unkown tool : {name}")
    entry = TOOLS[name]
    inp = entry.input_model(**payload)
    out = entry.handler(inp)

    if isinstance(out, BaseModel): ## in our metrics_tool we have defined that output of metrices_tool is MetricsOut() which is subclass of BaseModel
        return out.model_dump() 
    return entry.output_model(**out).model_dump()
