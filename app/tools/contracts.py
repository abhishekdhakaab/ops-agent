# this file only contains data semantics of all the tools call (eg. input and output semantic of metrics tool)
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

ToolName = Literal["metrics", "logs", "deployments"]

class MetricsIn(BaseModel):
    service: str
    time_range_minutes: int = 60

class MetricsOut(BaseModel):
    service: str
    time_range_minutes: int
    avg_latency_ms: float
    p95_latency_ms: float
    spike_detected: bool
    window_series_ms: List[float] = Field(default_factory=list)
    notes: str

class LogsIn(BaseModel):
    service: str
    contains: Optional[str] = None
    limit: int = 20

class LogsOut(BaseModel):
    service: str
    error_count: int
    common_messages: List[str]
    samples: List[str]

class DeploymentsIn(BaseModel):
    service: str

class DeploymentsOut(BaseModel):
    service: str
    recent: List[Dict[str, Any]]  # keep flexible early



class TicketIn(BaseModel):
    title :str
    description: str
    severity : str= "medium"
    service :str
    evidence_refs : List[str] = Field(default_factory = list)

class TicketOut(BaseModel):
    ticket_id : str
    status : str