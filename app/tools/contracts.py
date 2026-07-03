"""Typed request and response contracts for simulated operations tools."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

ToolName = Literal["metrics", "logs", "deployments"]

class MetricsIn(BaseModel):
    """Parameters for a latency-window query."""
    service: str
    time_range_minutes: int = 60

class MetricsOut(BaseModel):
    """Latency summary returned to the investigation graph."""
    service: str
    time_range_minutes: int
    avg_latency_ms: float
    p95_latency_ms: float
    spike_detected: bool
    window_series_ms: List[float] = Field(default_factory=list)
    notes: str

class LogsIn(BaseModel):
    """Parameters for filtered log sampling."""
    service: str
    contains: Optional[str] = None
    limit: int = 20

class LogsOut(BaseModel):
    """Aggregated errors plus a small set of representative lines."""
    service: str
    error_count: int
    common_messages: List[str]
    samples: List[str]

class DeploymentsIn(BaseModel):
    """Service whose recent deployment history should be inspected."""
    service: str

class DeploymentsOut(BaseModel):
    """Recent deployment records kept flexible for provider metadata."""
    service: str
    recent: List[Dict[str, Any]]



class TicketIn(BaseModel):
    """Validated incident ticket draft produced by the action node."""
    title :str
    description: str
    severity : str= "medium"
    service :str
    evidence_refs : List[str] = Field(default_factory = list)

class TicketOut(BaseModel):
    """Stable identifiers returned after ticket persistence."""
    ticket_id : str
    status : str
