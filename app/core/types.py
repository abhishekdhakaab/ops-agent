"""Request and response contracts for synchronous investigations."""

from pydantic import BaseModel
from typing import Any, Dict, List

class RunRequest(BaseModel):
    """User question submitted to the investigation graph."""
    question: str

class RunResponse(BaseModel):
    """Evidence-backed investigation returned by the API."""
    reasoning: List[str]
    tools_used: List[str]
    evidence: Dict[str, Any]
    final_answer: str
