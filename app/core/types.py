from pydantic import BaseModel
from typing import Any, Dict, List

class RunRequest(BaseModel):
    question: str

class RunResponse(BaseModel):
    reasoning: List[str]
    tools_used: List[str]
    evidence: Dict[str, Any]
    final_answer: str
