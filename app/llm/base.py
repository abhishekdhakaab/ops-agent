"""Provider-neutral interface used by every graph node."""

from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMClient(ABC):
    """Minimal async contract for text and structured completions."""
    @abstractmethod
    async def complete_json(self, *, system: str, user: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON object that conforms to json_schema."""
        raise NotImplementedError

    @abstractmethod
    async def complete_text(self, *, system: str, user: str) -> str:
        """Return an unconstrained text completion."""
        raise NotImplementedError
