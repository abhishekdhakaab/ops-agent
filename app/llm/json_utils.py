"""Recover a JSON object from imperfect model output."""

from __future__ import annotations
import json
import re
from typing import Any, Dict

class JSONParseError(Exception):
    """Raised when no valid object can be recovered from a completion."""
    pass

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()

def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse direct JSON first, then recover the first object-shaped block."""
    cleaned = _strip_code_fences(text)

    # The fast path preserves normal JSON without applying regex heuristics.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Small models often wrap the object in a sentence or Markdown fence.
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        raise JSONParseError("No JSON object found in model output.")

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        raise JSONParseError(f"Failed to parse extracted JSON: {e}") from e

    raise JSONParseError("Parsed JSON was not an object.")
