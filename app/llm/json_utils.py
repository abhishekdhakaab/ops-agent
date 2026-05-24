# this file contains code to extract json content from LLM response string
from __future__ import annotations
import json
import re
from typing import Any, Dict

class JSONParseError(Exception):
    pass

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Attempts to parse a JSON object even if the model adds extra text.
    Strategy:
      1) strip code fences
      2) try direct json.loads
      3) find first {...} block (greedy-ish) and parse
    """
    cleaned = _strip_code_fences(text)

    # direct
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first JSON object block
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
