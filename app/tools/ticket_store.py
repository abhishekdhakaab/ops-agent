"""Append-only local incident ticket storage for the demo service."""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import uuid

TICKETS_PATH = Path("data/tickets.jsonl")

def create_ticket(record: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a ticket while returning only its public identifier and status."""
    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ticket_id = "TCK-" + uuid.uuid4().hex[:10].upper()
    out = {
        "ticket_id": ticket_id,
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z",
        **record
    }
    with TICKETS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return {"ticket_id": ticket_id, "status": "open"}
