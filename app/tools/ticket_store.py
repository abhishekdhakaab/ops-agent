from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import uuid

TICKETS_PATH = Path("data/tickets.jsonl")

def create_ticket(record: Dict[str, Any]) -> Dict[str, Any]:
    ## takesn in a ticket , gets a ticket number and store the complete information about the ticket in tickets.josnl and only return ticket number and it's status to be used later from file fetch
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
