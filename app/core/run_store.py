"""Append-only persistence for asynchronous investigation runs."""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any,Dict,Optional,List
import uuid

RUNS_PATH = Path('data/runs.jsonl')

RUN_INDEX: Dict[str,Dict[str,Any]] = {}

def new_run_id()->str:
    """Create a compact public identifier for an investigation run."""
    return "RUN-"+uuid.uuid4().hex[:10].upper()

def _now()->str:
    return datetime.utcnow().isoformat()+'Z'


def create_run(question: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Create and persist a queued run record."""
    run_id = new_run_id()
    record = {
        "run_id": run_id,
        "question": question,
        "status": "queued",  # State transitions are persisted as new JSONL rows.
        "created_at": _now(),
        "updated_at": _now(),
        "meta": meta,
        "result": None,
        "error": None,
    }
    RUN_INDEX[run_id] = record
    _append(record)
    return record

def update_run(run_id:str, **updates:Any)->Dict[str,Any]:
    """Apply a state transition and append its new snapshot."""
    rec = RUN_INDEX.get(run_id)
    if not rec:
        raise KeyError(run_id)
    rec.update(updates)
    rec['updated_at'] = _now()
    RUN_INDEX[run_id] = rec
    _append(rec)
    return rec
def get_run(run_id:str)->Optional[Dict[str,Any]]:
    """Return the latest in-memory snapshot for a run."""
    return RUN_INDEX.get(run_id)
def list_runs(limit:int=25)->List[Dict[str,Any]]:
    """List the most recently updated runs first."""
    return sorted(RUN_INDEX.values(),key = lambda r : r['updated_at'],reverse=True)[:limit]

def _append(record:Dict[str,Any]) ->None:
    RUNS_PATH.parent.mkdir(parents=True,exist_ok=True)
    # One row per update keeps writes simple and replayable.
    with RUNS_PATH.open("a",encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def reload_from_disk() -> int:
    """Replay the append-only log, keeping the last snapshot per run."""
    if not RUNS_PATH.exists():
        return 0
    latest: Dict[str, Dict[str, Any]] = {}
    with RUNS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                run_id = record.get("run_id")
                if run_id:
                    latest[run_id] = record  # Later state transitions supersede earlier rows.
            except json.JSONDecodeError:
                # A partial final write should not prevent older runs from loading.
                continue
    RUN_INDEX.update(latest)
    return len(latest)

