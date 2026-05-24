from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any,Dict,Optional,List
import uuid

RUNS_PATH = Path('data/runs.jsonl')

RUN_INDEX: Dict[str,Dict[str,Any]] = {}

def new_run_id()->str:
    return "RUN-"+uuid.uuid4().hex[:10].upper()

def _now()->str:
    return datetime.utcnow().isoformat()+'Z'


# takes in the question and stores the question ,time along with metadata
def create_run(question: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    run_id = new_run_id()
    record = {
        "run_id": run_id,
        "question": question,
        "status": "queued",  # queued|running|done|error
        "created_at": _now(),
        "updated_at": _now(),
        "meta": meta,
        "result": None,
        "error": None,
    }
    RUN_INDEX[run_id] = record
    _append(record) # add run request logs to data/runs.jsonl
    return record

## updates the time for particular run id
def update_run(run_id:str, **updates:Any)->Dict[str,Any]:
    rec = RUN_INDEX.get(run_id)
    if not rec:
        raise KeyError(run_id)
    rec.update(updates)
    rec['updated_at'] = _now()
    RUN_INDEX[run_id] = rec
    _append(rec)
    return rec
## return the run id number
def get_run(run_id:str)->Optional[Dict[str,Any]]:
    return RUN_INDEX.get(run_id)
## returns complete list of runs
def list_runs(limit:int=25)->List[Dict[str,Any]]:
    return sorted(RUN_INDEX.values(),key = lambda r : r['updated_at'],reverse=True)[:limit]

## stores all runs to data/runs.jsonl file
def _append(record:Dict[str,Any]) ->None:
    RUNS_PATH.parent.mkdir(parents=True,exist_ok=True)
    with RUNS_PATH.open("a",encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def reload_from_disk() -> int:
    """
    Replay runs.jsonl into RUN_INDEX on startup.
    For each run_id, keeps only the latest record (last write wins).
    Returns the number of runs loaded.
    """
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
                    latest[run_id] = record  # last write wins
            except json.JSONDecodeError:
                continue
    RUN_INDEX.update(latest)
    return len(latest)



