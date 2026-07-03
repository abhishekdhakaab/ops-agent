"""FastAPI surface for investigations, tools, tickets, and runbook ingestion."""

from fastapi import FastAPI
from app.core.types import RunRequest, RunResponse
from app.agent.graph import build_graph
from app.core.storage import log_trajectory
from typing import Any, Dict, List
from fastapi import HTTPException
from app.tools.registry import list_tool_specs, invoke_tool, ToolSpec
from app.tools.ticket_store import create_ticket
from app.core.metadata import run_metadata
from fastapi import HTTPException, BackgroundTasks
from app.core.run_store import create_run, update_run, get_run, list_runs
from app.core.runner import run_agent_blocking
from app.core.metadata import run_metadata
from pathlib import Path
import json
from pydantic import BaseModel
from app.rag.ingest import ingest_runbooks

class StartRunRequest(BaseModel):
    """Question submitted to the background-run endpoint."""
    question: str

app = FastAPI(title="Autonomous AI Ops Agent (Code-First)")
graph = build_graph()

from app.core.run_store import reload_from_disk as _reload_runs

@app.on_event("startup")
async def _on_startup():
    """Restore the latest state of persisted asynchronous runs."""
    count = _reload_runs()
    print(f"[startup] Reloaded {count} runs from disk into RUN_INDEX")

@app.get("/health")
def health():
    """Report process health without invoking a model."""
    return {"status": "ok"}

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    """Run an investigation synchronously and log its training trajectory."""
    state = {"question": req.question}
    out = await graph.ainvoke(state)
    meta = run_metadata()
    

    log_trajectory({**meta,"question":req.question,"service":out.get('service'),
        'reasoning':out.get('reasoning'),
        'tool_used':out.get('tools_used',[]),
        "log_schema_version": 2,
        "tool_calls": out.get("tool_calls", []),
        "stop_reason": out.get("stop_reason"),
        "grounded_final": out.get("grounded_final"),
        "validation": out.get("validation", {}),
        "evidence": out.get("evidence", {}),
        "retrieved_context": out.get("retrieved_context", []),

    })
    return RunResponse(
        reasoning=out.get("reasoning", []),
        tools_used=out.get("tools_used", []),
        evidence=out.get("evidence", {}),
        final_answer=out.get("final_answer", ""),
    )

@app.get('/tools',response_model = List[ToolSpec])
def tools():
    """Expose tool contracts for inspection and integration tests."""
    return list_tool_specs()

@app.post('/tools/{tool_name}')
def tool_call(tool_name : str, payload:Dict[str, Any]):
    """Invoke one validated tool directly."""
    try: 
        return {'tool':tool_name,'output':invoke_tool(tool_name,payload)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Tool call failed : {e}')
    


@app.post("/tickets")
def create_ticket_endpoint(payload: dict):
    """Create a ticket directly; graph-driven calls still use the registry."""
    try:
        out = create_ticket(payload)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post('/runs')
def start_run(req: StartRunRequest, background_tasks:BackgroundTasks):
    """Queue an investigation and return its polling identifier immediately."""
    meta = run_metadata()
    run = create_run(req.question,meta=meta)

    def job(run_id:str, question:str):
        try:
            update_run(run_id,status='running')
            out = run_agent_blocking(question)
            update_run(run_id,status='done',result=out)

        except Exception as e:
            update_run(run_id,status='error',result=str(e))

    # FastAPI owns the task lifecycle after the queued record is persisted.
    background_tasks.add_task(job,run['run_id'],req.question)
    return {"run_id":run['run_id'],'status':run['status'],'created_at':run['created_at']}

@app.get('/runs')
def runs(limit:int=25):
    """List recent asynchronous investigations."""
    return {"runs":list_runs(limit=limit)}



@app.get('/runs/{run_id}')
def run_detail(run_id:str):
    """Return the latest state of one asynchronous investigation."""
    r = get_run(run_id)
    if not r:
        raise HTTPException(status_code=404,detail='run not found')

    return r


@app.get('/tickets')
def list_tickets(limit:int =50):
    """List persisted tickets in reverse chronological order."""
    path = Path('data/tickets.jsonl')
    if not path.exists():
        return {"tickets":[]}
    items = []

    with path.open('r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    items = sorted(items,key = lambda x : x.get('created_at'),reverse=True)[:limit]
    return {'tickets':items}


@app.post('/runbooks/ingest')
async def ingest():
    """Rebuild the runbook index used for retrieval."""
    return await ingest_runbooks()
