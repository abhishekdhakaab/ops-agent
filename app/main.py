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
    question: str

app = FastAPI(title="Autonomous AI Ops Agent (Code-First)")
graph = build_graph()

from app.core.run_store import reload_from_disk as _reload_runs

@app.on_event("startup")
async def _on_startup():
    count = _reload_runs()
    print(f"[startup] Reloaded {count} runs from disk into RUN_INDEX")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    # takes in the requests a string and invoke graph -> receives the result-> log the result 
    state = {"question": req.question}
    out = await graph.ainvoke(state)
    meta = run_metadata() ## with each request and it's response we also store meta data (LLM user, API version ,prompt version etc)
    

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
    #return RunResponse data model 
    return RunResponse(
        reasoning=out.get("reasoning", []),
        tools_used=out.get("tools_used", []),
        evidence=out.get("evidence", {}),
        final_answer=out.get("final_answer", ""),
    )

@app.get('/tools',response_model = List[ToolSpec]) # here response model will chcek if the outptu of the below defined fucntion is of pydantic type Tool[ToolSpec]
def tools():
    return list_tool_specs()

@app.post('/tools/{tool_name}')
def tool_call(tool_name : str, payload:Dict[str, Any]):
    try: 
        ## this function takes tool name as input and returns the tool output : exact working : take the tool name -> get all its properties -> call the tools mentioned in the properties 
        return {'tool':tool_name,'output':invoke_tool(tool_name,payload)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Tool call failed : {e}')
    


@app.post("/tickets")
def create_ticket_endpoint(payload: dict):
    # minimal direct endpoint (tool path is still preferred for agent usage)
    try:
        ## we take the payload and create a ticket in jsonl along with ticket number + payload + time , this function returns ticket number and output signal
        out = create_ticket(payload)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post('/runs')
def start_run(req: StartRunRequest, background_tasks:BackgroundTasks):
    meta = run_metadata() # get the model name and version detail
    run = create_run(req.question,meta=meta) # req : input query string, create_run will create a run (dictionary) and also stores the run request into logs to data/runs.jsonl

    def job(run_id:str, question:str):
        try:
            update_run(run_id,status='running')
            out = run_agent_blocking(question)
            update_run(run_id,status='done',result=out)

        except Exception as e:
            update_run(run_id,status='error',result=str(e))

    background_tasks.add_task(job,run['run_id'],req.question) # runs FastAPi's add background task 
    return {"run_id":run['run_id'],'status':run['status'],'created_at':run['created_at']} ## so the moment user request for a run a run id is created now the process is succcess or not depends but user will get a run id and will have to check serparately for the process progress
    ## basically request is added to background and user is notified

@app.get('/runs')
def runs(limit:int=25):
    return {"runs":list_runs(limit=limit)} # returns all runs in the run dictionary along with last updated time



@app.get('/runs/{run_id}')
def run_detail(run_id:str):
    r = get_run(run_id) # when creating a run we store the run detailed in a dictioinary and here we retrieve the same
    if not r:
        raise HTTPException(status_code=404,detail='run not found')

    return r


@app.get('/tickets')
def list_tickets(limit:int =50):
    # returns list of all the tickets 
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
    return await ingest_runbooks() # will create embedding for all the runbooks and store them in data/rag_index.jsonl for faster and quicker access