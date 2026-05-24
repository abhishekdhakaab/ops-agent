# this file contains the main agent graph flow 
from typing import TypedDict, List, Dict, Any, Literal, Optional
from pydantic import BaseModel, ValidationError
from langgraph.graph import StateGraph, END
from app.tools.registry import invoke_tool
import time
from app.llm.providers import get_llm
from app.agent.prompts import PLANNER_SYSTEM, FINAL_SYSTEM
from app.tools.contracts import MetricsIn, LogsIn, DeploymentsIn
from app.tools.impl import metrics_tool, logs_tool, deployments_tool
from app.core.config import settings
from app.agent.prompts import RETRIEVAL_SYSTEM, VALIDATOR_SYSTEM, ACTION_SYSTEM
from app.rag.embeddings import embed_texts
from app.rag.retrieve import retrieve
from pydantic import BaseModel, Field


Action = Literal["metrics", "logs", "deployments", "final"]
class ActionOut(BaseModel):
    severity: Literal["low", "medium", "high"]
    next_checks: List[str]
    recommended_action: str
    ticket: Optional[Dict[str, Any]] = None

class PlanOut(BaseModel):
    action: Action
    args: Dict[str, Any]
    rationale: str = "No rationale provided." ## because most of the small sized LLM struggle with rationale so setting it to default to prevent API stopping



class RetrievalPlan(BaseModel):
    queries: List[str] = Field(default_factory=list, alias="retrieval_queries")
    k: int = 2

    model_config = {
        "populate_by_name": True
    }


class ActionPayload(BaseModel):
    severity: Literal["low", "medium", "high"]
    next_checks: List[str]
    recommended_action: Literal["investigate", "mitigate", "rollback", "escalate"]
    create_ticket: bool
    ticket: Optional[Dict[str, Any]] = None  # will validate separately

class TicketDraft(BaseModel):
    title: str
    description: str
    severity: Literal["low", "medium", "high"]
    service: str
    evidence_refs: List[str] = []


class ValidationOut(BaseModel):
    grounded : bool
    missing_evidence : List[str] = []
    required_next_action : Literal["metrics",'logs','deployments','final'] = 'final'
    rationale : str
## single agent state shared across all nodes
class AgentState(TypedDict, total=False):
    question: str
    service: str
    reasoning: List[str]
    tools_used: List[str]
    evidence: Dict[str, Any]
    step: int
    tool_calls :List[str]
    stop_reason : str
    
    planner_output : Dict[str,Any]
    grounded_final : bool

    retrieved_context : List[Dict[str, Any]]
    investigation_plan : Dict[str, Any]
    draft_answer : str
    validation: Dict[str,Any]
    action_payload : Dict[str, Any]
    forced_next_action: Optional[str]

    final_answer: str


def _guess_service(question: str) -> str:
    q = question.lower()
    # All services from scenarios.json
    for s in [
        "order-svc", "checkout-svc", "api-gateway", "user-auth",
        "search-svc", "notification-svc", "payment-svc", "inventory-svc",
        "cdn-edge", "email-worker", "recommendation-svc", "session-store",
        "image-resize", "analytics-ingest", "auth-proxy", "billing-svc",
        "file-upload", "pricing-engine", "shipping-calc", "feed-svc",
        # Legacy service names for backward compatibility
        "service-x", "checkout", "api", "service-y", "payments",
    ]:
        if s in q:
            return s
    return "order-svc"  # default to a scenario that exists

async def intake_node(state: AgentState) -> AgentState:
    state["service"] = state.get("service") or _guess_service(state["question"])
    state["reasoning"] = state.get("reasoning") or []
    state["tools_used"] = state.get("tools_used") or []  ## tools model decided to use 
    state["evidence"] = state.get("evidence") or {}
    state["step"] = state.get("step") or 0
    state['tool_calls'] = state.get('tool_calls') or [] ## tools model executed ( sometime what model decide is not what is executed due to some error in processing so just to be double sure)
    return state

async def plan_node(state: AgentState) -> AgentState:
    llm = get_llm()
    forced = state.get('forced_next_action')
    if forced in {'metrics','logs','deployments'}: # if last call forces a particular tool use : then we direclty planOut with that tool call
        plan = PlanOut(
            action = forced, 
            args = {"service":state['service']},
            rationale = f"Forced by Validator Agent to gather missing evidence via {forced}."
        )
        state["reasoning"].append(plan.rationale)
        state["investigation_plan"] = plan.model_dump()
        state["planner_output"] = plan.model_dump()
        state["forced_next_action"] = None  # clear it so we don't loop forever
        return state ## we return the control to 'act' node which execute this forced action because here we manually decided to give the plan for forced action

    evidence_keys = list(state["evidence"].keys())
    # retrieve related content from 
    ctx_titles = [
    c.get("title") or c.get("ref") or c.get("source") or "context"
            for c in state.get("retrieved_context", [])
        ]


    user = f"""
    Question: {state['question']}
    Service: {state['service']}
    Current evidence keys: {list(state['evidence'].keys())}
    Retrieved context titles: {ctx_titles}

    Available tools:
    - metrics(service, time_range_minutes)
    - logs(service, contains, limit)
    - deployments(service)

    Decide the NEXT action.
    """

#     user = f"""
# Question: {state['question']}
# Service: {state['service']}
# Current evidence keys: {evidence_keys}

# Available tools:
# - metrics(service, time_range_minutes)
# - logs(service, contains, limit)
# - deployments(service)

# Decide the NEXT action.
# """
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["metrics", "logs", "deployments", "final"]},
            "args": {"type": "object"},
            "rationale": {"type": "string"}
        },
        "required": ["action", "args", "rationale"],
        "additionalProperties": False
    }
    raw = await llm.complete_json(system=PLANNER_SYSTEM, user=user, json_schema=schema)
    plan = PlanOut(**raw)
    ALLOWED_ACTIONS = {"metrics", "logs", "deployments", "final"}

    if plan.action not in ALLOWED_ACTIONS: # even if once we get a wrong tool call we stop to prevent from hallucination 
        plan.action = 'final'
        plan.args = {}
        plan.rationale = "Invalid tool requests, stopping to avoid unsafe action."
    state["reasoning"].append(plan.rationale)
    state['investigation_plan']= plan.model_dump()
    state["planner_output"] = plan.model_dump()
    return state

async def draft_answer_node(state: AgentState)->AgentState:
    llm = get_llm()
    user = f"""

            Question: {state['question']}
            Service: {state['service']}

            Retrieved context:
            {state.get('retrieved_context', [])}

            Evidence (JSON):
            {state['evidence']}

            Write a draft investigation answer. Use only evidence/context. Be concise.
            """
    draft = await llm.complete_text(system = FINAL_SYSTEM, user= user)
    state['draft_answer'] = draft
    return state

async def act_node(state: AgentState) -> AgentState:

    plan = PlanOut(**state["planner_output"])

    if plan.action == "final": ## if the tool==final then return without making any tool call 
        return state

    payload = dict(plan.args or {})
    ## if anything missing we will set defaul to service
    start = time.time()
    payload.setdefault("service",state['service'])
    try : 
        out = invoke_tool(plan.action, payload)
        success = not( isinstance(out,dict) and "error" in out) # if output anything other then dict that's a failure

    except Exception as e:
        out = {"error":str(e), "tool":plan.action, "payload":payload}
        success = False

    latency_ms = int((time.time() - start)*1000)
    state['tool_calls'].append({
        'tool':plan.action,
        'args':payload,
        'success':success,
        'latency_ms':latency_ms
    })
    state["tools_used"].append(plan.action)
    # Build a specific evidence key so repeated tool calls don't overwrite each other
    _action = plan.action
    _args = getattr(plan, "args", {}) or {}
    if _action == "metrics":
        _tr = _args.get("time_range_minutes", 60)
        _evidence_key = f"metrics_{_tr}min" if _tr != 60 else "metrics"
    elif _action == "logs":
        _contains = _args.get("contains")
        _evidence_key = f"logs_filter_{_contains}" if _contains else "logs"
    else:
        _evidence_key = _action
    state["evidence"][_evidence_key] = out
    state["step"] += 1

    return state


async def validation_agent_node(state :AgentState) -> AgentState:
    ## Validation Agent: checks if the draft answer is grounded in evidence ( means the output of act node) ; if not, requests more checks
    llm= get_llm()
    user = f"""
    Question: {state['question']}
    Draft answer:
    {state.get('draft_answer', '')}

    Evidence keys: {list(state.get('evidence', {}).keys())}
    Evidence JSON:
    {state.get('evidence', {})}

    Retrieved context:
    {state.get('retrieved_context', [])}

    Check grounding. If not grounded, specify what's missing and suggest the next tool.
    """
    schema = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "rationale": {"type": "string"},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "required_next_action": {"type": "string", "enum": ["metrics", "logs", "deployments", "final"]}
    },
    "required": ["grounded", "rationale", "required_next_action"],
    "additionalProperties": False
    }

    raw = await llm.complete_json(system = VALIDATOR_SYSTEM, user = user, json_schema = schema)
    val = ValidationOut(**raw)
    state['validation'] = val.model_dump()
    state["reasoning"].append(f"Validation Agent: grounded={val.grounded}, next={val.required_next_action}. {val.rationale}")
    if not val.grounded and val.required_next_action in {"metrics","logs","deployments"}:
        state["forced_next_action"] = val.required_next_action
    else :
        state["forced_next_action"] = None
    return state

async def action_agent_node(state: AgentState) -> AgentState:
    ## based on all the evidence and citation we decide which tool to call
    llm = get_llm()

   
    evidence_refs = list(state.get("evidence", {}).keys())
    ctx_refs = [c.get("id") for c in state.get("retrieved_context", [])]
    all_refs = [r for r in (evidence_refs + ctx_refs) if r]

    user = f"""
Question: {state['question']}
Service: {state['service']}

Draft answer:
{state.get('draft_answer', '')}

Validation:
{state.get('validation', {})}

Available evidence refs:
{all_refs}

Return JSON with:
- severity: low|medium|high
- next_checks: string[]
- recommended_action: investigate|mitigate|rollback|escalate
- create_ticket: boolean
- ticket (optional): {{title, description, severity, service, evidence_refs}}
"""

    schema = {
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "next_checks": {"type": "array", "items": {"type": "string"}},
            "recommended_action": {"type": "string", "enum": ["investigate", "mitigate", "rollback", "escalate"]},
            "create_ticket": {"type": "boolean"},
            "ticket": {
                "type": ["object", "null"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "service": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "description", "severity", "service", "evidence_refs"],
                "additionalProperties": False
            },
        },
        "required": ["severity", "next_checks", "recommended_action", "create_ticket"],
        "additionalProperties": False
    }

    raw = await llm.complete_json(system=ACTION_SYSTEM, user=user, json_schema=schema)
    # Normalize common model mistakes
    if raw.get("recommended_action") in {"metrics", "logs", "deployments", "final"}:
        raw["recommended_action"] = "investigate" # since we dont call tool recommended here and it's just a sign of 'more investigation'

    payload = ActionPayload(**raw)


    # If ticket is requested, validate and attach defaults
    ticket_id_info = None
    if payload.create_ticket:
        if payload.ticket is None:
            # Auto-generate safe ticket draft
            draft = TicketDraft(
                title=f"Investigate {state['service']} incident: {payload.severity} severity",
                description=state.get("draft_answer", "")[:1200],
                severity=payload.severity,
                service=state["service"],
                evidence_refs=all_refs[:10],
            )
        else:
            draft = TicketDraft(**payload.ticket)

        # Create ticket via MCP tool registry invocation (same path as agent tools)
        try:
            ticket_id_info = invoke_tool("ticket", draft.model_dump())
        except Exception as e:
            ticket_id_info = {"error": str(e)}

        state["tools_used"].append("ticket")
        state["evidence"]["ticket"] = ticket_id_info

    state["action_payload"] = payload.model_dump()
    state["reasoning"].append(
        f"Action Agent: recommended_action={payload.recommended_action}, create_ticket={payload.create_ticket}"
    )
    return state




def decide_next(state: AgentState) -> str:
    plan = PlanOut(**state["planner_output"])
    if plan.action == "final":
        return "finalize"
    if state.get("step", 0) >= settings.max_steps:
        return "finalize"
    return "plan"

async def finalize_node(state: AgentState) -> AgentState:
    ## we want to be honest about the fact that validator says ungrounded so will add that manually to your fiinalize output

    val = state.get("validation",{})
    ## reason for stopping
    if state.get('step',0) >= settings.max_steps: 
        state['stop_reason'] ="max_steps"
    elif val.get('grounded') is False:
        state['stop_reason'] = 'validator_exhausted'
    else:
        state['stop_reason'] = 'final'


    
    if val and  val.get("grounded") is False:
        missing = val.get("missing_evidence",[])
        note  = "Insufficient evidence to reach a confident root cause."
        if missing : 
            note += "Missing evidence: " + "; ".join(missing[:6])
        state['draft_answer'] = (note + "\n\n" + (state.get("draft_answer") or "")).strip()




    final = state.get("draft_answer", "").strip()
    action = state.get("action_payload", {})
    ticket = state.get("evidence", {}).get("ticket")

    if action:
        final += "\n\n---\nNext Actions (Structured):\n"
        final += f"- Severity: {action.get('severity')}\n"
        final += f"- Recommended: {action.get('recommended_action')}\n"
        for c in action.get("next_checks", [])[:6]:
            final += f"- Check: {c}\n"

    if ticket and isinstance(ticket, dict) and ticket.get("ticket_id"):
        final += f"\nTicket Created: {ticket['ticket_id']} (status={ticket.get('status','open')})\n"
    elif ticket and isinstance(ticket, dict) and ticket.get("error"):
        final += f"\nTicket Creation Failed: {ticket['error']}\n"

    
    state['grounded_final'] = bool(val.get('grounded',False))

    state["final_answer"] = final
    return state



async def retrieval_agent_node(state: AgentState)-> AgentState:
    llm = get_llm()
    user = f"""
    Question : {state['question']}
    Service : {state['service']}
    Propose retrieval queries for operational runbooks/notes.
    """
    schema = {
        "type":"object",
        "properties":{
            "queries": {"type":"array","items":{"type":"string"}},
            "k":{"type":"integer","minimum":1,"maximum":5}
        },
        "required":["queries"],
        "additionalProperties": False
    }
    raw = await llm.complete_json(system = RETRIEVAL_SYSTEM, user = user, json_schema = schema)
    plan = RetrievalPlan(**raw)
    retrieved_chunks: List[Dict[str, Any]] = []
    k = plan.k or 3

    for q in plan.queries[:3]:
        embs = await embed_texts([q])
        q_emb = embs[0] if embs else None

        hits = retrieve(query=q, query_emb=q_emb, k=k)

        # normalize to citation-ready format
        for h in hits:
            retrieved_chunks.append({
                "ref": f"{h['source']}#{h['chunk_id']}",
                "source": h["source"],
                "chunk_id": h["chunk_id"],
                "text": h["text"],
            })

    # de-dupe by ref
    seen = set()
    uniq = []
    for r in retrieved_chunks:
        if r["ref"] not in seen:
            uniq.append(r)
            seen.add(r["ref"])

    state["retrieved_context"] = uniq[:8]  # keep context bounded
    state["reasoning"].append(f"Retrieval Agent: retrieved {len(state['retrieved_context'])} chunks.")
    return state
    



def build_graph():
    g = StateGraph(AgentState)
    ## for now we have intake -> retrieval -> plan -> act-> loop to plan -> draft -> finalise)
    g.add_node("intake", intake_node)
    g.add_node('retrieve',retrieval_agent_node)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node('draft',draft_answer_node)
    g.add_node('validate',validation_agent_node)
    g.add_node('action',action_agent_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("intake")
    g.add_edge("intake", "retrieve")
    g.add_edge('retrieve','plan')
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", decide_next, {"plan": "plan", "finalize": "draft"})
    g.add_edge('draft','validate')

    # If not grounded and validator suggests a tool, loop back to plan; else proceed.

    def validate_route(state:AgentState)->str:
        ## if validaite_router says its not grounded and we still have step remaining then we will call Plan again  AND if steps exhausted then just call 'action' with 'insufficient evidence'
        val = state.get("validation",{})
        grounded = val.get('grounded',False)

        if grounded is True:
            return "action"
        
        if state.get('step',0) < settings.max_steps:
            return 'plan' ## think about it again 
        
        return 'action' ## mean finalise the action 

    g.add_conditional_edges("validate", validate_route, {"plan":"plan","action":"action"})
    g.add_edge("action","finalize")
    g.add_edge("finalize",END)
    return g.compile()






