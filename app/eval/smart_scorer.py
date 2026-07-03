"""Score investigations with deterministic checks and an optional LLM judge."""

import json
import httpx
from typing import Dict, Any, List, Optional

from app.core.config import settings


# ── Structured scoring (no LLM needed) ────────────────────────────

def score_tools(tools_used: List[str], expected_all: List[str], expected_any: List[str]) -> Dict[str, Any]:
    """Measure required-tool coverage and acceptable-tool presence."""
    used = set(tools_used)
    all_ok = all(t in used for t in expected_all)
    any_ok = True if not expected_any else any(t in used for t in expected_any)
    all_hits = sum(1 for t in expected_all if t in used)
    all_total = max(1, len(expected_all))
    return {
        "all_ok": all_ok,
        "any_ok": any_ok,
        "all_score": all_hits / all_total,
        "any_score": 1.0 if any_ok else 0.0,
    }


def score_tool_diversity(tools_used: List[str]) -> Dict[str, Any]:
    """Measure repeated tool calls within one investigation."""
    unique = set(tools_used)
    total = len(tools_used) or 1
    return {
        "unique_tools": len(unique),
        "total_calls": total,
        "diversity_ratio": round(len(unique) / total, 3),
        "repeated_calls": total - len(unique),
    }


def score_efficiency(tools_used: List[str], expected_min: int, expected_max: int) -> Dict[str, Any]:
    """Reward call counts inside the scenario-specific target range."""
    n = len(tools_used)
    in_range = expected_min <= n <= expected_max
    if in_range:
        eff_score = 1.0
    elif n < expected_min:
        eff_score = max(0.0, n / expected_min)
    else:
        eff_score = max(0.0, 1.0 - (n - expected_max) / expected_max)
    return {
        "tools_count": n,
        "in_range": in_range,
        "efficiency_score": round(eff_score, 3),
    }


def score_structured(agent_output: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Compare action enums and the grounding verdict with ground truth."""
    action_payload = agent_output.get("action_payload", {})
    validation = agent_output.get("validation", {})
    
    # Adjacent severity gets partial credit; action remains an exact match.
    agent_severity = (action_payload.get("severity") or "").lower()
    expected_severity = (ground_truth.get("severity") or "").lower()
    severity_match = agent_severity == expected_severity
    severity_close = agent_severity in _adjacent_severity(expected_severity)
    
    agent_action = (action_payload.get("recommended_action") or "").lower()
    expected_action = (ground_truth.get("correct_action") or "").lower()
    action_match = agent_action == expected_action
    
    grounded = validation.get("grounded", False)
    
    severity_score = 1.0 if severity_match else (0.5 if severity_close else 0.0)
    action_score = 1.0 if action_match else 0.0
    
    composite = round(
        severity_score * 0.3 +
        action_score * 0.4 +
        float(grounded) * 0.3,
        3
    )
    
    return {
        "severity_match": severity_match,
        "severity_close": severity_close,
        "action_match": action_match,
        "grounded": grounded,
        "agent_severity": agent_severity,
        "expected_severity": expected_severity,
        "agent_action": agent_action,
        "expected_action": expected_action,
        "structured_score": composite,
    }


def _adjacent_severity(s: str) -> set:
    """Return the target severity plus one step away."""
    adj = {"low": {"low", "medium"}, "medium": {"low", "medium", "high"}, "high": {"medium", "high"}}
    return adj.get(s, {s})


# ── LLM-as-Judge scoring ──────────────────────────────────────────

JUDGE_SYSTEM = """You are an evaluation judge for an AI operations investigation agent.

You will be given:
- The original question
- The evidence the agent collected (metrics, logs, deployments)
- The agent's diagnosis/final answer
- The ground truth (what actually happened)

Score the agent's diagnosis on these criteria:

1. ROOT CAUSE IDENTIFICATION (0-3):
   0 = completely wrong or missed the root cause
   1 = vaguely related but wrong specific cause
   2 = correct general direction but missing key details
   3 = correctly identified the root cause

2. EVIDENCE USAGE (0-2):
   0 = ignored evidence or hallucinated claims
   1 = referenced some evidence but missed key data
   2 = properly grounded conclusions in evidence

3. HARMFUL CONCLUSIONS (0 or -2):
   0 = no harmful/wrong recommendations
   -2 = recommended rollback for a healthy service, or said "healthy" for a critical issue

Return ONLY a JSON object:
{"root_cause_score": 0-3, "evidence_score": 0-2, "harmful_penalty": 0 or -2, "reasoning": "one sentence explanation"}
"""


async def llm_judge_score(
    question: str,
    evidence: Dict[str, Any],
    final_answer: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Use the LLM to judge the quality of the agent's diagnosis."""
    
    user_prompt = f"""
QUESTION: {question}

EVIDENCE COLLECTED:
{json.dumps(evidence, indent=2, default=str)[:3000]}

AGENT'S DIAGNOSIS:
{final_answer[:2000]}

GROUND TRUTH:
- Root cause: {ground_truth.get('root_cause', 'unknown')}
- Expected severity: {ground_truth.get('severity', 'unknown')}
- Correct action: {ground_truth.get('correct_action', 'unknown')}
- Key findings that should be mentioned: {ground_truth.get('should_mention', [])}
- Wrong conclusions to avoid: {ground_truth.get('should_not_conclude', [])}

Score the agent's diagnosis.
"""
    
    schema = {
        "type": "object",
        "properties": {
            "root_cause_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "evidence_score": {"type": "integer", "minimum": 0, "maximum": 2},
            "harmful_penalty": {"type": "integer", "enum": [0, -2]},
            "reasoning": {"type": "string"},
        },
        "required": ["root_cause_score", "evidence_score", "harmful_penalty", "reasoning"],
        "additionalProperties": False,
    }
    
    try:
        base_url = settings.llm_base_url.rstrip("/")
        api_key = settings.llm_api_key
        model = settings.llm_model
        
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt + f"\n\nReturn ONLY valid JSON matching: {json.dumps(schema)}"},
            ],
        }
        
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        
        # Judges can still wrap valid JSON despite the strict output request.
        import re
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            scores = json.loads(m.group(0))
        else:
            scores = json.loads(text)
        
        # Clamp harmful penalties at zero after normalizing to the five-point scale.
        raw_total = scores["root_cause_score"] + scores["evidence_score"] + scores["harmful_penalty"]
        max_possible = 5  # 3 + 2 + 0
        normalized = max(0.0, round(raw_total / max_possible, 3))
        
        return {
            "root_cause_score": scores["root_cause_score"],
            "evidence_score": scores["evidence_score"],
            "harmful_penalty": scores["harmful_penalty"],
            "reasoning": scores.get("reasoning", ""),
            "llm_judge_score": normalized,
            "judge_error": None,
        }
        
    except Exception as e:
        return {
            "root_cause_score": 0,
            "evidence_score": 0,
            "harmful_penalty": 0,
            "reasoning": f"Judge failed: {str(e)[:100]}",
            "llm_judge_score": 0.0,
            "judge_error": str(e)[:200],
        }


# ── Combined scoring ──────────────────────────────────────────────

async def score_full(
    question: str,
    agent_result: Dict[str, Any],
    ground_truth: Dict[str, Any],
    tools_expected_all: List[str],
    tools_expected_any: List[str],
    expected_min_tools: int,
    expected_max_tools: int,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run all scoring methods and return combined scores."""
    
    tools_used = agent_result.get("tools_used", [])
    final_answer = agent_result.get("final_answer", "")
    evidence = agent_result.get("evidence", {})
    
    tool_score = score_tools(tools_used, tools_expected_all, tools_expected_any)
    diversity = score_tool_diversity(tools_used)
    efficiency = score_efficiency(tools_used, expected_min_tools, expected_max_tools)
    structured = score_structured(agent_result, ground_truth)
    
    # The judge is optional because it dominates evaluation latency.
    if use_llm_judge:
        judge = await llm_judge_score(question, evidence, final_answer, ground_truth)
    else:
        judge = {"llm_judge_score": None, "judge_error": "skipped"}
    
    components = [
        ("tool_selection", tool_score["all_score"], 0.15),
        ("diversity", diversity["diversity_ratio"], 0.10),
        ("efficiency", efficiency["efficiency_score"], 0.10),
        ("structured", structured["structured_score"], 0.30),
    ]
    
    if judge.get("llm_judge_score") is not None:
        components.append(("llm_judge", judge["llm_judge_score"], 0.35))
    else:
        # Keep total weights at one when deterministic-only mode is requested.
        components = [
            ("tool_selection", tool_score["all_score"], 0.20),
            ("diversity", diversity["diversity_ratio"], 0.10),
            ("efficiency", efficiency["efficiency_score"], 0.15),
            ("structured", structured["structured_score"], 0.55),
        ]
    
    combined = round(sum(score * weight for _, score, weight in components), 3)
    
    return {
        "tool_score": tool_score,
        "diversity": diversity,
        "efficiency": efficiency,
        "structured": structured,
        "llm_judge": judge,
        "combined_score": combined,
    }
