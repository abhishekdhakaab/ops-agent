"""Canonical planner prompts shared by data generation and evaluation."""

import json
from typing import Dict, Any, List


# This text is repeated in every example, so additions have a real token cost.

PLANNER_SYSTEM = (
    "You are an SRE investigation planner. Choose the NEXT best action.\n"
    "\n"
    "Available tools:\n"
    "- metrics(service, time_range_minutes): latency stats, spike detection\n"
    "- logs(service, contains, limit): error counts, error messages\n"
    "- deployments(service): recent deploy history\n"
    "- final: stop investigating, you have enough evidence\n"
    "\n"
    "Rules:\n"
    "- Return ONLY valid JSON: {\"action\", \"args\", \"rationale\"}\n"
    "- Do NOT repeat a tool if its output is already in evidence\n"
    "- Call 2-3 tools then choose final. NEVER call more than 3 tools.\n"
    "- Rationale: 1-2 sentences referencing specific evidence values\n"
    "\n"
    "EXAMPLE CONVERSATION:\n"
    "Step 0 (no evidence): {\"action\": \"metrics\", \"args\": {\"service\": \"order-svc\"}, "
    "\"rationale\": \"Check latency first to assess severity.\"}\n"
    "Step 1 (has metrics): {\"action\": \"logs\", \"args\": {\"service\": \"order-svc\"}, "
    "\"rationale\": \"Spike detected at 420ms avg. Checking error logs for cause.\"}\n"
    "Step 2 (has metrics+logs): {\"action\": \"final\", \"args\": {}, "
    "\"rationale\": \"Metrics show spike, logs show OOM errors. Sufficient to diagnose.\"}\n"
)


def build_planner_state(
    question: str,
    service: str,
    step: int,
    tools_called: List[str],
    evidence: Dict[str, Any],       # already compressed
) -> dict:
    """Build planner state from evidence already compressed by the tool layer."""
    return {
        "question": question,
        "service": service,
        "step": step,
        "tools_called": tools_called,
        "evidence": evidence,
    }


def build_planner_user_message(state: dict) -> str:
    """Format the one planner-state representation used across all phases."""
    evidence = state.get("evidence", {})
    if evidence:
        evidence_str = json.dumps(evidence, indent=2, default=str)
    else:
        evidence_str = "  (none yet)"
    
    tools_called = state.get("tools_called", [])
    if tools_called:
        tools_str = json.dumps(tools_called)
    else:
        tools_str = "  (none yet)"
    
    return (
        f"Question: {state['question']}\n"
        f"Service: {state['service']}\n"
        f"Step: {state['step']}\n"
        f"Tools called so far: {tools_str}\n"
        f"Evidence collected:\n{evidence_str}\n"
        f"\n"
        f"Choose the NEXT action. Return JSON: "
        f"{{\"action\": \"...\", \"args\": {{...}}, \"rationale\": \"...\"}}"
    )


def format_sft_example(state: dict, action_json: dict) -> dict:
    """Create one TRL chat example from planner state and its next action."""
    return {
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": build_planner_user_message(state)},
            {"role": "assistant", "content": json.dumps(action_json, ensure_ascii=False)},
        ]
    }


# Hinting and neutral variants keep wording from becoming the label.
_QUESTION_TEMPLATES = {
    "deploy_regression": [
        "Investigate {service}: we're seeing issues that started recently.",
        "{service} has been degraded. Could this be related to a recent deployment?",
        "Something changed in {service} and it's not working right. Please investigate.",
        "We're getting alerts from {service}. Seems to have started in the last hour.",
    ],
    "resource_exhaustion": [
        "{service} is running slow and we're getting timeout alerts.",
        "High latency detected in {service}. What's causing the performance issues?",
        "Investigate {service}: response times are above normal.",
        "Customers are complaining about {service} being slow. What's going on?",
    ],
    "upstream_dependency": [
        "{service} is throwing errors intermittently. Please investigate.",
        "We're seeing sporadic failures in {service}. Not sure what's causing them.",
        "Investigate {service}: error rate has gone up.",
        "{service} seems unstable. Some requests fail while others succeed.",
    ],
    "healthy": [
        "Can you check if {service} is healthy? Someone reported a possible issue.",
        "Quick health check on {service} — anything unusual?",
        "We got a single alert from {service}. Is it a real problem or a false alarm?",
        "Investigate {service} — might be nothing but want to be sure.",
    ],
    "config_change": [
        "{service} is behaving differently. Can you investigate?",
        "Something seems off with {service} after a recent change.",
        "Investigate {service}: unexpected behavior detected.",
        "{service} config may have changed. Check if there are any issues.",
    ],
    "infrastructure": [
        "{service} is down or severely degraded. Investigate immediately.",
        "Major issues with {service}. Might be infrastructure-related.",
        "Investigate {service}: possible infrastructure problem.",
        "{service} is unreachable. What's the root cause?",
    ],
}


def generate_question(scenario: dict, variant: int = 0) -> str:
    """Generate a deterministic wording variant for one incident scenario."""
    root_cause = scenario.get("root_cause", "resource_exhaustion")
    service = scenario["service"]
    templates = _QUESTION_TEMPLATES.get(root_cause, _QUESTION_TEMPLATES["resource_exhaustion"])
    template = templates[variant % len(templates)]
    return template.format(service=service)
