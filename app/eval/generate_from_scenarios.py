"""
Generate eval questions and training data from scenarios.json.

Usage:
    python generate_from_scenarios.py

Outputs:
    app/eval/questions.json    — eval questions with ground truth
    data/trl_planner_sft.jsonl — SFT training data for planner
    data/trl_planner_grpo.jsonl — GRPO training data
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = REPO_ROOT / "data" / "scenarios.json"
QUESTIONS_OUT = REPO_ROOT / "app" / "eval" / "questions.json"
SFT_OUT = REPO_ROOT / "data" / "trl_planner_sft.jsonl"
GRPO_OUT = REPO_ROOT / "data" / "trl_planner_grpo.jsonl"

# ── Question templates per root cause ──────────────────────────────
# Multiple phrasings so the agent can't pattern-match on wording

TEMPLATES = {
    "deploy_regression": [
        "Why is {service} slow right now?",
        "Did the latest deployment cause issues in {service}?",
        "{service} started having problems recently. What happened?",
        "Investigate {service} — users are reporting errors after a release.",
        "Is there a deployment regression in {service}?",
    ],
    "upstream_dependency": [
        "We are seeing errors in {service}. What is causing this?",
        "{service} is throwing connection errors. What's wrong?",
        "Is {service} down? Users can't complete their actions.",
        "Investigate error spike in {service}.",
    ],
    "resource_exhaustion": [
        "{service} is running slow. Can you investigate?",
        "What's happening with {service}? It seems degraded.",
        "{service} is timing out for some users. What should we check?",
        "Performance issue in {service} — help me triage.",
    ],
    "config_change": [
        "Something changed in {service}. What's going on?",
        "{service} is behaving differently since this morning. Investigate.",
        "Are there any issues with {service} right now?",
        "Check if {service} is working correctly.",
    ],
    "infrastructure": [
        "{service} has errors but no recent deploys. What's happening?",
        "Investigate {service} — seems like an infrastructure issue.",
        "{service} is unstable. Could it be a platform problem?",
    ],
    "healthy": [
        "Is {service} having any issues right now?",
        "Quick health check on {service} please.",
        "Any problems with {service}?",
        "Give me a status update on {service}.",
    ],
}


def expected_tools_for(scenario: dict) -> tuple:
    """Determine expected tools based on what evidence the scenario has."""
    root = scenario["root_cause"]
    has_deploy = scenario["deployments"]["has_recent_deploy"]
    has_spike = scenario["metrics"]["has_latency_spike"]
    high_errors = scenario["logs"]["error_count"] > 20

    # expected_all: tools that MUST be used
    # expected_any: at least one of these
    if root == "deploy_regression":
        return (["deployments"], ["metrics", "logs", "deployments"])
    elif root == "upstream_dependency":
        return (["logs"], ["logs", "metrics"])
    elif root == "resource_exhaustion":
        return (["logs"], ["logs", "metrics"])
    elif root == "config_change":
        return (["logs"], ["logs", "deployments"])
    elif root == "infrastructure":
        return (["logs"], ["logs", "metrics"])
    elif root == "healthy":
        return ([], ["metrics", "logs"])
    else:
        return ([], ["metrics", "logs"])


def tool_count_range(scenario: dict) -> tuple:
    """Expected min/max tool calls."""
    root = scenario["root_cause"]
    if root == "healthy":
        return (1, 3)
    elif root == "deploy_regression":
        return (2, 5)
    else:
        return (1, 4)


def generate_questions(scenarios: list) -> list:
    """Generate eval questions from scenarios."""
    questions = []

    for sc in scenarios:
        service = sc["service"]
        root = sc["root_cause"]
        gt = sc["ground_truth"]

        # Pick 2 random question phrasings per scenario
        templates = TEMPLATES.get(root, TEMPLATES["healthy"])
        chosen = random.sample(templates, min(2, len(templates)))

        exp_all, exp_any = expected_tools_for(sc)
        min_t, max_t = tool_count_range(sc)

        for i, tmpl in enumerate(chosen):
            question = tmpl.format(service=service)
            qid = f"{sc['id']}_q{i+1}"

            questions.append({
                "id": qid,
                "question": question,
                "service": service,
                "scenario_id": sc["id"],
                "category": root,
                "expected_tools_all": exp_all,
                "expected_tools_any": exp_any,
                "expected_min_tools": min_t,
                "expected_max_tools": max_t,
                "ground_truth": {
                    "root_cause": root,
                    "must_mention": gt["must_mention"],
                    "should_mention": gt["should_mention"],
                    "should_not_conclude": gt["should_not_conclude"],
                },
            })

    return questions


# ── SFT Training Data ─────────────────────────────────────────────

PLANNER_SYSTEM = (
    "You are an ops investigation planner. Choose the NEXT best tool action.\n"
    "Return ONLY valid JSON: {action, args, rationale}.\n"
    "Allowed actions: metrics, logs, deployments, final.\n"
    "Never invent evidence. Be evidence-driven."
)

def first_tool_for(scenario: dict) -> str:
    """What tool should the planner pick FIRST for this scenario?"""
    root = scenario["root_cause"]
    has_spike = scenario["metrics"]["has_latency_spike"]
    high_errors = scenario["logs"]["error_count"] > 30

    if root == "deploy_regression":
        return "metrics" if has_spike else "logs"
    elif root in ("upstream_dependency", "resource_exhaustion", "infrastructure"):
        return "logs" if high_errors else "metrics"
    elif root == "config_change":
        return "logs"
    elif root == "healthy":
        return "metrics"
    return "metrics"


def second_tool_for(scenario: dict, first: str) -> str:
    """What should the second tool call be?"""
    root = scenario["root_cause"]
    has_deploy = scenario["deployments"]["has_recent_deploy"]

    if first == "metrics" and root in ("deploy_regression",):
        return "deployments" if has_deploy else "logs"
    elif first == "logs" and root in ("deploy_regression",):
        return "deployments"
    elif first == "metrics":
        return "logs"
    elif first == "logs":
        return "metrics"
    return "deployments" if has_deploy else "final"


def generate_sft(scenarios: list) -> list:
    """Generate SFT training rows: (system, user, assistant) triplets."""
    rows = []

    for sc in scenarios:
        service = sc["service"]

        # Step 1: no evidence yet — pick first tool
        first = first_tool_for(sc)
        user_1 = json.dumps({
            "question": f"Investigate {service}",
            "service": service,
            "evidence_keys": [],
            "available_tools": ["metrics", "logs", "deployments", "final"],
            "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
        })
        assistant_1 = json.dumps({
            "action": first,
            "args": {"service": service},
            "rationale": f"Start by checking {first} to gather initial evidence.",
        })
        rows.append({
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_1},
                {"role": "assistant", "content": assistant_1},
            ]
        })

        # Step 2: have first tool's evidence — pick second tool
        second = second_tool_for(sc, first)
        user_2 = json.dumps({
            "question": f"Investigate {service}",
            "service": service,
            "evidence_keys": [first],
            "available_tools": ["metrics", "logs", "deployments", "final"],
            "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
        })
        assistant_2 = json.dumps({
            "action": second,
            "args": {"service": service} if second != "final" else {},
            "rationale": f"Already have {first} data. Now check {second} to {'correlate findings' if second != 'final' else 'summarize'}.",
        })
        rows.append({
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_2},
                {"role": "assistant", "content": assistant_2},
            ]
        })

        # Step 3: have two tools — decide to finalize or get one more
        if second != "final":
            remaining = {"metrics", "logs", "deployments"} - {first, second}
            third = "final"
            # Deployment regressions need correlation across all three evidence types.
            if sc["root_cause"] == "deploy_regression" and remaining:
                third = list(remaining)[0]

            user_3 = json.dumps({
                "question": f"Investigate {service}",
                "service": service,
                "evidence_keys": [first, second],
                "available_tools": ["metrics", "logs", "deployments", "final"],
                "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
            })
            assistant_3 = json.dumps({
                "action": third,
                "args": {"service": service} if third != "final" else {},
                "rationale": f"Have {first} and {second} data. {'Gathering final evidence.' if third != 'final' else 'Sufficient evidence collected.'}",
            })
            rows.append({
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": user_3},
                    {"role": "assistant", "content": assistant_3},
                ]
            })

    return rows


def generate_grpo(scenarios: list) -> list:
    """Generate GRPO rows: prompt + expected_action."""
    rows = []
    for sc in scenarios:
        service = sc["service"]
        first = first_tool_for(sc)
        user = json.dumps({
            "question": f"Investigate {service}",
            "service": service,
            "evidence_keys": [],
            "available_tools": ["metrics", "logs", "deployments", "final"],
            "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
        })
        rows.append({
            "prompt": f"{PLANNER_SYSTEM}\n\nUSER:\n{user}\n\nASSISTANT:\n",
            "expected_action": first,
            "scenario_id": sc["id"],
            "root_cause": sc["root_cause"],
        })
    return rows


# ── Main ───────────────────────────────────────────────────────────

def main():
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios = raw["scenarios"]
    print(f"Loaded {len(scenarios)} scenarios from {SCENARIOS_PATH}")

    # Generate eval questions
    random.seed(42)  # reproducible
    questions = generate_questions(scenarios)
    QUESTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(questions)} eval questions to {QUESTIONS_OUT}")

    # Generate SFT training data
    sft_rows = generate_sft(scenarios)
    SFT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SFT_OUT.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sft_rows)} SFT training rows to {SFT_OUT}")

    # Generate GRPO training data
    grpo_rows = generate_grpo(scenarios)
    with GRPO_OUT.open("w", encoding="utf-8") as f:
        for r in grpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(grpo_rows)} GRPO training rows to {GRPO_OUT}")

    # Summary
    by_cause = {}
    for sc in scenarios:
        by_cause.setdefault(sc["root_cause"], []).append(sc["service"])
    print("\nScenario breakdown:")
    for cause, services in sorted(by_cause.items()):
        print(f"  {cause}: {len(services)} scenarios ({', '.join(services)})")


if __name__ == "__main__":
    main()
