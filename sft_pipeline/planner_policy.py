"""
Canonical planner policy definitions shared across training, eval, and scoring.

Import this module instead of defining best_first_tools() locally in each file.
"""

import re
from typing import Set


VALID_ACTIONS = {"metrics", "logs", "deployments", "final"}

ACTION_ALIASES = {
    "metric": "metrics",
    "log": "logs",
    "deploy": "deployments",
    "deployment": "deployments",
    "done": "final",
    "finish": "final",
    "complete": "final",
    "end": "final",
}


def normalize_action(action: str) -> str:
    """
    Normalize a model-produced action string to a valid action.
    Handles common typos and aliases the model produces.
    Returns the normalized string, or the original if no match.
    """
    if not isinstance(action, str):
        return ""
    action = action.lower().strip()
    if action in VALID_ACTIONS:
        return action
    return ACTION_ALIASES.get(action, action)


def best_first_tools(scenario: dict) -> Set[str]:
    """
    Returns the set of acceptable first tools for a given scenario.

    Based on Phase 2 synthesis results across 100 scenarios.
    This is the canonical version — use this everywhere.

    Mapping rationale:
      deploy_regression   → deployments (question usually mentions recent change)
      config_change       → deployments OR logs (either is reasonable)
      upstream_dependency → logs (question usually mentions errors)
      resource_exhaustion → metrics OR logs (both acceptable starting points)
      infrastructure      → logs (infrastructure issues surface in logs first)
      healthy             → metrics OR logs (either reasonable for health check)
    """
    root = scenario.get("root_cause", "")
    if root == "deploy_regression":
        return {"deployments"}
    elif root == "config_change":
        return {"deployments", "logs", "metrics"}
    elif root == "upstream_dependency":
        return {"logs"}
    elif root == "resource_exhaustion":
        return {"metrics", "logs"}
    elif root == "infrastructure":
        return {"logs"}
    elif root == "healthy":
        return {"metrics", "logs"}
    return {"metrics", "logs"}


def question_hint_first_tool(question: str) -> Set[str]:
    """
    Returns preferred first tools based on question wording alone.
    Use this as a secondary signal when question wording is unambiguous.

    Only returns a non-empty set when the wording is strongly indicative.
    """
    q = question.lower()
    deploy_words = ["deploy", "release", "shipped", "changed", "rollout", "push"]
    error_words = ["error", "5xx", "fail", "crash", "exception", "throwing", "broken"]
    perf_words = ["slow", "latency", "timeout", "performance", "p95", "response time"]

    hints: Set[str] = set()
    if any(w in q for w in deploy_words):
        hints.add("deployments")
    if any(w in q for w in error_words):
        hints.add("logs")
    if any(w in q for w in perf_words):
        hints.add("metrics")
    return hints
