"""
Tool simulation and evidence compression.

Replicates the tool behavior from app/tools/impl.py but self-contained.
Reads scenarios from data/scenarios.json and simulates tool responses.

IMPORTANT: Scenarios are indexed by scenario_id, NOT by service name.
Multiple scenarios share the same service (e.g., cdn-edge has 3 scenarios
with different root causes). The tool simulation must be scenario-aware
to return correct data for each.

Also defines compress_evidence() — the SHARED compression function used
in Phase 1 prompts, Phase 2 synthesis input, Phase 3 SFT formatting,
and future GRPO rollouts. One function, used everywhere.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from config import SCENARIOS_PATH



def load_scenarios_by_id() -> Dict[str, dict]:
    """Load scenario index keyed by scenario ID (unique)."""
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return {s["id"]: s for s in raw.get("scenarios", [])}

def load_scenarios_list() -> list:
    """Load scenarios as a list (preserves order)."""
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return raw.get("scenarios", [])

_SCENARIOS_BY_ID = load_scenarios_by_id()


DEFAULT_SCENARIO = {
    "metrics": {
        "has_latency_spike": False,
        "avg_latency_ms": 110.0,
        "p95_latency_ms": 240.0,
        "spike_start_minutes_ago": None,
    },
    "logs": {
        "error_count": 1,
        "error_messages": ["minor: GC pause 80ms"],
        "log_level": "WARN",
    },
    "deployments": {
        "has_recent_deploy": False,
        "deploy_minutes_ago": 2880,
        "deploy_summary": "Routine dependency update",
        "deploy_author": "ci-bot",
    },
}


def _get_scenario(scenario_id: str) -> dict:
    """Look up scenario by its unique ID."""
    # Unknown IDs get an empty fixture instead of leaking another scenario.
    return _SCENARIOS_BY_ID.get(scenario_id, {})



def run_tool(tool_name: str, args: dict, scenario_id: str = "") -> dict:
    """
    Run a simulated tool and return its output as a plain dict.
    
    `scenario_id` is required to get the correct scenario data
    (since multiple scenarios can share the same service name).
    
    This is the single entry point for tool simulation.
    Used by Phase 1 (raw runs) and Phase 3 (re-simulation).
    """
    service = args.get("service", "unknown")
    
    if tool_name == "metrics":
        return _metrics_tool(service, int(args.get("time_range_minutes", 60)), scenario_id)
    elif tool_name == "logs":
        # Planner output is untrusted here, so normalize filters before matching.
        contains = args.get("contains")
        if not isinstance(contains, str):  # catches list, bool, int, None
            contains = str(contains) if contains and not isinstance(contains, bool) else None
        if isinstance(contains, list):
            contains = contains[0] if contains else None
        return _logs_tool(service, contains, int(args.get("limit", 20)), scenario_id)
    elif tool_name == "deployments":
        return _deployments_tool(service, scenario_id)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _metrics_tool(service: str, time_range_minutes: int = 60, scenario_id: str = "") -> dict:
    sc = _get_scenario(scenario_id) if scenario_id else {}
    m = sc.get("metrics", DEFAULT_SCENARIO["metrics"])
    
    spike = m["has_latency_spike"]
    spike_start = m.get("spike_start_minutes_ago")
    
    # A query window that predates the spike should still look healthy.
    if spike and spike_start and time_range_minutes < spike_start:
        spike = False
    
    avg = m["avg_latency_ms"] if spike else m["avg_latency_ms"] * 0.42
    p95 = m["p95_latency_ms"] if spike else m["p95_latency_ms"] * 0.38
    
    # Stable seeds make training examples reproducible across pipeline phases.
    rng = random.Random(hash(service + "metrics"))
    if spike:
        normal = avg * 0.45
        series = [round(normal + (avg - normal) * (i / 10) + rng.uniform(-8, 8), 1)
                  for i in range(10)]
    else:
        series = [round(avg * (0.93 + 0.014 * i) + rng.uniform(-4, 4), 1)
                  for i in range(10)]
    
    if spike:
        notes = f"Latency spike detected: avg {avg:.0f}ms, p95 {p95:.0f}ms. Began ~{spike_start}min ago."
    else:
        notes = f"Latency stable: avg {avg:.0f}ms, p95 {p95:.0f}ms. No anomalies in window."
    
    return {
        "service": service,
        "time_range_minutes": time_range_minutes,
        "avg_latency_ms": round(avg, 1),
        "p95_latency_ms": round(p95, 1),
        "spike_detected": spike,
        "window_series_ms": series,
        "notes": notes,
    }


def _logs_tool(service: str, contains: Optional[str] = None, limit: int = 20, scenario_id: str = "") -> dict:
    sc = _get_scenario(scenario_id) if scenario_id else {}
    lg = sc.get("logs", DEFAULT_SCENARIO["logs"])
    
    error_count = lg["error_count"]
    common = lg["error_messages"]
    level = lg.get("log_level", "ERROR")
    
    # Samples stay deterministic so evidence does not drift between phases.
    rng = random.Random(hash(service + "logs"))
    base = datetime(2026, 4, 15, 12, 0, 0)
    samples = []
    # Six lines are enough context for a planner decision.
    for i in range(min(limit, 6)):
        ts = (base - timedelta(minutes=i * 2 + 1)).isoformat() + "Z"
        msg = common[i % len(common)]
        if contains and contains.lower() not in msg.lower():
            continue
        samples.append(f"{ts} {level} {service}: {msg}")
    
    return {
        "service": service,
        "error_count": error_count,
        "common_messages": common,
        "samples": samples,
    }


def _deployments_tool(service: str, scenario_id: str = "") -> dict:
    sc = _get_scenario(scenario_id) if scenario_id else {}
    dep = sc.get("deployments", DEFAULT_SCENARIO["deployments"])
    
    rng = random.Random(hash(service + "deploy"))
    base = datetime(2026, 4, 15, 12, 0, 0)
    
    recent = {
        "id": f"dep_{abs(hash(service + 'r')) % 100000:05d}",
        "time": (base - timedelta(minutes=dep["deploy_minutes_ago"])).isoformat() + "Z",
        "author": dep["deploy_author"],
        "summary": dep["deploy_summary"],
        "version": f"1.{rng.randint(10, 20)}.{rng.randint(0, 9)}",
        "minutes_ago": dep["deploy_minutes_ago"],
    }
    older = {
        "id": f"dep_{abs(hash(service + 'o')) % 100000:05d}",
        "time": (base - timedelta(hours=18)).isoformat() + "Z",
        "author": "ci-bot",
        "summary": "Routine dependency update",
        "version": f"1.{rng.randint(8, 12)}.0",
        "minutes_ago": 1080,
    }
    
    return {
        "service": service,
        "recent": [recent, older],
    }



def compress_evidence(tool_name: str, raw_output: dict) -> dict:
    """Keep only decision signals shared by training and inference prompts."""
    if tool_name == "metrics":
        return {
            "spike_detected": raw_output.get("spike_detected", False),
            "avg_latency_ms": raw_output.get("avg_latency_ms"),
            "p95_latency_ms": raw_output.get("p95_latency_ms"),
        }
    
    elif tool_name == "logs":
        return {
            "error_count": raw_output.get("error_count", 0),
            "common_messages": raw_output.get("common_messages", [])[:3],
        }
    
    elif tool_name == "deployments":
        recent_list = raw_output.get("recent", [])
        recent = recent_list[0] if recent_list else {}
        return {
            "has_recent_deploy": bool(recent) and recent.get("minutes_ago", 9999) < 120,
            "minutes_ago": recent.get("minutes_ago"),
            "summary": recent.get("summary"),
            "author": recent.get("author"),
        }
    
    return raw_output


def compress_all_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Compress an entire evidence dict (tool_name → raw_output)."""
    return {
        tool_name: compress_evidence(tool_name, output)
        for tool_name, output in evidence.items()
    }
