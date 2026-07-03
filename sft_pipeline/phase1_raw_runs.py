"""
Phase 1: Generate raw investigation runs.

For each scenario, run RUNS_PER_SCENARIO independent investigations
using a large model via Ollama. Each run is a multi-turn loop:

    model picks tool → we simulate it → feed output back → repeat

The model never sees ground truth. It reasons from the question
and the evidence it accumulates, just like the small model will
have to at inference time.

Saves results incrementally to phase1_raw_runs.jsonl for resumability.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import (
    RUNS_PER_SCENARIO,
    MAX_STEPS_PER_RUN,
    TEMPERATURE_PHASE1,
    NUM_PREDICT_PHASE1,
    RAW_RUNS_PATH,
    OUTPUT_DIR,
)
from tools import run_tool, compress_evidence, load_scenarios_list
from prompts import (
    PLANNER_SYSTEM,
    build_planner_state,
    build_planner_user_message,
    generate_question,
)
from ollama_client import call_ollama_json


VALID_ACTIONS = {"metrics", "logs", "deployments", "final"}


def run_single_investigation(
    scenario: dict,
    question: str,
    model: Optional[str] = None,
) -> dict:
    """
    Run one investigation attempt for a scenario.
    
    Returns a dict with:
      - steps: list of {action, args, rationale, tool_output_compressed}
      - completed: whether the model said "final" (vs hitting max_steps)
      - num_tool_calls: count of actual tool calls (excluding final)
    """
    service = scenario["service"]
    scenario_id = scenario["id"]
    evidence_raw = {}       # tool_name → raw tool output
    evidence_compressed = {}  # tool_name → compressed output (what model sees)
    tools_called = []       # ordered list of tools called
    steps = []
    completed = False
    
    for step_idx in range(MAX_STEPS_PER_RUN + 1):  # +1 to allow final after max tools
        state = build_planner_state(
            question=question,
            service=service,
            step=step_idx,
            tools_called=list(tools_called),
            evidence=dict(evidence_compressed),
        )
        
        user_msg = build_planner_user_message(state)
        
        response = call_ollama_json(
            system=PLANNER_SYSTEM,
            user=user_msg,
            model=model,
            temperature=TEMPERATURE_PHASE1,
            num_predict=NUM_PREDICT_PHASE1,
        )
        
        if response is None:
            steps.append({
                "action": "error",
                "args": {},
                "rationale": "LLM call failed",
                "tool_output_compressed": None,
            })
            break
        
        action = response.get("action", "").lower().strip()
        
        action_fixes = {"metric": "metrics", "log": "logs", "deployment": "deployments",
                        "deploy": "deployments", "finish": "final", "done": "final",
                        "stop": "final", "end": "final"}
        action = action_fixes.get(action, action)
        
        args = response.get("args", {})
        if not isinstance(args, dict):
            args = {}
        rationale = response.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale) if rationale else ""
        args = response.get("args", {})
        if not isinstance(args, dict):    # ← ADD THIS
            args = {}                      # ← ADD THIS
        rationale = response.get("rationale", "")
        if not isinstance(rationale, str):  # ← ADD THIS
            rationale = str(rationale) if rationale else ""  # ← ADD THIS

        
        
        if action not in VALID_ACTIONS:
            # Invalid tool — try to salvage by mapping to closest valid action,
            steps.append({
                "action": action,
                "args": args,
                "rationale": rationale,
                "tool_output_compressed": None,
                "error": f"Invalid action: {action}",
            })
            break
        
        if action == "final":
            steps.append({
                "action": "final",
                "args": {},
                "rationale": rationale,
                "tool_output_compressed": None,
            })
            completed = True
            break
        
        if isinstance(args, dict):
            args.setdefault("service", service)
            if isinstance(args.get("service"), str):
                args["service"] = args["service"].strip()
        else:
            args = {"service": service}
        
        raw_output = run_tool(action, args, scenario_id=scenario_id)
        compressed = compress_evidence(action, raw_output)
        
        steps.append({
            "action": action,
            "args": args,
            "rationale": rationale,
            "tool_output_compressed": compressed,
        })
        
        # Specific keys preserve repeated calls with different windows or filters;
        # the step list still records every decision for later synthesis.
        evidence_key = action
        tr = int(args.get("time_range_minutes", 60)) if action == "metrics" else 0
        if action == "metrics" and tr != 60:
            evidence_key = f"metrics_{tr}min"
        contains_val = args.get("contains")
        if action == "logs" and contains_val and isinstance(contains_val, str):
            evidence_key = f"logs_filter_{contains_val}"
        
        evidence_raw[evidence_key] = raw_output
        evidence_compressed[evidence_key] = compressed
        tools_called.append(action)
    
    auto_final = False
    if not completed and steps and steps[-1].get("action") != "error":
        steps.append({
            "action": "final",
            "args": {},
            "rationale": "Investigation complete, sufficient evidence gathered.",
            "tool_output_compressed": None,
        })
        auto_final = True

    return {
        "steps": steps,
        "completed": completed,
        "auto_final": auto_final,
        "num_tool_calls": sum(1 for s in steps if s["action"] not in ("final", "error")),
    }


def load_existing_runs() -> Dict[str, dict]:
    """Load already-generated runs for resumability."""
    existing = {}
    if RAW_RUNS_PATH.exists():
        with RAW_RUNS_PATH.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing[entry["scenario_id"]] = entry
    return existing


def run_phase1(model: Optional[str] = None, max_scenarios: int = 0) -> str:
    """
    Generate raw investigation runs for all scenarios.
    
    Args:
        model: Ollama model name (overrides config)
        max_scenarios: If > 0, only process this many scenarios (for testing)
    
    Returns:
        Path to the output file.
    """
    scenarios = load_scenarios_list()
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = load_existing_runs()
    
    total = len(scenarios)
    generated = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Raw Investigation Runs")
    print(f"  Scenarios: {total}")
    print(f"  Runs per scenario: {RUNS_PER_SCENARIO}")
    print(f"  Already completed: {len(existing)}")
    print(f"  Model: {model}")
    print(f"{'='*60}\n")
    
    for i, scenario in enumerate(scenarios):
        sc_id = scenario["id"]
        service = scenario["service"]
        
        if sc_id in existing:
            existing_runs = existing[sc_id].get("runs", [])
            if len(existing_runs) >= RUNS_PER_SCENARIO:
                skipped += 1
                continue
        
        print(f"  [{i+1}/{total}] {sc_id} ({service}, {scenario['root_cause']})")
        
        runs = []
        for run_idx in range(RUNS_PER_SCENARIO):
            question = generate_question(scenario, variant=run_idx)
            
            t0 = time.time()
            result = run_single_investigation(scenario, question, model=model)
            elapsed = time.time() - t0
            
            result["question"] = question
            result["run_index"] = run_idx
            result["elapsed_s"] = round(elapsed, 1)
            runs.append(result)
            
            status = "✓" if result["completed"] else "○"
            tools = [s["action"] for s in result["steps"] if s["action"] != "error"]
            print(f"    Run {run_idx+1}: {status} {tools} ({elapsed:.1f}s)")
        
        entry = {
            "scenario_id": sc_id,
            "service": service,
            "root_cause": scenario["root_cause"],
            "severity": scenario["severity"],
            "ground_truth": scenario.get("ground_truth", {}),
            "runs": runs,
        }
        
        with RAW_RUNS_PATH.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        generated += 1
    
    print(f"\n  Phase 1 complete:")
    print(f"    Generated: {generated}")
    print(f"    Skipped (resume): {skipped}")
    print(f"    Output: {RAW_RUNS_PATH}")
    
    return str(RAW_RUNS_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-scenarios", type=int, default=0)
    args = parser.parse_args()
    run_phase1(model=args.model, max_scenarios=args.max_scenarios)
