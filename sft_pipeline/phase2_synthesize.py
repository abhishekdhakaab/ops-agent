"""
Phase 2: Synthesize clean trajectories.

For each scenario, takes the 3 raw investigation runs from Phase 1
and asks the large model to produce ONE optimal trajectory.

The synthesizer sees complete information — every tool call, every
tool output, every rationale from all 3 runs. It picks the minimal
set of useful tool calls and writes new rationales that are:
  - Grounded only in evidence available at that step
  - Simple enough for a small model to learn
  - 1-2 sentences max

Output: one clean trajectory per scenario.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import (
    TEMPERATURE_PHASE2,
    NUM_PREDICT_PHASE2,
    RAW_RUNS_PATH,
    SYNTHESIZED_PATH,
    OUTPUT_DIR,
    MAX_RETRIES,
)
from ollama_client import call_ollama_json


VALID_ACTIONS = {"metrics", "logs", "deployments", "final"}



SYNTHESIZER_SYSTEM = """You are an expert SRE reviewing multiple investigation attempts for the same incident.
Your job: produce ONE optimal investigation trajectory.

CRITICAL RULES:
- The FIRST tool must be chosen based on the question context, NOT always metrics.
  - If the question mentions "deploy", "release", "changed" → start with deployments
  - If the question mentions "errors", "5xx", "throwing errors" → start with logs
  - If the question mentions "slow", "latency", "timeout" → start with metrics
  - If unclear, look at which first tool worked best across the attempts
- You do NOT need to call all 3 tools. If 2 tools provide sufficient evidence, stop there.
- A trajectory of 2 tools + final is BETTER than 3 tools + final when 2 suffice.

- IMPORTANT: Attempts marked "auto-final injected" hit the step limit and were not naturally completed.
  Prefer trajectories from naturally completed runs. Only use auto-final runs if no natural completion exists.

You must output valid JSON and nothing else."""


def _build_synthesizer_user_prompt(scenario_entry: dict) -> str:
    """Build the user prompt for the synthesizer, including all 3 runs."""
    
    sc_id = scenario_entry["scenario_id"]
    service = scenario_entry["service"]
    runs = scenario_entry["runs"]
    
    # Use the first run's question (they're all for the same scenario)
    question = runs[0].get("question", f"Investigate {service}")
    
    runs_text = ""
    for idx, run in enumerate(runs):
        runs_text += f"\n--- ATTEMPT {idx + 1} ---\n"
        runs_text += f"Question used: {run.get('question', question)}\n"
        auto = run.get("auto_final", False)
        if run["completed"] and not auto:
            completion_label = "Yes (model reached final naturally)"
        elif auto:
            completion_label = "No (hit step limit — auto-final injected, lower quality)"
        else:
            completion_label = "No (hit step limit)"
        runs_text += f"Completed: {completion_label}\n"
        
        for step in run["steps"]:
            action = step["action"]
            args = step.get("args", {})
            rationale = step.get("rationale", "")
            output = step.get("tool_output_compressed")
            
            runs_text += f"\n  Step: {action}"
            if args and action != "final":
                args_display = {k: v for k, v in args.items() if k != "service"}
                if args_display:
                    runs_text += f"({json.dumps(args_display)})"
            runs_text += f"\n  Rationale: {rationale[:150]}"
            if output:
                runs_text += f"\n  Output: {json.dumps(output, default=str)}"
            runs_text += "\n"
    
    return f"""INCIDENT:
Question: "{question}"  
Service: {service}

Here are {len(runs)} investigation attempts. Each shows tool calls, their outputs, and reasoning.
{runs_text}

YOUR TASK:
Analyze all attempts and produce ONE optimal investigation trajectory.

RULES:
1. Choose the FIRST tool based on the question wording, NOT defaulting to metrics
   - "errors", "5xx", "throwing errors", "failing" → start with logs
   - "deploy", "release", "changed", "recent change" → start with deployments  
   - "slow", "latency", "timeout", "performance" → start with metrics
   - If ambiguous, pick the first tool that was most informative across the 3 attempts
2. Do NOT always call all 3 tools. If 2 tools give enough evidence, go to final
   - Example: if logs show OOM and deployments confirm a recent release, you don't need metrics
   - Example: if metrics are normal and logs show minor GC pauses, you don't need deployments
3. If the same tool was called with same args and same output across runs, keep only one
4. CRITICAL: Each rationale must reference ONLY evidence from PREVIOUS steps
5. Keep each rationale to 1-2 simple sentences. Reference specific values
6. End with action "final" when sufficient evidence exists

Return JSON:
{{
  "reasoning": "1-2 sentences explaining why you chose this sequence and first tool",
  "trajectory": [
    {{"action": "tool_name", "args": {{"service": "{service}", ...}}, "rationale": "..."}},
    ...
    {{"action": "final", "args": {{}}, "rationale": "..."}}
  ]
}}"""



def _validate_trajectory(trajectory: list, service: str) -> tuple:
    """
    Validate a synthesized trajectory.
    Returns (is_valid, error_message).
    """
    if not trajectory:
        return False, "Empty trajectory"
    
    if not isinstance(trajectory, list):
        return False, "Trajectory is not a list"
    
    for i, step in enumerate(trajectory):
        if not isinstance(step, dict):
            return False, f"Step {i} is not a dict"
        
        action = step.get("action", "").lower().strip()
        if action not in VALID_ACTIONS:
            return False, f"Step {i} has invalid action: {action}"
        
        if not step.get("rationale"):
            return False, f"Step {i} has no rationale"
    
    if trajectory[-1].get("action") != "final":
        return False, "Trajectory doesn't end with 'final'"
    
    # Check for consecutive duplicates (same tool + same args)
    for i in range(1, len(trajectory)):
        if (trajectory[i].get("action") == trajectory[i-1].get("action") and
            trajectory[i].get("action") != "final"):
            args_i = {k: v for k, v in trajectory[i].get("args", {}).items() if k != "service"}
            args_prev = {k: v for k, v in trajectory[i-1].get("args", {}).items() if k != "service"}
            if args_i == args_prev:
                return False, f"Steps {i-1} and {i} are duplicate {trajectory[i]['action']} calls"
    
    tool_calls = [s for s in trajectory if s.get("action") != "final"]
    if len(tool_calls) > 4:
        return False, f"Too many tool calls ({len(tool_calls)}), max 4"
    
    if len(tool_calls) == 0:
        return False, "No tool calls before final"

    # Check for future-evidence leakage in rationales
    EVIDENCE_KEYWORDS = {
        "logs":        ["log", "error", "oom", "exception", "crash", "traceback",
                        "timeout", "warn", "stack", "message", "stderr"],
        "metrics":     ["latency", "p95", "spike", "avg", "ms", "throughput",
                        "request rate", "rps", "error rate"],
        "deployments": ["deploy", "release", "rollout", "version", "commit",
                        "author", "diff", "change", "minutes ago"],
    }

    tools_called_so_far = set()
    for i, step in enumerate(trajectory):
        action = step.get("action", "")
        rationale = step.get("rationale", "").lower()

        # Check if rationale mentions evidence from a tool not yet called
        for tool, keywords in EVIDENCE_KEYWORDS.items():
            if tool not in tools_called_so_far:
                for kw in keywords:
                    if kw in rationale:
                        return False, (
                            f"Step {i} rationale mentions '{kw}' "
                            f"but '{tool}' has not been called yet (leakage)"
                        )

        if action != "final":
            tools_called_so_far.add(action)

    return True, "OK"



def synthesize_one(scenario_entry: dict, model: Optional[str] = None) -> Optional[dict]:
    """
    Synthesize one clean trajectory from a scenario's raw runs.
    
    Returns a dict with:
      - trajectory: list of {action, args, rationale}
      - synthesis_reasoning: why this sequence was chosen
      - question: the canonical question for this scenario
    
    Returns None if synthesis fails after retries.
    """
    user_prompt = _build_synthesizer_user_prompt(scenario_entry)
    
    for attempt in range(MAX_RETRIES):
        response = call_ollama_json(
            system=SYNTHESIZER_SYSTEM,
            user=user_prompt,
            model=model,
            temperature=TEMPERATURE_PHASE2,
            num_predict=NUM_PREDICT_PHASE2,
        )
        
        if response is None:
            continue
        
        # Handle case where model returns a bare list instead of {trajectory: [...]}
        if isinstance(response, list):                        # ← ADD
            trajectory = response                              # ← ADD
            reasoning = ""                                     # ← ADD
        else:                                                  # ← ADD
            trajectory = response.get("trajectory", [])
            reasoning = response.get("reasoning", "")
        
        service = scenario_entry["service"]
        for step in trajectory:
            step["action"] = step.get("action", "").lower().strip()
            if not isinstance(step.get("args"), dict):       # ← ADD
                step["args"] = {}                              # ← ADD
            if not isinstance(step.get("rationale"), str):    # ← ADD
                step["rationale"] = str(step.get("rationale", ""))  # ← ADD
            if step["action"] not in ("final", "error"):
                args = step.get("args", {})
                if isinstance(args, dict):
                    args.setdefault("service", service)
                else:
                    step["args"] = {"service": service}
        
        is_valid, error = _validate_trajectory(trajectory, service)
        if is_valid:
            return {
                "trajectory": trajectory,
                "synthesis_reasoning": reasoning,
                "question": scenario_entry["runs"][0].get("question", f"Investigate {service}"),
            }
        else:
            if attempt < MAX_RETRIES - 1:
                print(f"      Validation failed: {error}. Retrying...")
    
    return None


def load_raw_runs() -> List[dict]:
    """Load Phase 1 results."""
    entries = []
    if not RAW_RUNS_PATH.exists():
        print(f"  ERROR: Phase 1 output not found at {RAW_RUNS_PATH}")
        print(f"  Run Phase 1 first.")
        sys.exit(1)
    
    with RAW_RUNS_PATH.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_existing_syntheses() -> Dict[str, dict]:
    """Load already-synthesized trajectories for resumability."""
    existing = {}
    if SYNTHESIZED_PATH.exists():
        with SYNTHESIZED_PATH.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing[entry["scenario_id"]] = entry
    return existing


def run_phase2(model: Optional[str] = None, max_scenarios: int = 0) -> str:
    """
    Synthesize clean trajectories for all scenarios.
    
    Reads from Phase 1 output, produces Phase 2 output.
    """
    raw_entries = load_raw_runs()
    if max_scenarios > 0:
        raw_entries = raw_entries[:max_scenarios]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = load_existing_syntheses()
    
    total = len(raw_entries)
    synthesized = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Trajectory Synthesis")
    print(f"  Scenarios to synthesize: {total}")
    print(f"  Already completed: {len(existing)}")
    print(f"  Model: {model}")
    print(f"{'='*60}\n")
    
    for i, entry in enumerate(raw_entries):
        sc_id = entry["scenario_id"]
        service = entry["service"]
        
        if sc_id in existing:
            skipped += 1
            continue
        
        print(f"  [{i+1}/{total}] {sc_id} ({service})")
        
        t0 = time.time()
        result = synthesize_one(entry, model=model)
        elapsed = time.time() - t0
        
        if result is None:
            print(f"    FAILED after {MAX_RETRIES} attempts")
            failed += 1
            continue
        
        trajectory = result["trajectory"]
        tool_calls = [s["action"] for s in trajectory if s["action"] != "final"]
        print(f"    ✓ {tool_calls} + final ({elapsed:.1f}s)")
        
        output_entry = {
            "scenario_id": sc_id,
            "service": service,
            "root_cause": entry["root_cause"],
            "severity": entry["severity"],
            "ground_truth": entry.get("ground_truth", {}),
            "question": result["question"],
            "synthesis_reasoning": result["synthesis_reasoning"],
            "trajectory": trajectory,
        }
        
        with SYNTHESIZED_PATH.open("a") as f:
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
        
        synthesized += 1
    
    print(f"\n  Phase 2 complete:")
    print(f"    Synthesized: {synthesized}")
    print(f"    Skipped (resume): {skipped}")
    print(f"    Failed: {failed}")
    print(f"    Output: {SYNTHESIZED_PATH}")
    
    return str(SYNTHESIZED_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-scenarios", type=int, default=0)
    args = parser.parse_args()
    run_phase2(model=args.model, max_scenarios=args.max_scenarios)
