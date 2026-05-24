"""
Phase 3: Format SFT dataset.

Takes synthesized trajectories from Phase 2, re-simulates the tools
in the synthesized order (to get real evidence states), and produces
the final SFT training dataset.

RE-SIMULATION IS KEY: The synthesizer output contains rationales but
we don't trust its evidence values — we re-run the actual tools to
get ground-truth evidence states. This prevents any data contamination
from the synthesis step.

The output is a JSONL file where each line is a TRL-compatible
chat-format training example:
  {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

from config import (
    SYNTHESIZED_PATH,
    SFT_DATASET_PATH,
    STATS_PATH,
    OUTPUT_DIR,
    MAX_FINAL_RATIO,
)
from tools import run_tool, compress_evidence
from prompts import (
    build_planner_state,
    format_sft_example,
)


def load_synthesized() -> List[dict]:
    """Load Phase 2 results."""
    entries = []
    if not SYNTHESIZED_PATH.exists():
        print(f"  ERROR: Phase 2 output not found at {SYNTHESIZED_PATH}")
        print(f"  Run Phase 2 first.")
        sys.exit(1)
    
    with SYNTHESIZED_PATH.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _rationale_has_numeric_mismatch(rationale: str, evidence: dict) -> bool:
    """
    Returns True if the rationale contains numeric values (e.g. latency ms)
    that are not present anywhere in the current evidence dict.
    This catches cases where the synthesizer hallucinated specific numbers.
    """
    # Extract numbers from rationale (e.g. 1900, 900ms, 99%)
    rationale_numbers = set(re.findall(r'\b\d{2,}\b', rationale))
    if not rationale_numbers:
        return False  # No numbers to check

    # Build flat string of all evidence values
    evidence_str = json.dumps(evidence)
    evidence_numbers = set(re.findall(r'\b\d{2,}\b', evidence_str))

    # If rationale has numbers not found anywhere in evidence, flag it
    orphan_numbers = rationale_numbers - evidence_numbers
    return len(orphan_numbers) > 0


def trajectory_to_sft_examples(entry: dict) -> List[dict]:
    """
    Convert one synthesized trajectory into step-level SFT examples.
    
    For a trajectory [logs, deployments, final], produces 3 examples:
      Example 1: state(step=0, evidence={})             → action=logs
      Example 2: state(step=1, evidence={logs:...})     → action=deployments
      Example 3: state(step=2, evidence={logs:..., deployments:...}) → action=final
    
    Evidence at each step is obtained by RE-SIMULATING the tools,
    not by copying from the synthesized trajectory.
    """
    question = entry["question"]
    service = entry["service"]
    scenario_id = entry["scenario_id"]
    trajectory = entry["trajectory"]
    
    examples = []
    evidence_compressed = {}
    tools_called = []
    
    for step_idx, step in enumerate(trajectory):
        action = step["action"]
        args = step.get("args", {})
        rationale = step.get("rationale", "")
        
        # Build the state BEFORE this action (what the model sees when deciding)
        state = build_planner_state(
            question=question,
            service=service,
            step=step_idx,
            tools_called=list(tools_called),
            evidence=dict(evidence_compressed),
        )
        
        # The correct action for this state
        action_json = {
            "action": action,
            "args": args if action != "final" else {},
            "rationale": rationale,
        }
        
        # Before appending each example:
        if _rationale_has_numeric_mismatch(rationale, evidence_compressed):
            # Replace the rationale with a safe generic one derived from evidence
            if action == "metrics":
                m = evidence_compressed.get("metrics", {})
                rationale = (
                    f"Metrics show avg latency {m.get('avg_latency_ms', '?')}ms, "
                    f"p95 {m.get('p95_latency_ms', '?')}ms, "
                    f"spike_detected={m.get('spike_detected', '?')}."
                )
            elif action == "logs":
                lg = evidence_compressed.get("logs", {})
                rationale = (
                    f"Logs show {lg.get('error_count', '?')} errors. "
                    f"Top message: {str(lg.get('common_messages', ['?'])[:1])}."
                )
            elif action == "deployments":
                dep = evidence_compressed.get("deployments", {})
                rationale = (
                    f"Deployment check: has_recent_deploy={dep.get('has_recent_deploy', '?')}, "
                    f"minutes_ago={dep.get('minutes_ago', '?')}."
                )
            # For final, keep rationale as-is (it summarizes all evidence)
            action_json = {
                "action": action,
                "args": args if action != "final" else {},
                "rationale": rationale,
            }
            example = format_sft_example(state, action_json)

        # Create SFT example
        example = format_sft_example(state, action_json)

        # Add metadata (not used in training, useful for analysis)
        example["_metadata"] = {
            "scenario_id": entry["scenario_id"],
            "service": service,
            "root_cause": entry.get("root_cause"),
            "step_index": step_idx,
            "trajectory_length": len(trajectory),
        }

        examples.append(example)
        
        # Now simulate the tool to build evidence for the NEXT step
        if action != "final":
            # Ensure service is in args
            sim_args = dict(args) if isinstance(args, dict) else {}
            sim_args.setdefault("service", service)
            
            # Run actual tool
            raw_output = run_tool(action, sim_args, scenario_id=scenario_id)
            compressed = compress_evidence(action, raw_output)
            
            # Build evidence key (handle multiple calls to same tool with different args)
            evidence_key = action
            if action == "metrics" and sim_args.get("time_range_minutes", 60) != 60:
                evidence_key = f"metrics_{sim_args['time_range_minutes']}min"
            if action == "logs" and sim_args.get("contains"):
                evidence_key = f"logs_filter_{sim_args['contains']}"
            
            evidence_compressed[evidence_key] = compressed
            tools_called.append(action)
    
    return examples


def balance_final_ratio(examples: List[dict], max_ratio: float) -> List[dict]:
    """
    Ensure 'final' examples don't exceed max_ratio of the dataset.
    
    If there are too many 'final' examples, randomly drop some.
    This prevents the model from learning a bias toward stopping early.
    """
    import random
    
    final_examples = []
    tool_examples = []
    
    for ex in examples:
        assistant_msg = ex["messages"][-1]["content"]
        try:
            action = json.loads(assistant_msg).get("action", "")
        except json.JSONDecodeError:
            action = ""
        
        if action == "final":
            final_examples.append(ex)
        else:
            tool_examples.append(ex)
    
    n_tool = len(tool_examples)
    max_final = int(n_tool * max_ratio / (1 - max_ratio))
    
    if len(final_examples) > max_final:
        random.seed(42)
        random.shuffle(final_examples)
        final_examples = final_examples[:max_final]
    
    combined = tool_examples + final_examples
    # Shuffle so finals aren't all at the end
    random.seed(42)
    random.shuffle(combined)
    
    return combined


def compute_stats(examples: List[dict], entries: List[dict]) -> dict:
    """Compute dataset statistics for inspection."""
    
    action_counts = Counter()
    scenario_coverage = set()
    root_cause_counts = Counter()
    step_distribution = Counter()
    
    for ex in examples:
        meta = ex.get("_metadata", {})
        scenario_coverage.add(meta.get("scenario_id", ""))
        root_cause_counts[meta.get("root_cause", "unknown")] += 1
        step_distribution[meta.get("step_index", 0)] += 1
        
        try:
            action = json.loads(ex["messages"][-1]["content"]).get("action", "")
            action_counts[action] += 1
        except:
            action_counts["parse_error"] += 1
    
    total = len(examples)
    
    return {
        "total_examples": total,
        "scenarios_covered": len(scenario_coverage),
        "scenarios_total": len(entries),
        "action_distribution": {
            k: {"count": v, "pct": round(v / max(1, total) * 100, 1)}
            for k, v in sorted(action_counts.items())
        },
        "root_cause_distribution": dict(root_cause_counts),
        "step_distribution": {
            f"step_{k}": v for k, v in sorted(step_distribution.items())
        },
        "avg_trajectory_length": round(
            sum(meta.get("trajectory_length", 0) 
                for ex in examples 
                for meta in [ex.get("_metadata", {})])
            / max(1, total),
            1
        ),
    }


def run_phase3(max_scenarios: int = 0) -> str:
    """
    Generate the final SFT dataset.
    
    No LLM calls needed — this is pure data transformation.
    Reads synthesized trajectories, re-simulates tools, formats examples.
    """
    entries = load_synthesized()
    if max_scenarios > 0:
        entries = entries[:max_scenarios]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total = len(entries)
    all_examples = []
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"  PHASE 3: SFT Dataset Formatting")
    print(f"  Synthesized trajectories: {total}")
    print(f"{'='*60}\n")
    
    for i, entry in enumerate(entries):
        sc_id = entry["scenario_id"]
        service = entry["service"]
        trajectory = entry.get("trajectory", [])
        
        if not trajectory:
            print(f"  [{i+1}/{total}] {sc_id}: SKIP (empty trajectory)")
            failed += 1
            continue
        
        try:
            examples = trajectory_to_sft_examples(entry)
            all_examples.extend(examples)
            
            tool_calls = [s["action"] for s in trajectory if s["action"] != "final"]
            print(f"  [{i+1}/{total}] {sc_id}: {len(examples)} examples from {tool_calls}+final")
            
        except Exception as e:
            print(f"  [{i+1}/{total}] {sc_id}: ERROR {e}")
            failed += 1
    
    # Balance final ratio
    before_balance = len(all_examples)
    all_examples = balance_final_ratio(all_examples, MAX_FINAL_RATIO)
    after_balance = len(all_examples)
    
    if before_balance != after_balance:
        print(f"\n  Balanced 'final' ratio: {before_balance} → {after_balance} examples")
    
    # Compute stats
    stats = compute_stats(all_examples, entries)
    
    # Write SFT dataset
    with SFT_DATASET_PATH.open("w") as f:
        for ex in all_examples:
            # Remove _metadata from training data (keep it clean)
            clean = {"messages": ex["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    
    # Write stats
    with STATS_PATH.open("w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n  {'─'*50}")
    print(f"  DATASET STATS:")
    print(f"  {'─'*50}")
    print(f"  Total SFT examples: {stats['total_examples']}")
    print(f"  Scenarios covered:  {stats['scenarios_covered']}/{stats['scenarios_total']}")
    print(f"  Action distribution:")
    for action, info in stats["action_distribution"].items():
        bar = "█" * int(info["pct"] / 3)
        print(f"    {action:12s}: {info['count']:4d} ({info['pct']:5.1f}%) {bar}")
    print(f"  Root cause distribution:")
    for rc, count in sorted(stats["root_cause_distribution"].items()):
        print(f"    {rc:25s}: {count}")
    print(f"  Step distribution:")
    for step, count in sorted(stats["step_distribution"].items()):
        print(f"    {step}: {count}")
    print(f"  {'─'*50}")
    print(f"  Output: {SFT_DATASET_PATH}")
    print(f"  Stats:  {STATS_PATH}")
    
    # Print a few sample examples for inspection
    print(f"\n  SAMPLE EXAMPLES (first 2):")
    print(f"  {'─'*50}")
    for ex in all_examples[:2]:
        user_msg = ex["messages"][1]["content"]
        asst_msg = ex["messages"][2]["content"]
        # Truncate for display
        print(f"  USER (first 200 chars):")
        print(f"    {user_msg[:200]}...")
        print(f"  ASSISTANT:")
        print(f"    {asst_msg}")
        print()
    
    return str(SFT_DATASET_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenarios", type=int, default=0)
    args = parser.parse_args()
    run_phase3(max_scenarios=args.max_scenarios)
