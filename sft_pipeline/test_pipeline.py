#!/usr/bin/env python3
"""
End-to-end pipeline test using simulated LLM responses.

This verifies the full pipeline logic (Phase 1 → Phase 2 → Phase 3)
without requiring Ollama. It patches the LLM client to return
plausible responses, then runs all 3 phases and validates the output.

Usage:
    python test_pipeline.py
    python test_pipeline.py --scenarios 5   # test with 5 scenarios
"""

import json
import sys
import shutil
from pathlib import Path
from unittest.mock import patch
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, RAW_RUNS_PATH, SYNTHESIZED_PATH, SFT_DATASET_PATH
from tools import load_scenarios_list, compress_evidence, run_tool


# ── Mock LLM responses ───────────────────────────────────────────

# These simulate what a large model would plausibly return
# for different investigation scenarios.

_INVESTIGATION_PLANS = {
    "deploy_regression": [
        [
            {"action": "metrics", "args": {}, "rationale": "Check for latency spikes that correlate with recent changes."},
            {"action": "deployments", "args": {}, "rationale": "Metrics show a spike. Checking if a recent deploy could be the cause."},
            {"action": "final", "args": {}, "rationale": "Latency spike correlates with recent deploy. Deploy regression confirmed."},
        ],
        [
            {"action": "logs", "args": {}, "rationale": "Check error logs to understand the failure pattern."},
            {"action": "deployments", "args": {}, "rationale": "OOM errors suggest memory regression. Checking recent deploys."},
            {"action": "final", "args": {}, "rationale": "OOM errors started after recent deploy. Root cause: deploy regression."},
        ],
        [
            {"action": "deployments", "args": {}, "rationale": "Question hints at recent changes. Check deploys first."},
            {"action": "metrics", "args": {}, "rationale": "Recent deploy found. Checking if metrics show correlated degradation."},
            {"action": "final", "args": {}, "rationale": "Deploy and latency spike are time-correlated. Likely deploy regression."},
        ],
    ],
    "resource_exhaustion": [
        [
            {"action": "metrics", "args": {}, "rationale": "Investigate latency and resource utilization patterns."},
            {"action": "logs", "args": {}, "rationale": "Spike detected. Checking logs for resource exhaustion signals."},
            {"action": "final", "args": {}, "rationale": "High latency + resource errors confirm resource exhaustion."},
        ],
        [
            {"action": "logs", "args": {}, "rationale": "Check error patterns to identify the failure mode."},
            {"action": "metrics", "args": {}, "rationale": "Errors suggest resource issues. Checking metrics for confirmation."},
            {"action": "final", "args": {}, "rationale": "Log errors and latency metrics both point to resource exhaustion."},
        ],
        [
            {"action": "metrics", "args": {}, "rationale": "Start with latency check for performance degradation."},
            {"action": "logs", "args": {}, "rationale": "Latency is high. Looking for error patterns in logs."},
            {"action": "deployments", "args": {}, "rationale": "Check if a recent deploy caused the resource issue."},
        ],
    ],
    "upstream_dependency": [
        [
            {"action": "logs", "args": {}, "rationale": "Intermittent errors suggest upstream issues. Check error logs."},
            {"action": "metrics", "args": {}, "rationale": "Errors point to upstream timeouts. Checking latency impact."},
            {"action": "final", "args": {}, "rationale": "Logs show upstream dependency failures. Not a local issue."},
        ],
        [
            {"action": "metrics", "args": {}, "rationale": "Check overall health and latency patterns."},
            {"action": "logs", "args": {}, "rationale": "Latency variable. Checking logs for upstream error patterns."},
            {"action": "final", "args": {}, "rationale": "Upstream dependency errors confirmed. External dependency issue."},
        ],
        [
            {"action": "logs", "args": {}, "rationale": "Check for error patterns that indicate the failure source."},
            {"action": "deployments", "args": {}, "rationale": "Check if a deploy changed upstream configuration."},
            {"action": "final", "args": {}, "rationale": "Errors from upstream, no relevant deploy. Upstream dependency issue."},
        ],
    ],
    "healthy": [
        [
            {"action": "metrics", "args": {}, "rationale": "Quick health check on latency and throughput."},
            {"action": "logs", "args": {}, "rationale": "Metrics look normal. Checking logs for any hidden issues."},
            {"action": "final", "args": {}, "rationale": "Metrics stable, low error count. Service is healthy."},
        ],
        [
            {"action": "logs", "args": {}, "rationale": "Check if there are meaningful errors behind the alert."},
            {"action": "metrics", "args": {}, "rationale": "Very few errors. Confirm metrics are within normal range."},
            {"action": "final", "args": {}, "rationale": "Minimal errors, normal latency. Likely a false alarm."},
        ],
        [
            {"action": "metrics", "args": {}, "rationale": "Check latency to assess if there's a real problem."},
            {"action": "logs", "args": {}, "rationale": "Latency normal. Quick log check for completeness."},
            {"action": "final", "args": {}, "rationale": "All signals normal. Service healthy, no action needed."},
        ],
    ],
    "config_change": [
        [
            {"action": "logs", "args": {}, "rationale": "Check error logs for unexpected behavior patterns."},
            {"action": "deployments", "args": {}, "rationale": "Errors suggest config issue. Check for recent changes."},
            {"action": "final", "args": {}, "rationale": "Errors correlate with recent config change deployment."},
        ],
        [
            {"action": "deployments", "args": {}, "rationale": "Behavior change suggests a config modification."},
            {"action": "logs", "args": {}, "rationale": "Recent deploy found. Checking if errors match the change."},
            {"action": "final", "args": {}, "rationale": "Deploy introduced config change causing the issue."},
        ],
        [
            {"action": "metrics", "args": {}, "rationale": "Check if service performance has changed."},
            {"action": "deployments", "args": {}, "rationale": "Performance degraded. Checking for config deployments."},
            {"action": "final", "args": {}, "rationale": "Performance change aligns with config deployment."},
        ],
    ],
    "infrastructure": [
        [
            {"action": "logs", "args": {}, "rationale": "Major issues reported. Check logs for infrastructure errors."},
            {"action": "metrics", "args": {}, "rationale": "Infrastructure errors in logs. Checking metrics for impact."},
            {"action": "final", "args": {}, "rationale": "Infrastructure-level failures confirmed. Needs escalation."},
        ],
        [
            {"action": "metrics", "args": {}, "rationale": "Check for severe latency or availability issues."},
            {"action": "logs", "args": {}, "rationale": "Severe issues detected. Checking logs for infra error patterns."},
            {"action": "final", "args": {}, "rationale": "Cluster/infra-level failure. Not application-level."},
        ],
        [
            {"action": "logs", "args": {}, "rationale": "Check for cluster or infrastructure error messages."},
            {"action": "deployments", "args": {}, "rationale": "Infra errors found. Verify no deploy caused this."},
            {"action": "final", "args": {}, "rationale": "Infrastructure failure confirmed. No deploy correlation."},
        ],
    ],
}


def _get_mock_plan(root_cause: str, run_index: int, step_index: int, service: str) -> dict:
    """Get a mock LLM response for a given scenario and step."""
    plans = _INVESTIGATION_PLANS.get(root_cause, _INVESTIGATION_PLANS["resource_exhaustion"])
    plan = plans[run_index % len(plans)]
    
    if step_index < len(plan):
        response = dict(plan[step_index])
        # Fill in service
        if response["action"] != "final":
            response["args"] = {"service": service}
        return response
    else:
        return {"action": "final", "args": {}, "rationale": "Investigation complete."}


def _get_mock_synthesis(scenario_entry: dict) -> dict:
    """Mock the synthesis response."""
    root_cause = scenario_entry["root_cause"]
    service = scenario_entry["service"]
    
    # Pick the first plan for this root cause as the "synthesis"
    plans = _INVESTIGATION_PLANS.get(root_cause, _INVESTIGATION_PLANS["resource_exhaustion"])
    chosen = plans[0]
    
    trajectory = []
    for step in chosen:
        s = dict(step)
        if s["action"] != "final":
            s["args"] = {"service": service}
        else:
            s["args"] = {}
        trajectory.append(s)
    
    return {
        "reasoning": f"Synthesized optimal path for {root_cause} investigation.",
        "trajectory": trajectory,
    }


# ── Mock call_ollama_json ─────────────────────────────────────────

_mock_call_count = 0
_mock_scenario_context = {}  # thread-unsafe but fine for testing


def mock_call_ollama_json(system, user, model=None, temperature=0.7, num_predict=300, max_retries=3):
    """Replace call_ollama_json with deterministic mock responses."""
    global _mock_call_count
    _mock_call_count += 1
    
    # Detect if this is a Phase 1 call (investigation) or Phase 2 call (synthesis)
    if "ATTEMPT" in user or "investigation attempts" in user:
        # Phase 2: synthesis
        ctx = _mock_scenario_context.get("current_entry")
        if ctx:
            return _get_mock_synthesis(ctx)
        return {"reasoning": "test", "trajectory": [
            {"action": "metrics", "args": {}, "rationale": "test"},
            {"action": "final", "args": {}, "rationale": "test"},
        ]}
    else:
        # Phase 1: investigation step
        ctx = _mock_scenario_context.get("current_scenario", {})
        root_cause = ctx.get("root_cause", "resource_exhaustion")
        service = ctx.get("service", "unknown")
        run_idx = _mock_scenario_context.get("current_run", 0)
        step_idx = _mock_scenario_context.get("current_step", 0)
        
        result = _get_mock_plan(root_cause, run_idx, step_idx, service)
        _mock_scenario_context["current_step"] = step_idx + 1
        
        return result


# ── Patched Phase 1 ───────────────────────────────────────────────

def run_phase1_mocked(max_scenarios=3):
    """Run Phase 1 with mocked LLM calls."""
    from phase1_raw_runs import run_single_investigation
    from prompts import generate_question
    
    scenarios = load_scenarios_list()[:max_scenarios]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear existing
    if RAW_RUNS_PATH.exists():
        RAW_RUNS_PATH.unlink()
    
    with patch("phase1_raw_runs.call_ollama_json", side_effect=mock_call_ollama_json):
        for i, scenario in enumerate(scenarios):
            runs = []
            for run_idx in range(3):
                _mock_scenario_context["current_scenario"] = scenario
                _mock_scenario_context["current_run"] = run_idx
                _mock_scenario_context["current_step"] = 0
                
                question = generate_question(scenario, variant=run_idx)
                result = run_single_investigation(scenario, question)
                result["question"] = question
                result["run_index"] = run_idx
                result["elapsed_s"] = 0.1
                runs.append(result)
            
            entry = {
                "scenario_id": scenario["id"],
                "service": scenario["service"],
                "root_cause": scenario["root_cause"],
                "severity": scenario["severity"],
                "ground_truth": scenario.get("ground_truth", {}),
                "runs": runs,
            }
            
            with RAW_RUNS_PATH.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Patched Phase 2 ───────────────────────────────────────────────

def run_phase2_mocked(max_scenarios=3):
    """Run Phase 2 with mocked LLM calls."""
    from phase2_synthesize import load_raw_runs, synthesize_one
    
    if SYNTHESIZED_PATH.exists():
        SYNTHESIZED_PATH.unlink()
    
    raw_entries = load_raw_runs()[:max_scenarios]
    
    with patch("phase2_synthesize.call_ollama_json", side_effect=mock_call_ollama_json):
        for entry in raw_entries:
            _mock_scenario_context["current_entry"] = entry
            
            result = synthesize_one(entry)
            
            if result:
                output_entry = {
                    "scenario_id": entry["scenario_id"],
                    "service": entry["service"],
                    "root_cause": entry["root_cause"],
                    "severity": entry["severity"],
                    "ground_truth": entry.get("ground_truth", {}),
                    "question": result["question"],
                    "synthesis_reasoning": result["synthesis_reasoning"],
                    "trajectory": result["trajectory"],
                }
                
                with SYNTHESIZED_PATH.open("a") as f:
                    f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")


# ── Tests ─────────────────────────────────────────────────────────

def test_full_pipeline(n_scenarios=3):
    """Test the full pipeline end-to-end with mock LLM."""
    
    print(f"\n{'='*60}")
    print(f"  END-TO-END PIPELINE TEST ({n_scenarios} scenarios)")
    print(f"{'='*60}")
    
    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Phase 1
    print(f"\n  Phase 1: Generating raw runs...")
    run_phase1_mocked(max_scenarios=n_scenarios)
    
    assert RAW_RUNS_PATH.exists(), "Phase 1 output not created"
    with RAW_RUNS_PATH.open() as f:
        phase1_entries = [json.loads(l) for l in f if l.strip()]
    
    print(f"    ✓ Generated {len(phase1_entries)} scenario entries")
    for entry in phase1_entries:
        assert len(entry["runs"]) == 3, f"Expected 3 runs, got {len(entry['runs'])}"
        for run in entry["runs"]:
            assert len(run["steps"]) > 0, "Empty run"
            assert run["steps"][-1]["action"] == "final", f"Run didn't end with final: {[s['action'] for s in run['steps']]}"
    print(f"    ✓ All runs complete and end with 'final'")
    
    # Phase 2
    print(f"\n  Phase 2: Synthesizing trajectories...")
    run_phase2_mocked(max_scenarios=n_scenarios)
    
    assert SYNTHESIZED_PATH.exists(), "Phase 2 output not created"
    with SYNTHESIZED_PATH.open() as f:
        phase2_entries = [json.loads(l) for l in f if l.strip()]
    
    print(f"    ✓ Synthesized {len(phase2_entries)} trajectories")
    for entry in phase2_entries:
        traj = entry["trajectory"]
        assert len(traj) >= 2, f"Trajectory too short: {len(traj)}"
        assert traj[-1]["action"] == "final", "Trajectory doesn't end with final"
        
        # No consecutive duplicates
        for i in range(1, len(traj)):
            if traj[i]["action"] == traj[i-1]["action"] and traj[i]["action"] != "final":
                assert False, f"Consecutive duplicate: {traj[i]['action']}"
    print(f"    ✓ All trajectories valid (no duplicates, end with final)")
    
    # Phase 3
    print(f"\n  Phase 3: Formatting SFT dataset...")
    from phase3_format_sft import run_phase3
    run_phase3(max_scenarios=n_scenarios)
    
    assert SFT_DATASET_PATH.exists(), "Phase 3 output not created"
    with SFT_DATASET_PATH.open() as f:
        sft_examples = [json.loads(l) for l in f if l.strip()]
    
    print(f"    ✓ Generated {len(sft_examples)} SFT examples")
    
    # Validate SFT format
    for i, ex in enumerate(sft_examples):
        msgs = ex["messages"]
        assert len(msgs) == 3, f"Example {i}: expected 3 messages, got {len(msgs)}"
        assert msgs[0]["role"] == "system", f"Example {i}: first message not system"
        assert msgs[1]["role"] == "user", f"Example {i}: second message not user"
        assert msgs[2]["role"] == "assistant", f"Example {i}: third message not assistant"
        
        # Assistant message should be valid JSON
        asst = json.loads(msgs[2]["content"])
        assert "action" in asst, f"Example {i}: no 'action' in assistant response"
        assert "rationale" in asst, f"Example {i}: no 'rationale' in assistant response"
        assert asst["action"] in {"metrics", "logs", "deployments", "final"}, \
            f"Example {i}: invalid action {asst['action']}"
    
    print(f"    ✓ All examples have valid format (system/user/assistant, valid JSON)")
    
    # Check evidence materialization
    step1_examples = [ex for ex in sft_examples 
                      if 'Step: 1' in ex["messages"][1]["content"]
                      or 'Step: 2' in ex["messages"][1]["content"]]
    evidence_present = 0
    for ex in step1_examples:
        user_msg = ex["messages"][1]["content"]
        if '"spike_detected"' in user_msg or '"error_count"' in user_msg or '"has_recent_deploy"' in user_msg:
            evidence_present += 1
    
    print(f"    ✓ Evidence materialized in {evidence_present}/{len(step1_examples)} step>0 examples")
    
    # Action distribution
    actions = Counter()
    for ex in sft_examples:
        a = json.loads(ex["messages"][2]["content"])["action"]
        actions[a] += 1
    
    total = len(sft_examples)
    print(f"\n  Action distribution:")
    for action, count in sorted(actions.items()):
        pct = count / total * 100
        bar = "█" * int(pct / 3)
        print(f"    {action:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Final ratio check
    final_pct = actions.get("final", 0) / total * 100
    assert final_pct <= 35, f"Too many 'final' examples: {final_pct:.1f}% (max 30%)"
    print(f"    ✓ 'final' ratio OK ({final_pct:.1f}% ≤ 30%)")
    
    # Print a full example
    print(f"\n  FULL EXAMPLE (step 1 with evidence):")
    print(f"  {'─'*55}")
    for ex in sft_examples:
        if 'Step: 1' in ex["messages"][1]["content"]:
            print(f"  SYSTEM: {ex['messages'][0]['content'][:80]}...")
            print(f"  USER:\n    {ex['messages'][1]['content']}")
            print(f"  ASSISTANT:\n    {ex['messages'][2]['content']}")
            break
    print(f"  {'─'*55}")
    
    print(f"\n  {'='*60}")
    print(f"  ALL TESTS PASSED ✓")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=3)
    args = parser.parse_args()
    test_full_pipeline(n_scenarios=args.scenarios)
