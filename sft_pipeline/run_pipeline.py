#!/usr/bin/env python3
"""
EvalLLM SFT Data Generation Pipeline
=====================================

End-to-end pipeline to generate high-quality SFT training data
for the ops agent planner.

Usage:
    # Full pipeline (all 3 phases):
    python run_pipeline.py --model gemma3:12b

    # Individual phases:
    python run_pipeline.py --model gemma3:12b --phase 1
    python run_pipeline.py --model gemma3:12b --phase 2
    python run_pipeline.py --phase 3          # no LLM needed

    # Quick test (3 scenarios only):
    python run_pipeline.py --model gemma3:12b --max-scenarios 3

    # Dry run (test tools and prompts, no LLM calls):
    python run_pipeline.py --dry-run

    # Clean previous outputs and start fresh:
    python run_pipeline.py --model gemma3:12b --clean

Pipeline:
    Phase 1: Run each scenario 3x with large model + tool simulation
             → output/phase1_raw_runs.jsonl
    
    Phase 2: Synthesize 3 runs into 1 clean trajectory per scenario
             → output/phase2_synthesized.jsonl
    
    Phase 3: Re-simulate tools, format as SFT training examples
             → output/sft_dataset.jsonl (final output)
"""

import argparse
import json
import sys
import time
import shutil
from pathlib import Path

# Ensure sft_pipeline/ is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    OLLAMA_MODEL,
    OUTPUT_DIR,
    RAW_RUNS_PATH,
    SYNTHESIZED_PATH,
    SFT_DATASET_PATH,
    SCENARIOS_PATH,
)


def run_dry_run():
    """Test pipeline components without making any LLM calls."""
    print("\n" + "=" * 60)
    print("  DRY RUN: Testing pipeline components")
    print("=" * 60)
    
    # Test 1: Scenario loading
    print("\n  [1/4] Testing scenario loading...")
    from tools import load_scenarios_list
    scenarios = load_scenarios_list()
    print(f"    ✓ Loaded {len(scenarios)} scenarios")
    print(f"    Sample: {scenarios[0]['id']} ({scenarios[0]['service']}, {scenarios[0]['root_cause']})")
    
    # Test 2: Tool simulation
    print("\n  [2/4] Testing tool simulation...")
    from tools import run_tool, compress_evidence
    
    service = scenarios[0]["service"]
    sc_id = scenarios[0]["id"]
    
    metrics_out = run_tool("metrics", {"service": service, "time_range_minutes": 60}, scenario_id=sc_id)
    metrics_comp = compress_evidence("metrics", metrics_out)
    print(f"    ✓ metrics({service})")
    print(f"      Raw keys: {list(metrics_out.keys())}")
    print(f"      Compressed: {json.dumps(metrics_comp)}")
    
    logs_out = run_tool("logs", {"service": service}, scenario_id=sc_id)
    logs_comp = compress_evidence("logs", logs_out)
    print(f"    ✓ logs({service})")
    print(f"      Compressed: {json.dumps(logs_comp)}")
    
    deploy_out = run_tool("deployments", {"service": service}, scenario_id=sc_id)
    deploy_comp = compress_evidence("deployments", deploy_out)
    print(f"    ✓ deployments({service})")
    print(f"      Compressed: {json.dumps(deploy_comp)}")
    
    # Test 3: Prompt building
    print("\n  [3/4] Testing prompt building...")
    from prompts import (
        PLANNER_SYSTEM,
        build_planner_state,
        build_planner_user_message,
        format_sft_example,
        generate_question,
    )
    
    question = generate_question(scenarios[0], variant=0)
    print(f"    ✓ Question: {question}")
    
    # State at step 0 (no evidence)
    state0 = build_planner_state(question, service, 0, [], {})
    msg0 = build_planner_user_message(state0)
    print(f"    ✓ Step 0 prompt ({len(msg0)} chars)")
    
    # State at step 1 (with metrics evidence)
    state1 = build_planner_state(question, service, 1, ["metrics"], {"metrics": metrics_comp})
    msg1 = build_planner_user_message(state1)
    print(f"    ✓ Step 1 prompt ({len(msg1)} chars)")
    
    # State at step 2 (with metrics + logs evidence)
    state2 = build_planner_state(question, service, 2, ["metrics", "logs"],
                                  {"metrics": metrics_comp, "logs": logs_comp})
    msg2 = build_planner_user_message(state2)
    print(f"    ✓ Step 2 prompt ({len(msg2)} chars)")
    
    # Format an SFT example
    action = {"action": "logs", "args": {"service": service}, "rationale": "Check error logs for patterns."}
    sft_ex = format_sft_example(state1, action)
    print(f"    ✓ SFT example has {len(sft_ex['messages'])} messages")
    
    # Test 4: Show a full example
    print("\n  [4/4] Full SFT example preview:")
    print(f"  {'─'*50}")
    print(f"  SYSTEM ({len(PLANNER_SYSTEM)} chars):")
    print(f"    {PLANNER_SYSTEM[:100]}...")
    print(f"  USER:")
    print(f"    {msg1[:300]}...")
    print(f"  ASSISTANT:")
    print(f"    {json.dumps(action)}")
    print(f"  {'─'*50}")
    
    # Token estimation
    # Rough: 1 token ≈ 4 chars for English text
    avg_example_chars = len(PLANNER_SYSTEM) + len(msg1) + len(json.dumps(action))
    est_tokens = avg_example_chars / 4
    print(f"\n  Estimated tokens per example: ~{int(est_tokens)}")
    print(f"  With 100 scenarios × ~3 steps: ~{int(est_tokens * 300)} total tokens")
    
    print(f"\n  ✓ Dry run passed. All components working.\n")


def clean_outputs():
    """Remove all pipeline outputs."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("  Cleaned output directory.")


def main():
    parser = argparse.ArgumentParser(
        description="EvalLLM SFT Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --model gemma3:12b              # full pipeline
  python run_pipeline.py --model gemma3:12b --phase 1    # only Phase 1
  python run_pipeline.py --phase 3                        # only Phase 3 (no LLM)
  python run_pipeline.py --dry-run                        # test components
  python run_pipeline.py --model gemma3:12b --max-scenarios 3  # quick test
        """,
    )
    parser.add_argument("--model", default=OLLAMA_MODEL,
                       help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                       help="Run a specific phase only (default: all)")
    parser.add_argument("--max-scenarios", type=int, default=0,
                       help="Limit scenarios for testing (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test components without LLM calls")
    parser.add_argument("--clean", action="store_true",
                       help="Clean previous outputs before running")
    args = parser.parse_args()
    
    # Dry run
    if args.dry_run:
        run_dry_run()
        return
    
    # Clean if requested
    if args.clean:
        clean_outputs()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check Ollama for phases that need it
    needs_llm = args.phase in (1, 2) or args.phase is None
    if needs_llm:
        from ollama_client import check_ollama_ready
        if not check_ollama_ready(args.model):
            sys.exit(1)
        print(f"  ✓ Ollama ready with model: {args.model}")
    
    # Timer
    pipeline_start = time.time()
    
    print(f"\n{'═'*60}")
    print(f"  EvalLLM SFT Data Generation Pipeline")
    print(f"  Model: {args.model}")
    print(f"  Scenarios: {'all' if args.max_scenarios == 0 else args.max_scenarios}")
    print(f"  Phase: {'all' if args.phase is None else args.phase}")
    print(f"{'═'*60}")
    
    # Phase 1
    if args.phase is None or args.phase == 1:
        from phase1_raw_runs import run_phase1
        run_phase1(model=args.model, max_scenarios=args.max_scenarios)
    
    # Phase 2
    if args.phase is None or args.phase == 2:
        from phase2_synthesize import run_phase2
        run_phase2(model=args.model, max_scenarios=args.max_scenarios)
    
    # Phase 3
    if args.phase is None or args.phase == 3:
        from phase3_format_sft import run_phase3
        run_phase3(max_scenarios=args.max_scenarios)
    
    # Final summary
    elapsed = time.time() - pipeline_start
    
    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE ({elapsed:.0f}s)")
    print(f"{'═'*60}")
    
    if SFT_DATASET_PATH.exists():
        n_examples = sum(1 for _ in SFT_DATASET_PATH.open())
        print(f"  SFT dataset: {SFT_DATASET_PATH}")
        print(f"  Total examples: {n_examples}")
        print(f"\n  To use this data for training:")
        print(f"  1. Copy {SFT_DATASET_PATH.name} to data/trl_planner_sft.jsonl")
        print(f"  2. Update train.py to use max_length=1024")
        print(f"  3. Run: python app/training/train.py sft --model qwen2.5:1.5b")
    
    print()


if __name__ == "__main__":
    main()
