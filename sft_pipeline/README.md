# SFT Data Generation Pipeline

Generates high-quality SFT training data for the EvalLLM ops agent planner.

## The Problem This Solves

The original SFT data had two critical issues:
1. **No evidence in prompts** — the model saw "evidence_keys: [logs]" but never what the logs actually said. It couldn't learn conditional reasoning.
2. **Training on failure modes** — 25/26 trajectories hit max_steps with massive tool repetition. The model was learning degenerate behavior.

## How It Works

```
Phase 1: Raw Runs
  For each scenario, run 3 independent investigations using a large model.
  Each run is a multi-turn loop: model picks tool → we simulate it → feed output back.
  The model never sees ground truth — it reasons from evidence only.

Phase 2: Synthesis  
  Feed all 3 runs (with tool outputs) to the large model.
  It produces ONE optimal trajectory: minimal tool calls, no redundancy,
  evidence-grounded rationales.

Phase 3: Format SFT Dataset
  Re-simulate tools in the synthesized order to get real evidence states.
  Extract step-level training examples with full materialized evidence.
  Balance action distribution (cap "final" at 30%).
```

## Output Format

Each SFT example is a TRL-compatible chat message:
```json
{
  "messages": [
    {"role": "system", "content": "<planner system prompt>"},
    {"role": "user", "content": "Question: ...\nService: ...\nStep: 1\nTools called: [\"metrics\"]\nEvidence: {\"metrics\": {\"spike_detected\": true, ...}}\n\nChoose the NEXT action."},
    {"role": "assistant", "content": "{\"action\": \"logs\", \"args\": {\"service\": \"X\"}, \"rationale\": \"Spike detected, checking error logs.\"}"}
  ]
}
```

## Usage

```bash
cd sft_pipeline/

# 1. Test components (no LLM needed):
python run_pipeline.py --dry-run

# 2. Quick test with 3 scenarios:
python run_pipeline.py --model gemma3:12b --max-scenarios 3

# 3. Full pipeline (all 100 scenarios):
python run_pipeline.py --model gemma3:12b

# 4. Run phases individually:
python run_pipeline.py --model gemma3:12b --phase 1   # raw runs
python run_pipeline.py --model gemma3:12b --phase 2   # synthesis
python run_pipeline.py --phase 3                       # format (no LLM)

# 5. Clean and restart:
python run_pipeline.py --model gemma3:12b --clean

# 6. Run end-to-end tests (no Ollama needed):
python test_pipeline.py --scenarios 20
```

## Pipeline is resumable

If interrupted, Phase 1 and Phase 2 skip already-completed scenarios on restart.

## Configuration

Edit `config.py`:
- `OLLAMA_MODEL`: which model to use (default: "gemma3:12b")
- `RUNS_PER_SCENARIO`: investigation attempts per scenario (default: 3)
- `MAX_STEPS_PER_RUN`: max tool calls per run (default: 5)
- `TEMPERATURE_PHASE1/2`: generation temperature
- `MAX_FINAL_RATIO`: cap on "final" action examples (default: 0.30)

## File Structure

```
sft_pipeline/
├── config.py              # All settings and paths
├── tools.py               # Tool simulation + evidence compression (SHARED)
├── prompts.py             # Prompt builder + state formatter (SHARED)
├── ollama_client.py       # Ollama API client with retries
├── phase1_raw_runs.py     # Phase 1: multi-turn investigation runs
├── phase2_synthesize.py   # Phase 2: trajectory synthesis
├── phase3_format_sft.py   # Phase 3: SFT dataset formatting
├── run_pipeline.py        # Main orchestrator CLI
├── test_pipeline.py       # End-to-end test with mock LLM
└── output/
    ├── phase1_raw_runs.jsonl     # Raw investigation attempts
    ├── phase2_synthesized.jsonl  # Clean synthesized trajectories
    ├── sft_dataset.jsonl         # FINAL: training data
    └── pipeline_stats.json       # Dataset statistics
```

## GRPO Compatibility

The key functions shared between SFT and future GRPO:
- `tools.compress_evidence()` — same state representation everywhere
- `prompts.build_planner_user_message()` — same prompt format everywhere
- `prompts.PLANNER_SYSTEM` — same system prompt everywhere

When adding GRPO, import these from this pipeline. The GRPO rollout loop
will use the same state builder and tool simulator, so there's zero
distribution shift between SFT and GRPO training.

## Expected Output

For 100 scenarios with 3-step trajectories:
- ~300 raw SFT examples before balancing
- ~250-280 after "final" ratio balancing
- Action distribution: ~33% metrics, ~27% logs, ~12% deployments, ~28% final
- All step>0 examples have full materialized evidence
