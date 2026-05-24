"""
GRPO Training for the Ops Agent Planner
========================================

Builds on the SFT pipeline — uses the SAME prompt format, evidence
compression, and tool simulation. Zero distribution shift.

Key difference from old GRPO: generates prompts at MULTIPLE evidence
states (step 0, 1, 2), not just empty state. This gives RL signal
for conditional decisions like "given OOM errors in logs, call deployments."

Usage:
    # Generate GRPO prompts from Phase 2 synthesized trajectories:
    python grpo_train.py generate --count 100

    # Run GRPO training on SFT model:
    python grpo_train.py train --model qwen2.5:1.5b --epochs 2

    # Full pipeline (generate + train):
    python grpo_train.py full --model qwen2.5:1.5b

    # Test reward function:
    python grpo_train.py test-reward

Requirements:
    pip install trl transformers peft datasets torch
"""

import json
import sys
import re
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

# ── Path setup ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
OUTPUT_DIR = SCRIPT_DIR / "output"
SYNTHESIZED_PATH = OUTPUT_DIR / "phase2_synthesized.jsonl"
GRPO_PROMPTS_PATH = OUTPUT_DIR / "grpo_prompts.jsonl"
TRAINED_MODELS_DIR = DATA_DIR / "trained_models"

# Add sft_pipeline to path for shared imports
sys.path.insert(0, str(SCRIPT_DIR))

from tools import run_tool, compress_evidence
from prompts import (
    PLANNER_SYSTEM,
    build_planner_state,
    build_planner_user_message,
)
from planner_policy import normalize_action, VALID_ACTIONS


# ══════════════════════════════════════════════════════════════════
# PHASE A: Generate GRPO prompts at multiple evidence states
# ══════════════════════════════════════════════════════════════════

def generate_grpo_prompts(max_scenarios: int = 0) -> str:
    """
    Generate GRPO prompts from Phase 2 synthesized trajectories.

    For each scenario's synthesized trajectory [tool_A, tool_B, final],
    creates prompts at each decision point:
      - Step 0: no evidence → expected action = tool_A
      - Step 1: has tool_A output → expected action = tool_B
      - Step 2: has tool_A + tool_B output → expected action = final

    Evidence is obtained by RE-SIMULATING tools (same as Phase 3 SFT).
    Prompt format uses the SHARED build_planner_user_message().

    Output: grpo_prompts.jsonl where each line has:
      - prompt: the formatted prompt string
      - step_info: metadata for reward computation (expected action, etc.)
    """
    # Load synthesized trajectories
    if not SYNTHESIZED_PATH.exists():
        print(f"  ERROR: {SYNTHESIZED_PATH} not found. Run Phase 2 first.")
        sys.exit(1)

    entries = []
    with SYNTHESIZED_PATH.open("r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if max_scenarios > 0:
        entries = entries[:max_scenarios]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    step_distribution = Counter()

    print(f"\n{'='*60}")
    print(f"  Generating GRPO prompts from {len(entries)} synthesized trajectories")
    print(f"{'='*60}\n")

    for i, entry in enumerate(entries):
        scenario_id = entry["scenario_id"]
        service = entry["service"]
        question = entry["question"]
        trajectory = entry["trajectory"]

        # Walk through the trajectory, building evidence at each step
        evidence_compressed = {}
        tools_called = []

        for step_idx, step in enumerate(trajectory):
            action = step["action"]
            args = step.get("args", {})

            # Build the prompt for this decision point
            # (state BEFORE the action — what the model sees when deciding)
            state = build_planner_state(
                question=question,
                service=service,
                step=step_idx,
                tools_called=list(tools_called),
                evidence=dict(evidence_compressed),
            )
            prompt_text = build_planner_user_message(state)

            # Store the prompt with metadata for reward computation
            step_info = {
                "scenario_id": scenario_id,
                "service": service,
                "root_cause": entry.get("root_cause", ""),
                "step": step_idx,
                "tools_called": list(tools_called),
                "expected_action": action,
                "trajectory_length": len(trajectory),
                "evidence_keys": list(evidence_compressed.keys()),
            }

            all_prompts.append({
                "prompt": prompt_text,
                "step_info": step_info,
            })
            step_distribution[step_idx] += 1

            # Now simulate the tool to build evidence for the NEXT step
            if action != "final":
                sim_args = dict(args) if isinstance(args, dict) else {}
                sim_args.setdefault("service", service)

                raw_output = run_tool(action, sim_args, scenario_id=scenario_id)
                compressed = compress_evidence(action, raw_output)

                # Evidence key (handle different args for same tool)
                evidence_key = action
                tr = int(sim_args.get("time_range_minutes", 60)) if action == "metrics" else 0
                if action == "metrics" and tr != 60:
                    evidence_key = f"metrics_{tr}min"
                contains_val = sim_args.get("contains")
                if action == "logs" and contains_val and isinstance(contains_val, str):
                    evidence_key = f"logs_filter_{contains_val}"

                evidence_compressed[evidence_key] = compressed
                tools_called.append(action)

    # Write prompts
    with GRPO_PROMPTS_PATH.open("w") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Stats
    print(f"  Total GRPO prompts: {len(all_prompts)}")
    print(f"  Step distribution:")
    for step, count in sorted(step_distribution.items()):
        print(f"    Step {step}: {count}")

    # Expected action distribution
    action_dist = Counter(p["step_info"]["expected_action"] for p in all_prompts)
    print(f"  Expected action distribution:")
    for a, c in sorted(action_dist.items()):
        pct = c / len(all_prompts) * 100
        print(f"    {a:12s}: {c:4d} ({pct:.1f}%)")

    print(f"\n  Output: {GRPO_PROMPTS_PATH}")
    return str(GRPO_PROMPTS_PATH)


# ══════════════════════════════════════════════════════════════════
# REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════

def compute_reward(completion: str, step_info: Dict[str, Any]) -> float:
    """
    Score a single GRPO completion against the expected action.

    Reward range: approximately [-1.0, 2.5]

    Scoring hierarchy:
      1. Can we parse valid JSON?              → gate (-1.0 if no)
      2. Is the action a valid tool name?      → gate
      3. Does it match the expert trajectory?  → main signal
      4. Is the service correct?               → minor bonus
      5. Is there a rationale?                 → minor bonus

    The expert match is the clear peak, but alternatives get
    differentiated scores so GRPO has variance to work with.
    """
    reward = 0.0

    expected_action = step_info.get("expected_action")
    if not expected_action:
        # Cannot score without knowing the expected action — penalize as invalid
        return -1.0
    tools_called = step_info.get("tools_called", [])
    service = step_info.get("service", "")
    step = step_info.get("step", 0)

    # ── 1. Parse JSON ─────────────────────────────────────────
    text = completion.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    parsed = None
    try:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
    except Exception:
        pass

    if parsed is None:
        return -1.0

    reward += 0.3  # valid JSON

    # ── 2. Valid action? ──────────────────────────────────────
    action = normalize_action(parsed.get("action", ""))

    if action not in VALID_ACTIONS:
        return reward - 0.5  # invalid action

    reward += 0.2  # valid action name

    # ── 3. Tool choice (main signal) ──────────────────────────
    if action == expected_action:
        # Perfect match with expert trajectory
        reward += 1.5
    elif action == "final":
        # Model wants to stop
        if step >= 2:
            reward += 0.4  # stopping after 2+ tools is reasonable
        elif step == 1 and len(tools_called) >= 1:
            reward += 0.0  # debatable — maybe enough evidence
        else:
            reward -= 0.5  # stopping at step 0 with no evidence
    elif action in tools_called:
        # Redundant call — tool output already in evidence
        # This is the behavioral issue we most want to fix
        reward -= 0.8
    elif action in {"metrics", "logs", "deployments"}:
        # Different tool than expert, but at least gathering evidence
        if action not in tools_called:
            reward += 0.3  # new information, just not the expert choice
        else:
            reward -= 0.3  # somehow still redundant

    # ── 4. Service correctness ────────────────────────────────
    args = parsed.get("args", {})
    if isinstance(args, dict):
        pred_service = args.get("service", "")
        if pred_service == service:
            reward += 0.3
        elif pred_service and action != "final":
            reward += 0.1  # has a service, just wrong one
    elif action != "final":
        # No args at all for a tool call
        reward -= 0.1

    # ── 5. Rationale present and non-trivial ──────────────────
    rationale = parsed.get("rationale", "")
    if isinstance(rationale, str) and len(rationale) > 15:
        reward += 0.2
    elif isinstance(rationale, str) and len(rationale) > 5:
        reward += 0.1

    return round(reward, 3)


# ══════════════════════════════════════════════════════════════════
# GRPO TRAINING
# ══════════════════════════════════════════════════════════════════

def _ollama_to_hf(model_name: str) -> str:
    mapping = {
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5:3b":   "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5:7b":   "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5:14b":  "Qwen/Qwen2.5-14B-Instruct",
        "llama3.2:1b":  "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2:3b":  "meta-llama/Llama-3.2-3B-Instruct",
        "gemma3:1b":    "google/gemma-3-1b-it",
        "gemma3:4b":    "google/gemma-3-4b-it",
        "gemma3:12b":   "google/gemma-3-12b-it",
        "phi4:14b":     "microsoft/phi-4",
    }
    return mapping.get(model_name, model_name)


def merge_sft(model_name: str) -> Path:
    """
    Merge the SFT LoRA adapter into the base model weights.

    Produces sft_{model}/merged_for_grpo/ — a standard HF checkpoint
    that GRPO and DPO can load without any PEFT machinery.
    Idempotent: skips if the merged model already exists.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel as _PM
        import torch as _t
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    sft_path = TRAINED_MODELS_DIR / f"sft_{model_name.replace(':', '_').replace('/', '_')}" / "final"
    if not sft_path.exists():
        print(f"  ERROR: SFT adapter not found at {sft_path}")
        print(f"  Run SFT training first.")
        sys.exit(1)

    merged_path = sft_path.parent / "merged_for_grpo"
    if merged_path.exists():
        print(f"  Merged model already exists: {merged_path}")
        return merged_path

    adapter_cfg = json.loads((sft_path / "adapter_config.json").read_text())
    base_name = adapter_cfg.get("base_model_name_or_path", "")
    print(f"  Merging SFT adapter into base model...")
    print(f"  Adapter: {sft_path}")
    print(f"  Base:    {base_name}")

    dtype = _t.bfloat16 if _t.cuda.is_available() else _t.float32
    base_model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True, torch_dtype=dtype)
    sft_model = _PM.from_pretrained(base_model, str(sft_path))
    merged = sft_model.merge_and_unload()
    merged.save_pretrained(str(merged_path))
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))
    del base_model, sft_model, merged
    print(f"  Saved merged model: {merged_path}")
    return merged_path


def run_grpo(
    model_name: str,
    epochs: int = 3,
    lr: float = 5e-6,
    batch_size: int = 1,
    num_generations: int = 8,
):
    """
    Run GRPO training on the SFT-trained model.

    Loads GRPO prompts (with step_info metadata), builds a reward
    function that scores completions against the expert trajectory,
    and trains with TRL's GRPOTrainer.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install trl transformers peft datasets torch")
        sys.exit(1)

    # ── Load prompts ──────────────────────────────────────────
    if not GRPO_PROMPTS_PATH.exists():
        print(f"  ERROR: {GRPO_PROMPTS_PATH} not found.")
        print(f"  Run: python grpo_train.py generate")
        sys.exit(1)

    grpo_data = []
    with GRPO_PROMPTS_PATH.open("r") as f:
        for line in f:
            if line.strip():
                grpo_data.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"  GRPO Training")
    print(f"  Model: {model_name}")
    print(f"  Prompts: {len(grpo_data)}")
    print(f"  Epochs: {epochs}, Generations/prompt: {num_generations}")
    print(f"{'='*60}\n")

    # ── Build dataset and reward lookup ───────────────────────
    # We need to map from the prompt string back to step_info
    # for reward computation. Use a dict keyed by prompt text.

    # First, format prompts with chat template
    # We need the tokenizer to apply the template
    sft_path = TRAINED_MODELS_DIR / f"sft_{model_name.replace(':', '_').replace('/', '_')}" / "final"
    is_sft = sft_path.exists()

    if is_sft:
        print(f"  Loading SFT adapter from: {sft_path}")
        merged_path = merge_sft(model_name)

    # Device detection: CUDA > MPS > CPU
    import torch as _torch
    if _torch.cuda.is_available():
        _device_type = "cuda"
        _torch_dtype = _torch.bfloat16
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        _device_type = "mps"
        _torch_dtype = _torch.float32
    else:
        _device_type = "cpu"
        _torch_dtype = _torch.float32
    print(f"  Device: {_device_type}")

    if is_sft:
        model = AutoModelForCausalLM.from_pretrained(str(merged_path), trust_remote_code=True, torch_dtype=_torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
    else:
        hf_model = _ollama_to_hf(model_name)
        print(f"  No SFT model found. Loading base: {hf_model}")
        model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, torch_dtype=_torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompts using chat template (matches SFT training format)
    step_info_lookup = {}
    dataset_rows = []

    for item in grpo_data:
        user_msg = item["prompt"]
        step_info = item["step_info"]

        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"{PLANNER_SYSTEM}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"

        step_info_lookup[formatted_prompt] = step_info
        dataset_rows.append({"prompt": formatted_prompt})

    # ── Inject redundancy-trap prompts ────────────────────────────────
    # These prompts show a state where one tool has already been called
    # and give the model a chance to either repeat it (bad) or move on (good).
    # This ensures the -0.8 redundancy penalty actually fires during training.

    trap_prompts_added = 0
    MAX_TRAP_PROMPTS = min(len(grpo_data) // 4, 50)  # at most 25% of dataset

    entries_for_traps = []
    with SYNTHESIZED_PATH.open("r") as _f:
        for _line in _f:
            if _line.strip():
                entries_for_traps.append(json.loads(_line))

    for entry in entries_for_traps:
        if trap_prompts_added >= MAX_TRAP_PROMPTS:
            break
        trajectory = entry.get("trajectory", [])
        tool_steps = [s for s in trajectory if s.get("action") != "final"]
        if len(tool_steps) < 2:
            continue

        # Build a prompt where the first tool has already been called,
        # but the expected next action is the second tool (not the first again)
        first_tool = tool_steps[0]["action"]
        second_tool = tool_steps[1]["action"]
        first_args = tool_steps[0].get("args", {})
        service = entry.get("service", "unknown")

        # Simulate the first tool output as evidence
        try:
            first_output = run_tool(first_tool, first_args, entry["scenario_id"])
            evidence_so_far = {first_tool: compress_evidence(first_tool, first_output)}
        except Exception:
            continue

        state = build_planner_state(
            question=entry["question"],
            service=service,
            step=1,
            tools_called=[first_tool],
            evidence=evidence_so_far,
        )
        user_msg = build_planner_user_message(state)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) if hasattr(tokenizer, "apply_chat_template") else (
            f"{PLANNER_SYSTEM}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"
        )

        trap_step_info = {
            "expected_action": second_tool,
            "tools_called": [first_tool],
            "service": service,
            "step": 1,
            "is_redundancy_trap": True,
        }
        step_info_lookup[formatted] = trap_step_info
        dataset_rows.append({"prompt": formatted})
        trap_prompts_added += 1

    print(f"  Added {trap_prompts_added} redundancy-trap prompts to GRPO dataset")
    # ─────────────────────────────────────────────────────────────────

    dataset = Dataset.from_list(dataset_rows)
    print(f"  Dataset size: {len(dataset_rows)} prompts")

    # ── Reward function ───────────────────────────────────────
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """Score each completion using step-aware reward."""
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            info = step_info_lookup.get(prompt, {})
            if not info:
                # Fallback: try to find closest match (handles whitespace diffs)
                for key, val in step_info_lookup.items():
                    if key[:200] == prompt[:200]:
                        info = val
                        break
            if not info:
                # Complete lookup failure — cannot score this completion
                rewards.append(-1.0)
                continue
            r = compute_reward(completion, info)
            rewards.append(r)
        return rewards

    # ── LoRA config — larger rank, more modules to avoid KL drift ──
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── GRPO config ───────────────────────────────────────────
    output_dir = TRAINED_MODELS_DIR / "grpo_on_sft_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tune grad_accum so generation_batch_size = batch_size * grad_accum
    # is divisible by num_generations. On GPU use larger batches.
    if _device_type == "cuda":
        grad_accum = 2
        use_bf16 = True
    elif _device_type == "mps":
        grad_accum = 2
        use_bf16 = False
    else:
        grad_accum = 4
        use_bf16 = False

    # generation_batch_size = batch_size * grad_accum; must be divisible by num_generations
    gen_batch = batch_size * grad_accum
    effective_generations = num_generations if gen_batch % num_generations == 0 else min(num_generations, gen_batch)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        num_generations=effective_generations,
        max_completion_length=256,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=20,
        beta=0.1,
        bf16=use_bf16,
    )

    # ── Train ─────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"\n  Starting GRPO training...")
    print(f"  The model will generate {num_generations} completions per prompt,")
    print(f"  score them with the step-aware reward function, and learn")
    print(f"  to prefer higher-scoring completions.\n")

    trainer.train()

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n  GRPO model saved to: {output_dir / 'final'}")

    return str(output_dir / "final")


# ══════════════════════════════════════════════════════════════════
# REWARD FUNCTION TESTS
# ══════════════════════════════════════════════════════════════════

def test_reward():
    """Test the reward function on hand-crafted examples."""
    print(f"\n{'='*60}")
    print(f"  Reward Function Tests")
    print(f"{'='*60}\n")

    # Step 0: no evidence, expected=logs
    info_step0 = {
        "expected_action": "logs",
        "tools_called": [],
        "service": "cdn-edge",
        "step": 0,
    }

    cases_step0 = [
        ("Expert match",
         '{"action": "logs", "args": {"service": "cdn-edge"}, "rationale": "Check error patterns first."}'),
        ("Wrong tool (not redundant)",
         '{"action": "metrics", "args": {"service": "cdn-edge"}, "rationale": "Check latency."}'),
        ("Final too early",
         '{"action": "final", "args": {}, "rationale": "Done."}'),
        ("Invalid action",
         '{"action": "restart", "args": {}, "rationale": "Restart the service."}'),
        ("Invalid JSON",
         'I think we should check the logs maybe?'),
        ("Wrong service",
         '{"action": "logs", "args": {"service": "wrong-svc"}, "rationale": "Check error patterns."}'),
    ]

    print(f"  Step 0 (no evidence, expected=logs):")
    for label, completion in cases_step0:
        r = compute_reward(completion, info_step0)
        print(f"    {r:>+6.2f}  {label}")

    # Step 1: has logs evidence, expected=deployments
    info_step1 = {
        "expected_action": "deployments",
        "tools_called": ["logs"],
        "service": "cdn-edge",
        "step": 1,
    }

    cases_step1 = [
        ("Expert match",
         '{"action": "deployments", "args": {"service": "cdn-edge"}, "rationale": "OOM errors suggest deploy regression."}'),
        ("Different tool (metrics, not redundant)",
         '{"action": "metrics", "args": {"service": "cdn-edge"}, "rationale": "Check latency impact."}'),
        ("Redundant (logs again)",
         '{"action": "logs", "args": {"service": "cdn-edge"}, "rationale": "Need more log details."}'),
        ("Final at step 1",
         '{"action": "final", "args": {}, "rationale": "Have enough evidence."}'),
    ]

    print(f"\n  Step 1 (has logs, expected=deployments):")
    for label, completion in cases_step1:
        r = compute_reward(completion, info_step1)
        print(f"    {r:>+6.2f}  {label}")

    # Step 2: has logs+deployments, expected=final
    info_step2 = {
        "expected_action": "final",
        "tools_called": ["logs", "deployments"],
        "service": "cdn-edge",
        "step": 2,
    }

    cases_step2 = [
        ("Expert match (final)",
         '{"action": "final", "args": {}, "rationale": "OOM errors correlate with deploy. Sufficient evidence."}'),
        ("Calls metrics (not redundant but unnecessary)",
         '{"action": "metrics", "args": {"service": "cdn-edge"}, "rationale": "Check latency too."}'),
        ("Redundant (logs again)",
         '{"action": "logs", "args": {"service": "cdn-edge"}, "rationale": "Check again."}'),
    ]

    print(f"\n  Step 2 (has logs+deployments, expected=final):")
    for label, completion in cases_step2:
        r = compute_reward(completion, info_step2)
        print(f"    {r:>+6.2f}  {label}")

    # Verify ordering
    print(f"\n  Reward ordering check:")
    print(f"    Expert > Alternative > Redundant > Invalid JSON?")

    r_expert = compute_reward(
        '{"action": "logs", "args": {"service": "cdn-edge"}, "rationale": "Check errors."}',
        info_step0)
    r_alt = compute_reward(
        '{"action": "metrics", "args": {"service": "cdn-edge"}, "rationale": "Check latency."}',
        info_step0)
    r_redundant = compute_reward(
        '{"action": "logs", "args": {"service": "cdn-edge"}, "rationale": "Check again."}',
        {"expected_action": "metrics", "tools_called": ["logs"], "service": "cdn-edge", "step": 1})
    r_invalid = compute_reward("not json at all", info_step0)

    print(f"    Expert:    {r_expert:+.2f}")
    print(f"    Alt tool:  {r_alt:+.2f}")
    print(f"    Redundant: {r_redundant:+.2f}")
    print(f"    Invalid:   {r_invalid:+.2f}")

    assert r_expert > r_alt > r_redundant > r_invalid, "ORDERING VIOLATED!"
    print(f"    ✓ Ordering correct")


# ══════════════════════════════════════════════════════════════════
# DPO TRAINING  (standard preference-learning fallback)
# ══════════════════════════════════════════════════════════════════

_CHOSEN_RATIONALE = {
    "deployments": "A recent deployment is the most likely cause. Checking deployment history to identify the offending change.",
    "metrics":     "Resource or performance issues surface in metrics first. Checking latency, error rate, and throughput.",
    "logs":        "Error logs will show the exact failure mode. Checking logs to identify the root exception.",
    "final":       "Evidence is sufficient to conclude. Reporting diagnosis.",
}

_REJECT_FOR = {
    "deployments": "metrics",
    "metrics":     "logs",
    "logs":        "deployments",
    "final":       "metrics",
}

def _dpo_completion(action: str, service: str) -> str:
    rationale = _CHOSEN_RATIONALE.get(action, f"Investigating {service}.")
    if action == "final":
        return json.dumps({"action": "final", "args": {}, "rationale": rationale}, ensure_ascii=False)
    return json.dumps({"action": action, "args": {"service": service}, "rationale": rationale}, ensure_ascii=False)


def run_dpo(
    model_name: str,
    epochs: int = 2,
    lr: float = 5e-6,
    batch_size: int = 2,
):
    """
    Direct Preference Optimization on the SFT model.

    For each GRPO prompt we create:
      chosen  = response with the expert-expected action (high reward)
      rejected = response with the most-confused wrong action (low reward)

    DPO directly maximises P(chosen) / P(rejected) without online sampling,
    making it much more stable than GRPO on CPU.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    if not GRPO_PROMPTS_PATH.exists():
        print(f"  ERROR: {GRPO_PROMPTS_PATH} not found. Run: python grpo_train.py generate")
        sys.exit(1)

    grpo_data = [json.loads(l) for l in GRPO_PROMPTS_PATH.open() if l.strip()]
    print(f"\n{'='*60}")
    print(f"  DPO Training (Direct Preference Optimization)")
    print(f"  Base: SFT merged model  |  Prompts: {len(grpo_data)}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"{'='*60}\n")

    # ── Load merged SFT model (auto-merge if not yet done) ───────────
    merged_path = merge_sft(model_name)

    # Device detection: CUDA > MPS > CPU
    import torch as _torch
    if _torch.cuda.is_available():
        _dpo_device, _dpo_dtype, _dpo_bf16 = "cuda", _torch.bfloat16, True
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        _dpo_device, _dpo_dtype, _dpo_bf16 = "mps", _torch.float32, False
    else:
        _dpo_device, _dpo_dtype, _dpo_bf16 = "cpu", _torch.float32, False
    print(f"  Device: {_dpo_device}")

    print(f"  Loading merged SFT: {merged_path}")
    model = AutoModelForCausalLM.from_pretrained(str(merged_path), trust_remote_code=True, torch_dtype=_dpo_dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build preference pairs ────────────────────────────────
    pairs = []
    skipped = 0
    for item in grpo_data:
        step_info = item["step_info"]
        expected = step_info.get("expected_action", "")
        service = step_info.get("service", "")
        if not expected or not service:
            skipped += 1
            continue

        wrong = _REJECT_FOR.get(expected, "metrics")

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": item["prompt"]},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_str = f"{PLANNER_SYSTEM}\n\nUSER:\n{item['prompt']}\n\nASSISTANT:\n"

        pairs.append({
            "prompt":   prompt_str,
            "chosen":   _dpo_completion(expected, service),
            "rejected": _dpo_completion(wrong, service),
        })

    print(f"  Preference pairs: {len(pairs)}  (skipped {skipped} with missing info)")

    dataset = Dataset.from_list(pairs)

    # ── LoRA — same spec as fixed GRPO ───────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_dir = TRAINED_MODELS_DIR / "dpo_on_sft_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        beta=0.1,
        logging_steps=5,
        save_strategy="epoch",
        warmup_steps=10,
        gradient_accumulation_steps=2 if _dpo_device == "cuda" else 1,
        max_length=1024,
        bf16=_dpo_bf16,
        dataloader_num_workers=4 if _dpo_device == "cuda" else 0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("  Starting DPO training...\n")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n  DPO model saved to: {output_dir / 'final'}")
    return str(output_dir / "final")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GRPO/DPO Training (SFT-aligned)")
    parser.add_argument("stage", choices=["generate", "train", "full", "dpo", "merge", "test-reward"],
                        help="What to run: generate|train|full|dpo|merge|test-reward")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-generations", type=int, default=8, help="Completions per prompt (GRPO only)")
    parser.add_argument("--count", type=int, default=0, help="Max scenarios (0=all)")
    args = parser.parse_args()

    if args.stage == "test-reward":
        test_reward()
        return

    if args.stage == "generate":
        generate_grpo_prompts(max_scenarios=args.count)
        return

    if args.stage == "merge":
        merge_sft(model_name=args.model)
        return

    if args.stage == "dpo":
        run_dpo(
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        return

    if args.stage == "train":
        run_grpo(
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_generations=args.num_generations,
        )
        return

    if args.stage == "full":
        print("\n  Running full GRPO pipeline: Generate prompts → Train\n")
        generate_grpo_prompts(max_scenarios=args.count)
        run_grpo(
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_generations=args.num_generations,
        )


if __name__ == "__main__":
    main()
