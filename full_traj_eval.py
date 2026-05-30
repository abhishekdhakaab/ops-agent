"""
Full Trajectory Evaluation — Ops Agent Planner
===============================================

Runs the model through ALL investigation steps, simulating real tool outputs
at each step. This is the correct eval for GRPO and DPO which were trained on
multi-step data — not just the first tool call.

For each scenario the agent must:
  1. Pick a first tool (step 0, no evidence)
  2. See the real simulated tool output
  3. Pick next tool based on evidence (step 1)
  4. Repeat until it says "final" or hits max steps

Scoring:
  - trajectory_accuracy : called required tools AND reached final in ≤3 steps
  - coverage            : fraction of required tools the agent actually called
  - no_redundancy_rate  : fraction of scenarios with zero repeated tool calls
  - completion_rate     : fraction where agent said "final" (vs hitting step limit)
  - efficiency_rate     : fraction finishing in ≤3 tool calls
  - avg_steps           : mean tool calls before stopping

Usage:
    cd /path/to/ops-agent
    python full_traj_eval.py --count 100                        # base model only
    python full_traj_eval.py --compare --count 100              # all 4 models
    python full_traj_eval.py --model-path data/trained_models/sft_qwen2.5_1.5b/final
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
RESULTS_DIR = DATA_DIR / "eval_results"

sys.path.insert(0, str(SCRIPT_DIR / "sft_pipeline"))
from tools import run_tool, compress_evidence
from prompts import PLANNER_SYSTEM, build_planner_state, build_planner_user_message
from planner_policy import normalize_action, VALID_ACTIONS

MAX_STEPS = 5          # hard cap — prevents infinite loops
OPTIMAL_MAX_STEPS = 3  # ≤3 tool calls is considered efficient


# ── Required tools per root cause ────────────────────────────────────────────
# A correct investigation must call at least these tools.
# Ordered: [must_call, secondary] — must_call is mandatory, secondary is a bonus.

REQUIRED_TOOLS: Dict[str, Set[str]] = {
    "deploy_regression":   {"deployments"},
    "resource_exhaustion": {"metrics"},
    "upstream_dependency": {"logs"},
    "config_change":       {"deployments"},
    "infrastructure":      {"logs"},
    "healthy":             {"metrics"},
}

BONUS_TOOLS: Dict[str, Set[str]] = {
    "deploy_regression":   {"logs"},
    "resource_exhaustion": {"logs"},
    "upstream_dependency": {"metrics"},
    "config_change":       {"logs"},
    "infrastructure":      {"metrics"},
    "healthy":             set(),
}


# ── Model loading ─────────────────────────────────────────────────────────────

def _ollama_to_hf(model_name: str) -> str:
    mapping = {
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5:3b":   "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5:7b":   "Qwen/Qwen2.5-7B-Instruct",
    }
    return mapping.get(model_name, model_name)


def load_model(model_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    hf_path = _ollama_to_hf(model_path) if ":" in model_path else model_path
    is_lora = Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        cfg = json.loads((Path(model_path) / "adapter_config.json").read_text())
        base = cfg["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True,
                        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_action(model, tokenizer, device, prompt: str) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_action(text: str) -> Optional[Dict]:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ── Trajectory scorer ─────────────────────────────────────────────────────────

def score_trajectory(tools_called: List[str], reached_final: bool, scenario: dict) -> dict:
    root = scenario.get("root_cause", "")
    required = REQUIRED_TOOLS.get(root, {"metrics"})
    bonus    = BONUS_TOOLS.get(root, set())

    called_set = set(tools_called)
    n_steps    = len(tools_called)

    # Coverage: fraction of required tools actually called
    coverage = len(called_set & required) / max(1, len(required))

    # Bonus coverage (secondary tools called)
    bonus_coverage = len(called_set & bonus) / max(1, len(bonus)) if bonus else 1.0

    # No redundancy: all tool calls were unique
    no_redundancy = len(tools_called) == len(called_set)

    # Completion: agent chose to stop
    completion = reached_final

    # Efficiency: stopped in ≤ OPTIMAL_MAX_STEPS
    efficiency = n_steps <= OPTIMAL_MAX_STEPS and reached_final

    # Trajectory accuracy: required tools called + no redundancy + completed
    traj_accurate = (coverage == 1.0) and no_redundancy and completion

    # Composite score (0–1)
    composite = (
        coverage       * 0.40 +
        bonus_coverage * 0.10 +
        no_redundancy  * 0.20 +
        completion     * 0.15 +
        efficiency     * 0.15
    )

    return {
        "traj_accurate":  traj_accurate,
        "composite":      round(composite, 3),
        "coverage":       round(coverage, 3),
        "bonus_coverage": round(bonus_coverage, 3),
        "no_redundancy":  no_redundancy,
        "completion":     completion,
        "efficiency":     efficiency,
        "steps":          n_steps,
        "tools_called":   tools_called,
        "required":       sorted(required),
    }


# ── Single-scenario runner ────────────────────────────────────────────────────

def run_scenario(model, tokenizer, device, scenario: dict) -> dict:
    service     = scenario["service"]
    scenario_id = scenario["id"]
    question    = f"Investigate {service}: we're seeing issues."

    tools_called: List[str] = []
    evidence:     Dict[str, Any] = {}
    reached_final = False
    steps_log = []

    for step in range(MAX_STEPS + 1):
        state = build_planner_state(
            question=question,
            service=service,
            step=step,
            tools_called=tools_called,
            evidence=evidence,
        )
        user_msg = build_planner_user_message(state)

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{PLANNER_SYSTEM}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"

        raw = generate_action(model, tokenizer, device, prompt)
        parsed = parse_action(raw)

        if parsed is None:
            steps_log.append({"step": step, "raw": raw[:80], "action": "INVALID_JSON"})
            break

        action = normalize_action(parsed.get("action", ""))
        steps_log.append({
            "step":    step,
            "action":  action,
            "args":    parsed.get("args", {}),
            "rationale": parsed.get("rationale", "")[:60],
        })

        if action == "final":
            reached_final = True
            break

        if action not in {"metrics", "logs", "deployments"}:
            break

        # Simulate the tool
        args = parsed.get("args", {})
        if "service" not in args:
            args["service"] = service
        raw_out = run_tool(action, args, scenario_id)
        compressed = compress_evidence(action, raw_out)
        evidence[action] = compressed
        tools_called.append(action)

    scores = score_trajectory(tools_called, reached_final, scenario)
    scores["steps_log"] = steps_log
    scores["scenario_id"] = scenario_id
    scores["root_cause"] = scenario.get("root_cause", "")
    return scores


# ── Full eval ─────────────────────────────────────────────────────────────────

def run_eval(model_path: str, scenarios: list, label: str = "model") -> dict:
    print(f"\n{'='*60}")
    print(f"  Full Trajectory Eval: {label}")
    print(f"  Model: {model_path}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"{'='*60}\n")

    model, tokenizer, device = load_model(model_path)
    print(f"  Device: {device}\n")

    results = []
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        sc_t0 = time.time()
        r = run_scenario(model, tokenizer, device, sc)
        elapsed = time.time() - sc_t0

        acc_sym = "✓" if r["traj_accurate"] else "✗"
        print(f"  [{i+1:3d}/{len(scenarios)}] {sc['service']:22s} "
              f"root={sc.get('root_cause','?'):20s} "
              f"steps={r['steps']} tools={r['tools_called']} "
              f"cov={r['coverage']:.0%} final={'Y' if r['completion'] else 'N'} "
              f"{acc_sym}  ({elapsed:.1f}s)")
        results.append(r)

    total_time = time.time() - t0

    # Aggregate
    n = len(results)
    traj_acc      = sum(r["traj_accurate"]  for r in results) / n
    avg_composite = sum(r["composite"]      for r in results) / n
    avg_coverage  = sum(r["coverage"]       for r in results) / n
    no_redund     = sum(r["no_redundancy"]  for r in results) / n
    completion    = sum(r["completion"]     for r in results) / n
    efficiency    = sum(r["efficiency"]     for r in results) / n
    avg_steps     = sum(r["steps"]          for r in results) / n

    # Per root-cause breakdown
    by_root: Dict[str, list] = {}
    for r in results:
        by_root.setdefault(r["root_cause"], []).append(r)

    per_root = {}
    for root, items in sorted(by_root.items()):
        per_root[root] = {
            "traj_accuracy": round(sum(x["traj_accurate"] for x in items) / len(items), 3),
            "avg_coverage":  round(sum(x["coverage"]      for x in items) / len(items), 3),
            "completion":    round(sum(x["completion"]     for x in items) / len(items), 3),
            "count":         len(items),
        }

    summary = {
        "label":              label,
        "model_path":         model_path,
        "scenarios":          n,
        "trajectory_accuracy":round(traj_acc, 3),
        "avg_composite_score":round(avg_composite, 3),
        "avg_coverage":       round(avg_coverage, 3),
        "no_redundancy_rate": round(no_redund, 3),
        "completion_rate":    round(completion, 3),
        "efficiency_rate":    round(efficiency, 3),
        "avg_steps":          round(avg_steps, 2),
        "total_time_s":       round(total_time, 1),
        "per_root_cause":     per_root,
    }

    # Print summary box
    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  {label:^54s} │")
    print(f"  ├────────────────────────────────────────────────────────┤")
    print(f"  │  Trajectory accuracy (correct tools + final):  {traj_acc*100:5.1f}%  │")
    print(f"  │  Avg composite score:                          {avg_composite:5.3f}   │")
    print(f"  │  Tool coverage (required tools called):        {avg_coverage*100:5.1f}%  │")
    print(f"  │  No redundancy rate (zero repeated calls):     {no_redund*100:5.1f}%  │")
    print(f"  │  Completion rate (said 'final'):               {completion*100:5.1f}%  │")
    print(f"  │  Efficiency (≤{OPTIMAL_MAX_STEPS} steps + final):               {efficiency*100:5.1f}%  │")
    print(f"  │  Avg steps per scenario:                       {avg_steps:5.2f}   │")
    print(f"  │  Total time: {total_time:.0f}s                                     │")
    print(f"  └────────────────────────────────────────────────────────┘")

    print(f"\n  Per root-cause:")
    for root, s in per_root.items():
        print(f"    {root:22s}  traj_acc={s['traj_accuracy']*100:5.1f}%  "
              f"coverage={s['avg_coverage']*100:5.1f}%  "
              f"n={s['count']}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ", "_").replace(":", "_").replace("/", "_")
    out = RESULTS_DIR / f"full_traj_eval_{safe}.json"
    with out.open("w") as f:
        json.dump({**summary, "scenario_results": results}, f, indent=2, default=str)
    print(f"\n  Saved: {out}")

    del model
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return summary


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(results: list):
    if len(results) < 2:
        return
    base = results[0]

    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║           FULL TRAJECTORY COMPARISON                             ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Model':<18} {'TrajAcc':>8} {'Coverage':>9} {'NoRedund':>9} {'Finish':>7} {'Steps':>6} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════╣")

    for r in results:
        delta = ""
        if r is not base:
            d = (r["trajectory_accuracy"] - base["trajectory_accuracy"]) * 100
            delta = f"({d:+.1f}pp)"
        print(f"  ║  {r['label']:<18} "
              f"{r['trajectory_accuracy']*100:>7.1f}% "
              f"{r['avg_coverage']*100:>8.1f}% "
              f"{r['no_redundancy_rate']*100:>8.1f}% "
              f"{r['completion_rate']*100:>6.1f}% "
              f"{r['avg_steps']:>5.2f}  "
              f"{delta} ║")

    print(f"  ╚══════════════════════════════════════════════════════════════════╝")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full trajectory evaluation for ops agent")
    parser.add_argument("--model",      default="qwen2.5:1.5b",  help="Base model name or HF id")
    parser.add_argument("--model-path", default=None,            help="Direct path to a trained model")
    parser.add_argument("--count",      type=int, default=100,   help="Number of scenarios (default 100)")
    parser.add_argument("--compare",    action="store_true",     help="Compare base, SFT, GRPO, DPO")
    args = parser.parse_args()

    # Load scenarios (training set — they have real tool simulation data)
    scenarios_path = DATA_DIR / "scenarios.json"
    all_scenarios  = json.loads(scenarios_path.read_text())["scenarios"]
    if args.count > 0:
        all_scenarios = all_scenarios[:args.count]
    print(f"  Loaded {len(all_scenarios)} scenarios from {scenarios_path}")

    model_safe = args.model.replace(":", "_").replace("/", "_")

    if args.compare:
        all_results = []

        # 1. Base model
        hf_base = _ollama_to_hf(args.model) if ":" in args.model else args.model
        r = run_eval(hf_base, all_scenarios, label="Base")
        all_results.append(r)

        # 2. SFT
        sft_path = DATA_DIR / f"trained_models/sft_{model_safe}/final"
        if sft_path.exists():
            r = run_eval(str(sft_path), all_scenarios, label="SFT")
            all_results.append(r)
        else:
            print(f"\n  [SKIP] SFT model not found: {sft_path}")

        # 3. GRPO
        grpo_path = DATA_DIR / "trained_models/grpo_on_sft_v2/final"
        if grpo_path.exists():
            r = run_eval(str(grpo_path), all_scenarios, label="GRPO")
            all_results.append(r)
        else:
            print(f"\n  [SKIP] GRPO model not found: {grpo_path}")

        # 4. DPO
        dpo_path = DATA_DIR / "trained_models/dpo_on_sft_v1/final"
        if dpo_path.exists():
            r = run_eval(str(dpo_path), all_scenarios, label="DPO")
            all_results.append(r)
        else:
            print(f"\n  [SKIP] DPO model not found: {dpo_path}")

        print_comparison(all_results)

    elif args.model_path:
        run_eval(args.model_path, all_scenarios, label=Path(args.model_path).name)
    else:
        hf_base = _ollama_to_hf(args.model) if ":" in args.model else args.model
        run_eval(hf_base, all_scenarios, label=args.model)


if __name__ == "__main__":
    main()
