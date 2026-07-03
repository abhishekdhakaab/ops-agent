"""
Eval script aligned with the new SFT pipeline.

Uses the SAME prompt format and system prompt as training
(from sft_pipeline/prompts.py), and an updated tool accuracy
mapping that matches what the SFT data actually teaches.

Usage:
    # Eval base model:
    python sft_eval.py --model qwen2.5:1.5b --count 100

    # Eval SFT model:
    python sft_eval.py --model-path data/trained_models/sft_qwen2.5_1.5b/final --count 100

    # Compare:
    python sft_eval.py --compare --model qwen2.5:1.5b --count 100
"""

import json
import sys
import re
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent / "sft_pipeline"))

from prompts import PLANNER_SYSTEM, build_planner_state, build_planner_user_message
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "sft_pipeline"))
from planner_policy import normalize_action, best_first_tools

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
RESULTS_DIR = DATA_DIR / "eval_results"



def compute_reward(completion: str, scenario: Dict) -> float:
    """Score a planner completion. Same structure as train.py but with updated tool mapping."""
    reward = 0.0
    
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
    
    reward += 0.5  # Valid JSON
    
    has_action = "action" in parsed
    has_args = "args" in parsed
    
    if has_action and has_args:
        reward += 0.3
    elif has_action:
        reward += 0.1
    
    if not has_action:
        return reward
    
    action = normalize_action(parsed.get("action", ""))
    valid_actions = {"metrics", "logs", "deployments", "final"}

    if action in valid_actions:
        reward += 0.2
    else:
        reward -= 0.3
        return reward
    
    best = best_first_tools(scenario)
    
    if action in best:
        reward += 1.0
    elif action in {"metrics", "logs", "deployments"} and action != "final":
        reward += 0.3
    elif action == "final":
        reward -= 0.2
    
    args = parsed.get("args", {})
    if isinstance(args, dict):
        service = args.get("service", "")
        if service and service == scenario.get("service", ""):
            reward += 0.3
        elif service:
            reward += 0.1
    
    rationale = parsed.get("rationale", "")
    if isinstance(rationale, str) and len(rationale) > 10:
        reward += 0.2
    
    return round(reward, 3)



def _ollama_to_hf(model_name: str) -> str:
    mapping = {
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
        "gemma3:1b": "google/gemma-3-1b-it",
        "gemma3:4b": "google/gemma-3-4b-it",
    }
    return mapping.get(model_name, model_name)


def load_model(model_path: str):
    """Load model and tokenizer, handling both base models and LoRA adapters."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    is_lora = (Path(model_path) / "adapter_config.json").exists() if ":" not in model_path else False
    
    if is_lora:
        from peft import PeftModel
        adapter_cfg = json.loads((Path(model_path) / "adapter_config.json").read_text())
        base_name = adapter_cfg.get("base_model_name_or_path", "")
        print(f"  Loading LoRA adapter from: {model_path}")
        print(f"  Base model: {base_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        hf_path = _ollama_to_hf(model_path)
        print(f"  Loading model: {hf_path}")
        model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


# SCOPE NOTE: This eval tests only the first tool decision (step 0).
# Every prompt is built with empty evidence and no prior tool calls.


def run_eval(model_path: str, label: str = "model", max_scenarios: int = 0) -> dict:
    """Evaluate using the SAME prompt format as SFT training."""
    import torch
    
    print(f"\n{'='*55}")
    print(f"  Eval: {label}")
    print(f"  Model: {model_path}")
    print(f"  Prompt format: SFT-aligned (with evidence fields)")
    print(f"{'='*55}\n")
    
    import warnings
    warnings.warn(
        "sft_eval.py tests on the TRAINING scenarios (scenarios.json). "
        "Results are not out-of-distribution. Use unseen_eval.py for honest evaluation.",
        UserWarning,
        stacklevel=2,
    )
    print("\n  WARNING: This eval uses training scenarios. Results may be inflated.")
    print("  For honest evaluation, use unseen_eval.py\n")

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]

    model, tokenizer, device = load_model(model_path)
    print(f"  Device: {device}")
    print(f"  Evaluating {len(scenarios)} scenarios...\n")
    
    rewards = []
    correct_tools = 0
    valid_json = 0
    total = len(scenarios)
    total_start = time.time()
    
    for i, sc in enumerate(scenarios):
        service = sc["service"]
        question = f"Investigate {service}"
        
        # Build prompt using the SHARED format from sft_pipeline/prompts.py
        state = build_planner_state(
            question=question,
            service=service,
            step=0,
            tools_called=[],
            evidence={},
        )
        user_msg = build_planner_user_message(state)
        
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{PLANNER_SYSTEM}\n\nUSER:\n{user_msg}\n\nASSISTANT:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        reward = compute_reward(completion, sc)
        rewards.append(reward)
        
        best = best_first_tools(sc)
        try:
            m = re.search(r"\{.*\}", completion, flags=re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                action = parsed.get("action", "")
                if action in best:
                    correct_tools += 1
                if isinstance(parsed, dict) and "action" in parsed:
                    valid_json += 1
        except Exception:
            pass
        
        status = "✓" if reward > 1.5 else "○" if reward > 0 else "✗"
        print(f"  [{i+1}/{total}] {service:20s} reward={reward:+.1f} {status}  {completion[:60].strip()}")
    
    total_time = time.time() - total_start
    
    avg_reward = sum(rewards) / max(1, len(rewards))
    tool_acc = correct_tools / max(1, total)
    json_rate = valid_json / max(1, total)
    positive = sum(1 for r in rewards if r > 0)
    strong = sum(1 for r in rewards if r > 1.5)
    
    result = {
        "label": label,
        "model_path": str(model_path),
        "scenarios": total,
        "first_step_avg_reward": round(avg_reward, 3),
        "first_tool_accuracy": round(tool_acc, 3),
        "json_valid_rate": round(json_rate, 3),
        "positive_reward_pct": round(positive / max(1, total), 3),
        "first_step_strong_pct": round(strong / max(1, total), 3),
        "total_time_s": round(total_time, 1),
        "eval_format": "sft_aligned",
    }
    
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  {label:^39s} │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  First-tool accuracy: {tool_acc*100:>6.1f}%            │")
    print(f"  │  First-step avg reward: {avg_reward:>+7.3f}          │")
    print(f"  │  Valid JSON rate:   {json_rate*100:>7.1f}%            │")
    print(f"  │  Positive reward:   {positive:>3d}/{total} ({positive/max(1,total)*100:.0f}%)         │")
    print(f"  │  Strong (>1.5):     {strong:>3d}/{total} ({strong/max(1,total)*100:.0f}%)         │")
    print(f"  │  Time: {total_time:.1f}s ({total_time/max(1,total):.1f}s/scenario)     │")
    print(f"  └─────────────────────────────────────────┘")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ", "_").replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"in_distribution_eval_{safe}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="SFT-aligned evaluation")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Base model name")
    parser.add_argument("--model-path", default=None, help="Path to trained model")
    parser.add_argument("--count", type=int, default=100, help="Number of scenarios")
    parser.add_argument("--compare", action="store_true", help="Run base vs SFT comparison")
    args = parser.parse_args()
    
    if args.compare:
        results = []
        
        print("\n  Evaluating BASE model...")
        r = run_eval(args.model, label="Base (no training)", max_scenarios=args.count)
        results.append(r)
        
        sft_path = str(DATA_DIR / f"trained_models/sft_{args.model.replace(':', '_').replace('/', '_')}" / "final")
        if Path(sft_path).exists():
            print("\n  Evaluating SFT model...")
            r = run_eval(sft_path, label="After SFT", max_scenarios=args.count)
            results.append(r)
        else:
            print(f"\n  SFT model not found at {sft_path}")
        
        if len(results) >= 2:
            base, sft = results[0], results[-1]
            print(f"\n  ╔═══════════════════════════════════════════╗")
            print(f"  ║           COMPARISON (SFT-aligned)         ║")
            print(f"  ╠═══════════════════════════════════════════╣")
            print(f"  ║  {'Metric':<20s} {'Base':>8s} {'SFT':>8s} {'Δ':>8s} ║")
            print(f"  ╠═══════════════════════════════════════════╣")
            
            for metric, key in [("Avg reward", "first_step_avg_reward"), ("Tool accuracy", "first_tool_accuracy"),
                                ("JSON valid", "json_valid_rate"), ("Strong %", "first_step_strong_pct")]:
                b, s = base[key], sft[key]
                d = s - b
                sign = "+" if d > 0 else ""
                fmt = ".3f" if key == "first_step_avg_reward" else ".1%"
                if fmt == ".1%":
                    print(f"  ║  {metric:<20s} {b*100:>7.1f}% {s*100:>7.1f}% {sign}{d*100:.1f}pp ║")
                else:
                    print(f"  ║  {metric:<20s} {b:>+8.3f} {s:>+8.3f} {sign}{d:.3f}  ║")
            
            print(f"  ╚═══════════════════════════════════════════╝")
    else:
        model_path = args.model_path or args.model
        label = args.model_path or args.model
        run_eval(model_path, label=label, max_scenarios=args.count)


if __name__ == "__main__":
    main()
