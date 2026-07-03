"""
Training pipeline for the Ops Agent planner.

Stage 1: SFT — supervised fine-tuning on correct tool choices
Stage 2: GRPO — RL with automatic reward from scenario ground truth

Usage:
    # Generate training data first:
    python app/eval/generate_scenarios.py --count 200

    # Run SFT:
    python app/training/train.py sft --model gemma3:1b --epochs 3

    # Run GRPO:
    python app/training/train.py grpo --model gemma3:1b --epochs 2

    # Full pipeline (SFT then GRPO):
    python app/training/train.py full --model gemma3:1b

    # Export trained model to Ollama:
    python app/training/train.py export --name ops-planner-v1

Requirements:
    pip install trl transformers peft datasets torch
"""

import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SFT_DATA = DATA_DIR / "trl_planner_sft.jsonl"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
OUTPUT_DIR = DATA_DIR / "trained_models"




def load_sft_data() -> List[Dict]:
    """Load SFT data in TRL chat format."""
    rows = []
    with SFT_DATA.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_grpo_prompts() -> List[Dict]:
    """Build GRPO prompts from scenarios."""
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    prompts = []
    
    planner_system = (
        "You are an ops investigation planner. Choose the NEXT best tool action.\n"
        "Return ONLY valid JSON: {action, args, rationale}.\n"
        "Allowed actions: metrics, logs, deployments, final.\n"
        "Never invent evidence. Be evidence-driven."
    )
    
    for sc in scenarios:
        service = sc["service"]
        user_text = json.dumps({
            "question": f"Investigate {service}",
            "service": service,
            "evidence_keys": [],
            "available_tools": ["metrics", "logs", "deployments", "final"],
            "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
        })
        
        prompts.append({
            "prompt": f"{planner_system}\n\nUSER:\n{user_text}\n\nASSISTANT:\n",
            "scenario": sc,  # Keep scenario for reward computation
        })
    
    return prompts



def run_sft(model_name: str, epochs: int = 3, lr: float = 2e-5, batch_size: int = 4, gpu: bool = False):
    """Run supervised fine-tuning on planner data."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install trl transformers peft datasets torch")
        sys.exit(1)

    use_gpu = gpu or (torch.cuda.is_available())
    effective_batch = batch_size * (8 if use_gpu else 1)
    print(f"\n{'='*50}")
    print(f"  SFT Training")
    print(f"  Model: {model_name}")
    print(f"  Device: {'GPU (bf16)' if use_gpu else 'CPU'}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch/device: {batch_size}")
    print(f"{'='*50}\n")
    
    sft_rows = load_sft_data()
    print(f"  Loaded {len(sft_rows)} SFT training examples")
    
    dataset = Dataset.from_list(sft_rows)
    
    # Resolve model path — if it's an Ollama model name, we need the HF equivalent
    hf_model = _ollama_to_hf(model_name)
    print(f"  Loading model: {hf_model}")
    
    # LoRA keeps the experiment small enough to rerun locally.
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    output_dir = OUTPUT_DIR / f"sft_{model_name.replace(':', '_').replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device detection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device_type = "cuda"
        use_bf16, use_fp16 = True, False
        grad_accum = 2
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
        use_bf16, use_fp16 = False, False   # MPS doesn't support bf16/fp16 training well
        grad_accum = 1
    else:
        device_type = "cpu"
        use_bf16, use_fp16 = False, False
        grad_accum = 1

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=5,
        max_length=1024,
        dataloader_num_workers=4 if device_type == "cuda" else 0,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    # Qwen needs an explicit padding token for batched training.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    print("  Starting SFT training...\n")
    trainer.train()
    
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n  SFT model saved to: {output_dir / 'final'}")
    
    return str(output_dir / "final")



def run_grpo(model_name: str, epochs: int = 2, lr: float = 1e-5, batch_size: int = 2, num_generations: int = 4):
    """Run GRPO (Group Relative Policy Optimization) with automatic reward."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install trl transformers peft datasets torch")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print(f"  GRPO Training (RL)")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}, Generations per prompt: {num_generations}")
    print(f"{'='*50}\n")
    
    grpo_data = load_grpo_prompts()
    print(f"  Loaded {len(grpo_data)} GRPO prompts")
    
    scenario_lookup = {}
    dataset_rows = []
    for item in grpo_data:
        sc = item["scenario"]
        prompt = item["prompt"]
        scenario_lookup[prompt] = sc
        dataset_rows.append({"prompt": prompt})
    
    dataset = Dataset.from_list(dataset_rows)
    
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """Score each completion against its scenario's ground truth."""
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            sc = scenario_lookup.get(prompt, {})
            r = compute_reward(completion, sc)
            rewards.append(r)
        return rewards
    
    # Load model — if model_name is a path to SFT adapter, merge and save as clean checkpoint first
    is_adapter_path = Path(model_name).exists() and (Path(model_name) / "adapter_config.json").exists()
    
    if is_adapter_path:
        adapter_cfg = json.loads((Path(model_name) / "adapter_config.json").read_text())
        base_name = adapter_cfg.get("base_model_name_or_path", "")
        print(f"  Loading SFT adapter from: {model_name}")
        print(f"  Base model: {base_name}")
        
        # Save merged model to disk so GRPO starts clean (no leftover PEFT state)
        merged_path = Path(model_name).parent / "merged"
        if not merged_path.exists():
            print(f"  Merging SFT weights into base model...")
            base_model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
            from peft import PeftModel as _PM
            sft_model = _PM.from_pretrained(base_model, model_name)
            merged_model = sft_model.merge_and_unload()
            merged_model.save_pretrained(str(merged_path))
            merged_tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
            merged_tokenizer.save_pretrained(str(merged_path))
            del base_model, sft_model, merged_model
            print(f"  Saved merged model to: {merged_path}")
        else:
            print(f"  Using cached merged model: {merged_path}")
        
        # Load the merged model fresh — completely clean, no PEFT artifacts
        model = AutoModelForCausalLM.from_pretrained(str(merged_path), trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
        output_dir = OUTPUT_DIR / "grpo_on_sft"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        hf_model = _ollama_to_hf(model_name)
        print(f"  Loading model: {hf_model}")
        model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        output_dir = OUTPUT_DIR / f"grpo_{model_name.replace(':', '_').replace('/', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Small LoRA for GRPO — gentle adjustments on top of SFT, not overwriting it
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        num_generations=num_generations,
        max_completion_length=64,
        gradient_accumulation_steps=1,
        warmup_steps=5,
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    print("  Starting GRPO training...\n")
    print("  The model will generate multiple completions per prompt,")
    print("  score them with the reward function, and learn to prefer")
    print("  higher-scoring completions.\n")
    trainer.train()
    
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n  GRPO model saved to: {output_dir / 'final'}")
    
    return str(output_dir / "final")



def export_to_ollama(model_path: str, name: str):
    """Export a trained model to Ollama for serving."""
    print(f"\n  Exporting {model_path} to Ollama as '{name}'...")
    print(f"  This requires converting to GGUF format.\n")
    
    try:
        import subprocess
        
        merged_path = Path(model_path).parent / "merged"
        if not merged_path.exists():
            print("  Merging LoRA weights...")
            from peft import PeftModel, AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
            
            model = AutoPeftModelForCausalLM.from_pretrained(model_path)
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_path))
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(str(merged_path))
            print(f"  Merged model saved to {merged_path}")
        
        print("\n  To convert to GGUF and import to Ollama:")
        print(f"  1. python llama.cpp/convert_hf_to_gguf.py {merged_path} --outfile {name}.gguf")
        print(f"  2. Create a Modelfile:")
        print(f"     FROM ./{name}.gguf")
        print(f'     SYSTEM "You are an ops investigation planner..."')
        print(f"     PARAMETER temperature 0.2")
        print(f"  3. ollama create {name} -f Modelfile")
        print(f"  4. Update .env: LLM_MODEL={name}")
        
    except Exception as e:
        print(f"  Export helper failed: {e}")
        print(f"  Manual steps:")
        print(f"  1. Convert {model_path} to GGUF format")
        print(f"  2. Create Ollama Modelfile pointing to the GGUF")
        print(f"  3. ollama create {name} -f Modelfile")



def run_eval_direct(model_path: str, label: str = "model", max_scenarios: int = 0):
    """Evaluate a model directly in Python — no server, no Ollama.
    
    Loads the model, runs each scenario prompt, scores with reward function.
    This is how we measure if fine-tuning actually helped.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)
    
    print(f"\n{'='*55}")
    print(f"  Direct Eval: {label}")
    print(f"  Model: {model_path}")
    print(f"{'='*55}\n")
    
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]
    
    # Force CPU for eval — MPS has a known bug where model.generate()
    # creates KV-cache tensors >4GB which crash with:
    # CPU is fast enough for 1.5B model inference (~3-5s per scenario)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"  Device: {device}")
    
    hf_path = _ollama_to_hf(model_path) if ":" in model_path else model_path
    is_lora = (Path(model_path) / "adapter_config.json").exists() if ":" not in model_path else False
    
    print(f"  Loading {'LoRA adapter' if is_lora else 'model'}: {hf_path}")
    
    if is_lora:
        adapter_cfg = json.loads((Path(model_path) / "adapter_config.json").read_text())
        base_name = adapter_cfg.get("base_model_name_or_path", "")
        print(f"  Base model: {base_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA for faster inference
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    print(f"  Model loaded. Evaluating {len(scenarios)} scenarios...\n")
    
    planner_system = (
        "You are an ops investigation planner. Choose the NEXT best tool action.\n"
        "Return ONLY valid JSON: {action, args, rationale}.\n"
        "Allowed actions: metrics, logs, deployments, final.\n"
        "Never invent evidence. Be evidence-driven."
    )
    
    rewards = []
    correct_tools = 0
    valid_json = 0
    total = len(scenarios)
    
    import time
    total_start = time.time()
    
    for i, sc in enumerate(scenarios):
        service = sc["service"]
        user_text = json.dumps({
            "question": f"Investigate {service}",
            "service": service,
            "evidence_keys": [],
            "available_tools": ["metrics", "logs", "deployments", "final"],
            "instruction": "Choose the NEXT action as JSON {action, args, rationale}.",
        })
        
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": planner_system},
                {"role": "user", "content": user_text},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{planner_system}\n\nUSER:\n{user_text}\n\nASSISTANT:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
        
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
        
        best = _best_first_tools(sc)
        import re
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
        "avg_reward": round(avg_reward, 3),
        "tool_accuracy": round(tool_acc, 3),
        "json_valid_rate": round(json_rate, 3),
        "positive_reward_pct": round(positive / max(1, total), 3),
        "strong_reward_pct": round(strong / max(1, total), 3),
        "total_time_s": round(total_time, 1),
        "per_scenario_s": round(total_time / max(1, total), 1),
    }
    
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  {label:^39s} │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Avg reward:        {avg_reward:>+8.3f}            │")
    print(f"  │  Tool accuracy:     {tool_acc*100:>7.1f}%            │")
    print(f"  │  Valid JSON rate:   {json_rate*100:>7.1f}%            │")
    print(f"  │  Positive reward:   {positive:>3d}/{total} ({positive/max(1,total)*100:.0f}%)         │")
    print(f"  │  Strong (>1.5):     {strong:>3d}/{total} ({strong/max(1,total)*100:.0f}%)         │")
    print(f"  │  Time: {total_time:.1f}s ({total_time/max(1,total):.1f}s/scenario)     │")
    print(f"  └─────────────────────────────────────────┘")
    
    RESULTS_DIR = DATA_DIR / "eval_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ", "_").replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"direct_eval_{safe}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    
    return result


def print_comparison_table(results: list):
    """Print a before/after comparison from direct eval results."""
    if not results:
        return
    
    print(f"\n  ╔═══════════════════════════════════════════════════════════╗")
    print(f"  ║              BEFORE / AFTER COMPARISON                    ║")
    print(f"  ╠═══════════════════════════════════════════════════════════╣")
    
    header = f"  ║  {'Model':<20s} {'Reward':>8s} {'ToolAcc':>8s} {'JSON%':>7s} {'Good%':>7s} ║"
    print(header)
    print(f"  ╠═══════════════════════════════════════════════════════════╣")
    
    for r in results:
        label = r['label'][:20]
        print(f"  ║  {label:<20s} {r['avg_reward']:>+8.3f} {r['tool_accuracy']*100:>7.1f}% {r['json_valid_rate']*100:>6.1f}% {r['positive_reward_pct']*100:>6.1f}% ║")
    
    print(f"  ╠═══════════════════════════════════════════════════════════╣")
    
    if len(results) >= 2:
        base = results[0]
        best = results[-1]
        reward_delta = best["avg_reward"] - base["avg_reward"]
        tool_delta = (best["tool_accuracy"] - base["tool_accuracy"]) * 100
        json_delta = (best["json_valid_rate"] - base["json_valid_rate"]) * 100
        
        print(f"  ║  IMPROVEMENT ({base['label'][:10]} → {best['label'][:10]}):               ║")
        print(f"  ║    Reward:      {reward_delta:>+.3f}                                ║")
        print(f"  ║    Tool acc:    {tool_delta:>+.1f}pp                                ║")
        print(f"  ║    JSON valid:  {json_delta:>+.1f}pp                                ║")
    
    print(f"  ╚═══════════════════════════════════════════════════════════╝")



def _ollama_to_hf(model_name: str) -> str:
    """Map Ollama model names to HuggingFace model IDs."""
    mapping = {
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
        "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",  # gated
        "gemma3:1b": "google/gemma-3-1b-it",
        "gemma3:4b": "google/gemma-3-4b-it",
        "gemma3:12b": "google/gemma-3-12b-it",
        "phi4:14b": "microsoft/phi-4",
    }
    return mapping.get(model_name, model_name)



def main():
    parser = argparse.ArgumentParser(description="Train the ops agent planner")
    parser.add_argument("stage", choices=["sft", "grpo", "full", "export", "reward-test", "eval", "eval-compare"],
                       help="Training stage to run")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Model name (Ollama or HF)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--name", default="ops-planner-v1", help="Ollama model name for export")
    parser.add_argument("--model-path", default=None, help="Path to trained model (for eval/export)")
    parser.add_argument("--count", type=int, default=0, help="Max scenarios to eval (0=all)")
    args = parser.parse_args()
    
    if args.stage == "reward-test":
        print("\n  Testing reward function...\n")
        scenarios = json.loads(SCENARIOS_PATH.read_text())["scenarios"][:3]
        
        for sc in scenarios:
            service = sc["service"]
            root = sc["root_cause"]
            best_tools = _best_first_tools(sc)
            
            good = json.dumps({"action": list(best_tools)[0], "args": {"service": service}, "rationale": "Checking evidence."})
            r_good = compute_reward(good, sc)
            
            wrong_tools = {"metrics", "logs", "deployments"} - best_tools
            bad = json.dumps({"action": list(wrong_tools)[0] if wrong_tools else "final", "args": {"service": service}, "rationale": "Not sure."})
            r_bad = compute_reward(bad, sc)
            
            r_garbage = compute_reward("I think we should check the metrics maybe?", sc)
            r_invalid = compute_reward('{"action": "metrics", "args": {', sc)
            
            print(f"  {service} ({root}):")
            print(f"    Best tools: {best_tools}")
            print(f"    Good completion:  reward={r_good}")
            print(f"    Wrong tool:       reward={r_bad}")
            print(f"    Garbage text:     reward={r_garbage}")
            print(f"    Invalid JSON:     reward={r_invalid}")
            print()
        return
    
    if args.stage == "eval":
        model_path = args.model_path or args.model
        run_eval_direct(model_path, label=model_path, max_scenarios=args.count)
        return
    
    if args.stage == "eval-compare":
        base_model = args.model
        sft_path = str(OUTPUT_DIR / f"sft_{base_model.replace(':', '_').replace('/', '_')}" / "final")
        grpo_path = str(OUTPUT_DIR / "grpo_on_sft" / "final")
        # Fallback: GRPO trained on base model
        grpo_path_alt = str(OUTPUT_DIR / f"grpo_{base_model.replace(':', '_').replace('/', '_')}" / "final")
        
        results = []
        
        print("\n  Evaluating BASE model...")
        r = run_eval_direct(base_model, label="Base (no training)", max_scenarios=args.count)
        results.append(r)
        
        if Path(sft_path).exists():
            print("\n  Evaluating SFT model...")
            r = run_eval_direct(sft_path, label="After SFT", max_scenarios=args.count)
            results.append(r)
        else:
            print(f"\n  SFT model not found at {sft_path} — skipping")
        
        found_grpo = None
        if Path(grpo_path).exists():
            found_grpo = grpo_path
        elif Path(grpo_path_alt).exists():
            found_grpo = grpo_path_alt
        
        if found_grpo:
            print("\n  Evaluating GRPO model...")
            r = run_eval_direct(found_grpo, label="After SFT+GRPO", max_scenarios=args.count)
            results.append(r)
        else:
            print(f"\n  GRPO model not found — skipping")
        
        print_comparison_table(results)
        return
    
    if args.stage == "sft":
        run_sft(args.model, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    
    elif args.stage == "grpo":
        run_grpo(args.model, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    
    elif args.stage == "full":
        print("\n  Running full pipeline: SFT → GRPO\n")
        sft_path = run_sft(args.model, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        print("\n  SFT complete. Starting GRPO on fine-tuned model...\n")
        run_grpo(sft_path, epochs=max(1, args.epochs - 1), lr=args.lr / 2, batch_size=args.batch_size)
    
    elif args.stage == "export":
        model_path = args.model_path or str(OUTPUT_DIR / f"grpo_{args.model.replace(':', '_')}" / "final")
        export_to_ollama(model_path, args.name)


if __name__ == "__main__":
    main()
