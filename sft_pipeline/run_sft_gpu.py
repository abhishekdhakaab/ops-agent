"""
SFT Training (GPU-optimized) for the Ops Agent Planner.

Loads Phase 3 SFT dataset and trains with TRL's SFTTrainer.
Compatible with `accelerate launch` for multi-GPU training.

Usage:
    # Single GPU or CPU:
    python sft_pipeline/run_sft_gpu.py --model qwen2.5:1.5b --epochs 3

    # Multi-GPU (8 GPUs, bf16):
    accelerate launch --num_processes 8 --mixed_precision bf16 \\
        sft_pipeline/run_sft_gpu.py --model qwen2.5:1.5b --epochs 3 --batch-size 8

Output:
    data/trained_models/sft_<model>/final/
"""

import json
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
SFT_DATA_PATH = SCRIPT_DIR / "output" / "sft_dataset.jsonl"
TRAINED_MODELS_DIR = DATA_DIR / "trained_models"


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


def run_sft(model_name: str, epochs: int = 3, lr: float = 2e-5, batch_size: int = 8):
    """
    Supervised fine-tuning on Phase 3 SFT dataset.

    Uses LoRA r=32 across all 7 projection modules for full expressive capacity.
    Saves the final adapter to data/trained_models/sft_{model}/final/.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall: pip install trl transformers peft datasets torch")
        sys.exit(1)

    # ── Device detection: CUDA > MPS > CPU ──────────────────────────
    if torch.cuda.is_available():
        device_type = "cuda"
        use_bf16, use_fp16 = True, False
        grad_accum = 2
        torch_dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
        use_bf16, use_fp16 = False, False
        grad_accum = 1
        torch_dtype = torch.float32
    else:
        device_type = "cpu"
        use_bf16, use_fp16 = False, False
        grad_accum = 1
        torch_dtype = torch.float32

    hf_model = _ollama_to_hf(model_name)
    model_safe = model_name.replace(":", "_").replace("/", "_")
    output_dir = TRAINED_MODELS_DIR / f"sft_{model_safe}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SFT Training (GPU-optimized)")
    print(f"  Model:        {model_name}  →  {hf_model}")
    print(f"  Device:       {device_type}  |  bf16={use_bf16}")
    print(f"  Epochs:       {epochs}  |  LR: {lr}  |  Batch/device: {batch_size}")
    print(f"  Grad accum:   {grad_accum}  (effective batch: {batch_size * grad_accum})")
    print(f"  Output:       {output_dir}")
    print(f"{'='*60}\n")

    if not SFT_DATA_PATH.exists():
        print(f"  ERROR: SFT dataset not found at {SFT_DATA_PATH}")
        print(f"  Run:  cd sft_pipeline && python phase3_format_sft.py")
        sys.exit(1)

    rows = []
    with SFT_DATA_PATH.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"  Loaded {len(rows)} SFT training examples")
    dataset = Dataset.from_list(rows)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=20,
        max_length=1024,
        dataloader_num_workers=4 if device_type == "cuda" else 0,
        report_to="none",
    )

    print(f"  Loading tokenizer: {hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {hf_model}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
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

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n  SFT model saved to: {final_path}")
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="SFT training for Ops Agent (GPU-optimized, accelerate-compatible)")
    parser.add_argument("--model",      default="qwen2.5:1.5b", help="Model name (Ollama tag or HF ID)")
    parser.add_argument("--epochs",     type=int,   default=3,    help="Training epochs")
    parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=8,    help="Per-device batch size")
    args = parser.parse_args()

    run_sft(args.model, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
