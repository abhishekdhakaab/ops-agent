#!/bin/bash
# =============================================================================
# Overnight GPU Training + Evaluation Pipeline — Ops Agent
#
# Runs on a single machine with 1–8 GPUs.
# Trains: SFT → GRPO (on SFT) → DPO (on SFT)
# Evaluates: all models on 100 in-distribution + 100 unseen scenarios
#
# Usage:
#   conda activate ops_agent
#   bash run_overnight.sh                          # Qwen2.5-1.5B (default)
#   bash run_overnight.sh qwen2.5:7b               # Qwen2.5-7B
#   bash run_overnight.sh qwen2.5:1.5b qwen2.5:7b  # Both models back-to-back
# =============================================================================
set -e

# ── Args ──────────────────────────────────────────────────────────
MODELS=("${@}")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("qwen2.5:1.5b")
fi

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/data/eval_results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── GPU Check ─────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo ""
echo "============================================================"
echo "  Ops Agent — Overnight GPU Pipeline"
echo "  GPUs detected: $GPU_COUNT"
echo "  Models to train: ${MODELS[*]}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "WARNING: No GPUs detected — falling back to CPU. Training will be slow."
    LAUNCH_CMD="python3"
else
    # Use accelerate for multi-GPU; single GPU also works
    LAUNCH_CMD="accelerate launch --config_file $SCRIPT_DIR/accelerate_config.yaml"
    # Override num_processes with actual GPU count
    LAUNCH_CMD="accelerate launch --num_processes $GPU_COUNT --mixed_precision bf16"
fi

# ── Helper ────────────────────────────────────────────────────────
run_step() {
    local STEP="$1"
    local CMD="$2"
    local LOG="$3"
    echo ""
    echo "------------------------------------------------------------"
    echo "  STEP: $STEP"
    echo "  $(date)"
    echo "------------------------------------------------------------"
    eval "$CMD" 2>&1 | tee "$LOG_DIR/$LOG.log"
    echo "  Done: $STEP ($(date))"
}

# ─────────────────────────────────────────────────────────────────
# PHASE 0 — Verify training data exists
# ─────────────────────────────────────────────────────────────────
echo "[Phase 0] Checking training data..."
SFT_DATA="$SCRIPT_DIR/sft_pipeline/output/sft_dataset.jsonl"
GRPO_DATA="$SCRIPT_DIR/sft_pipeline/output/grpo_prompts.jsonl"

if [ ! -f "$SFT_DATA" ]; then
    echo "  sft_dataset.jsonl not found — regenerating from phase2_synthesized..."
    cd "$SCRIPT_DIR/sft_pipeline"
    python3 phase3_format_sft.py
    cd "$SCRIPT_DIR"
fi

if [ ! -f "$GRPO_DATA" ]; then
    echo "  grpo_prompts.jsonl not found — regenerating..."
    cd "$SCRIPT_DIR/sft_pipeline"
    python3 grpo_train.py generate
    cd "$SCRIPT_DIR"
fi

SFT_COUNT=$(wc -l < "$SFT_DATA")
GRPO_COUNT=$(wc -l < "$GRPO_DATA")
echo "  SFT examples: $SFT_COUNT"
echo "  GRPO prompts: $GRPO_COUNT"

# ─────────────────────────────────────────────────────────────────
# Per-model training loop
# ─────────────────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    MODEL_SAFE="${MODEL//:/_}"
    MODEL_SAFE="${MODEL_SAFE//\//_}"
    echo ""
    echo "============================================================"
    echo "  MODEL: $MODEL"
    echo "============================================================"

    # ── Phase 1: SFT ──────────────────────────────────────────────
    SFT_FINAL="$SCRIPT_DIR/data/trained_models/sft_${MODEL_SAFE}/final"

    if [ -d "$SFT_FINAL" ]; then
        echo ""
        echo "  [SFT] Checkpoint found at $SFT_FINAL — skipping training."
        echo "  Delete the folder to retrain: rm -rf $SFT_FINAL"
    else
        run_step "SFT training ($MODEL)" \
            "cd $SCRIPT_DIR && $LAUNCH_CMD sft_pipeline/run_sft_gpu.py --model $MODEL --epochs 3 --lr 2e-5 --batch-size 8" \
            "sft_${MODEL_SAFE}"
    fi

    # ── Phase 2: Merge SFT adapter into base (needed by GRPO + DPO) ──
    MERGED_PATH="$SCRIPT_DIR/data/trained_models/sft_${MODEL_SAFE}/merged_for_grpo"
    if [ ! -d "$MERGED_PATH" ]; then
        run_step "Merging SFT weights ($MODEL)" \
            "cd $SCRIPT_DIR/sft_pipeline && python3 grpo_train.py merge --model $MODEL" \
            "merge_${MODEL_SAFE}"
    else
        echo ""
        echo "  [Merge] Already done at $MERGED_PATH — skipping."
    fi

    # ── Phase 3: GRPO on SFT ──────────────────────────────────────
    GRPO_FINAL="$SCRIPT_DIR/data/trained_models/grpo_on_sft_v2/final"

    if [ -d "$GRPO_FINAL" ]; then
        echo ""
        echo "  [GRPO] Checkpoint found — skipping. Delete to retrain: rm -rf $GRPO_FINAL"
    else
        run_step "GRPO training ($MODEL, on SFT)" \
            "cd $SCRIPT_DIR/sft_pipeline && $LAUNCH_CMD grpo_train.py train --model $MODEL --epochs 3 --lr 5e-6 --batch-size 4 --num-generations 8" \
            "grpo_${MODEL_SAFE}"
    fi

    # ── Phase 4: DPO on SFT ───────────────────────────────────────
    DPO_FINAL="$SCRIPT_DIR/data/trained_models/dpo_on_sft_v1/final"

    if [ -d "$DPO_FINAL" ]; then
        echo ""
        echo "  [DPO] Checkpoint found — skipping. Delete to retrain: rm -rf $DPO_FINAL"
    else
        run_step "DPO training ($MODEL, on SFT)" \
            "cd $SCRIPT_DIR/sft_pipeline && $LAUNCH_CMD grpo_train.py dpo --model $MODEL --epochs 2 --lr 5e-6 --batch-size 8" \
            "dpo_${MODEL_SAFE}"
    fi

done  # end model loop

# ─────────────────────────────────────────────────────────────────
# PHASE 5 — Evaluation (single GPU, no distributed needed)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PHASE 5: Evaluation"
echo "============================================================"

# Use single GPU for eval to avoid complexity
export CUDA_VISIBLE_DEVICES=0

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "  --- In-distribution eval ($MODEL, 100 scenarios) ---"
    run_step "sft_eval 100 ($MODEL)" \
        "cd $SCRIPT_DIR && python3 sft_eval.py --model $MODEL --count 100 --compare" \
        "eval_indist_${MODEL//:/_}"

    echo ""
    echo "  --- Unseen eval ($MODEL, 30 scenarios quick) ---"
    run_step "unseen_eval 30 ($MODEL)" \
        "cd $SCRIPT_DIR && python3 unseen_eval.py --model $MODEL --count 30" \
        "eval_unseen30_${MODEL//:/_}"

    echo ""
    echo "  --- Unseen eval ($MODEL, 100 scenarios full) ---"
    run_step "unseen_eval 100 ($MODEL)" \
        "cd $SCRIPT_DIR && python3 unseen_eval.py --model $MODEL --count 100" \
        "eval_unseen100_${MODEL//:/_}"
done

# ─────────────────────────────────────────────────────────────────
# DONE — Print summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo ""
echo "  Results saved to: $RESULTS_DIR"
echo ""
echo "  Key files:"
ls "$RESULTS_DIR"/*.json 2>/dev/null | xargs -I{} basename {} | grep -E "in_distribution|unseen" | sort
echo ""
echo "  Logs: $LOG_DIR"
echo "============================================================"
