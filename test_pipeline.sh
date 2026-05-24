#!/usr/bin/env bash
# ============================================================
#  Ops Agent — Training Pipeline Smoke Test v4
#
#  Tests training effectiveness with GRPO building on SFT.
#  Uses 10 scenarios for enough diversity to see improvement.
#
#  Usage: chmod +x test_pipeline.sh && ./test_pipeline.sh
# ============================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}━━━ $1 ━━━${NC}"; }
info()  { echo -e "${CYAN}  → $1${NC}"; }
warn()  { echo -e "${YELLOW}  ⚠ $1${NC}"; }
timer() { echo -e "${BOLD}  ⏱  $1${NC}"; }

now_s() { python3 -c "import time; print(time.time())"; }

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

TRAIN_MODEL="qwen2.5:1.5b"
NUM_SCENARIOS=10
SFT_EPOCHS=5
SFT_LR="1e-4"
GRPO_EPOCHS=2
GRPO_LR="2e-5"

TOTAL_START=$(now_s)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Ops Agent — Training Pipeline Smoke Test           ║"
echo "║  Model: $TRAIN_MODEL                                ║"
echo "║  Scenarios: $NUM_SCENARIOS, SFT: ${SFT_EPOCHS}ep (lr=${SFT_LR}), GRPO: ${GRPO_EPOCHS}ep   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ━━━ PHASE 0: Setup ━━━
step "PHASE 0: Setup"

info "Checking Python..."
python3 --version 2>&1 | head -1

info "Setting up venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

info "Installing dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -1
pip install -q -r requirements-training.txt 2>&1 | tail -1

# Clean old trained models to start fresh
info "Cleaning old trained models..."
rm -rf data/trained_models
rm -rf data/eval_results/direct_eval_*
mkdir -p data/trained_models data/eval_results

echo -e "  ${GREEN}✓ Setup OK${NC}"


# ━━━ PHASE 1: Generate scenarios ━━━
step "PHASE 1: Generate $NUM_SCENARIOS scenarios"

python3 app/eval/generate_scenarios.py --count $NUM_SCENARIOS --seed 42


# ━━━ PHASE 2: Eval BASE model ━━━
step "PHASE 2: Eval BASE $TRAIN_MODEL (before any training)"

info "Uses CPU for inference (avoids MPS tensor size bug)"

T0=$(now_s)
set +e
python3 app/training/train.py eval \
    --model $TRAIN_MODEL \
    --count $NUM_SCENARIOS
BASE_OK=$?
set -e
T1=$(now_s)
T_base_eval=$(python3 -c "print(round($T1-$T0, 1))")

if [ "$BASE_OK" -eq 0 ]; then
    timer "Base eval: ${T_base_eval}s"
else
    warn "Base eval FAILED"
fi


# ━━━ PHASE 3: SFT Training ━━━
step "PHASE 3: SFT Training ($SFT_EPOCHS epochs, lr=$SFT_LR)"

info "Higher LR and more epochs than default to ensure learning"

T0=$(now_s)
set +e
python3 app/training/train.py sft \
    --model $TRAIN_MODEL \
    --epochs $SFT_EPOCHS \
    --batch-size 2 \
    --lr $SFT_LR
SFT_OK=$?
set -e
T1=$(now_s)
T_sft=$(python3 -c "print(round($T1-$T0, 1))")

if [ "$SFT_OK" -eq 0 ]; then
    timer "SFT training: ${T_sft}s"
    echo -e "  ${GREEN}✓ SFT done${NC}"
else
    warn "SFT FAILED"
fi


# ━━━ PHASE 4: Eval SFT model ━━━
step "PHASE 4: Eval AFTER SFT"

SFT_PATH="data/trained_models/sft_$(echo $TRAIN_MODEL | tr ':' '_')/final"

SFT_EVAL_OK=1
if [ -d "$SFT_PATH" ]; then
    T0=$(now_s)
    set +e
    python3 app/training/train.py eval \
        --model-path "$SFT_PATH" \
        --count $NUM_SCENARIOS
    SFT_EVAL_OK=$?
    set -e
    T1=$(now_s)

    if [ "$SFT_EVAL_OK" -eq 0 ]; then
        timer "SFT eval: $(python3 -c "print(round($T1-$T0, 1))")s"
    else
        warn "SFT eval FAILED"
    fi
else
    warn "SFT model not found at $SFT_PATH"
fi


# ━━━ PHASE 5: GRPO Training (ON SFT MODEL) ━━━
step "PHASE 5: GRPO Training ON SFT model ($GRPO_EPOCHS epochs, lr=$GRPO_LR)"

info "GRPO builds on what SFT learned — not from scratch"

GRPO_OK=1
if [ -d "$SFT_PATH" ]; then
    T0=$(now_s)
    set +e
    python3 app/training/train.py grpo \
        --model "$SFT_PATH" \
        --epochs $GRPO_EPOCHS \
        --batch-size 4 \
        --lr $GRPO_LR
    GRPO_OK=$?
    set -e
    T1=$(now_s)
    T_grpo=$(python3 -c "print(round($T1-$T0, 1))")

    if [ "$GRPO_OK" -eq 0 ]; then
        timer "GRPO training: ${T_grpo}s"
        echo -e "  ${GREEN}✓ GRPO done${NC}"
    else
        warn "GRPO FAILED"
    fi
else
    warn "Skipping GRPO — SFT model not found"
fi


# ━━━ PHASE 6: Eval GRPO model ━━━
step "PHASE 6: Eval AFTER SFT+GRPO"

GRPO_PATH="data/trained_models/grpo_on_sft/final"

GRPO_EVAL_OK=1
if [ -d "$GRPO_PATH" ]; then
    T0=$(now_s)
    set +e
    python3 app/training/train.py eval \
        --model-path "$GRPO_PATH" \
        --count $NUM_SCENARIOS
    GRPO_EVAL_OK=$?
    set -e
    T1=$(now_s)

    if [ "$GRPO_EVAL_OK" -eq 0 ]; then
        timer "GRPO eval: $(python3 -c "print(round($T1-$T0, 1))")s"
    else
        warn "GRPO eval FAILED"
    fi
else
    warn "GRPO model not found at $GRPO_PATH"
fi


# ━━━ PHASE 7: Full comparison ━━━
step "PHASE 7: Before/After Comparison"

set +e
python3 app/training/train.py eval-compare \
    --model $TRAIN_MODEL \
    --count $NUM_SCENARIOS
set -e


# ━━━ Summary ━━━
TOTAL_END=$(now_s)
TOTAL_TIME=$(python3 -c "print(round($TOTAL_END-$TOTAL_START, 1))")

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              TEST SUMMARY                           ║"
echo "╠══════════════════════════════════════════════════════╣"

printf "║  %-40s %8s  ║\n" "Total time" "$(python3 -c "print(f'{$TOTAL_TIME/60:.1f}m')")"
printf "║  %-40s %8s  ║\n" "Scenarios" "$NUM_SCENARIOS"
echo "║                                                      ║"

checks="Base eval|$BASE_OK
SFT training|$SFT_OK
SFT eval|$SFT_EVAL_OK
GRPO training|$GRPO_OK
GRPO eval|$GRPO_EVAL_OK"

ALL_OK=true
while IFS='|' read -r name code; do
    if [ "$code" -eq 0 ]; then
        printf "║  ${GREEN}✓${NC} %-48s  ║\n" "$name"
    else
        printf "║  ${RED}✗${NC} %-48s  ║\n" "$name"
        ALL_OK=false
    fi
done <<< "$checks"

echo "║                                                      ║"
if [ "$ALL_OK" = true ]; then
    echo -e "║  ${GREEN}ALL PASSED${NC}                                        ║"
else
    echo -e "║  ${RED}SOME FAILED${NC} — check output above                 ║"
fi
echo "╚══════════════════════════════════════════════════════╝"
