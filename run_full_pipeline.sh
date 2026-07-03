#!/usr/bin/env bash

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

TRAIN_MODEL="qwen2.5:1.5b"
NUM_SCENARIOS=100
SFT_EPOCHS=3
SFT_LR="1e-4"
SFT_BATCH=2
GRPO_EPOCHS=2
GRPO_LR="2e-5"
GRPO_BATCH=4
SEED=42

LOG_FILE="$PROJ_DIR/full_run.log"
RESULTS_FILE="$PROJ_DIR/data/full_run_results.json"

now_s() { python3 -c "import time; print(time.time())"; }
now_human() { python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))"; }

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Ops Agent — Full Training & Evaluation Pipeline         ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Model:     $TRAIN_MODEL                                 ║"
echo "║  Scenarios: $NUM_SCENARIOS (seed=$SEED)                              ║"
echo "║  SFT:       ${SFT_EPOCHS} epochs, lr=${SFT_LR}, batch=${SFT_BATCH}                    ║"
echo "║  GRPO:      ${GRPO_EPOCHS} epochs, lr=${GRPO_LR}, batch=${GRPO_BATCH}                   ║"
echo "║  Started:   $(now_human)                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

PIPELINE_START=$(now_s)

echo "[$(now_human)] Setting up environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt 2>&1 | tail -1
pip install -q -r requirements-training.txt 2>&1 | tail -1

rm -rf data/trained_models
rm -rf data/eval_results/direct_eval_*
mkdir -p data/trained_models data/eval_results

echo "[$(now_human)] Setup complete."
echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 1: Generating $NUM_SCENARIOS scenarios..."
echo "================================================================"

python3 app/eval/generate_scenarios.py --count $NUM_SCENARIOS --seed $SEED

echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 2: Evaluating BASE model ($TRAIN_MODEL)..."
echo "================================================================"

T0=$(now_s)
python3 app/training/train.py eval \
    --model $TRAIN_MODEL \
    --count $NUM_SCENARIOS
T1=$(now_s)

echo ""
echo "[$(now_human)] Base eval done in $(python3 -c "print(f'{($T1-$T0)/60:.1f}')") minutes"
echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 3: SFT Training ($SFT_EPOCHS epochs, lr=$SFT_LR)..."
echo "================================================================"

T0=$(now_s)
python3 app/training/train.py sft \
    --model $TRAIN_MODEL \
    --epochs $SFT_EPOCHS \
    --batch-size $SFT_BATCH \
    --lr $SFT_LR
T1=$(now_s)

echo ""
echo "[$(now_human)] SFT training done in $(python3 -c "print(f'{($T1-$T0)/60:.1f}')") minutes"
echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 4: Evaluating SFT model..."
echo "================================================================"

SFT_PATH="data/trained_models/sft_$(echo $TRAIN_MODEL | tr ':' '_')/final"

T0=$(now_s)
python3 app/training/train.py eval \
    --model-path "$SFT_PATH" \
    --count $NUM_SCENARIOS
T1=$(now_s)

echo ""
echo "[$(now_human)] SFT eval done in $(python3 -c "print(f'{($T1-$T0)/60:.1f}')") minutes"
echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 5: GRPO Training on SFT model ($GRPO_EPOCHS epochs)..."
echo "================================================================"

T0=$(now_s)
python3 app/training/train.py grpo \
    --model "$SFT_PATH" \
    --epochs $GRPO_EPOCHS \
    --batch-size $GRPO_BATCH \
    --lr $GRPO_LR
T1=$(now_s)

echo ""
echo "[$(now_human)] GRPO training done in $(python3 -c "print(f'{($T1-$T0)/60:.1f}')") minutes"
echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 6: Evaluating SFT+GRPO model..."
echo "================================================================"

GRPO_PATH="data/trained_models/grpo_on_sft/final"

T0=$(now_s)
python3 app/training/train.py eval \
    --model-path "$GRPO_PATH" \
    --count $NUM_SCENARIOS
T1=$(now_s)

echo ""
echo "[$(now_human)] GRPO eval done in $(python3 -c "print(f'{($T1-$T0)/60:.1f}')") minutes"
echo ""


echo "================================================================"
echo "[$(now_human)] PHASE 7: Final comparison (re-evaluating all 3 models)..."
echo "================================================================"

python3 app/training/train.py eval-compare \
    --model $TRAIN_MODEL \
    --count $NUM_SCENARIOS


echo ""
echo "================================================================"
echo "[$(now_human)] PHASE 8: Generating results report..."
echo "================================================================"

PIPELINE_END=$(now_s)

python3 << PYEOF
import json, glob
from pathlib import Path

results_dir = Path("data/eval_results")
all_evals = {}

for f in sorted(results_dir.glob("direct_eval_*.json")):
    data = json.loads(f.read_text())
    all_evals[data["label"]] = data

pipeline = {
    "train_model": "$TRAIN_MODEL",
    "num_scenarios": $NUM_SCENARIOS,
    "seed": $SEED,
    "sft_epochs": $SFT_EPOCHS,
    "sft_lr": "$SFT_LR",
    "grpo_epochs": $GRPO_EPOCHS,
    "grpo_lr": "$GRPO_LR",
    "total_time_s": round($PIPELINE_END - $PIPELINE_START, 1),
    "total_time_human": f"{($PIPELINE_END - $PIPELINE_START)/60:.1f} minutes",
}

models = ["Base (no training)", "After SFT", "After SFT+GRPO"]
comparison = {}
for m in models:
    if m in all_evals:
        e = all_evals[m]
        comparison[m] = {
            "avg_reward": e["avg_reward"],
            "tool_accuracy": e["tool_accuracy"],
            "json_valid_rate": e["json_valid_rate"],
            "strong_pct": e["strong_reward_pct"],
            "positive_pct": e["positive_reward_pct"],
            "per_scenario_s": e["per_scenario_s"],
        }

if "Base (no training)" in comparison and "After SFT" in comparison:
    base = comparison["Base (no training)"]
    sft = comparison["After SFT"]
    pipeline["sft_improvement"] = {
        "reward_delta": round(sft["avg_reward"] - base["avg_reward"], 3),
        "reward_pct": round((sft["avg_reward"] - base["avg_reward"]) / max(0.001, abs(base["avg_reward"])) * 100, 1),
        "tool_acc_delta_pp": round((sft["tool_accuracy"] - base["tool_accuracy"]) * 100, 1),
        "strong_delta_pp": round((sft["strong_pct"] - base["strong_pct"]) * 100, 1),
        "speed_improvement_pct": round((1 - sft["per_scenario_s"] / max(0.1, base["per_scenario_s"])) * 100, 1),
    }

if "Base (no training)" in comparison and "After SFT+GRPO" in comparison:
    base = comparison["Base (no training)"]
    grpo = comparison["After SFT+GRPO"]
    pipeline["grpo_improvement"] = {
        "reward_delta": round(grpo["avg_reward"] - base["avg_reward"], 3),
        "reward_pct": round((grpo["avg_reward"] - base["avg_reward"]) / max(0.001, abs(base["avg_reward"])) * 100, 1),
        "tool_acc_delta_pp": round((grpo["tool_accuracy"] - base["tool_accuracy"]) * 100, 1),
        "strong_delta_pp": round((grpo["strong_pct"] - base["strong_pct"]) * 100, 1),
    }

pipeline["sft_training_curve"] = {
    "note": "Check full_run.log for per-epoch loss/accuracy progression"
}

output = {
    "pipeline": pipeline,
    "comparison": comparison,
    "all_evals": all_evals,
}

out_path = Path("$RESULTS_FILE")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(output, indent=2))
print(f"  Results saved to: {out_path}")

print()
print("╔═══════════════════════════════════════════════════════════════╗")
print("║                    FINAL RESULTS                             ║")
print("╠═══════════════════════════════════════════════════════════════╣")
print("║  Model                  Reward   ToolAcc  Strong   Speed    ║")
print("╠═══════════════════════════════════════════════════════════════╣")

for m in models:
    if m in comparison:
        c = comparison[m]
        name = m[:24]
        print(f"║  {name:<24s} {c['avg_reward']:>+6.3f}   {c['tool_accuracy']*100:>5.1f}%  {c['strong_pct']*100:>5.1f}%  {c['per_scenario_s']:>4.1f}s   ║")

print("╠═══════════════════════════════════════════════════════════════╣")

if "sft_improvement" in pipeline:
    s = pipeline["sft_improvement"]
    print(f"║  SFT improvement:                                           ║")
    print(f"║    Reward:     {s['reward_delta']:>+.3f} ({s['reward_pct']:>+.1f}%)                            ║")
    print(f"║    Tool acc:   {s['tool_acc_delta_pp']:>+.1f}pp                                       ║")
    print(f"║    Strong:     {s['strong_delta_pp']:>+.1f}pp                                       ║")
    print(f"║    Speed:      {s['speed_improvement_pct']:>+.1f}% faster                                ║")

if "grpo_improvement" in pipeline:
    g = pipeline["grpo_improvement"]
    print(f"║  SFT+GRPO improvement (vs base):                           ║")
    print(f"║    Reward:     {g['reward_delta']:>+.3f} ({g['reward_pct']:>+.1f}%)                            ║")
    print(f"║    Tool acc:   {g['tool_acc_delta_pp']:>+.1f}pp                                       ║")
    print(f"║    Strong:     {g['strong_delta_pp']:>+.1f}pp                                       ║")

print(f"║                                                             ║")
print(f"║  Total pipeline time: {pipeline['total_time_human']:>30s}      ║")
print("╠═══════════════════════════════════════════════════════════════╣")
print("║                                                             ║")
print("║  RESUME BULLET POINTS (fill in your actual numbers):        ║")
print("║                                                             ║")

base_r = comparison.get("Base (no training)", {})
sft_r = comparison.get("After SFT", {})
grpo_r = comparison.get("After SFT+GRPO", {})

if base_r and sft_r:
    ba = base_r["tool_accuracy"]*100
    sa = sft_r["tool_accuracy"]*100
    ga = grpo_r.get("tool_accuracy", sa)*100
    br = base_r["avg_reward"]
    sr = sft_r["avg_reward"]
    bs = base_r["strong_pct"]*100
    ss = sft_r["strong_pct"]*100
    gs = grpo_r.get("strong_pct", ss)*100
    rdelta = round((sr-br)/max(0.001,abs(br))*100, 0)

    print(f"║  1. \"Implemented SFT + GRPO training pipeline with       ║")
    print(f"║     automatic reward function. SFT improved planner      ║")
    print(f"║     reward by {rdelta:.0f}% and strong completion rate from     ║")
    print(f"║     {bs:.0f}% to {ss:.0f}% on {$NUM_SCENARIOS}-scenario benchmark.\"               ║")
    print(f"║                                                             ║")
    print(f"║  2. \"Trained on {$NUM_SCENARIOS} procedurally generated incident       ║")
    print(f"║     scenarios across 6 root cause categories. Training    ║")
    print(f"║     loss converged from 3.6 to <0.1 in 25 steps.\"        ║")
    print(f"║                                                             ║")
    print(f"║  3. \"Fine-tuned model produces correctly formatted tool   ║")
    print(f"║     calls with {ss:.0f}% strong reward rate vs {bs:.0f}%           ║")
    print(f"║     baseline, and {sft_r.get('per_scenario_s',0):.0f}s inference vs {base_r.get('per_scenario_s',0):.0f}s baseline.\"  ║")

print("║                                                             ║")
print("╚═══════════════════════════════════════════════════════════════╝")
PYEOF

echo ""
echo "================================================================"
echo "[$(now_human)] PIPELINE COMPLETE"
echo "================================================================"
echo ""
echo "  Results:  $RESULTS_FILE"
echo "  Log:      $LOG_FILE"
echo ""
echo "  Total time: $(python3 -c "
s = $PIPELINE_END - $PIPELINE_START
print(f'{int(s//3600)}h {int((s%3600)//60)}m {int(s%60)}s')
")"
echo ""
