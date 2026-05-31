# Ops Agent — Final Evaluation Results

Model: **Qwen2.5-1.5B-Instruct**  
Training data: 320 SFT examples  
GPU: Multi-GPU cluster (A100/H100, CUDA 12.4)  

---

## 1. Full Trajectory Evaluation (100 scenarios, end-to-end)

The agent runs through ALL investigation steps — picks a tool, sees simulated
output, picks next tool, repeats until it says "final" or hits the step limit.
This is the correct eval for multi-step planning.

| Model | Traj Accuracy | Coverage | No Redundancy | Completion | Avg Steps |
|-------|-------------|----------|--------------|-----------|----------|
| Base  | 18%         | 64%      | 30%          | 31%       | 4.77     |
| SFT   | 46%         | 73%      | 72%          | 72%       | 3.18     |
| GRPO  | **67%**     | **74%**  | **93%**      | **94%**   | **2.48** |
| DPO   | 40%         | 71%      | 66%          | 66%       | 3.36     |

- **Base → SFT**: +28pp trajectory accuracy
- **Base → GRPO**: +49pp trajectory accuracy (+21pp on top of SFT)
- **GRPO redundancy**: dropped from 70% repeated tool calls → 7%
- **GRPO efficiency**: average steps 4.77 → 2.48

---

## 2. First-Step Accuracy on 100 Unseen Scenarios

Evaluates only the first tool call with no evidence. Services are completely
new — never seen during training.

| Model | Tool Accuracy | Avg Reward | JSON Valid |
|-------|-------------|-----------|-----------|
| Base  | 54%         | +2.172    | 100%      |
| SFT   | **67%**     | **+2.269**| **100%**  |
| GRPO  | 65%         | +2.255    | 100%      |
| DPO   | 65%         | +2.255    | 100%      |

- **Base → SFT**: +13pp on never-seen services
- GRPO/DPO match SFT here because this metric only measures step 0
  (GRPO's real gains show in the full trajectory eval above)

---

## 3. Per Root-Cause Breakdown — SFT on Unseen

| Category            | Base Accuracy | SFT Accuracy |
|---------------------|-------------|-------------|
| deploy_regression   | 29%         | **100%**    |
| config_change       | 100%        | 100%        |
| healthy             | 100%        | 100%        |
| resource_exhaustion | 94%         | 71%         |
| upstream_dependency | 0%          | 24%         |
| infrastructure      | 0%          | 0%          |

---

## 4. SFT Training Convergence

Epochs: 3 | Batch: 8 per GPU | LR: 2e-5 | LoRA r=32

| Epoch | Loss  |
|-------|-------|
| 1     | 2.724 |
| 2     | 2.315 |
| 3     | 1.851 |

---

## 5. GRPO Training — Reward Progression

Epochs: 3 | 222 steps | LR: 5e-6 | beta=0.1 (KL penalty)

| Step | Mean Reward | KL Divergence |
|------|------------|--------------|
| 5    | 1.24       | 0.0004       |
| 75   | 1.45       | 0.0008       |
| 150  | 1.57       | 0.0018       |
| 222  | 1.57       | 0.0023       |

KL stayed low throughout — no catastrophic drift from SFT policy.

---

## Resume Summary

> Fine-tuned Qwen2.5-1.5B on 320 domain-specific SFT examples using a
> 3-stage pipeline (SFT → GRPO → DPO). On full end-to-end trajectory
> evaluation across 100 scenarios: trajectory accuracy improved from
> **18% (base) → 46% (SFT) → 67% (GRPO)**, with tool redundancy dropping
> from 70% → 7% and average investigation steps from 4.77 → 2.48.
> On 100 held-out unseen services: first-tool accuracy improved from
> **54% → 67%** with 100% JSON validity maintained.
