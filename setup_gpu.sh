#!/bin/bash
# =============================================================================
# GPU Environment Setup — Ops Agent Training
# Run once on a fresh machine before running run_overnight.sh
# Tested with: 8x A100/H100, CUDA 12.1+, Ubuntu 20.04/22.04
# =============================================================================
set -e

echo ""
echo "============================================================"
echo "  Ops Agent — GPU Environment Setup"
echo "============================================================"
echo ""

# ── 1. Check GPU ─────────────────────────────────────────────────
echo "[1/6] Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install CUDA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Found $GPU_COUNT GPU(s)"

# ── 2. Conda ──────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up conda environment..."
if ! command -v conda &> /dev/null; then
    echo "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
fi

ENV_NAME="ops_agent"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Conda env '$ENV_NAME' already exists — skipping creation"
else
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "  Created conda env: $ENV_NAME"
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 3. PyTorch (CUDA) ─────────────────────────────────────────────
echo ""
echo "[3/6] Installing PyTorch with CUDA 12.1..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── 4. ML Training dependencies ───────────────────────────────────
echo ""
echo "[4/6] Installing training dependencies..."
pip install --quiet \
    transformers==4.57.6 \
    trl==1.1.0 \
    peft==0.19.1 \
    accelerate==1.13.0 \
    datasets==4.8.3 \
    tokenizers==0.22.2 \
    bitsandbytes \
    sentencepiece \
    scipy \
    einops

# Flash Attention (optional but major speedup on A100/H100)
echo "  Installing flash-attn (this takes a few minutes)..."
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (not supported on this GPU — OK)"

# ── 5. App dependencies ───────────────────────────────────────────
echo ""
echo "[5/6] Installing app dependencies..."
pip install --quiet \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.8.2 \
    httpx==0.27.0 \
    python-dotenv==1.0.1 \
    "langgraph==0.2.44"

# ── 6. Accelerate config ──────────────────────────────────────────
echo ""
echo "[6/6] Configuring accelerate for $GPU_COUNT GPU(s)..."
mkdir -p ~/.cache/huggingface/accelerate

# Write accelerate config dynamically based on actual GPU count
cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $GPU_COUNT
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "  Accelerate configured for $GPU_COUNT GPUs with bf16"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next step:"
echo "    conda activate $ENV_NAME"
echo "    bash run_overnight.sh"
echo "============================================================"
echo ""
