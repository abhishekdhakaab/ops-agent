# Setup Guide — New Laptop

Everything you need, in order. Copy-paste each block.

## Step 1: Prerequisites

You need Python 3.10+ and Ollama installed.

```bash
# Check Python
python3 --version    # needs 3.10+

# Install Ollama (if not installed)
# Mac: https://ollama.ai/download
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (Mac — open the app, or:)
ollama serve
```

## Step 2: Unzip and Setup

```bash
unzip ops_agent_fixed.zip
cd ops_agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 3: Pull Models

```bash
# Required (pick one LLM + the embedding model):
ollama pull gemma3:4b         # recommended default (~3GB)
ollama pull all-minilm         # for RAG embeddings (~50MB)

# Optional (for model comparison):
ollama pull gemma3:1b          # baseline — small and bad (~1GB)
ollama pull qwen2.5:7b         # better — if you have 16GB RAM (~5GB)
```

## Step 4: Start the Server

```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

Leave this running in one terminal. Open a second terminal for the next steps.

## Step 5: Ingest Runbooks (one time)

```bash
curl -X POST http://localhost:8000/runbooks/ingest
```

You should see: `{"docs": 10, "chunks": 12, "embeddings": true}`

If embeddings is false, that's okay — keyword search will work as fallback.

## Step 6: Test It

```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is order-svc slow?"}' | python3 -m json.tool
```

You should see a JSON response with reasoning, tools_used, evidence, and final_answer.

Try a few more:
```bash
# Should detect upstream dependency (NOT deploy)
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"question": "Checkout-svc has errors. What is going on?"}' | python3 -m json.tool

# Should say healthy
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"question": "Is shipping-calc having any issues?"}' | python3 -m json.tool
```

## Step 7: Run Evaluation

```bash
# Quick test (10 questions, no LLM judge — fast)
cd app/eval
python run_eval.py --count 10 --no-judge

# Full eval (100 questions, with LLM judge — takes a while)
python run_eval.py

# Compare two models (automated — restarts server for each model)
python run_eval.py --compare gemma3:1b gemma3:4b

# View past results
python run_eval.py --report
```

## Step 8: Training (optional)

```bash
# Install training dependencies (heavy — only when you want to train)
pip install -r requirements-training.txt

# Test the reward function first
python app/training/train.py reward-test

# Run SFT
python app/training/train.py sft --model gemma3:1b --epochs 3

# Run GRPO (RL)
python app/training/train.py grpo --model gemma3:1b --epochs 2

# Full pipeline (SFT then GRPO)
python app/training/train.py full --model gemma3:1b
```

## Step 9: Generate More/Harder Scenarios

```bash
# Regenerate 100 scenarios (default)
python app/eval/generate_scenarios.py --count 100

# Generate 200 hard scenarios (more red herrings)
python app/eval/generate_scenarios.py --count 200 --hard

# Different random seed for variety
python app/eval/generate_scenarios.py --count 100 --seed 99
```

## Troubleshooting

**"Connection refused" on /run**: Ollama isn't running. Start it with `ollama serve`.

**"LLM did not return valid JSON after retries"**: The model is too small. Try a bigger one: `LLM_MODEL=gemma3:4b` in .env, then restart the server.

**Server won't start**: Make sure you're in the `ops_agent` directory and venv is activated.

**Slow responses**: Normal for the first request (model loading). Subsequent requests are faster. Expect 10-40s per question depending on model size.
