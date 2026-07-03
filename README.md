# Ops Agent

Ops Agent is an evidence-grounded incident investigation system built around a small local language model. It plans diagnostic steps, calls simulated metrics/log/deployment tools, retrieves relevant runbooks, checks its own conclusions for grounding, and produces a structured response with an optional incident ticket.

The project also includes the data pipeline and experiments used to fine-tune Qwen2.5-1.5B with SFT, GRPO, and DPO. The strongest policy improved full-trajectory accuracy from 18% to 67% while sharply reducing repeated tool calls.

## What is technically interesting

- A LangGraph state machine separates planning, evidence collection, drafting, validation, and action selection.
- Pydantic contracts validate every tool boundary and structured model response.
- The validator can send an ungrounded draft back through the planner for one more evidence-gathering pass.
- Runbook retrieval uses embeddings when available and a keyword fallback when they are not.
- The training pipeline materializes planner state at each step, preventing future evidence from leaking into next-action labels.
- Evaluation covers both first-tool selection and complete multi-step trajectories, including completion, coverage, efficiency, and redundancy.

## Architecture

```text
Question
   |
   v
Intake -> Runbook retrieval -> Planner -> Tool execution
                                  ^             |
                                  |_____________|
                                      loop
                                                 |
                                                 v
Draft answer -> Grounding validator -- missing evidence --> Planner
                       |
                    grounded
                       v
Structured action -> Optional ticket -> Final response
```

The API in `app/main.py` exposes the graph synchronously and as a background job. Tool responses come from reproducible incident scenarios in `data/scenarios.json`; the same evidence model is reused by the training pipeline.

## Stack

- Python 3.10+
- FastAPI and Uvicorn
- LangGraph
- Pydantic
- Ollama or another OpenAI-compatible endpoint
- PyTorch, Transformers, PEFT, TRL, Accelerate, and Datasets for training

## Local setup

Install [Ollama](https://ollama.com/), then clone the repository and create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Pull the default local models:

```bash
ollama pull gemma3:4b
ollama pull all-minilm
```

The defaults in `.env.example` target Ollama on localhost. For another OpenAI-compatible service, set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`; set the corresponding embedding variables if that service also supplies embeddings.

Start the API:

```bash
uvicorn app.main:app --reload
```

Build the runbook index once the server is running:

```bash
curl -X POST http://127.0.0.1:8000/runbooks/ingest
```

Run an investigation:

```bash
curl -s -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"question":"Did a deployment cause the checkout-svc latency spike?"}' \
  | python3 -m json.tool
```

`setup.sh` performs the same local setup interactively and finishes with a smoke test.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Process health check |
| `POST` | `/run` | Run an investigation synchronously |
| `POST` | `/runs` | Queue a background investigation |
| `GET` | `/runs` | List recent background runs |
| `GET` | `/runs/{run_id}` | Poll one background run |
| `GET` | `/tools` | Inspect tool schemas |
| `POST` | `/tools/{tool_name}` | Invoke one tool directly |
| `GET` / `POST` | `/tickets` | List or create incident tickets |
| `POST` | `/runbooks/ingest` | Rebuild the retrieval index |

Runtime JSONL files and the generated RAG index are intentionally ignored by Git.

## Training and evaluation

Training dependencies are separate from the API environment:

```bash
pip install -r requirements-training.txt
```

Useful entry points:

```bash
# Check the three-stage data pipeline without calling a model
python sft_pipeline/run_pipeline.py --dry-run

# Generate the SFT dataset with Ollama
python sft_pipeline/run_pipeline.py --model gemma3:12b --clean

# Train SFT on one GPU (use accelerate for multi-GPU)
python sft_pipeline/run_sft_gpu.py --model qwen2.5:1.5b

# Generate prompts and train GRPO on the merged SFT model
python sft_pipeline/grpo_train.py full --model qwen2.5:1.5b

# Compare complete trajectories across available checkpoints
python full_traj_eval.py --compare --count 100

# Compare first-step decisions on generated unseen services
python unseen_eval.py --count 100
```

`setup_gpu.sh` configures a Linux CUDA environment. `run_overnight.sh` runs the SFT, GRPO, DPO, and evaluation stages on available GPUs while avoiding devices already in use.

Model checkpoints are excluded because they are too large for a normal source repository. Evaluation commands expect them under `data/trained_models/`.

## Results

All headline results use Qwen2.5-1.5B-Instruct and 100 scenarios. The detailed records are in `data/eval_results/`.

### Full investigation trajectories

This benchmark runs the planner until it chooses `final` or reaches the five-step limit.

| Model | Trajectory accuracy | Required-tool coverage | No repeated calls | Completion | Average steps |
|---|---:|---:|---:|---:|---:|
| Base | 18% | 64% | 30% | 31% | 4.77 |
| SFT | 46% | 73% | 72% | 72% | 3.18 |
| GRPO | **67%** | **74%** | **93%** | **94%** | **2.48** |
| DPO | 40% | 71% | 66% | 66% | 3.36 |

GRPO improved trajectory accuracy by 49 percentage points over the base model. Repeated-call scenarios fell from 70% to 7%, and investigations used 2.29 fewer tool calls on average.

### First action on unseen services

The final four-model run evaluates the first decision before any evidence is available.

| Model | Tool accuracy | Average reward | Valid JSON |
|---|---:|---:|---:|
| Base | 54% | 2.172 | 100% |
| SFT | **67%** | **2.269** | 100% |
| GRPO | 65% | 2.255 | 100% |
| DPO | 65% | 2.255 | 100% |

### Run variance retained for transparency

Two earlier SFT-only runs produced different tradeoffs, so both were retained instead of selecting only the more flattering score.

| Benchmark | Run | SFT tool accuracy | Average reward | Valid JSON |
|---|---|---:|---:|---:|
| Unseen first action | Earlier SFT-only | 75% | 2.185 | 95% |
| Unseen first action | Final four-model comparison | 67% | 2.269 | 100% |
| In-distribution first action | Earlier SFT-only | 63% | 2.213 | 99% |
| In-distribution first action | Later rerun | 48% | 2.136 | 100% |

The in-distribution benchmark reuses training scenarios and should not be treated as a generalization result. The full-trajectory and unseen-service evaluations are the more useful measures of behavior.

## Repository map

```text
app/
  agent/        LangGraph workflow and role prompts
  core/         configuration, metadata, storage, and run lifecycle
  eval/         API evaluation, scoring, and scenario generation
  llm/          Ollama and OpenAI-compatible clients
  rag/          runbook ingestion and retrieval
  tools/        typed simulated operations tools
  training/     original SFT/GRPO training entry point
sft_pipeline/   staged data generation and current GPU training pipeline
data/
  eval_results/ retained benchmark records
  runbooks/     retrieval corpus
  scenarios.json
  trl_planner_sft.jsonl
```

## Scope

This is a research/demo system, not an automated production responder. Its tools operate on generated incident scenarios, ticket creation writes to a local JSONL file, and recommendations should be reviewed by an operator before any real action is taken.
