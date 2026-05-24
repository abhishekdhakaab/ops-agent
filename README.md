# Autonomous AI Operations Agent

An autonomous multi-agent system that investigates operational incidents by reasoning step-by-step, calling tools to gather evidence, and producing grounded explanations. Built with LangGraph, FastAPI, and local LLMs via Ollama.

## Architecture

```
User Question
    |
    v
[Intake] --> [Retrieval Agent] --> [Planner LLM] --> [Act: Tool Call]
                                        ^                   |
                                        |___ loop __________|
                                                            |
                                                            v
                                    [Draft Answer LLM] --> [Validation Agent LLM]
                                                            |
                                              grounded? ----|---- not grounded?
                                              |                      |
                                              v                      v
                                    [Action Agent LLM]          back to Planner
                                              |
                                              v
                                         [Finalize]
                                              |
                                              v
                                    Structured Response
                                    + Incident Ticket (optional)
```

**4 LLM calls per pass**: Planner, Draft, Validator, Action Agent  
**3 tools**: Metrics, Logs, Deployments (+ Ticket creation)  
**RAG**: Runbook retrieval with embeddings or keyword fallback  
**Validation loop**: Re-investigates if draft answer lacks evidence grounding  

## Quick Start (New Laptop Setup)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### Step 1: Install Ollama models
```bash
# Required - pick one (or all for comparison):
ollama pull gemma3:4b          # ~3GB, good balance (recommended start)
ollama pull gemma3:1b          # ~1GB, fast but weak at JSON
ollama pull qwen2.5:7b         # ~5GB, strong JSON/tool-use
ollama pull phi4:14b           # ~9GB, best quality (needs 16GB+ RAM)

# Required for RAG embeddings:
ollama pull all-minilm         # ~50MB, fast embeddings
```

### Step 2: Setup Python environment
```bash
cd ops_agent
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify .env
The `.env` file is in the project root. Default config:
```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=gemma3:4b
MAX_STEPS=6

EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=all-minilm
```

### Step 4: Ingest runbooks (one-time)
```bash
# Start the server first
uvicorn app.main:app --reload

# In another terminal, ingest the runbook documents:
curl -X POST http://localhost:8000/runbooks/ingest
```

### Step 5: Test it
```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is service-x slow?"}' | python3 -m json.tool
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/run` | Run agent investigation (synchronous) |
| POST | `/runs` | Start async investigation (returns run_id) |
| GET | `/runs` | List recent runs |
| GET | `/runs/{run_id}` | Get run status/result |
| GET | `/tools` | List available tools and schemas |
| POST | `/tools/{name}` | Call a tool directly |
| GET | `/tickets` | List created incident tickets |
| POST | `/runbooks/ingest` | Ingest runbook documents for RAG |
| GET | `/health` | Health check |

## Running Evaluations

### Single model eval
```bash
# With server running:
cd app/eval
python run_eval.py
```

### Compare multiple models (automated)
```bash
cd app/eval
python run_eval.py --compare gemma3:1b gemma3:4b qwen2.5:7b
```
This will:
1. Update `.env` for each model
2. Start/stop the server automatically
3. Run all 15 eval questions per model
4. Print a comparison table

### View past results
```bash
python run_eval.py --report
```

### Eval Metrics
| Metric | What it measures |
|--------|-----------------|
| Tool Selection Accuracy | Did the agent use the right tools? |
| Tool Any Accuracy | Did it use at least one correct tool? |
| Tool Diversity | Ratio of unique/total calls (1.0 = no repeats) |
| Efficiency Score | Were tool counts in expected range? |
| Grounding Score | Does the answer reference actual evidence? |
| Answer Quality | Is the answer well-structured and sized? |

## Training Data Pipeline

The agent logs every run to `data/trajectories.jsonl`. Convert to training format:

```bash
python app/eval/convert_for_training.py
```

Generates:
- `data/trl_planner_sft.jsonl` — SFT training data (prompt → tool choice)
- `data/trl_planner_grpo.jsonl` — GRPO/RL training data (with expected actions)

## Project Structure

```
ops_agent/
├── .env                        # LLM and embedding config
├── requirements.txt
├── app/
│   ├── main.py                 # FastAPI endpoints
│   ├── agent/
│   │   ├── graph.py            # LangGraph pipeline (7 nodes)
│   │   ├── prompts.py          # All LLM system prompts
│   │   └── retrieval_stub.py   # Fallback runbook data
│   ├── core/
│   │   ├── config.py           # Settings from .env
│   │   ├── types.py            # Request/Response models
│   │   ├── storage.py          # Trajectory logging
│   │   ├── run_store.py        # Async run management
│   │   ├── runner.py           # Blocking agent runner
│   │   └── metadata.py         # Version fingerprinting
│   ├── eval/
│   │   ├── questions.json      # 15 eval scenarios
│   │   ├── run_eval.py         # Multi-model eval runner
│   │   └── convert_for_training.py
│   ├── llm/
│   │   ├── base.py             # LLMClient ABC
│   │   ├── http_provider.py    # OpenAI-compatible provider
│   │   ├── ollama_provider.py  # Native Ollama provider
│   │   ├── json_utils.py       # JSON extraction/repair
│   │   └── providers.py        # Provider factory
│   ├── rag/
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── index.py            # Chunking and index I/O
│   │   ├── ingest.py           # Runbook ingestion
│   │   └── retrieve.py         # Cosine + keyword retrieval
│   └── tools/
│       ├── contracts.py        # Pydantic I/O schemas
│       ├── impl.py             # Tool implementations
│       ├── registry.py         # MCP-style tool registry
│       └── ticket_store.py     # Incident ticket creation
├── data/
│   ├── runbooks/               # 10 operational runbooks
│   ├── rag_index.json          # Pre-built RAG index
│   ├── trajectories.jsonl      # Agent run logs
│   ├── tickets.jsonl           # Created tickets
│   └── eval_results/           # Model comparison results
└── gemma-agent/
    └── Modelfile               # Custom Ollama model config
```

## Key Design Decisions

- **LangGraph over plain loops**: Gives us conditional edges (validation loop), state management, and graph visualization for free
- **Pydantic contracts for tools**: Every tool has typed input/output — the registry validates automatically
- **JSON extraction with retry**: Small models often wrap JSON in markdown; `json_utils.py` strips fences and retries
- **Validation agent loop**: Catches hallucinated conclusions by checking if claims have evidence backing
- **Trajectory logging**: Every run is logged with metadata (model, prompt version, graph version) for reproducibility
