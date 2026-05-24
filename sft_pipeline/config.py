"""
Configuration for the SFT data generation pipeline.

All paths, model settings, and pipeline parameters live here.
Change OLLAMA_MODEL to whatever large model you have pulled locally.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent                      # ops_agent(new changes)/
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"

# Pipeline outputs (all intermediate + final artifacts go here)
OUTPUT_DIR = BASE_DIR / "output"
RAW_RUNS_PATH = OUTPUT_DIR / "phase1_raw_runs.jsonl"
SYNTHESIZED_PATH = OUTPUT_DIR / "phase2_synthesized.jsonl"
SFT_DATASET_PATH = OUTPUT_DIR / "sft_dataset.jsonl"
STATS_PATH = OUTPUT_DIR / "pipeline_stats.json"

# ── Ollama settings ───────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma2:9b"           # ← change to your model

# ── Phase 1: Raw runs ────────────────────────────────────────────
RUNS_PER_SCENARIO = 3                 # how many investigation attempts per scenario
MAX_STEPS_PER_RUN = 5                 # cap on tool calls per run (excludes 'final')
TEMPERATURE_PHASE1 = 0.7             # higher = more diverse runs
NUM_PREDICT_PHASE1 = 300             # max tokens per model response

# ── Phase 2: Synthesis ────────────────────────────────────────────
TEMPERATURE_PHASE2 = 0.2             # lower = more consistent synthesis
NUM_PREDICT_PHASE2 = 600             # synthesis outputs are longer

# ── Phase 3: Formatting ──────────────────────────────────────────
MAX_FINAL_RATIO = 0.30               # cap 'final' examples at 30% of dataset

# ── Retry / robustness ───────────────────────────────────────────
MAX_RETRIES = 3                       # retries per LLM call on parse failure
RETRY_DELAY_S = 1.0                   # seconds between retries
