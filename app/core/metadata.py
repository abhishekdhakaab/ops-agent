from __future__ import annotations
from hashlib import sha256
from app.core.config import settings

PROMPT_VERSION = "v1.2"   # bump when you change prompts
GRAPH_VERSION  = "v1.3"   # bump when you change graph logic
DATASET_VERSION = "sft_grpo_v1"  # bump when you retrain


def model_fingerprint()->str:
    s = f"{settings.llm_provider}:{settings.llm_base_url}:{settings.llm_model}"
    return sha256(s.encode('utf-8')).hexdigest()[:12]

## basic model config will be saved with log for easy debug 
def run_metadata()->dict:
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "model_fp": model_fingerprint(),
        "prompt_version": PROMPT_VERSION,
        "graph_version": GRAPH_VERSION,
        "dataset_version": DATASET_VERSION,
    }