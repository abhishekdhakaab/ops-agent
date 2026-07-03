"""Attach reproducibility metadata to stored agent runs."""

from __future__ import annotations
from hashlib import sha256
from app.core.config import settings

# These versions make behavioral changes visible in otherwise similar run records.
PROMPT_VERSION = "v1.2"
GRAPH_VERSION  = "v1.3"
DATASET_VERSION = "sft_grpo_v1"


def model_fingerprint()->str:
    """Return a short, non-secret identifier for the active model endpoint."""
    s = f"{settings.llm_provider}:{settings.llm_base_url}:{settings.llm_model}"
    return sha256(s.encode('utf-8')).hexdigest()[:12]

def run_metadata()->dict:
    """Capture the model and artifact versions needed to interpret a run."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "model_fp": model_fingerprint(),
        "prompt_version": PROMPT_VERSION,
        "graph_version": GRAPH_VERSION,
        "dataset_version": DATASET_VERSION,
    }
