"""Rank indexed runbook chunks by embeddings or keyword overlap."""

from __future__ import annotations
from typing import List,Dict,Any,Optional,Tuple
import math
from .index import load_index


def cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity, rejecting mismatched embedding dimensions."""
    if len(a) != len(b):
        import logging
        logging.warning(
            f"cosine(): vector length mismatch ({len(a)} vs {len(b)}). "
            "Returning 0.0. This likely means embeddings were generated "
            "by different models. Re-run ingest.py to rebuild the index."
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def keyword_score(q:str, text:str)->float:
    """Score operational terms when no embedding model is available."""
    q = q.lower()
    t = text.lower()
    score = 0.0
    for w in ["latency", "slow", "timeout", "error", "deploy", "rollback", "p95", "database"]:
        if w in q and w in t:
            score += 1.0
    # A small general overlap bonus breaks ties between domain-term matches.
    for tok in set(q.split()):
        if len(tok) >= 4 and tok in t:
            score += 0.05
    return score


def retrieve(query:str, query_emb:Optional[List[float]],k:int=4,rows=load_index())->List[Dict[str,Any]]:
    """Return the highest-scoring positive-match chunks for a query."""
    rows = load_index()
    scored : List[Tuple[float,Dict[str,Any]]] = []
    for r in rows:
        if query_emb is not None and r.get('embedding') is not None:
            s = cosine(query_emb, r['embedding'])
        else:
            s = keyword_score(query,r['text'])
        scored.append((s,r))

    scored.sort(key=lambda x: x[0] ,reverse=True)
    top = [r for s,r in scored[:k] if s>0]
    return top
