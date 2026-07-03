"""Build the runbook retrieval index used by the agent graph."""

from __future__ import annotations
from typing import List,Dict, Any
from .index import load_runbooks, chunk_text,save_index
from .embeddings import embed_texts


async def ingest_runbooks() ->Dict[str,Any]:
    """Chunk all runbooks, embed them when configured, and save the index."""

    # Keeping rows without embeddings preserves the keyword-search fallback.
    docs = load_runbooks()
    rows : List[Dict[str,Any]] = []
    for d in docs:
        chunks = chunk_text(d['text'])
        for i,ch in enumerate(chunks):
            rows.append({"source": d["path"],"chunk_id": i,"text": ch,"embedding": None,})
    embs = await embed_texts([r['text'] for r in rows])
    if embs is not None and len(embs) == len(rows):
        for r,e in zip(rows,embs):
            r['embedding'] =e

    save_index(rows)
    return {"docs": len(docs), "chunks": len(rows), "embeddings": embs is not None}
