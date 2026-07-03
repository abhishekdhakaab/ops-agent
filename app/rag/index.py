"""Chunk runbooks and persist their retrieval index."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import json
import math

RUNBOOK_DIR = Path("data/runbooks")
INDEX_PATH = Path("data/rag_index.json")
def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    """Group paragraphs into chunks without splitting their internal text."""

    # Paragraph boundaries keep operational steps readable in retrieved context.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks
def load_runbooks() -> List[Dict[str, Any]]:
    """Load Markdown and text runbooks from the project data directory."""
    RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for fp in RUNBOOK_DIR.glob("**/*"):
        if fp.is_file() and fp.suffix.lower() in {".md", ".txt"}:
            docs.append({"path": str(fp), "text": fp.read_text(encoding="utf-8")})
    return docs
def save_index(rows: List[Dict[str, Any]]) -> None:
    """Persist retrieval rows as a human-inspectable JSON index."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
def load_index() -> List[Dict[str, Any]]:
    """Load the current index, returning an empty collection before ingestion."""
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
