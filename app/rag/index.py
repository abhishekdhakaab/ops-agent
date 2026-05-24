from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import json
import math

RUNBOOK_DIR = Path("data/runbooks")
INDEX_PATH = Path("data/rag_index.json")
# takes in complete text and divide it into chunks based on max character count limit
def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    # simple chunking by paragraphs
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= max_chars: ## bring smaller paragraphs together into buffer until we get it's size close to max_chars
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks
## loads the runbook file 
def load_runbooks() -> List[Dict[str, Any]]:
    RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for fp in RUNBOOK_DIR.glob("**/*"):
        if fp.is_file() and fp.suffix.lower() in {".md", ".txt"}:
            docs.append({"path": str(fp), "text": fp.read_text(encoding="utf-8")})
    return docs
## store the generated embedding in rag_index.json file
def save_index(rows: List[Dict[str, Any]]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
## get particular index from the saved embeddings
def load_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
