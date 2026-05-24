# takes in list of text alogn with embedding model name and API , returning embedding for each text
from __future__ import annotations
from typing import List, Optional
import httpx
from app.core.config import settings

async def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    provider = (settings.embedding_provider or "").lower()
    base_url = settings.embedding_base_url or settings.llm_base_url
    model = settings.embedding_model

    if not base_url or not model:
        return None

    # Ollama embeddings
    if provider == "ollama":
        payload = {"model": model, "input": texts}
        async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
            r = await client.post(f"{base_url.rstrip('/')}/api/embed", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("embeddings")

    # OpenAI-compatible embeddings (OpenRouter etc.)
    api_key = settings.embedding_api_key or settings.llm_api_key
    if not api_key:
        return None

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
        r = await client.post(f"{base_url.rstrip('/')}/api/embed", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return [d["embedding"] for d in data["data"]]