"""
Ollama API client with retry logic and JSON parsing.

Handles:
- Connection errors (Ollama not running)
- JSON parse failures (model outputs garbage)
- Retries with configurable delay
"""

import json
import re
import time
import sys
import httpx
from typing import Optional, Dict, Any

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    MAX_RETRIES,
    RETRY_DELAY_S,
)


def check_ollama_ready(model: Optional[str] = None) -> bool:
    """Verify Ollama is running and the model is available."""
    model = model or OLLAMA_MODEL
    
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException):
        print(f"\n  ERROR: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print(f"  Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"\n  ERROR: Ollama check failed: {e}")
        return False
    
    available = [m["name"] for m in r.json().get("models", [])]
    # Ollama returns names like "gemma3:12b" or "gemma3:12b-instruct-q4_0"
    model_found = any(m.startswith(model.split(":")[0]) for m in available)
    if not model_found:
        model_found = model in available or f"{model}:latest" in available
    
    if not model_found:
        print(f"\n  ERROR: Model '{model}' not found in Ollama.")
        print(f"  Available models: {available}")
        print(f"  Pull it with: ollama pull {model}")
        return False
    
    return True


def call_ollama_json(
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    num_predict: int = 300,
    max_retries: int = MAX_RETRIES,
) -> Optional[dict]:
    """
    Call Ollama chat API and parse JSON response.
    
    Returns parsed dict on success, None on failure after all retries.
    Uses format="json" to force valid JSON output from Ollama.
    """
    model = model or OLLAMA_MODEL
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    
    for attempt in range(max_retries):
        try:
            r = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            
            content = r.json()["message"]["content"].strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
            
            cleaned = content
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_S)
                continue
                
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                print(f"    Timeout (attempt {attempt+1}/{max_retries}), retrying...")
                time.sleep(RETRY_DELAY_S * 2)
                continue
            
        except httpx.HTTPStatusError as e:
            print(f"    HTTP error: {e.response.status_code}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_S)
                continue
                
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_S)
                continue
    
    return None


def call_ollama_text(
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    num_predict: int = 300,
) -> Optional[str]:
    """
    Call Ollama chat API and return raw text response.
    Used when we don't need JSON parsing (e.g., for rationale generation).
    """
    model = model or OLLAMA_MODEL
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    
    try:
        r = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        print(f"    Text call error: {e}")
        return None
