"""Select the configured LLM transport behind one stable interface."""

from .base import LLMClient
from app.core.config import settings
from .http_provider import OpenAICompatibleHTTP
from .ollama_provider import OllamaHTTP


def get_llm()->LLMClient:
    """Construct the provider selected by ``LLM_PROVIDER``."""
    if settings.llm_provider.lower() == "ollama":
        return OllamaHTTP()
    return OpenAICompatibleHTTP()
