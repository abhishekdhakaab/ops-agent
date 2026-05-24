import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env as soon as config module is imported
load_dotenv()
## basic model config will be saved with log for easy debug 
class Settings(BaseModel):
    llm_base_url : str = os.getenv("LLM_BASE_URL","")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")  # example default
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai_compatible")  # generic
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "600"))

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")

    max_steps: int = int(os.getenv("MAX_STEPS","6"))
    


settings = Settings()

