import httpx
from typing import Dict, Any, Optional
from .base import LLMClient
from app.core.config import settings
from .json_utils import extract_json_object, JSONParseError


class OpenAICompatibleHTTP(LLMClient):
    def __init__(self)->None:
        self.base_url = settings.llm_base_url.rstrip('/')
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_s

    ## calls LLM for response 
    async def _post_chat(self,payload : Dict[str,Any])->Dict[str,Any]:
        headers = {"Authorization":f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout = self.timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions",headers =headers, json=payload)
            r.raise_for_status()
            return r.json()
    ## crate payload prompt structure
    async def complete_text(self,*,system:str,user:str)->str:
        payload = {
            "model":self.model,
            "messages":[
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
            "temperature":0.2
        }
        data  = await self._post_chat(payload)
        return data["choices"][0]["message"]["content"]

    async def complete_json(self,*,system:str,user:str, json_schema:Dict[str,Any],max_retires:int=10)->Dict[str,Any]:
        schema_hint = f"""
                        Return ONLY valid JSON matching this schema (no markdown, no extra text, no extra keys):
                        {json_schema}
                        """
        base_payload = {
            "model":self.model,
            "temperature":0.0,
            "messages":[
                {"role":"system","content":system},
                {"role":"user","content":user+ "\n\n" + schema_hint}
            ]
        }
        last_text : Optional[str]=None
        for attempt in range(max_retires+1):
            data = await self._post_chat(base_payload)
            text = data["choices"][0]["message"]["content"]
            last_text = text
            try : 
                return extract_json_object(text)
            except JSONParseError:
                # Ollama's OpenAI-compatible /v1 often rejects "repair" prompts (400),
                # and small local models are flaky at JSON anyway.
                # For Ollama: retry once with a stricter instruction, then fail.
                if "localhost:11434" in self.base_url or "127.0.0.1:11434" in self.base_url:
                    base_payload["messages"] = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user + "\n\nRETURN ONLY MINIFIED JSON. NO TEXT. NO MARKDOWN. NO CODE FENCES."},
                    ]
                    continue

                # For non-Ollama providers: attempt repair
                repair_payload = {
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": "You are a JSON repair bot. Output ONLY valid JSON."},
                        {"role": "user", "content": f"Fix this to valid JSON matching schema:\nSCHEMA:\n{json_schema}\n\nTEXT:\n{text}"},
                    ],
                }
                data2 = await self._post_chat(repair_payload)
                repaired = data2["choices"][0]["message"]["content"]
                try:
                    return extract_json_object(repaired)
                except JSONParseError:
                    continue

            
        raise RuntimeError(f"LLM did not return valid JSON after retries. Last output: {last_text}")






