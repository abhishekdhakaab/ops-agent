import httpx
from typing import Any,Dict,Optional
from app.core.config import settings
from .base import LLMClient
from .json_utils import extract_json_object, JSONParseError


class OllamaHTTP(LLMClient):
    def __init__(self)->None:
        self.base_url = settings.llm_base_url.rstrip('/')
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_s
    async def complete_text(self,*,system:str, user:str)->str:
        payload = {
            'model':self.model,
            'messages':[
                {'role':'system','content':system},
                {'role':'user','content':user}
            ],
            'stream':False,
            'options':{'temperature':0.2}
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f'{self.base_url}/api/chat',json=payload)
            r.raise_for_status()
            data = r.json()
            return data['message']['content']
    async def complete_json(self,*,system:str,user:str,json_schema:Dict[str,Any],max_retries:int=2)->Dict[str,Any]:
            schema_hint = f"""
                                Return ONLY valid JSON matching this schema (no markdown, no extra text, no extra keys):
                                {json_schema}
                                """
            last_text: Optional[str] = None
            for _ in range(max_retries+1):
                 text = await self.complete_text(system=system,user=user+"\n\n"+ schema_hint)
                 last_text = text
                 try : 
                      return extract_json_object(text)
                 except JSONParseError:
                      repair = await self.complete_text(system="You are a JSON repair bot. Output ONLY valid JSON.",
                        user=f"Fix this to valid JSON matching schema:\nSCHEMA:\n{json_schema}\n\nTEXT:\n{text}"
                        )
                      try:
                           return extract_json_object(repair)
                      except JSONParseError:
                           continue
            raise RuntimeError(f"Ollama did not return valid JSON after retries. Last output: {last_text}")
    
                      

