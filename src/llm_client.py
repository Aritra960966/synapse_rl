import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import time
import re

class OllamaClient:
    """
    Optimized client for Llama 3.2 with Strict ChatML formatting and JSON mode support.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:latest", 
        timeout: int = 30
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.1)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 150,
        stop_tokens: Optional[List[str]] = None,
        format: Optional[str] = None  # Added for JSON mode support
    ) -> Dict[str, Any]:
        session = await self._get_session()
        
        # DEFAULT STOPS: Crucial to prevent "parroting"
        stops = stop_tokens if stop_tokens else ["<|eot_id|>", "<|end_of_text|>", "Assistant:", "User:"]

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "stop": stops
            }
        }
        
        # Enable structured output if format is specified (e.g., "json")
        if format:
            payload["format"] = format

        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "response": result.get("response", "").strip(),
                        "latency_ms": (time.time() - start_time) * 1000
                    }
                return {"response": "", "error": f"HTTP {response.status}", "latency_ms": 0}
        except Exception as e:
            return {"response": "", "error": str(e), "latency_ms": 0}

    async def extract_intent(self, user_input: str) -> Dict[str, Any]:
        u = user_input.lower()
        if any(x in u for x in ["hi", "hello"]): return {"intent": "greeting", "latency_ms": 0}
        if "?" in u: return {"intent": "question", "latency_ms": 0}
        return {"intent": "inform", "latency_ms": 0}

    async def extract_memories(self, user_input: str, previous_response: str = "") -> List[Dict]:
        """
        Extracts facts using Ollama's JSON mode for high reliability.
        """
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a Data Extraction Tool. Extract key facts from the user's sentence.\n"
            f"Output ONLY a valid JSON list of objects with 'key' and 'value'.\n"
            f"Allowed keys: name, profession, location, preference, restriction.\n"
            f"Example: [{{'key': 'name', 'value': 'Alex'}}]\n"
            f"If no facts exist, return [].\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Input: '{user_input}'\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Uses temperature=0.0 and format="json" for deterministic structured output
        result = await self.generate(
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=150,
            format="json" 
        )
        
        text = result["response"]
        
        try:
            # Safety regex in case of slight formatting variations
            match = re.search(r"\[.*\]", text, re.DOTALL)
            json_str = match.group(0) if match else text
            
            data = json.loads(json_str)
            
            if isinstance(data, dict):
                data = [data]

            valid_keys = {"name", "profession", "location", "preference", "restriction"}
            clean_memories = []
            
            for item in data:
                key = item.get("key")
                value = item.get("value")
                
                if key in valid_keys and value:
                    val = str(value).lower()
                    for prefix in ["is ", "my ", "a "]:
                        if val.startswith(prefix):
                            val = val[len(prefix):]
                    
                    clean_memories.append({
                        "type": "fact", 
                        "key": key, 
                        "value": val.strip(), 
                        "confidence": 1.0
                    })
            return clean_memories

        except (json.JSONDecodeError, Exception) as e:
            print(f"Extraction Error: {e} | Raw Text: {text}")
            return []

    async def generate_response(
        self,
        user_input: str,
        profile: str,
        policies: str,
        memories: List[Dict]
    ) -> str:
        # 1. Prepare Context
        facts = "\n".join([f"{m['key']}: {m['value']}" for m in memories]) if memories else "None"
        
        # 2. Strict Generation Prompt using Llama 3.2 ChatML
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a helpful assistant. Answer the user's question using ONLY the provided Facts.\n"
            f"Facts:\n{facts}\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_input}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        result = await self.generate(prompt=prompt, temperature=0.1, max_tokens=100)
        return result["response"]