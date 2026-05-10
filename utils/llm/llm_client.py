import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class LLMClient:
    """
    An HTTP client for OpenRouter, 
    optimized for RAG pipelines and strict JSON formatting.
    """
    
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "inclusionai/ring-2.6-1t:free", timeout: int = 30):
        """
        Args:
            api_key: OpenRouter API key.
            model: The target model string.
            timeout: Maximum seconds to wait for an HTTP response.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8000",
            "X-Title": "Traceability First Retrieval"
        }
        
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """
        Sends a chat completion request to OpenRouter.
        Includes an automatic fallback mechanism if the requested model 
        does not support native JSON mode formatting.
        """
        if json_mode and "json" not in system_prompt.lower():
            system_prompt += "\nReturn the response as a JSON object."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                self.OPENROUTER_URL, 
                headers=self.headers, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            # A 400 Bad Request usually means the model doesn't support the 'response_format' parameter
            if json_mode and response.status_code == 400:
                return self._fallback_json_chat(payload)
            else:
                raise RuntimeError(f"OpenRouter API Error: {response.status_code} - {response.text}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error communicating with OpenRouter: {str(e)}") from e

    def _fallback_json_chat(self, payload: dict) -> str:
        """
        Fallback method for models that reject the explicit response_format flag.
        Enforces JSON via system prompt injection and cleans up markdown.
        """
        # 1. Remove the unsupported format parameter
        if "response_format" in payload:
            del payload["response_format"]
            
        # 2. Inject a strong directive into the system prompt
        payload["messages"][0]["content"] += "\nIMPORTANT: You MUST respond with ONLY valid JSON. Do not include markdown formatting or conversational text."
        
        # 3. Retry the network request
        response = requests.post(
            self.OPENROUTER_URL, 
            headers=self.headers, 
            json=payload, 
            timeout=self.timeout
        )
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        
        # 4. Strip markdown wrappers if the model still outputs them
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
             content = content.split("```")[1].split("```")[0].strip()
             
        return content

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings using the local SentenceTransformer model.
        """
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return np.ascontiguousarray(embeddings, dtype=np.float32)
    