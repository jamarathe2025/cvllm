from __future__ import annotations
import os
from typing import Dict, Any
import json

try:
    import ollama  # type: ignore
except Exception as e:
    ollama = None


class OllamaClient:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        self.model = model or os.getenv("CVLLM_MODEL", "gemma:2b")
        self.temperature = temperature
        if ollama is None:
            raise RuntimeError(
                "ollama package not installed. Please `pip install ollama` and ensure Ollama is running."
            )

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        text = resp.get("message", {}).get("content", "").strip()
        # Try to locate JSON if the model wraps it with text
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
            return json.loads(text)
        except Exception:
            # Fallback: return as raw text under key
            return {"raw": text}

    def complete_text(self, prompt: str) -> str:
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return resp.get("message", {}).get("content", "").strip()
