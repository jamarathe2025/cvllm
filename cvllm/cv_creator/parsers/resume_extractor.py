from __future__ import annotations
import json
from typing import Tuple
from ..schema import Resume
from ..prompt_templates import EXTRACTION_PROMPT
from ..llm.ollama_client import OllamaClient


def extract_structured_resume(resume_text: str, client: OllamaClient) -> Tuple[Resume, dict]:
    prompt = EXTRACTION_PROMPT.format(resume_text=resume_text)
    result = client.complete_json(prompt)
    # Normalize keys and load into schema, leniently
    if "contact" in result:
        contact = result.get("contact") or {}
        result.setdefault("email", contact.get("email"))
        result.setdefault("phone", contact.get("phone"))
        result.setdefault("location", contact.get("location"))
        result.setdefault("linkedin", contact.get("linkedin"))
        result.setdefault("github", contact.get("github"))
        result.setdefault("website", contact.get("website"))
    try:
        resume = Resume.model_validate(result)
    except Exception:
        # Fallback to partial parsing
        resume = Resume()
        for k, v in result.items():
            if hasattr(resume, k):
                setattr(resume, k, v)
    return resume, result
