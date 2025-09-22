from __future__ import annotations
import json
import re
from typing import Tuple, Dict, Any, List

try:
    import ollama  # type: ignore
except Exception:
    ollama = None


RM_PROMPT = (
    """
You are an expert Resume Matcher. Given a candidate resume and a job description, assess overall fit.
Return ONLY JSON with the following fields:
- overall_score: float in [0,1] representing overall role fit (0=poor, 1=excellent)
- notes: brief string explaining the reasoning (no more than 3 sentences)
Consider: role alignment, responsibilities, must-have requirements, seniority, and evidence of impact.

Resume:
----------------
{resume_text}

Job Description:
----------------
{jd_text}
"""
).strip()


RM_DETAILED_PROMPT = (
    """
You are an expert Resume Matcher. Score how well the candidate satisfies each requirement from the job description, then provide an overall score and explanation.
Return ONLY JSON with keys:
- per_requirement: list of { requirement: string, score: float 0..1, explanation: string <= 2 sentences }
- overall_score: float 0..1
- overall_explanation: string <= 3 sentences

Resume:
----------------
{resume_text}

Job Description (focus on "requirements"):
----------------
{jd_text}
"""
).strip()


def _json_from_text(text: str) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _simple_keyword_coverage(resume_text: str, jd_text: str) -> float:
    # Tokenize naive keywords from JD and compute coverage in resume text
    def tokens(s: str):
        return [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+.#-]{3,}", s)]

    jd_tokens = tokens(jd_text)
    if not jd_tokens:
        return 0.5
    jd_unique = list(dict.fromkeys(jd_tokens))[:50]
    rtext = resume_text.lower()
    covered = sum(1 for k in jd_unique if k in rtext)
    return round(covered / max(1, len(jd_unique)), 3)


def score_resume_vs_jd(resume_text: str, jd_text: str, model: str | None = None) -> Tuple[float, float]:
    """Return (alignment_score, keyword_coverage) using an LLM-based Resume Matcher style rubric.

    Falls back gracefully if ollama is unavailable.
    """
    coverage = _simple_keyword_coverage(resume_text, jd_text)

    overall = 0.5
    if ollama is not None:
        try:
            resp = ollama.chat(
                model=model or "gemma:2b",
                messages=[{"role": "user", "content": RM_PROMPT.format(resume_text=resume_text, jd_text=jd_text)}],
                options={"temperature": 0.2},
            )
            text = resp.get("message", {}).get("content", "").strip()
            data = _json_from_text(text)
            val = data.get("overall_score")
            if isinstance(val, (int, float)):
                overall = float(val)
        except Exception:
            overall = 0.5
    return round(overall, 3), coverage


def score_resume_vs_jd_with_details(
    resume_text: str, jd_text: str, model: str | None = None
) -> Tuple[float, float, str | None, List[Dict[str, Any]]]:
    """Return (overall_score, keyword_coverage, overall_explanation, per_requirement[]) using an LLM-based rubric.

    per_requirement item: {"requirement": str, "score": float, "explanation": str}
    """
    coverage = _simple_keyword_coverage(resume_text, jd_text)
    overall = 0.5
    overall_expl = None
    per_req: List[Dict[str, Any]] = []

    if ollama is None:
        return round(overall, 3), coverage, overall_expl, per_req

    try:
        resp = ollama.chat(
            model=model or "gemma:2b",
            messages=[{"role": "user", "content": RM_DETAILED_PROMPT.format(resume_text=resume_text, jd_text=jd_text)}],
            options={"temperature": 0.2},
        )
        text = resp.get("message", {}).get("content", "").strip()
        data = _json_from_text(text)
        val = data.get("overall_score")
        if isinstance(val, (int, float)):
            overall = float(val)
        overall_expl = data.get("overall_explanation") or data.get("notes")
        for item in data.get("per_requirement", []) or []:
            req = str(item.get("requirement", ""))
            sc = item.get("score")
            try:
                scf = float(sc)
            except Exception:
                scf = 0.0
            per_req.append({
                "requirement": req,
                "score": scf,
                "explanation": item.get("explanation") or "",
            })
    except Exception:
        pass

    return round(overall, 3), coverage, overall_expl, per_req
