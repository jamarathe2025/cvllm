from __future__ import annotations
from typing import Tuple, List
import re
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_models import ChatOllama


class RubricScore(BaseModel):
    overall_score: float = Field(description="A value in [0,1] representing overall role fit")
    reasoning: str = Field(description="Brief rationale (1-3 sentences)")


RUBRIC_TEMPLATE = (
    """
You are an expert recruiter. Evaluate how well the candidate resume matches the job description.
Return ONLY a JSON object with keys: overall_score (float 0..1), reasoning (1-3 sentences).
Consider: responsibilities, must-have requirements, seniority, domain exposure, evidence of impact.

Resume (raw text):
----------------
{resume_text}

Job Description (parsed):
----------------
{jd_text}
"""
).strip()


def _jd_keywords(jd_text: str) -> List[str]:
    # naive keyword harvest from JD
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_+.#-]{4,}", jd_text or "")
    # dedupe and cap
    seen = []
    for t in toks:
        tl = t.lower()
        if tl not in seen:
            seen.append(tl)
        if len(seen) >= 50:
            break
    return seen


def _coverage(resume_text: str, jd_text: str) -> float:
    if not jd_text:
        return 0.5
    r = (resume_text or "").lower()
    kws = _jd_keywords(jd_text)
    if not kws:
        return 0.5
    covered = sum(1 for k in kws if k in r)
    return round(covered / max(1, len(kws)), 3)


def score_resume_vs_jd_langchain(resume_text: str, jd_text: str, model: str | None = None) -> Tuple[float, float]:
    """Orchestrated scoring using LangChain with structured parsing.

    - Builds a ChatOllama chain with a PydanticOutputParser to ensure JSON schema.
    - Returns (overall_score, keyword_coverage).
    """
    parser = PydanticOutputParser(pydantic_object=RubricScore)
    prompt = ChatPromptTemplate.from_template(RUBRIC_TEMPLATE)
    llm = ChatOllama(model=model or "gemma:2b", temperature=0.2)

    # Runnable graph: produce structured score
    chain = prompt | llm | parser
    result = chain.invoke({
        "resume_text": resume_text,
        "jd_text": jd_text,
    })
    overall = float(getattr(result, "overall_score", 0.5))

    cov = _coverage(resume_text, jd_text)
    return round(overall, 3), cov


# Detailed per-requirement structured outputs
class RequirementRubricItem(BaseModel):
    requirement: str
    score: float = Field(ge=0.0, le=1.0)
    explanation: str | None = None


class DetailedRubric(BaseModel):
    per_requirement: List[RequirementRubricItem] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)
    overall_explanation: str | None = None


DETAILED_RUBRIC_TEMPLATE = (
    """
You are an expert recruiter. Score how well the candidate satisfies each requirement from the job description.
Return ONLY a JSON object with keys:
- per_requirement: list of { requirement: string, score: float 0..1, explanation: string <= 2 sentences }
- overall_score: float 0..1
- overall_explanation: string <= 3 sentences

Resume (raw text):
----------------
{resume_text}

Job Description (requirements section included if present):
----------------
{jd_text}
"""
).strip()


def score_resume_vs_jd_langchain_with_details(
    resume_text: str, jd_text: str, model: str | None = None
) -> Tuple[float, float, str | None, List[dict]]:
    parser = PydanticOutputParser(pydantic_object=DetailedRubric)
    prompt = ChatPromptTemplate.from_template(DETAILED_RUBRIC_TEMPLATE)
    llm = ChatOllama(model=model or "gemma:2b", temperature=0.2)
    chain = prompt | llm | parser

    result: DetailedRubric = chain.invoke({
        "resume_text": resume_text,
        "jd_text": jd_text,
    })
    cov = _coverage(resume_text, jd_text)
    overall = float(getattr(result, "overall_score", 0.5))
    per_req_list: List[dict] = [
        {"requirement": r.requirement, "score": float(r.score), "explanation": r.explanation}
        for r in (result.per_requirement or [])
    ]
    return round(overall, 3), cov, result.overall_explanation, per_req_list
