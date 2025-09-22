from __future__ import annotations
from .schema import JobDescription


def explain_scores(alignment: float, coverage: float, jd: JobDescription) -> str:
    tips = []
    if coverage < 0.6:
        tips.append("Add more role-specific keywords found in the JD.")
    if alignment < 0.6:
        tips.append("Prioritize top responsibilities and highlight quantified impact.")
    if not jd.requirements:
        tips.append("Consider extracting explicit requirements from the JD for better alignment.")
    if not tips:
        tips.append("Great alignment. Consider fine-tuning bullet points for stronger action verbs.")
    return "- " + "\n- ".join(tips)
