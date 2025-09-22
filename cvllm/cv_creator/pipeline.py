from __future__ import annotations
import json
import os
from typing import Optional, Tuple, List
import csv
from .schema import Resume, JobDescription, TailoringConfig, PipelineArtifacts, RankingResult, CandidateScore
from .integrations.resume_matcher_client import score_resume_vs_jd, score_resume_vs_jd_with_details
from .integrations.lindex_matcher import score_resume_vs_jd_lindex, score_resume_vs_jd_lindex_with_evidence
from .orchestrators.langchain_orchestrator import score_resume_vs_jd_langchain, score_resume_vs_jd_langchain_with_details
from .prompt_templates import JD_PARSING_PROMPT, TAILORING_PROMPT
from .llm.ollama_client import OllamaClient
from .parsers.pdf_parser import extract_text_from_pdf
from .parsers.word_parser import extract_text_from_docx
from .parsers.resume_extractor import extract_structured_resume


def _read_resume_text(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(path)
    if lower.endswith(".docx"):
        return extract_text_from_docx(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_job_description(jd_text: str, client: OllamaClient) -> Tuple[JobDescription, dict]:
    prompt = JD_PARSING_PROMPT.format(job_text=jd_text)
    result = client.complete_json(prompt)
    try:
        jd = JobDescription.model_validate(result)
    except Exception:
        jd = JobDescription(raw_text=jd_text)
        for k, v in (result or {}).items():
            if hasattr(jd, k):
                setattr(jd, k, v)
    return jd, result


def tailor_resume(resume: Resume, jd: JobDescription, client: OllamaClient, config: Optional[TailoringConfig] = None) -> str:
    config = config or TailoringConfig()
    prompt = TAILORING_PROMPT.format(
        tone=config.tone,
        seniority=config.target_seniority or "auto",
        role=config.target_role or (jd.title or "target role"),
        length=config.length,
        resume_json=resume.model_dump_json(indent=2),
        jd_json=jd.model_dump_json(indent=2),
    )
    md = client.complete_text(prompt)
    return md


def evaluate_alignment(resume_md: str, jd: JobDescription) -> Tuple[float, float]:
    # Simple heuristic evaluation: keyword coverage and alignment score (0-1)
    text = resume_md.lower()
    keywords = [k.lower() for k in (jd.keywords or [])]
    if not keywords:
        # derive from requirements/responsibilities
        keywords = []
        for lst in (jd.requirements or []) + (jd.responsibilities or []):
            for token in (lst.split() if isinstance(lst, str) else []):
                if len(token) > 3:
                    keywords.append(token.lower().strip(",.()"))
        keywords = list(dict.fromkeys(keywords))[:30]

    if not keywords:
        return 0.5, 0.5

    covered = sum(1 for k in keywords if k in text)
    coverage = covered / max(1, len(keywords))
    # Alignment adds weighting for top 10 keywords
    top = keywords[:10]
    top_covered = sum(1 for k in top if k in text)
    alignment = 0.6 * coverage + 0.4 * (top_covered / max(1, len(top)))
    return round(alignment, 3), round(coverage, 3)


def run_pipeline(
    resume_path: str,
    jd_path: Optional[str] = None,
    jd_text: Optional[str] = None,
    model: Optional[str] = None,
    out_json: Optional[str] = None,
) -> PipelineArtifacts:
    assert (jd_path or jd_text) and not (jd_path and jd_text), "Provide either jd_path or jd_text"
    client = OllamaClient(model=model)

    resume_text = _read_resume_text(resume_path)
    resume, resume_raw = extract_structured_resume(resume_text, client)

    if jd_path:
        with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()
    assert jd_text is not None
    jd, jd_raw = parse_job_description(jd_text, client)

    md = tailor_resume(resume, jd, client)
    align, cov = evaluate_alignment(md, jd)

    artifacts = PipelineArtifacts(
        parsed_resume=resume,
        parsed_jd=jd,
        tailored_markdown=md,
        alignment_score=align,
        keyword_coverage=cov,
    )

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "resume": resume.model_dump(),
                    "resume_raw": resume_raw,
                    "jd": jd.model_dump(),
                    "jd_raw": jd_raw,
                    "alignment": align,
                    "keyword_coverage": cov,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return artifacts


def run_ranking(
    resume_paths: List[str],
    jd_path: Optional[str] = None,
    jd_text: Optional[str] = None,
    model: Optional[str] = None,
    out_json: Optional[str] = None,
    out_csv: Optional[str] = None,
    engine: str = "heuristic",  # 'heuristic' | 'resume_matcher' | 'lindex' | 'langchain'
) -> RankingResult:
    assert (jd_path or jd_text) and not (jd_path and jd_text), "Provide either jd_path or jd_text"
    client = OllamaClient(model=model)

    if jd_path:
        with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()
    assert jd_text is not None
    jd, jd_raw = parse_job_description(jd_text, client)

    candidates: List[CandidateScore] = []
    for path in resume_paths:
        try:
            resume_text = _read_resume_text(path)
            resume, _ = extract_structured_resume(resume_text, client)
            if engine == "resume_matcher":
                # Prefer detailed rubric; fallback to simple
                overall, cov, overall_expl, per_req = score_resume_vs_jd_with_details(
                    resume_text=resume_text, jd_text=jd.raw_text or jd_text or "", model=client.model
                )
                align = overall
            elif engine == "lindex":
                align, cov, evidence_snips = score_resume_vs_jd_lindex_with_evidence(
                    resume_text=resume_text, jd_text=jd.raw_text or jd_text or "", model=client.model
                )
            elif engine == "langchain":
                # Prefer detailed LC chain if available
                overall, cov, overall_expl, per_req = score_resume_vs_jd_langchain_with_details(
                    resume_text=resume_text, jd_text=jd.raw_text or jd_text or "", model=client.model
                )
                align = overall
            else:
                # Default heuristic: tailor then evaluate keyword alignment
                md = tailor_resume(resume, jd, client)
                align, cov = evaluate_alignment(md, jd)
            cs = CandidateScore(
                resume_path=path,
                name=resume.name,
                alignment_score=align,
                keyword_coverage=cov,
            )
            # Attach details if computed
            try:
                if engine in ("resume_matcher", "langchain"):
                    if 'overall_expl' in locals() and overall_expl:
                        cs.overall_explanation = overall_expl
                    if 'per_req' in locals() and per_req:
                        from .schema import RequirementScore
                        cs.per_requirement = [
                            RequirementScore(requirement=i.get("requirement", ""), score=float(i.get("score", 0.0)), explanation=i.get("explanation") or None)
                            for i in per_req
                        ]
                if engine == "lindex" and 'evidence_snips' in locals() and evidence_snips:
                    from .schema import EvidenceSnippet
                    cs.evidence = [
                        EvidenceSnippet(text=str(e.get("text", "")), score=float(e.get("score", 0.0)))
                        for e in evidence_snips
                    ]
            except Exception:
                pass
            candidates.append(cs)
        except Exception as e:
            # Record a failed parse with minimal info
            candidates.append(
                CandidateScore(
                    resume_path=path,
                    name=None,
                    alignment_score=0.0,
                    keyword_coverage=0.0,
                )
            )

    # Rank by alignment (desc), then coverage (desc)
    candidates.sort(key=lambda c: (c.alignment_score, c.keyword_coverage), reverse=True)
    for idx, c in enumerate(candidates, start=1):
        c.rank = idx

    result = RankingResult(parsed_jd=jd, candidates=candidates)

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "jd": jd.model_dump(),
                    "jd_raw": jd_raw,
                    "candidates": [c.model_dump() for c in candidates],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "name", "resume_path", "alignment_score", "keyword_coverage"])
            for c in candidates:
                writer.writerow([c.rank, c.name or "", c.resume_path, c.alignment_score, c.keyword_coverage])

    return result
