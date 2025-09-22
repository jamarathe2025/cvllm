from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None


class Experience(BaseModel):
    title: str
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    link: Optional[str] = None


class Resume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    publications: List[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None


class TailoringConfig(BaseModel):
    target_seniority: Optional[str] = None
    target_role: Optional[str] = None
    tone: str = "concise"  # concise, confident, impact-focused
    length: str = "1page"   # 1page or 2pages


class PipelineArtifacts(BaseModel):
    parsed_resume: Resume
    parsed_jd: JobDescription
    tailored_markdown: str
    alignment_score: float
    keyword_coverage: float


class RequirementScore(BaseModel):
    requirement: str
    score: float
    explanation: Optional[str] = None


class EvidenceSnippet(BaseModel):
    text: str
    score: float


class CandidateScore(BaseModel):
    resume_path: str
    name: Optional[str] = None
    alignment_score: float
    keyword_coverage: float
    rank: Optional[int] = None
    overall_explanation: Optional[str] = None
    per_requirement: List[RequirementScore] = Field(default_factory=list)
    evidence: List[EvidenceSnippet] = Field(default_factory=list)


class RankingResult(BaseModel):
    parsed_jd: JobDescription
    candidates: List[CandidateScore] = Field(default_factory=list)
