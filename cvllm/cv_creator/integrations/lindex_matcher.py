from __future__ import annotations
import os
from typing import Tuple, List, Dict, Any

# LlamaIndex imports (embedding-only engine)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    # simple paragraph-based chunking, merge small paragraphs
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _jd_queries(jd_text: str) -> List[str]:
    # derive multiple queries: full JD + top lines
    lines = [l.strip() for l in jd_text.split("\n") if l.strip()]
    head = " ".join(lines[:5])[:1000]
    queries = [jd_text[:2000]]
    if head:
        queries.append(head)
    return queries


def score_resume_vs_jd_lindex(resume_text: str, jd_text: str, model: str | None = None) -> Tuple[float, float]:
    """
    Compute semantic alignment (0..1) and keyword coverage proxy using LlamaIndex embeddings.

    Alignment is derived from average similarity of top-k resume chunks to JD queries.
    Coverage is approximated using fraction of queries with at least one strong match.
    """
    # Configure embedding model once per process
    embed_name = os.getenv("SEMANTIC_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if not isinstance(Settings.embed_model, HuggingFaceEmbedding) or getattr(Settings.embed_model, "model_name", None) != embed_name:
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_name)

    chunks = _chunk_text(resume_text)
    docs = [Document(text=c) for c in chunks]
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine(similarity_top_k=5)

    queries = _jd_queries(jd_text)
    sims: List[float] = []
    strong_hits = 0
    for q in queries:
        resp = engine.query(q)
        # source_nodes contain similarity scores in .score (higher is better)
        node_scores = []
        for sn in getattr(resp, "source_nodes", []) or []:
            try:
                node_scores.append(float(sn.score))
            except Exception:
                continue
        if node_scores:
            top = max(node_scores)
            sims.append(top)
            if top >= 0.6:
                strong_hits += 1
        else:
            sims.append(0.0)
    if not sims:
        return 0.5, 0.5

    # Normalize rough cosine scores from [0, 1] (empirical; LlamaIndex scores are typically cosine-like)
    avg_sim = sum(sims) / len(sims)
    coverage = strong_hits / len(queries)
    # Blend: emphasize semantic match and ensure within [0,1]
    alignment = max(0.0, min(1.0, 0.7 * avg_sim + 0.3 * coverage))
    return round(alignment, 3), round(coverage, 3)


def score_resume_vs_jd_lindex_with_evidence(
    resume_text: str, jd_text: str, model: str | None = None
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Same as score_resume_vs_jd_lindex, but also returns snippet-level evidence.

    Evidence items: {"text": chunk_text, "score": similarity}
    """
    # Configure embedding model
    embed_name = os.getenv("SEMANTIC_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if not isinstance(Settings.embed_model, HuggingFaceEmbedding) or getattr(Settings.embed_model, "model_name", None) != embed_name:
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_name)

    chunks = _chunk_text(resume_text)
    docs = [Document(text=c) for c in chunks]
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine(similarity_top_k=5)

    queries = _jd_queries(jd_text)
    sims: List[float] = []
    strong_hits = 0
    evidence: List[Dict[str, Any]] = []
    for q in queries:
        resp = engine.query(q)
        node_scores = []
        for sn in getattr(resp, "source_nodes", []) or []:
            try:
                node_scores.append(float(sn.score))
                evidence.append({"text": sn.node.get_content(), "score": float(sn.score)})
            except Exception:
                continue
        if node_scores:
            top = max(node_scores)
            sims.append(top)
            if top >= 0.6:
                strong_hits += 1
        else:
            sims.append(0.0)
    if not sims:
        return 0.5, 0.5, []

    avg_sim = sum(sims) / len(sims)
    coverage = strong_hits / len(queries)
    alignment = max(0.0, min(1.0, 0.7 * avg_sim + 0.3 * coverage))
    return round(alignment, 3), round(coverage, 3), evidence
