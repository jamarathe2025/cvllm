from __future__ import annotations
import argparse
import os
from rich import print
from docx import Document
from cv_creator.pipeline import run_pipeline, run_ranking
from cv_creator.schema import TailoringConfig
import glob


def save_docx(markdown_text: str, path: str):
    # Simple export: write paragraphs line by line; for production consider md->docx conversion
    doc = Document()
    for line in markdown_text.splitlines():
        doc.add_paragraph(line)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.save(path)


def main():
    parser = argparse.ArgumentParser(description="CV Creation and Ranking using LLMs (Local Ollama)")
    # Single resume mode
    parser.add_argument("--resume", required=False, help="Path to resume file (pdf, docx, or txt)")
    parser.add_argument("--out", required=False, help="Output Markdown resume path (single mode)")
    parser.add_argument("--json", required=False, help="Path to write extraction JSON (single mode)")
    parser.add_argument("--docx", required=False, help="Optional path to write DOCX export (single mode)")

    # Batch ranking mode
    parser.add_argument(
        "--resumes",
        required=False,
        help="Comma-separated list of resume paths or glob patterns for ranking (e.g., 'data/*.pdf,data/*.docx')",
    )
    parser.add_argument("--rank-json", required=False, help="Path to write ranking JSON (batch mode)")
    parser.add_argument("--rank-csv", required=False, help="Path to write ranking CSV (batch mode)")
    parser.add_argument(
        "--engine",
        required=False,
        default="heuristic",
        choices=["heuristic", "resume_matcher", "lindex", "langchain"],
        help="Scoring engine: 'heuristic' (default), 'resume_matcher', 'lindex' (LlamaIndex semantic), or 'langchain' (LLM rubric via LangChain)",
    )

    # Shared
    parser.add_argument("--job", required=False, help="Path to job description txt file")
    parser.add_argument("--job-text", required=False, help="Raw JD text (alternative to --job)")
    parser.add_argument("--model", required=False, help="Ollama model name (default from CVLLM_MODEL or gemma:2b)")

    args = parser.parse_args()

    # Validate mode selection
    single_mode = bool(args.resume)
    batch_mode = bool(args.resumes)
    assert single_mode ^ batch_mode, "Use either --resume (single) or --resumes (batch), not both."

    if single_mode:
        assert args.out, "--out is required in single resume mode"
        artifacts = run_pipeline(
            resume_path=args.resume,
            jd_path=args.job,
            jd_text=args.job_text,
            model=args.model,
            out_json=args.json,
        )

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(artifacts.tailored_markdown)

        if args.docx:
            save_docx(artifacts.tailored_markdown, args.docx)

        print({
            "alignment_score": artifacts.alignment_score,
            "keyword_coverage": artifacts.keyword_coverage,
        })
        print("[green]Done. Outputs written.")
        return

    # Batch mode: resolve glob patterns and comma-separated list
    paths = []
    for part in (args.resumes or "").split(","):
        part = part.strip()
        if not part:
            continue
        expanded = glob.glob(part)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(part)
    assert paths, "No resumes found for --resumes"

    result = run_ranking(
        resume_paths=paths,
        jd_path=args.job,
        jd_text=args.job_text,
        model=args.model,
        out_json=args.rank_json,
        out_csv=args.rank_csv,
        engine=args.engine,
    )

    print("[bold]Top candidates:")
    for c in result.candidates[:5]:
        print(f"#{c.rank}: {c.name or 'Unknown'} | Align={c.alignment_score} | Cover={c.keyword_coverage} | {c.resume_path}")
    print("[green]Ranking complete.")


if __name__ == "__main__":
    main()
