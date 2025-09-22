# CV Creation using LLMs (Capstone)

This project demonstrates a local, privacy-first pipeline to create and tailor ATS-friendly resumes using open models.

Highlights:
- Local inference via Ollama (configurable to Gemma 3 1B or other open models)
- Resume parsing from PDF/Word to structured JSON
- Job description parsing to requirements/keywords
- Tailored resume generation (Markdown and optional DOCX)
- Simple evaluation metrics (keyword coverage, alignment score)
- Modular design to extend with LangChain/LlamaIndex or integrate concepts from ResumeLM
- NEW: Batch ranking of multiple resumes for a single JD, with JSON/CSV outputs

## 1) Setup

Prerequisites:
- Python 3.10+
- Ollama installed and running (https://ollama.com)
- Pull at least one model (example Gemma):
  - If available: `ollama pull gemma3:1b`
  - Alternatives: `ollama pull gemma:2b`, `ollama pull llama3.1:8b`, or any local model you prefer.

Install dependencies:
```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Configure model (optional):
- Default model is read from environment variable `CVLLM_MODEL` (fallback: `gemma:2b`).
- Example: `set CVLLM_MODEL=gemma:2b`

## 2) Usage (CLI MVP)

Extract, parse, tailor, and generate a resume:
```
python app.py \
  --resume path/to/Resume.pdf \
  --job path/to/job_description.txt \
  --out out/tailored_resume.md \
  --json out/extractions.json \
  --docx out/tailored_resume.docx
```

Outputs:
- Structured JSON with parsed resume and JD fields
- Tailored resume as Markdown
- Optional DOCX export (nice for sharing)

### 2.1) Batch Ranking: Sort multiple resumes for one JD

Provide a set of resume paths (or glob patterns) and a job description. The tool extracts each resume, tailors it briefly to the JD, evaluates alignment and keyword coverage, and returns a ranked list.

Example:
```
python app.py \
  --resumes "examples/*.pdf,examples/*.docx" \
  --job examples/sample_job.txt \
  --rank-json out/ranking.json \
  --rank-csv out/ranking.csv \
  --engine resume_matcher
```

Console output shows the top candidates; full results are in the JSON/CSV files with fields: rank, name, resume_path, alignment_score, keyword_coverage.

Engines:
- `heuristic` (default): Tailors each resume and computes keyword-based alignment.
- `resume_matcher`: Uses an LLM rubric to compute an overall fit score (inspired by open-source Resume Matcher) and combines it with keyword coverage.

## 3) Project Structure

```
cvllm/
  app.py                  # CLI entry point
  requirements.txt
  README.md
  cv_creator/
    __init__.py
    schema.py             # Pydantic models for Resume and Job Description
    prompt_templates.py   # Prompt strings for extraction and tailoring
    llm/
      __init__.py
      ollama_client.py    # Thin wrapper around local Ollama
    parsers/
      __init__.py
      pdf_parser.py       # PDF text extraction
      word_parser.py      # DOCX text extraction
      resume_extractor.py # Heuristics + LLM-assisted structured extraction
    pipeline.py           # Orchestrates extract -> parse JD -> tailor -> generate; includes run_ranking for batch ranking
    evaluation.py         # Simple ATS-like evaluation metrics
  examples/
    sample_job.txt
    sample_resume.pdf     # (add your own; placeholder)
  out/                    # Outputs created at runtime
```

## 4) Extending with LangChain/LlamaIndex
- Replace direct calls to `ollama` with `langchain` wrappers (e.g., `ChatOllama`) and chains for extraction and generation.
- Or use `LlamaIndex` for document ingestion and query-based extraction from portfolio materials.
- Consider integrating ResumeLM-style templates and sections for strong ATS alignment.

## 5) Web UI (FastAPI)

Run a simple web app to upload multiple resumes and a JD, then view a ranked list.

Start the server:
```
uvicorn app.web_server:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000 in your browser.

Endpoints:
- `GET /`: Upload form.
- `POST /rank`: Processes uploads, runs ranking, and renders a results table.
- `POST /download/csv`: Downloads the current results as CSV.

Notes:
- The server stores uploaded files to a temporary directory per request.
- Set model via the UI (e.g., `gemma:2b`, `llama3.1:8b`) or default `CVLLM_MODEL` env var.
- Ensure `ollama` is running locally and the model is pulled.
- Choose scoring engine in the UI: Heuristic or Resume Matcher.

## 6) Notes
- Keep PII local; this pipeline runs entirely on your machine.
- Results depend on model quality and prompt tuning. Iterate on prompts in `cv_creator/prompt_templates.py`.
- For best ATS results, ensure strong, quantified bullet points and explicit keyword matching.
