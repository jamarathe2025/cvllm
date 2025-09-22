from __future__ import annotations
import os
import io
import csv
import tempfile
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cv_creator.pipeline import run_ranking

# App and static/template setup
app = FastAPI(title="CV Sorting using LLMs")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "out", "web_rankings")
os.makedirs(PERSIST_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/rank", response_class=HTMLResponse)
async def rank(
    request: Request,
    resumes: List[UploadFile] = File(..., description="Upload multiple resumes (PDF/DOCX/TXT)"),
    jd_text: str = Form(""),
    jd_file: UploadFile | None = File(None, description="Optional JD text file (txt)"),
    model: str | None = Form(None),
    engine: str = Form("heuristic"),
    required_skills: str = Form(""),
    min_alignment: float = Form(0.0),
    top_n: int = Form(0),
):
    # Prepare temp directory for this request
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    upload_dir = os.path.join(tempfile.gettempdir(), f"cvllm_uploads_{ts}")
    os.makedirs(upload_dir, exist_ok=True)

    # Save resumes to disk
    resume_paths: List[str] = []
    for f in resumes:
        name = os.path.basename(f.filename or "resume")
        dest = os.path.join(upload_dir, name)
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        resume_paths.append(dest)

    # Determine JD text content
    if (jd_file is None or (jd_file and (jd_file.filename or "").strip() == "")) and not jd_text.strip():
        # Redirect back with an error message in query
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Please provide JD text or upload a JD file.",
            },
            status_code=400,
        )

    if jd_file and (jd_file.filename or "").strip() != "":
        jd_bytes = await jd_file.read()
        jd_text_content = jd_bytes.decode("utf-8", errors="ignore")
    else:
        jd_text_content = jd_text

    # Run ranking
    result = run_ranking(
        resume_paths=resume_paths,
        jd_text=jd_text_content,
        jd_path=None,
        model=model,
        out_json=None,
        out_csv=None,
        engine=engine,
    )

    # Persist JSON to disk for later review
    ts_file = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    saved_name = f"ranking_{ts_file}.json"
    saved_path = os.path.join(PERSIST_DIR, saved_name)
    try:
        import json
        with open(saved_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "engine": engine,
                    "timestamp": ts_file,
                    "jd_text": jd_text_content,
                    "jd": result.parsed_jd.model_dump(),
                    "candidates": [c.model_dump() for c in result.candidates],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        saved_name = None

    # Apply interactive filters
    req_tokens = [t.strip().lower() for t in (required_skills or "").split(",") if t.strip()]
    def candidate_matches(c):
        if min_alignment and c.alignment_score < float(min_alignment):
            return False
        if req_tokens:
            hay = (c.name or "") + "\n" + (c.overall_explanation or "")
            # Include per-requirement texts and explanations
            for pr in getattr(c, 'per_requirement', []) or []:
                hay += "\n" + (getattr(pr, 'requirement', '') or '') + "\n" + (getattr(pr, 'explanation', '') or '')
            # Include evidence snippets if any (LlamaIndex engine)
            for ev in getattr(c, 'evidence', []) or []:
                hay += "\n" + (getattr(ev, 'text', '') or '')
            low = hay.lower()
            for token in req_tokens:
                if token not in low:
                    return False
        return True

    filtered = [c for c in result.candidates if candidate_matches(c)]
    if top_n and top_n > 0:
        filtered = filtered[: int(top_n)]
    # Replace candidates with filtered list for display
    result.candidates = filtered

    # Render results table
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": result,
            "error": None,
            "engine": engine,
            "saved_json_name": saved_name,
            "required_skills": required_skills,
            "min_alignment": min_alignment,
            "top_n": top_n,
        },
    )


@app.post("/download/csv")
async def download_csv(
    resumes: List[str] = Form(...),
    ranks: List[int] = Form(...),
    names: List[str] = Form(...),
    aligns: List[float] = Form(...),
    covers: List[float] = Form(...),
):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["rank", "name", "resume_path", "alignment_score", "keyword_coverage"])
    for r, n, p, a, c in zip(ranks, names, resumes, aligns, covers):
        writer.writerow([r, n, p, a, c])
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ranking.csv"},
    )


@app.get("/download/json/{name}")
async def download_json_file(name: str):
    # Security: restrict to PERSIST_DIR and simple filename
    safe_name = os.path.basename(name)
    path = os.path.join(PERSIST_DIR, safe_name)
    if not os.path.isfile(path):
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(path, media_type="application/json", filename=safe_name)


# Convenience local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.web_server:app", host="127.0.0.1", port=8000, reload=True)
