from __future__ import annotations
from docx import Document


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    lines = []
    for para in doc.paragraphs:
        lines.append(para.text)
    return "\n".join(lines)
