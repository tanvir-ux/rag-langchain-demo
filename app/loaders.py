"""Load TXT / PDF / DOCX into plain text for chunking."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}. Use TXT, PDF, or DOCX.")

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts).strip()

    # .docx
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
