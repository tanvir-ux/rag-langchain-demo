"""FastAPI entrypoint: health, ingest, query."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import __version__
from app.config import get_settings
from app.loaders import SUPPORTED_SUFFIXES
from app.rag import ingest_path, query_rag

app = FastAPI(
    title="RAG LangChain Demo",
    description="End-to-end retrieval-augmented generation skeleton (Chroma + LangChain).",
    version=__version__,
)


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_backend: str


class IngestResponse(BaseModel):
    ok: bool
    source: str
    chunks: int
    chars: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What regions does Acme support?"])


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    top_k: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        embedding_backend=settings.embedding_backend,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile | None = File(default=None),
    path: str | None = Form(default=None),
) -> IngestResponse:
    """
    Ingest a document by multipart upload **or** a server-local path.

    Supported: .txt, .pdf, .docx
    """
    settings = get_settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    if file is None and not path:
        raise HTTPException(status_code=400, detail="Provide either file upload or path.")

    try:
        if file is not None:
            suffix = Path(file.filename or "upload.txt").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported type {suffix}. Use TXT, PDF, or DOCX.",
                )
            dest = settings.upload_path / (file.filename or f"upload{suffix}")
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            result = ingest_path(dest, source_id=file.filename or dest.name)
        else:
            target = Path(path).expanduser().resolve()
            if not target.exists() or not target.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {path}")
            result = ingest_path(target)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    return IngestResponse(
        ok=True,
        source=result["source"],
        chunks=result["chunks"],
        chars=result["chars"],
    )


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    try:
        result = query_rag(body.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return QueryResponse(**result)


@app.get("/")
def root() -> dict:
    return {
        "service": "rag-langchain-demo",
        "docs": "/docs",
        "health": "/health",
    }
