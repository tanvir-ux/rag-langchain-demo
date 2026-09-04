import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force offline fake embeddings before app imports settings cache
os.environ["EMBEDDING_BACKEND"] = "fake"
os.environ["CHROMA_PERSIST_DIR"] = str(Path(__file__).resolve().parent / "_chroma_test")

from app.config import get_settings
from app.rag import reset_vectorstore

get_settings.cache_clear()
reset_vectorstore()

from app.main import app  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    reset_vectorstore()
    yield
    reset_vectorstore()
    get_settings.cache_clear()


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding_backend"] == "fake"


def test_ingest_and_query():
    client = TestClient(app)
    sample = SAMPLES / "onboarding_faq.txt"
    assert sample.exists()

    with sample.open("rb") as f:
        r = client.post(
            "/ingest",
            files={"file": ("onboarding_faq.txt", f, "text/plain")},
        )
    assert r.status_code == 200, r.text
    assert r.json()["chunks"] >= 1

    q = client.post("/query", json={"question": "Which regions are available?"})
    assert q.status_code == 200, q.text
    data = q.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
