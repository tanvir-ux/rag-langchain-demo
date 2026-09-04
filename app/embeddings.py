"""Embedding backends — default is a deterministic fake so the demo runs offline."""

from __future__ import annotations

import hashlib
import math
from typing import List

from langchain_core.embeddings import Embeddings

from app.config import Settings


class FakeDeterministicEmbeddings(Embeddings):
    """Local, no-API embeddings for demos and CI. Same text → same vector."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals: List[float] = []
        # Expand digest into a fixed-length vector
        seed = digest
        while len(vals) < self.dim:
            for b in seed:
                vals.append((b / 255.0) * 2.0 - 1.0)
                if len(vals) >= self.dim:
                    break
            seed = hashlib.sha256(seed).digest()
        # L2 normalize so cosine similarity is meaningful
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def build_embeddings(settings: Settings) -> Embeddings:
    backend = settings.embedding_backend.lower().strip()

    if backend in ("fake", "hash", "local-fake"):
        return FakeDeterministicEmbeddings()

    if backend in ("sentence-transformers", "st", "huggingface"):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Small, commonly available model; first run downloads weights.
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    if backend == "openai":
        from langchain_community.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
            model=settings.openai_embedding_model,
        )

    if backend == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

    if backend == "groq":
        # Groq is primarily chat; for embeddings point OPENAI_* at an
        # OpenAI-compatible embed endpoint, or use fake/sentence-transformers.
        raise ValueError(
            "Groq does not expose a public embedding model in this demo. "
            "Set EMBEDDING_BACKEND=fake|sentence-transformers|openai|ollama "
            "and use Groq for LLM generation separately."
        )

    raise ValueError(f"Unknown EMBEDDING_BACKEND={backend!r}")
