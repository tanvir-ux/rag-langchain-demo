"""Chunking, vector store, and retrieval chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.embeddings import build_embeddings
from app.loaders import load_file

_store: Chroma | None = None


def get_text_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    global _store
    settings = settings or get_settings()
    if _store is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        embeddings = build_embeddings(settings)
        _store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_path),
        )
    return _store


def reset_vectorstore() -> None:
    """Drop in-memory handle (tests / re-init). Does not delete on-disk files."""
    global _store
    _store = None


def ingest_path(path: Path, source_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    text = load_file(path)
    if not text.strip():
        raise ValueError(f"No extractable text in {path.name}")

    splitter = get_text_splitter(settings)
    docs = splitter.create_documents(
        [text],
        metadatas=[{"source": source_id or path.name, "path": str(path)}],
    )
    store = get_vectorstore(settings)
    ids = store.add_documents(docs)
    return {
        "source": source_id or path.name,
        "chunks": len(docs),
        "ids": ids,
        "chars": len(text),
    }


def format_docs(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source={src})\n{d.page_content}")
    return "\n\n".join(blocks)


def build_retrieval_chain(settings: Settings | None = None):
    """
    LangChain LCEL retrieval chain.

    Without an LLM API key we return a grounded extractive answer:
    top chunks joined with a short preface. Swap in ChatOpenAI / Groq / Ollama
    by replacing the final RunnableLambda — see README.
    """
    settings = settings or get_settings()
    store = get_vectorstore(settings)
    retriever = store.as_retriever(search_kwargs={"k": settings.top_k})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer only from the provided context. If the context is "
                "insufficient, say you do not know.",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
            ),
        ]
    )

    def extractive_answer(payload: dict) -> str:
        question = payload["question"]
        context = payload["context"]
        if not context.strip():
            return "I do not know — no matching documents were retrieved."
        # Demo path: no paid LLM required. Prefixed extractive summary.
        return (
            f"(extractive demo answer for: {question!r})\n\n"
            f"Retrieved context:\n{context}"
        )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(lambda msgs: {"question": msgs.messages[-1].content.split("Question: ", 1)[-1].split("\n\nContext:", 1)[0], "context": msgs.messages[-1].content.split("Context:\n", 1)[-1].rsplit("\n\nAnswer:", 1)[0]})
        | RunnableLambda(extractive_answer)
        | StrOutputParser()
    )
    return chain, retriever


def query_rag(question: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    store = get_vectorstore(settings)
    docs = store.similarity_search(question, k=settings.top_k)
    context = format_docs(docs)
    if not docs:
        answer = "I do not know — no matching documents were retrieved."
    else:
        answer = (
            f"(extractive demo answer for: {question!r})\n\n"
            f"Retrieved context:\n{context}"
        )
    return {
        "answer": answer,
        "sources": [
            {
                "source": d.metadata.get("source"),
                "snippet": d.page_content[:240],
            }
            for d in docs
        ],
        "top_k": settings.top_k,
    }
