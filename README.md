# rag-langchain-demo

I built this small end-to-end RAG skeleton for Freelancer project **End-to-End RAG (40686539)**. It is a working Python demo: upload or point at a document, chunk it, store embeddings in local ChromaDB, then ask questions over the retrieved context.

Stack: **FastAPI**, **LangChain** (LCEL-style retrieval helpers + optional LangGraph stub), **ChromaDB**, and a **no-API-key embedding backend** so `pip install` + `uvicorn` works offline for the demo.

## What you get

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness + embedding backend |
| `/ingest` | POST | Upload a PDF/DOCX/TXT **or** pass a local `path` form field |
| `/query` | POST | Ask a question; returns answer + source snippets |
| `/docs` | GET | OpenAPI UI |

Flow:

1. **Ingest** — load TXT / PDF / DOCX → recursive character chunking → embed → persist in Chroma.
2. **Query** — embed the question → similarity search (`TOP_K`) → grounded extractive answer from retrieved chunks.

By default there is **no paid LLM call**. The query path returns an extractive answer assembled from retrieved chunks so the API stays runnable without OpenAI/Groq keys. The LangGraph stub under `app/langgraph_stub.py` shows where a real generate node would plug in.

## Quick start (local)

```bash
cd rag-langchain-demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: copy env defaults
cp .env.example .env

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Ingest a sample and query:

```bash
curl -F "file=@samples/onboarding_faq.txt" http://127.0.0.1:8000/ingest

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Which regions are available?"}'
```

Or ingest by server-local path:

```bash
curl -F "path=./samples/company_overview.txt" http://127.0.0.1:8000/ingest
```

Run the smoke test:

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
# API on http://127.0.0.1:8000
```

## Embeddings and LLMs

`EMBEDDING_BACKEND` (env) controls embeddings:

| Value | Notes |
|-------|--------|
| `fake` **(default)** | Deterministic local vectors. No download, no API key. Good for demos/CI. |
| `sentence-transformers` | Uses `all-MiniLM-L6-v2` via HuggingFace (downloads model on first run). |
| `openai` | Needs `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_EMBEDDING_MODEL`). |
| `ollama` | Needs local Ollama; set `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`. |

**Groq / OpenAI chat:** Groq is great for generation but does not ship a public embedding model for this demo. Keep embeddings on `fake`, `sentence-transformers`, `openai`, or `ollama`, then replace the extractive step in `app/rag.py` (or the generate node in `app/langgraph_stub.py`) with `ChatGroq` / `ChatOpenAI` / `ChatOllama`.

Example env for OpenAI-compatible embeddings:

```bash
EMBEDDING_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Example for Ollama:

```bash
EMBEDDING_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Chunking / retrieval knobs

Set in `.env` or the process environment:

- `CHUNK_SIZE` (default `500`)
- `CHUNK_OVERLAP` (default `80`)
- `TOP_K` (default `4`)
- `CHROMA_PERSIST_DIR` (default `./data/chroma`)

## Layout

```
app/
  main.py            # FastAPI routes
  config.py          # settings from env
  embeddings.py      # fake / ST / OpenAI / Ollama
  loaders.py         # TXT, PDF, DOCX
  rag.py             # chunk + Chroma + query
  langgraph_stub.py  # optional retrieve→generate stub
samples/             # small TXT docs for a first ingest
Dockerfile
docker-compose.yml
requirements.txt
```

## Notes

- Chroma persists under `data/chroma` (gitignored).
- This repo is a **demo skeleton** for the Freelancer RAG brief — solid structure you can extend with a real LLM, auth, and richer evaluation.
- I kept dependencies pinned so a clean `pip install` is reproducible.

— Tanvir Alam
