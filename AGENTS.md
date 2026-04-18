# Stripe API Scout — Agent Guide

## Project Overview

Stripe API Scout is a RAG agent that lets developers ask natural-language questions about Stripe's API documentation and receive grounded, cited answers. It indexes Stripe's full API reference using OpenAI embeddings stored in a local Chroma vector store, then answers queries by retrieving the most relevant doc sections and passing them to an LLM for generation. Built to be A2A-compatible as a future iteration.

## Architecture

Two-phase pipeline:

**Phase 1 — Ingestion (runs once, or when docs are updated)**
1. Crawl Stripe API reference docs
2. Chunk text into retrieval-friendly segments
3. Embed each chunk with OpenAI `text-embedding-3-small`
4. Store embeddings + metadata in local Chroma vector store

**Phase 2 — Query (every request)**
1. Embed the user's question with OpenAI `text-embedding-3-small`
2. Similarity search in Chroma → retrieve top K chunks
3. Pass chunks + question to Groq `llama-3.3-70b-versatile`
4. Return grounded answer with citations to the source doc sections

## Key Files

| File | Purpose |
|------|---------|
| `backend/ingest.py` | Crawl, chunk, embed, and store. Run once before deploying. |
| `backend/main.py` | FastAPI app — query endpoint + OPIK instrumentation |
| `backend/chroma_db/` | Local Chroma vector store — **never commit to git** |
| `frontend/index.html` | Single-page chat UI |

## Environment Variables

```
OPENAI_API_KEY=
GROQ_API_KEY=
OPIK_API_KEY=
OPIK_PROJECT_NAME=stripe-api-scout
OPIK_WORKSPACE=ra-l-vallejo
```

Never commit API keys. Use a `.env` file locally and set these as secrets in your hosting environment.

## OPIK Instrumentation

- Configure exclusively via environment variables — never call `opik.configure()` in code
- Use the `_safe_track` decorator pattern for all traced functions

## Critical Rules

- `backend/chroma_db/` must be in `.gitignore` — never commit the vector store
- Always run `ingest.py` before starting the server for the first time
- Re-run `ingest.py` whenever Stripe's API docs are updated

## Known Gotchas

None yet — will be updated as we build.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma (local) |
| Generation | Groq `llama-3.3-70b-versatile` |
| RAG chain | LangChain |
| Backend | FastAPI |
| Observability | OPIK by Comet |
| Frontend | Single HTML file |
| Backend hosting | Render |
| Frontend hosting | Vercel |
