# Stripe API Scout

**Ask anything about Stripe's API. Get grounded answers with sources.**

A RAG agent that indexes Stripe's full API documentation and answers developer questions with cited, grounded responses. Built with OpenAI embeddings, Pinecone, Groq, and LangChain.

> [RAG Agent Template — coming soon]

---

## How it works

**Ingestion (one time)**
The agent crawls Stripe's API reference, splits it into chunks, and stores them as embeddings in Pinecone (cloud vector store). You run this once before deploying, and again whenever the Stripe docs are updated.

**Query (every request)**
When a developer asks a question, the agent finds the most relevant doc sections, passes them to an LLM, and returns an answer grounded in those sources — with citations so you can verify the response.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Pinecone |
| Generation | Groq `llama-3.3-70b-versatile` |
| RAG chain | LangChain |
| Observability | OPIK by Comet |
| Backend | FastAPI |
| Frontend hosting | Vercel |
| Backend hosting | Render |

---

## Live demo

[coming soon]

---

MIT License

Built by [Raul Vallejo](https://linkedin.com/in/raulvallejo) — PM building and shipping production AI agents.
