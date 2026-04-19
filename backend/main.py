from dotenv import load_dotenv
load_dotenv()

import os

import opik
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def _safe_track(*args, **kwargs):
    try:
        return opik.track(*args, **kwargs)
    except Exception:
        def noop(fn): return fn
        return noop


app = FastAPI(title="Stripe API Scout")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

SYSTEM_PROMPT = (
    "You are Stripe API Scout, a technical assistant specialized in Stripe's API documentation.\n\n"
    "Answer questions based solely on the provided context from Stripe's official documentation. "
    "Include code examples when relevant to the question. "
    "If the provided context does not contain the answer, respond with: "
    "\"I don't have information about that in Stripe's documentation.\" "
    "Never make up information or answer from general knowledge."
)


class AskRequest(BaseModel):
    question: str
    session_id: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str


@_safe_track(name="ask")
def _run_ask(question: str, session_id: str) -> tuple[str, list[str]]:
    query_vector = embeddings.embed_query(question)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    matches = results.get("matches", [])
    context = "\n\n---\n\n".join(m["metadata"]["text"] for m in matches if m.get("metadata", {}).get("text"))
    sources = list(dict.fromkeys(
        m["metadata"]["source"]
        for m in matches
        if m.get("metadata", {}).get("source")
    ))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context from Stripe documentation:\n\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    return response.content, sources


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Stripe API Scout"}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    answer, sources = _run_ask(request.question, request.session_id)
    return AskResponse(answer=answer, sources=sources, session_id=request.session_id)
