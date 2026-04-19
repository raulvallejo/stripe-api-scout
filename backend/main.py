from dotenv import load_dotenv
load_dotenv()

import opik
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
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
vector_store = Chroma(
    collection_name="stripe_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

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
    docs = vector_store.similarity_search(question, k=5)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = list(dict.fromkeys(
        doc.metadata["source"]
        for doc in docs
        if doc.metadata.get("source")
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
