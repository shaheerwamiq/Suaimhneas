"""
Single-file demo: a LangChain + LangGraph "support ticket triage" agent,
modeled on the classifier -> 3-path routing -> synthesizer architecture
described in FLOW_AI.pdf (Traceability Matrix Agent), scaled down to
something you can run in one file with no external datasets.

Real system                          -> This demo
------------------------------------    ------------------------------------
multilingual-e5 (OT Aviator)         -> sentence-transformers (HuggingFace)
llama-4-17b-maverick (OT Aviator)    -> Gemini (langchain_google_genai)
1,000 tickets JSON + 7,686 OTTR xlsx -> a handful of in-memory sample tickets
mmarco-rerank cross-encoder          -> omitted (single FAISS score used directly)
FastAPI + SSE + Angular UI           -> plain terminal loop

Architecture kept the same:
    CLASSIFIER (embed + FAISS search) decides one of 3 paths by score:
        score >= 0.65 -> PATH A: EXISTING issue  -> ticket_retriever -> synthesizer
        score >= 0.40 -> PATH B: NEW issue        -> root_cause_reasoner -> synthesizer
        score <  0.40 -> PATH C: FALLBACK          -> END (no LLM calls)

Requirements:
    langchain_community
    langchain_google_genai
    python-dotenv
    langchain_core
    pydantic
    langchain_huggingface
    sentence-transformers
    faiss-cpu
    pypdf
    bs4
    langgraph
    langchain-text-splitters

Setup:
    1. pip install -r requirements.txt
    2. .env file with: GOOGLE_API_KEY=your_key_here
    3. python triage_agent.py
"""

import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# Same thresholds as classifier.py in the real system
SCORE_EXISTING = 0.65
SCORE_NEW = 0.40


# ---------------------------------------------------------------------------
# PART 1 — DATA INGESTION (mirrors ticket_ingester.py, hugely simplified)
# ---------------------------------------------------------------------------
# In the real system this reads US_Calamus_1_jan_2024.json (1,000 tickets).
# Here we hardcode a handful of sample "past tickets" so the file is self
# contained. Swap this out for your own loader (CSV/JSON/DB) as needed.
SAMPLE_TICKETS = [
    {
        "id": "TCK-101",
        "name": "Password field not saving on draft",
        "description": "<p>When adding a password to a signature template and "
        "saving as <b>draft</b>, no confirmation banner appears and the "
        "password is silently dropped.</p>",
        "resolution_hint": "Draft-save handler skipped the password field; "
        "fixed by including it in the partial payload.",
    },
    {
        "id": "TCK-102",
        "name": "Signature template thumbnail not refreshing",
        "description": "<p>After editing a signature template, the thumbnail "
        "preview shows the old version until a full page refresh.</p>",
        "resolution_hint": "Frontend cache key wasn't invalidated on save.",
    },
    {
        "id": "TCK-103",
        "name": "Export to PDF times out for large documents",
        "description": "<p>Documents over 50 pages fail to export with a "
        "timeout error after ~30 seconds.</p>",
        "resolution_hint": "Increased worker timeout and added chunked "
        "rendering for large page counts.",
    },
]


def build_embedding_text(ticket: dict) -> str:
    """STEP 1 from ticket_ingester.py: strip HTML, build structured text."""
    clean_description = BeautifulSoup(ticket["description"], "html.parser").get_text()
    return f"Title: {ticket['name']}\nDescription: {clean_description.strip()}"


def build_tickets_vectorstore() -> FAISS:
    docs = [
        Document(
            page_content=build_embedding_text(t),
            metadata={"id": t["id"], "name": t["name"], "resolution_hint": t["resolution_hint"]},
        )
        for t in SAMPLE_TICKETS
    ]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"Indexed {len(docs)} sample tickets into FAISS (ve_tickets equivalent).")
    return vectorstore


# ---------------------------------------------------------------------------
# PART 2 — LangGraph state (mirrors agent/state.py)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    request_text: str
    classification: str          # "existing" | "new" | "fallback"
    best_score: float
    matched_ticket: Optional[dict]
    root_cause: Optional[str]
    solution: Optional[str]
    confidence: float


tickets_store = build_tickets_vectorstore()
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# NODE 1 — CLASSIFIER (mirrors nodes/classifier.py)
# ---------------------------------------------------------------------------
def classifier_node(state: AgentState) -> dict:
    results = tickets_store.similarity_search_with_relevance_scores(
        state["request_text"], k=1
    )
    if not results:
        return {"classification": "fallback", "best_score": 0.0, "matched_ticket": None}

    doc, score = results[0]
    print(f"[classifier] best match: {doc.metadata['name']!r} (score={score:.2f})")

    if score >= SCORE_EXISTING:
        classification = "existing"
    elif score >= SCORE_NEW:
        classification = "new"
    else:
        classification = "fallback"

    return {
        "classification": classification,
        "best_score": score,
        "matched_ticket": doc.metadata if score >= SCORE_NEW else None,
    }


def route_after_classifier(state: AgentState) -> str:
    return {"existing": "ticket_retriever", "new": "root_cause_reasoner", "fallback": "fallback"}[
        state["classification"]
    ]


# ---------------------------------------------------------------------------
# NODE 2A — TICKET RETRIEVER (PATH A, mirrors nodes/ticket_retriever.py)
# ---------------------------------------------------------------------------
TICKET_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """A new support request has come in. A similar past ticket was found.
Summarize in 2-3 sentences what the likely issue and fix are, based on the
past ticket's resolution hint.

New request: {request_text}

Matched past ticket: {ticket_name}
Resolution hint: {resolution_hint}

Summary:"""
)


def ticket_retriever_node(state: AgentState) -> dict:
    ticket = state["matched_ticket"]
    chain = TICKET_EXTRACT_PROMPT | llm
    response = chain.invoke(
        {
            "request_text": state["request_text"],
            "ticket_name": ticket["name"],
            "resolution_hint": ticket["resolution_hint"],
        }
    )
    return {"root_cause": response.content}


# ---------------------------------------------------------------------------
# NODE 2B — ROOT CAUSE REASONER (PATH B, mirrors nodes/root_cause_reasoner.py)
# ---------------------------------------------------------------------------
ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_template(
    """You are a root cause analyst. No closely-matching past ticket exists for
this request, so hypothesize the likely root cause in 1-2 sentences. Do not
suggest a fix yet, only diagnose.

Request: {request_text}"""
)


def root_cause_reasoner_node(state: AgentState) -> dict:
    chain = ROOT_CAUSE_PROMPT | llm
    response = chain.invoke({"request_text": state["request_text"]})
    return {"root_cause": response.content}


# ---------------------------------------------------------------------------
# NODE 3 — SYNTHESIZER (shared, mirrors nodes/synthesizer.py)
# ---------------------------------------------------------------------------
SYNTH_PROMPT = ChatPromptTemplate.from_template(
    """Write one clear paragraph explaining the likely issue and what the
developer should investigate next.

Request: {request_text}
Classification: {classification}
Analysis so far: {root_cause}"""
)


def synthesizer_node(state: AgentState) -> dict:
    chain = SYNTH_PROMPT | llm
    response = chain.invoke(
        {
            "request_text": state["request_text"],
            "classification": state["classification"],
            "root_cause": state["root_cause"],
        }
    )
    # Same confidence formula spirit as synthesizer.py (simplified: FAISS score only)
    confidence = round(state["best_score"], 2) if state["classification"] != "fallback" else 0.0
    return {"solution": response.content, "confidence": confidence}


# ---------------------------------------------------------------------------
# FALLBACK NODE (PATH C, mirrors synthesizer.py::fallback_node)
# ---------------------------------------------------------------------------
def fallback_node(state: AgentState) -> dict:
    print("[fallback] No relevant match found — skipping all LLM calls.")
    return {
        "solution": "Insufficient match found in the knowledge base — human review recommended.",
        "confidence": 0.0,
        "root_cause": None,
    }


# ---------------------------------------------------------------------------
# PART 3 — BUILD THE GRAPH (mirrors agent/graph.py)
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classifier", classifier_node)
    g.add_node("ticket_retriever", ticket_retriever_node)   # PATH A
    g.add_node("root_cause_reasoner", root_cause_reasoner_node)  # PATH B
    g.add_node("fallback", fallback_node)                   # PATH C
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "classifier")
    g.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "ticket_retriever": "ticket_retriever",
            "root_cause_reasoner": "root_cause_reasoner",
            "fallback": "fallback",
        },
    )
    g.add_edge("ticket_retriever", "synthesizer")       # PATH A converges
    g.add_edge("root_cause_reasoner", "synthesizer")    # PATH B converges
    g.add_edge("fallback", END)                         # PATH C skips synthesizer
    g.add_edge("synthesizer", END)

    return g.compile()


# ---------------------------------------------------------------------------
# PART 4 — MAIN
# ---------------------------------------------------------------------------
def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY not found. Add it to a .env file.")

    graph = build_graph()

    print("\nTriage agent ready. Try something close to a sample ticket, and")
    print("something totally unrelated, to see all 3 paths. Type 'exit' to quit.\n")

    while True:
        request_text = input("Support request: ").strip()
        if request_text.lower() in {"exit", "quit"}:
            break
        if not request_text:
            continue

        result = graph.invoke(
            {
                "request_text": request_text,
                "classification": "",
                "best_score": 0.0,
                "matched_ticket": None,
                "root_cause": None,
                "solution": None,
                "confidence": 0.0,
            }
        )

        print(f"\n--- classification: {result['classification']} "
              f"(confidence: {result['confidence']}) ---")
        print(result["solution"])
        print()


if __name__ == "__main__":
    main()