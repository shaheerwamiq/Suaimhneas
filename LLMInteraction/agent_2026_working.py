"""
Single-file demo: LangChain + LangGraph "support ticket triage" agent.

THIS FILE HAS THE SAME GRAPH SHAPE AS triage_agent_full.py:
    classifier -> {ticket_retriever | root_cause_reasoner | fallback} -> synthesizer -> END
Same node names, same routing thresholds, same main() control flow order.
Nothing about WHEN each stage runs, or WHICH path gets taken, has changed.

What changed is what happens INSIDE each stage, plus a couple of things
wrapped AROUND the graph (guardrails, human review, streaming) that don't
touch its internals. Each addition is marked "NEW:" so you can diff this
against the original mentally.

    1. LangSmith tracing & observability           (unchanged, richer tags)
    2. Cost & token tracking                        (unchanged)
    3. Latency SLA monitoring                       (unchanged)
    4. LLM-as-judge evaluation          NEW: structured output, not raw JSON
    5. Golden datasets & regression testing          (unchanged)
    6. Human feedback loop                           (unchanged)
       NEW: human-in-the-loop REVIEW GATE (different from feedback -- this
       happens before you'd act on a low-confidence result, not after)
    7. Prompt A/B testing                            (unchanged)
    8. Drift detection                               (unchanged)
    NEW: hybrid search + re-ranking in classifier_node
    NEW: agentic tool-calling sub-loop in root_cause_reasoner_node (PATH B)
    NEW: MCP-pluggable tool source for that loop (optional)
    NEW: episodic memory -- resolved cases feed back into future context
    NEW: explicit context engineering (assemble_context helper)
    NEW: input/output guardrails wrapped around graph.invoke()
    NEW: --stream flag using LangGraph's native message streaming

Requirements (add to requirements.txt):
    langchain_community langchain_google_genai python-dotenv langchain_core
    pydantic langchain_huggingface sentence-transformers faiss-cpu pypdf bs4
    langgraph langchain-text-splitters langsmith
    rank_bm25          # NEW: hybrid search (optional -- degrades gracefully)
    mcp                # NEW: optional MCP tool source (optional)

Setup:
    1. pip install -r requirements.txt
    2. .env: GOOGLE_API_KEY=..., optional LANGCHAIN_API_KEY=...
    3. Normal run:            python triage_agent_advanced.py
       Regression test run:   python triage_agent_advanced.py --test
       A/B report only:       python triage_agent_advanced.py --ab-report
       Streamed answers:      python triage_agent_advanced.py --stream
"""

import json
import os
import random
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

# NEW: optional deps -- hybrid search / rerank / MCP all degrade gracefully
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # pip install rank_bm25

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

SCORE_EXISTING = 0.65
SCORE_NEW = 0.40


def is_llm_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "resource_exhausted", "quota", "rate limit", "429",
            "exceeded your current quota", "not_found", "model",
            "no longer available", "unsupported",
        ]
    )


def invoke_llm_or_fallback(chain, payload: dict, fallback_text: str, node_name: str):
    try:
        response = chain.invoke(payload)
    except Exception as exc:
        if is_llm_quota_error(exc):
            print(f"[llm] {node_name} skipped due to API quota/rate limit: {exc}")
            return fallback_text, None
        raise
    token_tracker.record(response, node_name)
    return response.content, response


# =============================================================================
# 1. LANGSMITH TRACING & OBSERVABILITY (unchanged, tags enriched below)
# =============================================================================
TRACING_ENABLED = bool(os.getenv("LANGCHAIN_API_KEY"))
if TRACING_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "triage-agent-advanced")
    print(f"[langsmith] Tracing ON — project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("[langsmith] LANGCHAIN_API_KEY not set — tracing disabled.")

try:
    from langsmith import Client as LangSmithClient
    from langchain_core.tracers.context import collect_runs

    langsmith_client = LangSmithClient() if TRACING_ENABLED else None
except ImportError:
    langsmith_client = None
    collect_runs = None
    if TRACING_ENABLED:
        print("[langsmith] `langsmith` package not installed — run `pip install langsmith`.")


# =============================================================================
# 2. COST & TOKEN TRACKING (unchanged)
# =============================================================================
PRICE_PER_1M_INPUT = 0.075
PRICE_PER_1M_OUTPUT = 0.30


class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.total_cost = 0.0
        self.calls: List[dict] = []

    def record(self, response, node_name: str) -> float:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT + (
            output_tokens / 1_000_000
        ) * PRICE_PER_1M_OUTPUT
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_cost += cost
        self.calls.append({"node": node_name, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost})
        print(f"[tokens] {node_name}: in={input_tokens} out={output_tokens} cost=${cost:.5f}")
        return cost

    def session_summary(self) -> str:
        return f"[tokens] SESSION TOTAL — input={self.total_input} output={self.total_output} cost=${self.total_cost:.4f}"


token_tracker = TokenTracker()


# =============================================================================
# 3. LATENCY SLA MONITORING (unchanged)
# =============================================================================
SLA_SECONDS = {
    "classifier": 3.0,
    "ticket_retriever": 8.0,
    "root_cause_reasoner": 12.0,  # NEW: raised slightly -- this node now does a tool loop
    "synthesizer": 5.0,
}


def with_timing(node_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: dict) -> dict:
            start = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - start
            sla = SLA_SECONDS.get(node_name)
            if sla and elapsed > sla:
                print(f"[SLA BREACH] {node_name} took {elapsed:.2f}s (limit {sla:.1f}s)")
            else:
                print(f"[timing] {node_name}: {elapsed:.2f}s")
            timings = dict(state.get("timings", {}))
            timings[node_name] = round(elapsed, 3)
            result["timings"] = timings
            return result
        return wrapper
    return decorator


# =============================================================================
# DATA INGESTION (same sample data)
# =============================================================================
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
    clean_description = BeautifulSoup(ticket["description"], "html.parser").get_text() # t is primarily used to parse and extract data from HTML and XML documents, making it a popular tool for web scraping.
    return f"Title: {ticket['name']}\nDescription: {clean_description.strip()}"


# =============================================================================
# NEW: HYBRID SEARCH + RE-RANKING — wraps retrieval, keeps the same output
# contract classifier_node already expects: a list of (Document, score).
# =============================================================================
class TicketKnowledgeBase:
    def __init__(self, tickets: List[dict]):
        self.tickets = tickets
        self.docs = [
            Document(
                page_content=build_embedding_text(t),
                metadata={"id": t["id"], "name": t["name"], "resolution_hint": t["resolution_hint"]},
            )
            for t in tickets
        ]
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)

        if BM25Okapi:
            tokenized = [d.page_content.lower().split() for d in self.docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
            print("[hybrid] rank_bm25 not installed — falling back to vector-only search.")

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if CrossEncoder else None
        if not self.reranker:
            print("[rerank] sentence-transformers CrossEncoder unavailable — skipping rerank step.")

        print(f"Indexed {len(self.docs)} sample tickets (hybrid={'on' if self.bm25 else 'off'}, "
              f"rerank={'on' if self.reranker else 'off'}).")

    def _rrf_fuse(self, ranked_lists: List[List[int]], k: int = 60) -> List[int]:
        scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def best_match(self, query: str):
        """Returns (Document, score) for the single best match -- same shape
        classifier_node's original similarity_search_with_relevance_scores
        call returned, so downstream code needs zero changes."""
        vector_hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=5)
        candidates = [d for d, _ in vector_hits]
        vector_scores = {self.docs.index(d): s for d, s in vector_hits if d in self.docs}

        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.lower().split())
            bm25_rank = sorted(range(len(self.docs)), key=lambda i: bm25_scores[i], reverse=True)[:5]
            vector_rank = [self.docs.index(d) for d in candidates if d in self.docs]
            fused_order = self._rrf_fuse([vector_rank, bm25_rank])
            candidates = [self.docs[i] for i in fused_order[:5]]

        if self.reranker and candidates:
            pairs = [(query, d.page_content) for d in candidates]
            rerank_scores = self.reranker.predict(pairs)
            candidates = [d for d, _ in sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)]

        if not candidates:
            return None

        best_doc = candidates[0]
        # Preserve the original FAISS cosine score for that doc so the
        # existing SCORE_EXISTING/SCORE_NEW thresholds still mean the same thing.
        best_score = vector_scores.get(self.docs.index(best_doc), 0.0)
        return best_doc, best_score


# =============================================================================
# NEW: EPISODIC MEMORY — remembers resolved cases, recalled as extra context
# =============================================================================
MEMORY_LOG_PATH = "case_memory.jsonl"


class CaseMemory:
    """Long-term, cross-run memory of resolved cases. Backed by the same
    FAISS pattern as retrieval -- memory IS retrieval, just over a different,
    growing corpus that accumulates as the agent runs."""

    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        seed = [Document(page_content="(memory initialized)", metadata={"kind": "system"})]
        self._store = FAISS.from_documents(seed, embeddings)
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(MEMORY_LOG_PATH):
            return
        with open(MEMORY_LOG_PATH, "r", encoding="utf-8") as f:
            docs = []
            for line in f:
                entry = json.loads(line)
                docs.append(Document(page_content=entry["text"], metadata={"kind": "episodic", **entry}))
            if docs:
                self._store.add_documents(docs)
                print(f"[memory] Loaded {len(docs)} past cases from {MEMORY_LOG_PATH}")

    def remember(self, request_text: str, classification: str, solution: str):
        text = f"Request: {request_text}\nClassification: {classification}\nResolution: {solution}"
        entry = {
            "text": text,
            "request_text": request_text,
            "classification": classification,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._store.add_documents([Document(page_content=text, metadata=entry)])
        with open(MEMORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def recall(self, query: str, k: int = 2) -> List[Document]:
        return self._store.similarity_search(query, k=k)


case_memory = CaseMemory()


# =============================================================================
# NEW: CONTEXT ENGINEERING — assemble prompt context deliberately, don't
# just string-concatenate everything available.
# =============================================================================
def assemble_context(matched_ticket_summary: str, recalled_cases: List[Document], max_recalled: int = 2) -> str:
    recalled_text = "\n---\n".join(d.page_content for d in recalled_cases[:max_recalled] if d.metadata.get("kind") == "episodic")
    if not recalled_text:
        return matched_ticket_summary
    return f"{matched_ticket_summary}\n\nSimilar past cases (for extra context, may not be identical):\n{recalled_text}"


# =============================================================================
# NEW: TOOLS for the agentic sub-loop in PATH B (root_cause_reasoner)
# =============================================================================
@tool
def search_codebase(query: str) -> str:
    """Search the product codebase for functions/files related to the query."""
    # Mock implementation -- swap for a real code-search index or an MCP
    # server call (see get_mcp_tools() below) without changing the loop.
    return f"[mock code search] Found 2 files possibly related to {query!r}: signer.py, draft_handler.py"


@tool
def search_confluence_docs(query: str) -> str:
    """Search internal design docs for background on a feature or component."""
    return f"[mock docs search] Found 1 design doc mentioning {query!r}."


def get_mcp_tools() -> list:
    """
    NEW: MCP-pluggable extension point. If an MCP server (e.g. the Jira mock
    server from earlier) is configured, its tools get added to the same list
    the agentic loop below already uses -- the loop doesn't care whether a
    tool is a native @tool or came from MCP, same as SECTION 5 of the
    playbook file. Returns [] if mcp isn't installed/configured, so this is
    purely additive and never breaks the file if you skip it.
    """
    try:
        import mcp  # noqa: F401 -- presence check only; real wiring is async
        # Real wiring would use mcp.ClientSession + stdio_client here, as
        # shown in mcp_client_test.py, then wrap each remote tool as a
        # LangChain @tool. Omitted here to keep this node's execution
        # synchronous and simple; see that file for the async pattern.
        return []
    except ImportError:
        return []


LOCAL_TOOLS = [search_codebase, search_confluence_docs]
ALL_TOOLS = LOCAL_TOOLS + get_mcp_tools()


class AgentState(TypedDict):
    request_text: str
    classification: str
    best_score: float
    matched_ticket: Optional[dict]
    root_cause: Optional[str]
    solution: Optional[str]
    confidence: float
    timings: dict
    prompt_variant: str
    tool_trace: List[str]  # NEW: records which tools the agentic loop called


tickets_kb = TicketKnowledgeBase(SAMPLE_TICKETS)
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# =============================================================================
# NODE 1 — CLASSIFIER  (same routing logic, hybrid+reranked retrieval inside)
# =============================================================================
@with_timing("classifier")
def classifier_node(state: AgentState) -> dict:
    result = tickets_kb.best_match(state["request_text"])
    if not result:
        return {"classification": "fallback", "best_score": 0.0, "matched_ticket": None}

    doc, score = result
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
    # NOTE: this is the routing logic — the "supervisor" decision. Left
    # untouched, exactly as requested. It IS already a lightweight
    # multi-agent-style dispatch (existing/new/fallback = 3 specialists);
    # no separate multi-agent layer was added on top of it.
    return {"existing": "ticket_retriever", "new": "root_cause_reasoner", "fallback": "fallback"}[
        state["classification"]
    ]


# =============================================================================
# NODE 2A — TICKET RETRIEVER (PATH A)  — NEW: memory-enriched context
# =============================================================================
TICKET_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """A new support request has come in. A similar past ticket was found.
Summarize in 2-3 sentences what the likely issue and fix are, based on the
past ticket's resolution hint and any similar past cases provided.

New request: {request_text}

Context:
{context}

Summary:"""
)


@with_timing("ticket_retriever")
def ticket_retriever_node(state: AgentState) -> dict:
    ticket = state["matched_ticket"]
    ticket_summary = f"Matched past ticket: {ticket['name']}\nResolution hint: {ticket['resolution_hint']}"

    # NEW: recall similar past resolved cases and fold them into context
    recalled = case_memory.recall(state["request_text"])
    context = assemble_context(ticket_summary, recalled)

    chain = TICKET_EXTRACT_PROMPT | llm
    fallback_text = "Unable to reach the AI model right now due to quota limits; please retry shortly."
    root_cause_text, _ = invoke_llm_or_fallback(
        chain, {"request_text": state["request_text"], "context": context}, fallback_text, "ticket_retriever"
    )
    return {"root_cause": root_cause_text}


# =============================================================================
# NODE 2B — ROOT CAUSE REASONER (PATH B)
# NEW: this is now a bounded agentic tool-calling loop, not one flat call.
# The LLM decides whether it needs to search the codebase/docs before
# diagnosing -- same "reason -> act -> observe -> reason" pattern as any
# ReAct agent, just capped at a small number of iterations so a triage node
# can't run away with unbounded LLM calls.
# =============================================================================
ROOT_CAUSE_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a root cause analyst investigating a support request with no "
        "close match in past tickets. You have tools to search the codebase and "
        "internal docs. Use them if they'd help; otherwise reason directly. "
        "Once you have enough information, respond with a 1-2 sentence root "
        "cause hypothesis and STOP calling tools."
    )
)

MAX_TOOL_ITERATIONS = 3


@with_timing("root_cause_reasoner")
def root_cause_reasoner_node(state: AgentState) -> dict:
    messages = [ROOT_CAUSE_SYSTEM_PROMPT, HumanMessage(content=state["request_text"])]
    tool_trace: List[str] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            if is_llm_quota_error(exc):
                print(f"[llm] root_cause_reasoner skipped due to quota: {exc}")
                return {"root_cause": "Unable to reach the AI model right now due to quota limits.", "tool_trace": tool_trace}
            raise

        token_tracker.record(response, "root_cause_reasoner")
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return {"root_cause": response.content, "tool_trace": tool_trace}

        for call in tool_calls:
            tool_fn = next((t for t in ALL_TOOLS if t.name == call["name"]), None)
            result = tool_fn.invoke(call["args"]) if tool_fn else f"Unknown tool: {call['name']}"
            tool_trace.append(f"{call['name']}({call['args']})")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Loop budget exhausted without a final answer -- degrade gracefully.
    return {"root_cause": "Investigation inconclusive after available tool calls.", "tool_trace": tool_trace}


# =============================================================================
# 7. PROMPT A/B TESTING (unchanged)
# =============================================================================
SYNTH_PROMPT_A = ChatPromptTemplate.from_template(
    """Write one clear paragraph explaining the likely issue and what the
developer should investigate next.

Request: {request_text}
Classification: {classification}
Analysis so far: {root_cause}"""
)

SYNTH_PROMPT_B = ChatPromptTemplate.from_template(
    """Summarize this triage result as three short bullet points: (1) likely
issue, (2) suggested next step, (3) risk if not fixed.

Request: {request_text}
Classification: {classification}
Analysis so far: {root_cause}"""
)

SYNTH_VARIANTS = {"A": SYNTH_PROMPT_A, "B": SYNTH_PROMPT_B}
ab_results = {"A": {"count": 0, "positive": 0}, "B": {"count": 0, "positive": 0}}


def choose_prompt_variant() -> str:
    return random.choice(["A", "B"])


def print_ab_report():
    print("\n[A/B report]")
    for variant, stats in ab_results.items():
        count = stats["count"]
        rate = (stats["positive"] / count * 100) if count else 0.0
        print(f"  Variant {variant}: {count} rated responses, {rate:.0f}% positive")
    print()


# =============================================================================
# NODE 3 — SYNTHESIZER (unchanged logic; NEW: remembers the resolved case)
# =============================================================================
@with_timing("synthesizer")
def synthesizer_node(state: AgentState) -> dict:
    variant = choose_prompt_variant()
    prompt = SYNTH_VARIANTS[variant]
    chain = prompt | llm
    fallback_text = "The AI model is temporarily unavailable due to quota limits; please retry later."
    solution_text, _ = invoke_llm_or_fallback(
        chain,
        {"request_text": state["request_text"], "classification": state["classification"], "root_cause": state["root_cause"]},
        fallback_text,
        "synthesizer",
    )
    confidence = round(state["best_score"], 2) if state["classification"] != "fallback" else 0.0
    print(f"[ab] using prompt variant: {variant}")

    # NEW: write this resolved case into episodic memory for future recall
    case_memory.remember(state["request_text"], state["classification"], solution_text)

    return {"solution": solution_text, "confidence": confidence, "prompt_variant": variant}


def fallback_node(state: AgentState) -> dict:
    print("[fallback] No relevant match found — skipping all LLM calls.")
    return {
        "solution": "Insufficient match found in the knowledge base — human review recommended.",
        "confidence": 0.0,
        "root_cause": None,
        "prompt_variant": "none",
        "tool_trace": [],
    }


# =============================================================================
# BUILD THE GRAPH (identical structure/edges to the original)
# Reasoner = figures out why something is happening — diagnosis, not action. Input: a problem. Output: an explanation/hypothesis.
# Synthesizer = combines multiple pieces of information into one final, coherent output — the last step before showing something to a human. Input: several fragments. Output: one polished answer.
# =============================================================================
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classifier", classifier_node)
    g.add_node("ticket_retriever", ticket_retriever_node)
    g.add_node("root_cause_reasoner", root_cause_reasoner_node)
    g.add_node("fallback", fallback_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "classifier")
    g.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"ticket_retriever": "ticket_retriever", "root_cause_reasoner": "root_cause_reasoner", "fallback": "fallback"},
    )
    g.add_edge("ticket_retriever", "synthesizer")
    g.add_edge("root_cause_reasoner", "synthesizer")
    g.add_edge("fallback", END)
    g.add_edge("synthesizer", END)
    return g.compile()


def initial_state(request_text: str) -> AgentState:
    return {
        "request_text": request_text,
        "classification": "",
        "best_score": 0.0,
        "matched_ticket": None,
        "root_cause": None,
        "solution": None,
        "confidence": 0.0,
        "timings": {},
        "prompt_variant": "",
        "tool_trace": [],
    }


# =============================================================================
# 4. LLM-AS-JUDGE — NEW: structured output instead of raw JSON parsing
# =============================================================================
class TriageJudgment(BaseModel):
    relevance: int = Field(ge=1, le=5, description="1-5: does it address the request?")
    groundedness: int = Field(ge=1, le=5, description="1-5: consistent with the analysis, not made up?")
    reasoning: str = Field(description="One sentence explaining the scores")


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """Grade this support triage response 1-5 on relevance and groundedness.

Request: {request_text}
Analysis: {root_cause}
Generated solution: {solution}"""
)


def llm_judge(request_text: str, root_cause: Optional[str], solution: Optional[str]) -> Optional[dict]:
    if not solution:
        return None
    structured_llm = llm.with_structured_output(TriageJudgment)
    chain = JUDGE_PROMPT | structured_llm
    try:
        judgment = chain.invoke(
            {"request_text": request_text, "root_cause": root_cause or "(none)", "solution": solution}
        )
    except Exception as exc:
        if is_llm_quota_error(exc):
            print(f"[llm] llm_judge skipped due to quota: {exc}")
            return {"relevance": None, "groundedness": None, "reasoning": "llm_unavailable"}
        raise
    return judgment.model_dump()


# =============================================================================
# 5. GOLDEN DATASETS & REGRESSION TESTING (unchanged)
# =============================================================================
GOLDEN_DATASET = [
    {"request_text": "password on signature template disappears after saving as draft", "expected_classification": "existing"},
    {"request_text": "PDF export hangs and eventually times out on big files", "expected_classification": "existing"},
    {"request_text": "the search results seem to load slower than usual today", "expected_classification": "new"},
    {"request_text": "what's the weather forecast for tomorrow", "expected_classification": "fallback"},
]


def run_regression_tests():
    print("\n=== Regression test run against golden dataset ===\n")
    graph = build_graph()
    passed = 0
    for case in GOLDEN_DATASET:
        result = graph.invoke(initial_state(case["request_text"]))
        ok = result["classification"] == case["expected_classification"]
        passed += int(ok)
        judge = llm_judge(case["request_text"], result["root_cause"], result["solution"])
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] expected={case['expected_classification']!r} actual={result['classification']!r} "
              f"score={result['best_score']:.2f} judge={judge}")
    rate = passed / len(GOLDEN_DATASET) * 100
    print(f"\nRegression pass rate: {passed}/{len(GOLDEN_DATASET)} ({rate:.0f}%)")
    print(token_tracker.session_summary())


# =============================================================================
# 8. DRIFT DETECTION (unchanged)
# =============================================================================
class DriftMonitor:
    def __init__(self, window: int = 20, score_drift_threshold: float = 0.15, fallback_rate_threshold: float = 0.5):
        self.scores: deque = deque(maxlen=window)
        self.classifications: deque = deque(maxlen=window)
        self.baseline_mean: Optional[float] = None
        self.score_drift_threshold = score_drift_threshold
        self.fallback_rate_threshold = fallback_rate_threshold

    def record(self, score: float, classification: str):
        self.scores.append(score)
        self.classifications.append(classification)
        if self.baseline_mean is None:
            if len(self.scores) >= 5:
                self.baseline_mean = statistics.mean(self.scores)
                print(f"[drift] Baseline established: avg score {self.baseline_mean:.2f} over {len(self.scores)} requests")
            return
        current_mean = statistics.mean(self.scores)
        delta = self.baseline_mean - current_mean
        if delta > self.score_drift_threshold:
            print(f"[DRIFT WARNING] Avg match score dropped {delta:.2f} vs baseline "
                  f"({self.baseline_mean:.2f} -> {current_mean:.2f}). Knowledge base may need refreshing.")
        fallback_rate = self.classifications.count("fallback") / len(self.classifications)
        if fallback_rate > self.fallback_rate_threshold:
            print(f"[DRIFT WARNING] {fallback_rate:.0%} of the last {len(self.classifications)} requests "
                  f"fell back to no-match — possible topic drift.")


drift_monitor = DriftMonitor()


# =============================================================================
# 6. HUMAN FEEDBACK LOOP (unchanged)
# =============================================================================
FEEDBACK_LOG_PATH = "feedback_log.jsonl"


def collect_human_feedback(request_text: str, solution: str, prompt_variant: str, run_id: Optional[str]):
    raw = input("Was this helpful? (y/n/skip): ").strip().lower()
    if raw not in {"y", "n"}:
        return
    positive = raw == "y"
    if prompt_variant in ab_results:
        ab_results[prompt_variant]["count"] += 1
        ab_results[prompt_variant]["positive"] += int(positive)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_text": request_text,
        "solution": solution,
        "prompt_variant": prompt_variant,
        "positive": positive,
        "run_id": run_id,
    }
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    if langsmith_client and run_id:
        try:
            langsmith_client.create_feedback(run_id, key="user_helpful", score=int(positive))
            print("[langsmith] Feedback logged to run.")
        except Exception as e:
            print(f"[langsmith] Failed to log feedback: {e}")


# =============================================================================
# NEW: GUARDRAILS — wrapped AROUND graph.invoke(), not inside it
# =============================================================================
def input_guardrail(text: str) -> Optional[str]:
    blocked = ["ignore previous instructions", "system prompt"]
    lowered = text.lower()
    for pattern in blocked:
        if pattern in lowered:
            return f"blocked: input contains {pattern!r}"
    return None


def output_guardrail(text: str) -> Optional[str]:
    import re
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        return "blocked: output contains a possible SSN-shaped pattern"
    return None


# =============================================================================
# NEW: HUMAN-IN-THE-LOOP REVIEW GATE — distinct from feedback (#6): this
# happens BEFORE the low-confidence result is treated as final, not after.
# =============================================================================
REVIEW_CONFIDENCE_THRESHOLD = 0.5


def human_review_gate(result: dict) -> dict:
    if result["classification"] == "fallback" or result["confidence"] >= REVIEW_CONFIDENCE_THRESHOLD:
        return result
    print(f"\n[review] Confidence is low ({result['confidence']:.2f}) — flagging for human review before use.")
    decision = input("Approve this result for the requester anyway? (y/n): ").strip().lower()
    if decision != "y":
        result["solution"] = "PENDING HUMAN REVIEW — a low-confidence result was not approved for release."
    return result


# =============================================================================
# NEW: STREAMING — same graph, LangGraph's native token/message streaming
# =============================================================================
def stream_response(graph, state: AgentState, config: dict):
    print("\n[streaming] ", end="", flush=True)
    for message_chunk, metadata in graph.stream(state, config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "synthesizer" and getattr(message_chunk, "content", None):
            print(message_chunk.content, end="", flush=True)
    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY not found. Add it to a .env file.")

    if "--test" in sys.argv:
        run_regression_tests()
        return
    if "--ab-report" in sys.argv:
        print_ab_report()
        return

    stream_mode = "--stream" in sys.argv
    graph = build_graph()

    print("\nTriage agent (advanced build) ready.")
    print("Try something close to a sample ticket, then something unrelated,")
    print("to see all 3 paths. Type 'exit' to quit.\n")

    while True:
        request_text = input("Support request: ").strip()
        if request_text.lower() in {"exit", "quit"}:
            break
        if not request_text:
            continue

        # NEW: input guardrail, before the graph ever sees the request
        block_reason = input_guardrail(request_text)
        if block_reason:
            print(f"[guardrail] Request blocked — {block_reason}\n")
            continue

        # NEW: richer observability tags/metadata on the same invoke/stream call
        config = {"tags": ["triage-agent-advanced"], "metadata": {"request_length": len(request_text)}}

        run_id = None
        if stream_mode:
            stream_response(graph, initial_state(request_text), config)
            result = graph.invoke(initial_state(request_text), config)  # re-run for structured result below
        elif collect_runs is not None:
            with collect_runs() as cb:
                result = graph.invoke(initial_state(request_text), config)
            if cb.traced_runs:
                run_id = str(cb.traced_runs[-1].id)
        else:
            result = graph.invoke(initial_state(request_text), config)

        # NEW: output guardrail, before showing the result
        block_reason = output_guardrail(result["solution"] or "")
        if block_reason:
            print(f"[guardrail] Response blocked — {block_reason}\n")
            continue

        # NEW: human review gate for low-confidence results
        result = human_review_gate(result)

        print(f"\n--- classification: {result['classification']} "
              f"(confidence: {result['confidence']}) | variant: {result['prompt_variant']} ---")
        print(result["solution"])
        print(f"[timings] {result['timings']}")
        if result.get("tool_trace"):
            print(f"[tools used] {result['tool_trace']}")

        if result["classification"] != "fallback":
            judge = llm_judge(request_text, result["root_cause"], result["solution"])
            print(f"[judge] {judge}")

        drift_monitor.record(result["best_score"], result["classification"])
        collect_human_feedback(request_text, result["solution"], result["prompt_variant"], run_id)

        print(token_tracker.session_summary())
        print()


if __name__ == "__main__":
    main()