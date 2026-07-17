"""
Single-file demo: LangChain + LangGraph "support ticket triage" agent
(same architecture as triage_agent.py) with a full observability/eval
layer bolted on:

    1. LangSmith tracing & observability
    2. Cost & token tracking
    3. Latency SLA monitoring
    4. LLM-as-judge evaluation
    5. Golden datasets & regression testing
    6. Human feedback loop
    7. Prompt A/B testing
    8. Drift detection

None of these are production-grade implementations — they're the
simplest version of each pattern that still actually works, so you can
see how the pieces wire together and swap in real infra (a proper eval
dataset, a feedback DB, a stats library for drift, etc.) later.

Requirements (add to requirements.txt):
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
    langsmith

Setup:
    1. pip install -r requirements.txt
    2. .env file:
           GOOGLE_API_KEY=your_key_here
           # Optional — enables LangSmith tracing (#1) and feedback logging (#6):
           LANGCHAIN_API_KEY=your_langsmith_key_here
           LANGCHAIN_PROJECT=triage-agent-demo
    3. Normal run:            python triage_agent_full.py
       Regression test run:   python triage_agent_full.py --test
       A/B report only:       python triage_agent_full.py --ab-report
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
LLM_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

SCORE_EXISTING = 0.65
SCORE_NEW = 0.40


def is_llm_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "resource_exhausted",
            "quota",
            "rate limit",
            "429",
            "exceeded your current quota",
            "not_found",
            "model",
            "no longer available",
            "unsupported",
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
# 1. LANGSMITH TRACING & OBSERVABILITY
# =============================================================================
# LangChain/LangGraph auto-instrument every LLM call, chain, and graph node
# once these env vars are set — no code changes needed inside the graph.
# https://docs.smith.langchain.com/
TRACING_ENABLED = bool(os.getenv("LANGCHAIN_API_KEY"))
if TRACING_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "triage-agent-demo")
    print(f"[langsmith] Tracing ON — project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("[langsmith] LANGCHAIN_API_KEY not set — tracing disabled. "
          "Add it to .env to see traces at smith.langchain.com.")

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
# 2. COST & TOKEN TRACKING
# =============================================================================
# Approximate Gemini 1.5 Flash pricing (USD per 1M tokens) — update to match
# your actual plan/model; this is illustrative, not billing-accurate.
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
        self.calls.append(
            {"node": node_name, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost}
        )
        print(f"[tokens] {node_name}: in={input_tokens} out={output_tokens} cost=${cost:.5f}")
        return cost

    def session_summary(self) -> str:
        return (
            f"[tokens] SESSION TOTAL — input={self.total_input} output={self.total_output} "
            f"cost=${self.total_cost:.4f}"
        )


token_tracker = TokenTracker()


# =============================================================================
# 3. LATENCY SLA MONITORING
# =============================================================================
# Rough SLAs per node, loosely inspired by the PATH A/B timing breakdown
# in the original design doc (~35-38s end to end there; much lower here
# since this demo does far less work per node).
SLA_SECONDS = {
    "classifier": 3.0,
    "ticket_retriever": 8.0,
    "root_cause_reasoner": 8.0,
    "synthesizer": 5.0,
}


def with_timing(node_name: str):
    """Decorator: times a node, logs an SLA breach warning, and records the
    elapsed time into state['timings'] so it's visible in the final result."""

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
# PART: DATA INGESTION (same sample data as triage_agent.py)
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
    print(f"Indexed {len(docs)} sample tickets into FAISS.")
    return vectorstore


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


tickets_store = build_tickets_vectorstore()
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


# =============================================================================
# NODE 1 — CLASSIFIER
# =============================================================================
@with_timing("classifier")
def classifier_node(state: AgentState) -> dict:
    results = tickets_store.similarity_search_with_relevance_scores(state["request_text"], k=1)
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


# =============================================================================
# NODE 2A — TICKET RETRIEVER (PATH A)
# =============================================================================
TICKET_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """A new support request has come in. A similar past ticket was found.
Summarize in 2-3 sentences what the likely issue and fix are, based on the
past ticket's resolution hint.

New request: {request_text}

Matched past ticket: {ticket_name}
Resolution hint: {resolution_hint}

Summary:"""
)


@with_timing("ticket_retriever")
def ticket_retriever_node(state: AgentState) -> dict:
    ticket = state["matched_ticket"]
    chain = TICKET_EXTRACT_PROMPT | llm
    fallback_text = (
        "Unable to reach the AI model right now due to quota limits; "
        "please retry shortly."
    )
    root_cause_text, _ = invoke_llm_or_fallback(
        chain,
        {
            "request_text": state["request_text"],
            "ticket_name": ticket["name"],
            "resolution_hint": ticket["resolution_hint"],
        },
        fallback_text,
        "ticket_retriever",
    )
    return {"root_cause": root_cause_text}


# =============================================================================
# NODE 2B — ROOT CAUSE REASONER (PATH B)
# =============================================================================
ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_template(
    """You are a root cause analyst. No closely-matching past ticket exists for
this request, so hypothesize the likely root cause in 1-2 sentences. Do not
suggest a fix yet, only diagnose.

Request: {request_text}"""
)


@with_timing("root_cause_reasoner")
def root_cause_reasoner_node(state: AgentState) -> dict:
    chain = ROOT_CAUSE_PROMPT | llm
    fallback_text = (
        "Unable to reach the AI model right now due to quota limits; "
        "please retry shortly."
    )
    root_cause_text, _ = invoke_llm_or_fallback(
        chain,
        {"request_text": state["request_text"]},
        fallback_text,
        "root_cause_reasoner",
    )
    return {"root_cause": root_cause_text}


# =============================================================================
# 7. PROMPT A/B TESTING — two synthesizer prompt variants
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

# Running per-variant feedback tally, populated by the human feedback loop (#6)
ab_results = {"A": {"count": 0, "positive": 0}, "B": {"count": 0, "positive": 0}}


def choose_prompt_variant() -> str:
    """50/50 split. Swap for a hash-based split (e.g. by user id) for a
    consistent per-user experience instead of per-request randomness."""
    return random.choice(["A", "B"])


def print_ab_report():
    print("\n[A/B report]")
    for variant, stats in ab_results.items():
        count = stats["count"]
        rate = (stats["positive"] / count * 100) if count else 0.0
        print(f"  Variant {variant}: {count} rated responses, {rate:.0f}% positive")
    print()


# =============================================================================
# NODE 3 — SYNTHESIZER (shared, A/B tested)
# =============================================================================
@with_timing("synthesizer")
def synthesizer_node(state: AgentState) -> dict:
    variant = choose_prompt_variant()
    prompt = SYNTH_VARIANTS[variant]
    chain = prompt | llm
    fallback_text = (
        "The AI model is temporarily unavailable due to quota limits; "
        "please retry later."
    )
    solution_text, _ = invoke_llm_or_fallback(
        chain,
        {
            "request_text": state["request_text"],
            "classification": state["classification"],
            "root_cause": state["root_cause"],
        },
        fallback_text,
        "synthesizer",
    )
    confidence = round(state["best_score"], 2) if state["classification"] != "fallback" else 0.0
    print(f"[ab] using prompt variant: {variant}")
    return {"solution": solution_text, "confidence": confidence, "prompt_variant": variant}


def fallback_node(state: AgentState) -> dict:
    print("[fallback] No relevant match found — skipping all LLM calls.")
    return {
        "solution": "Insufficient match found in the knowledge base — human review recommended.",
        "confidence": 0.0,
        "root_cause": None,
        "prompt_variant": "none",
    }


# =============================================================================
# BUILD THE GRAPH
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
        {
            "ticket_retriever": "ticket_retriever",
            "root_cause_reasoner": "root_cause_reasoner",
            "fallback": "fallback",
        },
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
    }


# =============================================================================
# 4. LLM-AS-JUDGE EVALUATION
# =============================================================================
JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading an AI-generated support triage response. Score it 1-5 on:
- relevance: does it address the actual request?
- groundedness: is it consistent with the analysis provided, not made up?

Respond ONLY with JSON, no markdown fences:
{{"relevance": <1-5>, "groundedness": <1-5>, "reasoning": "<one sentence>"}}

Request: {request_text}
Analysis: {root_cause}
Generated solution: {solution}"""
)


def llm_judge(request_text: str, root_cause: Optional[str], solution: Optional[str]) -> Optional[dict]:
    if not solution:
        return None
    chain = JUDGE_PROMPT | llm
    fallback_text = '{"relevance": 1, "groundedness": 1, "reasoning": "llm_unavailable"}'
    raw, _ = invoke_llm_or_fallback(
        chain,
        {"request_text": request_text, "root_cause": root_cause or "(none)", "solution": solution},
        fallback_text,
        "llm_judge",
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[judge] Could not parse judge output: {raw[:200]}")
        return {"relevance": None, "groundedness": None, "reasoning": "parse_failed"}


# =============================================================================
# 5. GOLDEN DATASETS & REGRESSION TESTING
# =============================================================================
# A tiny hand-labeled set. In a real setup this would live in LangSmith as a
# Dataset (langsmith_client.create_dataset / create_examples) so you can run
# `langsmith_client.evaluate(...)` and get a hosted report instead of print().
GOLDEN_DATASET = [
    {
        "request_text": "password on signature template disappears after saving as draft",
        "expected_classification": "existing",
    },
    {
        "request_text": "PDF export hangs and eventually times out on big files",
        "expected_classification": "existing",
    },
    {
        "request_text": "the search results seem to load slower than usual today",
        "expected_classification": "new",
    },
    {
        "request_text": "what's the weather forecast for tomorrow",
        "expected_classification": "fallback",
    },
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
        print(
            f"[{status}] expected={case['expected_classification']!r} "
            f"actual={result['classification']!r} score={result['best_score']:.2f} "
            f"judge={judge}"
        )

    rate = passed / len(GOLDEN_DATASET) * 100
    print(f"\nRegression pass rate: {passed}/{len(GOLDEN_DATASET)} ({rate:.0f}%)")
    print(token_tracker.session_summary())


# =============================================================================
# 8. DRIFT DETECTION
# =============================================================================
class DriftMonitor:
    """Heuristic drift check over a rolling window of recent requests:
    (a) average FAISS match score dropping vs. an early baseline, and
    (b) fallback rate creeping up — both suggest the knowledge base no
    longer covers what people are actually asking about."""

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
                print(f"[drift] Baseline established: avg score {self.baseline_mean:.2f} "
                      f"over {len(self.scores)} requests")
            return

        current_mean = statistics.mean(self.scores)
        delta = self.baseline_mean - current_mean
        if delta > self.score_drift_threshold:
            print(
                f"[DRIFT WARNING] Avg match score dropped {delta:.2f} vs baseline "
                f"({self.baseline_mean:.2f} -> {current_mean:.2f}). Knowledge base "
                f"may need refreshing with newer tickets."
            )

        fallback_rate = self.classifications.count("fallback") / len(self.classifications)
        if fallback_rate > self.fallback_rate_threshold:
            print(
                f"[DRIFT WARNING] {fallback_rate:.0%} of the last {len(self.classifications)} "
                f"requests fell back to no-match — possible topic drift."
            )


drift_monitor = DriftMonitor()


# =============================================================================
# 6. HUMAN FEEDBACK LOOP
# =============================================================================
FEEDBACK_LOG_PATH = "feedback_log.jsonl"


def collect_human_feedback(request_text: str, solution: str, prompt_variant: str, run_id: Optional[str]):
    raw = input("Was this helpful? (y/n/skip): ").strip().lower()
    if raw not in {"y", "n"}:
        return

    positive = raw == "y"

    # Update A/B tally (#7)
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

    # Push to LangSmith too, if tracing is on and we captured a run_id
    if langsmith_client and run_id:
        try:
            langsmith_client.create_feedback(run_id, key="user_helpful", score=int(positive))
            print("[langsmith] Feedback logged to run.")
        except Exception as e:  # pragma: no cover - network/SDK errors
            print(f"[langsmith] Failed to log feedback: {e}")


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

    graph = build_graph()

    print("\nTriage agent (full observability build) ready.")
    print("Try something close to a sample ticket, then something unrelated,")
    print("to see all 3 paths. Type 'exit' to quit, or run with --test / --ab-report.\n")

    while True:
        request_text = input("Support request: ").strip()
        if request_text.lower() in {"exit", "quit"}:
            break
        if not request_text:
            continue

        # Capture the LangSmith run_id for this invocation, if tracing is on,
        # so human feedback (#6) can be attached to the exact trace.
        run_id = None
        if collect_runs is not None:
            with collect_runs() as cb:
                result = graph.invoke(initial_state(request_text))
            if cb.traced_runs:
                run_id = str(cb.traced_runs[-1].id)
        else:
            result = graph.invoke(initial_state(request_text))

        print(f"\n--- classification: {result['classification']} "
              f"(confidence: {result['confidence']}) | variant: {result['prompt_variant']} ---")
        print(result["solution"])
        print(f"[timings] {result['timings']}")

        # LLM-as-judge (#4) — skip for fallback since there's no LLM output to grade
        if result["classification"] != "fallback":
            judge = llm_judge(request_text, result["root_cause"], result["solution"])
            print(f"[judge] {judge}")

        # Drift detection (#8)
        drift_monitor.record(result["best_score"], result["classification"])

        # Human feedback (#6), feeding A/B tally (#7)
        collect_human_feedback(request_text, result["solution"], result["prompt_variant"], run_id)

        print(token_tracker.session_summary())
        print()


if __name__ == "__main__":
    main()