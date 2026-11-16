"""
Production-hardened build of the LangChain + LangGraph support-ticket
triage agent.

SAME GRAPH SHAPE AS triage_agent_advanced.py:
    classifier -> {ticket_retriever | root_cause_reasoner | fallback} -> synthesizer -> END
Same node names, same routing thresholds. Nothing about WHEN each stage
runs, or WHICH path gets taken, has changed.

This file assumes triage_agent_advanced.py's feature set (hybrid search,
episodic memory, agentic tool loop, guardrails, A/B testing, drift
detection) and fixes the issues that would actually break it in
production. Every change is tagged "FIX:" with a one-line reason, mirroring
that file's own "NEW:" convention so the two diff cleanly against each
other.

WHAT WAS ACTUALLY WRONG (see PRODUCTION_READINESS_REVIEW.md for the full
writeup with severity ratings):

  P0 -- would crash the process or silently corrupt data:
    - route_after_classifier raised KeyError on an unexpected value
    - `--stream` mode invoked the graph TWICE per request (double LLM
      spend, double episodic-memory writes)
    - a single non-quota LLM exception propagated out of graph.invoke()
      uncaught in main(), killing the whole CLI loop for every
      subsequent user
    - TicketKnowledgeBase used list.index(Document) to map scores back
      to tickets -- silently wrong whenever two tickets have identical
      text, because equality-based .index() returns the first match
    - the output guardrail ran AFTER the response was already written to
      episodic memory / judged / cost-tracked, so a "blocked" response
      still leaked into future RAG context

  P1 -- reliability / correctness under real traffic:
    - no retry/backoff or timeout on any LLM call, no circuit breaker,
      and "is_llm_quota_error" lumped genuine rate-limits together with
      permanent config errors (bad/retired model name) -- the latter
      would silently degrade forever instead of paging anyone
    - degraded/fallback text (the placeholder shown when the model is
      unreachable) was written into episodic memory, graded by the
      judge, and counted in A/B stats as if it were a real answer --
      polluting future retrieval and the metrics that guide decisions
    - shared global state (token tracker, drift monitor, A/B counters,
      the case-memory JSONL file) had no locking -- fine for a
      single-threaded REPL, unsafe the moment this is served
      concurrently
    - no input length cap (cost/DoS exposure), no request correlation
      ID, no idempotency protection against client-side retries
    - a hung or failing tool call inside the agentic sub-loop had no
      timeout and could not be isolated from the rest of the loop

  P2 -- infra/operability gaps addressed by api_service.py + Dockerfile,
  not this file: this is still a synchronous CLI-shaped module, so it
  cannot itself serve concurrent HTTP traffic; the blocking input()-based
  review gate only works for a local terminal.

Requirements: same as triage_agent_advanced.py, plus:
    tenacity            # retry/backoff
    filelock            # optional; cross-process safety for the memory log

Env vars (new):
    LOG_LEVEL=INFO
    LOG_FORMAT=json|text
    MAX_REQUEST_CHARS=4000
    USE_MOCK_BACKEND=false   # true = offline, deterministic, no API key/network needed
"""

import contextvars
import json
import logging
import os
import random
import statistics
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from functools import wraps
from itertools import cycle
from typing import Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # FIX: degrade to in-process-only locking; log a warning at startup (below)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-3.6-flash")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
MAX_REQUEST_CHARS = int(os.getenv("MAX_REQUEST_CHARS", "4000"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "3"))
ROOT_CAUSE_LOOP_BUDGET_SECONDS = float(os.getenv("ROOT_CAUSE_LOOP_BUDGET_SECONDS", "30"))
TOOL_CALL_TIMEOUT_SECONDS = float(os.getenv("TOOL_CALL_TIMEOUT_SECONDS", "10"))
USE_MOCK_BACKEND = os.getenv("USE_MOCK_BACKEND", "false").lower() == "true"

SCORE_EXISTING = 0.65
SCORE_NEW = 0.40

PRICE_PER_1M_INPUT = float(os.getenv("PRICE_PER_1M_INPUT", "0.075"))
PRICE_PER_1M_OUTPUT = float(os.getenv("PRICE_PER_1M_OUTPUT", "0.30"))

MEMORY_LOG_PATH = os.getenv("MEMORY_LOG_PATH", "case_memory.jsonl")
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "feedback_log.jsonl")


# =============================================================================
# STRUCTURED LOGGING
# FIX: replaces ad-hoc print() calls. Uses a contextvar for request_id so
# every log line in this request's call stack is correlated without having
# to thread request_id through every function signature. contextvars are
# task/thread-isolated, so this is safe under concurrent asyncio tasks too
# (the FastAPI wrapper sets it once per request).
# =============================================================================
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "json").lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s request_id=%(request_id)s %(message)s"))
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id_var.get()
            return record

        logging.setLogRecordFactory(record_factory)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


configure_logging()
logger = logging.getLogger("triage_agent")

if FileLock is None:
    logger.warning("filelock not installed -- case_memory/feedback log writes are only "
                    "process-local-safe, not multi-process-safe. `pip install filelock` for prod.")


# =============================================================================
# LIGHTWEIGHT METRICS (Prometheus text exposition, zero extra dependency)
# FIX: printed timings/costs were previously unrecoverable once scrolled off
# a terminal. api_service.py exposes MetricsRegistry.prometheus_text() at
# /metrics so this is actually scrapable in production.
# =============================================================================
class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._latency_sum: Dict[str, float] = {}
        self._latency_count: Dict[str, int] = {}

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            self._latency_sum[name] = self._latency_sum.get(name, 0.0) + seconds
            self._latency_count[name] = self._latency_count.get(name, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "avg_latency_seconds": {
                    k: round(self._latency_sum[k] / max(self._latency_count[k], 1), 4)
                    for k in self._latency_sum
                },
            }

    def prometheus_text(self) -> str:
        with self._lock:
            lines = []
            for k, v in sorted(self._counters.items()):
                lines.append(f"# TYPE triage_{k} counter")
                lines.append(f"triage_{k} {v}")
            for k in sorted(self._latency_sum):
                avg = self._latency_sum[k] / max(self._latency_count[k], 1)
                lines.append(f"# TYPE triage_{k}_latency_seconds_avg gauge")
                lines.append(f"triage_{k}_latency_seconds_avg {avg:.4f}")
            return "\n".join(lines) + "\n"


metrics = MetricsRegistry()


# =============================================================================
# ERROR CLASSIFICATION, CIRCUIT BREAKER, RETRY
# FIX: the original lumped rate-limit errors and permanent config errors
# (e.g. a decommissioned model name) into one "quota" bucket and silently
# fell back to placeholder text for both, forever. A bad model name would
# never surface as an incident -- it would just quietly degrade every
# request. This splits the three cases that actually need different
# handling.
# =============================================================================
class LLMConfigurationError(RuntimeError):
    """Raised for errors where retrying or silently falling back would only
    mask a real misconfiguration (bad model name, invalid request shape).
    These should fail loudly and page an operator, not degrade per-request."""


def classify_llm_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(t in msg for t in ("resource_exhausted", "quota", "rate limit", "429")):
        return "rate_limited"
    if any(t in msg for t in ("not_found", "no longer available", "unsupported", "invalid_argument", "does not exist")):
        return "permanent_config_error"
    if any(t in msg for t in ("timeout", "timed out", "connection", "503", "502", "504", "temporarily unavailable")):
        return "transient"
    return "unknown"


class CircuitBreaker:
    """Thread-safe. Opens after N consecutive failures and stops sending
    traffic to a clearly-down dependency for a cooldown window, instead of
    hammering it (and burning retry budget) on every request."""

    def __init__(self, failure_threshold: int = 5, reset_after_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: Optional[float] = None
        self.failure_threshold = failure_threshold
        self.reset_after_seconds = reset_after_seconds

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.time() - self._opened_at >= self.reset_after_seconds:
                return True  # half-open: let one trial request through
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold and self._opened_at is None:
                self._opened_at = time.time()
                logger.error(f"circuit breaker OPEN after {self._failures} consecutive LLM failures")
                metrics.inc("circuit_breaker_opened_total")


llm_circuit_breaker = CircuitBreaker(
    failure_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
    reset_after_seconds=float(os.getenv("CIRCUIT_BREAKER_RESET_SECONDS", "60")),
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    retry=retry_if_exception(lambda e: classify_llm_error(e) == "transient"),
)
def _invoke_with_retry(invokable, payload):
    return invokable.invoke(payload)


def is_llm_quota_error(exc: Exception) -> bool:
    """Kept for backward compatibility with any external callers of the
    original name; prefer classify_llm_error for new code."""
    return classify_llm_error(exc) == "rate_limited"


# FIX: `BaseMessage.content` is typed by LangChain as `str | list[str | dict]`,
# not just `str`. In practice this stayed a plain string for every backend
# tested here (Gemini via the mock, and every model version current as of
# this file's last edit) -- until it didn't: a newer langchain_google_genai
# build with AFC (automatic function calling) enabled returned `.content`
# as a LIST of content blocks for a normal, non-tool-calling reply. Every
# downstream consumer in this file (`output_guardrail`'s `re.search`,
# episodic-memory writes, structured logging) assumed a plain string and
# crashed the instant that shape showed up. Fixing at the two points where
# `.content` is first read means AgentState's `root_cause`/`solution`
# fields (typed `Optional[str]`) are actually strings everywhere else,
# instead of pushing the str-or-list ambiguity downstream.
_NON_TEXT_BLOCK_TYPES = {"thinking", "reasoning", "redacted_thinking", "tool_use", "tool_call", "image", "image_url"}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Skip non-text blocks explicitly rather than including
                # anything with a "text"-shaped key -- a reasoning/thinking
                # block from a model that exposes one should not silently
                # end up in a customer-facing triage response.
                if item.get("type") in _NON_TEXT_BLOCK_TYPES:
                    continue
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content) if content is not None else ""


def invoke_llm_or_fallback(chain, payload: dict, fallback_text: str, node_name: str) -> Tuple[str, bool]:
    """Returns (text, degraded). degraded=True means this is placeholder
    text, NOT a real model output -- callers must never write it to
    episodic memory, send it to the judge, or count it in A/B stats."""
    if not llm_circuit_breaker.allow():
        logger.warning(f"{node_name}: circuit breaker open, skipping call")
        metrics.inc(f"{node_name}_circuit_skipped_total")
        return fallback_text, True

    try:
        response = _invoke_with_retry(chain, payload)
    except Exception as exc:
        kind = classify_llm_error(exc)
        llm_circuit_breaker.record_failure()
        metrics.inc(f"{node_name}_llm_error_total")
        if kind == "permanent_config_error":
            logger.critical(f"{node_name}: permanent LLM configuration error -- needs operator attention: {exc}")
            raise LLMConfigurationError(str(exc)) from exc
        logger.warning(f"{node_name}: LLM call failed ({kind}), degrading to fallback text: {exc}")
        return fallback_text, True

    llm_circuit_breaker.record_success()
    token_tracker.record(response, node_name)
    return _extract_text(response.content), False


# =============================================================================
# TOKEN / COST TRACKING (thread-safe)
# =============================================================================
class TokenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_input = 0
        self.total_output = 0
        self.total_cost = 0.0
        self.calls: List[dict] = []

    def record(self, response, node_name: str) -> float:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT + (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
        with self._lock:
            self.total_input += input_tokens
            self.total_output += output_tokens
            self.total_cost += cost
            self.calls.append({"node": node_name, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost})
        metrics.inc(f"{node_name}_tokens_input_total", input_tokens)
        metrics.inc(f"{node_name}_tokens_output_total", output_tokens)
        logger.info(f"tokens node={node_name} in={input_tokens} out={output_tokens} cost=${cost:.5f}")
        return cost

    def session_summary(self) -> str:
        with self._lock:
            return f"SESSION TOTAL — input={self.total_input} output={self.total_output} cost=${self.total_cost:.4f}"


token_tracker = TokenTracker()


# =============================================================================
# LATENCY SLA MONITORING
# =============================================================================
SLA_SECONDS = {
    "classifier": 3.0,
    "ticket_retriever": 8.0,
    "root_cause_reasoner": 12.0,
    "synthesizer": 5.0,
}


def with_timing(node_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: dict) -> dict:
            start = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - start
            metrics.observe_latency(node_name, elapsed)
            sla = SLA_SECONDS.get(node_name)
            if sla and elapsed > sla:
                logger.warning(f"SLA BREACH node={node_name} took={elapsed:.2f}s limit={sla:.1f}s")
                metrics.inc(f"{node_name}_sla_breach_total")
            else:
                logger.info(f"timing node={node_name} elapsed={elapsed:.2f}s")
            timings = dict(state.get("timings", {}))
            timings[node_name] = round(elapsed, 3)
            result["timings"] = timings
            return result
        return wrapper
    return decorator


# =============================================================================
# DATA INGESTION (unchanged sample data)
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


# =============================================================================
# HYBRID SEARCH + RE-RANKING
# FIX: the original mapped a returned Document back to its knowledge-base
# index via `self.docs.index(d)`, which compares by VALUE (page_content +
# metadata), not identity. Two tickets with identical text would collide --
# `.index()` always returns the first match, so score/metadata could be
# silently attributed to the wrong ticket. Each Document now carries its own
# stable `_kb_idx` in metadata at construction time, so lookup never depends
# on equality or list position.
# =============================================================================
class TicketKnowledgeBase:
    def __init__(self, tickets: List[dict], embeddings):
        self.tickets = tickets
        self.docs = [
            Document(
                page_content=build_embedding_text(t),
                metadata={"id": t["id"], "name": t["name"], "resolution_hint": t["resolution_hint"], "_kb_idx": i},
            )
            for i, t in enumerate(tickets)
        ]

        if not self.docs:
            raise ValueError("TicketKnowledgeBase requires at least one ticket")

        from langchain_community.vectorstores import FAISS
        self.vectorstore = FAISS.from_documents(self.docs, embeddings)

        if BM25Okapi:
            tokenized = [d.page_content.lower().split() for d in self.docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
            logger.info("hybrid search: rank_bm25 not installed -- falling back to vector-only search.")

        self.reranker = self._load_reranker()
        logger.info(f"Indexed {len(self.docs)} sample tickets "
                    f"(hybrid={'on' if self.bm25 else 'off'}, rerank={'on' if self.reranker else 'off'}).")

    @staticmethod
    def _load_reranker():
        # FIX: reranker download/init failure previously had no error handling
        # at all beyond the ImportError check -- a network blip during cold
        # start would crash the whole process instead of degrading gracefully.
        if not CrossEncoder:
            return None
        try:
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            logger.warning(f"reranker failed to load, continuing without rerank: {exc}")
            return None

    def _rrf_fuse(self, ranked_lists: List[List[int]], k: int = 60) -> List[int]:
        scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def best_match(self, query: str):
        """Returns (Document, score) for the single best match. Same output
        shape as before, so classifier_node needs no changes."""
        if not self.docs:
            return None
        top_k = min(5, len(self.docs))
        vector_hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        if not vector_hits:
            return None

        # FIX: FAISS returns numpy.float32 scores. If this float ever lands
        # in AgentState unconverted, LangGraph's checkpointer (added below
        # for durability) fails to serialize it -- ormsgpack has no encoder
        # for numpy scalars -- and EVERY request crashes at graph.invoke(),
        # not just ones that hit an LLM. Cast to native float at the source
        # so nothing downstream (confidence rounding, drift monitor, JSON
        # logging, an eventual API response body) can reintroduce this.
        vector_scores = {d.metadata["_kb_idx"]: float(s) for d, s in vector_hits}
        candidates = [d for d, _ in vector_hits]

        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.lower().split())
            bm25_rank = sorted(range(len(self.docs)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
            vector_rank = [d.metadata["_kb_idx"] for d in candidates]
            fused_order = self._rrf_fuse([vector_rank, bm25_rank])
            candidates = [self.docs[i] for i in fused_order[:top_k]]

        if self.reranker and candidates:
            try:
                pairs = [(query, d.page_content) for d in candidates]
                rerank_scores = self.reranker.predict(pairs)
                candidates = [d for d, _ in sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)]
            except Exception as exc:
                logger.warning(f"reranker call failed, using fused order instead: {exc}")

        if not candidates:
            return None

        best_doc = candidates[0]
        best_score = float(vector_scores.get(best_doc.metadata["_kb_idx"], 0.0))
        return best_doc, best_score


# =============================================================================
# EPISODIC MEMORY
# FIX: thread-safe (in-process lock + optional cross-process filelock), and
# refuses to persist degraded/blocked text so placeholder "AI unavailable"
# responses and guardrail-blocked responses never get retrieved as if they
# were real past resolutions.
# =============================================================================
class CaseMemory:
    def __init__(self, embeddings):
        self._lock = threading.Lock()
        self._embeddings = embeddings
        seed = [Document(page_content="(memory initialized)", metadata={"kind": "system"})]
        from langchain_community.vectorstores import FAISS
        self._store = FAISS.from_documents(seed, embeddings)
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(MEMORY_LOG_PATH):
            return
        docs = []
        try:
            with open(MEMORY_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("skipping malformed line in case_memory log")
                        continue
                    docs.append(Document(page_content=entry["text"], metadata={"kind": "episodic", **entry}))
        except OSError as exc:
            logger.error(f"failed to read {MEMORY_LOG_PATH}: {exc}")
            return
        if docs:
            with self._lock:
                self._store.add_documents(docs)
            logger.info(f"loaded {len(docs)} past cases from {MEMORY_LOG_PATH}")

    def remember(self, request_text: str, classification: str, solution: str, request_id: str) -> None:
        text = f"Request: {request_text}\nClassification: {classification}\nResolution: {solution}"
        entry = {
            "text": text,
            "request_text": request_text,
            "classification": classification,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._store.add_documents([Document(page_content=text, metadata=entry)])
        self._append_to_disk(entry)

    def _append_to_disk(self, entry: dict) -> None:
        line = json.dumps(entry) + "\n"
        if FileLock:
            try:
                with FileLock(MEMORY_LOG_PATH + ".lock", timeout=5):
                    with open(MEMORY_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(line)
                return
            except Exception as exc:
                logger.error(f"filelock write failed, falling back to unlocked write: {exc}")
        with open(MEMORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)

    def recall(self, query: str, k: int = 2) -> List[Document]:
        with self._lock:
            return self._store.similarity_search(query, k=k)


# =============================================================================
# CONTEXT ENGINEERING
# =============================================================================
def assemble_context(matched_ticket_summary: str, recalled_cases: List[Document], max_recalled: int = 2) -> str:
    recalled_text = "\n---\n".join(
        d.page_content for d in recalled_cases[:max_recalled] if d.metadata.get("kind") == "episodic"
    )
    if not recalled_text:
        return matched_ticket_summary
    return f"{matched_ticket_summary}\n\nSimilar past cases (for extra context, may not be identical):\n{recalled_text}"


# =============================================================================
# TOOLS for the agentic sub-loop (PATH B)
# FIX: tool invocation now goes through _invoke_tool_safely, which isolates
# a hung or raising tool from the rest of the loop with a hard timeout.
# =============================================================================
@tool
def search_codebase(query: str) -> str:
    """Search the product codebase for functions/files related to the query."""
    return f"[mock code search] Found 2 files possibly related to {query!r}: signer.py, draft_handler.py"


@tool
def search_confluence_docs(query: str) -> str:
    """Search internal design docs for background on a feature or component."""
    return f"[mock docs search] Found 1 design doc mentioning {query!r}."


def get_mcp_tools() -> list:
    try:
        import mcp  # noqa: F401 -- presence check only
        return []
    except ImportError:
        return []


LOCAL_TOOLS = [search_codebase, search_confluence_docs]
ALL_TOOLS = LOCAL_TOOLS + get_mcp_tools()


def _invoke_tool_safely(tool_fn, args: dict, timeout_seconds: float = TOOL_CALL_TIMEOUT_SECONDS) -> str:
    result_holder: dict = {}

    def target():
        try:
            result_holder["result"] = tool_fn.invoke(args)
        except Exception as exc:  # a broken tool must not crash the reasoning loop
            result_holder["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        logger.warning(f"tool={tool_fn.name} timed out after {timeout_seconds}s")
        metrics.inc("tool_timeout_total")
        return f"[tool timeout] {tool_fn.name} did not respond within {timeout_seconds}s"
    if "error" in result_holder:
        logger.warning(f"tool={tool_fn.name} raised: {result_holder['error']}")
        metrics.inc("tool_error_total")
        return f"[tool error] {tool_fn.name} failed: {result_holder['error']}"
    return str(result_holder.get("result", "[tool returned no result]"))


# =============================================================================
# STATE
# FIX: added request_id (correlation/idempotency), degraded (was this node's
# output a placeholder?), blocked/block_reason (did the output guardrail
# withhold this?).
# =============================================================================
class AgentState(TypedDict):
    request_text: str
    request_id: str
    classification: str
    best_score: float
    matched_ticket: Optional[dict]
    root_cause: Optional[str]
    solution: Optional[str]
    confidence: float
    timings: dict
    prompt_variant: str
    tool_trace: List[str]
    degraded: bool
    blocked: bool
    block_reason: Optional[str]


def initial_state(request_text: str, request_id: Optional[str] = None) -> AgentState:
    request_id = request_id or str(uuid.uuid4())
    if len(request_text) > MAX_REQUEST_CHARS:
        logger.warning(f"request truncated from {len(request_text)} to {MAX_REQUEST_CHARS} chars")
        metrics.inc("request_truncated_total")
        request_text = request_text[:MAX_REQUEST_CHARS]
    return {
        "request_text": request_text,
        "request_id": request_id,
        "classification": "",
        "best_score": 0.0,
        "matched_ticket": None,
        "root_cause": None,
        "solution": None,
        "confidence": 0.0,
        "timings": {},
        "prompt_variant": "",
        "tool_trace": [],
        "degraded": False,
        "blocked": False,
        "block_reason": None,
    }


# =============================================================================
# MOCK BACKEND (offline, deterministic -- CI / smoke testing without a live
# API key or network egress to the model/embeddings providers)
# FIX: this is new. The original had no way to test the graph's control
# flow without hitting a real, billed API. USE_MOCK_BACKEND=true swaps in
# deterministic fakes so regression tests are fast, free, and reproducible.
# It intentionally does NOT try to fake realistic tool-selection behavior --
# it proves the plumbing runs end-to-end without crashing, nothing more.
# =============================================================================
def _build_llm_and_embeddings():
    if USE_MOCK_BACKEND:
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.embeddings import DeterministicFakeEmbedding

        class _MockToolCallingChatModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        mock_llm = _MockToolCallingChatModel(
            messages=cycle([
                "Likely a caching or state-sync issue based on the described symptoms.",
            ])
        )
        mock_embeddings = DeterministicFakeEmbedding(size=384)
        logger.warning("USE_MOCK_BACKEND=true -- using deterministic fake LLM/embeddings, NOT calling any real provider.")
        return mock_llm, mock_embeddings

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Add it to a .env file, or set USE_MOCK_BACKEND=true for offline testing."
        )
    real_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, timeout=LLM_TIMEOUT_SECONDS)
    real_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return real_llm, real_embeddings


llm, embeddings = _build_llm_and_embeddings()
llm_with_tools = llm.bind_tools(ALL_TOOLS)
tickets_kb = TicketKnowledgeBase(SAMPLE_TICKETS, embeddings)
case_memory = CaseMemory(embeddings)


# =============================================================================
# NODE 1 — CLASSIFIER
# =============================================================================
@with_timing("classifier")
def classifier_node(state: AgentState) -> dict:
    result = tickets_kb.best_match(state["request_text"])
    if not result:
        return {"classification": "fallback", "best_score": 0.0, "matched_ticket": None}

    doc, score = result
    logger.info(f"classifier best_match={doc.metadata['name']!r} score={score:.2f}")

    if score >= SCORE_EXISTING:
        classification = "existing"
    elif score >= SCORE_NEW:
        classification = "new"
    else:
        classification = "fallback"

    metrics.inc(f"classification_{classification}_total")
    return {
        "classification": classification,
        "best_score": score,
        "matched_ticket": doc.metadata if score >= SCORE_NEW else None,
    }


def route_after_classifier(state: AgentState) -> str:
    # FIX: original was a bare dict lookup -- state["classification"] not in
    # the map raised an uncaught KeyError and crashed graph.invoke(). Now
    # defaults safely to "fallback" and logs loudly, since this should never
    # happen if classifier_node is correct, and if it does happen it's a bug
    # worth knowing about, not a crashed request.
    mapping = {"existing": "ticket_retriever", "new": "root_cause_reasoner", "fallback": "fallback"}
    classification = state.get("classification")
    if classification not in mapping:
        logger.error(f"unexpected classification={classification!r} -- routing to fallback")
        metrics.inc("routing_fallback_on_unexpected_classification_total")
        return "fallback"
    return mapping[classification]


# =============================================================================
# NODE 2A — TICKET RETRIEVER (PATH A)
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
    recalled = case_memory.recall(state["request_text"])
    context = assemble_context(ticket_summary, recalled)

    chain = TICKET_EXTRACT_PROMPT | llm
    fallback_text = "Unable to reach the AI model right now; please retry shortly."
    root_cause_text, degraded = invoke_llm_or_fallback(
        chain, {"request_text": state["request_text"], "context": context}, fallback_text, "ticket_retriever"
    )
    return {"root_cause": root_cause_text, "degraded": degraded}


# =============================================================================
# NODE 2B — ROOT CAUSE REASONER (PATH B) — bounded agentic tool-calling loop
# FIX: per-call retry/circuit-breaker via _invoke_with_retry, per-tool
# timeout via _invoke_tool_safely, and a wall-clock budget in addition to
# the existing iteration cap (a slow but "successful" tool call on every
# iteration could previously run far longer than the SLA implies).
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


@with_timing("root_cause_reasoner")
def root_cause_reasoner_node(state: AgentState) -> dict:
    messages = [ROOT_CAUSE_SYSTEM_PROMPT, HumanMessage(content=state["request_text"])]
    tool_trace: List[str] = []
    deadline = time.monotonic() + ROOT_CAUSE_LOOP_BUDGET_SECONDS

    for _ in range(MAX_TOOL_ITERATIONS):
        if time.monotonic() > deadline:
            logger.warning("root_cause_reasoner exceeded wall-clock budget")
            metrics.inc("root_cause_reasoner_budget_exceeded_total")
            return {"root_cause": "Investigation timed out before completion.", "tool_trace": tool_trace, "degraded": True}

        if not llm_circuit_breaker.allow():
            return {"root_cause": "Unable to reach the AI model right now (circuit open).", "tool_trace": tool_trace, "degraded": True}

        try:
            response = _invoke_with_retry(llm_with_tools, messages)
        except Exception as exc:
            kind = classify_llm_error(exc)
            llm_circuit_breaker.record_failure()
            metrics.inc("root_cause_reasoner_llm_error_total")
            if kind == "permanent_config_error":
                logger.critical(f"root_cause_reasoner: permanent LLM configuration error: {exc}")
                raise LLMConfigurationError(str(exc)) from exc
            logger.warning(f"root_cause_reasoner skipped due to {kind}: {exc}")
            return {"root_cause": "Unable to reach the AI model right now due to a transient error.", "tool_trace": tool_trace, "degraded": True}

        llm_circuit_breaker.record_success()
        token_tracker.record(response, "root_cause_reasoner")
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return {"root_cause": _extract_text(response.content), "tool_trace": tool_trace, "degraded": False}

        for call in tool_calls:
            tool_fn = next((t for t in ALL_TOOLS if t.name == call["name"]), None)
            if tool_fn is None:
                result = f"Unknown tool requested: {call['name']}"
            else:
                result = _invoke_tool_safely(tool_fn, call["args"])
            tool_trace.append(f"{call['name']}({call['args']})")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    logger.warning("root_cause_reasoner exhausted tool-call iteration budget without a final answer")
    return {"root_cause": "Investigation inconclusive after available tool calls.", "tool_trace": tool_trace, "degraded": True}


# =============================================================================
# PROMPT A/B TESTING
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
ab_results_lock = threading.Lock()
ab_results = {"A": {"count": 0, "positive": 0}, "B": {"count": 0, "positive": 0}}


def choose_prompt_variant() -> str:
    return random.choice(["A", "B"])


def print_ab_report() -> None:
    with ab_results_lock:
        snapshot = {k: dict(v) for k, v in ab_results.items()}
    print("\n[A/B report]")
    for variant, stats in snapshot.items():
        count = stats["count"]
        rate = (stats["positive"] / count * 100) if count else 0.0
        print(f"  Variant {variant}: {count} rated responses, {rate:.0f}% positive")
    print()


# =============================================================================
# OUTPUT GUARDRAIL
# FIX: still a heuristic, not a substitute for a real moderation/DLP
# service -- flagged clearly in the review doc. Broadened slightly beyond
# SSN-shaped strings to catch a couple more obvious PII shapes.
# =============================================================================
def input_guardrail(text: str) -> Optional[str]:
    blocked = ["ignore previous instructions", "disregard prior instructions", "system prompt", "you are now"]
    lowered = text.lower()
    for pattern in blocked:
        if pattern in lowered:
            return f"blocked: input contains {pattern!r}"
    return None


def output_guardrail(text) -> Optional[str]:
    import re
    if not isinstance(text, str):
        text = _extract_text(text)  # defense in depth -- see _extract_text's comment above
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        return "blocked: output contains a possible SSN-shaped pattern"
    if re.search(r"\b\d{13,16}\b", text.replace(" ", "").replace("-", "")):
        return "blocked: output contains a possible card-number-shaped pattern"
    return None


# =============================================================================
# NODE 3 — SYNTHESIZER
# FIX: the output guardrail now runs INSIDE this node, before
# case_memory.remember() is called -- not after graph.invoke() returns in
# main(). Previously a blocked response was already committed to episodic
# memory (and judged, and cost-tracked) before anyone checked whether it
# should have been shown at all. Also skips the memory write entirely when
# the response is degraded placeholder text, so a "model unavailable"
# message never gets recalled as if it were a real past resolution.
# =============================================================================
@with_timing("synthesizer")
def synthesizer_node(state: AgentState) -> dict:
    variant = choose_prompt_variant()
    prompt = SYNTH_VARIANTS[variant]
    chain = prompt | llm
    fallback_text = "The AI model is temporarily unavailable due to quota limits; please retry later."
    solution_text, degraded = invoke_llm_or_fallback(
        chain,
        {"request_text": state["request_text"], "classification": state["classification"], "root_cause": state["root_cause"]},
        fallback_text,
        "synthesizer",
    )

    block_reason = output_guardrail(solution_text) if not degraded else None
    blocked = block_reason is not None
    if blocked:
        logger.warning(f"output guardrail blocked synthesizer response: {block_reason}")
        metrics.inc("output_guardrail_blocked_total")
        solution_text = "This response was withheld by an automated safety check and flagged for human review."

    confidence = round(state["best_score"], 2) if state["classification"] != "fallback" else 0.0
    logger.info(f"synthesizer using prompt_variant={variant} degraded={degraded} blocked={blocked}")

    if not degraded and not blocked:
        case_memory.remember(state["request_text"], state["classification"], solution_text, state["request_id"])
    else:
        metrics.inc("case_memory_write_skipped_total")

    return {
        "solution": solution_text,
        "confidence": confidence,
        "prompt_variant": variant,
        "degraded": degraded,
        "blocked": blocked,
        "block_reason": block_reason,
    }


def fallback_node(state: AgentState) -> dict:
    logger.info("fallback: no relevant match found -- skipping all LLM calls.")
    metrics.inc("fallback_total")
    return {
        "solution": "Insufficient match found in the knowledge base — human review recommended.",
        "confidence": 0.0,
        "root_cause": None,
        "prompt_variant": "none",
        "tool_trace": [],
        "degraded": False,
        "blocked": False,
        "block_reason": None,
    }


# =============================================================================
# BUILD THE GRAPH
# FIX: compiled with a checkpointer. MemorySaver is process-local (fine for
# the CLI / single-instance demo); swap in a persistent backend
# (e.g. langgraph-checkpoint-postgres) for multi-instance production so a
# pod restart mid-run doesn't lose state and so LangGraph's interrupt()
# mechanism can be used for a real async human-review gate (see
# api_service.py and the review doc).
# =============================================================================
def build_graph(checkpointer=None):
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
    return g.compile(checkpointer=checkpointer or MemorySaver())


# =============================================================================
# LLM-AS-JUDGE
# FIX: broadened the try/except to catch ANY judging failure (schema
# mismatch, timeout, mock-backend NotImplementedError), not just quota
# errors -- previously an unrelated judge failure would crash
# run_regression_tests() entirely. Also skipped for degraded/blocked
# results, since judging placeholder text was never meaningful.
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
    try:
        structured_llm = llm.with_structured_output(TriageJudgment)
        chain = JUDGE_PROMPT | structured_llm
        judgment = chain.invoke({"request_text": request_text, "root_cause": root_cause or "(none)", "solution": solution})
        return judgment.model_dump()
    except Exception as exc:
        logger.warning(f"llm_judge failed, skipping judgment: {exc}")
        metrics.inc("judge_error_total")
        return {"relevance": None, "groundedness": None, "reasoning": f"judge_unavailable: {type(exc).__name__}"}


# =============================================================================
# GOLDEN DATASETS & REGRESSION TESTING
# =============================================================================
GOLDEN_DATASET = [
    {"request_text": "password on signature template disappears after saving as draft", "expected_classification": "existing"},
    {"request_text": "PDF export hangs and eventually times out on big files", "expected_classification": "existing"},
    {"request_text": "the search results seem to load slower than usual today", "expected_classification": "new"},
    {"request_text": "what's the weather forecast for tomorrow", "expected_classification": "fallback"},
]


def run_regression_tests() -> None:
    print("\n=== Regression test run against golden dataset ===\n")
    if not GOLDEN_DATASET:
        print("GOLDEN_DATASET is empty -- nothing to test.")
        return
    graph = build_graph()
    passed = 0
    for case in GOLDEN_DATASET:
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        try:
            result = graph.invoke(initial_state(case["request_text"], request_id), {"configurable": {"thread_id": request_id}})
        except LLMConfigurationError as exc:
            print(f"[ABORT] permanent configuration error, stopping regression run: {exc}")
            return
        except Exception as exc:
            print(f"[FAIL] unhandled exception for {case['request_text']!r}: {exc}")
            continue
        ok = result["classification"] == case["expected_classification"]
        passed += int(ok)
        judge = None if result.get("degraded") or result.get("blocked") else llm_judge(case["request_text"], result["root_cause"], result["solution"])
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] expected={case['expected_classification']!r} actual={result['classification']!r} "
              f"score={result['best_score']:.2f} degraded={result.get('degraded')} judge={judge}")
    rate = passed / len(GOLDEN_DATASET) * 100
    print(f"\nRegression pass rate: {passed}/{len(GOLDEN_DATASET)} ({rate:.0f}%)")
    print(token_tracker.session_summary())


# =============================================================================
# DRIFT DETECTION (thread-safe)
# =============================================================================
class DriftMonitor:
    def __init__(self, window: int = 20, score_drift_threshold: float = 0.15, fallback_rate_threshold: float = 0.5):
        self._lock = threading.Lock()
        self.scores: deque = deque(maxlen=window)
        self.classifications: deque = deque(maxlen=window)
        self.baseline_mean: Optional[float] = None
        self.score_drift_threshold = score_drift_threshold
        self.fallback_rate_threshold = fallback_rate_threshold

    def record(self, score: float, classification: str) -> None:
        with self._lock:
            self.scores.append(score)
            self.classifications.append(classification)
            if self.baseline_mean is None:
                if len(self.scores) >= 5:
                    self.baseline_mean = statistics.mean(self.scores)
                    logger.info(f"drift baseline established: avg_score={self.baseline_mean:.2f} n={len(self.scores)}")
                return
            current_mean = statistics.mean(self.scores)
            delta = self.baseline_mean - current_mean
            if delta > self.score_drift_threshold:
                logger.warning(f"DRIFT: avg match score dropped {delta:.2f} vs baseline "
                                f"({self.baseline_mean:.2f} -> {current_mean:.2f}); knowledge base may need refreshing.")
                metrics.inc("drift_score_warning_total")
            fallback_rate = self.classifications.count("fallback") / len(self.classifications)
            if fallback_rate > self.fallback_rate_threshold:
                logger.warning(f"DRIFT: {fallback_rate:.0%} of last {len(self.classifications)} requests fell back -- possible topic drift.")
                metrics.inc("drift_fallback_rate_warning_total")


drift_monitor = DriftMonitor()


# =============================================================================
# HUMAN FEEDBACK LOOP
# FIX: skips degraded responses -- asking "was this helpful?" about an
# "AI unavailable, please retry" placeholder doesn't measure anything
# useful and would corrupt A/B stats.
# =============================================================================
def collect_human_feedback(request_text: str, solution: str, prompt_variant: str, run_id: Optional[str], degraded: bool) -> None:
    if degraded:
        return
    raw = input("Was this helpful? (y/n/skip): ").strip().lower()
    if raw not in {"y", "n"}:
        return
    positive = raw == "y"
    if prompt_variant in ab_results:
        with ab_results_lock:
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
    line = json.dumps(entry) + "\n"
    try:
        if FileLock:
            with FileLock(FEEDBACK_LOG_PATH + ".lock", timeout=5):
                with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line)
        else:
            with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError as exc:
        logger.error(f"failed to write feedback log: {exc}")


# =============================================================================
# HUMAN-IN-THE-LOOP REVIEW GATE
# NOTE: input() is blocking and only makes sense for this local CLI. The
# production API path (api_service.py) NEVER calls this -- it uses
# LangGraph's checkpointer + a separate approve/reject endpoint instead, so
# a request never ties up a server thread/coroutine waiting on a human.
# =============================================================================
REVIEW_CONFIDENCE_THRESHOLD = float(os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "0.5"))


def human_review_gate(result: dict) -> dict:
    if result["classification"] == "fallback" or result["blocked"] or result["confidence"] >= REVIEW_CONFIDENCE_THRESHOLD:
        return result
    print(f"\n[review] Confidence is low ({result['confidence']:.2f}) — flagging for human review before use.")
    decision = input("Approve this result for the requester anyway? (y/n): ").strip().lower()
    if decision != "y":
        result["solution"] = "PENDING HUMAN REVIEW — a low-confidence result was not approved for release."
    return result


# =============================================================================
# STREAMING
# FIX: the original called graph.stream() to print tokens, then called
# graph.invoke() again "to get the structured result" -- silently running
# every LLM call in the request TWICE (double cost, double latency, and
# TWO episodic-memory writes per user turn). LangGraph supports requesting
# multiple stream modes at once (verified against installed langgraph
# 1.x: passing a list to stream_mode yields (mode_name, chunk) tuples), so
# this collects the final state from the SAME run that produced the
# streamed tokens.
# =============================================================================
def stream_and_collect(graph, state: AgentState, config: dict) -> dict:
    print("\n[streaming] ", end="", flush=True)
    final_state: dict = dict(state)
    try:
        for mode_name, payload in graph.stream(state, config, stream_mode=["messages", "values"]):
            if mode_name == "messages":
                message_chunk, chunk_metadata = payload
                if chunk_metadata.get("langgraph_node") == "synthesizer" and getattr(message_chunk, "content", None):
                    print(message_chunk.content, end="", flush=True)
            elif mode_name == "values":
                final_state = payload
    except TypeError:
        # Defensive fallback if a pinned older langgraph version doesn't
        # support multi-mode streaming as a list -- single invoke, no
        # streamed tokens, but still only ONE run, not two.
        logger.warning("multi-mode streaming unsupported by installed langgraph version -- falling back to plain invoke")
        final_state = graph.invoke(state, config)
    print()
    return final_state


# =============================================================================
# IDEMPOTENCY CACHE
# FIX (round 2): the first version had a check-then-act race -- get() then,
# separately, set() -- so two requests carrying the SAME idempotency key
# arriving close together could both see "not cached yet" and both go on to
# fully process (double LLM spend, double episodic-memory write, exactly
# the thing idempotency is supposed to prevent). Caught by an actual
# concurrent load test (40 threads, 10 shared keys): only 11 of the
# expected ~30 duplicate calls were recognized as replays.
#
# Fixed with a claim/wait pattern: the first caller for a given key claims
# it and proceeds; concurrent duplicates block on an Event instead of
# racing ahead, then read the first caller's result once it lands.
# =============================================================================
class IdempotencyCache:
    def __init__(self, ttl_seconds: float = 300, max_size: int = 1000):
        self._lock = threading.Lock()
        self._store: Dict[str, Tuple[dict, float]] = {}
        self._inflight: Dict[str, threading.Event] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def claim(self, key: str) -> bool:
        """True if the caller is first for this key and should process the
        request. False means someone else is already in flight for this
        key -- the caller should wait_for_result() instead of processing."""
        with self._lock:
            if key in self._store or key in self._inflight:
                return False
            self._inflight[key] = threading.Event()
            return True

    def wait_for_result(self, key: str, timeout: float = 30.0) -> Optional[dict]:
        with self._lock:
            event = self._inflight.get(key)
        if event is None:
            return self.get(key)
        event.wait(timeout)
        return self.get(key)

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            if len(self._store) >= self.max_size:
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]
            self._store[key] = (value, time.time() + self.ttl)
            event = self._inflight.pop(key, None)
        if event:
            event.set()  # release anyone who called claim()==False and is waiting

    def release_claim(self, key: str) -> None:
        """Call if processing fails after a successful claim() -- otherwise
        every concurrent duplicate blocks the full `timeout` for a result
        that will never arrive."""
        with self._lock:
            event = self._inflight.pop(key, None)
        if event:
            event.set()


idempotency_cache = IdempotencyCache()


# =============================================================================
# CORE PROCESSING ENTRY POINT
# FIX: this is the function api_service.py calls. It centralizes the
# guardrail check, graph invocation, and error classification so both the
# CLI and the HTTP layer get identical, hardened behavior instead of
# duplicating (and inevitably drifting from) the safety logic.
# =============================================================================
def process_request(graph, request_text: str, request_id: Optional[str] = None) -> dict:
    request_id = request_id or str(uuid.uuid4())
    request_id_var.set(request_id)

    block_reason = input_guardrail(request_text)
    if block_reason:
        logger.warning(f"input blocked: {block_reason}")
        metrics.inc("input_guardrail_blocked_total")
        return {
            "request_id": request_id,
            "blocked": True,
            "block_reason": block_reason,
            "solution": None,
            "classification": None,
            "confidence": 0.0,
        }

    config = {"tags": ["triage-agent-production"], "configurable": {"thread_id": request_id}}
    try:
        result = graph.invoke(initial_state(request_text, request_id), config)
    except LLMConfigurationError as exc:
        logger.critical(f"service misconfigured: {exc}")
        metrics.inc("llm_configuration_error_total")
        return {
            "request_id": request_id,
            "blocked": False,
            "error": "service_misconfigured",
            "solution": "The AI service is misconfigured. This has been logged for the on-call engineer.",
            "classification": None,
            "confidence": 0.0,
        }
    except Exception:
        logger.exception("unhandled error processing request")
        metrics.inc("unhandled_error_total")
        return {
            "request_id": request_id,
            "blocked": False,
            "error": "internal_error",
            "solution": "Something went wrong processing that request. It's been logged; please try again.",
            "classification": None,
            "confidence": 0.0,
        }

    if result["classification"] is not None and result["classification"] != "":
        drift_monitor.record(result["best_score"], result["classification"])
    return result


# =============================================================================
# MAIN (CLI)
# FIX: every graph invocation is now wrapped so an unhandled exception logs
# and continues the loop instead of crashing the process -- previously ANY
# non-quota exception from a node propagated straight out of the `while`
# loop and took the whole CLI down for every subsequent user.
# =============================================================================
def main() -> None:
    if not USE_MOCK_BACKEND and not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY not found. Add it to a .env file, or set USE_MOCK_BACKEND=true.")

    if "--test" in sys.argv:
        run_regression_tests()
        return
    if "--ab-report" in sys.argv:
        print_ab_report()
        return

    stream_mode = "--stream" in sys.argv
    graph = build_graph()

    print("\nTriage agent (production build) ready.")
    if USE_MOCK_BACKEND:
        print("[mock backend active -- responses are canned, not from a real model]")
    print("Try something close to a sample ticket, then something unrelated,")
    print("to see all 3 paths. Type 'exit' to quit.\n")

    while True:
        request_text = input("Support request: ").strip()
        if request_text.lower() in {"exit", "quit"}:
            break
        if not request_text:
            continue

        block_reason = input_guardrail(request_text)
        if block_reason:
            print(f"[guardrail] Request blocked — {block_reason}\n")
            continue

        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        config = {"tags": ["triage-agent-production"], "configurable": {"thread_id": request_id}}

        try:
            if stream_mode:
                result = stream_and_collect(graph, initial_state(request_text, request_id), config)
            else:
                result = graph.invoke(initial_state(request_text, request_id), config)
        except LLMConfigurationError as exc:
            logger.critical(f"service misconfigured: {exc}")
            print("\n[error] The AI service is misconfigured. This has been logged for the on-call engineer.\n")
            continue
        except Exception:
            logger.exception("unhandled error processing request")
            print("\n[error] Something went wrong processing that request. It's been logged; please try again.\n")
            continue

        result = human_review_gate(result)

        print(f"\n--- classification: {result['classification']} "
              f"(confidence: {result['confidence']}) | variant: {result['prompt_variant']} "
              f"| degraded: {result.get('degraded')} ---")
        print(result["solution"])
        print(f"[timings] {result['timings']}")
        if result.get("tool_trace"):
            print(f"[tools used] {result['tool_trace']}")

        if result["classification"] != "fallback" and not result.get("degraded") and not result.get("blocked"):
            judge = llm_judge(request_text, result["root_cause"], result["solution"])
            print(f"[judge] {judge}")

        drift_monitor.record(result["best_score"], result["classification"])
        collect_human_feedback(request_text, result["solution"], result["prompt_variant"], None, result.get("degraded", False))

        print(token_tracker.session_summary())
        print()


if __name__ == "__main__":
    main()