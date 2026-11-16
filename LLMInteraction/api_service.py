"""
FastAPI wrapper around triage_agent_production's LangGraph agent.

WHY THIS FILE EXISTS: triage_agent_production.py (like the original
triage_agent_advanced.py) is a synchronous `while True: input()` CLI loop.
That is not a production service -- it can't handle concurrent requests,
and its human-review gate blocks on stdin, which has no meaning in an HTTP
request handler (there's no terminal to type into; the process would just
hang until a request timeout kills it). This wraps the same graph in an
API that:

  - handles concurrent requests without blocking the event loop, by
    running the (still-synchronous) LangGraph graph in a thread pool
    per request via Starlette's run_in_threadpool -- a deliberate,
    low-risk choice over rewriting every node as async right now; see
    "Next steps" in PRODUCTION_READINESS_REVIEW.md for the async-native
    migration path if throughput ever requires it
  - replaces the blocking input()-based review gate with a non-blocking
    pending-review workflow: a low-confidence result comes back with
    status="pending_review" and a request_id; a separate endpoint lets a
    human approve or reject it asynchronously
  - supports client-supplied idempotency keys so a retried HTTP call
    (e.g. after a client-side timeout) doesn't get double-processed,
    double-charged, or double-written to episodic memory
  - exposes /healthz, /readyz, and /metrics for a real deployment
    (load balancer health checks, Prometheus scraping)

Run locally:
    USE_MOCK_BACKEND=true uvicorn api_service:app --reload
Run against a real model:
    GOOGLE_API_KEY=... uvicorn api_service:app --host 0.0.0.0 --port 8080 --workers 1

NOTE on --workers: each worker process gets its OWN in-memory FAISS index,
episodic memory store, circuit breaker, and idempotency cache. That's
consistent within one worker but NOT shared across workers -- two requests
hitting different workers won't see each other's idempotency keys or
episodic memory writes at the same read-after-write speed as a single
process would. For multi-worker/multi-instance deployment, the case_memory
vector store and idempotency cache need to move to a shared backend (a
real vector DB + Redis) -- called out in PRODUCTION_READINESS_REVIEW.md,
not solved here so this stays honest about what it does and doesn't fix.
"""

import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import triage_agent_production as agent

app = FastAPI(title="Triage Agent API", version="1.0.0")

_graph = None  # built once at startup, reused across all requests


@app.on_event("startup")
def _startup() -> None:
    global _graph
    _graph = agent.build_graph()
    agent.logger.info("triage agent graph built, API ready to accept traffic")


# =============================================================================
# Pending-review store
# In-memory + process-local, matching the rest of this file's stated scope
# (see module docstring). For multi-instance deployment, back this with a
# shared store (Redis / a table) so an approval routed to a different
# instance than the one holding the pending result still works, and prefer
# LangGraph's own checkpointer + interrupt() mechanism over a bespoke dict
# once you're ready to invest in that -- it gives you resumability for
# free. This is the pragmatic version, not the final one.
# =============================================================================
class PendingReview:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._store: dict = {}

    def put(self, request_id: str, payload: dict) -> None:
        with self._lock:
            self._store[request_id] = payload

    def get(self, request_id: str) -> Optional[dict]:
        with self._lock:
            return self._store.get(request_id)

    def resolve(self, request_id: str, approved: bool) -> Optional[dict]:
        with self._lock:
            entry = self._store.get(request_id)
            if entry is None:
                return None
            entry["status"] = "approved" if approved else "rejected"
            if not approved:
                entry["solution"] = "PENDING HUMAN REVIEW — a low-confidence result was not approved for release."
            return entry


pending_reviews = PendingReview()


# =============================================================================
# Request/response models
# =============================================================================
class TriageRequest(BaseModel):
    request_text: str = Field(..., min_length=1, max_length=agent.MAX_REQUEST_CHARS * 2)


class TriageResponse(BaseModel):
    request_id: str
    status: str  # "completed" | "pending_review" | "blocked" | "error"
    classification: Optional[str] = None
    confidence: float = 0.0
    solution: Optional[str] = None
    degraded: bool = False
    block_reason: Optional[str] = None
    tool_trace: list = []


# =============================================================================
# POST /v1/triage
# =============================================================================
@app.post("/v1/triage", response_model=TriageResponse)
async def triage(
    body: TriageRequest,
    response: Response,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    if not body.request_text.strip():
        raise HTTPException(status_code=422, detail="request_text must not be empty or whitespace-only")

    if idempotency_key:
        cached = agent.idempotency_cache.get(idempotency_key)
        if cached is not None:
            response.headers["X-Idempotent-Replay"] = "true"
            response.headers["X-Request-ID"] = cached["request_id"]
            return cached

    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    result = await run_in_threadpool(agent.process_request, _graph, body.request_text, request_id)

    if result.get("blocked"):
        payload = TriageResponse(
            request_id=request_id,
            status="blocked",
            block_reason=result.get("block_reason"),
        ).model_dump()
    elif result.get("error"):
        payload = TriageResponse(
            request_id=request_id,
            status="error",
            solution=result.get("solution"),
        ).model_dump()
        # Config errors are an operator problem, not a client problem, but
        # the client still needs a 5xx so upstream retries/alerts behave
        # correctly instead of treating this like a normal 200.
        response.status_code = 503 if result.get("error") == "service_misconfigured" else 500
    else:
        needs_review = (
            result.get("classification") != "fallback"
            and not result.get("degraded")
            and not result.get("blocked")
            and result.get("confidence", 0.0) < agent.REVIEW_CONFIDENCE_THRESHOLD
        )
        status = "pending_review" if needs_review else "completed"
        payload = TriageResponse(
            request_id=request_id,
            status=status,
            classification=result.get("classification"),
            confidence=result.get("confidence", 0.0),
            solution=result.get("solution") if not needs_review else None,
            degraded=result.get("degraded", False),
            tool_trace=result.get("tool_trace") or [],
        ).model_dump()
        if needs_review:
            # Full result (including the actual solution) is retained
            # server-side for the reviewer; only withheld from the caller
            # until approved.
            pending_reviews.put(request_id, {**payload, "_full_solution": result.get("solution"), "status": "pending_review"})

    if idempotency_key:
        agent.idempotency_cache.set(idempotency_key, payload)

    return payload


# =============================================================================
# Human review endpoints (replace the CLI's blocking input() prompt)
# =============================================================================
@app.get("/v1/review/{request_id}")
async def get_pending_review(request_id: str):
    entry = pending_reviews.get(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="no pending review found for that request_id")
    return {k: v for k, v in entry.items() if not k.startswith("_")}


@app.post("/v1/review/{request_id}/approve")
async def approve_review(request_id: str):
    entry = pending_reviews.get(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="no pending review found for that request_id")
    entry["status"] = "approved"
    entry["solution"] = entry.pop("_full_solution", entry.get("solution"))
    return {k: v for k, v in entry.items() if not k.startswith("_")}


@app.post("/v1/review/{request_id}/reject")
async def reject_review(request_id: str):
    entry = pending_reviews.resolve(request_id, approved=False)
    if entry is None:
        raise HTTPException(status_code=404, detail="no pending review found for that request_id")
    entry.pop("_full_solution", None)
    return {k: v for k, v in entry.items() if not k.startswith("_")}


# =============================================================================
# Operability endpoints
# =============================================================================
@app.get("/healthz")
async def healthz():
    """Liveness: is the process up and able to respond at all."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(response: Response):
    """Readiness: is the graph built and the LLM circuit breaker not open.
    A load balancer should stop routing traffic here (but not restart the
    pod) when this returns 503 -- an open circuit breaker is often
    self-healing within its cooldown window."""
    if _graph is None:
        response.status_code = 503
        return {"status": "not_ready", "reason": "graph not initialized"}
    if not agent.llm_circuit_breaker.allow():
        response.status_code = 503
        return {"status": "degraded", "reason": "llm circuit breaker open"}
    return {"status": "ready"}


@app.get("/metrics")
async def metrics_endpoint():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(agent.metrics.prometheus_text())


@app.get("/v1/session-summary")
async def session_summary():
    return {"summary": agent.token_tracker.session_summary(), **agent.metrics.snapshot()}