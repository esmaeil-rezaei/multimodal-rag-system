from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import jwt
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.auth import (
    LoginRequest,
    RegisterRequest,
    decode_token,
    init_db,
    login_user,
    register_user,
)
from src.config.settings import get_config
from src.core.container import get_container, init_container
from src.indexing.vector_store import DenseVectorStore


logger = logging.getLogger(__name__)

_cfg = get_config()
_log_cfg = _cfg.log

UNKNOWN_LOG = Path(_log_cfg["unknown_query_dir"]) / "unknown_queries.jsonl"
LIKED_LOG = Path(_log_cfg["liked_response_dir"]) / "liked_responses.jsonl"
DISLIKED_LOG = Path(_log_cfg["disliked_response_dir"]) / "disliked_responses.jsonl"

for _p in (UNKNOWN_LOG, LIKED_LOG, DISLIKED_LOG):
    _p.parent.mkdir(parents=True, exist_ok=True)

MAX_HISTORY_TURNS = 20
_FAITHFULNESS_UNCERTAIN_THRESHOLD = 0.3

# Answer prefixes that signal the model could not answer from context.
# Compared case-insensitively after stripping leading whitespace.
_UNCERTAIN_PREFIXES: tuple[str, ...] = (
    "i'm sorry",
    "i was unable",
    "an internal error",
    "i cannot answer",
    "i could not find",
)

_session_histories: dict[str, list[dict[str, str]]] = defaultdict(list)


def _get_history(session_id: str) -> list[dict[str, str]]:
    return _session_histories[session_id]


def _append_history(session_id: str, question: str, answer: str) -> None:
    history = _session_histories[session_id]
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        _session_histories[session_id] = history[-max_messages:]


def _is_could_not_answer(answer: str, faithfulness: float | None) -> bool:
    """Return True when the model's answer signals insufficient context."""
    if not answer:
        return True
    lower = answer.strip().lower()
    if any(lower.startswith(prefix) for prefix in _UNCERTAIN_PREFIXES):
        return True
    if faithfulness is not None and faithfulness < _FAITHFULNESS_UNCERTAIN_THRESHOLD:
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    container = init_container()
    app.state.container = container
    yield
    container.shutdown()


app = FastAPI(
    title="RAG Knowledge Assistant",
    version="3.0.0",
    description="Multimodal retrieval-augmented generation with namespace federation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve namespace-organised extracted images
Path("artifacts").mkdir(exist_ok=True)
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    namespace: str | None = None  # guest-selected namespace
    token: str | None = None  # JWT for authenticated users


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list = []
    faithfulness_score: float | None = None
    has_conflict: bool = False
    conflict_resolution: str | None = None
    model_used: str = "unknown"
    could_not_answer: bool = False
    namespace_used: str | None = None


class FeedbackRequest(BaseModel):
    query_id: str
    question: str
    answer: str
    rating: str  # "like" | "dislike" | "unknown"
    comment: str | None = None
    session_id: str | None = None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    html_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post(
    "/auth/register",
    summary="Create a new user account",
    status_code=status.HTTP_201_CREATED,
)
async def auth_register(req: RegisterRequest):
    try:
        return register_user(req.username, req.password, req.namespace)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@app.post("/auth/login", summary="Authenticate and receive a JWT")
async def auth_login(req: LoginRequest):
    try:
        return login_user(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@app.get("/namespaces", summary="List available knowledge-base namespaces")
async def list_namespaces():
    """Return all namespaces inferred from knowledge_base/ subdirectories."""
    kb_root = Path(_cfg.knowledge_base["root_dir"])
    if not kb_root.exists():
        return {"namespaces": []}
    namespaces = sorted(
        [d.name for d in kb_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    return {"namespaces": namespaces}


@app.get("/chunk/{chunk_id}", summary="Retrieve chunk text for hover preview")
async def get_chunk(chunk_id: str):
    """Fetch a single chunk's payload from Qdrant by its ID."""
    dense_store: DenseVectorStore = get_container().dense_store
    try:
        points = dense_store._client.retrieve(
            collection_name=dense_store._collection,
            ids=[DenseVectorStore._chunk_id_to_int(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning("Chunk retrieval failed for '%s': %s", chunk_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach vector store.",
        ) from exc

    if not points:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk '{chunk_id}' not found.",
        )

    payload = points[0].payload or {}
    return {
        "chunk_id": chunk_id,
        "text": payload.get("text", ""),
        "source_file": payload.get("source_file", ""),
        "source_name": payload.get("source_name", ""),
        "modality": payload.get("modality", "text"),
        "namespace": payload.get("namespace", ""),
    }


@app.post("/query", response_model=QueryResponse, summary="Run a RAG query")
async def run_query(req: QueryRequest):
    query_id = str(uuid.uuid4())
    session_id = req.session_id or "default"
    history = _get_history(session_id)
    orchestrator = get_container().orchestrator

    # Namespace resolution: JWT claim > explicit body param > None (orchestrator default)
    namespace: str | None = None
    if req.token:
        try:
            claims = decode_token(req.token)
            namespace = claims.get("namespace")
        except jwt.PyJWTError as exc:
            logger.warning("Invalid JWT in query request: %s", exc)

    if namespace is None and req.namespace:
        namespace = req.namespace

    try:
        result = await orchestrator.run(
            raw_query=req.question,
            conversation_history=history,
            namespace=namespace,
        )
    except Exception as exc:
        logger.error("Orchestrator error for query '%s': %s", req.question[:80], exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query.",
        ) from exc

    could_not_answer = _is_could_not_answer(result.answer, result.faithfulness_score)

    log_record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": req.question,
        "answer": result.answer,
        "faithfulness_score": result.faithfulness_score,
        "model_used": result.model_used,
        "session_id": session_id,
        "namespace": namespace or "default",
    }
    if could_not_answer:
        _write_jsonl(UNKNOWN_LOG, log_record)
    else:
        _append_history(session_id, req.question, result.answer)

    return QueryResponse(
        query_id=query_id,
        answer=result.answer,
        citations=result.citations or [],
        faithfulness_score=result.faithfulness_score,
        has_conflict=result.has_conflict or False,
        conflict_resolution=result.conflict_resolution if result.has_conflict else None,
        model_used=result.model_used or "unknown",
        could_not_answer=could_not_answer,
        namespace_used=namespace or "default",
    )


@app.post("/query/stream", summary="Run a RAG query with server-sent event streaming")
async def run_query_stream(req: QueryRequest):
    """
    Streaming endpoint that yields SSE events as the query is processed.

    Event types:
      {"event": "status",  "message": "..."}   — phase updates (immediate feedback)
      {"event": "token",   "content": "..."}   — answer text tokens (streamed as generated)
      {"event": "done",    "citations": [...], "faithfulness_score": ..., ...}
      {"event": "error",   "message": "..."}   — on failure

    Connect with EventSource in the browser or `curl -N /query/stream`.
    The existing blocking /query endpoint is unchanged.
    """
    session_id = req.session_id or "default"
    history = _get_history(session_id)
    orchestrator = get_container().orchestrator

    namespace: str | None = None
    if req.token:
        try:
            claims = decode_token(req.token)
            namespace = claims.get("namespace")
        except jwt.PyJWTError as exc:
            logger.warning("Invalid JWT in stream request: %s", exc)

    if namespace is None and req.namespace:
        namespace = req.namespace

    async def event_stream():
        token_buf: list[str] = []
        try:
            async for chunk in orchestrator.run_stream(
                raw_query=req.question,
                conversation_history=history,
                namespace=namespace,
            ):
                # Collect tokens so we can update conversation history after streaming
                try:
                    raw = chunk.removeprefix("data: ").strip()
                    data = json.loads(raw)
                    if data.get("event") == "token":
                        token_buf.append(data.get("content", ""))
                except Exception:
                    pass
                yield chunk
        except Exception as exc:
            logger.error("Streaming error for query '%s': %s", req.question[:80], exc)
            yield f"data: {json.dumps({'event': 'error', 'message': 'An error occurred.'})}\n\n"

        # Update conversation history once the stream is complete
        full_answer = "".join(token_buf).strip()
        if full_answer and not _is_could_not_answer(full_answer, None):
            _append_history(session_id, req.question, full_answer)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/feedback", summary="Submit thumbs-up / thumbs-down feedback")
async def submit_feedback(req: FeedbackRequest):
    ts = datetime.now(timezone.utc).isoformat()
    record = {
        "query_id": req.query_id,
        "timestamp": ts,
        "question": req.question,
        "answer": req.answer,
        "rating": req.rating,
        "comment": req.comment or "",
        "session_id": req.session_id or "",
    }
    if req.rating == "like":
        _write_jsonl(LIKED_LOG, record)
    elif req.rating == "dislike":
        _write_jsonl(DISLIKED_LOG, record)
    elif req.rating == "unknown":
        _write_jsonl(UNKNOWN_LOG, {**record, "source": "user_flagged"})
    return {"status": "ok", "query_id": req.query_id}


@app.delete("/session/{session_id}", summary="Clear conversation history")
async def clear_session(session_id: str):
    _session_histories.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}


@app.get("/health", summary="Liveness probe")
async def health():
    return {"status": "ok"}


@app.get("/ready", summary="Readiness probe")
async def ready():
    import redis as _redis_mod

    from src.config.settings import get_secrets as _get_secrets

    checks: dict[str, str] = {}
    healthy = True

    try:
        container = get_container()
        container.dense_store._client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"
        healthy = False

    try:
        _r = _redis_mod.from_url(_get_secrets().redis_url, socket_connect_timeout=2)
        _r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
        healthy = False

    if not healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}


@app.get("/logs/summary", summary="Feedback log line counts")
async def logs_summary():
    def count_lines(p: Path) -> int:
        return sum(1 for _ in p.open()) if p.exists() else 0

    return {
        "liked": count_lines(LIKED_LOG),
        "disliked": count_lines(DISLIKED_LOG),
        "unknown": count_lines(UNKNOWN_LOG),
        "active_sessions": len(_session_histories),
    }
