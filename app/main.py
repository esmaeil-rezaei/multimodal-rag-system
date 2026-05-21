from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.agents.orchestrator import RAGOrchestrator
from src.config.settings import get_config

_cfg = get_config()
 
_log_cfg = _cfg.log 
 
UNKNOWN_LOG  = Path(_log_cfg["unknown_query_dir"])    / "unknown_queries.jsonl"
LIKED_LOG    = Path(_log_cfg["liked_response_dir"])   / "liked_responses.jsonl"
DISLIKED_LOG = Path(_log_cfg["disliked_response_dir"]) / "disliked_responses.jsonl"
 
UNKNOWN_LOG.parent.mkdir(parents=True, exist_ok=True)
LIKED_LOG.parent.mkdir(parents=True, exist_ok=True)
DISLIKED_LOG.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Query UI", version="1.0.0")

orchestrator = RAGOrchestrator()


MAX_HISTORY_TURNS = 20 
_session_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)


def _get_history(session_id: str) -> List[Dict[str, str]]:
    return _session_histories[session_id]


def _append_history(session_id: str, question: str, answer: str) -> None:
    history = _session_histories[session_id]
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        _session_histories[session_id] = history[-max_messages:]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list = []
    faithfulness_score: Optional[float] = None
    has_conflict: bool = False
    conflict_resolution: Optional[str] = None
    model_used: str = "unknown"
    could_not_answer: bool = False


class FeedbackRequest(BaseModel):
    query_id: str
    question: str
    answer: str
    rating: str          # "like" | "dislike" | "unknown"
    comment: Optional[str] = None
    session_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    query_id   = str(uuid.uuid4())
    session_id = req.session_id or "default"

    history = _get_history(session_id)

    try:
        result = await orchestrator.run(
            raw_query=req.question,
            conversation_history=history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    could_not_answer = (
        not result.answer
        or result.answer.strip().lower().startswith("i'm sorry")
        or result.answer.strip().lower().startswith("i was unable")
        or result.answer.strip().lower().startswith("an internal error")
        or (result.faithfulness_score is not None and result.faithfulness_score < 0.3)
    )

    if could_not_answer:
        _write_jsonl(UNKNOWN_LOG, {
            "query_id":          query_id,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "question":          req.question,
            "answer":            result.answer,
            "faithfulness_score": result.faithfulness_score,
            "model_used":        result.model_used,
            "session_id":        session_id,
        })
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
    )


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    ts = datetime.now(timezone.utc).isoformat()

    record = {
        "query_id":   req.query_id,
        "timestamp":  ts,
        "question":   req.question,
        "answer":     req.answer,
        "rating":     req.rating,
        "comment":    req.comment or "",
        "session_id": req.session_id or "",
    }


    if req.rating == "like":
        _write_jsonl(LIKED_LOG, record)
    elif req.rating == "dislike":
        _write_jsonl(DISLIKED_LOG, record)
    elif req.rating == "unknown":
        _write_jsonl(UNKNOWN_LOG, {**record, "source": "user_flagged"})

    return {"status": "ok", "query_id": req.query_id}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Optional endpoint to reset a session's conversation history."""
    _session_histories.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}


@app.get("/logs/summary")
async def logs_summary():
    def count_lines(p: Path) -> int:
        if not p.exists():
            return 0
        with p.open() as f:
            return sum(1 for _ in f)

    def read_last(p: Path, n: int = 5) -> list:
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]

    return {
        "liked":            count_lines(LIKED_LOG),
        "disliked":         count_lines(DISLIKED_LOG),
        "unknown":          count_lines(UNKNOWN_LOG),
        "active_sessions":  len(_session_histories),
    }