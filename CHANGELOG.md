# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-07-23

### Added

- LLM-based intent routing: `_classify_routing_intent` in `QueryUnderstanding`
  replaced keyword/heuristic lists with a `gpt-4o-mini` call (`max_tokens=5`,
  temperature 0). Routes to `conversational`, `followup`, or `retrieval` with
  full conversation-history context. Falls back to `retrieval` on error.
- Direct followup streaming path: followup queries bypass the Agents SDK and
  stream directly from conversation history. Buffers the first 16 chars to
  detect a `NEEDS_RETRIEVAL` signal before yielding tokens; falls through to
  the retrieval pipeline when history is insufficient.
- Direct conversational path: social/greeting queries bypass retrieval entirely
  and stream a short `gpt-4o-mini` reply with conversation-history context.
- `agent_name` field in SSE `done` events; the UI badge now shows `RAG Agent`,
  `Follow-up Agent`, or `Assistant` instead of the internal model string.
- `src/utils/retry.py`: `openai_retry` tenacity decorator (3 attempts,
  exponential backoff 1–8 s) for `RateLimitError`, `APITimeoutError`,
  `APIConnectionError`, and `InternalServerError`. Applied to `generate()`,
  `_detect_conflicts()`, `_ragas_faithfulness()`, `_selfcheck_gpt_faithfulness()`,
  and the streaming create helper `_async_create_stream()`.
- `httpx.Timeout` on all OpenAI client instantiations (connect 5 s, read 60 s
  for generation / 30 s elsewhere) and the Cohere client; `max_retries=0`
  hands retry control to tenacity.
- `GET /ready` readiness probe: checks Qdrant and Redis; returns `200` when
  both are available, `503` with a per-service breakdown otherwise.

### Changed

- Routing classification, history compression, and preference detection now run
  concurrently via `asyncio.gather` in both `run()` and `run_stream()`.
- `model_used` in `done` SSE events now carries the actual model identifier
  instead of internal routing labels (`direct_conversational`, `no-context`,
  `guardrail`).

## [1.1.0] - 2026-06-28

### Added

- Per-session user preference store: behavioral instructions detected by
  `gpt-4o-mini` and persisted as JSON at
  `logs/user_preferences/<namespace>.json`.
- Preference reconciler: REPLACE / ADD / SKIP logic prevents contradicting
  preferences from stacking.
- REDO action: re-runs the previous question with an updated preference applied.
- CLEAR action: wipes all stored preferences for the namespace.
- Mixed-message handling: a single turn can store a preference and route a
  new question through retrieval simultaneously.
- History compression: `compress_history()` in `QueryUnderstanding` summarises
  overflow turns into a `[CONVERSATION SUMMARY]` system message via
  `gpt-4o-mini`; configurable via `query.conversation.compress_*` in
  `config.yaml`.
- RLHF pipeline (`src/rlhf/`): preference pair dataset builder, reward model
  training (LoRA on `Qwen/Qwen2.5-0.5B-Instruct`), composite reward signal
  (learned reward + citation presence + RAGAS faithfulness + length penalty),
  PPO fine-tuning, and DPO fine-tuning via TRL.
- Embedding fine-tuning (`src/tuning/`): triplet dataset builder with hard
  negative mining, `MultipleNegativesRankingLoss` trainer for
  `BAAI/bge-large-en-v1.5`.
- Reranker fine-tuning (`src/tuning/`): pairwise dataset builder and
  `CrossEncoder` trainer for `BAAI/bge-reranker-large`.
- Layer 5 — Cost & latency SLO evaluation (`cost_latency_eval.py`): per-query
  token cost and USD calculation, p50/p95/p99 latency tracking, and
  configurable SLO thresholds with score ledger.
- Layer 6 — Multi-turn evaluation (`multiturn_eval.py`): condensation quality
  checks, follow-up retrieval hit rate, and optional full-pipeline faithfulness
  scoring across `tests/golden_conversations.json`.
- Layer 7 — Fairness evaluation (`fairness_eval.py`): pairwise Jaccard
  similarity of retrieved chunk sets and cosine similarity of answer embeddings
  across counterfactual demographic query groups
  (`tests/golden_fairness_pairs.json`).
- Pytest unit test suite (`tests/unit/`) covering config/settings, auth,
  chunking, retrieval metrics, cost/latency evaluation, multi-turn evaluation,
  fairness evaluation, RLHF feedback dataset, reward functions, reward model,
  embedding dataset, and reranker dataset.
- GitHub Actions CI workflow (lint, format check, type-check, unit tests).
- `pyproject.toml` with project metadata and ruff/black/mypy configuration.
- Split dependency files: `requirements.in`, `requirements.txt`,
  `requirements-dev.txt`.
- `Dockerfile` and `docker-compose.yml`.
- `.pre-commit-config.yaml` with ruff, black, mypy, and hygiene hooks.
- `LICENSE` (MIT), `CONTRIBUTING.md`, and this changelog.

### Changed

- Preference detection runs on both `retrieval` and `conversational` routes,
  so short behavioral notes like "be concise" are captured regardless of
  routing intent.
- `.gitignore` extended to exclude `logs/`, `qdrant_storage/`, `eval_results/`,
  and `artifacts/`.

## [1.0.0] - 2026-06-07

### Added

- 4-layer offline evaluation suite: indexing quality (Layer 1), retrieval
  metrics — Precision/Recall/MRR/Hit Rate/NDCG@K (Layer 2), RAGAS-based
  generation quality (Layer 3), and golden regression suite with baseline
  comparison (Layer 4).
- Neo4j-backed GraphRAG layer with community detection.
- JWT-based authentication with namespace federation.
- Multi-agent orchestration layer for query routing, condensation, and
  sub-question retrieval.
- End-to-end query pipeline (retrieval, reranking, generation).
- End-to-end ingestion pipeline: parsing, chunking, deduplication,
  metadata stamping, PII handling, and indexing.

### Fixed

- Follow-up query routing and history-based answering in the multi-agent
  pipeline.

## [0.1.0] - 2026-04-24

### Added

- Initial project scaffold for the multimodal AI shopping recommendation
  assistant.
