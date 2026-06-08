# Multimodal RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for querying large document corpora with grounded, cited answers. Combines dense vector search, BM25 sparse retrieval, a Neo4j knowledge graph (GraphRAG), and a multi-agent orchestration layer built on the OpenAI Agents SDK.
<p align="center"><img src="docs/demo.gif" alt="Demo"/></p>

---

## Architecture Overview

```
User Query
    │
    ▼
FastAPI  ──►  AccessControlMiddleware (JWT + namespace isolation)
    │
    ▼
QueryUnderstanding  ──►  intent classification · HyDE expansion · NER · decomposition
    │
    ├──► SemanticCache (Redis)  ──►  cache hit → return immediately
    │
    ▼
OrchestratorAgent  ──►  routes to RetrievalAgent · ConversationalAgent · FollowUpAgent
    │
    ▼
HybridSearchEngine
    ├── DenseVectorStore  (Qdrant HNSW, cosine ANN)
    ├── SparseIndex       (Elasticsearch BM25)
    └── GraphRetriever    (Neo4j — LOCAL k-hop · GLOBAL community · HYBRID)
    │
    ▼
Post-Retrieval  ──►  parent expansion · sentence window · Cohere rerank · LLM Lingua compression
    │
    ▼
Generator  ──►  conflict detection · grounded prompting · citation extraction · faithfulness check
    │
    ▼
PIIGuard  ──►  Presidio output scan
    │
    ▼
RAGEvaluator  ──►  RAGAS metrics · query drift detection · feedback logging
```

---

## Key Features

**Ingestion**
- Multimodal parsing: PDF (`hi_res` + OCR), DOCX, Markdown, HTML, images (GPT-4o captioning + CLIP embeddings)
- Table extraction via Camelot with merged-cell and multi-page support
- 3-strategy chunking: fixed-window, semantic (cosine breakpoints), hierarchical (document → section → paragraph)
- PII redaction via Microsoft Presidio before indexing
- SHA-256 fingerprinting, MinHash near-dedup (Jaccard ≥ 0.85), delta ingestion to skip unchanged files

**Retrieval**
- Hybrid search: Qdrant HNSW dense + Elasticsearch BM25 sparse, fused via Reciprocal Rank Fusion (k=60)
- HyDE: GPT-4-turbo writes a hypothetical answer passage; its embedding bridges query ↔ document space
- Multi-query decomposition: complex questions split into ≤4 sub-questions, each retrieved independently
- Parent-child expansion: paragraph hits promoted to their section node for broader LLM context
- Cohere cross-encoder reranking with empty-list guard

**GraphRAG** (optional, toggle via `config.yaml`)
- Entity/relationship extraction via async GPT-4o with rate limiting and retry
- Neo4j storage with 1024-d cosine vector index on entity embeddings
- Three retrieval modes:
  - **LOCAL** — k-hop neighbourhood traversal from NER-extracted entities
  - **GLOBAL** — ANN search over Louvain/Leiden community summaries
  - **HYBRID** — all three sources fused via RRF (default)

**Generation & Safety**
- Conflict detection across retrieved passages before generation
- Grounded prompting: model instructed to cite every claim with `[CITE:chunk_id]`
- Faithfulness guardrail: blocks responses scoring below 0.40 (RAGAS)
- Output PII scanning; 3-stage Presidio pipeline (ingest → input → output)

**Operations**
- JWT multi-tenancy: namespace isolation enforced at every retrieval layer
- Redis semantic cache: cosine similarity ≥ 0.95, TTL 3600s, namespace-scoped keys
- Query drift detection: rolling 1000-query reference window, cosine divergence alert
- TraceSpan structured latency tracing, LangSmith / Arize Phoenix / Helicone backends
- RAGAS online evaluation + custom LLM judge (GPT-4-turbo)

---

## Project Structure

```
multimodal-rag-system/
├── app/
│   ├── main.py              # FastAPI application, all endpoints
│   ├── auth.py              # JWT auth, user registration, SQLite auth.db
│   └── ui.html              # Browser chat UI
├── src/
│   ├── agents/              # OpenAI Agents SDK — orchestrator, retrieval, conversational, followup
│   ├── config/              # settings.py — typed config loader from config.yaml
│   ├── core/                # container.py — dependency wiring and startup
│   ├── evaluation/          # evaluator.py — RAGAS, custom judge, drift detection
│   ├── generation/          # generator.py — conflict detection, grounded prompting, citations
│   ├── graphrag/            # extractor, neo4j_store, graph_retriever, community, schema
│   ├── indexing/            # embedder.py, vector_store.py (Qdrant)
│   ├── ingestion/           # parser, chunker, consolidator, deduplicator, pipeline, graph_handler
│   ├── operations/          # ops_middleware.py — PII, semantic cache, tracing, ACL
│   ├── query/               # understanding.py, pipeline.py
│   ├── retrieval/           # retriever.py — hybrid search, RRF, reranking, context management
│   └── utils/               # logger.py, file_utils.py
├── scripts/
│   ├── ingest.py            # CLI ingestion runner
│   ├── query.py             # CLI query runner
│   └── build_communities.py # Run Louvain/Leiden community detection (post-ingest)
├── config/
│   └── config.yaml          # All system configuration
├── knowledge_base/          # Source documents, organized by namespace subfolder
│   └── <namespace>/
├── .env.example             # Environment variable template
└── requirements.txt
```

---

## Quickstart

### 1. Prerequisites

| Service | Purpose | Default port |
|---|---|---|
| Qdrant | Dense vector store | 6333 |
| Elasticsearch | BM25 sparse index | 9200 |
| Redis | Semantic cache | 6379 |
| Neo4j *(optional)* | GraphRAG knowledge graph | 7687 |

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, COHERE_API_KEY, service URLs, JWT_SECRET_KEY
```

### 4. Add documents

Place documents inside `knowledge_base/<namespace>/` — subfolders become tenant namespaces.

```
knowledge_base/
├── ProjectA/
│   ├── report.pdf
│   └── guidelines.docx
└── ProjectB/
    └── dataset.md
```

### 5. Ingest

```bash
# Ingest a specific namespace
python scripts/ingest.py --namespace ProjectA

# Ingest all namespaces
python scripts/ingest.py --all

# After ingestion, optionally build GraphRAG communities (expensive, one-time)
python scripts/build_communities.py
```

### 6. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the browser UI, or use the API directly.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register a new user |
| `POST` | `/auth/login` | Authenticate; returns JWT |
| `GET` | `/namespaces` | List available knowledge-base namespaces |
| `POST` | `/query` | Run a RAG query |
| `GET` | `/chunk/{chunk_id}` | Fetch raw chunk text by ID |
| `POST` | `/feedback` | Submit thumbs-up / thumbs-down feedback |
| `DELETE` | `/session/{session_id}` | Clear conversation history |
| `GET` | `/health` | Liveness probe |
| `GET` | `/logs/summary` | Feedback log line counts |

### Query request

```json
{
  "question": "What are the data retention requirements for clinical trial records?",
  "namespace": "ProjectA",
  "session_id": "abc-123",
  "pii_block_on_input": false
}
```

### Query response

```json
{
  "answer": "Clinical trial records must be retained for at least 2 years after the last marketing application approval or discontinuation of development [1][2].",
  "citations": [
    { "source_file": "report.pdf", "ingestion_ts": "2025-01-01", "excerpt": "..." }
  ],
  "faithfulness_score": 0.91,
  "session_id": "abc-123"
}
```

---

## Configuration

All behaviour is controlled by `config/config.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `knowledge_base` | Source document root, supported formats |
| `ingestion` | PDF strategy, OCR, table extraction, image captioning, deduplication |
| `chunking` | Strategy (`fixed` / `semantic` / `hierarchical`), sizes, overlap |
| `embeddings` | Model selection, domain-specific overrides, multilingual fallback |
| `vector_store` | Qdrant collection, HNSW parameters, hybrid search, RRF constant |
| `query` | HyDE, decomposition, NER, conversation history, intent routing |
| `retrieval` | top-k, parent-child, sentence window, Cohere reranking, context compression |
| `generation` | Model, faithfulness check, citations, conflict handling |
| `evaluation` | RAGAS metrics, LLM judge, drift detection, synthetic QA generation |
| `operations` | Semantic cache, JWT ACL, PII entities, observability backends |
| `graphrag` | Enable/disable, extraction model, entity types, relationship types, retrieval mode |

---

## GraphRAG Configuration

Enable in `config.yaml`:

```yaml
graphrag:
  enabled: true
  retrieval:
    mode: "hybrid"          # local | global | hybrid
    local_hop_depth: 2
    community_top_k: 5
```

The system ships with a rich domain-specific schema covering 60+ entity types and 80+ relationship types across clinical research, biomarker science, cognitive assessment, digital health, genetics, statistics, and ML methodology.

After ingestion, run community detection once:

```bash
python scripts/build_communities.py
```

GraphRAG falls back gracefully to vector-only retrieval if Neo4j is unavailable.

---

## Embedding Models

The system routes chunks to domain-appropriate embedding models:

| Domain | Model |
|---|---|
| Default | `BAAI/bge-large-en-v1.5` (1024d) |
| Medical | `pritamdeka/S-PubMedBert-MS-MARCO` |
| Multilingual | `intfloat/multilingual-e5-large` |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | GPT-4o / GPT-4-turbo for generation and extraction |
| `COHERE_API_KEY` | ✅ | Cohere reranker |
| `QDRANT_URL` | ✅ | Qdrant instance URL |
| `ELASTICSEARCH_URL` | ✅ | Elasticsearch instance URL |
| `REDIS_URL` | ✅ | Redis instance URL |
| `NEO4J_URI` | GraphRAG only | Neo4j bolt URI |
| `NEO4J_USERNAME` | GraphRAG only | Neo4j username |
| `NEO4J_PASSWORD` | GraphRAG only | Neo4j password |
| `JWT_SECRET_KEY` | ✅ | HS256 signing secret (32+ chars) |

---

## Application Domains

This system is designed for document-heavy domains where answers must be grounded, cited, and auditable:

- **Clinical & Scientific Research** — querying literature corpora, surfacing findings, biomarkers, and study results
- **Regulatory & Compliance** — FDA submissions, clinical trial documentation, audit trails
- **Healthcare Knowledge Management** — multi-tenant knowledge bases for research teams
- **Enterprise Document Retrieval** — large unstructured document stores (PDFs, reports, guidelines)
- **Systematic Review Acceleration** — cross-document reasoning via GraphRAG entity/relationship extraction
