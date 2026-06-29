# src/tuning — Embedding & Reranker Fine-tuning

This package fine-tunes the two retrieval models — the bi-encoder embedder and the cross-encoder reranker — on domain-specific examples mined from the indexed corpus and the golden retrieval labels in `tests/golden_queries.json`.

Fine-tuning is optional: the system ships with strong off-the-shelf defaults (`BAAI/bge-large-en-v1.5` for embedding, `BAAI/bge-reranker-large` for reranking). Run these pipelines when you have accumulated enough annotated retrieval labels and want to specialise the models to your domain.

---

## Files

### `embedding_dataset.py`

Builds `(anchor, positive, negative)` triplets for contrastive embedding training.

- Positives are the `relevant_chunk_ids` from golden queries
- Negatives are supplied by `HardNegativeMiner` — passages ranked highly by the current retriever but absent from the golden set
- Writes to `data/tuning/embedding_triplets.jsonl`

```bash
python -m src.tuning.embedding_dataset
```

---

### `hard_negative_miner.py`

Mines hard negatives for both the embedding and reranker datasets.

For each golden query, retrieves the top-`top_k_candidates` (default 20) passages from Qdrant and filters to those whose cosine similarity to the query falls within the band `[similarity_floor, similarity_ceiling]` (default 0.3–0.85). Passages inside this band are confusable but not relevant — the hardest and most informative negatives.

Key configuration (`config/config.yaml` under `tuning.hard_negatives`):

| Parameter | Default | Notes |
|---|---|---|
| `num_per_query` | 3 | Hard negatives per query |
| `top_k_candidates` | 20 | Retriever candidates to filter from |
| `similarity_floor` | 0.3 | Minimum similarity (below = too easy) |
| `similarity_ceiling` | 0.85 | Maximum similarity (above = likely relevant) |

---

### `embedding_trainer.py`

Fine-tunes `BAAI/bge-large-en-v1.5` on `(anchor, positive, negative)` triplets.

- Uses `sentence_transformers` `MultipleNegativesRankingLoss` (in-batch negatives + explicit hard negatives)
- Outputs fine-tuned checkpoint to `models/embedder/`
- To deploy: set `embeddings.default_model` in `config/config.yaml` to `models/embedder/`

```bash
python -m src.tuning.embedding_trainer
```

Key hyperparameters (all in `config/config.yaml` under `tuning.embedder`):

| Parameter | Default |
|---|---|
| `num_epochs` | 3 |
| `batch_size` | 32 |
| `learning_rate` | 2e-5 |
| `warmup_ratio` | 0.1 |

---

### `reranker_dataset.py`

Builds `(query, passage, label)` pairs for cross-encoder training.

- Positives: `relevant_chunk_ids` from golden queries (`label = 1`)
- Negatives: hard-mined passages from `HardNegativeMiner` + BM25 candidates (`label = 0`)
- Writes to `data/tuning/reranker_pairs.jsonl`

```bash
python -m src.tuning.reranker_dataset
```

---

### `reranker_trainer.py`

Fine-tunes `BAAI/bge-reranker-large` on `(query, passage, label)` pairs.

- Uses `sentence_transformers` `CrossEncoder.fit` with binary cross-entropy loss
- Outputs fine-tuned checkpoint to `models/reranker/`
- To deploy: set `retrieval.reranking.bge_model` in `config/config.yaml` to `models/reranker/`

```bash
python -m src.tuning.reranker_trainer
```

Key hyperparameters (all in `config/config.yaml` under `tuning.reranker`):

| Parameter | Default |
|---|---|
| `num_epochs` | 2 |
| `batch_size` | 16 |
| `learning_rate` | 2e-5 |
| `warmup_ratio` | 0.1 |
| `max_length` | 512 |
| `negatives_per_query` | 2 |

---

## Full workflow

```bash
# 1. Mine hard negatives and build both datasets in one pass
python -m src.tuning.embedding_dataset   # also calls HardNegativeMiner internally
python -m src.tuning.reranker_dataset

# 2. Fine-tune
python -m src.tuning.embedding_trainer
python -m src.tuning.reranker_trainer

# 3. Point config/config.yaml at the new checkpoints
#    embeddings.default_model: "models/embedder/"
#    retrieval.reranking.bge_model: "models/reranker/"

# 4. Re-ingest (new embedder) or restart the server (new reranker) to activate
```

After swapping in the fine-tuned embedder you must re-index the corpus (`python scripts/ingest.py --all`) since embedding vectors will differ from the original model. The reranker can be hot-swapped without re-indexing.

---

## When to fine-tune

- **Embedder**: when Recall@10 on the golden query set plateaus and you have ≥ 500 labeled query–chunk pairs.
- **Reranker**: when the pipeline retrieves the right chunks but they are not ranked in the top-3 positions (low MRR despite acceptable Recall@10).

Run `scripts/evaluate_offline.py --layer2` before and after fine-tuning to measure lift.
