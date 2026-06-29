# Embedding & Reranker Fine-Tuning

Both pipelines optionally build their own dataset before training. A live Qdrant connection is required for `--build-dataset`.

## Embedding model (bi-encoder)

```bash
# Build data/tuning/embedding_triplets.jsonl, then train:
python scripts/tuning/train_embedder.py --build-dataset

# Train from an already-built dataset:
python scripts/tuning/train_embedder.py
python scripts/tuning/train_embedder.py --dataset data/tuning/embedding_triplets.jsonl
```

Fine-tunes the bi-encoder on `(anchor, positive, negative)` triplets built from `tests/golden_queries.json`. Requires `sentence-transformers`, `torch`.

## Reranker (cross-encoder)

```bash
# Build data/tuning/reranker_pairs.jsonl, then train:
python scripts/tuning/train_reranker.py --build-dataset

# Train from an already-built dataset:
python scripts/tuning/train_reranker.py
python scripts/tuning/train_reranker.py --dataset data/tuning/reranker_pairs.jsonl
```

Fine-tunes the cross-encoder on `(query, passage, label)` triples. Requires `sentence-transformers`, `torch`.
