# Offline Evaluation

`scripts/evaluate_offline.py` is the CLI entry point for the seven-layer RAG evaluation suite.

## Layers

| Flag | Layer | What it measures |
|------|-------|-----------------|
| `--layer1` | Indexing quality | Chunk coherence, embedding spot-check, entity coverage |
| `--layer2` | Retrieval quality | Precision@K, Recall@K, MRR, Hit Rate, NDCG |
| `--layer4` | System-level | Golden regression suite + baseline comparison |
| `--layer5` | Cost & latency | Token cost, p50/p95/p99 latency SLO tracking |
| `--layer6` | Multi-turn | Context carryover, follow-up retrieval |
| `--layer7` | Fairness/bias | Counterfactual demographic framings |

**Default run (no flags):** layers 1, 2, and 4.

Layers 5–7 are opt-in; they make additional LLM calls and are excluded from the default run.

Layer 3 (generation/RAGAs) runs online via `RAGEvaluator.evaluate_online()` at inference time, where the full query + context + answer triad is available, so it is intentionally omitted here.

## Usage

```bash
# Default suite (layers 1, 2, 4)
python scripts/evaluate_offline.py --golden tests/golden_queries.json

# Retrieval-only with K=5
python scripts/evaluate_offline.py --layer2 --k 5 --golden tests/golden_queries.json

# Indexing layer on a specific namespace
python scripts/evaluate_offline.py --layer1 --namespace radiology --sample 30

# Cost/latency SLO tracking
python scripts/evaluate_offline.py --layer5 --ledger eval_results/scores.jsonl

# Multi-turn evaluation (condensation + follow-up retrieval only)
python scripts/evaluate_offline.py --layer6

# Multi-turn evaluation with full pipeline answers (keyword + faithfulness checks)
python scripts/evaluate_offline.py --layer6 --answer

# Fairness/bias evaluation (retrieval consistency only)
python scripts/evaluate_offline.py --layer7

# Fairness/bias evaluation with full pipeline answers (answer-similarity checks)
python scripts/evaluate_offline.py --layer7 --answer

# Save scores to a tracking ledger and JSON report
python scripts/evaluate_offline.py \
    --golden tests/golden_queries.json \
    --ledger eval_results/scores.jsonl \
    --output eval_results/report.json
```
