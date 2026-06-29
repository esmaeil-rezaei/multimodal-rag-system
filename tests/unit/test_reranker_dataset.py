from __future__ import annotations

import json

from src.tuning.reranker_dataset import (
    InMemoryChunkResolver,
    RerankerExample,
    build_reranker_dataset,
    save_reranker_dataset,
)

GOLDEN_QUERIES = [
    {
        "query": "What is X?",
        "relevant_chunk_ids": ["c1", "c2"],
        "namespace": "ns1",
    },
    {
        "query": "What is Y?",
        "relevant_chunk_ids": ["c3"],
        "namespace": "ns2",
    },
]

CHUNKS = {
    "c1": "X is the first thing.",
    "c2": "X is also described here.",
    "c3": "Y is the second thing.",
}


def test_build_reranker_dataset_creates_positives_and_negatives():
    resolver = InMemoryChunkResolver(CHUNKS)

    examples = build_reranker_dataset(GOLDEN_QUERIES, resolver, negatives_per_query=1)

    positives = [e for e in examples if e.label == 1.0]
    negatives = [e for e in examples if e.label == 0.0]

    assert len(positives) == 3  # c1, c2 for query 1; c3 for query 2
    assert len(negatives) == 2  # one negative per query
    assert all(isinstance(e, RerankerExample) for e in examples)

    # Negative for "What is X?" must come from the other query's chunks (c3),
    # never one of its own positives.
    x_negatives = [e for e in negatives if e.query == "What is X?"]
    assert all(e.passage == CHUNKS["c3"] for e in x_negatives)


def test_build_reranker_dataset_skips_unresolvable_chunks():
    resolver = InMemoryChunkResolver({"c1": "X is the first thing."})

    examples = build_reranker_dataset(GOLDEN_QUERIES, resolver, negatives_per_query=1)

    # Only c1 resolves; c2 and c3 are skipped silently.
    assert all(e.passage == "X is the first thing." or e.label == 0.0 for e in examples)
    resolved_passages = {e.passage for e in examples}
    assert "X is the first thing." in resolved_passages


def test_build_reranker_dataset_skips_queries_without_relevant_ids():
    queries = [{"query": "No chunks", "relevant_chunk_ids": []}]
    resolver = InMemoryChunkResolver(CHUNKS)

    assert build_reranker_dataset(queries, resolver) == []


def test_build_reranker_dataset_custom_negative_sampler():
    resolver = InMemoryChunkResolver(CHUNKS)

    def sampler(query, all_queries, n):
        return ["c3"] if query["query"] == "What is X?" else []

    examples = build_reranker_dataset(GOLDEN_QUERIES, resolver, negative_sampler=sampler)

    negatives = [e for e in examples if e.label == 0.0]
    assert len(negatives) == 1
    assert negatives[0].query == "What is X?"
    assert negatives[0].passage == CHUNKS["c3"]


def test_save_reranker_dataset_writes_jsonl(tmp_path):
    examples = [
        RerankerExample(query="Q", passage="P", label=1.0, meta={"chunk_id": "c1"}),
    ]
    output_path = tmp_path / "data" / "tuning" / "reranker_pairs.jsonl"

    result = save_reranker_dataset(examples, output_path)

    assert result == output_path
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"query": "Q", "passage": "P", "label": 1.0, "meta": {"chunk_id": "c1"}}
