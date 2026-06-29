from __future__ import annotations

import json
from src.tuning.embedding_dataset import (
    EmbeddingTriplet,
    build_embedding_dataset,
    save_embedding_dataset,
)
from src.tuning.reranker_dataset import InMemoryChunkResolver

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


def test_build_embedding_dataset_creates_one_triplet_per_positive():
    resolver = InMemoryChunkResolver(CHUNKS)

    triplets = build_embedding_dataset(GOLDEN_QUERIES, resolver)

    # Query 1 has 2 positives -> 2 triplets, query 2 has 1 positive -> 1 triplet.
    assert len(triplets) == 3
    assert all(isinstance(t, EmbeddingTriplet) for t in triplets)
    for t in triplets:
        assert t.anchor in {"What is X?", "What is Y?"}
        assert t.positive
        assert t.negative
        # Negative must not be one of the query's own positive passages.
        assert t.negative != t.positive


def test_build_embedding_dataset_skips_query_with_no_negatives_available():
    # Single query -> no other query's positives exist as negatives.
    queries = [GOLDEN_QUERIES[0]]
    resolver = InMemoryChunkResolver(CHUNKS)

    assert build_embedding_dataset(queries, resolver) == []


def test_build_embedding_dataset_skips_unresolvable_positive(tmp_path):
    resolver = InMemoryChunkResolver({"c2": CHUNKS["c2"], "c3": CHUNKS["c3"]})

    triplets = build_embedding_dataset(GOLDEN_QUERIES, resolver)

    # c1 is unresolvable, so query 1 only yields a triplet for c2.
    anchors_with_positive = {(t.anchor, t.positive) for t in triplets}
    assert ("What is X?", CHUNKS["c1"]) not in anchors_with_positive


def test_build_embedding_dataset_custom_negative_sampler():
    resolver = InMemoryChunkResolver(CHUNKS)

    def sampler(query, all_queries, n):
        return ["c3"] * n if query["query"] == "What is X?" else ["c1"] * n

    triplets = build_embedding_dataset(GOLDEN_QUERIES, resolver, negative_sampler=sampler)

    x_triplets = [t for t in triplets if t.anchor == "What is X?"]
    assert all(t.negative == CHUNKS["c3"] for t in x_triplets)


def test_save_embedding_dataset_writes_jsonl(tmp_path):
    triplets = [
        EmbeddingTriplet(anchor="A", positive="P", negative="N", meta={"k": "v"}),
    ]
    output_path = tmp_path / "data" / "tuning" / "embedding_triplets.jsonl"

    result = save_embedding_dataset(triplets, output_path)

    assert result == output_path
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"anchor": "A", "positive": "P", "negative": "N", "meta": {"k": "v"}}
