from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_QUERIES = PROJECT_ROOT / "tests" / "golden_queries.json"
GOLDEN_CONVERSATIONS = PROJECT_ROOT / "tests" / "golden_conversations.json"
GOLDEN_FAIRNESS_PAIRS = PROJECT_ROOT / "tests" / "golden_fairness_pairs.json"


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)




def test_golden_queries_file_exists_and_is_a_list():
    assert GOLDEN_QUERIES.exists()
    data = _load(GOLDEN_QUERIES)
    assert isinstance(data, list)
    assert len(data) > 0


def test_golden_queries_have_required_fields():
    data = _load(GOLDEN_QUERIES)
    for item in data:
        assert isinstance(item["query"], str) and item["query"]
        assert isinstance(item["relevant_chunk_ids"], list)
        assert all(isinstance(cid, str) for cid in item["relevant_chunk_ids"])
        assert isinstance(item.get("expected_answer_keywords", []), list)
        assert isinstance(item.get("namespace", "default"), str)
        assert isinstance(item.get("metadata", {}), dict)


def test_golden_queries_have_unique_query_text():
    data = _load(GOLDEN_QUERIES)
    queries = [item["query"] for item in data]
    assert len(queries) == len(set(queries)), "duplicate queries found in golden_queries.json"




def test_golden_conversations_file_exists_and_is_a_list():
    assert GOLDEN_CONVERSATIONS.exists()
    data = _load(GOLDEN_CONVERSATIONS)
    assert isinstance(data, list)
    assert len(data) > 0


def test_golden_conversations_have_required_fields():
    data = _load(GOLDEN_CONVERSATIONS)
    for convo in data:
        assert isinstance(convo["conversation_id"], str) and convo["conversation_id"]
        assert isinstance(convo.get("namespace", "default"), str)
        turns = convo["turns"]
        assert isinstance(turns, list) and len(turns) > 0
        for turn in turns:
            assert isinstance(turn["query"], str) and turn["query"]
            assert isinstance(turn.get("expected_answer_keywords", []), list)
            assert isinstance(turn.get("relevant_chunk_ids", []), list)
            assert isinstance(turn.get("expects_condensation", False), bool)
            if turn.get("expects_condensation"):
                # condensation checks should specify at least one constraint
                must_contain = turn.get("condensation_must_contain", [])
                must_not_contain = turn.get("condensation_must_not_contain", [])
                assert isinstance(must_contain, list)
                assert isinstance(must_not_contain, list)


def test_golden_conversations_have_unique_ids():
    data = _load(GOLDEN_CONVERSATIONS)
    ids = [c["conversation_id"] for c in data]
    assert len(ids) == len(set(ids)), "duplicate conversation_id values found"




def test_golden_fairness_pairs_file_exists_and_is_a_list():
    assert GOLDEN_FAIRNESS_PAIRS.exists()
    data = _load(GOLDEN_FAIRNESS_PAIRS)
    assert isinstance(data, list)
    assert len(data) > 0


def test_golden_fairness_pairs_have_required_fields():
    data = _load(GOLDEN_FAIRNESS_PAIRS)
    for pair in data:
        assert isinstance(pair["pair_id"], str) and pair["pair_id"]
        assert isinstance(pair.get("dimension", "unknown"), str)
        assert isinstance(pair.get("namespace", "default"), str)
        variants = pair["variants"]
        assert isinstance(variants, list)
        assert len(variants) >= 2, f"pair {pair['pair_id']} needs at least 2 variants to compare"
        for variant in variants:
            assert isinstance(variant["label"], str) and variant["label"]
            assert isinstance(variant["query"], str) and variant["query"]


def test_golden_fairness_pairs_have_unique_ids():
    data = _load(GOLDEN_FAIRNESS_PAIRS)
    ids = [p["pair_id"] for p in data]
    assert len(ids) == len(set(ids)), "duplicate pair_id values found"


def test_golden_fairness_pairs_variant_labels_unique_within_pair():
    data = _load(GOLDEN_FAIRNESS_PAIRS)
    for pair in data:
        labels = [v["label"] for v in pair["variants"]]
        assert len(labels) == len(
            set(labels)
        ), f"duplicate variant labels in pair {pair['pair_id']}"
