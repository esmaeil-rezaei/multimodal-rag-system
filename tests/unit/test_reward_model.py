from __future__ import annotations

import json

import pytest

from src.rlhf.reward_model import build_reward_dataset, load_preference_pairs


def test_load_preference_pairs_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preference_pairs(tmp_path / "missing.jsonl")


def test_load_preference_pairs_empty_file_raises(tmp_path):
    path = tmp_path / "preference_pairs.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_preference_pairs(path)


def test_load_preference_pairs_reads_jsonl(tmp_path):
    path = tmp_path / "preference_pairs.jsonl"
    records = [
        {"prompt": "Q1", "chosen": "good", "rejected": "bad", "meta": {}},
        {"prompt": "Q2", "chosen": "good2", "rejected": "bad2", "meta": {"k": "v"}},
    ]
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    loaded = load_preference_pairs(path)

    assert loaded == records


def test_build_reward_dataset_wraps_records_in_hf_dataset():
    pytest.importorskip("datasets")

    records = [
        {"prompt": "Q1", "chosen": "good", "rejected": "bad", "meta": {}},
    ]

    dataset = build_reward_dataset(records)

    assert dataset.column_names == ["prompt", "chosen", "rejected"]
    assert len(dataset) == 1
    assert dataset[0]["prompt"] == "Q1"
