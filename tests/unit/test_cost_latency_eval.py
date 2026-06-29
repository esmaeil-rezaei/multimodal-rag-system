from __future__ import annotations

import time

import pytest

from src.evaluation.cost_latency_eval import (
    CostCalculator,
    CostLatencyEvaluator,
    CostLatencySLO,
    LatencyTimer,
    QueryCostLatencyRecord,
    _percentile,
)


def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_value():
    assert _percentile([42.0], 95) == 42.0


def test_percentile_p50_of_sorted_values():
    values = [10, 20, 30, 40, 50]
    # nearest-rank p50 of 5 values -> rank = round(0.5*5) = 2 (1-indexed) -> 20
    assert _percentile(values, 50) == 20


def test_percentile_p99_returns_max_for_small_sample():
    values = [10, 20, 30]
    assert _percentile(values, 99) == 30


def test_latency_timer_records_stage_durations():
    timer = LatencyTimer()
    with timer.stage("retrieval"):
        time.sleep(0.001)
    with timer.stage("generation"):
        time.sleep(0.001)

    assert "retrieval" in timer.stage_ms
    assert "generation" in timer.stage_ms
    assert timer.stage_ms["retrieval"] > 0
    assert timer.stage_ms["generation"] > 0
    assert timer.total_ms >= timer.stage_ms["retrieval"]


def test_latency_timer_accumulates_repeated_stage():
    timer = LatencyTimer()
    with timer.stage("retrieval"):
        time.sleep(0.001)
    first = timer.stage_ms["retrieval"]
    with timer.stage("retrieval"):
        time.sleep(0.001)
    assert timer.stage_ms["retrieval"] > first


def test_cost_calculator_llm_cost_known_model():
    calc = CostCalculator()
    # gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
    cost = calc.llm_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_cost_calculator_llm_cost_unknown_model_falls_back():
    calc = CostCalculator()
    fallback_cost = calc.llm_cost("gpt-4-turbo", prompt_tokens=1_000_000, completion_tokens=0)
    unknown_cost = calc.llm_cost(
        "totally-unknown-model-xyz", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert unknown_cost == pytest.approx(fallback_cost)


def test_cost_calculator_embedding_cost_local_model_is_free():
    calc = CostCalculator()
    cost = calc.embedding_cost("BAAI/bge-large-en-v1.5", num_tokens=1_000_000)
    assert cost == 0.0


def test_cost_calculator_embedding_cost_openai_model():
    calc = CostCalculator()
    cost = calc.embedding_cost("text-embedding-3-small", num_tokens=1_000_000)
    assert cost == pytest.approx(0.02)


def test_cost_calculator_rerank_cost():
    calc = CostCalculator()
    cost = calc.rerank_cost("rerank-english-v3.0", num_searches=1000)
    assert cost == pytest.approx(2.00)


def test_cost_calculator_pricing_overrides_merge_with_defaults():
    overrides = {"gpt-4o-mini": {"input": 1.0}}
    calc = CostCalculator(pricing_overrides=overrides)
    # input price overridden to $1/1M, output price retained from default ($0.60/1M)
    cost = calc.llm_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(1.0 + 0.60)


def test_cost_latency_slo_from_config_loads_section():
    slo = CostLatencySLO.from_config()
    assert isinstance(slo, CostLatencySLO)
    assert isinstance(slo.enabled, bool)


def test_cost_latency_slo_defaults_when_no_config():
    slo = CostLatencySLO()
    assert slo.enabled is True
    assert slo.p50_latency_ms == 2000.0
    assert slo.p95_latency_ms == 6000.0
    assert slo.p99_latency_ms == 10000.0


def test_query_cost_latency_record_total_tokens():
    record = QueryCostLatencyRecord(
        query="test",
        prompt_tokens=100,
        completion_tokens=50,
        judge_prompt_tokens=10,
        judge_completion_tokens=5,
        embedding_tokens=20,
    )
    assert record.total_tokens == 100 + 50 + 10 + 5 + 20


def test_query_cost_latency_record_passed_slo_true_when_no_violations():
    record = QueryCostLatencyRecord(query="test")
    assert record.passed_slo is True


def test_query_cost_latency_record_passed_slo_false_with_violations():
    record = QueryCostLatencyRecord(query="test", slo_violations=["latency too high"])
    assert record.passed_slo is False


def test_check_slo_flags_latency_violation():
    slo = CostLatencySLO(
        p99_latency_ms=100.0, max_cost_per_query_usd=None, max_tokens_per_query=None
    )
    evaluator = CostLatencyEvaluator(slo=slo)
    record = QueryCostLatencyRecord(query="slow query", total_latency_ms=500.0)
    violations = evaluator._check_slo(record)
    assert any("latency" in v for v in violations)


def test_check_slo_flags_cost_violation():
    slo = CostLatencySLO(
        p99_latency_ms=None, max_cost_per_query_usd=0.01, max_tokens_per_query=None
    )
    evaluator = CostLatencyEvaluator(slo=slo)
    record = QueryCostLatencyRecord(query="expensive query", total_cost_usd=0.05)
    violations = evaluator._check_slo(record)
    assert any("cost" in v for v in violations)


def test_check_slo_flags_token_violation():
    slo = CostLatencySLO(p99_latency_ms=None, max_cost_per_query_usd=None, max_tokens_per_query=10)
    evaluator = CostLatencyEvaluator(slo=slo)
    record = QueryCostLatencyRecord(query="big query", prompt_tokens=20)
    violations = evaluator._check_slo(record)
    assert any("tokens" in v for v in violations)


def test_check_slo_disabled_returns_no_violations():
    slo = CostLatencySLO(
        enabled=False, p99_latency_ms=1.0, max_cost_per_query_usd=0.0, max_tokens_per_query=0
    )
    evaluator = CostLatencyEvaluator(slo=slo)
    record = QueryCostLatencyRecord(
        query="anything", total_latency_ms=99999.0, total_cost_usd=99.0, prompt_tokens=99999
    )
    assert evaluator._check_slo(record) == []


def test_check_slo_passes_within_budget():
    slo = CostLatencySLO(
        p99_latency_ms=10000.0, max_cost_per_query_usd=1.0, max_tokens_per_query=10000
    )
    evaluator = CostLatencyEvaluator(slo=slo)
    record = QueryCostLatencyRecord(
        query="ok query", total_latency_ms=100.0, total_cost_usd=0.001, prompt_tokens=10
    )
    assert evaluator._check_slo(record) == []


def test_build_record_computes_cost_breakdown():
    evaluator = CostLatencyEvaluator(slo=CostLatencySLO())
    timer = LatencyTimer()
    with timer.stage("end_to_end"):
        time.sleep(0.001)

    record = evaluator.build_record(
        query="what is x?",
        namespace="default",
        timer=timer,
        embedding_tokens=100,
        rerank_searches=1,
    )

    assert record.query == "what is x?"
    assert "generation" in record.cost_breakdown_usd
    assert "judge" in record.cost_breakdown_usd
    assert "embedding" in record.cost_breakdown_usd
    assert "rerank" in record.cost_breakdown_usd
    assert record.total_cost_usd == pytest.approx(sum(record.cost_breakdown_usd.values()))
