#!/usr/bin/env python3
"""CLI entry point for the offline seven-layer RAG evaluation suite. See docs/evaluation.md."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import openai

from src.config.settings import get_config, get_secrets
from src.core.container import init_container
from src.evaluation.cost_latency_eval import CostLatencyEvaluator
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.fairness_eval import FairnessEvaluator, load_golden_fairness_pairs
from src.evaluation.indexing_eval import IndexingEvaluator
from src.evaluation.multiturn_eval import MultiTurnEvaluator, load_golden_conversations
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.evaluation.system_eval import GoldenQuery, SystemEvaluator
from src.utils.logger import get_logger

logger = get_logger("evaluate_offline")


def load_golden_queries(path: str) -> list[GoldenQuery]:
    """
    Load golden queries from a JSON file.

    Expected format:
    [
      {
        "query": "What is the recommended dosage for ...",
        "relevant_chunk_ids": ["chunk_001", "chunk_042"],
        "expected_answer_keywords": ["dosage", "mg", "twice daily"],
        "namespace": "default",
        "metadata": {"category": "treatment"}
      },
      ...
    ]
    """
    p = Path(path)
    if not p.exists():
        logger.error("Golden queries file not found: %s", path)
        sys.exit(1)

    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    queries = [
        GoldenQuery(
            query=item["query"],
            relevant_chunk_ids=item.get("relevant_chunk_ids", []),
            expected_answer_keywords=item.get("expected_answer_keywords", []),
            namespace=item.get("namespace", "default"),
            metadata=item.get("metadata", {}),
        )
        for item in raw
    ]
    logger.info("Loaded %d golden queries from %s", len(queries), path)
    return queries


def make_answer_fn(container):
    """
    Wrap the OrchestratorAgent to produce (answer, context_texts) tuples.

    This is passed to SystemEvaluator so Layer 4 can evaluate end-to-end
    generation quality, not just retrieval.
    """

    def answer_fn(query: str, namespace: str):
        response = container.orchestrator.run(query=query, namespace=namespace)
        answer = response.get("answer", "")
        context_texts = [c.chunk.text for c in response.get("context_items", [])]
        return answer, context_texts

    return answer_fn


def run_layer1(
    container,
    openai_client: openai.OpenAI,
    args: argparse.Namespace,
) -> dict:
    """Layer 1: Indexing quality."""
    logger.info("=== Layer 1: Indexing Quality ===")

    evaluator = IndexingEvaluator(
        embedder=container.embedder,
        openai_client=openai_client,
        judge_model=get_config().evaluation["llm_judge"]["model"],
        coherence_threshold=0.6,
    )

    # Fetch a sample of chunks from the dense store for coherence scoring
    try:
        sample_chunks = container.dense_store.sample(n=args.sample, namespace=args.namespace)
    except Exception:
        logger.warning(
            "dense_store.sample() not available — skipping indexing eval. "
            "Implement AppContainer.dense_store.sample() to enable this layer."
        )
        return {"layer1": "skipped — dense_store.sample() not implemented"}

    report = evaluator.evaluate(
        chunks=sample_chunks,
        coherence_sample_size=args.sample,
    )
    return {"layer1": report.to_dict()}


def run_layer2(
    container,
    golden_queries: list[GoldenQuery],
    args: argparse.Namespace,
) -> dict:
    """Layer 2: Retrieval quality."""
    logger.info("=== Layer 2: Retrieval Quality ===")

    evaluator = RetrievalEvaluator(
        embedder=container.embedder,
        retriever=container.retriever,
        request_delay=args.request_delay,
    )

    labeled = [gq.to_labeled_query() for gq in golden_queries]
    report = evaluator.evaluate(labeled, k=args.k)
    print(report.summary())
    return {"layer2": report.to_dict()}


def run_layer4(
    container,
    rag_evaluator: RAGEvaluator,
    golden_queries: list[GoldenQuery],
    args: argparse.Namespace,
) -> dict:
    """Layer 4: System-level golden regression suite."""
    logger.info("=== Layer 4: System-Level Evaluation ===")

    ret_evaluator = RetrievalEvaluator(
        embedder=container.embedder,
        retriever=container.retriever,
    )

    sys_evaluator = SystemEvaluator(
        retrieval_evaluator=ret_evaluator,
        rag_evaluator=rag_evaluator,
        answer_fn=make_answer_fn(container) if args.answer else None,
        score_ledger_path=args.ledger,
        k=args.k,
    )

    report = sys_evaluator.run_golden_suite(
        golden_queries,
        run_baseline=not args.no_baseline,
    )
    print(report.summary())
    sys_evaluator.print_trend(metric="pass_rate")
    return {"layer4": report.to_dict()}


async def run_layer5(
    container,
    golden_queries: list[GoldenQuery],
    args: argparse.Namespace,
) -> dict:
    """Layer 5: Cost & latency SLO tracking."""
    logger.info("=== Layer 5: Cost & Latency SLO Tracking ===")

    evaluator = CostLatencyEvaluator(score_ledger_path=args.ledger)
    report = await evaluator.run_suite(container.orchestrator, golden_queries)
    print(report.summary())
    evaluator.print_trend(metric="p95")
    return {"layer5": report.to_dict()}


async def run_layer6(
    container,
    rag_evaluator: RAGEvaluator,
    args: argparse.Namespace,
) -> dict:
    """Layer 6: Multi-turn conversational evaluation."""
    logger.info("=== Layer 6: Multi-Turn Conversational Evaluation ===")

    conversations = load_golden_conversations(args.conversations)
    if not conversations:
        logger.error("Golden conversations file is empty. Exiting.")
        sys.exit(1)

    mt_cfg = get_config().evaluation.get("multi_turn", {})

    ret_evaluator = RetrievalEvaluator(
        embedder=container.embedder,
        retriever=container.retriever,
        request_delay=args.request_delay,
    )

    evaluator = MultiTurnEvaluator(
        query_understanding=container.query_understanding,
        retrieval_evaluator=ret_evaluator,
        rag_evaluator=rag_evaluator if args.answer else None,
        score_ledger_path=args.ledger,
        keyword_pass_threshold=mt_cfg.get("keyword_pass_threshold", 0.6),
        k=args.k,
    )

    report = await evaluator.run_suite(
        conversations,
        orchestrator=container.orchestrator if args.answer else None,
    )
    print(report.summary())
    return {"layer6": report.to_dict()}


async def run_layer7(
    container,
    args: argparse.Namespace,
) -> dict:
    """Layer 7: Fairness / bias evaluation."""
    logger.info("=== Layer 7: Fairness / Bias Evaluation ===")

    pairs = load_golden_fairness_pairs(args.fairness_pairs)
    if not pairs:
        logger.error("Golden fairness pairs file is empty. Exiting.")
        sys.exit(1)

    fairness_cfg = get_config().evaluation.get("fairness", {})

    ret_evaluator = RetrievalEvaluator(
        embedder=container.embedder,
        retriever=container.retriever,
        request_delay=args.request_delay,
    )

    evaluator = FairnessEvaluator(
        retrieval_evaluator=ret_evaluator,
        embedding_model=fairness_cfg.get("embedding_model", "text-embedding-3-small"),
        score_ledger_path=args.ledger,
        retrieval_jaccard_threshold=fairness_cfg.get("retrieval_jaccard_threshold", 0.5),
        answer_similarity_threshold=fairness_cfg.get("answer_similarity_threshold", 0.80),
        k=args.k,
    )

    report = await evaluator.run_suite(
        pairs,
        orchestrator=container.orchestrator if args.answer else None,
    )
    print(report.summary())
    return {"layer7": report.to_dict()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline four-layer RAG evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Layer selection
    p.add_argument("--layer1", action="store_true", help="Run Layer 1: indexing quality")
    p.add_argument("--layer2", action="store_true", help="Run Layer 2: retrieval quality")
    p.add_argument("--layer4", action="store_true", help="Run Layer 4: system-level golden suite")
    p.add_argument("--layer5", action="store_true", help="Run Layer 5: cost & latency SLO tracking")
    p.add_argument(
        "--layer6", action="store_true", help="Run Layer 6: multi-turn conversational evaluation"
    )
    p.add_argument("--layer7", action="store_true", help="Run Layer 7: fairness / bias evaluation")

    # Input
    p.add_argument(
        "--golden",
        default="tests/golden_queries.json",
        metavar="FILE",
        help="Path to golden queries JSON (default: tests/golden_queries.json)",
    )
    p.add_argument(
        "--conversations",
        default="tests/golden_conversations.json",
        metavar="FILE",
        help="Path to golden multi-turn conversations JSON for Layer 6 "
        "(default: tests/golden_conversations.json)",
    )
    p.add_argument(
        "--fairness-pairs",
        default="tests/golden_fairness_pairs.json",
        metavar="FILE",
        help="Path to golden fairness/counterfactual query pairs JSON for Layer 7 "
        "(default: tests/golden_fairness_pairs.json)",
    )

    # Tuning knobs
    p.add_argument("--k", type=int, default=10, help="Top-K cutoff for retrieval metrics")
    p.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Delay before each retrieval call, to avoid third-party rate limits "
        "(e.g. 6.5 for a Cohere Trial key's 10 calls/minute limit)",
    )
    p.add_argument("--sample", type=int, default=50, help="Chunks sampled for coherence scoring")
    p.add_argument("--namespace", default="default", help="Vector store namespace to query")
    p.add_argument(
        "--answer",
        action="store_true",
        help="Call the full pipeline to generate answers (enables generation metrics in Layer 4, "
        "and full-pipeline answer checks in Layers 6-7)",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison in Layer 4",
    )

    # Output
    p.add_argument(
        "--ledger",
        default=None,
        metavar="FILE",
        help="JSON-lines file to append scores for trend tracking",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write combined JSON report to this path",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Default: run the original three-layer suite if no layer flag is given.
    # Layers 5-7 are opt-in and never part of the implicit default run.
    any_layer_selected = (
        args.layer1 or args.layer2 or args.layer4 or args.layer5 or args.layer6 or args.layer7
    )
    run_all = not any_layer_selected

    # Bootstrap
    sec = get_secrets()
    openai_client = openai.OpenAI(api_key=sec.openai_api_key)
    container = init_container()
    rag_evaluator = container.evaluator

    # Load golden queries (needed for layers 2, 4, and 5)
    golden_queries: list[GoldenQuery] | None = None
    if args.layer2 or args.layer4 or args.layer5 or run_all:
        golden_queries = load_golden_queries(args.golden)
        if not golden_queries:
            logger.error("Golden queries file is empty. Exiting.")
            sys.exit(1)

    combined_report: dict = {}

    if args.layer1 or run_all:
        combined_report.update(run_layer1(container, openai_client, args))

    if args.layer2 or run_all:
        combined_report.update(run_layer2(container, golden_queries, args))

    if args.layer4 or run_all:
        combined_report.update(run_layer4(container, rag_evaluator, golden_queries, args))

    if args.layer5:
        combined_report.update(asyncio.run(run_layer5(container, golden_queries, args)))

    if args.layer6:
        combined_report.update(asyncio.run(run_layer6(container, rag_evaluator, args)))

    if args.layer7:
        combined_report.update(asyncio.run(run_layer7(container, args)))

    # Persist combined report
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(combined_report, fh, indent=2)
        logger.info("Combined report written to %s", args.output)
    else:
        print(json.dumps(combined_report, indent=2))


if __name__ == "__main__":
    main()
