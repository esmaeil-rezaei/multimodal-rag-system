
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (     
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config.settings import get_config, get_secrets
from src.generation.generator import GenerationResult
from src.indexing.vector_store import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)



@dataclass
class RetrievalMetrics:
    """Metrics measuring the quality of the retrieval stage in isolation."""
    recall_at_k: float = 0.0        
    ndcg: float = 0.0              
    mrr: float = 0.0              
    context_precision: float = 0.0 
    context_recall: float = 0.0   


@dataclass
class GenerationMetrics:
    """Metrics measuring the quality of the generation stage given gold context."""
    faithfulness: float = 0.0       
    answer_relevancy: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report for a single query-answer pair."""
    query: str
    answer: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    ragas_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0



class RAGEvaluator:
    """
    Comprehensive evaluation suite for the end-to-end RAG pipeline.

    RAGAS metrics (faithfulness, answer_relevancy, context_precision,
    context_recall) are computed via the official `ragas` library.

    IR metrics (Recall@k, NDCG, MRR) are computed directly and require
    ground-truth relevant chunk IDs.

    Drift detection uses a sliding-window cosine-distance proxy.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._eval_cfg = cfg.evaluation
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)

        judge_model = self._eval_cfg["llm_judge"]["model"]

        # ragas requires LangChain-wrapped clients
        self._ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(model=judge_model, api_key=sec.openai_api_key, temperature=0)
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self._eval_cfg.get("embedding_model", "text-embedding-3-small"),
                api_key=sec.openai_api_key,
            )
        )

        self._reference_embeddings: List[np.ndarray] = []
        self._reference_window = self._eval_cfg["drift_detection"]["reference_window"]


    def evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        context_texts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Run RAGAS metrics on a single query-answer-context triple.
        Without `ground_truth`, only faithfulness and answer_relevancy are computed;
        passing it also enables context_precision and context_recall.
        """
        if not self._eval_cfg["llm_judge"]["enabled"]:
            return EvaluationReport(query=query, answer=answer)

        # ragas maps `reference` to ground_truth; None is valid for faithfulness and answer_relevancy.
        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=context_texts,
            reference=ground_truth,
        )
        dataset = EvaluationDataset(samples=[sample])

        # ragas 0.4.x: reset llm/embeddings to None before each call so that
        # evaluate() injects our llm/embeddings via its own initialisation loop.
        # If they are already set (from a previous call), evaluate() skips them.
        for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
            if hasattr(m, 'llm'):
                m.llm = None
            if hasattr(m, 'embeddings'):
                m.embeddings = None

        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics += [context_precision, context_recall]

        logger.info(f"Running RAGAS evaluation with metrics: {[getattr(m, 'name', type(m).__name__) for m in metrics]}")

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
            raise_exceptions=False,
            show_progress=False,
        )

        # result.scores is a list of per-sample dicts; we have one sample
        scores: Dict[str, float] = dict(result.scores[0]) if result.scores else {}

        report = EvaluationReport(query=query, answer=answer)
        report.ragas_scores = scores
        report.generation.faithfulness = scores.get("faithfulness", 0.0)
        report.generation.answer_relevancy = scores.get("answer_relevancy", 0.0)
        report.retrieval.context_precision = scores.get("context_precision", 0.0)
        report.retrieval.context_recall = scores.get("context_recall", 0.0)

        valid_scores = [v for v in scores.values() if v is not None]
        report.overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        logger.info("RAGAS evaluation complete", extra={"scores": scores})
        return report


    def evaluate_with_llm_judge(
        self,
        query: str,
        answer: str,
        context_texts: List[str],
        generation_result: GenerationResult,
        ground_truth: Optional[str] = None,
    ) -> EvaluationReport:
        """Alias for evaluate_with_ragas (backward-compatible)."""
        return self.evaluate_with_ragas(
            query=query,
            answer=answer,
            context_texts=context_texts,
            ground_truth=ground_truth,
        )


    def evaluate_batch_with_ragas(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[EvaluationReport]:
        """
        Evaluate a batch in a single ragas call.
        Each dict must have `query`, `answer`, `context_texts`, and optionally `ground_truth`.
        Returns one EvaluationReport per sample in the same order.
        """
        ragas_samples = [
            SingleTurnSample(
                user_input=s["query"],
                response=s["answer"],
                retrieved_contexts=s["context_texts"],
                reference=s.get("ground_truth"),
            )
            for s in samples
        ]
        dataset = EvaluationDataset(samples=ragas_samples)

        has_ground_truth = any(s.get("ground_truth") for s in samples)
        metrics = [faithfulness, answer_relevancy]
        if has_ground_truth:
            metrics += [context_precision, context_recall]

        for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
            if hasattr(m, 'llm'):
                m.llm = None
            if hasattr(m, 'embeddings'):
                m.embeddings = None

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
            raise_exceptions=False,
            show_progress=True,
        )

        reports = []
        for i, s in enumerate(samples):
            scores = dict(result.scores[i]) if i < len(result.scores) else {}
            report = EvaluationReport(query=s["query"], answer=s["answer"])
            report.ragas_scores = scores
            report.generation.faithfulness = scores.get("faithfulness", 0.0)
            report.generation.answer_relevancy = scores.get("answer_relevancy", 0.0)
            report.retrieval.context_precision = scores.get("context_precision", 0.0)
            report.retrieval.context_recall = scores.get("context_recall", 0.0)
            valid = [v for v in scores.values() if v is not None]
            report.overall_score = sum(valid) / len(valid) if valid else 0.0
            reports.append(report)

        logger.info("RAGAS evaluation complete", extra={"scores": scores})
        logger.info(
            f"\n{'='*50}\n"
            f"  RAGAS EVALUATION RESULTS\n"
            f"  Faithfulness:      {report.generation.faithfulness:.3f}\n"
            f"  Answer Relevancy:  {report.generation.answer_relevancy:.3f}\n"
            f"  Context Precision: {report.retrieval.context_precision:.3f}\n"
            f"  Context Recall:    {report.retrieval.context_recall:.3f}\n"
            f"  Overall Score:     {report.overall_score:.3f}\n"
            f"{'='*50}"
        )

        return reports


    def generate_synthetic_qa(self, chunk_text: str, n: int = 3) -> List[Dict[str, str]]:
        """
        Generate `n` question-answer pairs from `chunk_text` using the configured generator model.
        Returns a list of {"question": ..., "answer": ...} dicts, or an empty list on failure.
        """
        prompt = (
            f"Generate {n} diverse, specific question-answer pairs from the passage below. "
            "Answers must be fully grounded in the passage text. "
            "Return ONLY valid JSON: [{\"question\": \"...\", \"answer\": \"...\"}, ...]\n\n"
            f"Passage: {chunk_text[:1000]}"
        )
        try:
            response = self._openai.chat.completions.create(
                model=self._eval_cfg["synthetic_qa"]["generator_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600,
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)[:n]
        except Exception as exc:
            logger.warning(f"Synthetic QA generation failed: {exc}")
            return []


    def evaluate_retrieval(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10,
    ) -> RetrievalMetrics:
        """
        Compute Recall@k, NDCG, and MRR given ordered retrieved IDs and ground-truth relevant IDs.

        Args:
            retrieved_ids: Ordered chunk IDs (rank 1 = index 0).
            relevant_ids:  Ground-truth relevant chunk IDs.
            k:             Cutoff rank.
        """
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)

        hits = sum(1 for cid in retrieved_k if cid in relevant_set)
        recall_at_k = hits / len(relevant_set) if relevant_set else 0.0

        mrr = 0.0
        for rank, cid in enumerate(retrieved_k, start=1):
            if cid in relevant_set:
                mrr = 1.0 / rank
                break

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, cid in enumerate(retrieved_k, start=1)
            if cid in relevant_set
        )
        ideal_positions = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_positions + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics = RetrievalMetrics(recall_at_k=recall_at_k, ndcg=ndcg, mrr=mrr)
        logger.info(
            f"Retrieval metrics: recall@{k}={recall_at_k:.3f} NDCG={ndcg:.3f} MRR={mrr:.3f}"
        )
        return metrics



    def update_reference_distribution(self, query_embedding: np.ndarray) -> None:
        """Add a query embedding to the sliding FIFO reference window."""
        self._reference_embeddings.append(query_embedding)
        if len(self._reference_embeddings) > self._reference_window:
            self._reference_embeddings.pop(0)

    def detect_drift(self, recent_embeddings: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Compare recent query embeddings to the reference distribution.

        Args:
            recent_embeddings: Embeddings from the most recent query batch.

        Returns:
            (drift_detected: bool, divergence_score: float)
        """
        drift_cfg = self._eval_cfg["drift_detection"]
        threshold = drift_cfg["drift_threshold"]

        if not drift_cfg["enabled"] or len(self._reference_embeddings) < 50:
            return False, 0.0

        ref_matrix = np.stack(self._reference_embeddings)
        cur_matrix = np.stack(recent_embeddings)

        ref_proj = ref_matrix[:, :32].mean(axis=0)
        cur_proj = cur_matrix[:, :32].mean(axis=0)

        ref_norm = ref_proj / (np.linalg.norm(ref_proj) + 1e-8)
        cur_norm = cur_proj / (np.linalg.norm(cur_proj) + 1e-8)
        divergence = float(np.linalg.norm(ref_norm - cur_norm))

        drift_detected = divergence > threshold
        if drift_detected:
            logger.warning(
                f"Query drift detected: divergence={divergence:.3f} > threshold={threshold}",
                extra={"divergence": divergence, "threshold": threshold},
            )
        else:
            logger.debug(f"No drift detected: divergence={divergence:.3f}")

        return drift_detected, divergence