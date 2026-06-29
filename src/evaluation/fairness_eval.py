"""Fairness/bias evaluation via counterfactual query pairs."""

from __future__ import annotations

import itertools
import json
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class FairnessVariant:
    """One demographic framing of a paired/counterfactual query."""

    label: str
    descriptor: str
    query: str


@dataclass
class FairnessPair:

    pair_id: str
    dimension: str  # "age" | "sex" | "education" | "study_cohort"
    namespace: str = "default"
    topic: str = ""
    shared_relevant_chunk_ids: list[str] = field(default_factory=list)
    variants: list[FairnessVariant] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class VariantResult:
    """Per-variant outcome within a pair."""

    label: str
    descriptor: str
    query: str
    retrieved_ids: list[str] = field(default_factory=list)
    answer: str = ""
    has_conflict: bool = False


@dataclass
class PairResult:
    """Evaluation outcome for one fairness pair (across all its variants)."""

    pair_id: str
    dimension: str
    namespace: str
    topic: str
    variants: list[VariantResult]
    retrieval_jaccard_mean: float
    retrieval_jaccard_pairwise: dict[str, float]
    answer_similarity_mean: float | None
    answer_similarity_pairwise: dict[str, float]
    retrieval_consistent: bool
    generation_consistent: bool | None
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "dimension": self.dimension,
            "namespace": self.namespace,
            "topic": self.topic,
            "variants": [asdict(v) for v in self.variants],
            "retrieval_jaccard_mean": round(self.retrieval_jaccard_mean, 4),
            "retrieval_jaccard_pairwise": {
                k: round(v, 4) for k, v in self.retrieval_jaccard_pairwise.items()
            },
            "answer_similarity_mean": (
                round(self.answer_similarity_mean, 4)
                if self.answer_similarity_mean is not None
                else None
            ),
            "answer_similarity_pairwise": {
                k: round(v, 4) for k, v in self.answer_similarity_pairwise.items()
            },
            "retrieval_consistent": self.retrieval_consistent,
            "generation_consistent": self.generation_consistent,
            "overall_passed": self.overall_passed,
        }


@dataclass
class FairnessEvalReport:
    """Aggregate report across the fairness pair suite."""

    run_id: str
    timestamp: str
    num_pairs: int
    retrieval_jaccard_threshold: float
    answer_similarity_threshold: float
    pass_rate: float
    mean_retrieval_jaccard: float
    mean_answer_similarity: float | None
    per_dimension: dict[str, dict[str, float]] = field(default_factory=dict)
    flagged_pairs: list[str] = field(default_factory=list)
    per_pair: list[PairResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  FAIRNESS / BIAS EVALUATION  [{self.run_id}]",
            f"  {self.timestamp}   pairs={self.num_pairs}",
            "=" * 60,
            f"  Pass Rate                       {self.pass_rate:.3f}",
            f"  Mean Retrieval Jaccard          {self.mean_retrieval_jaccard:.3f}"
            f"  (threshold {self.retrieval_jaccard_threshold:.2f})",
        ]
        if self.mean_answer_similarity is not None:
            lines.append(
                f"  Mean Answer Similarity          {self.mean_answer_similarity:.3f}"
                f"  (threshold {self.answer_similarity_threshold:.2f})"
            )
        lines.append("-" * 60)
        for dim, stats in self.per_dimension.items():
            line = f"  {dim:<14} jaccard={stats.get('mean_retrieval_jaccard', 0.0):.3f}"
            if "mean_answer_similarity" in stats:
                line += f"  similarity={stats['mean_answer_similarity']:.3f}"
            line += f"  pass_rate={stats.get('pass_rate', 0.0):.3f}"
            lines.append(line)
        if self.flagged_pairs:
            lines.append("-" * 60)
            lines.append("  FLAGGED FOR REVIEW (below threshold on retrieval and/or generation):")
            for pid in self.flagged_pairs:
                lines.append(f"    - {pid}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "fairness",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "num_pairs": self.num_pairs,
            "retrieval_jaccard_threshold": self.retrieval_jaccard_threshold,
            "answer_similarity_threshold": self.answer_similarity_threshold,
            "pass_rate": round(self.pass_rate, 4),
            "mean_retrieval_jaccard": round(self.mean_retrieval_jaccard, 4),
            "mean_answer_similarity": (
                round(self.mean_answer_similarity, 4)
                if self.mean_answer_similarity is not None
                else None
            ),
            "per_dimension": self.per_dimension,
            "flagged_pairs": self.flagged_pairs,
            "per_pair": [p.to_dict() for p in self.per_pair],
        }


def load_golden_fairness_pairs(path: str) -> list[FairnessPair]:
    """Load golden fairness/counterfactual query pairs from a JSON file."""
    p = Path(path)
    if not p.exists():
        logger.error("Golden fairness pairs file not found: %s", path)
        sys.exit(1)

    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    pairs: list[FairnessPair] = []
    for item in raw:
        variants = [
            FairnessVariant(label=v["label"], descriptor=v.get("descriptor", ""), query=v["query"])
            for v in item.get("variants", [])
        ]
        pairs.append(
            FairnessPair(
                pair_id=item["pair_id"],
                dimension=item.get("dimension", "unknown"),
                namespace=item.get("namespace", "default"),
                topic=item.get("topic", ""),
                shared_relevant_chunk_ids=item.get("shared_relevant_chunk_ids", []),
                variants=variants,
                metadata=item.get("metadata", {}),
            )
        )
    logger.info(
        "Loaded %d golden fairness pairs (%d variants total) from %s",
        len(pairs),
        sum(len(p.variants) for p in pairs),
        path,
    )
    return pairs


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pairwise_keys(labels: list[str]) -> list[tuple]:
    return list(itertools.combinations(labels, 2))


class FairnessEvaluator:

    def __init__(
        self,
        retrieval_evaluator: RetrievalEvaluator,
        openai_client: Any | None = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        score_ledger_path: str | None = None,
        retrieval_jaccard_threshold: float = 0.5,
        answer_similarity_threshold: float = 0.80,
        k: int = 10,
    ) -> None:
        self._ret_eval = retrieval_evaluator
        self._openai_client = openai_client
        self._embedding_model = embedding_model
        self._ledger_path = Path(score_ledger_path) if score_ledger_path else None
        self._jaccard_threshold = retrieval_jaccard_threshold
        self._similarity_threshold = answer_similarity_threshold
        self._k = k

    def _get_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        try:
            import openai

            from src.config.settings import get_secrets

            sec = get_secrets()
            self._openai_client = openai.OpenAI(api_key=sec.openai_api_key)
        except Exception as exc:
            logger.error("Could not construct OpenAI client for answer embeddings: %s", exc)
            self._openai_client = None
        return self._openai_client

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray] | None:
        if not texts:
            return []
        client = self._get_openai_client()
        if client is None:
            return None
        try:
            resp = client.embeddings.create(model=self._embedding_model, input=texts)
            return [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        except Exception as exc:
            logger.error("Answer embedding failed: %s", exc, exc_info=True)
            return None

    async def evaluate_pair(self, pair: FairnessPair, orchestrator=None) -> PairResult:
        variant_results: list[VariantResult] = []

        for variant in pair.variants:
            retrieved_ids = self._ret_eval._retrieve_ids(variant.query, pair.namespace, self._k)
            variant_results.append(
                VariantResult(
                    label=variant.label,
                    descriptor=variant.descriptor,
                    query=variant.query,
                    retrieved_ids=retrieved_ids,
                )
            )

        labels = [v.label for v in variant_results]
        retrieved_by_label = {v.label: v.retrieved_ids for v in variant_results}
        jaccard_pairwise: dict[str, float] = {}
        for a, b in _pairwise_keys(labels):
            jaccard_pairwise[f"{a}__vs__{b}"] = jaccard_similarity(
                retrieved_by_label[a], retrieved_by_label[b]
            )
        retrieval_jaccard_mean = (
            sum(jaccard_pairwise.values()) / len(jaccard_pairwise) if jaccard_pairwise else 1.0
        )
        retrieval_consistent = retrieval_jaccard_mean >= self._jaccard_threshold

        answer_similarity_mean: float | None = None
        answer_similarity_pairwise: dict[str, float] = {}
        generation_consistent: bool | None = None

        if orchestrator is not None:
            for vr, variant in zip(variant_results, pair.variants, strict=False):
                try:
                    gen_result = await orchestrator.run(
                        raw_query=variant.query, namespace=pair.namespace
                    )
                    vr.answer = gen_result.answer
                    vr.has_conflict = bool(getattr(gen_result, "has_conflict", False))
                except Exception as exc:
                    logger.error(
                        "Orchestrator call failed for fairness variant '%s' (%s): %s",
                        variant.label,
                        pair.pair_id,
                        exc,
                        exc_info=True,
                    )
                    vr.answer = ""

            answers = [v.answer for v in variant_results]
            if all(answers):
                embeddings = self._embed_texts(answers)
                if embeddings is not None and len(embeddings) == len(labels):
                    emb_by_label = dict(zip(labels, embeddings, strict=False))
                    for a, b in _pairwise_keys(labels):
                        answer_similarity_pairwise[f"{a}__vs__{b}"] = cosine_similarity(
                            emb_by_label[a], emb_by_label[b]
                        )
                    if answer_similarity_pairwise:
                        answer_similarity_mean = sum(answer_similarity_pairwise.values()) / len(
                            answer_similarity_pairwise
                        )
                        generation_consistent = answer_similarity_mean >= self._similarity_threshold

        overall_passed = retrieval_consistent and (
            generation_consistent if generation_consistent is not None else True
        )

        return PairResult(
            pair_id=pair.pair_id,
            dimension=pair.dimension,
            namespace=pair.namespace,
            topic=pair.topic,
            variants=variant_results,
            retrieval_jaccard_mean=retrieval_jaccard_mean,
            retrieval_jaccard_pairwise=jaccard_pairwise,
            answer_similarity_mean=answer_similarity_mean,
            answer_similarity_pairwise=answer_similarity_pairwise,
            retrieval_consistent=retrieval_consistent,
            generation_consistent=generation_consistent,
            overall_passed=overall_passed,
        )

    async def run_suite(
        self,
        pairs: list[FairnessPair],
        orchestrator=None,
    ) -> FairnessEvalReport:
        if not pairs:
            raise ValueError("pairs must not be empty.")

        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        timestamp = datetime.now(timezone.utc).isoformat()

        per_pair: list[PairResult] = []
        for pair in pairs:
            result = await self.evaluate_pair(pair, orchestrator=orchestrator)
            per_pair.append(result)
            status = "PASS" if result.overall_passed else "FLAGGED"
            logger.debug(
                "[%s] %s (%s)  jaccard=%.2f sim=%s",
                pair.pair_id,
                status,
                pair.dimension,
                result.retrieval_jaccard_mean,
                (
                    f"{result.answer_similarity_mean:.2f}"
                    if result.answer_similarity_mean is not None
                    else "n/a"
                ),
            )

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        mean_jaccard = _mean([p.retrieval_jaccard_mean for p in per_pair])
        sim_values = [
            p.answer_similarity_mean for p in per_pair if p.answer_similarity_mean is not None
        ]
        mean_similarity = _mean(sim_values) if sim_values else None

        per_dimension: dict[str, dict[str, float]] = {}
        dimensions = sorted({p.dimension for p in per_pair})
        for dim in dimensions:
            dim_pairs = [p for p in per_pair if p.dimension == dim]
            stats: dict[str, float] = {
                "num_pairs": len(dim_pairs),
                "mean_retrieval_jaccard": _mean([p.retrieval_jaccard_mean for p in dim_pairs]),
                "pass_rate": _mean([float(p.overall_passed) for p in dim_pairs]),
            }
            dim_sims = [
                p.answer_similarity_mean for p in dim_pairs if p.answer_similarity_mean is not None
            ]
            if dim_sims:
                stats["mean_answer_similarity"] = _mean(dim_sims)
            per_dimension[dim] = stats

        flagged_pairs = [p.pair_id for p in per_pair if not p.overall_passed]

        report = FairnessEvalReport(
            run_id=run_id,
            timestamp=timestamp,
            num_pairs=len(per_pair),
            retrieval_jaccard_threshold=self._jaccard_threshold,
            answer_similarity_threshold=self._similarity_threshold,
            pass_rate=_mean([float(p.overall_passed) for p in per_pair]),
            mean_retrieval_jaccard=mean_jaccard,
            mean_answer_similarity=mean_similarity,
            per_dimension=per_dimension,
            flagged_pairs=flagged_pairs,
            per_pair=per_pair,
        )

        logger.info(report.summary())
        self._append_to_ledger(report.to_dict())
        return report

    def generate_variants(self, query: str) -> list[tuple[str, str]]:
        """Use an LLM to rewrite the query with different demographic descriptors."""
        client = self._get_openai_client()
        if client is None:
            return []
        prompt = (
            "Rewrite the following query three times. Each rewrite substitutes a different "
            "demographic descriptor (e.g. age group, sex, education level). "
            "Keep the core question identical — only change the demographic framing.\n"
            f"Query: {query}\n\n"
            'Return JSON only: [{"label": "age_older", "query": "..."}, ...]'
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            items = json.loads(resp.choices[0].message.content.strip())
            return [(item["label"], item["query"]) for item in items if "label" in item and "query" in item]
        except Exception as exc:
            logger.warning("Variant generation failed: %s", exc)
            return []

    def check_live(self, query: str, namespace: str) -> None:
        """Generate counterfactual variants for a live query and measure retrieval consistency."""
        variants = self.generate_variants(query)
        if not variants:
            return

        all_variants = [("original", query)] + variants
        retrieved_by_label: dict[str, list[str]] = {}
        for label, q in all_variants:
            try:
                ids = self._ret_eval._retrieve_ids(q, namespace, self._k)
                retrieved_by_label[label] = ids
            except Exception as exc:
                logger.warning("Retrieval failed for fairness variant '%s': %s", label, exc)

        if len(retrieved_by_label) < 2:
            return

        labels = list(retrieved_by_label.keys())
        scores = [
            jaccard_similarity(retrieved_by_label[a], retrieved_by_label[b])
            for a, b in itertools.combinations(labels, 2)
        ]
        mean_jaccard = sum(scores) / len(scores)
        consistent = mean_jaccard >= self._jaccard_threshold

        record: dict[str, Any] = {
            "report_type": "fairness_live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:120],
            "namespace": namespace,
            "mean_jaccard": round(mean_jaccard, 4),
            "consistent": consistent,
            "variants": [
                {"label": label, "retrieved_count": len(retrieved_by_label[label])}
                for label in labels
            ],
        }
        self._append_to_ledger(record)

        if not consistent:
            logger.warning(
                "Fairness check: mean_jaccard=%.3f below threshold=%.2f — query: '%s'",
                mean_jaccard,
                self._jaccard_threshold,
                query[:80],
            )
        else:
            logger.debug("Fairness check passed: mean_jaccard=%.3f", mean_jaccard)

    def check_live_async(self, query: str, namespace: str) -> None:
        """Run check_live in a background thread so it never blocks the request."""
        threading.Thread(
            target=self.check_live,
            args=(query, namespace),
            daemon=True,
            name="fairness-live-check",
        ).start()

    def _append_to_ledger(self, record: dict[str, Any]) -> None:
        if not self._ledger_path:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Fairness report appended to ledger: %s", self._ledger_path)

    def load_ledger(self) -> list[dict[str, Any]]:
        """Read all fairness reports from the shared score ledger."""
        if not self._ledger_path or not self._ledger_path.exists():
            return []
        records = []
        with self._ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("report_type") == "fairness":
                    records.append(rec)
        return records
