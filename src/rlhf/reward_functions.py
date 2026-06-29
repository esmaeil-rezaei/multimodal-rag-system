"""Composite reward signal for PPO fine-tuning: learned reward, citation, faithfulness, length."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RewardBreakdown:
    """Component scores that make up a composite reward, for logging/debugging."""

    learned_reward: float
    citation_score: float
    faithfulness_score: float
    length_penalty: float
    total: float
    weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "learned_reward": self.learned_reward,
            "citation_score": self.citation_score,
            "faithfulness_score": self.faithfulness_score,
            "length_penalty": self.length_penalty,
            "total": self.total,
            "weights": self.weights,
        }


def score_with_reward_model(prompt: str, response: str, model, tokenizer) -> float:
    """
    Score a single `(prompt, response)` pair with a trained reward model.

    Returns the raw scalar logit, squashed to `[-1, 1]` via `tanh` so it
    can be combined with the bounded heuristic scores below.
    """
    import torch

    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)

    return float(torch.tanh(logits).item())


def citation_score(response: str, pattern: str = r"\[\d+\]") -> float:
    """Return 1.0 if the response contains at least one inline citation marker, else 0.0."""
    return 1.0 if re.search(pattern, response) else 0.0


def _tokenize_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def faithfulness_score(response: str, context: str | None) -> float:
    """Fraction of the response's content words that also appear in the retrieved context."""
    if not context:
        return 1.0

    response_words = _tokenize_words(response)
    if not response_words:
        return 0.0

    context_words = _tokenize_words(context)
    overlap = response_words & context_words
    return len(overlap) / len(response_words)


def length_penalty(
    response: str,
    target_length_tokens: int = 256,
    max_length_tokens: int = 512,
) -> float:
    """Score 1.0 at target length, decaying linearly toward 0.0 for shorter or longer responses."""
    n_tokens = len(response.split())

    if n_tokens <= 0:
        return 0.0
    if n_tokens <= target_length_tokens:
        return n_tokens / target_length_tokens
    if n_tokens >= max_length_tokens:
        return 0.0

    span = max_length_tokens - target_length_tokens
    return 1.0 - (n_tokens - target_length_tokens) / span


def composite_reward(
    prompt: str,
    response: str,
    reward_model=None,
    reward_tokenizer=None,
    context: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> RewardBreakdown:
    """Compute the weighted composite reward for a (prompt, response) rollout."""
    rf_cfg = cfg or get_config().rlhf["reward_function"]
    weights = dict(rf_cfg["weights"])

    learned = 0.0
    if reward_model is not None and reward_tokenizer is not None:
        learned = (
            score_with_reward_model(prompt, response, reward_model, reward_tokenizer) + 1.0
        ) / 2.0
    else:
        dropped = weights.pop("learned_reward", 0.0)
        remaining = sum(weights.values()) or 1.0
        for key in weights:
            weights[key] += dropped * (weights[key] / remaining)
        weights["learned_reward"] = 0.0

    citation = citation_score(response, rf_cfg.get("citation_pattern", r"\[\d+\]"))
    faithfulness = faithfulness_score(response, context)
    length = length_penalty(
        response,
        target_length_tokens=rf_cfg.get("target_length_tokens", 256),
        max_length_tokens=rf_cfg.get("max_length_tokens", 512),
    )

    total = (
        weights.get("learned_reward", 0.0) * learned
        + weights.get("citation_score", 0.0) * citation
        + weights.get("faithfulness", 0.0) * faithfulness
        + weights.get("length_penalty", 0.0) * length
    )

    return RewardBreakdown(
        learned_reward=learned,
        citation_score=citation,
        faithfulness_score=faithfulness,
        length_penalty=length,
        total=total,
        weights=weights,
    )
