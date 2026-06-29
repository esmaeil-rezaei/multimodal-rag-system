# RLHF Fine-Tuning

Three training pipelines are available. All require preference pairs built first.

## Step 0 — Build datasets

```bash
python scripts/rlhf/build_preference_dataset.py
# or with explicit paths:
python scripts/rlhf/build_preference_dataset.py --feedback-dir logs/feedback
```

Produces:
- `data/rlhf/preference_pairs.jsonl` — `(prompt, chosen, rejected)` pairs from operational feedback logs; used by reward model training and DPO.
- `data/rlhf/ppo_prompts.jsonl` — prompt pool for PPO rollouts, drawn from feedback logs and `tests/golden_queries.json`. Context is omitted (`null`) so the policy fetches fresh context live during rollouts.

## Step 1a — Train reward model (PPO path)

```bash
python scripts/rlhf/train_reward_model.py
python scripts/rlhf/train_reward_model.py --preference-pairs data/rlhf/preference_pairs.jsonl
```

Trains a LoRA-wrapped sequence-classification reward model via TRL `RewardTrainer`. Requires `transformers`, `trl`, `peft`, `datasets`, `torch`.

## Step 1b — PPO fine-tuning

```bash
python scripts/rlhf/train_ppo.py
python scripts/rlhf/train_ppo.py --ppo-prompts data/rlhf/ppo_prompts.jsonl
```

Runs PPO against the trained reward model. Requires a trained reward model at the path configured under `rlhf.reward_model.output_dir`.

## Alternative — DPO fine-tuning

```bash
python scripts/rlhf/train_dpo.py
python scripts/rlhf/train_dpo.py --preference-pairs data/rlhf/preference_pairs.jsonl
```

Cheaper alternative to the reward model + PPO pipeline. Optimizes the policy directly against preference pairs in a single training run — no separate reward model or rollout loop.
