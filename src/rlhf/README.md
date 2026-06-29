# src/rlhf — Reinforcement Learning from Human Feedback

This package fine-tunes a local open-weight policy model using feedback collected from the production RAG pipeline. The production generation layer calls hosted APIs (OpenAI / Cohere) which cannot be fine-tuned directly. RLHF targets the configured `policy_model` (default: `Qwen/Qwen2.5-1.5B-Instruct`), which can later be swapped in as a self-hosted generation backend by pointing `generation.model` in `config/config.yaml` to the local checkpoint.

---

## Files

### `feedback_dataset.py`

Converts raw thumbs-up / thumbs-down feedback logs into preference pairs.

- Reads `logs/feedback/liked_responses/` (chosen) and `logs/feedback/disliked_responses/` (rejected)
- Each record pairs the same prompt with its liked and disliked responses: `{prompt, chosen, rejected}`
- Writes the dataset to `data/rlhf/preference_pairs.jsonl`
- Also writes `data/rlhf/ppo_prompts.jsonl` — raw `{prompt, context}` records for PPO rollouts

```bash
python -m src.rlhf.feedback_dataset
```

---

### `reward_model.py`

Trains a scalar reward model on preference pairs.

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` (small, fast to train)
- LoRA-wrapped (rank 16, alpha 32, dropout 0.05) via `peft`
- Trained with TRL `RewardTrainer` on `(prompt, chosen, rejected)` triples
- Outputs to `models/reward_model/`

```bash
python -m src.rlhf.reward_model
# optional: override dataset path
python -m src.rlhf.reward_model --pairs data/rlhf/preference_pairs.jsonl
```

---

### `reward_functions.py`

Defines the composite reward signal used during PPO rollouts. The reward is a weighted sum:

| Component | Weight | What it measures |
|---|---|---|
| Learned reward model score | 0.70 | Alignment with human preferences |
| Citation score | 0.15 | Presence of `[N]` citation markers in the response |
| Faithfulness score | 0.10 | RAGAS faithfulness of the generated answer |
| Length penalty | 0.05 | Penalises responses shorter than `target_length_tokens` (256) or longer than `max_length_tokens` (512) |

All weights and thresholds are configurable under `rlhf.reward_function` in `config/config.yaml`.

Key function: `composite_reward(prompt, response, context, reward_model) → float`

---

### `ppo_trainer.py`

Online PPO fine-tuning of the policy model against the composite reward signal.

- Uses TRL `PPOTrainer`
- Loads rollout prompts from `data/rlhf/ppo_prompts.jsonl`
- Generates responses from the LoRA policy, scores them with `composite_reward`, and updates via PPO
- KL divergence penalty prevents the policy from drifting too far from the reference model (`init_kl_coef: 0.2`, `target_kl: 6.0`)
- Outputs to `models/rlhf_policy/ppo/`

```bash
python -m src.rlhf.ppo_trainer
```

Key hyperparameters (all in `config/config.yaml` under `rlhf.ppo`):

| Parameter | Default | Notes |
|---|---|---|
| `learning_rate` | 1e-5 | |
| `batch_size` | 8 | Number of prompts per PPO step |
| `mini_batch_size` | 2 | Mini-batches for gradient update |
| `ppo_epochs` | 4 | Gradient steps per rollout batch |
| `total_episodes` | 256 | Total rollout episodes |
| `init_kl_coef` | 0.2 | Initial KL penalty coefficient |
| `target_kl` | 6.0 | Adaptive KL target |

---

### `dpo_trainer.py`

Offline DPO fine-tuning on preference pairs — simpler than PPO since no reward model is needed at training time.

- Uses TRL `DPOTrainer`
- Directly optimises on `(prompt, chosen, rejected)` triples from `data/rlhf/preference_pairs.jsonl`
- `beta` (default 0.1) controls the KL penalty between the policy and reference model
- Outputs to `models/rlhf_policy/dpo/`

```bash
python -m src.rlhf.dpo_trainer
```

Key hyperparameters (all in `config/config.yaml` under `rlhf.dpo`):

| Parameter | Default | Notes |
|---|---|---|
| `learning_rate` | 5e-6 | Lower than PPO — DPO is more sensitive |
| `beta` | 0.1 | KL regularisation strength |
| `num_train_epochs` | 1 | |
| `per_device_train_batch_size` | 2 | |
| `max_length` | 1024 | Max tokens for chosen/rejected |
| `max_prompt_length` | 512 | |

---

## Prerequisites

- `HF_TOKEN` must be set in `.env` to download gated model weights (Qwen models)
- Dependencies: `transformers`, `peft`, `trl`, `datasets` (all in `requirements.txt`)
- GPU strongly recommended for PPO; DPO can run on CPU for small datasets

---

## Full workflow

```bash
# 1. Collect enough feedback (thumbs-up / thumbs-down) via the production UI
# 2. Build preference pairs
python -m src.rlhf.feedback_dataset

# 3. Train the reward model
python -m src.rlhf.reward_model

# 4a. PPO (online, composite reward)
python -m src.rlhf.ppo_trainer

# 4b. DPO (offline, cheaper — good starting point)
python -m src.rlhf.dpo_trainer

# 5. Point config/config.yaml generation.model at the local checkpoint to deploy
```

DPO and PPO are complementary: start with DPO for a quick alignment pass, then apply PPO if you want to optimise further against the learned reward signal.
