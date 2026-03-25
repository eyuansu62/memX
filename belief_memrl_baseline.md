# Belief-MemRL baseline

This baseline is a minimal, drop-in modification of MemRL.

## What changes

1. **Memory metadata** gains a lightweight belief posterior:
   - `belief_key`
   - `belief_text`
   - `belief_alpha`, `belief_beta`
   - `belief_reuse`, `belief_conflict`
   - `belief_score`

2. **Retrieval** keeps MemRL's original hybrid similarity + Q score, but reranks with:

```text
final_score =
    w_legacy * legacy_memrl_score
  + w_belief_similarity * cosine(query_belief, memory_belief)
  + w_belief_posterior * (2 * posterior_mean - 1)
  + w_reuse_bonus * log(1 + reuse)
  - w_uncertainty_penalty * posterior_std
  - w_conflict_penalty * conflict_rate
```

3. **Update** still runs MemRL's original `q_value` update, and additionally updates a memory-side Beta posterior.

4. **Optional probe signal**: retrieval returns `probe_needed=True` when the top two memory states are close in score but disagree in belief state under high uncertainty.

## Files

- `memrl/service/belief_memory_service.py`

## Minimal wiring

Replace this import:

```python
from memrl.service.memory_service import MemoryService
```

with:

```python
from memrl.service.belief_memory_service import BeliefMemoryService, BeliefConfig
```

Then instantiate:

```python
memsvc = BeliefMemoryService(
    mos_config_path=mos_config_path,
    llm_provider=llm,
    embedding_provider=embedder,
    strategy_config=...,
    user_id=user_id,
    num_workers=cfg.experiment.batch_size,
    max_keywords=cfg.memory.max_keywords,
    add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
    enable_value_driven=cfg.experiment.enable_value_driven,
    rl_config=cfg.rl_config,
    belief_config=BeliefConfig(),
)
```

## Good first ablations

1. `MemoryService` vs `BeliefMemoryService`
2. with / without `index_belief_text`
3. with / without `dedup_by_belief`
4. stronger vs weaker `weight_legacy`

## Suggested default interpretation

This is **not** a full external belief-state system yet.
It is a conservative baseline that moves MemRL from:

- memory ranked mainly by **query similarity + q_value**

toward:

- memory ranked by **query similarity + q_value + memory-side belief posterior**

That makes it a reasonable first implementation of the broader memory-first idea without rewriting MemRL's full runtime.
