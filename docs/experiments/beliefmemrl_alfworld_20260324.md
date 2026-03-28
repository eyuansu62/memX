# BeliefMemRL ALFWorld Experiment — 2026-03-24

## Overview

Full 10-section ALFWorld training run comparing **BeliefMemRL** against the **MemRL baseline** (run 2026-03-20).
BeliefMemRL is a lightweight extension of MemRL that augments each episodic memory with a Beta posterior belief state,
belief-aware retrieval scoring, belief-based deduplication, and conflict tracking.

- **Start:** 2026-03-24 15:07
- **End:** 2026-03-26 19:45
- **Duration:** ~52.6 hours
- **Log:** `logs/alfworld_memrl/alfworld_memrl_20260324-150815.log`
- **Results CSV:** `logs/experiment_results_alfworld_memrl_20260326-194543.csv`

---

## Environment & Config

| Parameter | Value |
|-----------|-------|
| Config | `configs/rl_alf_config.local.yaml` |
| LLM | Qwen3-4B-Instruct-2507 (vLLM, tp=2, GPU 0+1, port 8000) |
| Embedding | Qwen3-Embedding-4B (vLLM, tp=1, GPU 1, port 8001, dim=2560) |
| Sections | 10 |
| Batch size | 32 |
| Train games | 3553 (112 mini-batches/section) |
| Eval games | 140 (eval_in_distribution) |
| Random seed | 42 |

---

## Results

### Eval Success Rate (in-distribution, 140 games)

| Section | BeliefMemRL | MemRL Baseline | Δ |
|---------|------------|---------------|---|
| 0 (no memory) | 27.14% | 27.14% | 0 |
| 1 | **52.14%** | 29.29% | +22.9pp |
| 2 | 45.71% | 34.29% | +11.4pp |
| 3 | 49.29% | 36.43% | +12.9pp |
| 4 | 46.43% | 38.57% | +7.9pp |
| 5 | 47.86% | 38.57% | +9.3pp |
| 6 | **52.14%** | 36.43% | +15.7pp |
| 7 | 51.43% | 44.29% | +7.1pp |
| 8 | **52.14%** | 45.71% | +6.4pp |
| 9 | 51.43% | 45.00% | +6.4pp |
| 10 | **52.14%** | 39.29% | **+12.9pp** |

### Training Success Rate & Steps

| Section | Belief Train SR | Baseline Train SR | Δ | Belief Steps | Baseline Steps | Δ |
|---------|----------------|------------------|---|--------------|----------------|---|
| 1 | 40.25% | 28.24% | +12.0pp | 21.39 | 23.90 | -2.5 |
| 2 | 47.26% | 32.25% | +15.0pp | 19.83 | 23.06 | -3.2 |
| 3 | 49.20% | 33.78% | +15.4pp | 19.41 | 22.68 | -3.3 |
| 4 | 48.42% | 34.46% | +14.0pp | 19.61 | 22.55 | -2.9 |
| 5 | 49.77% | 34.98% | +14.8pp | 19.38 | 22.49 | -3.1 |
| 6 | 48.57% | 35.53% | +13.0pp | 19.63 | 22.32 | -2.7 |
| 7 | 49.99% | 36.76% | +13.2pp | 19.30 | 22.08 | -2.8 |
| 8 | 50.49% | 36.01% | +14.5pp | 19.28 | 22.13 | -2.9 |
| 9 | 51.21% | 37.17% | +14.0pp | 19.06 | 21.95 | -2.9 |
| 10 | 51.11% | 38.67% | +12.4pp | 19.14 | 21.67 | -2.5 |

### Cumulative Accuracy

| Section | BeliefMemRL | MemRL Baseline |
|---------|------------|---------------|
| 1 | 40.25% | 28.23% |
| 5 | 60.88% | 53.79% |
| 10 | **64.85%** | 62.76% |

---

## Key Findings

1. **Immediate improvement**: BeliefMemRL jumped from 27.14% → 52.14% eval SR after just one section of training (+22.9pp). The baseline only reached 29.29% after section 1, requiring 7+ sections to approach 45%.

2. **Sustained advantage**: BeliefMemRL held eval SR at 51–52% for sections 6–10, while the baseline peaked at 45.71% (section 8) and degraded to 39.29% at section 10. The belief posterior prevents the retrieval degradation seen in baseline's late sections.

3. **Training efficiency**: BeliefMemRL train SR converged to ~50% and held steady; baseline plateaued at ~37–38%. The ~13–15pp gap was consistent across all sections.

4. **Faster task completion**: BeliefMemRL averaged 19.1–19.8 steps per episode vs baseline's 21.7–23.9 — roughly 2.7–3.3 fewer steps, indicating the retrieved memories provide more actionable guidance.

5. **Stability**: The belief-based deduplication and conflict penalty prevented the late-stage regression seen in the baseline. BeliefMemRL's eval SR variance across sections 6–10 was only 0.71pp (51.43–52.14%) vs baseline's 9.00pp (36.43–45.71%).

---

## BeliefMemRL Modifications (vs MemRL)

| Phase | MemRL | BeliefMemRL |
|-------|-------|-------------|
| Build | Store memory + Q-init | + Annotate belief state (α, β, n_reuse, n_conflict, belief_key) |
| | | + Dual-index by belief text |
| Retrieve | Rank by sim + Q, top-k | + Rescore with 6-term belief score |
| | | + Belief-based deduplication (1 per belief_key) |
| | | + Ambiguity probe signal |
| Update | Q-learning TD update | + Beta posterior update (α/β counts) |
| | | + Conflict tracking |

**Retrieval scoring formula:**
```
score = 0.45·legacy + 0.25·belief_sim + 0.20·(2μ-1) + 0.08·log(1+n_reuse) - 0.08·σ - 0.10·conflict_rate
```

---

## OOD Inference — Qwen3-30B-A3B-FP8 (2026-03-28)

Cross-model OOD test: load BeliefMemRL snapshot 10 (built by Qwen3-4B) and run inference with a larger model (Qwen3-30B-A3B-FP8). Tests whether belief-annotated memories transfer across model scales.

| Eval Set | SR | Avg Steps | Notes |
|----------|----|-----------|-------|
| In-distribution (140 games) | 51.43% (72/140) | 18.97 | Same as training eval |
| **Out-of-distribution** (134 games) | **59.70%** (80/134) | **17.53** | Unseen task types |

- OOD SR (+8.27pp above in-dist) and avg steps (−1.44) are both better on unseen tasks
- Memories encode structural patterns (GOAL_TERMS, CONSTRAINTS) rather than model-specific text — they transfer effectively to a 30B model
- The 30B model leverages the belief-ranked memories more efficiently: fewer steps on OOD tasks suggests cleaner action selection with higher-quality retrieved context
- **Script:** `scripts/test_belief_ood.sh`, **Config:** `configs/rl_alf_config.qwen30b_ood.local.yaml`
- **Results CSV:** `logs/experiment_results_alfworld_belief_ood_20260328-174919.csv`

---

## Implementation Notes

- `memrl/service/belief_memory_service.py` — `BeliefMemoryService(MemoryService)` subclass, no changes to agent or runners beyond import swap
- All 4 benchmark runners wired: `run_alfworld.py`, `run_bcb.py`, `run_llb.py`, `run_hle.py`
- Methods section written at `docs/belief_memrl_method.tex`
- Code pushed to branch `belief-memrl` on GitHub
