# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MemRL** implements Runtime Reinforcement Learning on Episodic Memory — a framework for self-evolving agents that improve via external memory updates rather than parameter changes. The core idea: stable LLM reasoning + plastic episodic memory (backed by MemoryOS).

## Setup

```bash
conda create -n memoryrl python=3.10 -y
conda activate memoryrl
pip install -r requirements.txt
```

## Running Benchmarks

All configs live in `configs/`. Set `llm.api_key` and `embedding.api_key` in your config (or create a `.local.yaml` sibling file for private values).

```bash
# BigCodeBench (code generation, multi-epoch)
python run/run_bcb.py --config configs/rl_bcb_config.yaml --split instruct --epochs 10

# Lifelong Agent Bench (requires Docker pre-started per LLB docs)
python run/run_llb.py  # reads configs/rl_llb_config.yaml

# ALFWorld (interactive environments)
python run/run_alfworld.py --config configs/rl_alf_config.yaml

# Humanity's Last Exam (QA)
python run/run_hle.py --config configs/rl_hle_config.yaml --train /path/to/hle_train.parquet

# LoComo (conversational memory)
python run/run_locomo.py --config configs/rl_locomo_config.yaml
```

## Running Tests

```bash
pytest tests/
pytest tests/test_belief_config.py   # BeliefConfigModel + YAML loading
pytest tests/test_state_first.py     # redaction + state-first config wiring
```

## Architecture

### Core Loop
Each benchmark runner (`memrl/run/`) orchestrates the same RL loop:
1. **Begin epoch**: `MemoryService.begin_epoch()` expires stale memories and enforces budget
2. **Build**: Agent runs a task → trajectory stored as memory via `MemoryService.build()`
3. **Retrieve**: For future tasks, `MemoryService.retrieve()` finds relevant memories (Two-Phase: keyword-based → RL Q-value filtering, optionally reranked by belief posterior)
4. **Update**: After task outcome, `MemoryService.update()` adjusts Q-values and belief posteriors; triggers auto-refine if conflict rate exceeds threshold

### Key Components

**`memrl/service/memory_service.py`** — Central component. Integrates MemoryOS for storage, implements the RL Q-learning loop, and manages checkpoint/snapshot lifecycle. Also owns:
- `compile_state(task, k, threshold)` → structured dict of `active_beliefs`, `uncertain_beliefs`, `budget_info`, and `raw_retrieval`
- `format_state_prompt(state)` → concise text block for agent prompts
- `begin_epoch(epoch)` → expiry + budget enforcement hook
- `delete_memories()` / `redact_memories()` / `redact_by_entity()` → lifecycle management

**`memrl/service/belief_memory_service.py`** — Extends `MemoryService` with a Beta conjugate prior over memory usefulness. Key additions:
- Write routing: directs to Create vs Refine based on budget pressure and similarity
- Belief annotation: attaches `alpha`/`beta` counts and `belief_key`/`belief_text` metadata
- Retrieval reranking: blends Q-value score with belief similarity and posterior stats
- Auto-refine: triggers LLM rewrite when `conflict_rate` exceeds `auto_refine_conflict_threshold`
- External intervention: `intervene(refine|override|delete|redact)`

**`memrl/service/budget.py`** — `BudgetManager` enforces the hard memory budget |z_t| ≤ B. Eviction policies: `q_weighted` (default), `fifo`, `lru`, `posterior`. Computes composite retention score: `r = w1×Q + w2×posterior_mean + w3×recency`.

**`memrl/service/strategies.py`** — Strategy enums:
- `BuildStrategy`: `trajectory | script | proceduralization`
- `RetrieveStrategy`: `random | query | avefact`
- `UpdateStrategy`: `vanilla | validation | adjustment`

**`memrl/configs/config.py`** — Pydantic models assembled into `MempConfig`. Load with `MempConfig.from_yaml(path)`. Relevant new fields:

*In `MemoryConfig`:*
```yaml
memory_budget: 0              # max memories (0 = unlimited)
budget_policy: q_weighted     # q_weighted | fifo | lru | posterior
budget_utilization_threshold: 0.8  # prefer Refine over Create above this utilization
default_ttl_epochs: null      # per-memory TTL; null = never expire
expire_check_on_retrieve: true
```

*In `ExperimentConfig`:*
```yaml
state_first: false            # use compiled belief state as primary agent context
```

*`BeliefConfigModel`* — 20+ fields for retrieval weights, Beta prior params (`prior_alpha`, `prior_beta`), belief extraction (`max_goal_terms`, `dedup_by_belief`), ambiguity flagging (`probe_margin`, `probe_uncertainty`), and auto-refine thresholds. Convert to dataclass with `.to_dataclass()`.

**`memrl/agent/memp_agent.py`** — Stateless LLM agent. When `state_first=True`, `_construct_messages_state_first()` uses compiled belief state (confident + uncertain beliefs with success rates) as primary context; raw memories are truncated fallback evidence only.

**`memrl/providers/`** — OpenAI-compatible LLM (`OpenAILLM`) and embedding (`OpenAIEmbedder`) clients. Any OpenAI-compatible API (vLLM, ollama, etc.) works.

**`memrl/run/`** — Benchmark runners. Each is a self-contained orchestrator. BCB outputs TensorBoard logs; all runners support epoch checkpointing. Runners call `begin_epoch()` + `check_divergence_and_refine()` to wire in adaptive state tracking.

**`memrl/analysis/`** — Observability layer:
- `memory_logger.py` (`MemoryEventLogger`) — thread-safe JSONL event log for write/retrieve/update/refine/delete/redact events
- `logged_services.py` (`LoggedMemoryService`, `LoggedBeliefMemoryService`) — transparent wrappers that intercept service calls and emit events
- `summary_table.py` — combines experiment CSV results with JSONL diagnostic metrics

### External Dependencies
- **MemoryOS** (`MemoryOS==1.0.0`): The episodic memory storage backend; auto-generates `configs/mos_config.json` at runtime
- **BigCodeBench**: Vendored at `3rdparty/bigcodebench-main/`
- **LifelongAgentBench**: Vendored at `3rdparty/LifelongAgentBench/`; requires Docker + MySQL
- **LoComo**: Vendored at `3rdparty/locomo/`

### Outputs
- Logs: `logs/{bcb,llb,alfworld,hle,locomo}/`
- JSONL memory event logs: written alongside experiment logs when using `LoggedMemoryService`
- TensorBoard: `logs/tensorboard/` (BCB only)
- Checkpoints: Per-epoch memory snapshots (path configured in YAML under `experiment.checkpoint_dir`)
