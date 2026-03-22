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
```

## Architecture

### Core Loop
Each benchmark runner (`memrl/run/`) orchestrates the same RL loop:
1. **Build**: Agent runs a task → trajectory stored as memory via `MemoryService.build()`
2. **Retrieve**: For future tasks, `MemoryService.retrieve()` finds relevant memories (Two-Phase: keyword-based → RL Q-value filtering)
3. **Update**: After task outcome, `MemoryService.update()` adjusts memory and Q-values

### Key Components

**`memrl/service/memory_service.py`** — Central component (~900 lines). Integrates MemoryOS for storage, implements the RL Q-learning loop on memory retrieval, and manages checkpoint/snapshot lifecycle.

**`memrl/service/strategies.py`** — Strategy enums:
- `BuildStrategy`: `trajectory | script | proceduralization`
- `RetrieveStrategy`: `random | query | avefact`
- `UpdateStrategy`: `vanilla | validation | adjustment`

These are configured via YAML and compose into 27 possible combinations.

**`memrl/configs/config.py`** — Pydantic models (`LLMConfig`, `EmbeddingConfig`, `MemoryConfig`, `RLConfig`, etc.) assembled into `MempConfig`. Load with `MempConfig.from_yaml(path)`.

**`memrl/providers/`** — OpenAI-compatible LLM (`OpenAILLM`) and embedding (`OpenAIEmbedder`) clients. All model endpoints are configured via the YAML config — any OpenAI-compatible API (vLLM, ollama, etc.) works.

**`memrl/agent/memp_agent.py`** — Stateless LLM agent. Receives retrieved memories as context; has no internal state of its own. Decoupled from `MemoryService`.

**`memrl/run/`** — Benchmark runners. Each is a self-contained orchestrator. BCB outputs TensorBoard logs; all runners support epoch checkpointing.

### External Dependencies
- **MemoryOS** (`MemoryOS==1.0.0`): The episodic memory storage backend; auto-generates `configs/mos_config.json` at runtime
- **BigCodeBench**: Vendored at `3rdparty/bigcodebench-main/`
- **LifelongAgentBench**: Vendored at `3rdparty/LifelongAgentBench/`; requires Docker + MySQL

### Outputs
- Logs: `logs/{bcb,llb,alfworld,hle}/`
- TensorBoard: `logs/tensorboard/` (BCB only)
- Checkpoints: Per-epoch memory snapshots (path configured in YAML under `experiment.checkpoint_dir`)
