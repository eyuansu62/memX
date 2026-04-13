# Compile-Time Resolve Ablation Plan

## Goal

Prove that **state-first compile-time resolve** (not just "a stronger model summarizing") is what drives performance gains. Separate three sources of improvement: (1) LLM compiler itself, (2) belief metadata, (3) state-resolve instruction.

## Four Ablation Groups

| # | Name | compile_mode | compiler_use_belief | What it tests | Config |
|---|------|-------------|--------------------|----|--------|
| 1 | **annotate** | `annotate` | true | Current v2 baseline (no LLM compiler call) | `rl_alf_ablation_annotate.yaml` |
| 2 | **summary** | `summary` | false | Generic summary control — isolates "is it just because a compiler helps?" | `rl_alf_ablation_summary.yaml` |
| 3 | **summary+belief** | `summary` | true | Summary + belief metadata — isolates "is it just belief signals?" | `rl_alf_ablation_summary_belief.yaml` |
| 4 | **resolve+belief** | `resolve` | true | **Our method** — state-resolve with belief | `rl_alf_ablation_resolve_belief.yaml` |

Also keep the existing **off** (raw memory list, no state_first) run as the base baseline.

## Run Order & Rationale

### Round 1: resolve+belief (our method) — run FIRST

```bash
python run/run_alfworld.py --config configs/rl_alf_ablation_resolve_belief.yaml
```

**Why first:** If resolve doesn't beat annotate, we stop and debug the prompt before wasting GPU on controls. This is the make-or-break experiment.

**What to watch:** Compare S1-S4 validation against the existing `annotate` run (currently in progress) and the `off` baseline (~47-50%).

### Round 2: summary (no belief) — generic summary control

```bash
python run/run_alfworld.py --config configs/rl_alf_ablation_summary.yaml
```

**Why second:** This is the most critical control. If `summary ≈ resolve+belief`, then our gain is just from having a compiler, not from state-resolve. We need this result before claiming anything in the paper.

**What to watch:** If summary is significantly worse than resolve+belief → our resolve instruction matters. If summary ≈ resolve+belief → the gain is from compilation, not resolve.

### Round 3: summary+belief — belief metadata ablation

```bash
python run/run_alfworld.py --config configs/rl_alf_ablation_summary_belief.yaml
```

**Why third:** Separates "belief metadata" from "resolve instruction". Together with Round 2, answers: does giving the compiler confidence/conflict signals help even in generic summarization mode?

**What to watch:**
- If `summary+belief > summary` → belief metadata helps the compiler regardless of instruction type.
- If `summary+belief ≈ resolve+belief` → belief matters, resolve instruction doesn't.
- If `summary+belief < resolve+belief` → resolve instruction adds value beyond just metadata.

### Round 4: annotate (if not already done)

```bash
python run/run_alfworld.py --config configs/rl_alf_ablation_annotate.yaml
```

Already running. Just let it finish for the comparison.

## Expected Results Table

| Group | compile_mode | belief | Expected result | What it proves |
|-------|-------------|--------|----------------|---------------|
| off (baseline) | off | - | ~47-50% | Base performance |
| annotate | annotate | yes | ~45-48% | Belief tags alone don't help 4B |
| summary | summary | no | ? | Does compilation alone help? |
| summary+belief | summary | yes | ? | Does belief metadata help compilation? |
| **resolve+belief** | **resolve** | **yes** | **target: >50%** | **State-resolve is the key** |

## Interpretation Guide

**Best case** (supports our paper): `resolve+belief >> summary > annotate ≈ off`
→ State-resolve instruction with belief signals provides unique value beyond generic compilation.

**Acceptable case**: `resolve+belief > summary+belief > summary > annotate`
→ Both resolve instruction and belief metadata contribute. Paper claim is solid.

**Needs pivot**: `summary ≈ resolve+belief >> annotate`
→ Gain comes from compilation, not resolve-specific instruction. Shift paper emphasis to "maintained belief substrate enables effective compilation" rather than "state-resolve instruction is key."

**Worst case**: `summary ≈ resolve+belief ≈ annotate ≈ off`
→ Compilation doesn't help at this scale. Likely need stronger compiler model (30B).

## 30B Compiler Variant

If results are promising with same-model compilation, run the key groups again with 30B compiler:

In each config, change:
```yaml
compiler:
  use_actor_model: false
  provider: "openai"
  api_key: "sk-REPLACE_ME"
  base_url: "http://localhost:8001/v1"  # 30B endpoint
  model: "Qwen3-30B-A3B"
  temperature: 0.3
  max_tokens: 1024
```

Priority: `resolve+belief (30B)` > `summary (30B)` > others.

## Notes

- All groups share the same RL config, memory config, embedding, seed (42), and dataset split.
- The only variables are `compile_mode` and `compiler_use_belief`.
- Each run produces logs in `results/alfworld/exp_{experiment_name}_*/`.
- Compare S0-S4 validation accuracy across groups.
- S0 should be identical across all groups (no memories yet).
