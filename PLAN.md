# Implementation Plan: Budgeted Learned State Tracking for MemRL

## Context

This plan closes the gap between the paper's claims (budgeted learned state tracking with 7 state-transition operators and a state-first execution interface) and the current codebase. The current code has a solid BRU pipeline, 5 benchmark runners, Q-learning value updates, and a lightweight BeliefMemoryService. What's missing: explicit memory budget enforcement, Expire/Delete/Redact operators, and a state-first agent interface.

Priority order: (1) Budget mechanism, (2) Expire + Delete, (3) State-first interface on ALFWorld, (4) Redact, (5) Evaluation metrics for state maintenance.

---

## Phase 1: Memory Budget Mechanism

**Goal**: Make the `B` in `z_t = f_θ(x_{1:t}; B)` real. Memories should be capped and evicted when the budget is exceeded.

### 1.1 Add budget config to `memrl/configs/config.py`

In `MemoryConfig`, add:

```python
# Memory budget
memory_budget: int = Field(default=0, ge=0, description="Maximum number of memories allowed (0 = unlimited)")
budget_policy: str = Field(default="q_weighted", description="Eviction policy when budget exceeded: q_weighted, fifo, lru, posterior")
budget_check_interval: int = Field(default=1, ge=1, description="Check budget every N memory additions")
```

### 1.2 Create `memrl/service/budget.py`

Create a new module with a `BudgetManager` class:

```python
class BudgetManager:
    """Enforces memory budget by evicting low-utility memories."""

    def __init__(self, budget: int, policy: str, check_interval: int = 1):
        ...

    def needs_eviction(self, current_count: int) -> bool:
        """Return True if current memory count exceeds budget."""
        ...

    def select_eviction_candidates(self, memories: List[dict], n_to_evict: int) -> List[str]:
        """Select memory IDs to evict based on policy.
        
        Policies:
        - q_weighted: evict memories with lowest composite score = w1*q_value + w2*posterior_mean - w3*recency
        - fifo: evict oldest memories by creation timestamp
        - lru: evict least recently used (by last_used_at)
        - posterior: evict memories with lowest Beta posterior mean (alpha / (alpha + beta))
        """
        ...
```

Key design decisions:
- The composite eviction score for `q_weighted` should combine Q-value, belief posterior mean, recency, and reuse count. Use the same weights as retrieval scoring where applicable.
- Keep a `protected` flag in memory metadata — memories marked protected are never evicted (useful for user-pinned memories).
- Return a list of IDs to evict; actual deletion is handled by the caller (MemoryService).

### 1.3 Integrate BudgetManager into `memrl/service/memory_service.py`

In `MemoryService.__init__()`:
- Instantiate `BudgetManager` if `memory_budget > 0`

In `add_memory()` and `add_memories()`:
- After adding new memories, call `self._enforce_budget()` every `check_interval` additions

New private method `_enforce_budget()`:
1. Count current memories via MemOS query
2. If over budget, call `budget_manager.select_eviction_candidates()`
3. Delete selected memories via MemOS `text_mem.delete()` (you'll need to verify this API exists in MemOS 1.0.0; if not, use the lowest-level available delete)
4. Log evictions with memory_logger if available

Also integrate into `BeliefMemoryService` — it inherits from `MemoryService`, so budget enforcement should propagate automatically. Verify this.

### 1.4 Add budget tracking to analysis

In `memrl/analysis/memory_logger.py`, add a new event type `"evict"` that logs:
- `memory_id`, `eviction_reason` (policy name), `eviction_score`, `epoch`, `episode`

In `memrl/analysis/summary_table.py`, add:
- `eviction_rate()`: fraction of memories evicted per epoch
- `budget_utilization()`: average memory count / budget over time

---

## Phase 2: Expire and Delete Operators

**Goal**: Implement explicit Expire (time-based staleness) and Delete (on-demand removal) as first-class state transitions.

### 2.1 Add expiry metadata to memory items

In `MemoryService.add_memory()`, when building metadata, add:

```python
meta["created_at"] = _now_iso()
meta["ttl_epochs"] = ttl_epochs  # None means never expires; integer means expire after N epochs since creation
meta["validity_window"] = validity_window  # optional (start_epoch, end_epoch) tuple
meta["expired"] = False
```

Add to `MemoryConfig`:
```python
default_ttl_epochs: Optional[int] = Field(default=None, description="Default TTL in epochs for new memories (None = no expiry)")
expire_check_on_retrieve: bool = Field(default=True, description="Check and mark expired memories during retrieval")
```

### 2.2 Implement expire logic in `MemoryService`

New method `expire_stale_memories(current_epoch: int)`:
1. Query all memories
2. For each memory, check if `current_epoch - created_epoch > ttl_epochs` or if `current_epoch > validity_window[1]`
3. Mark as `expired=True` in metadata (soft delete)
4. Optionally hard-delete if budget pressure exists (delegate to BudgetManager)
5. Log as event type `"expire"`

Integrate into retrieval: in `retrieve_query()` and `retrieve_avefact()`, filter out memories where `expired == True` before scoring.

Call `expire_stale_memories()` at the start of each epoch in the runners (or in a new `MemoryService.begin_epoch(epoch_num)` lifecycle method).

### 2.3 Implement explicit Delete

New public method `delete_memories(memory_ids: List[str], reason: str = "explicit")`:
1. Remove from MemOS storage
2. Log deletion events
3. Return count of successfully deleted memories

New public method `delete_by_predicate(predicate_fn)`:
- Useful for bulk cleanup, e.g., delete all memories with Q < threshold, or delete all memories matching a task pattern

### 2.4 Wire into runners

In `memrl/run/base_runner.py` or each individual runner, add an epoch lifecycle hook:

```python
# At the start of each epoch/section
self.memory_service.begin_epoch(epoch)  # triggers expire check + budget enforcement
```

---

## Phase 3: State-First Agent Interface (ALFWorld)

**Goal**: Demonstrate that the agent can operate "state-first" — reading a maintained belief state as primary context rather than raw retrieved memories. Implement this on ALFWorld first since it's the simplest interactive benchmark.

### 3.1 Add state compilation to `BeliefMemoryService`

New method `compile_state(task_description: str, k: int) -> dict`:

```python
def compile_state(self, task_description: str, k: int) -> dict:
    """Compile current belief state relevant to the given task.
    
    Returns a structured state dict containing:
    - active_beliefs: list of high-confidence, non-expired belief summaries
    - uncertain_beliefs: beliefs with high posterior variance
    - constraints: extracted active constraints
    - metadata: budget utilization, staleness stats
    """
    # 1. Retrieve top-k relevant memories (using belief-augmented retrieval)
    # 2. Instead of returning raw memory text, compile into structured state:
    #    - Group by belief_key
    #    - For each group: extract the highest-confidence belief, its posterior stats, last update time
    #    - Separate into "confident" (low variance) and "uncertain" (high variance)
    # 3. Add system-level metadata: total memory count, budget remaining, expired count
    # Return structured dict that can be formatted into a prompt
```

New method `format_state_prompt(state: dict) -> str`:
- Converts the structured state into a concise text block for the agent prompt
- Format example:
  ```
  [ACTIVE STATE]
  - Task pattern "heat+place": Success rate 85% (17/20), Strategy: use microwave then navigate to counter
  - Task pattern "clean+put": Success rate 60% (6/10), Strategy: find sinkbasin, use toggle
  [UNCERTAIN]
  - Task pattern "cool+examine": Only 2 attempts, 50% success — strategy unclear
  [BUDGET] 45/50 memories active, 3 expired this epoch
  ```

### 3.2 Modify `memrl/agent/memp_agent.py`

Add a new method or mode `act_state_first(task, state_prompt, history)`:
- The agent receives the compiled state as PRIMARY context
- Raw retrieved memory text is available as secondary/fallback context only
- The prompt structure changes from:
  ```
  Here are relevant past experiences: [raw memory text]
  ```
  to:
  ```
  Current operating state: [compiled state]
  ---
  (Supporting evidence from memory, if needed): [raw memory text, truncated]
  ```

### 3.3 Modify ALFWorld runner (`memrl/run/alfworld_rl_runner.py`)

Add a config flag:
```python
state_first: bool = Field(default=False, description="Use state-first agent interface")
```

When `state_first=True`:
- Before each episode, call `memory_service.compile_state(task_description, k)`
- Pass compiled state to agent via `act_state_first()`
- Keep raw retrieval as fallback (truncated to save tokens)

This is the minimum viable demonstration of state-first execution.

---

## Phase 4: Redact Operator

**Goal**: Support privacy-preserving memory edits — remove or mask sensitive content from memory while preserving structural utility.

### 4.1 Implement in `memrl/service/memory_service.py`

New method `redact_memories(memory_ids: List[str], redact_patterns: List[str], replacement: str = "[REDACTED]")`:
1. For each memory, retrieve its content text
2. Apply regex/string replacement for each pattern
3. Update the memory content in MemOS (in-place update)
4. Log as event type `"redact"` with field `redacted_fields`
5. Set `meta["redacted"] = True` so downstream code knows this memory was modified

New method `redact_by_entity(entity_type: str, entity_value: str)`:
- Higher-level API: redact all occurrences of a specific entity (e.g., a username, API key, file path) across all memories

### 4.2 This is lower priority

Redact is important for the paper's completeness (7 operators) but less critical for experiments. Implement the basic version and mention more sophisticated approaches (LLM-driven redaction, differential privacy) as future work.

---

## Phase 5: State Maintenance Evaluation Metrics

**Goal**: Add metrics that go beyond recall/F1 to measure state maintenance quality. These are needed to support the paper's claim that recall-oriented benchmarks are insufficient.

### 5.1 Add to `memrl/analysis/summary_table.py`

New metrics:

```python
def consistency_score(log_path: str) -> float:
    """Measure whether the memory state is internally consistent.
    
    Checks: after an Override/Delete/Expire event, do subsequent retrievals
    correctly reflect the updated state? Score = fraction of post-update 
    retrievals that return the updated (not stale) memory.
    """

def intervention_fidelity(log_path: str) -> float:
    """After explicit user corrections (Override events), measure how quickly
    the correction propagates to downstream behavior.
    
    Score = 1.0 if the very next retrieval for the same task pattern returns
    the corrected memory; decays based on lag.
    """

def conflict_resolution_rate(log_path: str) -> float:
    """Fraction of detected contradictions that are resolved (via Override, 
    Delete, or Expire) within K epochs of detection."""

def budget_utility_curve(log_path: str) -> List[Tuple[int, float]]:
    """At different budget levels, what is the task success rate?
    Returns (budget, success_rate) pairs for plotting."""
```

### 5.2 Add conflict detection to `BeliefMemoryService`

During `add_memories()` or `update_values()`, check for contradictions:
- Two memories with the same `belief_key` but opposite outcomes (one success, one failure) = potential conflict
- Log conflict events
- Optionally trigger auto-resolution: keep the more recent or higher-posterior memory, expire the other

### 5.3 Instrument LoComo runner

In `memrl/run/locomo_runner.py`, add state maintenance metrics alongside the existing token-F1:
- After each session update, measure whether the memory state correctly reflects the latest session info (not stale info from earlier sessions)
- This addresses the self-contradiction concern (paper criticizes recall-oriented benchmarks but LoComo uses token-F1)

---

## Phase 6: Config and Runner Wiring

### 6.1 Update YAML configs

Add to `configs/rl_alf_config.yaml`:
```yaml
memory:
  memory_budget: 50
  budget_policy: q_weighted
  default_ttl_epochs: null
  expire_check_on_retrieve: true

experiment:
  state_first: false  # toggle for ablation
```

### 6.2 Ensure backward compatibility

All new features must be OFF by default (`memory_budget: 0`, `state_first: false`, `default_ttl_epochs: null`). Existing experiments must reproduce without config changes.

### 6.3 Add integration tests

Create `tests/test_budget.py`:
- Test that eviction triggers when count > budget
- Test each eviction policy (q_weighted, fifo, lru, posterior)
- Test that expired memories are filtered from retrieval
- Test delete API

Create `tests/test_state_first.py`:
- Test compile_state returns structured output
- Test format_state_prompt produces valid text
- Test that state-first agent receives compiled state

---

## Execution Order (suggested)

1. **Phase 1** (Budget) — ~2-3 hours. This is the foundation; Expire/Delete depend on it.
2. **Phase 2** (Expire/Delete) — ~2 hours. Straightforward once budget infrastructure exists.
3. **Phase 6.1-6.2** (Config wiring) — ~30 min. Wire everything up, ensure backward compat.
4. **Phase 3** (State-first on ALFWorld) — ~3-4 hours. Most design-intensive phase.
5. **Phase 5** (Metrics) — ~2 hours. Needed for paper evaluation.
6. **Phase 4** (Redact) — ~1 hour. Simplest operator; low priority.
7. **Phase 6.3** (Tests) — ~1-2 hours. Validate everything.

Total estimated: ~12-14 hours of implementation work.

---

## Important Notes

- **MemOS 1.0.0 API**: Before implementing delete, verify that `text_mem.delete(memory_id)` exists in MemOS 1.0.0. If it doesn't, you'll need a workaround (e.g., soft delete via metadata flag + retrieval-time filtering).
- **BeliefMemoryService inheritance**: Budget enforcement in `MemoryService` should automatically propagate to `BeliefMemoryService`. Verify after Phase 1 that no methods are inadvertently bypassed.
- **Backward compatibility is critical**: All new features default to OFF. Running `python run/run_alfworld.py --config configs/rl_alf_config.yaml` with the current config must produce identical behavior to the pre-change code.
- **Don't modify the core BRU loop unnecessarily**: The Build/Retrieve/Update pipeline is battle-tested on 5 benchmarks. New operators should integrate cleanly without restructuring existing methods.
