"""
logged_services.py — Thin logging wrappers around MemoryService / BeliefMemoryService.

Usage in run_alfworld.py:
    from memrl.analysis.logged_services import LoggedMemoryService, LoggedBeliefMemoryService
    from memrl.analysis.memory_logger import MemoryEventLogger

    log = MemoryEventLogger("logs/belief_memrl_alfworld.jsonl")
    svc = LoggedBeliefMemoryService(..., belief_config=BeliefConfig())
    svc._mem_event_logger = log
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from memrl.service.memory_service import MemoryService, _meta_to_dict
from memrl.service.belief_memory_service import BeliefMemoryService, BeliefConfig


class LoggedMemoryService(MemoryService):
    """MemoryService with JSONL event logging hooks."""

    # -----------------------------------------------------------------------
    # retrieve_query
    # -----------------------------------------------------------------------
    def retrieve_query(self, task_description: str, k: int = 5, threshold: float = 0.0) -> Any:
        ret = super().retrieve_query(task_description, k=k, threshold=threshold)
        log = getattr(self, "_mem_event_logger", None)
        if log is None:
            return ret

        result = ret[0] if isinstance(ret, tuple) else ret
        selected = (result or {}).get("selected", []) or []
        # Store pending retrieves so add_memories can finalize retrieval_used_in_response.
        pending = getattr(self, "_pending_retrieves", {})
        pending_for_task = []
        for rank, mem in enumerate(selected, 1):
            mem_id = str(mem.get("memory_id") or "")
            mem_text = str(mem.get("memory", "") or "")[:200]
            q_val = float(mem.get("q_estimate", mem.get("score", 0)) or 0)
            md = _meta_to_dict(mem.get("metadata"))
            entry = {
                "memory_id": mem_id,
                "memory_text": mem_text,
                "q_value": q_val,
                "retrieval_rank": rank,
                "belief_alpha": float(md.get("belief_alpha", 1.0)),
                "belief_beta": float(md.get("belief_beta", 1.0)),
                "belief_score": float(md.get("belief_score", 0.5)),
                "belief_reuse": float(md.get("belief_reuse", 0.0)),
                "belief_conflict": float(md.get("belief_conflict", 0.0)),
            }
            pending_for_task.append(entry)

        pending[task_description] = pending_for_task
        self._pending_retrieves = pending
        return ret

    # -----------------------------------------------------------------------
    # update_values
    # -----------------------------------------------------------------------
    def update_values(
        self,
        successes: List[float],
        retrieved_ids_list: List[List[str]],
        rewards: Optional[List[float]] = None,
    ) -> Dict[str, Optional[float]]:
        results = super().update_values(successes, retrieved_ids_list, rewards=rewards)
        log = getattr(self, "_mem_event_logger", None)
        if log is None:
            return results

        # Build mem_id → success map
        mem_success: Dict[str, bool] = {}
        for succ, mem_ids in zip(successes, retrieved_ids_list):
            for mid in (mem_ids or []):
                if mid and mid not in mem_success:
                    mem_success[mid] = bool(succ)

        for mem_id, new_q in (results or {}).items():
            log.log_update(
                memory_id=mem_id,
                q_value=float(new_q or 0),
                task_success=mem_success.get(mem_id),
            )
        return results

    # -----------------------------------------------------------------------
    # add_memories
    # -----------------------------------------------------------------------
    def add_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_memory_queries=None,
        retrieved_memory_ids_list=None,
        metadatas=None,
    ) -> Any:
        results = super().add_memories(
            task_descriptions=task_descriptions,
            trajectories=trajectories,
            successes=successes,
            retrieved_memory_queries=retrieved_memory_queries,
            retrieved_memory_ids_list=retrieved_memory_ids_list,
            metadatas=metadatas,
        )
        log = getattr(self, "_mem_event_logger", None)
        if log is None:
            return results

        pending = getattr(self, "_pending_retrieves", {})

        # Normalize results to list of (task_desc, mem_id)
        if isinstance(results, dict):
            pairs = list(results.items())  # (task_desc, mem_id)
        elif isinstance(results, (list, tuple)):
            pairs = [(str(r[0]), r[1] if isinstance(r, (list, tuple)) else r) for r in results]
        else:
            pairs = [(td, None) for td in task_descriptions]

        for i, (td, mem_id) in enumerate(pairs):
            traj = trajectories[i] if i < len(trajectories) else ""
            succ = bool(successes[i]) if i < len(successes) else None
            traj_lower = str(traj or "").lower()

            # Finalize pending retrieve events for this task_desc.
            for entry in pending.pop(td, []):
                snippet = entry["memory_text"][:50].lower().strip()
                used = bool(snippet and snippet in traj_lower)
                log.log_retrieve(
                    memory_id=entry["memory_id"],
                    memory_text=entry["memory_text"],
                    retrieval_rank=entry["retrieval_rank"],
                    q_value=entry["q_value"],
                    task_success=succ,
                    retrieval_used_in_response=used,
                    belief_alpha=entry["belief_alpha"],
                    belief_beta=entry["belief_beta"],
                    belief_score=entry["belief_score"],
                    belief_reuse=entry["belief_reuse"],
                    belief_conflict=entry["belief_conflict"],
                )

            if not mem_id:
                continue
            log.log_write(
                memory_id=str(mem_id),
                memory_text=str(td)[:200],
                task_success=succ,
            )

        return results


class LoggedBeliefMemoryService(LoggedMemoryService, BeliefMemoryService):
    """BeliefMemoryService with JSONL event logging.

    MRO: LoggedBeliefMemoryService → LoggedMemoryService → BeliefMemoryService → MemoryService

    retrieve_query and update_values come from BeliefMemoryService (which calls super()).
    We override here so that logging happens AFTER the belief-enriched results are built.
    """

    def __init__(self, *args: Any, belief_config: Optional[BeliefConfig] = None, **kwargs: Any) -> None:
        # BeliefMemoryService.__init__ handles belief_config
        super().__init__(*args, belief_config=belief_config, **kwargs)

    # -----------------------------------------------------------------------
    # retrieve_query — use BeliefMemoryService result (with belief scores)
    # -----------------------------------------------------------------------
    def retrieve_query(self, task_description: str, k: int = 5, threshold: float = 0.0) -> Any:
        # Call BeliefMemoryService.retrieve_query (skips LoggedMemoryService.retrieve_query)
        ret = BeliefMemoryService.retrieve_query(self, task_description, k=k, threshold=threshold)
        log = getattr(self, "_mem_event_logger", None)
        if log is None:
            return ret

        result = ret[0] if isinstance(ret, tuple) else ret
        selected = (result or {}).get("selected", []) or []
        pending = getattr(self, "_pending_retrieves", {})
        pending_for_task = []
        for rank, mem in enumerate(selected, 1):
            mem_id = str(mem.get("memory_id") or "")
            mem_text = str(mem.get("memory", "") or "")[:200]
            q_val = float(mem.get("score", mem.get("q_estimate", 0)) or 0)
            md = _meta_to_dict(mem.get("metadata"))
            entry = {
                "memory_id": mem_id,
                "memory_text": mem_text,
                "q_value": q_val,
                "retrieval_rank": rank,
                "belief_alpha": float(md.get("belief_alpha", 1.0)),
                "belief_beta": float(md.get("belief_beta", 1.0)),
                "belief_score": float(md.get("belief_score", float(mem.get("belief_score", 0.5)))),
                "belief_reuse": float(md.get("belief_reuse", 0.0)),
                "belief_conflict": float(md.get("belief_conflict", 0.0)),
            }
            pending_for_task.append(entry)

        pending[task_description] = pending_for_task
        self._pending_retrieves = pending
        return ret

    # -----------------------------------------------------------------------
    # update_values — read updated belief metadata after BeliefMemoryService update
    # -----------------------------------------------------------------------
    def update_values(
        self,
        successes: List[float],
        retrieved_ids_list: List[List[str]],
        rewards: Optional[List[float]] = None,
    ) -> Dict[str, Optional[float]]:
        # Call BeliefMemoryService.update_values (which updates belief posteriors)
        results = BeliefMemoryService.update_values(self, successes, retrieved_ids_list, rewards=rewards)
        log = getattr(self, "_mem_event_logger", None)
        if log is None:
            return results

        mem_success: Dict[str, bool] = {}
        for succ, mem_ids in zip(successes, retrieved_ids_list):
            for mid in (mem_ids or []):
                if mid and mid not in mem_success:
                    mem_success[mid] = bool(succ)

        for mem_id, new_q in (results or {}).items():
            # Read updated belief stats from metadata.
            try:
                item = self._read_memory(mem_id)
                md = _meta_to_dict(getattr(item, "metadata", None))
            except Exception:
                md = {}
            log.log_update(
                memory_id=mem_id,
                q_value=float(new_q or 0),
                task_success=mem_success.get(mem_id),
                belief_alpha=float(md.get("belief_alpha", 1.0)),
                belief_beta=float(md.get("belief_beta", 1.0)),
                belief_score=float(md.get("belief_score", 0.5)),
                belief_reuse=float(md.get("belief_reuse", 0.0)),
                belief_conflict=float(md.get("belief_conflict", 0.0)),
            )
        return results

    # add_memories is inherited from LoggedMemoryService (calls BeliefMemoryService.add_memories
    # via MRO since LoggedMemoryService.add_memories calls super() which resolves to
    # BeliefMemoryService.add_memories in this MRO chain).
