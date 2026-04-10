"""
memory_logger.py — Thread-safe JSONL event logger for MemRL / BeliefMemRL.

Attach to a MemoryService instance via:
    svc._mem_event_logger = MemoryEventLogger("logs/belief_memrl_alfworld.jsonl")

The service calls log_write / log_retrieve / log_update at the appropriate points.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryEventLogger:
    """Appends one JSON object per line to a JSONL file. Thread-safe."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")
        # Mutable context set by the runner each section/mini-batch.
        self._epoch: int = 0
        self._episode: int = 0

    # ------------------------------------------------------------------
    # Context setters (called by the runner)
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int, episode: int = 0) -> None:
        self._epoch = epoch
        self._episode = episode

    def inc_episode(self) -> None:
        self._episode += 1

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------
    def _write(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def _base(self, event_type: str) -> Dict[str, Any]:
        return {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": self._epoch,
            "episode": self._episode,
        }

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------
    def log_write(
        self,
        memory_id: str,
        memory_text: str,
        q_value: float = 0.0,
        task_success: Optional[bool] = None,
        belief_alpha: float = 1.0,
        belief_beta: float = 1.0,
        belief_score: float = 0.5,
        belief_reuse: float = 0.0,
        belief_conflict: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("write")
        ev.update(
            {
                "memory_id": memory_id,
                "memory_text": memory_text[:200],
                "q_value": q_value,
                "belief_alpha": belief_alpha,
                "belief_beta": belief_beta,
                "belief_score": belief_score,
                "belief_reuse": belief_reuse,
                "belief_conflict": belief_conflict,
                "retrieval_rank": None,
                "retrieval_used_in_response": None,
                "task_success": task_success,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_retrieve(
        self,
        memory_id: str,
        memory_text: str,
        retrieval_rank: int,
        q_value: float = 0.0,
        task_success: Optional[bool] = None,
        retrieval_used_in_response: Optional[bool] = None,
        belief_alpha: float = 1.0,
        belief_beta: float = 1.0,
        belief_score: float = 0.5,
        belief_reuse: float = 0.0,
        belief_conflict: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("retrieve")
        ev.update(
            {
                "memory_id": memory_id,
                "memory_text": memory_text[:200],
                "q_value": q_value,
                "belief_alpha": belief_alpha,
                "belief_beta": belief_beta,
                "belief_score": belief_score,
                "belief_reuse": belief_reuse,
                "belief_conflict": belief_conflict,
                "retrieval_rank": retrieval_rank,
                "retrieval_used_in_response": retrieval_used_in_response,
                "task_success": task_success,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_update(
        self,
        memory_id: str,
        q_value: float,
        task_success: Optional[bool] = None,
        belief_alpha: float = 1.0,
        belief_beta: float = 1.0,
        belief_score: float = 0.5,
        belief_reuse: float = 0.0,
        belief_conflict: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("update")
        ev.update(
            {
                "memory_id": memory_id,
                "memory_text": None,
                "q_value": q_value,
                "belief_alpha": belief_alpha,
                "belief_beta": belief_beta,
                "belief_score": belief_score,
                "belief_reuse": belief_reuse,
                "belief_conflict": belief_conflict,
                "retrieval_rank": None,
                "retrieval_used_in_response": None,
                "task_success": task_success,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_evict(
        self,
        memory_id: str,
        eviction_score: float = 0.0,
        policy: str = "q_weighted",
        q_value: float = 0.0,
        belief_alpha: float = 1.0,
        belief_beta: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("evict")
        ev.update(
            {
                "memory_id": memory_id,
                "eviction_score": eviction_score,
                "policy": policy,
                "q_value": q_value,
                "belief_alpha": belief_alpha,
                "belief_beta": belief_beta,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_expire(
        self,
        memory_id: str,
        created_epoch: int = 0,
        ttl_epochs: int = 0,
        current_epoch: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("expire")
        ev.update(
            {
                "memory_id": memory_id,
                "created_epoch": created_epoch,
                "ttl_epochs": ttl_epochs,
                "current_epoch": current_epoch,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_delete(
        self,
        memory_id: str,
        reason: str = "explicit",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("delete")
        ev.update(
            {
                "memory_id": memory_id,
                "reason": reason,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_refine(
        self,
        memory_id: str,
        trigger: str = "belief_instability",
        q_value: float = 0.0,
        belief_alpha: float = 1.0,
        belief_beta: float = 1.0,
        posterior_reset: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("refine")
        ev.update(
            {
                "memory_id": memory_id,
                "trigger": trigger,
                "q_value": q_value,
                "belief_alpha": belief_alpha,
                "belief_beta": belief_beta,
                "posterior_reset": posterior_reset,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_intervention(
        self,
        memory_id: str,
        operator: str = "refine",
        source: str = "external",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("intervention")
        ev.update(
            {
                "memory_id": memory_id,
                "operator": operator,
                "source": source,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def log_redact(
        self,
        memory_id: str,
        n_patterns: int = 0,
        n_replacements: int = 0,
        source: str = "explicit",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = self._base("redact")
        ev.update(
            {
                "memory_id": memory_id,
                "n_patterns": n_patterns,
                "n_replacements": n_replacements,
                "source": source,
            }
        )
        if extra:
            ev.update(extra)
        self._write(ev)

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
