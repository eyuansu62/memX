"""
Budget enforcement for the memory system.

Implements the hard budget constraint |z_t| <= B from the paper.
When memory count exceeds the budget after a Create, the BudgetManager
selects low-value entries for eviction using a configurable policy.
"""

import logging
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BudgetManager:
    """Enforces memory budget by selecting eviction candidates.

    The manager does NOT perform deletion itself; it returns a list of
    memory IDs that should be evicted.  The caller (MemoryService) is
    responsible for the actual deletion via MemOS.

    Supports four eviction policies:
      - q_weighted : composite retention score  r = w1*Q + w2*posterior_mean + w3*recency
      - fifo       : evict oldest memories by creation timestamp
      - lru        : evict least recently used (by last_used_at)
      - posterior   : evict memories with lowest Beta posterior mean
    """

    # Default weights for q_weighted retention score (Eq. retention in the paper)
    DEFAULT_W1 = 0.5   # Q-value weight
    DEFAULT_W2 = 0.3   # Posterior mean weight
    DEFAULT_W3 = 0.2   # Recency weight

    # Recency half-life in seconds (memories unused for this long get recency ~ 0.5)
    RECENCY_HALFLIFE = 86400.0 * 7  # 7 days

    def __init__(
        self,
        budget: int,
        policy: str = "q_weighted",
        check_interval: int = 1,
        utilization_threshold: float = 0.8,
        w1: float = DEFAULT_W1,
        w2: float = DEFAULT_W2,
        w3: float = DEFAULT_W3,
    ):
        if budget <= 0:
            raise ValueError(f"Budget must be positive, got {budget}")
        self.budget = budget
        self.policy = policy
        self.check_interval = check_interval
        self.utilization_threshold = utilization_threshold
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # Counter for check_interval gating
        self._add_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def needs_eviction(self, current_count: int) -> bool:
        """Return True if current memory count exceeds budget."""
        return current_count > self.budget

    def should_prefer_refine(self, current_count: int) -> bool:
        """Return True if budget utilization exceeds threshold (prefer Refine over Create)."""
        return current_count / self.budget >= self.utilization_threshold

    def should_check(self) -> bool:
        """Return True if enough additions have occurred to warrant a budget check."""
        self._add_count += 1
        if self._add_count >= self.check_interval:
            self._add_count = 0
            return True
        return False

    def select_eviction_candidates(
        self,
        memories: List[Dict[str, Any]],
        n_to_evict: int,
    ) -> List[str]:
        """Select memory IDs to evict based on the configured policy.

        Args:
            memories: List of memory dicts, each containing at minimum:
                - "memory_id": str
                - "metadata": dict with optional keys:
                    "q_value", "belief_alpha", "belief_beta",
                    "created_at", "last_used_at", "protected"
            n_to_evict: Number of memories to evict.

        Returns:
            List of memory_id strings to evict (length <= n_to_evict).
        """
        if n_to_evict <= 0:
            return []

        # Never evict protected memories
        eligible = [m for m in memories if not self._is_protected(m)]

        if not eligible:
            logger.warning("All memories are protected; cannot evict any.")
            return []

        # Separate expired memories — they are evicted first
        expired = [m for m in eligible if self._is_expired(m)]
        non_expired = [m for m in eligible if not self._is_expired(m)]

        evict_ids: List[str] = []

        # Phase 1: evict expired memories first
        for m in expired:
            if len(evict_ids) >= n_to_evict:
                break
            evict_ids.append(m["memory_id"])

        remaining = n_to_evict - len(evict_ids)
        if remaining <= 0:
            return evict_ids

        # Phase 2: evict non-expired memories by policy
        scored = self._score_by_policy(non_expired)
        # Sort ascending by score — lowest score = first to evict
        scored.sort(key=lambda x: x[1])

        for mem_id, _score in scored[:remaining]:
            evict_ids.append(mem_id)

        return evict_ids

    @property
    def utilization(self) -> float:
        """Current utilization is not tracked here; caller provides current_count."""
        raise NotImplementedError("Call needs_eviction(current_count) instead.")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_by_policy(
        self, memories: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Return (memory_id, retention_score) pairs for the given policy."""
        if self.policy == "q_weighted":
            return self._score_q_weighted(memories)
        elif self.policy == "fifo":
            return self._score_fifo(memories)
        elif self.policy == "lru":
            return self._score_lru(memories)
        elif self.policy == "posterior":
            return self._score_posterior(memories)
        else:
            logger.warning(f"Unknown budget policy '{self.policy}', falling back to q_weighted.")
            return self._score_q_weighted(memories)

    def _score_q_weighted(
        self, memories: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Retention score: r = w1*Q + w2*posterior_mean + w3*recency (Eq. retention)."""
        now = datetime.now()
        results = []
        for m in memories:
            meta = m.get("metadata", {})
            q = float(meta.get("q_value", 0.0))
            alpha = float(meta.get("belief_alpha", 1.0))
            beta = float(meta.get("belief_beta", 1.0))
            posterior_mean = alpha / (alpha + beta)

            recency = self._compute_recency(meta, now)

            score = self.w1 * q + self.w2 * posterior_mean + self.w3 * recency
            results.append((m["memory_id"], score))
        return results

    def _score_fifo(
        self, memories: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """FIFO: oldest creation time gets lowest score."""
        results = []
        for m in memories:
            meta = m.get("metadata", {})
            created = self._parse_ts(meta.get("created_at"))
            # Use timestamp as score; oldest = smallest score = evicted first
            score = created.timestamp() if created else 0.0
            results.append((m["memory_id"], score))
        return results

    def _score_lru(
        self, memories: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """LRU: least recently used gets lowest score."""
        results = []
        for m in memories:
            meta = m.get("metadata", {})
            last_used = self._parse_ts(meta.get("last_used_at"))
            if last_used is None:
                last_used = self._parse_ts(meta.get("created_at"))
            score = last_used.timestamp() if last_used else 0.0
            results.append((m["memory_id"], score))
        return results

    def _score_posterior(
        self, memories: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Posterior: lowest Beta posterior mean gets evicted first."""
        results = []
        for m in memories:
            meta = m.get("metadata", {})
            alpha = float(meta.get("belief_alpha", 1.0))
            beta = float(meta.get("belief_beta", 1.0))
            score = alpha / (alpha + beta)
            results.append((m["memory_id"], score))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_recency(self, meta: Dict[str, Any], now: datetime) -> float:
        """Exponential decay recency based on last_used_at or created_at."""
        last_used = self._parse_ts(meta.get("last_used_at"))
        if last_used is None:
            last_used = self._parse_ts(meta.get("created_at"))
        if last_used is None:
            return 0.0
        elapsed = max((now - last_used).total_seconds(), 0.0)
        return math.exp(-math.log(2) * elapsed / self.RECENCY_HALFLIFE)

    @staticmethod
    def _is_protected(m: Dict[str, Any]) -> bool:
        meta = m.get("metadata", {})
        return bool(meta.get("protected", False))

    @staticmethod
    def _is_expired(m: Dict[str, Any]) -> bool:
        meta = m.get("metadata", {})
        return bool(meta.get("expired", False))

    @staticmethod
    def _parse_ts(value: Any) -> Optional[datetime]:
        """Best-effort parse of datetime values."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.strip())
            except Exception:
                return None
        return None
