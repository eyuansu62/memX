from __future__ import annotations

"""
A minimal memory-first baseline built on top of MemRL's MemoryService.

Design goals:
1. Keep MemRL's storage, retrieval, and Q-update pipeline intact.
2. Add a lightweight "belief" layer into memory metadata, so memories are
   ranked not only by query similarity and q_value, but also by a memory-side
   posterior over usefulness.
3. Avoid benchmark-specific assumptions. The baseline operates on generic task
   descriptions / trajectories / metadata.

Usage inside MemRL:
    from memrl.service.belief_memory_service import BeliefMemoryService, BeliefConfig

    memsvc = BeliefMemoryService(
        ..., belief_config=BeliefConfig(),
    )

This class is intentionally conservative: it subclasses MemoryService and only
changes three places:
- add_memories(): annotate each memory with belief metadata and optionally add
  a second retrieval key (belief text) into dict_memory.
- retrieve_query(): rerank MemRL's candidates using belief similarity + belief
  posterior statistics, then optionally emit `probe_needed` when the top states
  are ambiguous.
- update_values(): keep MemRL's q update, and in parallel update the belief
  posterior inside memory metadata.
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from memrl.service.memory_service import MemoryService, _meta_to_dict


# A small, generic English stopword set. Keeping it local avoids extra deps.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "if", "in", "into", "is", "it", "its", "of", "on", "or", "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to", "was",
    "will", "with", "you", "your", "must", "should", "can", "cannot", "not",
    "use", "using", "avoid", "need", "needs", "required", "require", "return",
}

_CONSTRAINT_MARKERS = (
    "must",
    "should",
    "avoid",
    "cannot",
    "can't",
    "required",
    "require",
    "exact",
    "without",
    "only",
    "error",
    "exception",
    "fail",
    "failed",
)


@dataclass
class BeliefConfig:
    # Retrieval score = legacy MemRL score + belief-side terms.
    weight_legacy: float = 0.45
    weight_belief_similarity: float = 0.25
    weight_belief_posterior: float = 0.20
    weight_reuse_bonus: float = 0.08
    weight_uncertainty_penalty: float = 0.08
    weight_conflict_penalty: float = 0.10

    # Posterior prior. Each memory starts with one pseudo-success/failure prior.
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    # Belief extraction controls.
    max_goal_terms: int = 8
    max_constraint_lines: int = 4
    max_error_chars: int = 240
    index_belief_text: bool = True       # add belief text as a second retrieval key
    dedup_by_belief: bool = True         # keep at most one selected memory per belief_key
    dedup_by_memory_id: bool = True      # merge duplicate candidates caused by dual indexing

    # Ambiguity flagging (does not change action space; just exposes a signal).
    probe_margin: float = 0.15           # top-2 score gap below this => ambiguous
    probe_uncertainty: float = 0.22      # uncertainty threshold for ambiguity

    # Update strength for posterior counts.
    success_step: float = 1.0
    failure_step: float = 1.0


class BeliefMemoryService(MemoryService):
    """
    A drop-in MemoryService variant that stores a lightweight belief posterior
    in memory metadata and uses it during retrieval and update.
    """

    def __init__(self, *args: Any, belief_config: Optional[BeliefConfig] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.belief_config = belief_config or BeliefConfig()
        self._belief_embedding_cache: Dict[str, List[float]] = {}
        self._belief_text_cache: Dict[str, str] = {}
        self._query_belief_cache: Dict[str, List[float]] = {}

    # ---------------------------------------------------------------------
    # Helpers: text / embedding / metadata I/O
    # ---------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_+#.-]{2,}", str(text or "").lower())

    def _extract_goal_terms(self, task_description: str) -> List[str]:
        text = str(task_description or "").strip()
        if not text:
            return []

        # Prefer the provider's keyword extractor when available.
        try:
            extractor = getattr(self.llm_provider, "extract_keywords", None)
            if callable(extractor):
                kws = extractor(text, self.belief_config.max_goal_terms)
                kws = [str(x).strip().lower() for x in (kws or []) if str(x).strip()]
                seen: set[str] = set()
                deduped: List[str] = []
                for kw in kws:
                    if kw not in seen:
                        seen.add(kw)
                        deduped.append(kw)
                if deduped:
                    return deduped[: self.belief_config.max_goal_terms]
        except Exception:
            pass

        counts: Dict[str, int] = {}
        for tok in self._tokenize(text):
            if tok in _STOPWORDS or tok.isdigit():
                continue
            counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [t for t, _ in ranked[: self.belief_config.max_goal_terms]]

    def _extract_constraint_lines(
        self,
        task_description: str,
        trajectory: str = "",
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> List[str]:
        meta = dict(metadata or {})
        blocks: List[str] = [str(task_description or "")]
        err = str(meta.get("eval_error") or meta.get("error") or "")
        if err:
            blocks.append(err[: self.belief_config.max_error_chars])
        if trajectory:
            blocks.append(str(trajectory)[: 800])

        out: List[str] = []
        seen: set[str] = set()
        for line in "\n".join(blocks).splitlines():
            clean = re.sub(r"\s+", " ", line).strip()
            if not clean:
                continue
            low = clean.lower()
            if any(marker in low for marker in _CONSTRAINT_MARKERS):
                if clean not in seen:
                    seen.add(clean)
                    out.append(clean[: 180])
            if len(out) >= self.belief_config.max_constraint_lines:
                break
        return out

    def _compose_belief_text(
        self,
        task_description: str,
        trajectory: str = "",
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        goals = self._extract_goal_terms(task_description)
        constraints = self._extract_constraint_lines(task_description, trajectory, meta)

        tags: List[str] = []
        for k in ("source_benchmark", "task_id", "entry_point", "phase"):
            val = meta.get(k)
            if val not in (None, "", [], {}):
                tags.append(f"{k}:{val}")

        belief_key_parts = goals[:4] + [c.lower()[:32] for c in constraints[:2]]
        belief_key = " | ".join(x for x in belief_key_parts if x).strip()
        if not belief_key:
            belief_key = "generic"

        text_parts = [f"GOAL_TERMS: {', '.join(goals) if goals else 'generic'}"]
        if constraints:
            text_parts.append(f"CONSTRAINTS: {' || '.join(constraints)}")
        if tags:
            text_parts.append(f"TAGS: {' | '.join(tags)}")

        return {
            "belief_key": belief_key,
            "belief_terms": goals,
            "belief_constraints": constraints,
            "belief_text": "\n".join(text_parts),
        }

    def _embed_text(self, text: str) -> Optional[List[float]]:
        text = str(text or "").strip()
        if not text:
            return None
        embed = getattr(self.embedding_provider, "embed", None)
        if not callable(embed):
            return None
        try:
            vecs = embed([text])
            if vecs:
                return list(vecs[0])
        except Exception:
            return None
        return None

    @staticmethod
    def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a)) or 1e-8
        nb = math.sqrt(sum(float(y) * float(y) for y in b)) or 1e-8
        return float(dot / (na * nb))

    def _get_text_mem(self) -> Any:
        cube_id = getattr(self, "default_cube_id", None)
        if cube_id is None or cube_id not in self.mos.mem_cubes:
            raise RuntimeError("default_cube_id is unavailable in BeliefMemoryService")
        text_mem = self.mos.mem_cubes[cube_id].text_mem
        if text_mem is None:
            raise RuntimeError("Textual memory is not initialized")
        return text_mem

    def _read_memory(self, memory_id: str) -> Any:
        if memory_id in getattr(self, "_mem_cache", {}):
            return self._mem_cache[memory_id]
        text_mem = self._get_text_mem()
        item = text_mem.get(memory_id)
        self._mem_cache[memory_id] = item
        return item

    def _patch_metadata(self, memory_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        text_mem = self._get_text_mem()
        item = self._read_memory(memory_id)
        old_meta = _meta_to_dict(getattr(item, "metadata", None))
        new_meta = old_meta | patch
        text_mem.update(
            memory_id,
            {
                "id": memory_id,
                "memory": getattr(item, "memory", ""),
                "metadata": new_meta,
            },
        )
        # Force fresh reads after metadata updates.
        self._mem_cache.pop(memory_id, None)
        return new_meta

    def _posterior_stats(self, meta: MutableMapping[str, Any]) -> Dict[str, float]:
        cfg = self.belief_config
        alpha = float(meta.get("belief_alpha", cfg.prior_alpha))
        beta = float(meta.get("belief_beta", cfg.prior_beta))
        reuse = float(meta.get("belief_reuse", 0.0))
        conflict = float(meta.get("belief_conflict", 0.0))

        denom = max(alpha + beta, 1e-8)
        mean = alpha / denom
        # Beta posterior stddev
        uncertainty = math.sqrt((alpha * beta) / ((denom ** 2) * (denom + 1.0)))
        reuse_bonus = math.log1p(max(reuse, 0.0)) / math.log(2.0)
        conflict_rate = conflict / max(reuse + 1.0, 1.0)

        return {
            "belief_alpha": alpha,
            "belief_beta": beta,
            "belief_mean": mean,
            "belief_uncertainty": uncertainty,
            "belief_reuse_bonus": reuse_bonus,
            "belief_conflict_rate": conflict_rate,
        }

    def _normalize_add_results(
        self,
        results: Any,
        task_descriptions: Sequence[str],
    ) -> List[Tuple[str, Optional[str]]]:
        if results is None:
            return [(td, None) for td in task_descriptions]
        if isinstance(results, dict):
            # Two possible styles: {task_description: mem_id} or {idx: mem_id}
            out: List[Tuple[str, Optional[str]]] = []
            for td in task_descriptions:
                out.append((td, results.get(td)))
            return out
        if isinstance(results, (list, tuple)):
            out = []
            for i, row in enumerate(results):
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    out.append((str(row[0]), None if row[1] is None else str(row[1])))
                else:
                    td = task_descriptions[i] if i < len(task_descriptions) else str(i)
                    out.append((td, None if row is None else str(row)))
            # pad if needed
            while len(out) < len(task_descriptions):
                td = task_descriptions[len(out)]
                out.append((td, None))
            return out
        return [(td, None) for td in task_descriptions]

    def _index_belief_text(self, memory_id: str, belief_text: str) -> None:
        if not belief_text or not getattr(self, "dict_memory", None):
            return
        lst = self.dict_memory.setdefault(belief_text, [])
        if memory_id not in lst:
            lst.append(memory_id)
        if belief_text not in getattr(self, "query_embeddings", {}):
            vec = self._embed_text(belief_text)
            if vec is not None:
                self.query_embeddings[belief_text] = vec

    # ---------------------------------------------------------------------
    # Write path: keep MemRL add_memories, then annotate memory-side beliefs.
    # ---------------------------------------------------------------------
    def add_memories(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_memory_queries: Optional[List[List[Tuple[str, float]]]] = None,
        retrieved_memory_ids_list: Optional[List[Optional[List[str]]]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Any:
        results = super().add_memories(
            task_descriptions=task_descriptions,
            trajectories=trajectories,
            successes=successes,
            retrieved_memory_queries=retrieved_memory_queries,
            retrieved_memory_ids_list=retrieved_memory_ids_list,
            metadatas=metadatas,
        )

        normalized = self._normalize_add_results(results, task_descriptions)
        metas = list(metadatas or [None] * len(task_descriptions))

        for i, (task_text, memory_id) in enumerate(normalized):
            if not memory_id:
                continue
            td = task_descriptions[i] if i < len(task_descriptions) else task_text
            traj = trajectories[i] if i < len(trajectories) else ""
            succ = bool(successes[i]) if i < len(successes) else True
            meta = dict(metas[i] or {})

            belief = self._compose_belief_text(td, traj, meta)
            patch = {
                **belief,
                # Store a Beta-posterior style memory-side utility statistic.
                "belief_alpha": float(meta.get("belief_alpha", self.belief_config.prior_alpha + (1.0 if succ else 0.0))),
                "belief_beta": float(meta.get("belief_beta", self.belief_config.prior_beta + (0.0 if succ else 1.0))),
                "belief_reuse": float(meta.get("belief_reuse", 0.0)),
                "belief_conflict": float(meta.get("belief_conflict", 0.0)),
                "belief_last_outcome": "success" if succ else "failure",
            }
            stats = self._posterior_stats(patch)
            patch["belief_score"] = float(stats["belief_mean"])
            new_meta = self._patch_metadata(memory_id, patch)

            belief_text = str(new_meta.get("belief_text", "") or "")
            if belief_text:
                self._belief_text_cache[memory_id] = belief_text
                vec = self._embed_text(belief_text)
                if vec is not None:
                    self._belief_embedding_cache[memory_id] = vec
                if self.belief_config.index_belief_text:
                    self._index_belief_text(memory_id, belief_text)

        return results

    # ---------------------------------------------------------------------
    # Read path: rerank MemRL candidates with memory-side belief statistics.
    # ---------------------------------------------------------------------
    def _get_memory_belief_embedding(
        self,
        memory_id: str,
        metadata: MutableMapping[str, Any],
    ) -> Optional[List[float]]:
        if memory_id in self._belief_embedding_cache:
            return self._belief_embedding_cache[memory_id]
        belief_text = str(metadata.get("belief_text", "") or "")
        if not belief_text:
            return None
        vec = self._embed_text(belief_text)
        if vec is not None:
            self._belief_embedding_cache[memory_id] = vec
            self._belief_text_cache[memory_id] = belief_text
        return vec

    def _build_query_belief_embedding(self, task_description: str) -> Optional[List[float]]:
        belief = self._compose_belief_text(task_description)
        belief_text = str(belief.get("belief_text", "") or "")
        if not belief_text:
            return None
        if belief_text in self._query_belief_cache:
            return self._query_belief_cache[belief_text]
        vec = self._embed_text(belief_text)
        if vec is not None:
            self._query_belief_cache[belief_text] = vec
        return vec

    def _score_candidate(
        self,
        candidate: MutableMapping[str, Any],
        query_belief_vec: Optional[List[float]],
    ) -> MutableMapping[str, Any]:
        cfg = self.belief_config
        md = _meta_to_dict(candidate.get("metadata"))
        stats = self._posterior_stats(md)
        belief_vec = self._get_memory_belief_embedding(str(candidate.get("memory_id") or ""), md)
        belief_sim = self._cosine(query_belief_vec, belief_vec)

        legacy_score = float(candidate.get("score", 0.0) or 0.0)
        # Center the posterior mean so that 0.5 = neutral.
        posterior_centered = 2.0 * float(stats["belief_mean"]) - 1.0

        final_score = (
            cfg.weight_legacy * legacy_score
            + cfg.weight_belief_similarity * belief_sim
            + cfg.weight_belief_posterior * posterior_centered
            + cfg.weight_reuse_bonus * float(stats["belief_reuse_bonus"])
            - cfg.weight_uncertainty_penalty * float(stats["belief_uncertainty"])
            - cfg.weight_conflict_penalty * float(stats["belief_conflict_rate"])
        )

        candidate["legacy_score"] = legacy_score
        candidate["belief_similarity"] = belief_sim
        candidate["belief_mean"] = float(stats["belief_mean"])
        candidate["belief_uncertainty"] = float(stats["belief_uncertainty"])
        candidate["belief_reuse_bonus"] = float(stats["belief_reuse_bonus"])
        candidate["belief_conflict_rate"] = float(stats["belief_conflict_rate"])
        candidate["belief_key"] = md.get("belief_key")
        candidate["score"] = final_score
        return candidate

    def _dedup_candidates(self, candidates: Iterable[MutableMapping[str, Any]]) -> List[MutableMapping[str, Any]]:
        if not self.belief_config.dedup_by_memory_id:
            return list(candidates)
        best_by_id: Dict[str, MutableMapping[str, Any]] = {}
        for cand in candidates:
            mid = str(cand.get("memory_id") or "")
            if not mid:
                continue
            prev = best_by_id.get(mid)
            if prev is None or float(cand.get("score", -1e18)) > float(prev.get("score", -1e18)):
                best_by_id[mid] = cand
        return list(best_by_id.values())

    def _select_with_optional_belief_dedup(
        self,
        ranked: List[MutableMapping[str, Any]],
        topk: int,
    ) -> List[MutableMapping[str, Any]]:
        if not self.belief_config.dedup_by_belief:
            return ranked[:topk]
        selected: List[MutableMapping[str, Any]] = []
        seen_beliefs: set[str] = set()
        for cand in ranked:
            bkey = str(cand.get("belief_key") or f"__no_belief__:{cand.get('memory_id')}")
            if bkey in seen_beliefs:
                continue
            seen_beliefs.add(bkey)
            selected.append(cand)
            if len(selected) >= topk:
                break
        return selected

    def _needs_probe(self, ranked: List[MutableMapping[str, Any]]) -> bool:
        if len(ranked) < 2:
            return False
        c1, c2 = ranked[0], ranked[1]
        score_gap = abs(float(c1.get("score", 0.0)) - float(c2.get("score", 0.0)))
        if score_gap > self.belief_config.probe_margin:
            return False
        u1 = float(c1.get("belief_uncertainty", 0.0))
        u2 = float(c2.get("belief_uncertainty", 0.0))
        k1 = str(c1.get("belief_key") or "")
        k2 = str(c2.get("belief_key") or "")
        ambiguous_keys = (not k1) or (not k2) or (k1 != k2)
        return ambiguous_keys and max(u1, u2) >= self.belief_config.probe_uncertainty

    def retrieve_query(self, task_description: str, k: int = 5, threshold: float = 0.0) -> Any:
        ret = super().retrieve_query(task_description, k=k, threshold=threshold)
        if isinstance(ret, tuple):
            result, sim_list = ret
        else:
            result, sim_list = ret, None

        result = dict(result or {})
        candidates = list(result.get("candidates", []) or [])
        if not candidates:
            result["probe_needed"] = False
            return (result, sim_list) if sim_list is not None else result

        query_belief_vec = self._build_query_belief_embedding(task_description)
        rescored = [self._score_candidate(dict(c), query_belief_vec) for c in candidates]
        rescored = self._dedup_candidates(rescored)
        rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        topk = min(int(getattr(self.rl_config, "topk", k) or k), len(rescored))
        selected = self._select_with_optional_belief_dedup(rescored, topk)

        result["candidates"] = rescored
        result["selected"] = selected
        result["actions"] = [str(x.get("memory_id")) for x in selected if x.get("memory_id")]
        result["probe_needed"] = self._needs_probe(rescored)

        return (result, sim_list) if sim_list is not None else result

    # ---------------------------------------------------------------------
    # Update path: preserve MemRL q-learning and add belief posterior updates.
    # ---------------------------------------------------------------------
    def _belief_update_for_memory(self, memory_id: str, success: bool) -> None:
        item = self._read_memory(memory_id)
        meta = _meta_to_dict(getattr(item, "metadata", None))
        stats = self._posterior_stats(meta)
        alpha = float(stats["belief_alpha"])
        beta = float(stats["belief_beta"])
        reuse = float(meta.get("belief_reuse", 0.0)) + 1.0
        conflict = float(meta.get("belief_conflict", 0.0))

        last_outcome = str(meta.get("belief_last_outcome", "") or "").strip().lower()
        if last_outcome:
            if (last_outcome == "success" and not success) or (last_outcome == "failure" and success):
                conflict += 1.0

        if success:
            alpha += float(self.belief_config.success_step)
        else:
            beta += float(self.belief_config.failure_step)

        patch = {
            "belief_alpha": alpha,
            "belief_beta": beta,
            "belief_reuse": reuse,
            "belief_conflict": conflict,
            "belief_last_outcome": "success" if success else "failure",
        }
        patch["belief_score"] = alpha / max(alpha + beta, 1e-8)
        self._patch_metadata(memory_id, patch)

    def update_values(self, successes: List[float], retrieved_ids_list: List[List[str]], rewards: Optional[List[float]] = None) -> Dict[str, Optional[float]]:
        # Keep MemRL's original Q update path.
        q_updates = super().update_values(successes=successes, retrieved_ids_list=retrieved_ids_list, rewards=rewards)

        for success, mem_ids in zip(successes, retrieved_ids_list):
            ok = bool(success)
            for mem_id in (mem_ids or []):
                if not mem_id:
                    continue
                try:
                    self._belief_update_for_memory(str(mem_id), ok)
                except Exception:
                    # Best-effort: belief metadata must not break the original MemRL path.
                    continue
        return q_updates
