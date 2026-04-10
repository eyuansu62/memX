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

    # Auto-refine trigger: when conflict_rate exceeds this, trigger posterior reset.
    auto_refine_conflict_threshold: float = 0.5
    auto_refine_min_reuse: int = 3  # only trigger after this many reuse events


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
    # Write routing: decide Create vs Refine based on budget pressure
    # ---------------------------------------------------------------------
    def _find_similar_memory(
        self,
        task_description: str,
        trajectory: str,
        sim_threshold: float = 0.75,
    ) -> Optional[str]:
        """Find an existing memory similar to this task, for write routing.

        Returns memory_id if a similar memory exists, None otherwise.
        """
        # Use belief key overlap as a fast similarity proxy
        new_belief = self._compose_belief_text(task_description, trajectory)
        new_key = new_belief.get("belief_key", "")
        if not new_key or new_key == "generic":
            return None

        new_terms = set(new_belief.get("belief_terms", []))
        if not new_terms:
            return None

        best_id: Optional[str] = None
        best_overlap = 0.0

        # Check existing memories via belief text cache
        for mid, btext in self._belief_text_cache.items():
            # Extract terms from cached belief text
            existing_terms = set(self._tokenize(btext)) - _STOPWORDS
            if not existing_terms:
                continue
            intersection = new_terms & existing_terms
            union = new_terms | existing_terms
            overlap = len(intersection) / max(len(union), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = mid

        if best_overlap >= sim_threshold and best_id is not None:
            return best_id
        return None

    # ---------------------------------------------------------------------
    # Write path: route to Refine or Create, then annotate belief metadata.
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
        # ----- Write routing: check budget pressure -----
        bm = getattr(self, "_budget_manager", None)
        routed_to_refine: Dict[int, str] = {}  # index -> existing memory_id

        if bm is not None:
            try:
                mem_count = self.get_memory_count()
            except Exception:
                mem_count = 0

            if bm.should_prefer_refine(mem_count):
                for i, (td, traj) in enumerate(zip(task_descriptions, trajectories)):
                    similar_id = self._find_similar_memory(td, traj)
                    if similar_id is not None:
                        routed_to_refine[i] = similar_id

        # ----- Separate items into Create vs Refine paths -----
        create_indices = [i for i in range(len(task_descriptions)) if i not in routed_to_refine]

        # Handle Refine-routed items: update existing memory instead of creating new
        refine_results: Dict[str, Optional[str]] = {}
        for i, existing_mid in routed_to_refine.items():
            td = task_descriptions[i]
            traj = trajectories[i]
            succ = successes[i]
            try:
                self.refine_memory(
                    existing_mid,
                    trigger="write_routing",
                    reset_posterior=False,
                    new_trajectory=traj,
                    task_description=td,
                    success=succ,
                )
                refine_results[td] = existing_mid
            except Exception:
                # Fall back to Create if Refine fails
                create_indices.append(i)

        # ----- Create path: delegate to parent for remaining items -----
        if create_indices:
            create_tds = [task_descriptions[i] for i in create_indices]
            create_trajs = [trajectories[i] for i in create_indices]
            create_succs = [successes[i] for i in create_indices]
            create_queries = [
                (retrieved_memory_queries or [None] * len(task_descriptions))[i]
                for i in create_indices
            ]
            create_ids = [
                (retrieved_memory_ids_list or [None] * len(task_descriptions))[i]
                for i in create_indices
            ]
            create_metas = [
                (metadatas or [None] * len(task_descriptions))[i]
                for i in create_indices
            ]

            create_results = super().add_memories(
                task_descriptions=create_tds,
                trajectories=create_trajs,
                successes=create_succs,
                retrieved_memory_queries=create_queries,
                retrieved_memory_ids_list=create_ids,
                metadatas=create_metas,
            )
        else:
            create_results = {}

        # ----- Merge results back into original order -----
        # Normalize create_results
        if create_indices:
            normalized_create = self._normalize_add_results(create_results, [task_descriptions[i] for i in create_indices])
        else:
            normalized_create = []

        # Build combined results
        all_results: Dict[str, Optional[str]] = {}
        create_idx = 0
        for i in range(len(task_descriptions)):
            td = task_descriptions[i]
            if i in routed_to_refine:
                all_results[td] = refine_results.get(td)
            elif create_idx < len(normalized_create):
                _, mid = normalized_create[create_idx]
                all_results[td] = mid
                create_idx += 1

        # ----- Annotate belief metadata on all entries -----
        metas = list(metadatas or [None] * len(task_descriptions))
        for i in range(len(task_descriptions)):
            td = task_descriptions[i]
            memory_id = all_results.get(td)
            if not memory_id:
                continue
            traj = trajectories[i] if i < len(trajectories) else ""
            succ = bool(successes[i]) if i < len(successes) else True
            meta = dict(metas[i] or {})

            belief = self._compose_belief_text(td, traj, meta)
            patch = {
                **belief,
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

        return all_results

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
        conflict_detected = False
        if last_outcome:
            if (last_outcome == "success" and not success) or (last_outcome == "failure" and success):
                conflict += 1.0
                conflict_detected = True

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

        # Check auto-refine trigger: high conflict rate after sufficient reuse
        cfg = self.belief_config
        conflict_rate = conflict / max(reuse, 1.0)
        if (
            conflict_detected
            and reuse >= cfg.auto_refine_min_reuse
            and conflict_rate >= cfg.auto_refine_conflict_threshold
        ):
            self.refine_memory(memory_id, trigger="auto_conflict")

    def _detect_cross_memory_conflicts(
        self,
        retrieved_ids: List[str],
        success: bool,
    ) -> List[Tuple[str, str]]:
        """Detect conflicts between memories with the same belief_key but opposite outcomes.

        Returns list of (memory_id_a, memory_id_b) conflict pairs.
        """
        by_key: Dict[str, List[Tuple[str, str]]] = {}  # belief_key -> [(mem_id, last_outcome)]
        for mid in retrieved_ids:
            if not mid:
                continue
            try:
                item = self._read_memory(mid)
                meta = _meta_to_dict(getattr(item, "metadata", None))
                bkey = str(meta.get("belief_key", ""))
                last = str(meta.get("belief_last_outcome", ""))
                if bkey:
                    by_key.setdefault(bkey, []).append((mid, last))
            except Exception:
                continue

        conflicts: List[Tuple[str, str]] = []
        for bkey, entries in by_key.items():
            successes = [e for e in entries if e[1] == "success"]
            failures = [e for e in entries if e[1] == "failure"]
            if successes and failures:
                conflicts.append((successes[0][0], failures[0][0]))
        return conflicts

    _REFINE_PROMPT_TEMPLATE = (
        "You are revising a stored memory entry to make it more accurate and general.\n\n"
        "## Original Memory Content\n{old_content}\n\n"
        "## Belief Key\n{belief_key}\n\n"
        "{new_evidence_section}"
        "## Instructions\n"
        "Rewrite the memory to incorporate the new evidence. "
        "Keep the same format and level of detail. "
        "If the new trajectory contradicts the old content, prefer the newer information. "
        "Output ONLY the revised memory text, no commentary."
    )

    def _llm_rewrite_content(
        self,
        old_content: str,
        belief_key: str,
        new_trajectory: Optional[str] = None,
        task_description: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> Optional[str]:
        """Call LLM to rewrite memory content with new evidence."""
        llm = getattr(self, "llm_provider", None)
        if llm is None or not callable(getattr(llm, "generate", None)):
            return None

        evidence_parts = []
        if task_description:
            evidence_parts.append(f"Task: {task_description[:500]}")
        if new_trajectory:
            evidence_parts.append(f"New trajectory:\n{new_trajectory[:1500]}")
        if success is not None:
            evidence_parts.append(f"Outcome: {'success' if success else 'failure'}")

        if evidence_parts:
            new_evidence_section = "## New Evidence\n" + "\n".join(evidence_parts) + "\n\n"
        else:
            new_evidence_section = ""

        prompt = self._REFINE_PROMPT_TEMPLATE.format(
            old_content=old_content[:2000],
            belief_key=belief_key[:200],
            new_evidence_section=new_evidence_section,
        )

        try:
            revised = llm.generate([{"role": "user", "content": prompt}], max_tokens=1024)
            revised = str(revised or "").strip()
            if revised and len(revised) > 20:
                return revised
        except Exception:
            pass
        return None

    def refine_memory(
        self,
        memory_id: str,
        trigger: str = "belief_instability",
        reset_posterior: bool = True,
        new_trajectory: Optional[str] = None,
        task_description: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> None:
        """Refine operator: rewrite content and/or reset unstable posterior.

        Called automatically when conflict_rate exceeds threshold, via write
        routing under budget pressure, or explicitly by external intervention.

        Args:
            memory_id: Target memory to refine.
            trigger: What caused the refinement.
            reset_posterior: If True, soft-reset Beta posterior counts.
            new_trajectory: New trajectory to incorporate (for content rewrite).
            task_description: Task description for context (for content rewrite).
            success: Task outcome (for content rewrite).
        """
        item = self._read_memory(memory_id)
        meta = _meta_to_dict(getattr(item, "metadata", None))
        cfg = self.belief_config

        old_alpha = float(meta.get("belief_alpha", cfg.prior_alpha))
        old_beta = float(meta.get("belief_beta", cfg.prior_beta))
        q_val = float(getattr(self, "_q_cache", {}).get(memory_id, 0.0))
        old_content = str(getattr(item, "memory", "") or "")
        belief_key = str(meta.get("belief_key", ""))

        # ----- Content rewrite via LLM -----
        content_rewritten = False
        if new_trajectory or trigger in ("write_routing", "successful_divergence"):
            revised = self._llm_rewrite_content(
                old_content=old_content,
                belief_key=belief_key,
                new_trajectory=new_trajectory,
                task_description=task_description,
                success=success,
            )
            if revised is not None:
                try:
                    text_mem = self._get_text_mem()
                    text_mem.update(memory_id, {
                        "id": memory_id,
                        "memory": revised,
                        "metadata": meta,
                    })
                    self._mem_cache.pop(memory_id, None)
                    content_rewritten = True
                except Exception:
                    pass

        # ----- Posterior soft-reset -----
        patch: Dict[str, Any] = {}
        if reset_posterior:
            new_alpha = max(cfg.prior_alpha, old_alpha * 0.5)
            new_beta = max(cfg.prior_beta, old_beta * 0.5)
            patch["belief_alpha"] = new_alpha
            patch["belief_beta"] = new_beta
            patch["belief_score"] = new_alpha / max(new_alpha + new_beta, 1e-8)
            patch["belief_conflict"] = 0.0

        # Recompute belief key if content was rewritten
        if content_rewritten and task_description:
            new_belief = self._compose_belief_text(task_description, new_trajectory or "")
            patch["belief_key"] = new_belief.get("belief_key", belief_key)
            patch["belief_text"] = new_belief.get("belief_text", "")
            patch["belief_terms"] = new_belief.get("belief_terms", [])

        if patch:
            self._patch_metadata(memory_id, patch)

        # Log the refine event
        _ev_log = getattr(self, "_mem_event_logger", None)
        if _ev_log is not None:
            _ev_log.log_refine(
                memory_id=memory_id,
                trigger=trigger,
                q_value=q_val,
                belief_alpha=float(patch.get("belief_alpha", old_alpha)),
                belief_beta=float(patch.get("belief_beta", old_beta)),
                posterior_reset=reset_posterior,
                content_rewritten=content_rewritten,
            )

    def intervene(
        self,
        memory_id: str,
        operator: str = "refine",
        source: str = "external",
        new_text: Optional[str] = None,
        reset_posterior: bool = False,
        metadata_patch: Optional[Dict[str, Any]] = None,
        redact_patterns: Optional[List[str]] = None,
    ) -> None:
        """External intervention operator: correct a memory with immediate effect.

        This supports the paper's "intervention-ready" property — operators can be
        externally triggered (e.g., by a human or a monitoring system).

        Args:
            memory_id: Target memory to intervene on.
            operator: Type of intervention ("refine", "override", "delete", "redact").
            source: Who triggered it ("external", "human", "monitor").
            new_text: If provided, replace the memory text (for override).
            reset_posterior: If True, reset belief posterior to prior.
            metadata_patch: Additional metadata to patch.
            redact_patterns: Regex patterns to redact (required when operator="redact").
        """
        if operator == "delete":
            self.delete_memories([memory_id], reason=f"intervention:{source}")
        elif operator == "redact":
            self.redact_memories(
                memory_ids=[memory_id],
                patterns=redact_patterns or [],
                source=f"intervention:{source}",
            )
        elif operator == "refine":
            self.refine_memory(memory_id, trigger=f"intervention:{source}", reset_posterior=reset_posterior)
        elif operator == "override" and new_text is not None:
            # Replace memory content
            try:
                text_mem = self._get_text_mem()
                text_mem.update(memory_id, {"id": memory_id, "memory": new_text})
                self._mem_cache.pop(memory_id, None)
            except Exception:
                pass

        # Apply optional metadata patch
        if metadata_patch:
            self._patch_metadata(memory_id, metadata_patch)

        # Log the intervention event
        _ev_log = getattr(self, "_mem_event_logger", None)
        if _ev_log is not None:
            _ev_log.log_intervention(
                memory_id=memory_id,
                operator=operator,
                source=source,
            )

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

            # Cross-memory conflict detection
            try:
                conflicts = self._detect_cross_memory_conflicts(mem_ids or [], ok)
                if conflicts:
                    _ev_log = getattr(self, "_mem_event_logger", None)
                    if _ev_log is not None:
                        for mid_a, mid_b in conflicts:
                            _ev_log.log_update(
                                memory_id=mid_a,
                                q_value=float(self.q_cache.get(mid_a, 0.0)),
                                task_success=ok,
                                belief_conflict=1.0,
                                extra={"cross_conflict_with": mid_b},
                            )
            except Exception:
                pass

        return q_updates

    # ------------------------------------------------------------------
    # Successful divergence detection
    # ------------------------------------------------------------------

    def check_divergence_and_refine(
        self,
        task_descriptions: List[str],
        trajectories: List[str],
        successes: List[bool],
        retrieved_ids_list: List[List[str]],
        divergence_threshold: float = 0.35,
    ) -> int:
        """Check for successful divergence and trigger Refine where needed.

        Called by runners after update_values(). If a task succeeded but the
        actual trajectory is significantly different from the retrieved memory's
        content, the memory should be generalized via Refine.

        Args:
            task_descriptions: Task descriptions for context.
            trajectories: Actual trajectories executed by the agent.
            successes: Whether each task succeeded.
            retrieved_ids_list: Memory IDs that were retrieved for each task.
            divergence_threshold: Maximum belief-key overlap below which
                trajectory is considered divergent from memory.

        Returns:
            Number of memories refined due to divergence.
        """
        refined_count = 0
        for i, (td, traj, succ) in enumerate(zip(task_descriptions, trajectories, successes)):
            if not succ:
                continue  # Only trigger on success
            mem_ids = retrieved_ids_list[i] if i < len(retrieved_ids_list) else []
            if not mem_ids:
                continue

            # Compare trajectory against each retrieved memory's content
            new_terms = set(self._tokenize(traj[:1500])) - _STOPWORDS
            if len(new_terms) < 3:
                continue

            for mid in mem_ids:
                if not mid:
                    continue
                try:
                    item = self._read_memory(str(mid))
                    content = str(getattr(item, "memory", "") or "")
                    if not content:
                        continue
                    mem_terms = set(self._tokenize(content[:1500])) - _STOPWORDS
                    if not mem_terms:
                        continue

                    # Jaccard overlap as divergence measure
                    overlap = len(new_terms & mem_terms) / max(len(new_terms | mem_terms), 1)
                    if overlap < divergence_threshold:
                        # Trajectory diverged significantly from memory — refine
                        self.refine_memory(
                            str(mid),
                            trigger="successful_divergence",
                            reset_posterior=False,
                            new_trajectory=traj,
                            task_description=td,
                            success=True,
                        )
                        refined_count += 1
                except Exception:
                    continue
        return refined_count

    # ------------------------------------------------------------------
    # State-first interface: compile retrieved memories into structured state
    # ------------------------------------------------------------------

    def compile_state(
        self,
        task_description: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Compile current belief state relevant to the given task.

        Instead of returning raw memory text, this method retrieves memories
        and compiles them into a structured state dict grouped by confidence.

        Returns:
            {
                "active_beliefs": [...],    # high-confidence strategies
                "uncertain_beliefs": [...], # low-confidence / high-conflict
                "budget_info": {...},       # utilization stats
                "raw_retrieval": {...},     # original retrieve_query result
            }
        """
        raw = self.retrieve_query(task_description, k=k, threshold=threshold)
        # retrieve_query may return (result_dict, sim_list) tuple
        if isinstance(raw, tuple):
            raw, _ = raw
        raw = dict(raw or {})
        candidates = raw.get("selected", []) or raw.get("candidates", [])

        active = []
        uncertain = []
        variance_threshold = 0.25  # Beta variance threshold for uncertainty

        for c in candidates:
            meta = _meta_to_dict(c.get("metadata", {}))
            memory_id = c.get("memory_id", "")
            content = c.get("content") or meta.get("full_content", "")

            alpha = float(meta.get("belief_alpha", 1.0))
            beta_val = float(meta.get("belief_beta", 1.0))
            posterior_mean = alpha / (alpha + beta_val)
            posterior_var = (alpha * beta_val) / ((alpha + beta_val) ** 2 * (alpha + beta_val + 1))
            reuse = int(float(meta.get("belief_reuse", meta.get("n_reuse", 0))))
            conflict = float(meta.get("belief_conflict", meta.get("n_conflict", 0)))
            conflict_rate = conflict / (reuse + 1) if reuse >= 0 else 0.0
            q_value = float(meta.get("q_value", 0.0))
            belief_key = meta.get("belief_key", meta.get("belief_text", ""))
            success_flag = meta.get("success", None)

            entry = {
                "memory_id": memory_id,
                "belief_key": str(belief_key)[:100],
                "content_summary": str(content)[:500] if content else "",
                "posterior_mean": round(posterior_mean, 3),
                "posterior_var": round(posterior_var, 4),
                "conflict_rate": round(conflict_rate, 3),
                "reuse_count": reuse,
                "q_value": round(q_value, 3),
                "success": success_flag,
            }

            if posterior_var > variance_threshold or conflict_rate > 0.3:
                uncertain.append(entry)
            else:
                active.append(entry)

        # Budget info
        bm = getattr(self, "_budget_manager", None)
        mem_count = self.get_memory_count() if hasattr(self, "get_memory_count") else 0
        budget_info = {
            "total_memories": mem_count,
            "budget": bm.budget if bm else 0,
            "utilization": round(mem_count / bm.budget, 2) if bm and bm.budget > 0 else 0.0,
        }

        return {
            "active_beliefs": active,
            "uncertain_beliefs": uncertain,
            "budget_info": budget_info,
            "raw_retrieval": raw,
        }

    def format_state_prompt(self, state: Dict[str, Any]) -> str:
        """Convert compiled state into a concise text block for the agent prompt.

        Args:
            state: Output of compile_state().

        Returns:
            Formatted string for injection into agent context.
        """
        parts = []
        active = state.get("active_beliefs", [])
        uncertain = state.get("uncertain_beliefs", [])
        budget = state.get("budget_info", {})

        if active:
            parts.append("[OPERATING STATE — Confident strategies]")
            for i, entry in enumerate(active, 1):
                success_str = f"success rate {entry['posterior_mean']:.0%}" if entry["reuse_count"] > 0 else "untested"
                line = (
                    f"  {i}. {entry['belief_key']}: "
                    f"{success_str} ({entry['reuse_count']} uses, Q={entry['q_value']:.2f})"
                )
                if entry["content_summary"]:
                    # Extract first line of content as strategy hint
                    first_line = entry["content_summary"].split("\n")[0][:150]
                    line += f"\n     Strategy: {first_line}"
                parts.append(line)

        if uncertain:
            parts.append("\n[UNCERTAIN — Use with caution]")
            for i, entry in enumerate(uncertain, 1):
                reason = "high conflict" if entry["conflict_rate"] > 0.3 else "low confidence"
                parts.append(
                    f"  {i}. {entry['belief_key']}: "
                    f"{entry['posterior_mean']:.0%} success, {reason} "
                    f"({entry['reuse_count']} uses)"
                )

        if budget.get("budget", 0) > 0:
            parts.append(
                f"\n[BUDGET] {budget['total_memories']}/{budget['budget']} memories "
                f"({budget['utilization']:.0%} utilization)"
            )

        if not active and not uncertain:
            parts.append("[NO RELEVANT MEMORIES]")

        return "\n".join(parts)
