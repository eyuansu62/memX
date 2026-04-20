"""
MemoryAgentBench (MAB) runner for MemRL.

Design:
- For each example: reset memory, stream `haystack_sessions` into memory,
  then answer each question by retrieving + generating.
- Evaluation is per-example (memory does not persist across examples).
- Questions within an example are answered in parallel via ThreadPoolExecutor.

This is a BASELINE-style runner: memory is built from the example's context,
and Q-values can optionally be updated within-example using answer correctness.
No cross-example training loop.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memrl.mab_eval.task_wrappers import (
    SPLIT_NAMES,
    get_sessions,
    iter_qa_pairs,
    load_mab_split,
    score_answer,
    write_samples,
)

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

DEFAULT_SYSTEM_PROMPT = """You are an assistant that answers questions using memory of previously seen context.

You will be given [Retrieved Memory Context] from your memory store (pieces of a long document you have read).
Answer the question based on the memory. Be concise — give only the answer, no extra explanation.

If the memory does not contain enough information, answer with your best guess based on any related context."""


@dataclass
class MABSelection:
    split: str = "Accurate_Retrieval"
    num_examples: Optional[int] = None  # limit examples (None = all)
    max_questions_per_example: Optional[int] = None
    cache_dir: Optional[str] = None


class MABRunner:
    def __init__(
        self,
        *,
        selection: MABSelection,
        llm: Any,
        memory_service: Any,
        output_dir: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        retrieve_k: int = 5,
        retrieve_threshold: Optional[float] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory_budget_tokens: int = 2000,
        update_q_values: bool = True,
        tb_log_dir: Optional[str] = None,
    ) -> None:
        self.sel = selection
        self.llm = llm
        self.mem = memory_service
        self.output_dir = os.path.abspath(output_dir)
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retrieve_k = int(retrieve_k)
        self.retrieve_threshold = (
            None if retrieve_threshold is None else float(retrieve_threshold)
        )
        self.system_prompt = str(system_prompt or "")
        self.memory_budget_tokens = int(memory_budget_tokens)
        self.update_q_values = bool(update_q_values)

        os.makedirs(self.output_dir, exist_ok=True)

        # TensorBoard (optional)
        if tb_log_dir and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))
        else:
            class _NoOpWriter:
                def add_scalar(self, *args: Any, **kwargs: Any) -> None:
                    return
                def close(self) -> None:
                    return
            self.writer = _NoOpWriter()

    # ── Memory helpers ────────────────────────────────────────────────────

    def _get_retrieve_threshold(self) -> float:
        if self.retrieve_threshold is not None:
            return float(self.retrieve_threshold)
        try:
            rl_cfg = getattr(self.mem, "rl_config", None)
            if rl_cfg is None:
                return 0.0
            return float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
        except Exception:
            return 0.0

    def _reset_memory(self) -> None:
        """Clear the memory store before processing a new example."""
        for method in ("clear", "reset", "clear_all"):
            if hasattr(self.mem, method):
                try:
                    getattr(self.mem, method)()
                    return
                except Exception:
                    logger.debug("Memory reset via %s failed", method, exc_info=True)
        logger.warning("MemoryService has no clear/reset method; memory may persist across examples.")

    def _ingest_sessions(self, sessions: List[str], example_idx: int) -> None:
        """Stream sessions into memory as successful trajectories."""
        if not sessions:
            return
        task_descriptions: List[str] = []
        trajectories: List[str] = []
        successes: List[bool] = []
        metadatas: List[Dict[str, Any]] = []
        for i, sess in enumerate(sessions):
            if not sess or not str(sess).strip():
                continue
            # Treat each session as a "successful trajectory": the memory stores the raw content.
            task_descriptions.append(f"MAB context session #{i + 1}")
            trajectories.append(str(sess))
            successes.append(True)
            metadatas.append({
                "source_benchmark": "memoryagentbench",
                "example_idx": example_idx,
                "session_idx": i,
                "outcome": "success",
                "outcome_success": True,
                "source": "haystack",
            })
        if not task_descriptions:
            return
        try:
            self.mem.add_memories(
                task_descriptions=task_descriptions,
                trajectories=trajectories,
                successes=successes,
                retrieved_memory_queries=[None] * len(task_descriptions),
                retrieved_memory_ids_list=[[] for _ in task_descriptions],
                metadatas=metadatas,
            )
        except Exception:
            logger.warning("MAB add_memories failed for example %d", example_idx, exc_info=True)

    def _retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], List[str], Optional[List[Tuple[str, float]]]]:
        """Retrieve memories for a query. Returns (selected_mems, ids, topk_queries)."""
        if self.mem is None or self.retrieve_k <= 0:
            return [], [], None
        try:
            thr = self._get_retrieve_threshold()
            ret = self.mem.retrieve_query(query, k=self.retrieve_k, threshold=thr)
            if isinstance(ret, tuple):
                ret_result, topk_q = ret
            else:
                ret_result, topk_q = ret, None
            selected_mems = (ret_result or {}).get("selected", []) if ret_result else []
            if not isinstance(selected_mems, list):
                selected_mems = []
            ids = [
                str(m.get("memory_id") or m.get("id"))
                for m in selected_mems
                if isinstance(m, dict) and (m.get("memory_id") or m.get("id"))
            ]
            return selected_mems, ids, topk_q
        except Exception:
            logger.debug("MAB retrieval failed", exc_info=True)
            return [], [], None

    def _format_memory_context(self, selected_mems: List[Dict[str, Any]]) -> str:
        if not selected_mems:
            return ""
        parts: List[str] = ["# Retrieved Memory Context"]
        for i, m in enumerate(selected_mems, 1):
            content = str(m.get("content") or m.get("full_content") or "")
            if not content:
                continue
            if len(content) > self.memory_budget_tokens // max(1, len(selected_mems)):
                content = content[: self.memory_budget_tokens // max(1, len(selected_mems))] + "..."
            parts.append(f"## Memory {i}")
            parts.append(content)
            parts.append("")
        return "\n".join(parts)

    # ── LLM ───────────────────────────────────────────────────────────────

    def _generate(self, question: str, memory_context: str) -> str:
        messages: List[Dict[str, str]] = []
        system_parts: List[str] = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        if memory_context:
            system_parts.append(memory_context)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": question})
        try:
            resp = self.llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            logger.warning("MAB generation failed for question: %s...", question[:80], exc_info=True)
            return ""
        return resp or ""

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        examples = load_mab_split(self.sel.split, cache_dir=self.sel.cache_dir)
        if self.sel.num_examples is not None:
            examples = examples[: int(self.sel.num_examples)]

        all_samples: List[Dict[str, Any]] = []
        total_questions = 0
        total_correct = 0
        total_exact = 0

        gen_workers = 8

        for ex_idx, example in enumerate(examples):
            t_start = time.time()
            self._reset_memory()

            sessions = get_sessions(example, max_chars_per_chunk=4000)
            ctx_chars = sum(len(s) for s in sessions)
            logger.info(
                "[mab] example %d/%d: %d sessions, %d chars ctx",
                ex_idx + 1, len(examples), len(sessions), ctx_chars,
            )
            self._ingest_sessions(sessions, example_idx=ex_idx)

            # Collect QA pairs
            qa_pairs = list(iter_qa_pairs(example))
            if self.sel.max_questions_per_example is not None:
                qa_pairs = qa_pairs[: int(self.sel.max_questions_per_example)]

            # Answer all questions in parallel
            ex_results: List[Dict[str, Any]] = [None] * len(qa_pairs)

            def _answer_one(idx: int) -> Tuple[int, Dict[str, Any]]:
                q_i, question, gold, qid = qa_pairs[idx]
                sel_mems, sel_ids, topk = self._retrieve(question)
                mem_ctx = self._format_memory_context(sel_mems)
                prediction = self._generate(question, mem_ctx).strip()
                score = score_answer(prediction, gold)
                return idx, {
                    "example_idx": ex_idx,
                    "q_idx": q_i,
                    "qa_pair_id": qid,
                    "question": question,
                    "gold_answers": gold,
                    "prediction": prediction,
                    "retrieved_ids": sel_ids,
                    "num_retrieved": len(sel_ids),
                    "exact": score["exact"],
                    "substr": score["substr"],
                    "correct": score["correct"],
                    "matched": score["matched"],
                }

            with ThreadPoolExecutor(max_workers=gen_workers) as pool:
                futures = [pool.submit(_answer_one, i) for i in range(len(qa_pairs))]
                for fut in as_completed(futures):
                    i, rec = fut.result()
                    ex_results[i] = rec

            # Optional Q-value updates (within-example learning signal)
            if self.update_q_values and self.mem is not None and hasattr(self.mem, "update_values"):
                successes = [float(r["correct"]) for r in ex_results]
                retrieved_ids = [r["retrieved_ids"] for r in ex_results]
                try:
                    self.mem.update_values(successes, retrieved_ids)
                except Exception:
                    logger.debug("MAB update_values failed", exc_info=True)

            # Aggregate
            ex_correct = sum(1 for r in ex_results if r["correct"])
            ex_exact = sum(1 for r in ex_results if r["exact"])
            total_correct += ex_correct
            total_exact += ex_exact
            total_questions += len(ex_results)
            elapsed = time.time() - t_start
            acc = ex_correct / max(1, len(ex_results))
            logger.info(
                "[mab] example %d done: %d/%d correct (acc=%.3f), %.1fs",
                ex_idx + 1, ex_correct, len(ex_results), acc, elapsed,
            )
            self.writer.add_scalar(f"mab/{self.sel.split}/example_accuracy", acc, global_step=ex_idx)
            all_samples.extend(ex_results)

        # Save results
        samples_path = os.path.join(self.output_dir, "samples.jsonl")
        write_samples(all_samples, samples_path)

        overall = {
            "split": self.sel.split,
            "model": self.model_name,
            "num_examples": len(examples),
            "num_questions": total_questions,
            "num_correct": total_correct,
            "num_exact": total_exact,
            "accuracy": (total_correct / total_questions) if total_questions else 0.0,
            "exact_match": (total_exact / total_questions) if total_questions else 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)

        logger.info(
            "[mab] DONE split=%s examples=%d questions=%d correct=%d acc=%.4f exact=%.4f",
            self.sel.split, len(examples), total_questions, total_correct,
            overall["accuracy"], overall["exact_match"],
        )
        try:
            self.writer.close()
        except Exception:
            pass
        return overall
