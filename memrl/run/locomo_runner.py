"""LoComo benchmark runner for MemRL.

LoComo (Long-term Conversational Memory) tests whether a memory system can
answer questions about long, multi-session conversations.  The RL loop works
as follows:

1. **Seed phase** – For each training conversation, session observations and
   summaries are ingested as episodic memories via ``memory_service.add_memories``.
2. **QA phase (per section)** – Each QA item becomes a *task*:
   retrieve relevant memories → generate a short answer → evaluate with token-F1.
3. **Update** – Q-values on retrieved memory IDs are updated with the F1 reward.
4. **Eval** – Held-out conversation QAs are evaluated (no memory update).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .base_runner import BaseRunner
from memrl.providers.llm import OpenAILLM
from memrl.service.memory_service import MemoryService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation helpers – thin wrappers around LoComo's own metric code.
# ---------------------------------------------------------------------------

try:
    import sys as _sys
    _locomo_root = str(Path(__file__).resolve().parent.parent.parent / "3rdparty" / "locomo")
    if _locomo_root not in _sys.path:
        _sys.path.insert(0, _locomo_root)
    from task_eval.evaluation import f1_score as _locomo_f1_single, f1 as _locomo_f1_multi, normalize_answer
    _HAS_LOCOMO_EVAL = True
except Exception:
    _HAS_LOCOMO_EVAL = False

    def normalize_answer(s: str) -> str:
        import string
        s = s.replace(",", "")
        s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        return " ".join(s.split())

    def _locomo_f1_single(pred: str, gold: str) -> float:
        from collections import Counter
        pred_toks = normalize_answer(pred).split()
        gold_toks = normalize_answer(gold).split()
        common = Counter(pred_toks) & Counter(gold_toks)
        n = sum(common.values())
        if n == 0:
            return 0.0
        p = n / len(pred_toks)
        r = n / len(gold_toks)
        return 2 * p * r / (p + r)

    def _locomo_f1_multi(pred: str, gold: str) -> float:
        import numpy as np
        preds = [p.strip() for p in pred.split(",")]
        golds = [g.strip() for g in gold.split(",")]
        return float(np.mean([max(_locomo_f1_single(p, g) for p in preds) for g in golds]))


def _compute_f1(prediction: str, ground_truth: str, category: int) -> float:
    """Return token-F1 following LoComo category rules."""
    if category == 5:
        low = prediction.lower()
        if "no information available" in low or "not mentioned" in low:
            return 1.0
        return 0.0
    if category == 1:
        return _locomo_f1_multi(prediction, ground_truth)
    # Categories 2, 3, 4 – standard single-answer F1
    if category == 3:
        ground_truth = ground_truth.split(";")[0].strip()
    return _locomo_f1_single(prediction, ground_truth)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class LoCoMoSelection:
    data_path: str = ""
    num_train_conv: Optional[int] = None
    num_valid_conv: Optional[int] = None
    categories: Optional[List[int]] = None  # filter QAs to these categories


def _load_locomo_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_conversation(conv: Dict[str, Any]) -> str:
    """Flatten a LoComo conversation dict into readable text."""
    session_nums = sorted(
        int(k.split("_")[1])
        for k in conv.keys()
        if k.startswith("session_") and not k.endswith("date_time")
    )
    lines: List[str] = []
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    lines.append(f"Conversation between {speaker_a} and {speaker_b}.\n")
    for sn in session_nums:
        dt = conv.get(f"session_{sn}_date_time", "")
        lines.append(f"\n--- Session {sn} ({dt}) ---")
        for turn in conv.get(f"session_{sn}", []):
            lines.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(lines)


def _extract_observations(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (observation_text, dia_id) from the observation field."""
    obs_dict = sample.get("observation", {})
    results: List[Tuple[str, str]] = []
    for key in sorted(obs_dict.keys()):
        per_speaker = obs_dict[key]
        if isinstance(per_speaker, dict):
            for speaker, obs_list in per_speaker.items():
                for item in obs_list:
                    if isinstance(item, list) and len(item) >= 2:
                        results.append((f"[{speaker}] {item[0]}", item[1]))
                    elif isinstance(item, str):
                        results.append((f"[{speaker}] {item}", ""))
    return results


def _extract_summaries(sample: Dict[str, Any]) -> List[str]:
    """Return session summaries as a list of strings."""
    summ_dict = sample.get("session_summary", {})
    return [str(v) for _, v in sorted(summ_dict.items())]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LoCoMoRunner(BaseRunner):
    """LoComo benchmark runner — QA over long multi-session conversations."""

    F1_SUCCESS_THRESHOLD = 0.5  # F1 >= this counts as "success" for RL

    def __init__(
        self,
        name: str,
        llm: OpenAILLM,
        selection: LoCoMoSelection,
        output_dir: Path,
        memory_service: Optional[MemoryService] = None,
        run_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        retrieve_k: int = 5,
        num_sections: int = 5,
        batch_size: int = 16,
        dataset_ratio: float = 1.0,
        random_seed: int = 42,
        train_valid_split: float = 0.8,
        ckpt_eval_enabled: bool = False,
        ckpt_eval_path: Optional[str] = None,
        ckpt_resume_enabled: bool = False,
        ckpt_resume_path: Optional[str] = None,
        ckpt_resume_epoch: Optional[int] = None,
        baseline_mode: Optional[str] = None,
        baseline_k: int = 10,
    ) -> None:
        self.name = name
        self.llm = llm
        self.sel = selection
        self.output_dir = Path(output_dir)
        self.memory_service = memory_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieve_k = max(0, int(retrieve_k))
        self.num_sections = num_sections
        self.batch_size = max(1, int(batch_size))
        self.dataset_ratio = float(dataset_ratio)
        self.random_seed = random_seed
        self.train_valid_split = float(train_valid_split)
        self.ckpt_eval_enabled = bool(ckpt_eval_enabled)
        self.ckpt_eval_path = str(ckpt_eval_path) if ckpt_eval_path else None
        self.ckpt_resume_enabled = ckpt_resume_enabled
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch
        self.baseline_mode = (baseline_mode or "").strip().lower() or None
        self.baseline_k = max(1, int(baseline_k))

        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.ck_dir = self.output_dir / "locomo" / f"exp_{self.name}_{self.run_id}"
        self.log_dir = self.ck_dir / "local_cache"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = self.output_dir / "tensorboard" / f"exp_locomo_{self.name}_{self.run_id}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info("TensorBoard logs at: %s", tb_dir)

        self.llm_log_path = self.log_dir / "llm_calls.jsonl"
        self._log_lock = threading.Lock()

        # Will be populated by _load()
        self.train_samples: List[Dict[str, Any]] = []
        self.valid_samples: List[Dict[str, Any]] = []

        self.train_cumulative_correct_map: Dict[str, bool] = {}
        self.valid_cumulative_correct_map: Dict[str, bool] = {}

    # ------------------------------------------------------------------ data

    def _load(self) -> Tuple[List[Dict], List[Dict]]:
        raw = _load_locomo_data(self.sel.data_path)
        rng = random.Random(self.random_seed)
        rng.shuffle(raw)

        # Apply dataset_ratio
        if 0 < self.dataset_ratio < 1:
            n = max(1, int(len(raw) * self.dataset_ratio))
            raw = raw[:n]

        # Split conversations into train / valid
        n_train = max(1, int(len(raw) * self.train_valid_split))
        self.train_samples = raw[:n_train]
        self.valid_samples = raw[n_train:]

        # Optional category filtering on QAs
        if self.sel.categories:
            cats = set(self.sel.categories)
            for s in self.train_samples + self.valid_samples:
                s["qa"] = [q for q in s["qa"] if q.get("category") in cats]

        n_train_qa = sum(len(s["qa"]) for s in self.train_samples)
        n_valid_qa = sum(len(s["qa"]) for s in self.valid_samples)
        logger.info(
            "LoComo loaded: %d conversations (%d train / %d valid), %d train QAs, %d valid QAs",
            len(raw), len(self.train_samples), len(self.valid_samples), n_train_qa, n_valid_qa,
        )
        return self.train_samples, self.valid_samples

    # ---------------------------------------------------------- memory seeding

    def _seed_memories_from_conversation(self, sample: Dict[str, Any]) -> None:
        """Ingest conversation observations and summaries as episodic memories."""
        if not self.memory_service:
            return
        conv_id = sample.get("sample_id", "unknown")

        # 1) Observations — fine-grained facts per dialog turn
        observations = _extract_observations(sample)
        if observations:
            task_descs = [f"[conv:{conv_id}] {obs}" for obs, _ in observations]
            trajectories = [f"Observation: {obs}\nEvidence: {dia_id}" for obs, dia_id in observations]
            successes = [True] * len(observations)
            metadatas = [
                {
                    "source_benchmark": "LoComo",
                    "source_type": "observation",
                    "conv_id": conv_id,
                    "success": True,
                    "q_value": 0.0,
                    "q_visits": 0,
                    "q_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "last_used_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "reward_ma": 0.0,
                }
            ] * len(observations)
            try:
                self.memory_service.add_memories(
                    task_descriptions=task_descs,
                    trajectories=trajectories,
                    successes=successes,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.warning("Failed to seed observations for conv %s: %s", conv_id, e)

        # 2) Summaries — session-level abstractions
        summaries = _extract_summaries(sample)
        if summaries:
            task_descs = [f"[conv:{conv_id}] Session summary {i+1}" for i in range(len(summaries))]
            trajectories = [f"Session Summary:\n{s}" for s in summaries]
            successes = [True] * len(summaries)
            metadatas = [
                {
                    "source_benchmark": "LoComo",
                    "source_type": "summary",
                    "conv_id": conv_id,
                    "success": True,
                    "q_value": 0.0,
                    "q_visits": 0,
                    "q_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "last_used_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "reward_ma": 0.0,
                }
            ] * len(summaries)
            try:
                self.memory_service.add_memories(
                    task_descriptions=task_descs,
                    trajectories=trajectories,
                    successes=successes,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.warning("Failed to seed summaries for conv %s: %s", conv_id, e)

    # -------------------------------------------------------------- QA logic

    QA_SYSTEM_PROMPT = (
        "You are a conversational memory assistant.  You have access to retrieved "
        "memories from past conversations.  Answer the question in a short phrase. "
        "Answer with exact words from the memories whenever possible.  "
        "If the information is not available, say 'No information available'."
    )

    def _build_memory_context(self, selected_mems: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        if not selected_mems:
            return "", []
        retrieved_ids: List[str] = []
        blocks: List[str] = []
        for m in selected_mems[: self.retrieve_k]:
            mid = m.get("memory_id") or m.get("id")
            if mid:
                retrieved_ids.append(str(mid))
            content = m.get("content") or m.get("full_content") or ""
            blocks.append(content)
        ctx = "=== Retrieved Memories ===\n" + "\n\n".join(blocks)
        return ctx, retrieved_ids

    def _build_messages(
        self,
        question: str,
        category: int,
        memory_ctx: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Category-specific question formatting (following LoComo conventions)
        if category == 2:
            question += " Use dates from the conversation to answer with an approximate date."
        elif category == 5:
            question += " If the information is not mentioned, say 'No information available'."

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": self.QA_SYSTEM_PROMPT}]
        if memory_ctx:
            msgs.append({"role": "system", "content": memory_ctx})
        msgs.append({"role": "user", "content": question})
        return msgs

    def _log_llm_call(self, call_type: str, messages: Any, response: Any, meta: Optional[Dict] = None) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": call_type,
            "meta": meta or {},
            "messages": messages,
            "response": response,
        }
        try:
            payload = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            return
        try:
            with self._log_lock:
                with open(self.llm_log_path, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
        except Exception:
            pass

    def _evaluate_qa(self, qa: Dict[str, Any], conv_id: str) -> Dict[str, Any]:
        """Evaluate a single QA item: retrieve → generate → score."""
        question = str(qa["question"])
        gold = str(qa["answer"])
        category = int(qa.get("category", 4))

        # Retrieve
        memory_ctx = None
        retrieved_ids: List[str] = []
        retrieved_topk_queries = None
        if self.memory_service and self.retrieve_k > 0:
            try:
                rl_cfg = getattr(self.memory_service, "rl_config", None)
                tau = float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
            except Exception:
                tau = 0.0
            try:
                ret = self.memory_service.retrieve_query(question, k=self.retrieve_k, threshold=tau)
                if isinstance(ret, tuple):
                    ret_result, retrieved_topk_queries = ret
                else:
                    ret_result, retrieved_topk_queries = ret, None
                selected = ret_result.get("selected", []) if ret_result else []
                memory_ctx, retrieved_ids = self._build_memory_context(selected)
            except Exception as e:
                logger.warning("Memory retrieval failed for QA: %s", e)

        # Generate
        messages = self._build_messages(question, category, memory_ctx)
        output = ""
        try:
            output = self.llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error("LLM error: %s", e)

        self._log_llm_call("qa", messages, output, meta={"conv_id": conv_id, "category": category})

        # Score
        f1 = _compute_f1(output or "", gold, category)
        success = f1 >= self.F1_SUCCESS_THRESHOLD

        return {
            "conv_id": conv_id,
            "question": question,
            "gold": gold,
            "category": category,
            "prediction": (output or "").strip(),
            "f1": float(f1),
            "success": bool(success),
            "retrieved_ids": retrieved_ids,
            "retrieved_topk_queries": retrieved_topk_queries,
            "trajectory": f"QUESTION\n{question}\n\nANSWER\n{(output or '').strip()}\n",
        }

    # -------------------------------------------------------- section training

    def _flatten_qas(self, samples: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Return list of (conv_id, qa_dict) for all samples."""
        items: List[Tuple[str, Dict[str, Any]]] = []
        for s in samples:
            cid = str(s.get("sample_id", "unknown"))
            for qa in s.get("qa", []):
                items.append((cid, qa))
        return items

    def _qa_key(self, conv_id: str, question: str) -> str:
        return f"{conv_id}::{question}"

    def _train_one_section(self, qa_items: List[Tuple[str, Dict]], sec_idx: int) -> Dict[str, float]:
        n = len(qa_items)
        if n == 0:
            return {"acc": 0.0, "avg_f1": 0.0}

        idxs = list(range(n))
        batches = [idxs[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        all_recs: List[Dict[str, Any]] = []
        processed = 0
        correct_so_far = 0
        f1_sum = 0.0

        for b in tqdm(batches, desc=f"Training Section {sec_idx}/{self.num_sections}"):
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(b)
            with ThreadPoolExecutor(max_workers=min(len(b), self.batch_size)) as ex:
                fut2pos = {}
                for pos, i in enumerate(b):
                    conv_id, qa = qa_items[i]
                    fut2pos[ex.submit(self._evaluate_qa, qa, conv_id)] = pos
                for fut in as_completed(fut2pos):
                    pos = fut2pos[fut]
                    try:
                        batch_results[pos] = fut.result()
                    except Exception as e:
                        logger.warning("[train sec %d] QA eval failed at pos %d: %s", sec_idx, pos, e)

            batch_recs = [r for r in batch_results if r is not None]
            all_recs.extend(batch_recs)
            processed += len(batch_recs)
            correct_so_far += sum(1 for r in batch_recs if r.get("success"))
            f1_sum += sum(r.get("f1", 0.0) for r in batch_recs)
            acc_so_far = correct_so_far / max(1, processed)
            avg_f1_so_far = f1_sum / max(1, processed)
            logger.info(
                "[train sec %d] %d/%d | Acc: %.2f%% | Avg F1: %.4f",
                sec_idx, processed, n, acc_so_far * 100, avg_f1_so_far,
            )

            # Memory update
            if self.memory_service and batch_recs:
                try:
                    task_descriptions = [r["question"] for r in batch_recs]
                    trajectories = [r["trajectory"] for r in batch_recs]
                    successes = [r["success"] for r in batch_recs]
                    retrieved_ids_list = [r.get("retrieved_ids") or [] for r in batch_recs]
                    retrieved_queries = [r.get("retrieved_topk_queries") for r in batch_recs]
                    # Use F1 as continuous reward
                    rewards = [r.get("f1", 0.0) for r in batch_recs]
                    metadatas = [
                        {
                            "source_benchmark": "LoComo",
                            "source_type": "qa_attempt",
                            "conv_id": r["conv_id"],
                            "category": r["category"],
                            "success": r["success"],
                            "f1": r["f1"],
                            "q_value": r["f1"],
                            "q_visits": 0,
                            "q_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "last_used_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "reward_ma": 0.0,
                        }
                        for r in batch_recs
                    ]
                    self.memory_service.update_values(
                        [float(r["success"]) for r in batch_recs],
                        retrieved_ids_list,
                        rewards=rewards,
                    )
                    self.memory_service.add_memories(
                        task_descriptions=task_descriptions,
                        trajectories=trajectories,
                        successes=successes,
                        retrieved_memory_queries=retrieved_queries,
                        retrieved_memory_ids_list=retrieved_ids_list,
                        metadatas=metadatas,
                    )
                except Exception as e:
                    logger.warning("[train sec %d] memory update failed: %s", sec_idx, e)

        if not all_recs:
            return {"acc": 0.0, "avg_f1": 0.0}

        acc = correct_so_far / len(all_recs)
        avg_f1 = f1_sum / len(all_recs)
        logger.info("Section %d Train — Acc: %.2f%%, Avg F1: %.4f", sec_idx, acc * 100, avg_f1)

        try:
            self.writer.add_scalar("Train/Section_Acc", acc, sec_idx)
            self.writer.add_scalar("Train/Section_AvgF1", avg_f1, sec_idx)
            self.writer.flush()
        except Exception:
            pass

        # Save checkpoint
        if self.memory_service:
            ckpt_meta = self.memory_service.save_checkpoint_snapshot(self.ck_dir, ckpt_id=sec_idx)
            logger.info("Saved ckpt: %s", ckpt_meta)

        # Save per-item results
        out_path = self.log_dir / f"train_sec_{sec_idx}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_recs, f, ensure_ascii=False, indent=2, default=str)

        return {
            "acc": float(acc),
            "avg_f1": float(avg_f1),
            "per_item": {self._qa_key(r["conv_id"], r["question"]): r["success"] for r in all_recs},
        }

    # -------------------------------------------------------------- eval

    def _eval_split(
        self, qa_items: List[Tuple[str, Dict]], tag: str, step: Optional[int] = None,
    ) -> Dict[str, Any]:
        n = len(qa_items)
        if n == 0:
            logger.warning("No QAs for %s; skip.", tag)
            return {"acc": 0.0, "avg_f1": 0.0}

        results: List[Dict[str, Any]] = []
        correct = 0
        f1_sum = 0.0
        idxs = list(range(n))
        batches = [idxs[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        processed = 0

        for b in tqdm(batches, desc=f"Evaluating {tag}"):
            batch_results: List[Optional[Dict]] = [None] * len(b)
            with ThreadPoolExecutor(max_workers=min(len(b), self.batch_size)) as ex:
                fut2pos = {}
                for pos, i in enumerate(b):
                    conv_id, qa = qa_items[i]
                    fut2pos[ex.submit(self._evaluate_qa, qa, conv_id)] = pos
                for fut in as_completed(fut2pos):
                    pos = fut2pos[fut]
                    try:
                        batch_results[pos] = fut.result()
                    except Exception as e:
                        logger.warning("[%s] eval failed at pos %d: %s", tag, pos, e)
            batch_valid = [r for r in batch_results if r is not None]
            results.extend(batch_valid)
            processed += len(batch_valid)
            correct += sum(1 for r in batch_valid if r.get("success"))
            f1_sum += sum(r.get("f1", 0.0) for r in batch_valid)
            logger.info(
                "[%s] %d/%d | Acc: %.2f%% | Avg F1: %.4f",
                tag, processed, n, correct / max(1, processed) * 100, f1_sum / max(1, processed),
            )

        acc = correct / max(1, len(results))
        avg_f1 = f1_sum / max(1, len(results))
        logger.info("[%s] Done — Acc: %.2f%%, Avg F1: %.4f (%d items)", tag, acc * 100, avg_f1, n)

        # Per-category breakdown
        by_cat: Dict[int, List[float]] = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r["f1"])
        for cat in sorted(by_cat):
            vals = by_cat[cat]
            cat_f1 = sum(vals) / len(vals)
            cat_acc = sum(1 for v in vals if v >= self.F1_SUCCESS_THRESHOLD) / len(vals)
            logger.info("  Cat %d: n=%d, Avg F1=%.4f, Acc=%.2f%%", cat, len(vals), cat_f1, cat_acc * 100)
            try:
                self.writer.add_scalar(f"{tag}/Cat{cat}_F1", cat_f1, step or 0)
            except Exception:
                pass

        try:
            if step is not None:
                self.writer.add_scalar(f"Evaluation/{tag}_Acc", acc, step)
                self.writer.add_scalar(f"Evaluation/{tag}_AvgF1", avg_f1, step)
            self.writer.flush()
        except Exception:
            pass

        out_path = self.log_dir / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        return {
            "acc": float(acc),
            "avg_f1": float(avg_f1),
            "per_item": {self._qa_key(r["conv_id"], r["question"]): r["success"] for r in results},
        }

    # ----------------------------------------------------------------- run

    def run(self, num_episodes: int = 0):
        train_samples, valid_samples = self._load()

        # Seed memories from training conversations
        logger.info("Seeding memories from %d training conversations...", len(train_samples))
        for s in tqdm(train_samples, desc="Seeding memories"):
            self._seed_memories_from_conversation(s)
        logger.info("Memory seeding complete.")

        train_qas = self._flatten_qas(train_samples)
        valid_qas = self._flatten_qas(valid_samples)

        # Shuffle train QAs with fixed seed
        rng = random.Random(self.random_seed)
        rng.shuffle(train_qas)

        # Initial eval on valid set
        if valid_qas:
            valid_res = self._eval_split(valid_qas, tag="valid_initial", step=0)
            for k, v in valid_res.get("per_item", {}).items():
                self.valid_cumulative_correct_map[k] = bool(v)

        # Section loop
        for sec_idx in range(1, self.num_sections + 1):
            logger.info("=== Section %d/%d ===", sec_idx, self.num_sections)

            if train_qas:
                res = self._train_one_section(train_qas, sec_idx)
                for k, v in res.get("per_item", {}).items():
                    if k not in self.train_cumulative_correct_map:
                        self.train_cumulative_correct_map[k] = False
                    if v:
                        self.train_cumulative_correct_map[k] = True

                total = len(self.train_cumulative_correct_map)
                cum_correct = sum(1 for x in self.train_cumulative_correct_map.values() if x)
                cum_acc = cum_correct / max(1, total)
                logger.info(
                    "[Train] Cumulative Acc after section %d: %.2f%% (%d/%d)",
                    sec_idx, cum_acc * 100, cum_correct, total,
                )
                try:
                    self.writer.add_scalar("Train/Cumulative_Acc", cum_acc, sec_idx)
                except Exception:
                    pass

            # Valid eval
            if valid_qas:
                valid_res = self._eval_split(valid_qas, tag=f"valid_sec_{sec_idx}", step=sec_idx)
                for k, v in valid_res.get("per_item", {}).items():
                    if k not in self.valid_cumulative_correct_map:
                        self.valid_cumulative_correct_map[k] = False
                    if v:
                        self.valid_cumulative_correct_map[k] = True

                total_v = len(self.valid_cumulative_correct_map)
                cum_v = sum(1 for x in self.valid_cumulative_correct_map.values() if x)
                cum_v_acc = cum_v / max(1, total_v)
                logger.info(
                    "[Valid] Cumulative Acc after section %d: %.2f%% (%d/%d)",
                    sec_idx, cum_v_acc * 100, cum_v, total_v,
                )
                try:
                    self.writer.add_scalar("Valid/Cumulative_Acc", cum_v_acc, sec_idx)
                except Exception:
                    pass

        try:
            self.writer.close()
        except Exception:
            pass
