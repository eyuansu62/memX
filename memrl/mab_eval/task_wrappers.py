"""
MemoryAgentBench (MAB) task wrappers.

MAB splits:
  - Accurate_Retrieval (AR)        : 22 examples
  - Test_Time_Learning (TTL)       :  6 examples
  - Long_Range_Understanding (LRU) : 110 examples
  - Conflict_Resolution (CR)       :  8 examples

Each example fields:
  - context    : long concatenated text
  - questions  : list[str]  (60-100 questions)
  - answers    : list[list[str]]  (multiple valid answers per question)
  - metadata   : dict, including 'haystack_sessions' (list[str] — context split by session),
                 'qa_pair_ids', 'keypoints', ...
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

SPLIT_NAMES = (
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
)


def load_mab_split(split: str, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load a MemoryAgentBench split from HuggingFace (auto-download on first use).

    Returns a list of example dicts with keys: context, questions, answers, metadata.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown MAB split: {split}. Valid: {SPLIT_NAMES}")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Loading MemoryAgentBench requires the 'datasets' package. "
            "Install via: pip install datasets"
        ) from e
    ds = load_dataset("ai-hyz/MemoryAgentBench", split=split, cache_dir=cache_dir)
    examples: List[Dict[str, Any]] = []
    for ex in ds:
        examples.append({
            "context": ex.get("context", "") or "",
            "questions": list(ex.get("questions", []) or []),
            "answers": list(ex.get("answers", []) or []),
            "metadata": dict(ex.get("metadata", {}) or {}),
        })
    logger.info("Loaded MAB split=%s: %d examples", split, len(examples))
    return examples


def get_sessions(example: Dict[str, Any], *, max_chars_per_chunk: int = 4000) -> List[str]:
    """Return the list of context chunks (sessions) to be streamed into memory.

    Prefers `metadata.haystack_sessions` when present (already split by the benchmark);
    otherwise falls back to chunking the raw context by character length.
    """
    meta = example.get("metadata") or {}
    sessions = meta.get("haystack_sessions")
    if isinstance(sessions, list) and len(sessions) > 0:
        return [str(s) for s in sessions if s is not None]

    # Fallback: character-based chunking
    ctx = str(example.get("context", "") or "")
    if not ctx:
        return []
    if max_chars_per_chunk <= 0 or len(ctx) <= max_chars_per_chunk:
        return [ctx]
    return [ctx[i : i + max_chars_per_chunk] for i in range(0, len(ctx), max_chars_per_chunk)]


# ── Scoring ───────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = str(text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def score_answer(predicted: str, gold_answers: List[str]) -> Dict[str, Any]:
    """Score a single prediction against a list of acceptable gold answers.

    Returns dict with:
      - exact   : True if any normalized gold exactly matches normalized prediction
      - substr  : True if any normalized gold appears as substring of normalized prediction
      - correct : exact OR substr (the default 'pass' signal)
      - matched : the gold string that matched (or None)
    """
    pred_norm = _normalize(predicted)
    exact = False
    substr = False
    matched: Optional[str] = None
    for g in gold_answers or []:
        g_norm = _normalize(g)
        if not g_norm:
            continue
        if pred_norm == g_norm:
            exact = True
            matched = g
            break
        if g_norm in pred_norm:
            substr = True
            matched = g
    return {
        "exact": exact,
        "substr": substr,
        "correct": exact or substr,
        "matched": matched,
    }


def iter_qa_pairs(example: Dict[str, Any]) -> Iterator[Tuple[int, str, List[str], str]]:
    """Yield (idx, question, gold_answers, qa_pair_id) over a single example."""
    qs = example.get("questions") or []
    ans = example.get("answers") or []
    qa_ids = (example.get("metadata") or {}).get("qa_pair_ids") or []
    n = min(len(qs), len(ans))
    for i in range(n):
        qid = str(qa_ids[i]) if i < len(qa_ids) else f"qa_{i}"
        gold = ans[i]
        if not isinstance(gold, list):
            gold = [str(gold)] if gold is not None else []
        yield i, str(qs[i]), [str(x) for x in gold], qid


def write_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Append samples to a JSONL file (create parent dirs as needed)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
