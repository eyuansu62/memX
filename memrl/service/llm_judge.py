"""
ALFWorld LLM-as-Judge

Evaluates agent trajectories against task goals and returns a continuous
score in [-1, 1].  Designed to blend with the environment's binary reward.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are evaluating an AI agent's performance on a household task.

Task goal: {task}

Agent trajectory (actions and observations):
{trajectory}

Environment result: {env_result}

Score the agent's performance on a scale from -1.0 to 1.0:
  1.0  — Task fully completed, efficient approach
  0.5  — Task completed but with unnecessary steps or minor errors
  0.0  — Significant partial progress (found key objects, reached subgoals) but did not finish
 -0.5  — Minimal progress, mostly wrong actions
 -1.0  — No meaningful progress or completely wrong approach

Respond with a JSON object only:
{{
  "score": <float between -1.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}}"""

_MAX_TRAJ_CHARS = 3000


def _format_trajectory(trajectory: List[Dict]) -> str:
    """Condense a message-list trajectory to a readable string."""
    lines = []
    for msg in trajectory:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role in ("user", "assistant") and content:
            prefix = "Obs:" if role == "user" else "Act:"
            lines.append(f"{prefix} {content.strip()}")
    full = "\n".join(lines)
    if len(full) > _MAX_TRAJ_CHARS:
        half = _MAX_TRAJ_CHARS // 2
        full = full[:half] + "\n...[truncated]...\n" + full[-half:]
    return full


class ALFWorldJudge:
    """LLM judge for ALFWorld trajectories."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def judge(
        self,
        task: str,
        trajectory: List[Dict],
        env_success: bool,
    ) -> Dict:
        """
        Judge a single trajectory.

        Returns:
            {"score": float, "reasoning": str}
        """
        traj_str = _format_trajectory(trajectory)
        env_result = "SUCCESS (environment confirmed task complete)" if env_success else "FAILURE (environment did not confirm task complete)"
        prompt = _JUDGE_PROMPT.format(
            task=task,
            trajectory=traj_str,
            env_result=env_result,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.llm.generate(messages, temperature=0.0, max_tokens=256)
        except Exception as e:
            logger.warning("LLM judge call failed: %s", e)
            return {"score": 1.0 if env_success else -1.0, "reasoning": "judge unavailable"}

        return self._parse(raw, env_success)

    def _parse(self, raw: str, env_success: bool) -> Dict:
        fallback = {"score": 1.0 if env_success else -1.0, "reasoning": raw[:120]}
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return fallback
            data = json.loads(m.group(0))
            score = float(data.get("score", fallback["score"]))
            score = max(-1.0, min(1.0, score))
            return {"score": score, "reasoning": str(data.get("reasoning", ""))}
        except Exception:
            # Try regex fallback
            m = re.search(r'"score"\s*:\s*(-?[\d.]+)', raw)
            if m:
                score = max(-1.0, min(1.0, float(m.group(1))))
                return {"score": score, "reasoning": raw[:120]}
            return fallback

    def judge_batch(
        self,
        tasks: List[str],
        trajectories: List[List[Dict]],
        env_successes: List[bool],
        max_workers: int = 8,
    ) -> List[Dict]:
        """Judge a batch of trajectories in parallel."""
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.judge, task, traj, succ): i
                for i, (task, traj, succ) in enumerate(
                    zip(tasks, trajectories, env_successes)
                )
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.warning("Judge batch slot %d failed: %s", i, e)
                    results[i] = {
                        "score": 1.0 if env_successes[i] else -1.0,
                        "reasoning": "judge error",
                    }
        return results
