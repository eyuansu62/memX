# FILE: memp/agent/memp_agent.py

import logging
import re
from typing import List, Dict, Any, Optional
import copy
import ast

from .base import BaseAgent
from .history import EpisodeHistory
from . import prompts
from memrl.providers.llm import OpenAILLM

logger = logging.getLogger(__name__)

# Compiled regex for structural delimiters in memory content
_TRAJECTORY_DELIM = re.compile(r'\n\nTRAJECTORY:\n')
_FAILED_APPROACH_DELIM = re.compile(r'\n\nFailed approach:\n')
_ADJUSTMENT_NOTE_DELIM = re.compile(r'\n\n--- ADJUSTMENT NOTE \(.*?\) ---\n')

class MempAgent(BaseAgent):
    """
    A stateless agent that uses an LLM to make decisions.
    It receives all necessary context (history, retrieved memories) from an
    external controller (the Runner) at the moment of action.
    """
    def __init__(self, llm_provider: OpenAILLM, few_shot_examples: Dict[str, Any]):
        # The agent is now independent of the memory service.
        self.llm = llm_provider
        self.few_shot_examples = few_shot_examples
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

    def reset(self, task_description: str) -> None:
        """Resets the agent for a new episode and retrieves relevant long-term memories."""
        self.task_description = task_description.strip()
        logger.info(f"Agent has been reset for new task: '{self.task_description}'")
        
    def _get_examples_for_task(self, task_type: str) -> str:
        """
        [NEW] Selects the relevant few-shot examples based on the task type.
        """
        for prefix, key in self.prefixes.items():
            if task_type.startswith(prefix):
                # This logic mirrors your example script: load two relevant examples
                for example in self.few_shot_examples:
                    if example['task'] == key:
                        return copy.deepcopy(example['example'])
        return "No specific examples found for this task type."

    def _extract_list_from_str(self, text: str) -> Optional[list]:
        """Find the first top-level [...] in *text* by bracket counting,
        then parse it with ast.literal_eval.  Returns the parsed list or None."""
        start = text.find('[')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        quote_char = None
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if in_string:
                if ch == quote_char:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        result = ast.literal_eval(candidate)
                        if isinstance(result, list):
                            return result
                    except Exception:
                        pass
                    return None
        return None

    def _format_trajectory_list(self, trajectory_list: list) -> str:
        """Format a parsed trajectory list into clean text for the agent."""
        # Keep only the portion after the last "Now, it's your turn"
        turn_idx = -1
        for i, msg in enumerate(trajectory_list):
            if (isinstance(msg, dict)
                    and msg.get("role") == "user"
                    and isinstance(msg.get("content", ""), str)
                    and "Now, it's your turn" in msg["content"]):
                turn_idx = i
        if turn_idx != -1:
            trajectory_list = trajectory_list[turn_idx:]

        clean_trajectory = []
        for message in trajectory_list:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if role == "assistant":
                clean_trajectory.append(f"> {content}")
            elif role == "user" and content.strip().startswith("Observation:"):
                clean_trajectory.append(content)

        if clean_trajectory:
            return "Archived Trajectory:\n" + "\n".join(clean_trajectory)
        return ""

    def _format_retrieved_memory(self, raw_content: str) -> str:
        """
        Parses the raw memory content to extract only the most useful parts
        (SCRIPT and the core Thought/Action/Observation sequence), removing
        redundant headers, system prompts, and old task descriptions.

        Handles five memory content formats:
        1. Proceduralization  (SCRIPT + TRAJECTORY delimiter)
        2. Adjustment/Reflection  (What went wrong + Failed approach delimiter)
        3. Raw trajectory  (Python list string, no delimiter)
        4. Script only  (plain text, no trajectory list)
        5. Inplace adjustment  (TRAJECTORY + trailing ADJUSTMENT NOTE)
        """
        if not raw_content or not raw_content.strip():
            return ""

        clean_parts = []
        trajectory_list = None

        # --- Format 1 & 5: SCRIPT + TRAJECTORY delimiter ---
        traj_match = _TRAJECTORY_DELIM.search(raw_content)
        if traj_match:
            header = raw_content[:traj_match.start()]
            trajectory_region = raw_content[traj_match.end():]

            # Strip trailing adjustment notes (Format 5)
            adj_match = _ADJUSTMENT_NOTE_DELIM.search(trajectory_region)
            if adj_match:
                trajectory_region = trajectory_region[:adj_match.start()]

            if 'SCRIPT:' in header:
                script_part = header.split('SCRIPT:', 1)[1].strip()
                clean_parts.append(f"Archived Script:\n{script_part}")

            trajectory_list = self._extract_list_from_str(trajectory_region)

        # --- Format 2: TASK REFLECTION + Failed approach ---
        elif _FAILED_APPROACH_DELIM.search(raw_content):
            fa_match = _FAILED_APPROACH_DELIM.search(raw_content)
            header = raw_content[:fa_match.start()]
            trajectory_region = raw_content[fa_match.end():]

            if 'What went wrong:' in header:
                reflection_part = header.split('What went wrong:', 1)[1].strip()
                clean_parts.append(f"Archived Reflection:\n{reflection_part}")

            trajectory_list = self._extract_list_from_str(trajectory_region)

        # --- Format 3: Raw trajectory (no delimiter, list present) ---
        elif '[' in raw_content:
            trajectory_list = self._extract_list_from_str(raw_content)

        # --- Format trajectory if found ---
        if trajectory_list and isinstance(trajectory_list, list):
            formatted = self._format_trajectory_list(trajectory_list)
            if formatted:
                clean_parts.append(formatted)
        elif not clean_parts:
            # Format 4 or completely unparseable: return cleaned raw content
            cleaned = raw_content.strip()
            if cleaned.startswith('Task:'):
                lines = cleaned.split('\n', 1)
                if len(lines) > 1:
                    cleaned = lines[1].strip()
            if cleaned:
                clean_parts.append(f"Archived Note:\n{cleaned}")

        return "\n\n".join(clean_parts) if clean_parts else ""
        
    def _construct_messages(self, task_description: str, retrieved_memories: List[Dict], task_type: str) -> List[Dict[str, str]]:
        """
        [REFACTORED]
        Builds the message list in a conversational ReAct style.
        """
        # 1. Start with the system prompt
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]

        # 2. Add the selected few-shot example as a complete dialogue
        example_dialogue = self._get_examples_for_task(task_type)
        if example_dialogue:
            # Modify the first user message in the example to introduce it
            example_dialogue[0]['content'] = "Here is an example of how to solve the task:\n" + example_dialogue[0]['content']
            messages.extend(example_dialogue)

        # 3. Add retrieved memories as additional context for the agent
        if retrieved_memories:
            successful_mems = retrieved_memories.get('successed', [])
            failed_mems = retrieved_memories.get('failed', [])

            successful_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in successful_mems
            ] if successful_mems else []

            failed_mems_formatted = [
                self._format_retrieved_memory(mem['content']) for mem in failed_mems
            ] if failed_mems else []

            memory_parts = [
                "In addition to the example, you have the following memories from your own past experiences. "
                "Use them to help you if they are relevant:"
            ]

            if successful_mems_formatted:
                memory_parts.append(
                    "--- SUCCESSFUL MEMORIES (Examples to follow) ---\n" +
                    "\n".join(successful_mems_formatted)
                )

            if failed_mems_formatted:
                memory_parts.append(
                    "--- FAILED MEMORIES (Examples to avoid or learn from) ---\n" +
                    "\n".join(failed_mems_formatted)
                )

            if successful_mems_formatted or failed_mems_formatted:
                memory_context = "\n\n".join(memory_parts)
                messages.append({"role": "system", "content": memory_context})

        # 4. Add the current task description as the new user prompt
        # The history of the current task will be appended in the `act` method
        current_task_prompt = f"Now, it's your turn to solve a new task.\n{task_description}"
        messages.append({"role": "user", "content": current_task_prompt})
        # logger.info(f"\nPrompt {messages}")
        return messages

    def _construct_messages_state_first(
        self,
        task_description: str,
        state_prompt: str,
        task_type: str,
        raw_memories: dict = None,
        belief_annotations: dict = None,
    ) -> List[Dict[str, str]]:
        """Build message list with belief-annotated memories (v2).

        Single system message for memories (same structure as
        ``_construct_messages``), but each memory is prefixed with a
        one-line belief annotation and memories are sorted by confidence.
        """
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]

        # Few-shot example
        example_dialogue = self._get_examples_for_task(task_type)
        if example_dialogue:
            example_dialogue[0]['content'] = "Here is an example of how to solve the task:\n" + example_dialogue[0]['content']
            messages.extend(example_dialogue)

        # Build annotated memory context (single system message)
        if raw_memories:
            annotations = belief_annotations or {}
            successful_mems = raw_memories.get('successed', [])
            failed_mems = raw_memories.get('failed', [])

            def _sort_key(mem):
                mid = mem.get("memory_id", "")
                ann = annotations.get(mid)
                return ann["sort_key"] if ann else 0.0

            memory_parts = [
                "In addition to the example, you have the following memories from your own past experiences. "
                "Use them to help you if they are relevant:"
            ]

            if successful_mems:
                sorted_mems = sorted(successful_mems, key=_sort_key, reverse=True)
                formatted = []
                for mem in sorted_mems:
                    content = self._format_retrieved_memory(mem.get('content', ''))
                    if not content:
                        continue
                    ann = annotations.get(mem.get("memory_id", ""))
                    if ann:
                        tag = "CONFIDENT" if ann["confidence"] == "confident" else "UNCERTAIN"
                        prefix = f"[{tag}] {ann['label']}\n"
                    else:
                        prefix = ""
                    formatted.append(prefix + content)
                if formatted:
                    memory_parts.append(
                        "--- SUCCESSFUL MEMORIES (Examples to follow) ---\n" +
                        "\n".join(formatted)
                    )

            if failed_mems:
                sorted_mems = sorted(failed_mems, key=_sort_key, reverse=True)
                formatted = []
                for mem in sorted_mems:
                    content = self._format_retrieved_memory(mem.get('content', ''))
                    if not content:
                        continue
                    ann = annotations.get(mem.get("memory_id", ""))
                    if ann:
                        tag = "CONFIDENT" if ann["confidence"] == "confident" else "UNCERTAIN"
                        prefix = f"[{tag}] {ann['label']}\n"
                    else:
                        prefix = ""
                    formatted.append(prefix + content)
                if formatted:
                    memory_parts.append(
                        "--- FAILED MEMORIES (Examples to avoid or learn from) ---\n" +
                        "\n".join(formatted)
                    )

            if len(memory_parts) > 1:
                memory_context = "\n\n".join(memory_parts)
                messages.append({"role": "system", "content": memory_context})

        # Current task
        current_task_prompt = f"Now, it's your turn to solve a new task.\n{task_description}"
        messages.append({"role": "user", "content": current_task_prompt})
        return messages

    def _construct_messages_resolved(
        self,
        task_description: str,
        compiled_state: str,
        task_type: str,
    ) -> List[Dict[str, str]]:
        """Build message list with LLM-compiled state view.

        The agent reads resolved state entries (current strategies, pitfalls,
        uncertainties) instead of a list of raw memory entries.  This is the
        core of the state-first interface: the agent reads *what currently
        holds*, not a memory archive.
        """
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]

        # Few-shot example
        example_dialogue = self._get_examples_for_task(task_type)
        if example_dialogue:
            example_dialogue[0]['content'] = "Here is an example of how to solve the task:\n" + example_dialogue[0]['content']
            messages.extend(example_dialogue)

        # Compiled state as memory context (single system message)
        if compiled_state and compiled_state.strip():
            memory_context = (
                "Based on your past experiences, here is what is currently known "
                "about tasks like this. Use this to guide your actions:\n\n"
                + compiled_state
            )
            messages.append({"role": "system", "content": memory_context})

        # Current task
        current_task_prompt = f"Now, it's your turn to solve a new task.\n{task_description}"
        messages.append({"role": "user", "content": current_task_prompt})
        return messages

    def _parse_action(self, llm_response: str) -> str:
        """
        Extracts the 'Action:' part from the ReAct response.
        """
        if llm_response:
            if "Action:" in llm_response:
                return llm_response.split("Action:")[-1].strip()
            # Fallback if the model doesn't follow the format correctly
            logger.warning(f"Could not find 'Action:' in LLM response. Returning the full response: '{llm_response}'")
            return llm_response.strip()
        else:
            return 'look around'
    def act(self, observation: str, history_messages: List[Dict[str, str]], first_step: bool = False):
        """
        Agent performs one step of action generation.
        Ensures robustness: if LLM fails or returns invalid output, action=None is returned.
        """
        import json

        current_messages = copy.deepcopy(history_messages)
        if not first_step:
            current_messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})

        filtered_messages = []
        for i, m in enumerate(current_messages):
            if m.get("content") is None:
                logger.warning(f"[Message Filter] Message {i} has None content, removed: {m}")
                continue
            if isinstance(m.get("content"), str) and not m["content"].strip():
                logger.warning(f"[Message Filter] Message {i} has empty content, removed: {m}")
                continue
            filtered_messages.append(m)
        current_messages = filtered_messages

        logger.debug("Querying LLM for the next action...")

        response = None
        try:
            response = self.llm.generate(current_messages)
        except Exception as e:
            logger.error("LLM generation failed: %s", str(e))
            logger.error("Messages before failure:\n%s", json.dumps(current_messages, indent=2, ensure_ascii=False))
            response = None  # fallback

        if not first_step:
            history_messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})
        history_messages.append({"role": "assistant", "content": response if response is not None else "No response."})

        action = None
        if response:
            try:
                action = self._parse_action(response)
            except Exception as e:
                logger.warning(f"Action parsing failed for response='{response}': {e}")
                action = "inventory"

        return action



    def get_trajectory(self) -> List[Dict[str, str]]:
        """Returns the complete trajectory for the finished episode."""
        pass
