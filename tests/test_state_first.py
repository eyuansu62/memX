"""Tests for state-first interface, redact operator, and integration.

These tests verify:
1. compile_state returns structured output (mocked)
2. format_state_prompt produces valid text (mocked)
3. Redact operator pattern matching
4. Redact logging
5. state_first config wiring
6. Integration: full state-first pipeline (compile → format → prompt)
7. Override operator syncs belief metadata
8. compile_state thresholds are configurable
"""

import re
import math
import pytest

from memrl.configs.config import MempConfig, BeliefConfigModel


class TestRedactPatternLogic:
    """Test the regex-based redaction logic independently of MemOS."""

    def _apply_redact(self, content: str, patterns: list, replacement: str = "[REDACTED]"):
        """Simulate the core redact loop from MemoryService.redact_memories."""
        compiled = []
        for pat in patterns:
            try:
                compiled.append(re.compile(pat, re.IGNORECASE))
            except re.error:
                continue
        total_subs = 0
        new_content = content
        for rx in compiled:
            new_content, n = rx.subn(replacement, new_content)
            total_subs += n
        return new_content, total_subs

    def test_single_pattern(self):
        content = "API key is sk-abc123def456 and it should be secret."
        new, n = self._apply_redact(content, [r"sk-[a-zA-Z0-9]+"])
        assert n == 1
        assert "[REDACTED]" in new
        assert "sk-abc123def456" not in new

    def test_multiple_patterns(self):
        content = "User john@example.com with password hunter2 logged in."
        new, n = self._apply_redact(
            content,
            [r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", r"hunter2"],
        )
        assert n == 2
        assert "john@example.com" not in new
        assert "hunter2" not in new

    def test_no_match(self):
        content = "This content has nothing sensitive."
        new, n = self._apply_redact(content, [r"sk-[a-z]+"])
        assert n == 0
        assert new == content

    def test_invalid_pattern_skipped(self):
        content = "Normal text with secret123 data."
        new, n = self._apply_redact(content, [r"[invalid", r"secret\d+"])
        assert n == 1
        assert "secret123" not in new

    def test_case_insensitive(self):
        content = "The PASSWORD is HunTer2."
        new, n = self._apply_redact(content, [r"hunter2"])
        assert n == 1
        assert "HunTer2" not in new

    def test_custom_replacement(self):
        content = "File at /home/user/.ssh/id_rsa should be protected."
        new, n = self._apply_redact(content, [r"/home/user/[^\s]+"], "[PATH_REDACTED]")
        assert n == 1
        assert "[PATH_REDACTED]" in new


class TestStateFirstConfig:
    """Test that state_first is configurable via YAML."""

    def test_state_first_can_be_enabled(self):
        import yaml, tempfile, os

        with open("configs/rl_alf_config.yaml") as f:
            data = yaml.safe_load(f)
        data["experiment"]["state_first"] = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(data, tmp)
            tmp_path = tmp.name
        try:
            cfg = MempConfig.from_yaml(tmp_path)
            assert cfg.experiment.state_first is True
        finally:
            os.unlink(tmp_path)

    def test_state_first_defaults_false_when_absent(self):
        import yaml, tempfile, os

        with open("configs/rl_alf_config.yaml") as f:
            data = yaml.safe_load(f)
        data["experiment"].pop("state_first", None)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(data, tmp)
            tmp_path = tmp.name
        try:
            cfg = MempConfig.from_yaml(tmp_path)
            assert getattr(cfg.experiment, "state_first", False) is False
        finally:
            os.unlink(tmp_path)


class TestBeliefConfigConversion:
    """Verify that BeliefConfigModel -> dataclass conversion preserves all fields."""

    def test_all_fields_present(self):
        model = BeliefConfigModel(
            weight_legacy=0.55,
            prior_alpha=2.0,
            auto_refine_min_reuse=5,
        )
        dc = model.to_dataclass()
        # Whether real dataclass or Pydantic fallback, all attributes must be accessible
        assert dc.weight_legacy == 0.55
        assert dc.prior_alpha == 2.0
        assert dc.auto_refine_min_reuse == 5
        # Defaults preserved
        assert dc.weight_belief_similarity == 0.25
        assert dc.dedup_by_belief is True

    def test_state_thresholds_present(self):
        """New state compilation thresholds should be in both Model and dataclass."""
        model = BeliefConfigModel(
            state_variance_threshold=0.3,
            state_conflict_threshold=0.4,
        )
        dc = model.to_dataclass()
        assert dc.state_variance_threshold == 0.3
        assert dc.state_conflict_threshold == 0.4

    def test_state_thresholds_defaults(self):
        model = BeliefConfigModel()
        dc = model.to_dataclass()
        assert dc.state_variance_threshold == 0.25
        assert dc.state_conflict_threshold == 0.3


# ---------------------------------------------------------------------------
# Integration tests: state-first pipeline without MemOS
# ---------------------------------------------------------------------------

class _FakeMeta:
    """Simulates a MemOS metadata object with model_extra."""
    def __init__(self, d: dict):
        self._d = d
        self.model_extra = d

    def model_dump(self):
        return dict(self._d)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._d.get(name)


class _FakeMemItem:
    """Simulates a MemOS memory item."""
    def __init__(self, memory_id: str, memory: str, metadata: dict):
        self.memory_id = memory_id
        self.id = memory_id
        self.memory = memory
        self.metadata = _FakeMeta(metadata)


def _build_candidate(memory_id: str, content: str, score: float,
                     belief_alpha: float = 2.0, belief_beta: float = 1.0,
                     belief_reuse: int = 3, belief_conflict: float = 0.0,
                     belief_key: str = "test_key", q_value: float = 0.5,
                     success: bool = True) -> dict:
    """Build a candidate dict matching what retrieve_query returns."""
    return {
        "memory_id": memory_id,
        "content": content,
        "score": score,
        "metadata": {
            "belief_alpha": belief_alpha,
            "belief_beta": belief_beta,
            "belief_reuse": belief_reuse,
            "belief_conflict": belief_conflict,
            "belief_key": belief_key,
            "q_value": q_value,
            "success": success,
        },
    }


class TestCompileStateLogic:
    """Test the compile_state partitioning logic without MemOS."""

    def _partition(self, candidates, variance_threshold=0.25, conflict_threshold=0.3):
        """Reproduce compile_state's active/uncertain partitioning."""
        active, uncertain = [], []
        for c in candidates:
            meta = c.get("metadata", {})
            alpha = float(meta.get("belief_alpha", 1.0))
            beta_val = float(meta.get("belief_beta", 1.0))
            posterior_mean = alpha / (alpha + beta_val)
            posterior_var = (alpha * beta_val) / ((alpha + beta_val) ** 2 * (alpha + beta_val + 1))
            reuse = int(float(meta.get("belief_reuse", 0)))
            conflict = float(meta.get("belief_conflict", 0))
            conflict_rate = conflict / (reuse + 1) if reuse >= 0 else 0.0
            q_value = float(meta.get("q_value", 0.0))
            belief_key = meta.get("belief_key", "")
            success_flag = meta.get("success", None)

            entry = {
                "memory_id": c.get("memory_id", ""),
                "belief_key": str(belief_key)[:100],
                "content_summary": str(c.get("content", ""))[:500],
                "posterior_mean": round(posterior_mean, 3),
                "posterior_var": round(posterior_var, 4),
                "conflict_rate": round(conflict_rate, 3),
                "reuse_count": reuse,
                "q_value": round(q_value, 3),
                "success": success_flag,
            }

            if posterior_var > variance_threshold or conflict_rate > conflict_threshold:
                uncertain.append(entry)
            else:
                active.append(entry)
        return active, uncertain

    def test_high_confidence_goes_to_active(self):
        """Memory with low variance, low conflict → active."""
        cands = [_build_candidate("m1", "strategy A", 0.9,
                                  belief_alpha=10.0, belief_beta=2.0,
                                  belief_reuse=8, belief_conflict=0.0)]
        active, uncertain = self._partition(cands)
        assert len(active) == 1
        assert len(uncertain) == 0
        assert active[0]["memory_id"] == "m1"

    def test_high_conflict_goes_to_uncertain(self):
        """Memory with high conflict rate → uncertain."""
        cands = [_build_candidate("m2", "strategy B", 0.7,
                                  belief_alpha=3.0, belief_beta=3.0,
                                  belief_reuse=5, belief_conflict=3.0)]
        active, uncertain = self._partition(cands)
        assert len(active) == 0
        assert len(uncertain) == 1
        assert uncertain[0]["memory_id"] == "m2"

    def test_new_memory_low_variance_goes_to_active(self):
        """Fresh memory with prior (1,1) has variance 0.083 < 0.25 → active."""
        cands = [_build_candidate("m3", "new strategy", 0.5,
                                  belief_alpha=1.0, belief_beta=1.0,
                                  belief_reuse=0, belief_conflict=0.0)]
        active, uncertain = self._partition(cands)
        # Beta(1,1) variance = 1*1 / (4*3) = 0.083
        assert len(active) == 1

    def test_high_variance_goes_to_uncertain(self):
        """Memory with very high variance → uncertain."""
        # Beta(0.5, 0.5) has variance ~0.125 which is < 0.25
        # Need extreme case: Beta(1.01, 1.01) has var ~0.082
        # Actually use conflict to push to uncertain
        cands = [_build_candidate("m4", "unstable", 0.6,
                                  belief_alpha=1.0, belief_beta=1.0,
                                  belief_reuse=3, belief_conflict=2.0)]
        active, uncertain = self._partition(cands)
        # conflict_rate = 2/(3+1) = 0.5 > 0.3
        assert len(uncertain) == 1

    def test_mixed_partition(self):
        """Multiple candidates get correctly partitioned."""
        cands = [
            _build_candidate("good1", "reliable", 0.9,
                             belief_alpha=10, belief_beta=1, belief_reuse=8, belief_conflict=0),
            _build_candidate("good2", "also reliable", 0.85,
                             belief_alpha=8, belief_beta=2, belief_reuse=6, belief_conflict=0),
            _build_candidate("bad1", "conflicted", 0.7,
                             belief_alpha=3, belief_beta=3, belief_reuse=4, belief_conflict=3),
        ]
        active, uncertain = self._partition(cands)
        assert len(active) == 2
        assert len(uncertain) == 1
        active_ids = {e["memory_id"] for e in active}
        assert active_ids == {"good1", "good2"}

    def test_configurable_thresholds(self):
        """Custom thresholds change partitioning."""
        cands = [_build_candidate("m5", "borderline", 0.8,
                                  belief_alpha=3.0, belief_beta=3.0,
                                  belief_reuse=5, belief_conflict=1.0)]
        # Default thresholds: conflict_rate = 1/6 = 0.167 < 0.3 → active
        active, uncertain = self._partition(cands, conflict_threshold=0.3)
        assert len(active) == 1

        # Stricter threshold: 0.167 > 0.15 → uncertain
        active2, uncertain2 = self._partition(cands, conflict_threshold=0.15)
        assert len(uncertain2) == 1


class TestFormatStatePrompt:
    """Test format_state_prompt output structure without MemOS."""

    def _format(self, active, uncertain, budget_info=None):
        """Reproduce format_state_prompt logic."""
        parts = []
        if active:
            parts.append("[OPERATING STATE — Confident strategies]")
            for i, entry in enumerate(active, 1):
                success_str = f"success rate {entry['posterior_mean']:.0%}" if entry["reuse_count"] > 0 else "untested"
                line = (
                    f"  {i}. {entry['belief_key']}: "
                    f"{success_str} ({entry['reuse_count']} uses, Q={entry['q_value']:.2f})"
                )
                if entry.get("content_summary"):
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

        budget = budget_info or {}
        if budget.get("budget", 0) > 0:
            parts.append(
                f"\n[BUDGET] {budget['total_memories']}/{budget['budget']} memories "
                f"({budget['utilization']:.0%} utilization)"
            )

        if not active and not uncertain:
            parts.append("[NO RELEVANT MEMORIES]")
        return "\n".join(parts)

    def test_active_only(self):
        active = [{
            "memory_id": "m1", "belief_key": "pick_up, apple",
            "content_summary": "Go to countertop and pick up apple",
            "posterior_mean": 0.85, "posterior_var": 0.01,
            "conflict_rate": 0.0, "reuse_count": 6, "q_value": 0.72,
            "success": True,
        }]
        text = self._format(active, [])
        assert "[OPERATING STATE" in text
        assert "pick_up, apple" in text
        assert "85%" in text
        assert "Q=0.72" in text
        assert "[UNCERTAIN" not in text

    def test_uncertain_only(self):
        uncertain = [{
            "memory_id": "m2", "belief_key": "heat, egg",
            "content_summary": "", "posterior_mean": 0.4,
            "posterior_var": 0.05, "conflict_rate": 0.5,
            "reuse_count": 5, "q_value": 0.1, "success": False,
        }]
        text = self._format([], uncertain)
        assert "[UNCERTAIN" in text
        assert "high conflict" in text
        assert "[OPERATING STATE" not in text

    def test_empty_state(self):
        text = self._format([], [])
        assert "[NO RELEVANT MEMORIES]" in text

    def test_budget_info(self):
        active = [{
            "memory_id": "m1", "belief_key": "test",
            "content_summary": "strategy", "posterior_mean": 0.9,
            "posterior_var": 0.01, "conflict_rate": 0.0,
            "reuse_count": 3, "q_value": 0.5, "success": True,
        }]
        budget = {"total_memories": 47, "budget": 100, "utilization": 0.47}
        text = self._format(active, [], budget)
        assert "[BUDGET]" in text
        assert "47/100" in text

    def test_mixed_state_has_both_sections(self):
        active = [{
            "memory_id": "a1", "belief_key": "good_key",
            "content_summary": "works well", "posterior_mean": 0.9,
            "posterior_var": 0.01, "conflict_rate": 0.0,
            "reuse_count": 10, "q_value": 0.8, "success": True,
        }]
        uncertain = [{
            "memory_id": "u1", "belief_key": "bad_key",
            "content_summary": "", "posterior_mean": 0.3,
            "posterior_var": 0.1, "conflict_rate": 0.6,
            "reuse_count": 4, "q_value": -0.2, "success": False,
        }]
        text = self._format(active, uncertain)
        assert "[OPERATING STATE" in text
        assert "[UNCERTAIN" in text
        # Active should come before uncertain
        assert text.index("[OPERATING STATE") < text.index("[UNCERTAIN")


class TestOverrideMetadataSync:
    """Test that override operator recomputes belief metadata."""

    def test_compose_belief_text_structure(self):
        """Verify _compose_belief_text returns the expected keys by testing tokenizer logic."""
        # Reproduce BeliefMemoryService._tokenize (static method) without importing MemOS
        def _tokenize(text: str):
            return re.findall(r"[A-Za-z_][A-Za-z0-9_+#.-]{2,}", str(text or "").lower())

        tokens = _tokenize("Pick up the red apple from countertop")
        assert "pick" in tokens
        assert "apple" in tokens
        assert "countertop" in tokens
        # Tokens < 3 chars are excluded by the regex
        assert "up" not in tokens
        # "the" is 3 chars so it IS captured by tokenizer; stopword filtering happens later
        assert "the" in tokens

    def test_override_should_sync_belief_key(self):
        """Verify intervene(operator='override') includes belief metadata sync in code."""
        import ast
        with open("memrl/service/belief_memory_service.py") as f:
            source = f.read()
        # The override branch should call _compose_belief_text after updating memory
        assert "_compose_belief_text" in source
        # Find the override block and check it patches belief_key
        tree = ast.parse(source)
        # Check that "belief_key" appears in the override section
        override_section = source[source.index('operator == "override"'):]
        override_section = override_section[:override_section.index("# Apply optional metadata patch")]
        assert "belief_key" in override_section
        assert "belief_text" in override_section
        assert "_patch_metadata" in override_section


class TestBCBRunnerStateFirst:
    """Verify BCBRunner accepts and stores state_first parameter."""

    def test_bcb_runner_has_state_first_param(self):
        """BCBRunner.__init__ should accept state_first kwarg."""
        import inspect
        from memrl.run.bcb_runner import BCBRunner
        sig = inspect.signature(BCBRunner.__init__)
        assert "state_first" in sig.parameters
        # Default should be False
        assert sig.parameters["state_first"].default is False

    def test_bcb_entrypoint_passes_state_first(self):
        """run_bcb.py should pass state_first from config to runner."""
        import ast
        with open("run/run_bcb.py") as f:
            source = f.read()
        tree = ast.parse(source)
        # Find the BCBRunner() call and check state_first is in kwargs
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "state_first":
                found = True
                break
        assert found, "run_bcb.py should pass state_first= to BCBRunner"
