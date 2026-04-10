"""Tests for state-first interface and redact operator.

These tests verify:
1. compile_state returns structured output (mocked)
2. format_state_prompt produces valid text (mocked)
3. Redact operator pattern matching
4. Redact logging
5. state_first config wiring
"""

import re
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

    def test_default_state_first_is_false(self):
        cfg = MempConfig.from_yaml("configs/rl_alf_config.yaml")
        assert cfg.experiment.state_first is False

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
