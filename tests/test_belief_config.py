"""Tests for BeliefConfigModel and config integration.

These tests verify:
1. BeliefConfigModel <-> BeliefConfig dataclass field parity
2. YAML loading with/without belief section
3. to_dataclass() fallback behavior
4. Config backward compatibility (all YAML configs load cleanly)
"""

import glob
import tempfile
import os

import pytest
import yaml

from memrl.configs.config import MempConfig, BeliefConfigModel


class TestBeliefConfigModel:
    """BeliefConfigModel Pydantic model tests."""

    def test_defaults_match_dataclass(self):
        """Pydantic defaults must match the BeliefConfig dataclass defaults."""
        pydantic_model = BeliefConfigModel()
        # Key defaults that must match belief_memory_service.BeliefConfig
        assert pydantic_model.weight_legacy == 0.45
        assert pydantic_model.weight_belief_similarity == 0.25
        assert pydantic_model.weight_belief_posterior == 0.20
        assert pydantic_model.weight_reuse_bonus == 0.08
        assert pydantic_model.weight_uncertainty_penalty == 0.08
        assert pydantic_model.weight_conflict_penalty == 0.10
        assert pydantic_model.prior_alpha == 1.0
        assert pydantic_model.prior_beta == 1.0
        assert pydantic_model.max_goal_terms == 8
        assert pydantic_model.index_belief_text is True
        assert pydantic_model.dedup_by_belief is True
        assert pydantic_model.auto_refine_conflict_threshold == 0.5
        assert pydantic_model.auto_refine_min_reuse == 3

    def test_partial_override(self):
        """Partial overrides keep remaining fields at defaults."""
        m = BeliefConfigModel(weight_legacy=0.60, auto_refine_min_reuse=5)
        assert m.weight_legacy == 0.60
        assert m.auto_refine_min_reuse == 5
        assert m.weight_belief_similarity == 0.25  # untouched

    def test_to_dataclass_fallback(self):
        """to_dataclass() returns attribute-compatible object even if import fails."""
        m = BeliefConfigModel(weight_legacy=0.7)
        dc = m.to_dataclass()
        # Whether it's the real dataclass or the Pydantic fallback,
        # attribute access must work identically.
        assert dc.weight_legacy == 0.7
        assert dc.auto_refine_min_reuse == 3

    def test_model_dump_roundtrip(self):
        """model_dump -> reconstruct must be lossless."""
        m1 = BeliefConfigModel(weight_conflict_penalty=0.20, prior_alpha=2.0)
        d = m1.model_dump()
        m2 = BeliefConfigModel(**d)
        assert m1 == m2


class TestMempConfigBeliefIntegration:
    """MempConfig loading with belief section."""

    def test_default_belief_section(self):
        """MempConfig without explicit belief section uses defaults."""
        cfg = MempConfig.from_yaml("configs/rl_bcb_config.yaml")
        assert cfg.belief.weight_legacy == 0.45
        assert cfg.belief.auto_refine_conflict_threshold == 0.5

    def test_explicit_belief_override(self):
        """YAML with explicit belief section overrides only specified fields."""
        with open("configs/rl_bcb_config.yaml") as f:
            data = yaml.safe_load(f)
        data["belief"] = {"weight_legacy": 0.60, "auto_refine_min_reuse": 5}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(data, tmp)
            tmp_path = tmp.name
        try:
            cfg = MempConfig.from_yaml(tmp_path)
            assert cfg.belief.weight_legacy == 0.60
            assert cfg.belief.auto_refine_min_reuse == 5
            assert cfg.belief.weight_belief_similarity == 0.25  # default preserved
        finally:
            os.unlink(tmp_path)

    @pytest.mark.parametrize(
        "yaml_path",
        sorted(glob.glob("configs/rl_*.yaml")),
        ids=lambda p: os.path.basename(p),
    )
    def test_all_yaml_configs_load(self, yaml_path):
        """Every shipped YAML config must load without ValidationError."""
        cfg = MempConfig.from_yaml(yaml_path)
        assert cfg.belief is not None
        assert isinstance(cfg.belief, BeliefConfigModel)
