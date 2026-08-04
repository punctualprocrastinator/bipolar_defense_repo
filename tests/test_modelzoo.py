"""Registry tests.

These pin the *verified* geometry so a future edit cannot quietly reintroduce the
`hidden_size / num_attention_heads` assumption that breaks Gemma.
"""

from __future__ import annotations

import pytest

from bsc import modelzoo


class TestRegistry:
    def test_lookup(self):
        assert modelzoo.get("qwen2.5-7b").hf_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_unknown_key_rejected(self):
        with pytest.raises(KeyError, match="unknown model"):
            modelzoo.get("gpt-5")

    def test_ungated_models_available(self):
        keys = {m.key for m in modelzoo.ungated()}
        assert {"qwen2.5-7b", "qwen2.5-1.5b", "olmo2-7b", "ministral-8b"} <= keys

    def test_every_model_has_notes(self):
        assert all(m.notes for m in modelzoo.MODELS.values())


class TestGeometryTraps:
    """The reason this registry exists."""

    @pytest.mark.parametrize("key", ["gemma2-9b", "gemma3-4b"])
    def test_gemma_head_dim_differs_from_derived(self, key):
        spec = modelzoo.get(key)
        assert spec.head_dim_matches_derived is False, (
            f"{key} is registered as a geometry trap; if this now matches, re-verify the config"
        )

    def test_gemma2_exact_values(self):
        spec = modelzoo.get("gemma2-9b")
        assert (spec.head_dim, spec.derived_head_dim) == (256, 224)

    def test_gemma3_derived_is_larger_than_real(self):
        # Opposite direction from gemma-2: derived 320 > real 256, so o_proj input is narrower
        # than hidden_size and naive slicing runs off the end of the tensor.
        spec = modelzoo.get("gemma3-4b")
        assert spec.derived_head_dim == 320
        assert spec.head_dim == 256
        assert spec.num_attention_heads * spec.head_dim < spec.hidden_size

    @pytest.mark.parametrize("key", ["qwen2.5-7b", "qwen2.5-1.5b", "olmo2-7b", "ministral-8b"])
    def test_safe_models_derive_correctly(self, key):
        spec = modelzoo.get(key)
        effective = spec.head_dim or spec.derived_head_dim
        assert effective == spec.derived_head_dim
        assert effective == 128

    def test_olmo_is_mha_not_gqa(self):
        # The control for "is the bipolar circuit an artifact of GQA?"
        spec = modelzoo.get("olmo2-7b")
        assert spec.is_gqa is False
        assert spec.num_key_value_heads == spec.num_attention_heads

    @pytest.mark.parametrize("key", ["qwen2.5-7b", "llama3.1-8b", "gemma2-9b", "ministral-8b"])
    def test_others_are_gqa(self, key):
        assert modelzoo.get(key).is_gqa is True


class TestMemory:
    def test_all_verified_models_fit_on_blackwell(self):
        # 95 GB card. Weights alone must leave headroom for activations, KV cache, and GCG's
        # candidate batch — assert a generous margin rather than just "fits".
        for spec in modelzoo.verified():
            assert spec.bf16_weight_gb() < 40, f"{spec.key} weights too large for comfort"

    def test_weight_estimate(self):
        assert modelzoo.get("qwen2.5-7b").bf16_weight_gb() == pytest.approx(15.2)
