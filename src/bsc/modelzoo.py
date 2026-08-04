"""Verified model registry for cross-architecture circuit discovery.

The paper's generalisation claim needs more than one model family. Geometry below was read from
each model's actual HF config on the Blackwell node (see ``verified`` field), not assumed --
because the assumption ``head_dim == hidden_size / num_attention_heads`` is false for some of
these, and the legacy code hardcodes it everywhere (``CLAIMS_AUDIT.md`` / CLAUDE.md §3.1).

Why these models:

* **Qwen2.5 7B / 1.5B** — the existing results. Same family, two scales.
* **Llama-3.1-8B** — the cross-family transfer test already on the roadmap
  (``RESEARCH_OVERVIEW.md`` §3.1). Different pretraining, different safety tuning, same GQA shape.
* **OLMo-2-7B** — **fully open weights *and* open training data**, and the only candidate with
  **MHA rather than GQA** (32 query heads, 32 KV heads). That makes it the control for a
  question the Qwen-only results cannot answer: is the bipolar circuit a property of safety
  tuning, or an artifact of grouped-query attention?
* **Gemma-2-9B** — the geometry stress test. ``head_dim=256`` while
  ``hidden_size/num_heads = 3584/16 = 224``. Any code assuming the derived value slices the
  wrong columns and silently produces noise instead of a circuit.
* **Ministral-8B** — carries an explicit ``head_dim`` that happens to equal the derived value,
  so it distinguishes "reads head_dim correctly" from "gets lucky".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A candidate model and what we know about running it."""

    key: str
    hf_name: str
    family: str
    params_b: float
    n_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None  # explicit config value; None means derive
    gated: bool = False
    verified: bool = False
    notes: str = ""

    @property
    def derived_head_dim(self) -> int | None:
        if self.hidden_size is None or self.num_attention_heads is None:
            return None
        return self.hidden_size // self.num_attention_heads

    @property
    def head_dim_matches_derived(self) -> bool | None:
        """False means code assuming ``hidden/n_heads`` will slice the wrong columns."""
        if self.head_dim is None or self.derived_head_dim is None:
            return None
        return self.head_dim == self.derived_head_dim

    @property
    def is_gqa(self) -> bool | None:
        if self.num_key_value_heads is None or self.num_attention_heads is None:
            return None
        return self.num_key_value_heads != self.num_attention_heads

    def bf16_weight_gb(self) -> float:
        """Weights only, bf16. Activations, KV cache, and GCG's candidate batch are extra."""
        return self.params_b * 2.0


MODELS: dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec(
        key="qwen2.5-7b",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        params_b=7.6,
        n_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=None,
        verified=True,
        notes="Primary model for existing results. Derived head_dim=128 is correct.",
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen",
        params_b=1.5,
        n_layers=28,
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=None,
        verified=True,
        notes="Small-scale replication. Derived head_dim=128 is correct.",
    ),
    "llama3.1-8b": ModelSpec(
        key="llama3.1-8b",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        family="llama",
        params_b=8.0,
        n_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,  # explicit, equals derived
        gated=True,
        verified=True,
        notes=(
            "Cross-family transfer test (RESEARCH_OVERVIEW.md 3.1). Gated but accessible with "
            "an HF token whose account accepted the Llama license."
        ),
    ),
    "llama3.2-3b": ModelSpec(
        key="llama3.2-3b",
        hf_name="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        params_b=3.2,
        n_layers=28,
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,
        head_dim=128,
        gated=True,
        verified=True,
        notes="Small Llama for cheap scale comparison within the family.",
    ),
    "olmoe-1b-7b": ModelSpec(
        key="olmoe-1b-7b",
        hf_name="allenai/OLMoE-1B-7B-0924-Instruct",
        family="olmoe",
        params_b=6.9,  # 7B total params, ~1B active per token
        n_layers=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,  # MHA
        head_dim=128,
        gated=False,
        verified=True,
        notes=(
            "Sparse Mixture-of-Experts (64 experts, 8 active), fully open weights + data. "
            "ATTENTION IS DENSE (MHA) -- the MoE routing is in the MLP only, so the bipolar "
            "attention-head methodology transfers unchanged. Tests whether the refusal/compliance "
            "head structure appears in a sparse-MoE model. The expert-routing question (is "
            "compliance routed through specific experts?) is a separate, larger study."
        ),
    ),
    "olmo2-7b": ModelSpec(
        key="olmo2-7b",
        hf_name="allenai/OLMo-2-1124-7B-Instruct",
        family="olmo",
        params_b=7.3,
        n_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,  # MHA, not GQA
        head_dim=None,
        gated=False,
        verified=True,
        notes=(
            "Ungated, fully open training data. MHA (32 Q / 32 KV) -- the only candidate "
            "without GQA, so it isolates whether the bipolar circuit depends on GQA."
        ),
    ),
    "gemma2-9b": ModelSpec(
        key="gemma2-9b",
        hf_name="google/gemma-2-9b-it",
        family="gemma",
        params_b=9.2,
        n_layers=42,
        hidden_size=3584,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,  # VERIFIED: derived would be 3584/16 = 224. Off by 32 columns per head.
        gated=True,
        verified=True,
        notes=(
            "GEOMETRY TRAP, verified against the real config: head_dim=256 but "
            "hidden/n_heads = 224. Legacy slicing is wrong by 32 columns at every head offset "
            "and fails silently. o_proj.in_features = 16*256 = 4096 != hidden_size 3584."
        ),
    ),
    "gemma3-4b": ModelSpec(
        key="gemma3-4b",
        hf_name="google/gemma-3-4b-it",
        family="gemma",
        params_b=4.3,
        n_layers=34,
        hidden_size=2560,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,  # VERIFIED: derived would be 2560/8 = 320. Off by -64 columns per head.
        gated=True,
        verified=True,
        notes=(
            "Second geometry trap, opposite direction: head_dim=256 vs derived 320. "
            "o_proj.in_features = 8*256 = 2048, narrower than hidden_size 2560."
        ),
    ),
    "ministral-8b": ModelSpec(
        key="ministral-8b",
        hf_name="mistralai/Ministral-8B-Instruct-2410",
        family="mistral",
        params_b=8.0,
        n_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,  # explicit in config, equals derived
        gated=False,
        verified=True,
        notes="Ungated. Explicit head_dim that equals the derived value.",
    ),
}


def get(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; available: {sorted(MODELS)}")
    return MODELS[key]


def ungated() -> list[ModelSpec]:
    """Models runnable with no license approval — start here."""
    return [m for m in MODELS.values() if not m.gated]


def verified() -> list[ModelSpec]:
    """Models whose geometry was read from the real config."""
    return [m for m in MODELS.values() if m.verified]
