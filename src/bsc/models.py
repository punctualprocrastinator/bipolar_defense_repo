"""Model loading and device/dtype resolution.

This module and :mod:`bsc.provenance` are the *only* places allowed to touch ``torch.cuda``
directly (CLAUDE.md §1.7). Everything downstream receives an already-placed model and a
:class:`ModelBundle` describing what it got.

Hardware notes that matter for this project:

* **Blackwell (sm_100 / sm_120)** — full bf16 and FP8 support; bf16 is the right default. Needs a
  CUDA 12.8+ / PyTorch build with matching kernels. If PyTorch was compiled without sm_100
  kernels it will still "see" the GPU and then fail at the first matmul with a cryptic error, so
  :func:`resolve_device` checks capability against ``torch.cuda.get_arch_list()`` up front.
* **Turing (sm_75, e.g. GTX 1650)** — *no* native bf16. Silently emulating it is catastrophically
  slow, so we fall back to fp16 and say so. 4 GB VRAM will not hold a 7B model in any dtype;
  use tiny models for smoke tests only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from bsc.config import ModelConfig

log = logging.getLogger("bsc.models")

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Architectures with native bf16 arithmetic. Ampere is sm_80/86, Ada sm_89, Hopper sm_90,
# Blackwell sm_100/sm_120.
_BF16_MIN_MAJOR = 8


@dataclass(frozen=True)
class HeadGeometry:
    """Attention shape, resolved once so no call site recomputes head offsets.

    GQA is the trap this exists to close (CLAUDE.md §3.1). Qwen2.5-7B has 28 query heads and
    4 key/value heads. The input to ``o_proj`` is the concatenation of the **query**-head
    outputs, so it is indexed by ``num_attention_heads`` and ``head_dim = hidden_size /
    num_attention_heads``. Any code touching K or V instead needs ``num_key_value_heads``.
    Confusing the two silently slices the wrong head — an indexing bug that looks like a
    scientific result.
    """

    n_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads != self.num_attention_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def head_slice(self, head: int) -> slice:
        """Column span of one query head inside ``o_proj``'s input."""
        if not 0 <= head < self.num_attention_heads:
            raise ValueError(
                f"head {head} out of range for {self.num_attention_heads} attention heads"
            )
        return slice(head * self.head_dim, (head + 1) * self.head_dim)

    def kv_head_for(self, head: int) -> int:
        """Which KV head a given query head reads from, under GQA."""
        if not 0 <= head < self.num_attention_heads:
            raise ValueError(f"head {head} out of range")
        return head // self.kv_group_size

    @classmethod
    def from_hf_config(cls, config: Any) -> HeadGeometry:
        n_heads = int(config.num_attention_heads)
        hidden = int(config.hidden_size)
        n_kv = int(getattr(config, "num_key_value_heads", n_heads))
        # Newer configs carry head_dim explicitly and it is not always hidden/n_heads.
        head_dim = int(getattr(config, "head_dim", None) or hidden // n_heads)
        return cls(
            n_layers=int(config.num_hidden_layers),
            hidden_size=hidden,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
        )


@dataclass(frozen=True)
class ResolvedDevice:
    device: torch.device
    dtype: torch.dtype
    reason: str
    capability: str | None = None
    device_name: str | None = None


def resolve_device(cfg: ModelConfig) -> ResolvedDevice:
    """Pick device and dtype, explaining the choice.

    The ``reason`` string lands in the manifest so a run that quietly fell back to fp32 on CPU
    is visible in the artifact rather than discovered when the numbers look wrong.
    """
    want_device = cfg.device
    want_dtype = cfg.dtype

    if want_device not in {"auto", "cpu"} and not torch.cuda.is_available():
        raise RuntimeError(
            f"model.device={want_device!r} requested but CUDA is unavailable. "
            f"torch={torch.__version__} (a '+cpu' build cannot see any GPU). "
            "Install a CUDA build, or set model.device=cpu for a smoke test."
        )

    if want_device == "cpu" or (want_device == "auto" and not torch.cuda.is_available()):
        dtype = _DTYPES.get(want_dtype, torch.float32) if want_dtype != "auto" else torch.float32
        if dtype is torch.float16:
            # fp16 on CPU is unsupported by most kernels and silently miserable where it works.
            dtype = torch.float32
        return ResolvedDevice(
            device=torch.device("cpu"),
            dtype=dtype,
            reason=(
                "CPU: no CUDA device visible"
                if want_device == "auto"
                else "CPU: explicitly requested"
            ),
        )

    index = 0 if want_device in {"auto", "cuda"} else int(want_device.split(":")[1])
    props = torch.cuda.get_device_properties(index)
    capability = f"sm_{props.major}{props.minor}"

    # PyTorch can see a GPU whose SM it has no compiled kernels for. Catch it here with a clear
    # message instead of at the first matmul.
    arch_list = torch.cuda.get_arch_list()
    if arch_list and not any(a.startswith(f"sm_{props.major}") for a in arch_list):
        raise RuntimeError(
            f"This PyTorch build has no kernels for {capability} ({props.name}). "
            f"Compiled architectures: {arch_list}. "
            "For Blackwell you need a CUDA 12.8+ PyTorch build."
        )

    supports_bf16 = props.major >= _BF16_MIN_MAJOR
    if want_dtype == "auto":
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
        reason = (
            f"{props.name} ({capability}): bf16 native"
            if supports_bf16
            else f"{props.name} ({capability}): pre-Ampere, no native bf16 — using fp16"
        )
    else:
        dtype = _DTYPES[want_dtype]
        reason = f"{props.name} ({capability}): dtype pinned to {want_dtype} by config"
        if dtype is torch.bfloat16 and not supports_bf16:
            log.warning(
                "bfloat16 pinned on %s (%s) which lacks native bf16 — this will be emulated "
                "and very slow. Prefer dtype=float16 on this device.",
                props.name,
                capability,
            )

    return ResolvedDevice(
        device=torch.device(f"cuda:{index}"),
        dtype=dtype,
        reason=reason,
        capability=capability,
        device_name=props.name,
    )


def resolve_attn_implementation(cfg: ModelConfig) -> str:
    """Choose the attention kernel.

    Head-level interventions hook ``o_proj``'s input. That tensor exists under every
    implementation, but fused kernels change what is available *upstream* of it (attention
    probabilities, per-head outputs before concat). Interpretability runs therefore default to
    ``eager``, which keeps the whole computation in exposed PyTorch ops.
    """
    if cfg.attn_implementation != "auto":
        return cfg.attn_implementation
    return "eager" if cfg.require_eager_for_hooks else "sdpa"


@dataclass
class ModelBundle:
    """A loaded model plus everything downstream code needs to know about it."""

    model: Any
    tokenizer: Any
    geometry: HeadGeometry
    resolved: ResolvedDevice
    attn_implementation: str
    name: str

    @property
    def device(self) -> torch.device:
        return self.resolved.device

    @property
    def dtype(self) -> torch.dtype:
        return self.resolved.dtype

    def layer(self, index: int) -> Any:
        """The transformer block at ``index``, across the naming conventions we support."""
        model = self.model
        for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj[index]
        raise AttributeError(f"cannot locate transformer layers on {type(model).__name__}")

    def o_proj(self, index: int) -> Any:
        """The attention output projection of layer ``index`` — the hook site for head edits."""
        attn = getattr(self.layer(index), "self_attn", None) or getattr(self.layer(index), "attn")
        for attr in ("o_proj", "out_proj", "c_proj", "dense"):
            module = getattr(attn, attr, None)
            if module is not None:
                return module
        raise AttributeError(f"cannot locate output projection on {type(attn).__name__}")

    def describe(self) -> dict[str, Any]:
        """Manifest-friendly summary."""
        return {
            "name": self.name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "device_reason": self.resolved.reason,
            "capability": self.resolved.capability,
            "device_name": self.resolved.device_name,
            "attn_implementation": self.attn_implementation,
            "geometry": {
                "n_layers": self.geometry.n_layers,
                "hidden_size": self.geometry.hidden_size,
                "num_attention_heads": self.geometry.num_attention_heads,
                "num_key_value_heads": self.geometry.num_key_value_heads,
                "head_dim": self.geometry.head_dim,
                "is_gqa": self.geometry.is_gqa,
                "kv_group_size": self.geometry.kv_group_size,
            },
        }


def load_model(cfg: ModelConfig) -> ModelBundle:
    """Load a causal LM and its tokenizer, placed and typed per :func:`resolve_device`."""
    resolved = resolve_device(cfg)
    attn = resolve_attn_implementation(cfg)
    log.info("loading %s | %s | dtype=%s | attn=%s", cfg.name, resolved.reason, resolved.dtype, attn)

    hf_config = AutoConfig.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    geometry = HeadGeometry.from_hf_config(hf_config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batched generation must left-pad or the decode step reads padding as context.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        dtype=resolved.dtype,
        attn_implementation=attn,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.to(resolved.device)
    model.eval()
    model.requires_grad_(False)  # every experiment here is inference-only; guards against leaks

    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        geometry=geometry,
        resolved=resolved,
        attn_implementation=attn,
        name=cfg.name,
    )
    verify_geometry(bundle)
    return bundle


def verify_geometry(bundle: ModelBundle) -> None:
    """Assert the resolved head geometry matches the real ``o_proj`` weight.

    This is the check the legacy code never had. Verified on real configs:

    * Qwen2.5-7B: ``head_dim`` absent, derived 3584/28 = 128 — correct.
    * Llama-3.1-8B: explicit 128 == derived 128 — correct.
    * **Gemma-2-9B: explicit ``head_dim=256`` but ``hidden/n_heads`` = 3584/16 = 224.**
    * **Gemma-3-4B: explicit 256 but derived 2560/8 = 320.**

    On the Gemma models, code assuming the derived value slices the wrong columns at every head
    offset. It does not raise — the tensor is still wide enough for the early heads — so the
    result is a plausible-looking circuit map computed from misaligned activations. Failing
    loudly here is the difference between a caught bug and a retracted paper.
    """
    geometry = bundle.geometry
    expected = geometry.num_attention_heads * geometry.head_dim
    o_proj = bundle.o_proj(0)
    actual = getattr(o_proj, "in_features", None)
    if actual is None:  # non-Linear projection; nothing to check against
        return
    if expected != actual:
        raise ValueError(
            f"{bundle.name}: head geometry does not match o_proj. "
            f"num_attention_heads({geometry.num_attention_heads}) * head_dim({geometry.head_dim}) "
            f"= {expected}, but o_proj.in_features = {actual}. "
            f"hidden_size/num_attention_heads would give "
            f"{geometry.hidden_size // geometry.num_attention_heads}. "
            "Head slicing would target the wrong columns."
        )
    log.info(
        "geometry verified: %d heads x %d dims = %d == o_proj.in_features%s",
        geometry.num_attention_heads,
        geometry.head_dim,
        expected,
        f" (GQA {geometry.kv_group_size}:1)" if geometry.is_gqa else " (MHA)",
    )
