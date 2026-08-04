"""Head-level activation interventions.

All head editing in this project goes through here (CLAUDE.md §3). The three properties this
module guarantees, each of which the legacy scripts violated somewhere:

1. **Identity is a strict no-op.** ``multiplier=1.0`` with no ablations must not perturb a
   single bit. ``tests/test_hooks.py`` asserts this against real logits, which is what catches
   an off-by-one in head slicing — a bug that otherwise presents as a plausible "effect".
2. **One code path for measuring and generating.** The legacy Crescendo defense measured
   activation norms in one pass and generated in another with separately-written hooks; the two
   disagreed by exactly the amplification factor. Here, :class:`HeadIntervention` is a single
   object that both edits and records, so the recorded norm is by construction the value that
   was fed into ``o_proj``.
3. **Hooks are never leaked.** Use :func:`applied`; on any exception the handles are removed in
   a ``finally``. A leaked forward hook silently contaminates every later run in the process,
   which in a sweep means the baseline is not a baseline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from bsc.models import HeadGeometry, ModelBundle

log = logging.getLogger("bsc.hooks")


@dataclass(frozen=True, order=True)
class Head:
    """A single attention head, identified by layer and *query*-head index."""

    layer: int
    head: int

    def __str__(self) -> str:
        return f"L{self.layer}-H{self.head}"

    @classmethod
    def parse(cls, text: str) -> Head:
        """Parse the ``L25-H1`` notation used throughout the papers and JSON artifacts."""
        try:
            layer_part, head_part = text.strip().upper().split("-")
            return cls(int(layer_part.lstrip("L")), int(head_part.lstrip("H")))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"cannot parse head spec {text!r}; expected e.g. 'L25-H1'") from exc

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Head:
        """Build from the ``{"layer": int, "head": int}`` dicts stored in circuit maps."""
        return cls(int(node["layer"]), int(node["head"]))


@dataclass
class HeadEdit:
    """What to do to one head's contribution to ``o_proj``'s input.

    Applied as ``x <- x * scale + alpha * vector``. Composing both in a single edit lets the
    multiplicative and additive-steering defenses share one hook implementation instead of the
    two divergent ones in the legacy code.
    """

    scale: float = 1.0
    alpha: float = 0.0
    vector: torch.Tensor | None = None

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and (self.alpha == 0.0 or self.vector is None)


@dataclass
class InterventionRecord:
    """Per-head activations actually seen at the hook site, after editing."""

    pre_norms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    post_norms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    n_calls: int = 0

    def summary(self) -> dict[str, dict[str, float]]:
        """Mean pre/post L2 norm per head, over every forward call while hooks were active."""
        out: dict[str, dict[str, float]] = {}
        for name in self.pre_norms:
            pre = self.pre_norms[name]
            post = self.post_norms[name]
            out[name] = {
                "pre_mean": sum(pre) / len(pre) if pre else 0.0,
                "post_mean": sum(post) / len(post) if post else 0.0,
                "n": len(pre),
            }
        return out


class HeadIntervention:
    """Edits per-head slices of the attention output projection input.

    The hook site is ``o_proj``'s **input**: the concatenation of per-query-head outputs, just
    before they are mixed back into the residual stream. This is the finest granularity at
    which a head's contribution is still separable with a plain forward pre-hook.

    Args:
        bundle: The loaded model.
        edits: Head -> edit mapping.
        positions: ``"last"`` edits only the final sequence position; ``"all"`` edits every
            position. During KV-cached decode there is only one position, so these coincide;
            they differ during prefill, which is why it is explicit (CLAUDE.md §3.3).
        record: Capture pre/post norms while running.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        edits: dict[Head, HeadEdit],
        *,
        positions: str = "last",
        record: bool = True,
    ) -> None:
        if positions not in {"last", "all"}:
            raise ValueError(f"positions must be 'last' or 'all', got {positions!r}")
        self.bundle = bundle
        self.geometry: HeadGeometry = bundle.geometry
        self.positions = positions
        self.record_enabled = record
        self.record = InterventionRecord()
        self._handles: list[Any] = []

        for head in edits:
            if not 0 <= head.layer < self.geometry.n_layers:
                raise ValueError(f"{head}: layer out of range (n_layers={self.geometry.n_layers})")
            if not 0 <= head.head < self.geometry.num_attention_heads:
                raise ValueError(
                    f"{head}: head out of range "
                    f"(num_attention_heads={self.geometry.num_attention_heads})"
                )

        self.edits = edits
        self._by_layer: dict[int, list[tuple[Head, HeadEdit]]] = defaultdict(list)
        for head, edit in edits.items():
            self._by_layer[head.layer].append((head, edit))

    @property
    def is_identity(self) -> bool:
        return all(edit.is_identity for edit in self.edits.values())

    def _make_hook(self, layer: int):
        entries = self._by_layer[layer]
        geometry = self.geometry
        positions = self.positions
        record_enabled = self.record_enabled
        record = self.record

        def pre_hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not args:
                return None
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return None

            # Clone: the incoming tensor may be a view into an autograd-free buffer reused
            # elsewhere in the forward pass, and in-place edits on it have bitten this project
            # before. The cost is one attention-width tensor per hooked layer per token.
            edited = hidden.clone()
            pos = slice(None) if positions == "all" or edited.dim() < 3 else slice(-1, None)

            for head, edit in entries:
                cols = geometry.head_slice(head.head)
                view = edited[..., pos, cols] if edited.dim() >= 3 else edited[..., cols]

                if record_enabled:
                    record.pre_norms[str(head)].append(float(view.float().norm().item()))

                if edit.scale != 1.0:
                    view = view * edit.scale
                if edit.alpha != 0.0 and edit.vector is not None:
                    view = view + edit.alpha * edit.vector.to(
                        device=view.device, dtype=view.dtype
                    )

                if edited.dim() >= 3:
                    edited[..., pos, cols] = view
                else:
                    edited[..., cols] = view

                if record_enabled:
                    record.post_norms[str(head)].append(float(view.float().norm().item()))

            record.n_calls += 1
            return (edited, *args[1:])

        return pre_hook

    def register(self) -> None:
        if self._handles:
            raise RuntimeError("intervention already registered; call remove() first")
        for layer in sorted(self._by_layer):
            module = self.bundle.o_proj(layer)
            self._handles.append(module.register_forward_pre_hook(self._make_hook(layer)))
        log.debug("registered %d head hooks across %d layers", len(self.edits), len(self._handles))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class ResidualSteering:
    """Add a fixed direction to the residual stream at one or more layers' output.

    This is the *sign-correct* alternative to multiplicative head amplification. P0-4 showed
    that ``hidden *= multiplier`` on an activation-patching-selected "refusal" head can push
    toward compliance, because the head's sign in the unembedding basis was never checked.
    A difference-of-means refusal direction (Arditi et al. / CAA) is sign-correct by
    construction: it points from the complying activation mean to the refusing one, so adding a
    positive multiple always steers toward refusal.

    The direction is added at each hooked decoder layer's *output* (the residual stream), not at
    ``o_proj``'s input, because the CAA direction lives in residual space, not per-head space.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        direction: torch.Tensor,
        layers: Sequence[int],
        *,
        alpha: float = 1.0,
        positions: str = "all",
    ) -> None:
        if positions not in {"last", "all"}:
            raise ValueError(f"positions must be 'last' or 'all', got {positions!r}")
        self.bundle = bundle
        self.direction = direction
        self.layers = list(layers)
        self.alpha = alpha
        self.positions = positions
        self._handles: list[Any] = []

    @property
    def is_identity(self) -> bool:
        return self.alpha == 0.0

    def _make_hook(self):
        alpha = self.alpha
        positions = self.positions

        def hook(module: Any, args: Any, output: Any):
            # Decoder layers return either a tensor or a tuple whose first element is the hidden
            # state. Handle both without assuming a specific transformers version.
            hidden = output[0] if isinstance(output, tuple) else output
            vec = self.direction.to(device=hidden.device, dtype=hidden.dtype)
            if positions == "all" or hidden.dim() < 3:
                hidden = hidden + alpha * vec
            else:
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * vec
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

    def register(self) -> None:
        if self._handles:
            raise RuntimeError("steering already registered; call remove() first")
        for layer in self.layers:
            self._handles.append(self.bundle.layer(layer).register_forward_hook(self._make_hook()))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def applied(intervention: HeadIntervention | ResidualSteering) -> Iterator[Any]:
    """Register hooks for the duration of the block, always removing them afterwards."""
    intervention.register()
    try:
        yield intervention
    finally:
        intervention.remove()


@contextmanager
def no_intervention() -> Iterator[None]:
    """Explicit null context, so baseline and treatment code paths are structurally identical."""
    yield


def build_bipolar_edits(
    refusal_heads: Sequence[Head],
    compliance_heads: Sequence[Head],
    *,
    mode: str = "multiplicative",
    refusal_multiplier: float = 3.0,
    steering_alpha: float = 1.0,
    steering_vectors: dict[str, torch.Tensor] | None = None,
    compliance_scale: float = 0.0,
    ablate_compliance: bool = True,
) -> dict[Head, HeadEdit]:
    """Construct the bipolar defense: amplify refusal heads, suppress compliance heads.

    Ported from ``bipolar_defense_repo/crescendo/crescendo_bipolar_defense.py`` (multiplicative)
    and ``crescendo_7b_pipeline.py`` (additive steering). Those two used separate, subtly
    different hook implementations; unifying them here is what makes a head-to-head comparison
    of the mechanisms valid (NEXT_EXPERIMENTS.md §2).

    A head appearing in both lists is an error, not a silent last-writer-wins.
    """
    overlap = set(refusal_heads) & set(compliance_heads)
    if overlap:
        raise ValueError(
            f"heads classified as both refusal and compliance: {sorted(map(str, overlap))}"
        )

    edits: dict[Head, HeadEdit] = {}

    if mode == "multiplicative":
        for head in refusal_heads:
            edits[head] = HeadEdit(scale=refusal_multiplier)
    elif mode == "additive_steering":
        if steering_vectors is None:
            raise ValueError("mode='additive_steering' requires steering_vectors")
        for head in refusal_heads:
            key = str(head)
            if key not in steering_vectors:
                raise KeyError(
                    f"no steering vector for {key}; available: {sorted(steering_vectors)[:8]}..."
                )
            edits[head] = HeadEdit(alpha=steering_alpha, vector=steering_vectors[key])
    elif mode == "none":
        pass
    else:
        raise ValueError(f"unknown intervention mode {mode!r}")

    if ablate_compliance:
        for head in compliance_heads:
            edits[head] = HeadEdit(scale=compliance_scale)

    return edits
