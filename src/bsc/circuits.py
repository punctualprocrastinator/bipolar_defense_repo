"""Circuit maps: loading legacy artifacts, selecting heads, and sparsity metrics.

A "circuit map" is the output of harm-contrastive activation patching: a causal score per
attention head, where positive means patching the clean activation in *restores* refusal
(a refusal head) and negative means it pushes toward compliance (a compliance head).

Two things here are load-bearing for the paper:

* :func:`sparsity_fraction` is the metric behind the "top-10 heads carry X% of causal effect"
  claim. ``CLAIMS_AUDIT.md`` P0-1 found the published GCG value of ~85% has no source and
  recomputes to 60.1%, with the GCG-vs-Crescendo contrast *inverting* under the other available
  circuit map. One implementation, used for every attack, is how that stops happening.
* :func:`load_circuit_map` records provenance, because the repo contains two conflicting maps
  for each model and nothing states which supersedes which (``CLAIMS_AUDIT.md`` P1).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bsc.hooks import Head

log = logging.getLogger("bsc.circuits")


@dataclass(frozen=True)
class HeadScore:
    """One head's causal patching score."""

    head: Head
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.head.layer, "head": self.head.head, "score": self.score}


@dataclass
class CircuitMap:
    """A full per-head causal map, plus where it came from.

    Legacy schema (``circuit_map.json``)::

        {"model": str, "method": str, "num_heads": int, "patch_layers": [int],
         "circuit_nodes": [{"layer": int, "head": int, "score": float}],   # all heads, sorted desc
         "top_10": [...], "compliance_bottom_5": [...]}                    # derived views

    Only ``circuit_nodes`` is read: the ``top_10`` / ``compliance_bottom_5`` views bake in a
    selection rule (k=10, k=5) that this project needs to vary and report sensitivity to. Note
    that the published defense actually used **11** refusal heads (``K_HEADS = 11`` in
    ``adaptive_defense.py``) while the JSON stores only ``top_10`` — reading the derived view
    would silently reproduce a different circuit than the paper's.
    """

    scores: list[HeadScore]
    model: str | None = None
    method: str | None = None
    source_path: str | None = None
    num_heads: int | None = None
    patch_layers: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[Head] = set()
        for entry in self.scores:
            if entry.head in seen:
                raise ValueError(f"duplicate head in circuit map: {entry.head}")
            seen.add(entry.head)

    @property
    def sorted_scores(self) -> list[HeadScore]:
        """Descending by score: refusal heads first, compliance heads last."""
        return sorted(self.scores, key=lambda s: -s.score)

    def refusal_heads(self, k: int) -> list[Head]:
        """Top-k heads by score."""
        if k > len(self.scores):
            raise ValueError(f"requested {k} refusal heads but map has {len(self.scores)}")
        return [s.head for s in self.sorted_scores[:k]]

    def compliance_heads(self, k: int) -> list[Head]:
        """Bottom-k heads by score (most negative), returned most-negative-first."""
        if k > len(self.scores):
            raise ValueError(f"requested {k} compliance heads but map has {len(self.scores)}")
        return [s.head for s in sorted(self.scores, key=lambda s: s.score)[:k]]

    def heads_above(self, threshold: float) -> list[Head]:
        return [s.head for s in self.sorted_scores if s.score > threshold]

    def heads_below(self, threshold: float) -> list[Head]:
        return [s.head for s in sorted(self.scores, key=lambda s: s.score) if s.score < threshold]

    def score_of(self, head: Head) -> float:
        for entry in self.scores:
            if entry.head == head:
                return entry.score
        raise KeyError(f"{head} not in circuit map")

    def select(
        self,
        *,
        selection: str = "top_k",
        top_k_refusal: int = 11,
        top_k_compliance: int = 5,
        score_threshold: float = 0.0,
    ) -> tuple[list[Head], list[Head]]:
        """Apply the configured selection rule, returning (refusal, compliance).

        Overlap is impossible by construction for ``top_k`` unless k values exceed the map size,
        but ``threshold`` selection with a negative threshold could produce it — checked here so
        the failure is a clear error rather than an ambiguous intervention.
        """
        if selection == "top_k":
            refusal = self.refusal_heads(top_k_refusal)
            compliance = self.compliance_heads(top_k_compliance)
        elif selection == "threshold":
            refusal = self.heads_above(score_threshold)
            compliance = self.heads_below(-abs(score_threshold))
        else:
            raise ValueError(f"unknown selection rule {selection!r}")

        overlap = set(refusal) & set(compliance)
        if overlap:
            raise ValueError(
                f"selection produced overlapping refusal/compliance heads: "
                f"{sorted(map(str, overlap))}"
            )
        return refusal, compliance

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "method": self.method,
            "source_path": self.source_path,
            "num_heads": self.num_heads,
            "patch_layers": self.patch_layers,
            "n_scored_heads": len(self.scores),
            "circuit_nodes": [s.to_dict() for s in self.sorted_scores],
        }


def load_circuit_map(path: str | Path) -> CircuitMap:
    """Load a legacy or current circuit map JSON."""
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    nodes = payload.get("circuit_nodes")
    if nodes is None:
        # Crescendo override maps store only the selected heads under a different key.
        nodes = payload.get("context_override_heads")
        if nodes is None:
            raise ValueError(
                f"{p}: no 'circuit_nodes' or 'context_override_heads' key; "
                f"found {sorted(payload)}"
            )
        log.warning(
            "%s contains only pre-selected heads (%d), not a full per-head map. "
            "Sparsity metrics computed from it are not comparable to a full map.",
            p.name,
            len(nodes),
        )

    scores = [HeadScore(Head.from_node(n), float(n["score"])) for n in nodes]
    known = {"model", "method", "num_heads", "patch_layers", "circuit_nodes"}
    return CircuitMap(
        scores=scores,
        model=payload.get("model"),
        method=payload.get("method"),
        source_path=str(p),
        num_heads=payload.get("num_heads"),
        patch_layers=list(payload.get("patch_layers", [])),
        extra={k: v for k, v in payload.items() if k not in known},
    )


def save_circuit_map(circuit: CircuitMap, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(circuit.to_dict(), indent=2), encoding="utf-8")
    return p


# -- sparsity ---------------------------------------------------------------------


@dataclass(frozen=True)
class SparsityReport:
    """How concentrated a circuit's causal effect is.

    ``top_k_fraction`` is ``sum(top-k positive scores) / sum(all positive scores)``, matching
    ``crescendo_circuit_discovery_v3.py:243-250`` so new numbers stay comparable with the
    verified 0.4444 Crescendo value.
    """

    top_k: int
    top_k_fraction: float
    positive_mass: float
    n_positive: int
    n_heads: int
    sparse_threshold: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "top_k_fraction": self.top_k_fraction,
            "positive_mass": self.positive_mass,
            "n_positive": self.n_positive,
            "n_heads": self.n_heads,
            "sparse_threshold": self.sparse_threshold,
            "interpretation": self.interpretation,
        }


def sparsity_fraction(
    circuit: CircuitMap | Sequence[float],
    *,
    top_k: int = 10,
    sparse_threshold: float = 0.6,
) -> SparsityReport:
    """Fraction of total positive causal mass carried by the top-k heads.

    Only positive scores enter the denominator, matching the legacy metric: negative-scoring
    (compliance) heads are a separate population, and including them would let a strong
    compliance head inflate the apparent concentration of the refusal circuit.

    ``sparse_threshold=0.6`` is the legacy cutoff for calling a circuit "sparse". The paper's own
    7B GCG map lands at 0.6008 — on the boundary to four decimals — so any claim of sparsity
    must report the value, not just the label (``CLAIMS_AUDIT.md`` P0-1).
    """
    raw = (
        [s.score for s in circuit.scores]
        if isinstance(circuit, CircuitMap)
        else list(circuit)
    )
    if not raw:
        raise ValueError("empty circuit")

    positives = sorted((s for s in raw if s > 0), reverse=True)
    positive_mass = sum(positives)
    if positive_mass <= 0:
        raise ValueError("no positive causal mass; sparsity is undefined")

    fraction = sum(positives[:top_k]) / positive_mass
    return SparsityReport(
        top_k=top_k,
        top_k_fraction=fraction,
        positive_mass=positive_mass,
        n_positive=len(positives),
        n_heads=len(raw),
        sparse_threshold=sparse_threshold,
        interpretation="sparse_localized" if fraction > sparse_threshold else "diffuse_distributed",
    )


def compare_sparsity(maps: dict[str, CircuitMap], *, top_k: int = 10) -> dict[str, Any]:
    """Sparsity across several circuits under one metric.

    This is the direct fix for ``CLAIMS_AUDIT.md`` P0-1: the published GCG-vs-Crescendo sparsity
    contrast came from two different computations, one of which was never actually run.
    """
    reports = {name: sparsity_fraction(m, top_k=top_k) for name, m in maps.items()}
    return {
        "top_k": top_k,
        "reports": {name: r.to_dict() for name, r in reports.items()},
        "sources": {name: m.source_path for name, m in maps.items()},
        "methods": {name: m.method for name, m in maps.items()},
    }


# -- steering vectors --------------------------------------------------------------


def load_steering_vectors(path: str | Path) -> dict[str, Any]:
    """Load per-head steering vectors, keyed by the ``L{layer}-H{head}`` string form.

    Legacy checkpoints use several key conventions (tuple keys, ``(layer, head)`` strings, nested
    dicts). Everything is normalised to :class:`Head`'s string form here so
    :func:`bsc.hooks.build_bipolar_edits` has exactly one lookup convention to handle.
    """
    import torch

    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"expected a dict of steering vectors, got {type(raw).__name__}")

    normalised: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(key, tuple) and len(key) == 2:
            head = Head(int(key[0]), int(key[1]))
        elif isinstance(key, str):
            try:
                head = Head.parse(key)
            except ValueError:
                normalised[key] = value  # non-head metadata; keep as-is
                continue
        else:
            raise TypeError(f"unrecognised steering-vector key: {key!r}")
        normalised[str(head)] = value
    return normalised
