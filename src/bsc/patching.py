"""Reusable activation-patching circuit discovery.

Factored out so the harm-contrastive (benign vs harmful) and adversarial-contrastive
(harmful vs harmful+suffix) circuits are discovered by *identical* code — the only difference
is which prompt pairs are passed in. That is what makes the two circuits comparable when asking
whether the GCG circuit and the Crescendo/harm circuit are the same structure
(`CRESCENDO_CIRCUIT_METHOD.md`, and the cross-attack transfer question).

Method: for each head, patch its clean (refusing-run) last-token o_proj-input activation into
the corrupt (complying-run) forward pass, and score the change in a refusal logit-difference
metric. Positive => patching that head *restores* refusal => refusal head. Negative => the head
pushes toward compliance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from bsc.circuits import CircuitMap, HeadScore
from bsc.hooks import Head

if TYPE_CHECKING:
    from bsc.models import ModelBundle

REFUSAL_TOKENS = ["Sorry", "I", "As", "Unfortunately", "However", "No", "Cannot"]
COMPLIANCE_TOKENS = ["Sure", "Here", "Certainly", "Absolutely", "Yes", "Of", "Step"]


def token_ids(bundle: ModelBundle, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        for form in (f" {w}", w):
            toks = bundle.tokenizer.encode(form, add_special_tokens=False)
            if toks:
                ids.append(toks[0])
                break
    return sorted(set(ids))


def discover_circuit_scores(
    bundle: ModelBundle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    *,
    method: str,
) -> CircuitMap:
    """Per-head causal map from clean->corrupt activation patching, averaged over prompt pairs.

    ``clean_prompts`` are the refusing runs (their activations get patched in); ``corrupt_prompts``
    are the complying runs (patched into). The two lists must be aligned and equal length.
    """
    import torch

    if len(clean_prompts) != len(corrupt_prompts):
        raise ValueError("clean and corrupt prompt lists must be aligned and equal length")

    geometry = bundle.geometry
    n_layers, n_heads = geometry.n_layers, geometry.num_attention_heads

    refusal_ids = torch.tensor(token_ids(bundle, REFUSAL_TOKENS), device=bundle.device)
    compliance_ids = torch.tensor(token_ids(bundle, COMPLIANCE_TOKENS), device=bundle.device)

    def refusal_metric(logits_last: torch.Tensor) -> float:
        return (logits_last[refusal_ids].mean() - logits_last[compliance_ids].mean()).item()

    effect = torch.zeros(n_layers, n_heads, dtype=torch.float64)

    for clean_text, corrupt_text in zip(clean_prompts, corrupt_prompts, strict=True):
        clean_acts: dict[int, torch.Tensor] = {}

        def cap(layer: int):
            def hook(module, args):
                clean_acts[layer] = args[0][..., -1, :].detach().clone()
            return hook

        handles = [bundle.o_proj(l).register_forward_pre_hook(cap(l)) for l in range(n_layers)]
        try:
            with torch.no_grad():
                bundle.model(**bundle.tokenizer(clean_text, return_tensors="pt").to(bundle.device))
        finally:
            for h in handles:
                h.remove()

        corrupt_inputs = bundle.tokenizer(corrupt_text, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            base_metric = refusal_metric(bundle.model(**corrupt_inputs).logits[0, -1].float())

        for layer in range(n_layers):
            clean = clean_acts[layer]
            for head in range(n_heads):
                cols = geometry.head_slice(head)

                def patch(module, args, _cols=cols, _clean=clean):
                    x = args[0].clone()
                    x[..., -1, _cols] = _clean[..., _cols]
                    return (x, *args[1:])

                handle = bundle.o_proj(layer).register_forward_pre_hook(patch)
                try:
                    with torch.no_grad():
                        patched = refusal_metric(bundle.model(**corrupt_inputs).logits[0, -1].float())
                finally:
                    handle.remove()
                effect[layer, head] += patched - base_metric

    effect /= len(clean_prompts)
    scores = [
        HeadScore(Head(l, h), float(effect[l, h]))
        for l in range(n_layers)
        for h in range(n_heads)
    ]
    return CircuitMap(scores=scores, model=bundle.name, method=method,
                      num_heads=n_heads, patch_layers=list(range(n_layers)))


def circuit_overlap(a: CircuitMap, b: CircuitMap, *, k: int) -> dict:
    """Compare two circuits: Jaccard overlap of top-k refusal and bottom-k compliance heads,
    plus Spearman rank correlation of the shared per-head scores.

    This is the quantitative form of "does the GCG circuit transfer to the Crescendo/harm
    circuit" — high overlap means the attacks target the same heads."""
    import numpy as np

    ref_a, ref_b = set(a.refusal_heads(k)), set(b.refusal_heads(k))
    comp_a, comp_b = set(a.compliance_heads(k)), set(b.compliance_heads(k))

    def jaccard(x: set, y: set) -> float:
        return len(x & y) / len(x | y) if (x | y) else 0.0

    # Rank correlation over the full head set (both maps cover the same heads).
    scores_a = {s.head: s.score for s in a.scores}
    scores_b = {s.head: s.score for s in b.scores}
    common = sorted(set(scores_a) & set(scores_b), key=str)
    va = np.array([scores_a[h] for h in common])
    vb = np.array([scores_b[h] for h in common])

    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = np.sqrt((rx**2).sum() * (ry**2).sum())
        return float((rx * ry).sum() / denom) if denom else 0.0

    return {
        "k": k,
        "refusal_jaccard": jaccard(ref_a, ref_b),
        "refusal_shared": sorted(str(h) for h in ref_a & ref_b),
        "compliance_jaccard": jaccard(comp_a, comp_b),
        "compliance_shared": sorted(str(h) for h in comp_a & comp_b),
        "spearman_full": spearman(va, vb),
        "n_common_heads": len(common),
    }
