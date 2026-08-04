"""Harm-contrastive activation patching -- circuit discovery for any model.

**What it produces:** a per-head causal map in the same schema as the legacy
``circuit_map.json`` (so downstream experiments and the sparsity metric consume it unchanged),
computed with device-agnostic raw-HF hooks instead of nnsight, and with the head geometry
verified rather than assumed. This is what lets the bipolar-circuit question be asked on models
the legacy code never touched (OLMo, Llama, Gemma) -- the cross-architecture generalisation test
that is the paper's spotlight-tier claim (RESEARCH_OVERVIEW.md §3).

**Method** (ported from ``curcuit_discovry.py``, corrected):

For each head, patch its *clean* (benign-prompt) last-token o_proj-input activation into the
*harmful* run, and measure how much that restores the refusal signal. The refusal signal is a
logit-difference metric: ``mean_logit(refusal_tokens) - mean_logit(compliance_tokens)`` at the
first generated position. A head whose clean activation raises that difference is a refusal head
(positive score); one that lowers it is a compliance head (negative score).

Corrections over the legacy version:
* Device/dtype from ``bsc.models``; no hardcoded ``cuda`` or model name.
* GQA-correct, bounds-checked head slicing via ``bsc.hooks`` geometry; asserted at load.
* Averaged over N paired (harmful, benign) prompts with a config-set N and a recorded seed.
* Writes a full run manifest; never overwrites a previous map.

Cost: one patched forward pass per (head x prompt-pair). For a 28-layer/28-head model at N=8
pairs that is ~28*28*8 forward passes -- fine on the Blackwell, slow on CPU, so keep N small
for smoke tests.

Writes:
    circuit_map.json   {model, method, num_heads, circuit_nodes[], top_k[], compliance_bottom_k[]}
    summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bsc.circuits import CircuitMap, HeadScore
from bsc.config import ExperimentConfig
from bsc.data import DATA_DIR
from bsc.hooks import Head
from bsc.models import ModelBundle, load_model
from bsc.runs import RunContext

REFUSAL_TOKENS = ["Sorry", "I", "As", "Unfortunately", "However", "No", "Cannot"]
COMPLIANCE_TOKENS = ["Sure", "Here", "Certainly", "Absolutely", "Yes", "Of", "Step"]


def _load_prompts(n: int) -> tuple[list[str], list[str]]:
    harmful_path = DATA_DIR / "advbench.csv"
    with harmful_path.open(encoding="utf-8") as f:
        harmful = [row["goal"] for row in csv.DictReader(f)]
    benign = json.loads((DATA_DIR / "benign_prompts.json").read_text(encoding="utf-8"))["prompts"]
    n = min(n, len(harmful), len(benign))
    return harmful[:n], benign[:n]


def _token_ids(bundle: ModelBundle, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        for form in (f" {w}", w):
            toks = bundle.tokenizer.encode(form, add_special_tokens=False)
            if toks:
                ids.append(toks[0])
                break
    return sorted(set(ids))


def _chat_prompt(bundle: ModelBundle, instruction: str) -> str:
    return bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    # repo_root kept in the signature for a uniform experiment interface (tests inject it).
    import torch

    n_pairs = cfg.data.limit or 8

    with RunContext.create("discover_circuit", cfg, notes=cfg.notes) as run_ctx:
        bundle = load_model(cfg.model)
        run_ctx.save_json("model.json", bundle.describe())
        geometry = bundle.geometry

        harmful, benign = _load_prompts(n_pairs)
        run_ctx.log.info("patching %d (harmful, benign) pairs on %s", len(harmful), bundle.name)

        refusal_ids = torch.tensor(_token_ids(bundle, REFUSAL_TOKENS), device=bundle.device)
        compliance_ids = torch.tensor(_token_ids(bundle, COMPLIANCE_TOKENS), device=bundle.device)

        def refusal_metric(logits_last: torch.Tensor) -> float:
            return (logits_last[refusal_ids].mean() - logits_last[compliance_ids].mean()).item()

        n_layers = geometry.n_layers
        n_heads = geometry.num_attention_heads
        # Accumulate per-head causal effect across pairs.
        effect = torch.zeros(n_layers, n_heads, dtype=torch.float64)

        for pair_idx, (h_text, b_text) in enumerate(zip(harmful, benign, strict=True)):
            h_prompt = _chat_prompt(bundle, h_text)
            b_prompt = _chat_prompt(bundle, b_text)

            # Capture clean (benign) last-token o_proj inputs for every layer, one pass.
            clean_acts: dict[int, torch.Tensor] = {}

            def cap(layer: int):
                def hook(module, args):
                    clean_acts[layer] = args[0][..., -1, :].detach().clone()
                return hook

            handles = [bundle.o_proj(l).register_forward_pre_hook(cap(l)) for l in range(n_layers)]
            try:
                with torch.no_grad():
                    bundle.model(**bundle.tokenizer(b_prompt, return_tensors="pt").to(bundle.device))
            finally:
                for hd in handles:
                    hd.remove()

            h_inputs = bundle.tokenizer(h_prompt, return_tensors="pt").to(bundle.device)
            with torch.no_grad():
                harmful_logits = bundle.model(**h_inputs).logits[0, -1].float()
            baseline_metric = refusal_metric(harmful_logits)

            # Patch each head individually: clean activation into the harmful run.
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
                            patched_logits = bundle.model(**h_inputs).logits[0, -1].float()
                    finally:
                        handle.remove()
                    effect[layer, head] += refusal_metric(patched_logits) - baseline_metric

            run_ctx.log.info("pair %d/%d done", pair_idx + 1, len(harmful))

        effect /= len(harmful)
        scores = [
            HeadScore(Head(l, h), float(effect[l, h]))
            for l in range(n_layers)
            for h in range(n_heads)
        ]
        circuit = CircuitMap(
            scores=scores,
            model=bundle.name,
            method="harm_contrastive_patching",
            num_heads=n_heads,
            patch_layers=list(range(n_layers)),
        )
        run_ctx.save_json("circuit_map.json", circuit.to_dict())

        top = circuit.sorted_scores[: cfg.circuit.top_k_refusal]
        bottom = sorted(circuit.scores, key=lambda s: s.score)[: cfg.circuit.top_k_compliance]
        run_ctx.save_text("summary.md", _summary_md(bundle, top, bottom, len(harmful)))

        run_ctx.record_metric("top_refusal_head", str(top[0].head))
        run_ctx.record_metric("top_refusal_score", round(top[0].score, 4))
        run_ctx.record_metric("top_compliance_head", str(bottom[0].head))
        run_ctx.record_metric("top_compliance_score", round(bottom[0].score, 4))
        run_ctx.log.info("top refusal %s (%+.3f) | top compliance %s (%+.3f)",
                         top[0].head, top[0].score, bottom[0].head, bottom[0].score)
        return {"model": bundle.name, "top_refusal": str(top[0].head),
                "top_compliance": str(bottom[0].head), "n_pairs": len(harmful)}


def _summary_md(bundle, top, bottom, n_pairs) -> str:
    lines = [
        f"# Circuit discovery: {bundle.name}",
        "",
        f"Harm-contrastive activation patching over {n_pairs} (harmful, benign) pairs. "
        f"Geometry: {bundle.geometry.num_attention_heads} query heads, "
        f"head_dim {bundle.geometry.head_dim}, "
        f"{'GQA' if bundle.geometry.is_gqa else 'MHA'}.",
        "",
        "## Top refusal heads",
        "| head | score |",
        "|---|---|",
    ]
    lines += [f"| {s.head} | {s.score:+.4f} |" for s in top]
    lines += ["", "## Top compliance heads", "| head | score |", "|---|---|"]
    lines += [f"| {s.head} | {s.score:+.4f} |" for s in bottom]

    # Intra-layer antagonism: the striking 7B finding was top refusal and top compliance in the
    # same layer. Report whether it recurs.
    if top and bottom and top[0].head.layer == bottom[0].head.layer:
        lines += ["", f"**Intra-layer antagonism:** top refusal and top compliance are both in "
                      f"layer {top[0].head.layer}."]
    return "\n".join(lines) + "\n"
