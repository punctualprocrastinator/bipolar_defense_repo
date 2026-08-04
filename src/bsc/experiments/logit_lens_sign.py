"""Logit-lens sign check on refusal heads -- tests the CLAIMS_AUDIT.md P0-4 root cause.

**The finding to verify:** the repo's own ``FINDINGS_7B_DEFENSE.md`` reports that the dominant
"refusal" head L25-H1, when its contribution is read through the unembedding, most promotes the
tokens *Sure / Certainly / definitely / surely* -- so multiplying its output by 3x pushes the
model **toward** compliance, not away. If true, this inverts the assumption under the entire
multiplicative defense, and is a more interesting result than the defense itself.

Method (direct logit attribution, done correctly):

For each head h in the refusal set, isolate its contribution to the residual stream --
``head_output_h @ o_proj.weight[:, head_slice(h)].T`` -- then project that single-head
contribution through the model's final norm and unembedding. Compare the promoted mass on a
curated refusal-token set vs a compliance-token set. A genuine refusal head promotes refusal
tokens (positive score); a sign-inverted head promotes compliance tokens (negative score).

This is the one place the legacy repo already sliced ``o_proj.weight`` columns correctly
(``logit_lens_poc.py``); the value it adds here is doing it inside the reproducibility harness,
across every refusal head and both models, with the head geometry verified rather than assumed.

Writes:
    head_sign_scores.json   per-head refusal-minus-compliance logit score + top promoted tokens
    summary.md              ranked table; flags heads whose sign contradicts their circuit score
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bsc.circuits import load_circuit_map
from bsc.config import ExperimentConfig
from bsc.data import get_scenario
from bsc.generation import build_chat_prompt
from bsc.hooks import Head
from bsc.models import ModelBundle, load_model
from bsc.runs import REPO_ROOT, RunContext

# Token sets for the refusal-vs-compliance projection. Leading spaces matter: these are the
# BPE forms that actually begin an assistant turn ("▁Sure", not "Sure").
REFUSAL_TOKENS = ["Sorry", "I", "As", "Unfortunately", "However", "No", "Cannot"]
COMPLIANCE_TOKENS = ["Sure", "Certainly", "Here", "Absolutely", "Yes", "Of", "Definitely"]


def _token_ids(bundle: ModelBundle, words: list[str]) -> list[int]:
    """First-subtoken id for each word in its turn-initial (space-prefixed) form."""
    ids = []
    for w in words:
        for form in (f" {w}", w):
            toks = bundle.tokenizer.encode(form, add_special_tokens=False)
            if toks:
                ids.append(toks[0])
                break
    return sorted(set(ids))


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    import torch

    root = repo_root or REPO_ROOT
    if not cfg.circuit.circuit_map_path:
        raise ValueError("logit_lens_sign needs circuit.circuit_map_path")
    circuit_path = Path(cfg.circuit.circuit_map_path)
    if not circuit_path.is_absolute():
        circuit_path = root / circuit_path

    scenario_name = cfg.data.path if cfg.data.path else "RDX_Synthesis"

    with RunContext.create("logit_lens_sign", cfg, notes=cfg.notes) as run_ctx:
        circuit = load_circuit_map(circuit_path)
        refusal_heads, _ = circuit.select(
            selection=cfg.circuit.selection,
            top_k_refusal=cfg.circuit.top_k_refusal,
            top_k_compliance=cfg.circuit.top_k_compliance,
        )

        bundle = load_model(cfg.model)
        run_ctx.save_json("model.json", bundle.describe())
        geometry = bundle.geometry

        refusal_ids = _token_ids(bundle, REFUSAL_TOKENS)
        compliance_ids = _token_ids(bundle, COMPLIANCE_TOKENS)
        run_ctx.log.info("refusal tok ids=%s compliance tok ids=%s", refusal_ids, compliance_ids)

        scenario = get_scenario(scenario_name)
        prompt = build_chat_prompt(bundle, scenario.messages_up_to_turn(5))
        inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)

        # Capture each hooked layer's o_proj input (= per-head concatenated outputs) at the last
        # position, in one forward pass.
        captured: dict[int, torch.Tensor] = {}
        layers = sorted({h.layer for h in refusal_heads})

        def make_capture(layer: int):
            def hook(module, args):
                captured[layer] = args[0][..., -1, :].detach().clone()
            return hook

        handles = [bundle.o_proj(l).register_forward_pre_hook(make_capture(l)) for l in layers]
        try:
            with torch.no_grad():
                bundle.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # Project each head's isolated contribution through norm + unembedding.
        final_norm = _final_norm(bundle)
        unembed = bundle.model.get_output_embeddings().weight  # (vocab, hidden)

        head_scores: list[dict[str, Any]] = []
        for head in refusal_heads:
            cols = geometry.head_slice(head.head)
            head_out = captured[head.layer][..., cols]  # (..., head_dim)
            # Contribution to the residual stream from this head alone.
            o_weight = bundle.o_proj(head.layer).weight  # (hidden, in_features)
            contribution = head_out.to(o_weight.dtype) @ o_weight[:, cols].T  # (..., hidden)

            with torch.no_grad():
                normed = final_norm(contribution)
                logits = (normed.to(unembed.dtype) @ unembed.T).float().flatten()

            refusal_mass = logits[refusal_ids].mean().item()
            compliance_mass = logits[compliance_ids].mean().item()
            sign_score = refusal_mass - compliance_mass

            topk = torch.topk(logits, 8)
            top_tokens = [bundle.tokenizer.decode([i]) for i in topk.indices.tolist()]

            circuit_score = circuit.score_of(head)
            inverted = circuit_score > 0 and sign_score < 0
            head_scores.append({
                "head": str(head),
                "circuit_score": circuit_score,
                "refusal_logit_mean": refusal_mass,
                "compliance_logit_mean": compliance_mass,
                "sign_score": sign_score,
                "sign_inverted": inverted,
                "top_promoted_tokens": top_tokens,
            })
            run_ctx.log.info(
                "%s circuit=%+.3f sign=%+.3f%s top=%s",
                head, circuit_score, sign_score,
                "  <-- INVERTED" if inverted else "",
                top_tokens[:4],
            )

        n_inverted = sum(h["sign_inverted"] for h in head_scores)
        payload = {
            "scenario": scenario_name,
            "refusal_tokens": REFUSAL_TOKENS,
            "compliance_tokens": COMPLIANCE_TOKENS,
            "n_refusal_heads": len(refusal_heads),
            "n_sign_inverted": n_inverted,
            "dominant_head": head_scores[0] if head_scores else None,
            "heads": head_scores,
        }
        run_ctx.save_json("head_sign_scores.json", payload)
        run_ctx.save_text("summary.md", _summary_md(payload))
        run_ctx.record_metric("n_sign_inverted", n_inverted)
        run_ctx.record_metric("dominant_head_inverted",
                              head_scores[0]["sign_inverted"] if head_scores else None)
        return payload


def _final_norm(bundle: ModelBundle):
    """The model's pre-unembedding norm, across naming conventions."""
    model = bundle.model
    for path in (("model", "norm"), ("transformer", "ln_f"), ("gpt_neox", "final_layer_norm")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("cannot locate final norm layer")


def _summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Logit-lens sign check on refusal heads",
        "",
        f"Scenario: `{payload['scenario']}`. "
        f"Refusal tokens {payload['refusal_tokens']}, compliance {payload['compliance_tokens']}.",
        "",
        "`sign_score` = mean refusal-token logit − mean compliance-token logit, from each head's "
        "isolated contribution. A head with a **positive circuit score but negative sign** is "
        "**sign-inverted**: amplifying it pushes toward compliance.",
        "",
        "| head | circuit score | sign score | inverted? | top promoted tokens |",
        "|---|---|---|---|---|",
    ]
    for h in payload["heads"]:
        toks = ", ".join(repr(t) for t in h["top_promoted_tokens"][:4])
        lines.append(
            f"| {h['head']} | {h['circuit_score']:+.3f} | {h['sign_score']:+.3f} "
            f"| {'**YES**' if h['sign_inverted'] else 'no'} | {toks} |"
        )
    lines += [
        "",
        f"**{payload['n_sign_inverted']} of {payload['n_refusal_heads']}** refusal heads are "
        "sign-inverted.",
    ]
    dom = payload.get("dominant_head")
    if dom:
        verdict = ("**is sign-inverted** -- this is the P0-4 root cause confirmed"
                   if dom["sign_inverted"] else "is not inverted")
        lines.append(
            f"\nThe dominant head **{dom['head']}** (highest circuit score) {verdict}: "
            f"it most promotes {', '.join(repr(t) for t in dom['top_promoted_tokens'][:4])}."
        )
    return "\n".join(lines) + "\n"
