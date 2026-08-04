"""GCG circuit discovery + bipolar defense + cross-attack transfer.

Answers three linked questions on a single model, single run:

1. **GCG circuit.** Optimize an adversarial suffix (nanoGCG) per harmful prompt, then discover
   the adversarial-contrastive circuit: patch clean (harmful, refusing) activations into the
   corrupt (harmful+suffix, complying) run. Same patching code as the harm-contrastive circuit.

2. **Does the bipolar defense stop GCG?** On the attacked prompts, measure attack success under
   (a) no defense, (b) additive CAA refusal steering, (c) compliance-head zero-ablation.

3. **Does the circuit / steering transfer across attacks?** Compare the GCG circuit to the
   harm-contrastive circuit (Jaccard overlap of top heads, rank correlation) — high overlap =
   the two attacks target the same heads. The steering transfer is answered directly by whether
   the *same* CAA direction that was built without any GCG data reduces GCG ASR.

nanoGCG is incompatible with transformers 5.x out of the box (it imports the removed
``is_tf_available``); a one-line shim installs the missing symbols before import.

Writes:
    suffixes.json, gcg_circuit.json, harm_circuit.json, circuit_transfer.json,
    defense_results.json, summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _install_nanogcg_shim() -> None:
    """nanoGCG imports symbols removed in transformers 5.x; provide them before import."""
    import transformers.utils as U

    for name in ("is_tf_available", "is_flax_available"):
        if not hasattr(U, name):
            setattr(U, name, lambda: False)


import csv  # noqa: E402
import json  # noqa: E402

from bsc.config import ExperimentConfig  # noqa: E402
from bsc.data import DATA_DIR  # noqa: E402
from bsc.generation import run_trials  # noqa: E402
from bsc.hooks import HeadEdit, HeadIntervention, ResidualSteering, applied  # noqa: E402
from bsc.metrics import wilson_interval  # noqa: E402
from bsc.models import ModelBundle, load_model  # noqa: E402
from bsc.patching import circuit_overlap, discover_circuit_scores  # noqa: E402
from bsc.runs import RunContext  # noqa: E402

# GCG hyperparameters. Small by research standards, sized to run several prompts on one Blackwell
# in minutes; documented here rather than buried as magic numbers.
GCG_STEPS = 150
GCG_SEARCH_WIDTH = 256
GCG_TOPK = 256


def _chat(bundle: ModelBundle, text: str) -> str:
    return bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def _load_harmful(n: int) -> list[tuple[str, str]]:
    with (DATA_DIR / "advbench.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["goal"], r["target"]) for r in rows[:n]]


def _extract_caa_direction(bundle: ModelBundle, n_pairs: int, dev_frac: float):
    """Reuse the steering_defense extraction (difference-of-means refusal direction)."""
    from bsc.experiments.steering_defense import _extract_direction

    return _extract_direction(bundle, n_pairs, dev_frac)


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    n_prompts = cfg.data.limit or 5

    with RunContext.create("gcg_transfer", cfg, notes=cfg.notes) as run_ctx:
        bundle = load_model(cfg.model)
        run_ctx.save_json("model.json", bundle.describe())

        # --- 1. get GCG suffixes ------------------------------------------------------
        # Two modes. Precomputed (data.path -> a suffixes JSON): reuse already-optimized suffixes
        # (e.g. the verified N=100 7B set) -- no nanoGCG, and far larger N. The suffixes are
        # model-specific, so this MUST run on the model they were optimized for. Fresh mode:
        # optimize with nanoGCG (small N; needs the transformers-5 shim).
        precomputed_path = cfg.data.path
        if precomputed_path:
            p = Path(precomputed_path)
            if not p.is_absolute():
                p = DATA_DIR / p.name if (DATA_DIR / p.name).exists() else (repo_root or Path.cwd()) / p
            payload = json.loads(p.read_text(encoding="utf-8"))
            expected_model = payload.get("model")
            if expected_model and expected_model != bundle.name:
                raise ValueError(
                    f"suffixes were optimized for {expected_model!r} but model is {bundle.name!r}; "
                    "GCG suffixes are model-specific and will not transfer across models"
                )
            pairs = payload["pairs"][:n_prompts]
            suffixes = [{"goal": r["goal"], "suffix": r["suffix"]} for r in pairs]
            harmful = [(r["goal"], "") for r in pairs]
            run_ctx.log.info("loaded %d precomputed suffixes from %s (model %s)",
                             len(suffixes), p.name, expected_model)
        else:
            _install_nanogcg_shim()
            import nanogcg
            from nanogcg import GCGConfig

            harmful = _load_harmful(n_prompts)
            suffixes = []
            gcg_cfg = GCGConfig(num_steps=GCG_STEPS, search_width=GCG_SEARCH_WIDTH,
                                topk=GCG_TOPK, seed=cfg.seed, verbosity="ERROR")
            for i, (goal, target) in enumerate(harmful):
                run_ctx.log.info("GCG %d/%d: %s", i + 1, n_prompts, goal[:60])
                result = nanogcg.run(bundle.model, bundle.tokenizer, goal, target, gcg_cfg)
                suffixes.append({"goal": goal, "suffix": result.best_string,
                                 "loss": float(result.best_loss)})
        run_ctx.save_json("suffixes.json", suffixes)

        # --- 2. circuits: harm-contrastive and GCG-contrastive, same code -------------
        benign = json.loads((DATA_DIR / "benign_prompts.json").read_text(encoding="utf-8"))["prompts"]
        harm_clean = [_chat(bundle, b) for b in benign[:n_prompts]]          # benign -> complies
        harm_corrupt = [_chat(bundle, g) for g, _ in harmful]               # harmful -> refuses
        # Harm-contrastive: clean=refusing(harmful), corrupt=complying(benign). Patch refusing
        # activation into the complying run to find refusal heads.
        harm_circuit = discover_circuit_scores(
            bundle, clean_prompts=harm_corrupt, corrupt_prompts=harm_clean,
            method="harm_contrastive_patching",
        )
        # GCG-contrastive: clean=refusing(harmful, no suffix), corrupt=complying(harmful+suffix).
        gcg_clean = [_chat(bundle, g) for g, _ in harmful]
        gcg_corrupt = [_chat(bundle, f"{g} {s['suffix']}") for (g, _), s in zip(harmful, suffixes, strict=True)]
        gcg_circuit = discover_circuit_scores(
            bundle, clean_prompts=gcg_clean, corrupt_prompts=gcg_corrupt,
            method="adversarial_contrastive_patching",
        )
        run_ctx.save_json("harm_circuit.json", harm_circuit.to_dict())
        run_ctx.save_json("gcg_circuit.json", gcg_circuit.to_dict())

        # --- 3. circuit transfer ------------------------------------------------------
        overlap = circuit_overlap(harm_circuit, gcg_circuit, k=cfg.circuit.top_k_refusal)
        run_ctx.save_json("circuit_transfer.json", overlap)
        run_ctx.log.info("circuit transfer: refusal Jaccard=%.2f, compliance Jaccard=%.2f, spearman=%.2f",
                         overlap["refusal_jaccard"], overlap["compliance_jaccard"], overlap["spearman_full"])

        # --- 4. defenses on GCG-attacked prompts --------------------------------------
        direction, best_layer, _ = _extract_caa_direction(bundle, cfg.data.limit or 16, cfg.data.dev_fraction)
        steer_layers = [l for l in (best_layer, best_layer + 1, best_layer + 2) if l < bundle.geometry.n_layers]

        # Compliance heads from the harm-contrastive circuit (the transfer test: defense built
        # from harm/Crescendo-side data, applied to GCG).
        _, compliance_heads = harm_circuit.select(
            selection="top_k",
            top_k_refusal=cfg.circuit.top_k_refusal,
            top_k_compliance=cfg.circuit.top_k_compliance,
        )
        ablate_edits = {h: HeadEdit(scale=0.0) for h in compliance_heads}

        attacked = [f"{g} {s['suffix']}" for (g, _), s in zip(harmful, suffixes, strict=True)]
        defense_results: dict[str, list[dict[str, Any]]] = {"none": [], "caa_steer": [], "compliance_ablate": []}

        for i, atk in enumerate(attacked):
            prompt = _chat(bundle, atk)
            base_seed = cfg.seed + i

            defense_results["none"].append(run_trials(
                bundle, prompt, cfg.generation, prompt_id=f"gcg_{i}", condition="none",
                base_seed=base_seed).to_dict())

            steering = ResidualSteering(bundle, direction, steer_layers,
                                        alpha=cfg.intervention.steering_alpha or 8.0, positions="all")
            with applied(steering):
                defense_results["caa_steer"].append(run_trials(
                    bundle, prompt, cfg.generation, prompt_id=f"gcg_{i}", condition="caa_steer",
                    base_seed=base_seed).to_dict())

            with applied(HeadIntervention(bundle, ablate_edits, positions="last")):
                defense_results["compliance_ablate"].append(run_trials(
                    bundle, prompt, cfg.generation, prompt_id=f"gcg_{i}", condition="compliance_ablate",
                    base_seed=base_seed).to_dict())

        run_ctx.save_json("defense_results.json", defense_results)

        # ASR per defense (any trial complies).
        def asr(cond: str):
            flags = [d["any_success"] for d in defense_results[cond]]
            return wilson_interval(sum(flags), len(flags), cfg.eval.confidence).to_dict()

        asr_summary = {c: asr(c) for c in defense_results}
        run_ctx.save_text("summary.md", _summary_md(overlap, asr_summary, best_layer, len(attacked)))
        for c, a in asr_summary.items():
            run_ctx.record_metric(f"gcg_asr_{c}", round(a["value"], 3))
        run_ctx.record_metric("refusal_jaccard", round(overlap["refusal_jaccard"], 3))
        run_ctx.record_metric("circuit_spearman", round(overlap["spearman_full"], 3))
        return {"overlap": overlap, "asr": asr_summary}


def _summary_md(overlap, asr_summary, best_layer, n) -> str:
    lines = [
        "# GCG circuit, bipolar defense, and cross-attack transfer",
        "",
        f"n={n} GCG-attacked prompts. CAA steering direction from layer {best_layer} "
        "(built from harm/benign data, NOT GCG data — so reducing GCG ASR = steering transfer).",
        "",
        "## Circuit transfer: GCG vs harm-contrastive",
        f"- Top-k refusal-head Jaccard overlap: **{overlap['refusal_jaccard']:.2f}** "
        f"(shared: {overlap['refusal_shared'] or 'none'})",
        f"- Top-k compliance-head Jaccard overlap: **{overlap['compliance_jaccard']:.2f}** "
        f"(shared: {overlap['compliance_shared'] or 'none'})",
        f"- Full per-head score rank correlation (Spearman): **{overlap['spearman_full']:.2f}**",
        "",
        "High overlap = GCG and harm/Crescendo attacks target the *same* heads (the circuit "
        "transfers). Near-zero = they exploit different structure.",
        "",
        "## Does the bipolar defense stop GCG?",
        "| defense | GCG ASR | 95% CI |",
        "|---|---|---|",
    ]
    for c, a in asr_summary.items():
        lines.append(f"| {c} | {a['value']:.0%} | [{a['ci_low']:.0%}, {a['ci_high']:.0%}] |")
    lines += ["",
              "`caa_steer` and `compliance_ablate` are both built from harm/benign data with no "
              "GCG information, so any ASR reduction there is cross-attack transfer of the defense."]
    return "\n".join(lines) + "\n"
