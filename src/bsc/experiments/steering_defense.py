"""Additive (sign-correct) bipolar steering defense -- does OUR steering actually work?

**Context:** P0-4 showed the *multiplicative* head-amplification defense makes attack success
*worse* (ASR 70% -> 90% at 3x -> 100% at 6x on 1.5B), because the amplified head is
sign-inverted. This experiment tests the sign-correct alternative: additive steering along a
difference-of-means refusal direction (Arditi et al. / CAA), which points from the complying
activation mean to the refusing one and is therefore correctly signed by construction.

Pipeline (correlational discovery -> causal validation, the discipline from CRESCENDO_CIRCUIT_METHOD.md):

1. **Extract** the refusal direction: mean last-token residual on harmful prompts (which the
   model refuses) minus mean on benign prompts (which it complies with), per layer. Pick the
   layer whose direction most separates a held-out split -- reported, not hand-picked.
2. **Apply** it additively to the residual stream during Crescendo turn-5 generation, sweeping
   the coefficient, and measure ASR with the same paired/seeded/degeneracy-aware protocol as the
   multiplicative sweep, so the two are directly comparable.

A working defense shows ASR *decreasing* with the coefficient (the opposite of the multiplicative
result), without collapsing the model into NONRESPONSE degeneration.

Writes:
    refusal_direction.json   per-layer separation + chosen layer
    steering_results.json    baseline vs coefficient trial sets
    comparison.json          ASR per coefficient with CIs and McNemar vs baseline
    summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bsc.config import ExperimentConfig
from bsc.data import DATA_DIR, get_scenario, load_crescendo_scenarios
from bsc.generation import build_chat_prompt, run_trials
from bsc.hooks import ResidualSteering, applied
from bsc.metrics import cohens_d, mcnemar_paired, wilson_interval
from bsc.models import ModelBundle, load_model
from bsc.runs import RunContext

# Coefficients on the raw difference-of-means vector. The vector has a natural scale (the gap
# between the two class means), so alpha=1 adds roughly one "refuse-minus-comply" step.
DEFAULT_COEFFS = (0.0, 2.0, 4.0, 8.0, 16.0)


def _load_contrast_prompts(bundle: ModelBundle, n: int) -> tuple[list[str], list[str]]:
    with (DATA_DIR / "advbench.csv").open(encoding="utf-8") as f:
        harmful = [row["goal"] for row in csv.DictReader(f)]
    benign = json.loads((DATA_DIR / "benign_prompts.json").read_text(encoding="utf-8"))["prompts"]
    n = min(n, len(harmful), len(benign))
    fmt = lambda text: bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    return [fmt(h) for h in harmful[:n]], [fmt(b) for b in benign[:n]]


def _layer_residuals(bundle: ModelBundle, prompts: list[str]):
    """Last-token residual at every layer output for each prompt. Shape: (n_prompts, n_layers, hidden)."""
    import torch

    per_prompt = []
    for prompt in prompts:
        inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            out = bundle.model(**inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding; [1:] are the layer outputs.
        layers = torch.stack([h[0, -1].float() for h in out.hidden_states[1:]])
        per_prompt.append(layers)
    return torch.stack(per_prompt)


def _extract_direction(bundle: ModelBundle, n_pairs: int, dev_frac: float):
    """Per-layer difference-of-means refusal direction, choosing the best layer on a dev split."""
    import torch

    harmful, benign = _load_contrast_prompts(bundle, n_pairs)
    h_res = _layer_residuals(bundle, harmful)  # (n, L, hidden)
    b_res = _layer_residuals(bundle, benign)

    n = h_res.shape[0]
    n_dev = max(2, int(n * dev_frac))
    # Direction from TRAIN split; layer chosen by separation on DEV split (no leakage).
    h_tr, h_dev = h_res[n_dev:], h_res[:n_dev]
    b_tr, b_dev = b_res[n_dev:], b_res[:n_dev]

    directions = (h_tr.mean(0) - b_tr.mean(0))  # (L, hidden)

    per_layer = []
    for layer in range(directions.shape[0]):
        d = directions[layer]
        d_unit = d / (d.norm() + 1e-8)
        # Project dev activations onto the direction; separation = Cohen's d between classes.
        h_proj = (h_dev[:, layer] @ d_unit).tolist()
        b_proj = (b_dev[:, layer] @ d_unit).tolist()
        try:
            sep = cohens_d(h_proj, b_proj)
        except ValueError:
            sep = 0.0
        per_layer.append({"layer": layer, "separation_d": sep, "norm": float(d.norm())})

    best = max(per_layer, key=lambda r: abs(r["separation_d"]))
    best_layer = best["layer"]
    return directions[best_layer].to(bundle.dtype), best_layer, per_layer


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    coeffs = tuple(cfg.intervention_sweep) if cfg.intervention_sweep else DEFAULT_COEFFS
    n_pairs = cfg.data.limit or 16
    terminal_turn = 5

    scenario_names = sorted(load_crescendo_scenarios())

    with RunContext.create("steering_defense", cfg, notes=cfg.notes) as run_ctx:
        bundle = load_model(cfg.model)
        run_ctx.save_json("model.json", bundle.describe())

        direction, best_layer, per_layer = _extract_direction(
            bundle, n_pairs, cfg.data.dev_fraction
        )
        run_ctx.save_json(
            "refusal_direction.json",
            {
                "best_layer": best_layer,
                "n_pairs": n_pairs,
                "per_layer_separation": per_layer,
                "method": "difference_of_means(harmful_refused - benign_complied), CAA-style",
            },
        )
        # Steer at the chosen layer and the two above it -- a small band, as in Arditi et al.
        steer_layers = [l for l in (best_layer, best_layer + 1, best_layer + 2)
                        if l < bundle.geometry.n_layers]
        run_ctx.log.info("refusal direction from layer %d (sep d=%.2f); steering layers %s",
                         best_layer, per_layer[best_layer]["separation_d"], steer_layers)

        conditions = ["baseline"] + [f"alpha_{a:g}" for a in coeffs if a != 0.0]
        results: dict[str, dict[str, Any]] = {c: {} for c in conditions}

        for name in scenario_names:
            scenario = get_scenario(name)
            prompt = build_chat_prompt(bundle, scenario.messages_up_to_turn(terminal_turn))
            base_seed = cfg.seed + hash(name) % 10_000

            results["baseline"][name] = run_trials(
                bundle, prompt, cfg.generation,
                prompt_id=name, condition="baseline", base_seed=base_seed,
                metadata={"category": scenario.category},
            ).to_dict()

            for a in coeffs:
                if a == 0.0:
                    continue
                cond = f"alpha_{a:g}"
                steering = ResidualSteering(bundle, direction, steer_layers,
                                            alpha=a, positions=cfg.intervention.positions)
                with applied(steering):
                    ts = run_trials(
                        bundle, prompt, cfg.generation,
                        prompt_id=name, condition=cond, base_seed=base_seed,
                        metadata={"category": scenario.category, "alpha": a},
                    )
                results[cond][name] = ts.to_dict()

        run_ctx.save_json("steering_results.json", {"conditions": conditions, "results": results})
        comparison = _compare(results, scenario_names, conditions, cfg)
        run_ctx.save_json("comparison.json", comparison)
        run_ctx.save_text("summary.md",
                          _summary_md(results, scenario_names, conditions, comparison, best_layer))

        for cond in conditions:
            run_ctx.record_metric(f"{cond}_asr", round(comparison["condition_asr"][cond]["value"], 3))
        run_ctx.record_metric("best_layer", best_layer)
        run_ctx.log.info("baseline ASR %.0f%% -> best %s",
                         100 * comparison["condition_asr"]["baseline"]["value"],
                         comparison["best_condition"])
        return comparison


def _compare(results, scenario_names, conditions, cfg) -> dict[str, Any]:
    def flags(cond: str) -> list[bool]:
        return [results[cond][n]["any_success"] for n in scenario_names]

    baseline_flags = flags("baseline")
    condition_asr, vs_baseline = {}, {}
    for cond in conditions:
        f = flags(cond)
        condition_asr[cond] = wilson_interval(sum(f), len(f), cfg.eval.confidence).to_dict()
        if cond != "baseline":
            vs_baseline[cond] = mcnemar_paired(
                baseline_flags, f, n_samples=cfg.eval.bootstrap_samples,
                confidence=cfg.eval.confidence, seed=cfg.eval.seed).to_dict()

    best = min((c for c in conditions if c != "baseline"),
               key=lambda c: condition_asr[c]["value"], default="baseline")
    return {
        "n_scenarios": len(scenario_names),
        "condition_asr": condition_asr,
        "vs_baseline_mcnemar": vs_baseline,
        "best_condition": best,
        "note": ("Working defense: ASR decreases with alpha, unlike the multiplicative sweep "
                 "where it increased. Watch the NONRESPONSE share -- steering that only works by "
                 "degrading the model into gibberish is not a defense."),
    }


def _summary_md(results, scenario_names, conditions, comparison, best_layer) -> str:
    lines = [
        "# Additive (sign-correct) bipolar steering defense",
        "",
        f"Refusal direction: difference-of-means (harmful − benign), chosen layer **{best_layer}**. "
        "Per-cell = compliance rate over trials; `NR` = degenerate/non-response share.",
        "",
        "| scenario | " + " | ".join(conditions) + " |",
        "|" + "---|" * (len(conditions) + 1),
    ]
    for name in scenario_names:
        row = [name]
        for c in conditions:
            ts = results[c][name]
            nr = ts["verdict_counts"].get("nonresponse", 0)
            row.append(f"{ts['compliance_rate']:.0%}" + (f" ({nr}NR)" if nr else ""))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## ASR (any trial complies), 95% CI", "",
              "| condition | ASR | 95% CI | vs baseline |", "|---|---|---|---|"]
    for c in conditions:
        asr = comparison["condition_asr"][c]
        cmp = comparison["vs_baseline_mcnemar"].get(c)
        cmp_s = f"Δ={cmp['delta']:+.2f} p={cmp['p_value']:.3f}" if cmp else "—"
        lines.append(f"| {c} | {asr['value']:.0%} | [{asr['ci_low']:.0%}, {asr['ci_high']:.0%}] | {cmp_s} |")
    lines += ["", f"Best condition: **{comparison['best_condition']}**.", "", comparison["note"]]
    return "\n".join(lines) + "\n"
