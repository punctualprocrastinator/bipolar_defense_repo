"""Calibrate the compliance-head intervention: steer vs zero-ablate.

Earlier the compliance heads were only *zero-ablated*, which did not help (ASR >= baseline). But
zeroing just removes a head's contribution; if compliance heads *actively write toward
compliance*, the right lever is to steer them the other way with a calibrated direction and
strength — exactly as the refusal heads are steered. This experiment tests that.

Per-head direction: the unit difference-of-means `mean(refusing) - mean(complying)` at each head's
o_proj-input slice — the same construction used for refusal heads, so "add +beta" always steers a
head toward refusal regardless of which population it belongs to. For a compliance head that is
"subtract its compliance component"; for a refusal head it is "reinforce refusal".

Conditions decompose the question:
* baseline
* comp_zero — the old zero-ablation, for comparison
* comp_steer_{beta} — steer ONLY the compliance heads toward refusal (refusal heads untouched)
* refusal_only — steer only the refusal heads (reference)
* combined — steer both

If comp_steer beats comp_zero, the compliance heads are a usable defense lever after all — just via
calibrated steering, not ablation. If comp_steer alone does little while refusal steering carries
it, the earlier "refusal side does the work" conclusion holds even under proper calibration.

Writes: results.json, comparison.json, summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bsc.circuits import load_circuit_map
from bsc.config import ExperimentConfig
from bsc.data import get_scenario, load_crescendo_scenarios
from bsc.determinism import prompt_seed
from bsc.experiments.bipolar_steering import _extract_head_steering_vectors
from bsc.generation import build_chat_prompt, run_trials
from bsc.hooks import HeadEdit, HeadIntervention, applied
from bsc.metrics import mcnemar_paired, wilson_interval
from bsc.runs import REPO_ROOT, RunContext

DEFAULT_BETAS = (4.0, 8.0, 16.0)
REFERENCE_ALPHA = 8.0  # refusal-head steering strength used in the reference/combined conditions


def _run(bundle, scenario_names, cfg, condition, edits, terminal_turn=5):
    out = {}
    for name in scenario_names:
        scenario = get_scenario(name)
        prompt = build_chat_prompt(bundle, scenario.messages_up_to_turn(terminal_turn))
        base_seed = prompt_seed(cfg.seed, name)
        if edits is None:
            out[name] = run_trials(bundle, prompt, cfg.generation, prompt_id=name,
                                   condition=condition, base_seed=base_seed).to_dict()
        else:
            with applied(HeadIntervention(bundle, edits, positions=cfg.intervention.positions)):
                out[name] = run_trials(bundle, prompt, cfg.generation, prompt_id=name,
                                       condition=condition, base_seed=base_seed).to_dict()
    return out


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    from bsc.models import load_model

    root = repo_root or REPO_ROOT
    betas = tuple(cfg.intervention_sweep) if cfg.intervention_sweep else DEFAULT_BETAS
    scenario_names = sorted(load_crescendo_scenarios())

    with RunContext.create("compliance_calibration", cfg, notes=cfg.notes) as run_ctx:
        circuit_path = Path(cfg.circuit.circuit_map_path)
        if not circuit_path.is_absolute():
            circuit_path = root / circuit_path
        circuit = load_circuit_map(circuit_path)
        refusal_heads, compliance_heads = circuit.select(
            selection=cfg.circuit.selection,
            top_k_refusal=cfg.circuit.top_k_refusal,
            top_k_compliance=cfg.circuit.top_k_compliance,
        )

        bundle = load_model(cfg.model)
        run_ctx.save_json("model.json", bundle.describe())

        # Per-head unit refuse-comply direction for EVERY head we might steer.
        all_heads = list(refusal_heads) + list(compliance_heads)
        vectors, norms = _extract_head_steering_vectors(bundle, all_heads, cfg.data.limit or 16)
        run_ctx.save_json("head_directions.json",
                          {"refusal_heads": [str(h) for h in refusal_heads],
                           "compliance_heads": [str(h) for h in compliance_heads],
                           "per_head_diff_norm": norms})

        def comp_steer(beta):
            return {h: HeadEdit(alpha=beta, vector=vectors[str(h)]) for h in compliance_heads}

        def refusal_steer(alpha):
            return {h: HeadEdit(alpha=alpha, vector=vectors[str(h)]) for h in refusal_heads}

        results: dict[str, dict[str, Any]] = {}
        results["baseline"] = _run(bundle, scenario_names, cfg, "baseline", None)
        results["comp_zero"] = _run(bundle, scenario_names, cfg, "comp_zero",
                                    {h: HeadEdit(scale=0.0) for h in compliance_heads})
        results["refusal_only"] = _run(bundle, scenario_names, cfg, "refusal_only",
                                       refusal_steer(REFERENCE_ALPHA))
        for b in betas:
            results[f"comp_steer_{b:g}"] = _run(bundle, scenario_names, cfg,
                                                f"comp_steer_{b:g}", comp_steer(b))
            combined = {**refusal_steer(REFERENCE_ALPHA), **comp_steer(b)}
            results[f"combined_a{REFERENCE_ALPHA:g}_b{b:g}"] = _run(
                bundle, scenario_names, cfg, f"combined_a{REFERENCE_ALPHA:g}_b{b:g}", combined)
            run_ctx.log.info("beta=%g done", b)

        run_ctx.save_json("results.json", results)
        comparison = _compare(results, scenario_names, cfg)
        run_ctx.save_json("comparison.json", comparison)
        run_ctx.save_text("summary.md", _summary_md(results, scenario_names, comparison))
        for cond, a in comparison["condition_asr"].items():
            run_ctx.record_metric(f"{cond}_asr", round(a["value"], 3))
        run_ctx.log.info("baseline %.0f%% | comp_zero %.0f%% | best %s %.0f%%",
                         100 * comparison["condition_asr"]["baseline"]["value"],
                         100 * comparison["condition_asr"]["comp_zero"]["value"],
                         comparison["best_condition"],
                         100 * comparison["condition_asr"][comparison["best_condition"]]["value"])
        return comparison


def _compare(results, scenario_names, cfg):
    def flags(cond):
        return [results[cond][n]["any_success"] for n in scenario_names]
    base = flags("baseline")
    asr, vs = {}, {}
    for cond in results:
        f = flags(cond)
        asr[cond] = wilson_interval(sum(f), len(f), cfg.eval.confidence).to_dict()
        if cond != "baseline":
            vs[cond] = mcnemar_paired(base, f, n_samples=cfg.eval.bootstrap_samples,
                                      confidence=cfg.eval.confidence, seed=cfg.eval.seed).to_dict()
    best = min((c for c in results if c != "baseline"), key=lambda c: asr[c]["value"], default="baseline")
    return {"condition_asr": asr, "vs_baseline": vs, "best_condition": best}


def _summary_md(results, scenario_names, comparison):
    def coherence(cond):
        ref = deg = comp = 0
        for r in results[cond].values():
            for t in r["trials"]:
                v = t["judgement"]["verdict"]
                ref += v == "refusal"; deg += v == "nonresponse"; comp += v == "compliance"
        return ref, deg, comp

    lines = [
        "# Calibrating the compliance-head intervention: steer vs zero-ablate",
        "",
        "`comp_zero` = zero-ablate compliance heads (old). `comp_steer_b` = steer ONLY compliance "
        "heads toward refusal at strength b. `refusal_only` = steer only refusal heads (alpha=8). "
        "`combined` = both. Key question: does calibrated compliance-head **steering** beat "
        "**zeroing** them?",
        "",
        "## ASR (any trial complies) and coherence",
        "| condition | ASR | 95% CI | vs baseline | refusal/degenerate/compliance trials |",
        "|---|---|---|---|---|",
    ]
    for c, a in comparison["condition_asr"].items():
        v = comparison["vs_baseline"].get(c)
        vs = f"Δ={v['delta']:+.2f} p={v['p_value']:.3f}" if v else "—"
        ref, deg, comp = coherence(c)
        lines.append(f"| {c} | {a['value']:.0%} | [{a['ci_low']:.0%}, {a['ci_high']:.0%}] | {vs} "
                     f"| {ref}/{deg}/{comp} |")
    lines += ["", f"Best: **{comparison['best_condition']}**.", "",
              "Read `comp_steer_*` vs `comp_zero`: if steering the compliance heads (rather than "
              "zeroing them) reduces ASR while keeping outputs coherent (high refusal / low "
              "degenerate), the compliance side IS a usable lever under proper calibration."]
    return "\n".join(lines) + "\n"
