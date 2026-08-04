"""The *actual* bipolar defense: per-head refusal steering + compliance-head suppression.

The steering defense that worked in `steering_defense` used a single residual-stream direction and
touched no heads — it is not "bipolar". The head-based bipolar defense tested so far was the
*multiplicative* one, which failed (sign-inverted head). This experiment tests the version that
uses **both head sets the correct way**:

* **Refusal heads:** add a per-head, correctly-signed steering vector (difference of means of the
  head's o_proj-input slice between refusing and complying runs). Because it is the measured
  activation difference, each head's sign is empirically correct — unlike multiplicative
  amplification, which assumed a sign and hit the inverted head.
* **Compliance heads:** zero-ablate.

This is the mechanism behind the legacy 7B 66%->33% GCG result. Here we test it on Crescendo and
decompose it, so we can say whether the compliance-head ablation adds anything on top of the
refusal-head steering, and whether the head-based bipolar defense matches the simpler
single-direction one.

Conditions: baseline, compliance_ablate (heads only), refusal_steer_{alpha} (refusal heads only),
bipolar_{alpha} (both). Same paired/seeded/degeneracy-aware protocol as the other sweeps.

Writes: head_steering_vectors.json (norms), results.json, comparison.json, summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bsc.circuits import load_circuit_map
from bsc.config import ExperimentConfig
from bsc.data import DATA_DIR, get_scenario, load_crescendo_scenarios
from bsc.determinism import prompt_seed
from bsc.generation import build_chat_prompt, run_trials
from bsc.hooks import Head, HeadEdit, HeadIntervention, applied, build_bipolar_edits
from bsc.metrics import mcnemar_paired, wilson_interval
from bsc.models import ModelBundle, load_model
from bsc.runs import REPO_ROOT, RunContext

DEFAULT_ALPHAS = (4.0, 8.0, 16.0)


def _contrast_prompts(bundle: ModelBundle, n: int) -> tuple[list[str], list[str]]:
    with (DATA_DIR / "advbench.csv").open(encoding="utf-8") as f:
        harmful = [r["goal"] for r in csv.DictReader(f)]
    benign = json.loads((DATA_DIR / "benign_prompts.json").read_text(encoding="utf-8"))["prompts"]
    n = min(n, len(harmful), len(benign))
    fmt = lambda t: bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
    )
    return [fmt(h) for h in harmful[:n]], [fmt(b) for b in benign[:n]]


def _extract_head_steering_vectors(
    bundle: ModelBundle, refusal_heads: list[Head], n_pairs: int
):
    """Per-head refusal steering vector = mean(o_proj-input slice | refusing) − mean(| complying).

    Points from the complying activation to the refusing one for each head, so adding a positive
    multiple steers that head toward refusal — with the sign taken from data, not assumed.
    """
    import torch

    harmful, benign = _contrast_prompts(bundle, n_pairs)
    layers = sorted({h.layer for h in refusal_heads})
    geometry = bundle.geometry

    def collect(prompts: list[str]) -> dict[int, torch.Tensor]:
        sums: dict[int, torch.Tensor] = {}
        for prompt in prompts:
            captured: dict[int, torch.Tensor] = {}

            def cap(layer: int):
                def hook(module, args):
                    captured[layer] = args[0][..., -1, :].detach().float().clone()
                return hook

            handles = [bundle.o_proj(l).register_forward_pre_hook(cap(l)) for l in layers]
            try:
                with torch.no_grad():
                    bundle.model(**bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device))
            finally:
                for h in handles:
                    h.remove()
            for l, act in captured.items():
                sums[l] = act if l not in sums else sums[l] + act
        return {l: v / len(prompts) for l, v in sums.items()}

    mean_refuse = collect(harmful)
    mean_comply = collect(benign)

    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    for head in refusal_heads:
        cols = geometry.head_slice(head.head)
        diff = (mean_refuse[head.layer] - mean_comply[head.layer])[..., cols].flatten()
        unit = diff / (diff.norm() + 1e-8)  # unit per head, so alpha is comparable across heads
        vectors[str(head)] = unit.to(bundle.dtype)
        norms[str(head)] = float(diff.norm())
    return vectors, norms


def _run_condition(bundle, scenario_names, cfg, condition, edits_fn, terminal_turn=5):
    out = {}
    for name in scenario_names:
        scenario = get_scenario(name)
        prompt = build_chat_prompt(bundle, scenario.messages_up_to_turn(terminal_turn))
        base_seed = prompt_seed(cfg.seed, name)
        edits = edits_fn()
        if edits is None:  # baseline
            out[name] = run_trials(bundle, prompt, cfg.generation, prompt_id=name,
                                   condition=condition, base_seed=base_seed).to_dict()
        else:
            with applied(HeadIntervention(bundle, edits, positions=cfg.intervention.positions)):
                out[name] = run_trials(bundle, prompt, cfg.generation, prompt_id=name,
                                       condition=condition, base_seed=base_seed).to_dict()
    return out


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    alphas = tuple(cfg.intervention_sweep) if cfg.intervention_sweep else DEFAULT_ALPHAS
    scenario_names = sorted(load_crescendo_scenarios())

    with RunContext.create("bipolar_steering", cfg, notes=cfg.notes) as run_ctx:
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

        vectors, norms = _extract_head_steering_vectors(bundle, refusal_heads, cfg.data.limit or 16)
        run_ctx.save_json("head_steering_vectors.json",
                          {"refusal_heads": [str(h) for h in refusal_heads],
                           "compliance_heads": [str(h) for h in compliance_heads],
                           "per_head_diff_norm": norms})

        results: dict[str, dict[str, Any]] = {}
        # baseline
        results["baseline"] = _run_condition(bundle, scenario_names, cfg, "baseline", lambda: None)
        # compliance ablation only (heads, no alpha)
        results["compliance_ablate"] = _run_condition(
            bundle, scenario_names, cfg, "compliance_ablate",
            lambda: {h: HeadEdit(scale=0.0) for h in compliance_heads})
        # per alpha: refusal-steer-only and full bipolar
        for a in alphas:
            results[f"refusal_steer_{a:g}"] = _run_condition(
                bundle, scenario_names, cfg, f"refusal_steer_{a:g}",
                lambda a=a: build_bipolar_edits(refusal_heads, [], mode="additive_steering",
                                                steering_alpha=a, steering_vectors=vectors,
                                                ablate_compliance=False))
            results[f"bipolar_{a:g}"] = _run_condition(
                bundle, scenario_names, cfg, f"bipolar_{a:g}",
                lambda a=a: build_bipolar_edits(refusal_heads, compliance_heads,
                                                mode="additive_steering", steering_alpha=a,
                                                steering_vectors=vectors, ablate_compliance=True))
            run_ctx.log.info("alpha=%g done", a)

        run_ctx.save_json("results.json", results)
        comparison = _compare(results, scenario_names, cfg)
        run_ctx.save_json("comparison.json", comparison)
        run_ctx.save_text("summary.md", _summary_md(results, scenario_names, comparison))
        for cond, asr in comparison["condition_asr"].items():
            run_ctx.record_metric(f"{cond}_asr", round(asr["value"], 3))
        run_ctx.log.info("baseline ASR %.0f%% -> best %s (%.0f%%)",
                         100 * comparison["condition_asr"]["baseline"]["value"],
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
    lines = [
        "# The actual bipolar defense: per-head refusal steering + compliance ablation",
        "",
        "Isolates the two head sets. `refusal_steer_*` = additive per-head steering on refusal "
        "heads only; `bipolar_*` = that PLUS zero-ablating compliance heads; `compliance_ablate` "
        "= compliance heads only. Per-cell = compliance rate; `NR` = degenerate share.",
        "",
        "| scenario | " + " | ".join(results.keys()) + " |",
        "|" + "---|" * (len(results) + 1),
    ]
    for name in scenario_names:
        row = [name]
        for c in results:
            ts = results[c][name]
            nr = ts["verdict_counts"].get("nonresponse", 0)
            row.append(f"{ts['compliance_rate']:.0%}" + (f" ({nr}NR)" if nr else ""))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## ASR (any trial complies), 95% CI", "",
              "| condition | ASR | 95% CI | vs baseline |", "|---|---|---|---|"]
    for c, a in comparison["condition_asr"].items():
        v = comparison["vs_baseline"].get(c)
        vs = f"Δ={v['delta']:+.2f} p={v['p_value']:.3f}" if v else "—"
        lines.append(f"| {c} | {a['value']:.0%} | [{a['ci_low']:.0%}, {a['ci_high']:.0%}] | {vs} |")
    lines += ["", f"Best: **{comparison['best_condition']}**.",
              "", "Read: does `bipolar_*` beat `refusal_steer_*` (i.e. does suppressing compliance "
              "heads add anything over steering refusal heads)? And does either head-based variant "
              "match the single residual-direction defense from `steering_defense`?"]
    return "\n".join(lines) + "\n"
