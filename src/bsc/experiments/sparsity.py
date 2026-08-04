"""Recompute circuit sparsity for every available circuit map under one metric.

**Claim under test** (``RESEARCH_OVERVIEW.md`` §2.3): "the Crescendo override circuit is diffuse
-- the top 10 heads carry only 44% of the causal effect, versus ~85% for the GCG circuit."

``CLAIMS_AUDIT.md`` P0-1 found the 44% is real and reproducible, but the ~85% appears nowhere
except prose: no script computes it and no artifact stores it. This experiment computes the
sparsity of *every* circuit map in the repo with a single implementation
(:func:`bsc.circuits.sparsity_fraction`, which reproduces the published 0.4444 exactly), so the
comparison is finally apples-to-apples.

Needs no model and no GPU: it reads JSON. That is deliberate -- it is the cheapest possible fix
for the most falsifiable claim in the paper.

Writes:
    sparsity_comparison.json   per-map report, plus the pairwise GCG-vs-Crescendo contrasts
    summary.md                 human-readable table
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bsc.circuits import (
    CircuitMap,
    HeadScore,
    compare_sparsity,
    load_circuit_map,
    sparsity_fraction,
)
from bsc.config import ExperimentConfig
from bsc.hooks import Head
from bsc.runs import REPO_ROOT, RunContext

# Every circuit map in the repo, with the attack family it belongs to. Two maps exist per model
# with nothing recording which supersedes which (CLAIMS_AUDIT.md P1), so all are computed and
# the ambiguity is made visible rather than resolved silently.
KNOWN_MAPS: tuple[tuple[str, str, str], ...] = (
    ("gcg_7b_harm_contrastive", "gcg", "bipolar_defense_repo/code/circuit_map.json"),
    ("gcg_7b_adversarial_contrastive", "gcg", "7b_results/gcg_circuit_map.json"),
    ("gcg_1_5b_harm_contrastive", "gcg", "7b_results/circuit_map_1_5B.json"),
    ("gcg_1_5b_legacy", "gcg", "results_1.5b/circuit_map.json"),
)

# The Crescendo circuit must come from the raw per-head score arrays, NOT from
# crescendo_override_circuit_v3.json: that file stores only the 10 already-selected heads, so
# its top-10 fraction is trivially 1.0. Reading it as a full map is exactly the apples-to-oranges
# error that produced the unsupported published comparison in the first place.
CRESCENDO_SCORE_ARRAYS: tuple[str, ...] = (
    "crescendo_mech_interp/scores_Term_B_Technical.npy",
    "crescendo_mech_interp/scores_Term_C_Editor.npy",
)

# The value stored in crescendo_override_circuit_v3.json. Recomputing it is the regression test
# proving this implementation matches the one that produced the published number.
PUBLISHED_CRESCENDO_SPARSITY = 0.4444
PUBLISHED_GCG_SPARSITY_CLAIM = 0.85


def load_crescendo_scores(root: Path) -> tuple[list[float], list[str]]:
    """Mean per-head causal score across the valid Crescendo contrast pairs.

    Reproduces ``crescendo_circuit_discovery_v3.py``: scores are averaged over the pairs that
    passed the verification gate (base refuses AND corrupted complies), then flattened. Only 2
    of 3 pairs passed, so every Crescendo number rests on n=2 -- stated here rather than buried.
    """
    import numpy as np

    arrays, used = [], []
    for rel in CRESCENDO_SCORE_ARRAYS:
        path = root / rel
        if path.exists():
            arrays.append(np.load(path))
            used.append(rel)
    if not arrays:
        raise FileNotFoundError(f"no Crescendo score arrays found under {root}")

    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Crescendo score arrays have mismatched shapes: {shapes}")
    return np.mean(np.stack(arrays), axis=0).flatten().tolist(), used


def run(cfg: ExperimentConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    top_k = cfg.circuit.top_k_refusal

    with RunContext.create("sparsity", cfg, notes=cfg.notes) as run_ctx:
        maps: dict[str, CircuitMap] = {}
        families: dict[str, str] = {}
        missing: list[dict[str, str]] = []

        for name, family, rel in KNOWN_MAPS:
            path = root / rel
            if not path.exists():
                run_ctx.log.warning("missing circuit map: %s", rel)
                missing.append({"name": name, "path": rel})
                continue
            maps[name] = load_circuit_map(path)
            families[name] = family
            run_ctx.log.info("loaded %-34s %4d heads  method=%s", name, len(maps[name].scores), maps[name].method)

        if not maps:
            raise FileNotFoundError(f"no circuit maps found under {root}")

        # Crescendo enters from raw scores, as a synthetic full map (see CRESCENDO_SCORE_ARRAYS).
        crescendo_scores, crescendo_sources = load_crescendo_scores(root)
        maps["crescendo_override_v3"] = CircuitMap(
            scores=[
                HeadScore(Head(i // 100, i % 100), s) for i, s in enumerate(crescendo_scores)
            ],
            model="Qwen/Qwen2.5-1.5B-Instruct",
            method="context_override_patching (raw scores)",
            source_path=", ".join(crescendo_sources),
        )
        families["crescendo_override_v3"] = "crescendo"
        run_ctx.log.info(
            "loaded %-34s %4d heads  from %d raw score arrays",
            "crescendo_override_v3",
            len(crescendo_scores),
            len(crescendo_sources),
        )

        comparison = compare_sparsity(maps, top_k=top_k)

        # Regression check: the published 0.4444 used k=10, so verify at k=10 regardless of the
        # configured top_k. If this stops matching, this implementation has drifted from the one
        # that produced the published number and every value in the table is suspect.
        recomputed = sparsity_fraction(maps["crescendo_override_v3"], top_k=10).top_k_fraction
        verification = {
            "published_crescendo_sparsity": PUBLISHED_CRESCENDO_SPARSITY,
            "recomputed_at_k10": recomputed,
            "matches": abs(recomputed - PUBLISHED_CRESCENDO_SPARSITY) < 5e-4,
            "n_contrast_pairs": len(crescendo_sources),
            "sources": crescendo_sources,
        }
        run_ctx.log.info(
            "crescendo sparsity: published %.4f, recomputed %.4f -> %s",
            PUBLISHED_CRESCENDO_SPARSITY,
            recomputed,
            "MATCH" if verification["matches"] else "MISMATCH",
        )
        run_ctx.record_metric("crescendo_regression_match", verification["matches"])

        # Every GCG-vs-Crescendo pairing, because the published contrast inverts depending on
        # which 7B map is used, and the paper must state which one it means.
        contrasts = []
        crescendo = [n for n, f in families.items() if f == "crescendo"]
        for gcg_name in (n for n, f in families.items() if f == "gcg"):
            for cres_name in crescendo:
                g = comparison["reports"][gcg_name]["top_k_fraction"]
                c = comparison["reports"][cres_name]["top_k_fraction"]
                contrasts.append(
                    {
                        "gcg_map": gcg_name,
                        "crescendo_map": cres_name,
                        "gcg_fraction": g,
                        "crescendo_fraction": c,
                        "gcg_more_sparse": g > c,
                        "gap": g - c,
                    }
                )

        inverted = [c for c in contrasts if not c["gcg_more_sparse"]]
        payload = {
            "claim_under_test": (
                "Crescendo circuit is diffuse (44% in top-10) vs GCG sparse (~85%)"
            ),
            "published_gcg_claim": PUBLISHED_GCG_SPARSITY_CLAIM,
            "top_k": top_k,
            "comparison": comparison,
            "families": families,
            "verification": verification,
            "contrasts": contrasts,
            "n_inverted_contrasts": len(inverted),
            "missing_maps": missing,
        }

        run_ctx.save_json("sparsity_comparison.json", payload)
        run_ctx.save_text("summary.md", _summary_markdown(payload, comparison, top_k))

        for name, report in comparison["reports"].items():
            run_ctx.record_metric(f"{name}_top{top_k}", round(report["top_k_fraction"], 4))
        run_ctx.record_metric("n_inverted_contrasts", len(inverted))

        best_gcg = max(
            (r["top_k_fraction"] for n, r in comparison["reports"].items() if families[n] == "gcg"),
            default=float("nan"),
        )
        run_ctx.log.info(
            "highest GCG sparsity found: %.4f vs published claim of %.2f",
            best_gcg,
            PUBLISHED_GCG_SPARSITY_CLAIM,
        )
        return payload


def _summary_markdown(payload: dict[str, Any], comparison: dict[str, Any], top_k: int) -> str:
    lines = [
        "# Circuit sparsity, recomputed under a single metric",
        "",
        f"Metric: fraction of total **positive** causal mass carried by the top-{top_k} heads,",
        "matching `crescendo_circuit_discovery_v3.py:243-250`.",
        "",
        "| circuit map | family | method | heads | top-k fraction | interpretation |",
        "|---|---|---|---|---|---|",
    ]
    for name, report in sorted(
        comparison["reports"].items(), key=lambda kv: -kv[1]["top_k_fraction"]
    ):
        lines.append(
            f"| `{name}` | {payload['families'][name]} | {comparison['methods'][name] or 'n/a'} "
            f"| {report['n_heads']} | **{report['top_k_fraction']:.4f}** | {report['interpretation']} |"
        )

    verification = payload.get("verification") or {}
    if verification:
        lines += [
            "",
            "## Regression check",
            "",
            f"Published Crescendo sparsity `{verification['published_crescendo_sparsity']}` vs "
            f"recomputed at k=10 `{verification['recomputed_at_k10']:.4f}` -> "
            f"**{'MATCH' if verification['matches'] else 'MISMATCH'}**.",
            "",
            f"Averaged over **{verification['n_contrast_pairs']} contrast pairs** "
            "(only 2 of 3 passed the verification gate) - every Crescendo number rests on n=2.",
        ]

    lines += ["", "## GCG vs Crescendo contrasts", "", "| GCG map | GCG | Crescendo | gap | GCG sparser? |", "|---|---|---|---|---|"]
    for c in payload["contrasts"]:
        lines.append(
            f"| `{c['gcg_map']}` | {c['gcg_fraction']:.4f} | {c['crescendo_fraction']:.4f} "
            f"| {c['gap']:+.4f} | {'yes' if c['gcg_more_sparse'] else '**NO - INVERTED**'} |"
        )

    lines += [
        "",
        f"Published claim: GCG ~{payload['published_gcg_claim']:.0%}. "
        f"Inverted contrasts: {payload['n_inverted_contrasts']} of {len(payload['contrasts'])}.",
    ]
    return "\n".join(lines) + "\n"
