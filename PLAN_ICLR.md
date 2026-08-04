# Plan — ICLR main submission

Dated 2026-08-04. The target is an ICLR **main-conference** paper, not a workshop. This plan is
built on the novelty review (see bottom) and the results already in `FINDINGS.md`.

## 0. The thesis (locked)

**Do NOT frame this as "we discovered a bipolar refusal/compliance circuit."** That is already
published — *The Struggle Between Continuation and Refusal* (arXiv 2603.08234) identifies opposing
safety vs continuation heads and shows scaling them in opposite directions helps/hurts safety, and
*Attention Head Specialization* (2606.28153) independently finds two opposing head types. A
discovery framing gets desk-rejected on novelty.

**Frame it as a defense + generalization paper:**

> *"The first calibrated, deployable bipolar defense — steering refusal and compliance heads in
> their (opposite) correct directions — and evidence that a single such intervention transfers
> across attack families (single-turn GCG and multi-turn Crescendo) and across architectures
> (GQA, MHA, and Mixture-of-Experts)."*

The mechanism (opposing heads) is cited background. Our four novel legs, none of which any single
prior paper combines:
1. A **working, calibrated, additive both-sides defense** with significance (89→34% ASR, p<0.001;
   additive over one-sided), not just an analysis probe.
2. **Cross-attack transfer**: the same direction defends GCG (46→0%) and Crescendo.
3. **Cross-architecture generalization** including **MoE** (OLMoE) and MHA (OLMo) — prior work is
   1–2 dense 7B models.
4. **Sign inversion**: a head labelled "refusal" by activation patching writes *toward compliance*
   in the unembedding, so naive amplification backfires — a crisp cautionary result.

## 1. Claims to DROP or de-emphasise (honesty gate)

- ❌ "We discovered bipolar circuits" → scooped. Cite, don't claim.
- ❌ Multi-turn KV-cache "85% composed of model's own output" → measured 10.5% (CLAIMS_AUDIT P0-2).
  Do not lead with a multi-turn *mechanism* claim; a rep-eng paper (2507.02956) already covers
  Crescendo representations. Lead with the *defense that transfers*.
- ❌ Sparsity "~85%" and the mislabelled Cohen's d → already corrected in CLAIMS_AUDIT.
- ⚠️ Keyword-judge ASR is an upper bound (we found this ourselves) → must be backed by an LLM judge.

## 2. Work plan (≈4 focused weeks)

### Phase A — Lock positioning (2 days, no compute) ← DO FIRST
- Write the related-work section against the 8 papers in the review below; state the delta
  explicitly in a table.
- Freeze the claim set (§0, §1). Everything downstream cites this.
- *Why first:* an hour of positioning prevents weeks of running the wrong experiments.

### Phase B — Make the core defense bulletproof (1 week) ← HIGHEST LEVERAGE
1. **LLM judge** (`bsc.judge` method="llm"): validate keyword ASR, report agreement rate, rerun
   the headline numbers under it. This is a prerequisite for every ASR claim.
2. **Baselines — the biggest missing piece.** Implement and compare on the same benchmark/judge:
   - Circuit Breakers / RepBend (representation rerouting)
   - SmoothLLM (input perturbation)
   - Self-reminder / system-prompt defense
   - The *Struggle* paper's head-scaling intervention (the closest prior method)
   The bipolar defense must **win or match at lower cost/coherence loss**. If it doesn't, we learn
   that now, not in review.
3. **HarmBench-scale single-turn eval** (300+ behaviours) replacing custom sets.
4. **RMS-corrected DLA** for the sign-inversion result across models (drop the artifact-prone
   isolated-RMSNorm version).

### Phase C — Generalization evidence (1 week) ← THE NOVELTY CORE
1. **Cross-model study**: Qwen2.5-{1.5B,7B}, Llama-3.1-8B, OLMo-2-7B, OLMoE-1B-7B, Gemma-2-9B,
   + one 32B. For each: circuit → sign check → calibrate α/β → bipolar-vs-one-sided defense.
   (~8–10 h sequential / ~4–5 h parallel; the report of record for "generalizes across
   architectures".) Gemma is the head_dim geometry check; OLMoE is the MoE check.
2. **Third attack family** (PAIR or AutoDAN) with the *same* defense → strengthens "transfers
   across attacks" beyond GCG+Crescendo.
3. Grow the harvested benchmark per target (7B/8B are harder to jailbreak than 1.5B).

### Phase D — Adversarial robustness (3–4 days)
- **Adaptive white-box attacker** against the *defended* model (knows the steering params). Report
  ASR under adaptation. Standard security-eval requirement; its absence is an easy reject.

### Phase E — Write-up (1 week)
- Figures generated from run artifacts only (`bsc.paper.collect` → numbers.json → LaTeX).
- Ablations: refusal-only vs compliance-only vs bipolar; α/β sensitivity; per-architecture.
- Reproducibility appendix (the framework already gives per-run manifests + seeds).

## 3. Priority order (if compute/time is limited)

1. **Positioning section** (Phase A) — cheap, prevents wasted work.
2. **Baselines** (Phase B.2) — the make-or-break: does bipolar beat Circuit Breakers?
3. **LLM judge** (Phase B.1) — every number depends on it.
4. **Cross-model study** (Phase C.1) — the generalization headline.
5. **Adaptive attacker** (Phase D) — the robustness gate.
6. Everything else.

The two that most change the paper's fate: **baselines** (is the defense actually competitive?)
and **cross-model generalization** (is it more than a 2-model curiosity?). Do those before writing.

## 4. What we already have (assets, verified this session)

- Reproducible `bsc` framework (14 experiments, 92 tests, per-run manifests, stable seeds).
- Additive bipolar defense: Crescendo **89→34%** (n=35, p<0.001, additive), GCG **46→0%**.
- Cross-attack transfer (one direction defends both).
- Cross-architecture circuits: compliance heads present in GQA/MHA/**MoE**; dominant in OLMo family.
- **Compliance-head sign inversion** (defends only with the sign opposite the refusal heads).
- Validated scenario harvester (10→35).
- Full claims audit (P0 fixes) + degeneration-aware judge fix.

## 5. Related work to position against (from the 2026-08-04 review)

| Paper | What they do | Our delta |
|---|---|---|
| The Struggle Between Continuation and Refusal (2603.08234) | Opposing safety/continuation heads; opposite-scaling analysis; Llama-2-7B + Qwen2.5-7B | We build a **deployed calibrated defense**, add **multi-turn**, **cross-attack transfer**, **MoE/cross-arch**, and **sign inversion** |
| Attention Head Specialization (2606.28153) | Two opposing head types (ACH/SAH); detection | We do steering **defense** + generalization, not detection |
| Zhou et al. SAHARA | Safety attention heads (refusal side) | We add the **compliance side as an active, oppositely-signed lever** + a defense |
| Arditi et al. (NeurIPS'24) | Single refusal **direction** | We decompose to **heads**, find the **opposing** side, and the sign inversion |
| Attention Slipping (2507.04365) | Attention drifts from unsafe prototype under attack | Different mechanism; we intervene on head OV, not attention scores |
| Rep-Eng Multi-Turn (2507.02956) | Crescendo represented as benign; Llama-3-8B | We give a **head-level defense** that transfers to multi-turn, not just a representation account |
| There Is More to Refusal… (2602.02132) / concept cones | Multiple refusal directions | Compatible; we work at head granularity with a defense |
| Steering Externalities (2602.04896) | Benign steering can *increase* jailbreak risk | Consistent with our **sign-inversion** caution; we characterise it at head level |

**Bottom line:** as a discovery paper this is scooped; as a *defense-that-generalizes* paper with
baselines + cross-model + adaptive-attacker evidence, it is viable for ICLR main.
