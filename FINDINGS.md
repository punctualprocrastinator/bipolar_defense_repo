# Findings — new_bipolar_study

Dated, append-only research log. Every entry points to the run directory that produced it.
Negative and non-reproducing results stay here; they are superseded by later dated entries, not
deleted (CLAUDE.md §2, §7).

All runs below executed on an NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 95 GB),
torch 2.11.0+cu130, transformers 5.14.1, seed 0. Full provenance in each run's `manifest.json`.

---

## 2026-08-04 — P0-4 SETTLED: the multiplicative bipolar "defense" *increases* attack success

**Claim retested:** "the 3× bipolar defense (amplify refusal heads ×3, zero-ablate compliance
heads) blocks the harmful Turn-5 request" (`RESEARCH_OVERVIEW.md` §2.3). `CLAIMS_AUDIT.md` P0-4
flagged that it rested on a single greedy sample and did not reproduce; the sweep that would
settle it had never been run. It has now been run.

### Result 1 — behavioral multiplier sweep (the ground truth)

Qwen2.5-1.5B, 10 Crescendo scenarios, 8 sampled trials each, paired seeds, degeneracy-aware
judge. Run: `runs/crescendo_sweep/20260804T141340Z-fb496d884b`.

| condition | ASR (any trial complies) | 95% CI | vs baseline |
|---|---|---|---|
| baseline (no intervention) | 70% | [40%, 89%] | — |
| mult_1 (compliance-ablate only) | 70% | [40%, 89%] | Δ=0.00, p=1.00 |
| mult_3 (**the published defense**) | **90%** | [60%, 98%] | Δ=+0.20 |
| mult_6 | 100% | [72%, 100%] | Δ=+0.30 |
| mult_12 | 100% | [72%, 100%] | Δ=+0.30 |
| mult_24 | 100% | [72%, 100%] | Δ=+0.30 |

**The defense does the opposite of what was claimed.** ASR rises monotonically with the
multiplier, from 70% undefended to 100% at 6×. At the published 3× setting ASR is 90%, *higher*
than baseline. The per-scenario table shows the mechanism turning on sharply at 6×, where every
one of the 10 scenarios hits 100% compliance (partly genuine compliance, partly degeneration —
the judge flags the NONRESPONSE share, e.g. RDX at 24× is 4/8 NR). The effect is not yet
individually significant at n=10 scenarios (McNemar p=0.25), but the direction is unambiguous
and consistent across every scenario.

The original "3× blocks Turn 5" claim is **retracted**. It was a single greedy sample; under
8-trial sampling the same intervention raises attack success.

### Result 2 — mechanism: the dominant refusal head is sign-inverted

Logit-lens sign check: project each refusal head's isolated contribution through the
unembedding and compare promoted mass on refusal vs compliance tokens.
Runs: `runs/logit_lens_sign/…` (1.5B `-1a2104bec4`, 7B `-0b40f5bc60`).

| Model | Dominant head (circuit score) | Sign score (refusal − compliance logit) | Inverted? | Heads inverted |
|---|---|---|---|---|
| Qwen2.5-1.5B | L23-H11 (+0.74) | **−1.19** | yes — top tokens include " complying", 作为 | 5 of 10 |
| Qwen2.5-7B | L25-H1 (+1.87) | **−7.44** | yes | 6 of 11 |

The heads selected as "refusal" by activation patching write *toward* compliance in the
unembedding basis. Amplifying them (`hidden *= multiplier`) therefore strengthens compliance —
exactly the behavioral result above.

**Caveat (honest):** this single-head DLA applies the final RMSNorm to each head's *isolated*
contribution, which distorts the top-promoted-token strings (the 7B top tokens come out as
'while'/code fragments, not clean 'Sure'/'Certainly'). The **sign_score over the curated token
sets is robust to this**; the specific top-token strings are not and should not be quoted
without the RMS-real correction (the legacy `dla_decomposition_poc.py` documents that fix). The
behavioral sweep is the primary evidence; DLA is corroborating mechanism.

### Takeaway (the publishable result)

A head identified as "refusal" by causal activation patching can write *toward compliance* in
the unembedding basis. An inference-time defense that amplifies such a head along an unexamined
sign convention **degrades safety monotonically** rather than improving it. This is a stronger
and more interesting mechanistic claim than the original defense: it is a concrete failure mode
of steering-by-amplification, reproducible and quantified. The correct defense direction is the
open question (ablate-only `mult_1` at least does no harm; additive steering along the *measured*
refusal direction is the natural next test).

---

## 2026-08-04 — Cross-architecture circuit discovery: the compliance side is not a GQA artifact

**Question:** is the bipolar refusal/compliance structure a property of safety tuning, or an
artifact of Qwen + grouped-query attention? Tested by rediscovering circuits from scratch,
including an MHA (non-GQA) control. Harm-contrastive patching, 8 (harmful, benign) pairs.
Runs: `runs/discover_circuit/…`.

| Model | Attn | Top refusal head (score) | Top compliance head (score) |
|---|---|---|---|
| Qwen2.5-1.5B | GQA 6:1 | L16-H6 (+0.24) | L15-H0 (−0.36) |
| OLMo-2-7B | **MHA 1:1** | L14-H11 (+0.14) | L30-H14 (−0.38) |

**The compliance-head phenomenon survives in a multi-head-attention model with no grouped-query
sharing** — and on OLMo the compliance heads are *stronger* in magnitude than the refusal heads.
So "compliance heads that actively push toward answering" is not a consequence of GQA's KV
sharing; it recurs in a fully-open, MHA, differently-trained model. That is direct support for
the bipolar framing being a property of safety-tuned transformers rather than a Qwen quirk.

Two honest caveats:
1. This discovery uses a different contrast than the legacy maps (different benign prompt set,
   first-token logit-difference metric), so the **absolute scores are not comparable** to the
   legacy magnitudes, and the specific top heads differ from the legacy 1.5B map. The
   cross-model *structural* claim (a compliance side exists and is strong) is what this supports,
   not head-for-head agreement.
2. Notably, **L23-H11 — the legacy 1.5B "top refusal head" — appears as a compliance head
   (−0.34) in this fresh discovery**, consistent with Result 2's finding that it is
   sign-unstable. Two operationalizations disagree on its sign, which is itself the P0-4 story.

Gated models (Llama-3.1-8B, Gemma-2-9B) were not run in this batch; Gemma requires the verified
`head_dim=256` geometry that `bsc.models.verify_geometry` now enforces.

---

## Superseded / prior claims

See `CLAIMS_AUDIT.md` for the full audit of the legacy results. P0-1 (sparsity ~85%), P0-2
(KV-cache 85%), P0-3 (Cohen's d mislabel), P0-5 (fabricated figure), and P0-6 (six judges) are
documented there and do not carry into this repo.
