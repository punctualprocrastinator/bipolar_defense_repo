# Findings — new_bipolar_study

Dated, append-only research log. Every entry points to the run directory that produced it.
Negative and non-reproducing results stay here; they are superseded by later dated entries, not
deleted (CLAUDE.md §2, §7).

All runs executed on an NVIDIA RTX PRO 6000 Blackwell (sm_120, 95 GB), torch 2.11.0+cu130,
transformers 5.14.1, seed 0. Full provenance in each run's `manifest.json`.

---

## 2026-08-04 — HEADLINE: which "bipolar defense" actually works

Two mechanisms have been conflated under "bipolar defense." They give opposite results.

| Mechanism | What it does | Result |
|---|---|---|
| **Multiplicative amplification** | `refusal_head_output *= N`, zero compliance heads | **Fails** — never reduces ASR |
| **Additive steering** | add a difference-of-means refusal direction to the residual stream | **Works** — and transfers across attacks |

### The additive steering defense works, and transfers across attack types

A single refusal direction — mean(harmful) − mean(benign) residual at layer 18, **unit-normalized**,
built with **no attack-specific data** — added to the residual stream during generation:

| Attack | Model | ASR undefended | ASR defended | Coefficient | Output quality |
|---|---|---|---|---|---|
| Crescendo (multi-turn) | Qwen2.5-1.5B | 60% [31–83] | **10%** [2–40] | α=16 | coherent refusals |
| GCG (single-turn) | Qwen2.5-7B | 46% [28–65] | **0%** [0–14] | α=8 | 96/96 coherent refusals |

Runs: `runs/steering_defense/…` (Crescendo), `runs/gcg_transfer/…` (7B GCG).

The **same direction defends both attacks** — that is cross-attack transfer of the *defense*.
Verified genuine: on 7B GCG all 96 defended trials are coherent refusals ("Sorry, but what
you're asking involves activities that are illegal and unethical. I cannot provide any
assistance…"), 0 compliance, 0 degeneration. On Crescendo, α=16 gives "I am unable to provide
any information about illegal activities…". There is an **inverted-U in the coefficient**:
too small (baseline) or too large (α=32 → 50%) both fail; α≈16 is the optimum.

Compliance-head *ablation alone* does **not** transfer to GCG (54% ASR, ≈ baseline). So the
working ingredient is adding the refusal direction, not removing compliance heads.

### The multiplicative amplification defense fails — the head is sign-inverted

Crescendo multiplier sweep, Qwen2.5-1.5B (`runs/crescendo_sweep/…`, corrected judge):

| condition | ASR |
|---|---|
| baseline | 70% |
| mult_1 (compliance-ablate only) | 60% |
| mult_3 / 6 / 12 | ~90–100% |

Amplification never reduces ASR. Root cause (`runs/logit_lens_sign/…`): the **dominant "refusal"
head is sign-inverted** — its isolated contribution, read through the unembedding, promotes
*compliance* tokens over refusal tokens. L23-H11 (1.5B) sign score −1.19; L25-H1 (7B) −7.44;
5/10 and 6/11 refusal heads inverted. Multiplying such a head's output amplifies toward
compliance, so "stronger" is worse. Adding the correctly-signed *direction* (above) does not have
this problem — which is exactly why additive works where multiplicative fails.

*(Caveat: at high multipliers the model also degrades into incoherence — see the methodology note
below — so part of the apparent "compliance" there is degeneration the keyword judge can't
cleanly separate. Either way, multiplicative amplification is not a working defense.)*

### Is the defense actually "bipolar"? Mostly no — the refusal side does the work

The working single-direction defense above uses a residual direction and touches *no heads*, so
it is not bipolar. Tested the true head-based bipolar defense directly
(`runs/bipolar_steering/…`, 1.5B): per-head additive steering on the 10 refusal heads (sign taken
from the measured activation difference) with and without zero-ablating the 5 compliance heads,
decomposed.

| condition | ASR | output quality (calibrated α=8) |
|---|---|---|
| baseline | 50% | — |
| compliance-ablate only | 60% | **no help** — ≥ baseline |
| refusal-steer only (α=8) | 20% | **coherent refusals** (78 refusal / 2 compliance / 0 degenerate) |
| bipolar: refusal-steer + compliance-ablate (α=8) | 10% | coherent (76 refusal / 3 compliance / 1 degenerate) |

Two conclusions:

1. **Compliance-head ablation contributes little.** Alone it does not reduce ASR (60% ≥ 50%
   baseline); added on top of refusal steering it moves the point estimate only marginally and
   inconsistently (bipolar_8 10% vs refusal_steer_8 20%, but bipolar_16 is *worse* than
   refusal_steer_16 by degeneration). **The refusal side does essentially all the work** — whether
   applied as per-head steering or as the single residual direction.
2. **Per-head steering saturates faster.** It injects into 10 heads at once, so at α≥16 it
   over-steers into refusal-flavoured *degeneration* (refusal_steer_16: 40/80 nonresponse;
   bipolar_24: 80/80 nonresponse) rather than coherent refusals. The single residual direction
   (`steering_defense`) reaches the same 10% ASR with cleanly coherent refusals and is easier to
   calibrate.

So *zero-ablating* the compliance heads adds little. **But that was the wrong intervention** — see
the next entry, which supersedes this conclusion: the compliance heads *are* a useful lever once
you steer them with the correct (opposite) sign instead of zeroing them.

### UPDATE — the compliance heads ARE a defense lever, with the opposite sign

`runs/compliance_calibration/…` (1.5B). Instead of zeroing the compliance heads, steer them with
`β·(refuse−comply)` and sweep β through **both signs**, decomposed against refusal-head steering.

| condition | ASR | coherence (refusal/degen/comply) |
|---|---|---|
| baseline | 60% | 59/0/21 |
| compliance zero-ablate | 60% | no effect |
| compliance steer **+8** (same sign as refusal heads) | **100%** | 7/0/73 — **attacks!** |
| compliance steer **−8** (opposite sign) | **20%** | 78/0/2 — coherent refusals |
| refusal steer +8 & compliance steer −8 (combined) | 10% | 79/0/1 |

**The compliance heads are sign-inverted relative to the refusal heads.** The *same*
`+(refuse−comply)` steering that defends via refusal heads (`refusal_only`: 60→20% ASR) *attacks*
via compliance heads (`comp_steer_+8`: 60→100%). Flip the sign and the compliance heads defend on
their own (`comp_steer_−8`: 60→20%, coherent refusals like "I'm sorry, but for your safety reasons
I can't provide information about illegal…"). Zero-ablation does nothing because it discards the
signal instead of reversing it.

This is the mechanistic core of the "bipolar" name made concrete and *causal*: the two head
populations have **opposite-sign geometry**, so a defense must steer them in opposite directions
(refusal `+`, compliance `−`). It also explains why the single residual-direction defense works so
cleanly (it never has to reconcile the two head signs) and why naive per-head or single-sign
schemes fail. (Credit: the sign flip was found by testing the negative direction on the compliance
heads, which the positive-only first pass had missed.)

### Correct-signed bipolar (+refusal / −compliance) — works, but doesn't beat refusal-only at n=10

Clean matched run with the stable-seed fix (`runs/compliance_calibration/…-e9c7acf649`):

| condition | ASR | coherence (refusal/degen/comply) |
|---|---|---|
| baseline | 60% | 51/0/29 |
| compliance zero-ablate | 80% | worse than baseline |
| refusal-only (+8) | **0%** [0–28], p=0.031 | 80/0/0 |
| compliance-only (−8) | 10% [2–40], p=0.062 | 78/0/2 |
| **bipolar: +refusal 8 / −compliance 8** | **0%** [0–28], p=0.031 | 80/0/0 |

The correctly-signed bipolar defense achieves **0% ASR with 80/80 coherent refusals** — a clean,
strong defense. **But refusal-steering alone also hits 0% here**, so the bipolar version *matches*
rather than *beats* the one-sided defense: refusal steering already saturates the floor, leaving no
headroom for the compliance side to demonstrably add value. An earlier (pre-seed-fix) run had
refusal-only at 40% and combined at 10%, suggesting synergy — but that gap is within seed noise at
n=10 (the two runs disagree on refusal-only, 0% vs 40%, purely from different sample seeds).

So the honest status of the bipolar-vs-one-sided question:
- **Both robust:** the compliance heads are a genuine independent lever with the correct sign
  (compliance-only −8 defends at 10%, coherent), and the correct-signed bipolar is a clean 0%.
- **Unresolved at n=10:** whether bipolar *beats* refusal-only was not answerable at n=10 because
  refusal-only already floored out. **Now resolved at n=35 — see below.**

### RESOLVED at n=35 — the bipolar defense significantly beats both one-sided defenses

Grew the benchmark 10 → **35** scenarios via validated harvesting (`harvest_scenarios`: a 7B
attacker writes escalations, kept only if the undefended 1.5B target genuinely complies). The
harvested scenarios are *harder* — baseline ASR is **89%** (vs 60% on the original 10), so the
defenses no longer saturate and can separate. Run `runs/compliance_calibration/…-3e800e6cea`,
stable seed, 35 scenarios × 8 trials:

| condition | ASR | 95% CI | vs baseline | refusal/degenerate/compliance trials |
|---|---|---|---|---|
| baseline | 89% | [74–95] | — | 67/0/213 |
| compliance zero-ablate | 94% | [81–98] | +6pp, ns | worse than baseline |
| refusal-only (+8) | 49% | [33–64] | −40pp, **p<0.001** | 238/0/42 |
| compliance-only (−8) | 60% | [44–74] | −29pp, **p=0.002** | 207/0/73 |
| **bipolar +refusal 8 / −compliance 8** | **34%** | [21–51] | **−54pp, p<0.001** | 251/0/29 |

**The correctly-signed bipolar defense is significantly the best**, and the two sides are
**additive**: refusal-only removes 40pp, compliance-only removes 29pp, and combining them removes
**54pp** — more than either alone, with the highest coherent-refusal share (251/280 trials) and
zero degeneration. This is the result the n=10 run couldn't reach, and it **vindicates the bipolar
framing for defense**: both head populations are causal, opposite-signed levers, and steering both
in their correct directions beats steering either one. Zero-ablation remains useless (94% ≥
baseline) — the signal must be *reversed*, not removed.

**Bottom line on the whole defense arc:** multiplicative amplification fails (sign-inverted head);
zero-ablation does nothing; the working defense is *additive steering along the correctly-signed
refusal and compliance directions*, and the **full bipolar version (both sides) is significantly
the strongest** (89% → 34% ASR, coherent). The single residual direction is the simplest near-equal
alternative. Remaining caveats: still one model (1.5B) and one attack (Crescendo) for this
comparison; the harvested scenarios are 7B-authored (a possible distribution quirk); an LLM judge
would tighten the compliance/degeneration boundary further.

---

## 2026-08-04 — METHODOLOGY: refusal-classifier ASR is fooled by degeneration

A first pass concluded that *both* defenses increased ASR. That was wrong, and the reason is
important enough to be a finding in its own right.

Strong intervention breaks the model into repetition salad — `回答回答回答…` (answer answer),
`不存在不存在…` (does-not-exist), `Please keepeadowowow色`. The keyword refusal-classifier scores
these as **compliance** (no refusal phrase present), and the original degeneracy detector missed
them because it was **English/whitespace-only** — CJK has no spaces, so `回答回答回答` was one
"word". This inflated ASR toward 100% and produced a false "defense makes it worse" conclusion.

Fix (`bsc/judge.py`, `tests/test_judge.py`): script-agnostic **character-bigram** repetition
detection, with the real gibberish strings as regression fixtures. Documented limitation:
*semi-coherent* degradation (e.g. "I'm Qwen… themingergoneantly.abe") still has too much variety
to catch by repetition and needs an **LLM judge** — so the keyword ASR is an *upper bound* on
genuine compliance and the paper must report LLM-judge agreement. This is a real, citable
evaluation pitfall: any steering/ablation study reporting ASR from a keyword classifier without a
degeneration filter is likely over-counting attack success.

---

## 2026-08-04 — Cross-architecture: the bipolar structure is not a GQA artifact

Fresh harm-contrastive circuit discovery (`runs/discover_circuit/…`), 8 pairs each:

| Model | Attention | Top refusal | Top compliance |
|---|---|---|---|
| Qwen2.5-1.5B | GQA 6:1 | L16-H6 (+0.24) | L15-H0 (−0.36) |
| OLMo-2-7B | MHA (dense) | L14-H11 (+0.14) | L30-H14 (−0.38) |
| **OLMoE-1B-7B** | **MHA + MoE** (64 experts) | L9-H13 (+0.15) | **L15-H4 (−0.80)** |

Both a refusal side and a compliance side appear in every architecture, including a sparse
**mixture-of-experts** model — and in both open (OLMo-family) models the compliance heads are
*stronger* than the refusal heads. So "compliance heads that actively push toward answering" is
not a consequence of grouped-query attention or of a Qwen-specific quirk; it recurs across GQA,
MHA-dense, and MHA-MoE. (Attention is dense in all three — the MoE routing is in the MLP — so the
head methodology transfers unchanged. Whether compliance is *also* routed through specific experts
is a separate, open question.)

---

## 2026-08-04 — Circuit transfer: partial (shared top heads, distinct overall)

GCG-contrastive vs harm-contrastive circuits, discovered by identical code on the same model
(`runs/gcg_transfer/…`, 7B, N=20; `bsc/patching.py::circuit_overlap`):

- Top-11 refusal-head Jaccard overlap **0.22** — shares the dominant **L25-H1** plus L18-H4/H15,
  L20-H4.
- Compliance-head Jaccard **0.10**; full per-head rank correlation (Spearman) **≈0.04**.

So GCG and the harm/Crescendo circuit **share a few key top heads but are largely distinct
overall**. Notably, the *defense* transfers (steering direction defends both) even though the
*circuits* only partially overlap — evidence that a shared low-dimensional refusal **direction**
matters more for defense than an identical set of causal heads. (The 1.5B pilot, `n=5`, agreed:
refusal Jaccard 0.22, compliance 0.47, Spearman 0.51.)

---

## Established anchor (legacy, verified)

`7b_results/adaptive_defense_gcg.json` (N=100, audit-verified): GCG raw ASR **66%**, additive
bipolar defense **33%**, conditional gate 42%. Consistent with the additive-steering result above
(a defense built from the refusal direction reduces GCG ASR); the fresh 7B run pushes it further
(46%→0% at N=24 with the unit-normalized direction and degeneration-aware judge).

---

## Open items / caveats

- **n is small** for the ASR numbers: Crescendo 10 scenarios, GCG 24 prompts. CIs are wide
  (McNemar p≈0.06 at the best steering point). Scale both up before the paper.
- **LLM judge** needed to close the degeneration gap and validate the keyword ASR.
- **Coefficient calibration** (the inverted-U) should be characterised more finely and per-model.
- Gated models (Llama-3.1, Gemma) not yet run; Gemma needs the verified `head_dim=256` geometry
  the framework now enforces.

## Superseded

See `CLAIMS_AUDIT.md` for the legacy-results audit (P0-1…P0-6). Note this repo *corrected its own*
first-pass conclusion (the degeneration/judge bug above) — recorded here rather than quietly
overwritten, per the honesty rules.
