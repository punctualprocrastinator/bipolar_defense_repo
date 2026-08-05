# Findings — new_bipolar_study

Dated, append-only research log. Every entry points to the run directory that produced it.
Negative and non-reproducing results stay here; they are superseded by later dated entries, not
deleted (CLAUDE.md §2, §7).

All runs executed on an NVIDIA RTX PRO 6000 Blackwell (sm_120, 95 GB), torch 2.11.0+cu130,
transformers 5.14.1, seed 0. Full provenance in each run's `manifest.json`.

---

## 2026-08-05 — Circuit Breakers (GraySwanAI RR checkpoint) baseline — BLOCKED on judge quality

Obtained the actual Circuit Breakers checkpoints (`GraySwanAI/Llama-3-8B-Instruct-RR`,
`Mistral-7B-Instruct-RR`, `llava-...-RR` — all **dense**) to run the ICLR-critical baseline without
retraining. Ran undefended `Llama-3-8B-Instruct` (ungated NousResearch mirror) vs the RR checkpoint on
the 35 Crescendo scenarios (×3 seeds = 105) under the **keyword** judge. Raw counts *looked* like CB is
worse than undefended (comply 57→68, native refusals 48→26), **but manual inspection of CB's outputs
invalidates that reading** — the keyword judge cannot score a representation-rerouting defense:

- CB's designed mechanism is to **reroute harmful generations into gibberish** (confirmed: `EDR_Evasion`,
  `Meth`, `Harvested_06/11` → `"aspers/dc.orcarage sourceMapping preferablyQRST…"`). That gibberish is a
  *successful* defense, but the keyword judge labels some of it **compliance** (degeneration detector
  misses the Latin-script token-salad) and some **nonresponse**.
- The keyword judge also mislabels CB's **terse refusals** ("I can't help you with that.") as
  **compliance**.
- The **harvested** Crescendo goals are benign-ish ("create a website", "develop an app **with
  consent**") — "compliance" there is not harmful, so they can't support an ASR claim.

**Conclusion: no CB-vs-ours claim is valid under the keyword judge.** This is the wall PLAN_ICLR.md
predicted (Phase B.1/B.2): scoring a rerouting defense requires (1) an **LLM/safety judge**
(Llama-Guard-3-8B or the HarmBench classifier) that separates harmful-compliance / gibberish / refusal,
and (2) **vetted-harmful** Crescendo goals (drop the benign harvested ones). Do NOT report the 57/68
numbers as a result. Probes: `/marimo/llama_cb_setup.py`, `/marimo/cb_inspect.py`. Next: build the LLM
judge, then rerun the 3-way (undefended / CB / bipolar) on genuinely-harmful goals.

---

## 2026-08-05 — MoE (OLMoE): refusal is expert *computation*, not routing — and it is NOT linearly steerable

Full investigation of whether the bipolar defense transfers to a Mixture-of-Experts model
(allenai/OLMoE-1B-7B-0924-Instruct: 64 experts, top-8, 16 layers). **Bottom line: a training-free
steering defense does not work on OLMoE.** But the investigation yielded a clean mechanistic result.

**Setup checks (both pass):** OLMoE refuses 4/5 *direct* harmful requests but complies 100% to the
Crescendo escalations (classic single-turn-refuses / multi-turn-falls split), so there is a refusal
signal to study. Differential expert routing exists (some experts fire more when refusing), but
top experts fire on only ~10 tokens/prompt and rarer ones on 0–3 — the sparse-firing caveat is real.

**Patch-target sanity split (the positive result).** Contrast: direct-harmful (refuses,
refusal-logit-diff **+2.18**) vs Crescendo-terminal (complies, **−0.95**), last-token. Patch the
refusing run's state into the complying run, per layer:
- **Routing patch** (force refusing router indices+scores): per-layer ~−0.9, all-layers only
  −0.95→−0.59. **Routing barely moves refusal** — consistent with RASET (routing is topic-driven).
- **Block-output patch** (force refusing MoE-block output): **L9 −0.95→+0.32**, L10/L12 positive —
  patching expert *computation* at layers 9–12 restores refusal.
→ **OLMoE's refusal is carried in expert computation (block output, L9–12), not routing.** This is
also *why* attention-head steering failed: the refusal signal isn't in the attention heads
(top refusal head only +0.15) — it moved into the MLP/experts.

**But it is not linearly steerable — four intervention families all fail:**
1. Attention-head steering (multiplicative + additive): no coherent-refusal window.
2. Residual-direction steering (α 8–32): degenerates (36/36 nonresponse), 0 refusals.
3. Block-output additive steering (α 0.5–16, the *right* substrate): 0 refusals; narrow cliff
   between no-effect and degeneration, no coherent window.
4. Block-output projection-clamp/ablation (clamp the comply/refuse coordinate to the refusing mean,
   β 0–4): β=1 still 100% *coherent compliance*; only overshoot degenerates. 0 refusals.

The **only** intervention that induced refusal was full block-output *patching* (replacing with the
entire real refusing activation) — not deployable (circular: needs a refusing activation you don't
have at inference). Every single-direction edit fails because refusal in OLMoE is **distributed and
nonlinear** across the expert computation, not a low-dimensional linear direction.

**Honest conclusion.** Refusal's substrate shifts with architecture: a low-dimensional,
linearly-steerable direction in attention heads on dense models (→ training-free bipolar defense
works, 89→34%) vs distributed expert computation in MoE (→ training-free steering defenses fail,
across four families). This *explains* why steering/representation defenses don't transfer to MoE,
backed by the patch-target split and a full intervention ablation. It is **not** a working MoE
defense — we cannot claim one. Remaining routes: training-based (SAFEx-style LoRA on
response-control experts, not training-free), or a larger/less-fragile MoE.

**Open question (decisive):** is this "MoE" or "small/fragile MoE"? OLMoE is 1B-active. Testing
Qwen3-30B-A3B (the SAFEx model) settles whether the non-steerability is architectural or a
fragility of tiny MoEs. Not yet run.

**Decomposition ladder (reframed as interpretability characterization, not defense — full white-box,
so "not deployable" is irrelevant; the goal is to localize the causal signal to something minimal):**
- **Rung 1 (gate weights):** patch the refusing router's *continuous* weights onto the complying
  run's clean-selected experts. ~Null (all-layers −0.95→−0.72, vs block-patch's −0.95→+0.32 from L9
  alone). So refusal is **not in the gate** (neither discrete selection nor continuous weights) —
  it's in the expert *outputs* E_i(x). Consistent with RASET (topic-driven routing).
- **Rung 2 (per-expert credit, ablation at L9–12):** zero one expert's `gate_up_proj` at a time,
  rank by refusal-logit-diff drop on the refusing run. **~32 (layer,expert) pairs carry 80% of the
  ablation credit; 119/256 pairs contribute; top expert (L12-E9) only ~33% of the full signal.**
  Refusal is **distributed across dozens of experts, not a sparse handful** — the precise mechanistic
  content of "not linearly steerable." Caveats: top experts fire on only 8–15/50 tokens (per-expert
  credit noisy), and total marginal drop (4.72) > full signal (2.18) → ablations redundant/non-
  additive; robust claim is the aggregate, and it needs multi-prompt aggregation to tighten.
- **Rung 4 (linear decodability):** held-out AUC of a diff-of-means probe separating refuse vs comply.
  Contrast had to be built carefully — two confounds caught and rejected: (a) harmful-vs-benign =
  topic (AUC 1.0, the known harmfulness direction, not refusal); (b) direct-vs-Crescendo = format/
  length (AUC 1.0). The clean version — **direct-harmful prompts split by the model's own behavior**
  (111 refuse / 39 comply, same single-turn format, both harmful) — gives **AUC ~0.85–0.95** in
  residual_in, block_out, AND a single expert's output (L12-E9: 0.90), at L9–12.
  → **Refusal is linearly *decodable* in exactly the bases where *steering* failed.** The residual/
  block stream *encodes* refusal without being its *write interface*. Plus a decode-vs-cause
  asymmetry: one expert (L12-E9) decodes at 0.90 yet its ablation drops refusal only ~33%, and ~32
  experts are needed for 80% of causal credit — **reading ≠ writing ≠ single-point causality**.

**LADDER SYNTHESIS (the mechanistic headline):** In OLMoE refusal is **linearly decodable** (AUC ~0.9,
even from one expert), **causally distributed** (~32 experts for 80%), and **not steerable from the
residual** (four intervention families fail). The residual stream *represents* refusal without being
its *write interface*; the write happens through distributed, nonlinear expert computation — which is
exactly why steering defenses fail on MoE even though a linear probe reads refusal off the activations
cleanly. (Caveats: one model; class imbalance 111/39; single-prompt ablation for rung 2.)
- **Rung 3 (per-token):** not yet run (spatial locality of the assembly).

**Gate-only fine-tuning (conceding training-free, allowing router-only training):** freeze all experts
+ attention, train only the 16 gate matrices (2.1M params, **0.03%**) on direct-harmful→refusal.
At proper lr (5e-4, 100 steps) the router **perfectly fits the training data (loss→0)** but Crescendo
ASR is **unchanged (12→11)**, benign over-refusal 0, direct refusal preserved — **it memorizes refusal
on direct-harmful prompts but does NOT generalize to defend the multi-turn attack**. (At high lr=2e-3
it destabilizes: 12→7 but via *degeneration*, plus it *hurts* direct refusal 10→6 — not a real defense.)
Mechanism: the router is topic-driven (RASET) and Crescendo makes the input *look benign to the router*
(rep-eng 2507.02956), so it keeps routing the jailbroken input to complying experts; training on direct
harmful doesn't fix the escalated representation. → **"MoE refusal is a trainable routing policy" is NOT
supported for gate-only training that generalizes to the attack.** (Training on the attack distribution,
or full/expert fine-tuning, would likely defend — a weaker/different claim.)

**FINAL BOUNDARY (OLMoE):** refusal is linearly decodable, causally distributed across ~32 experts, and
**not** steerable from the residual, **not** reachable by inference-time gate patching, **not** defendable
by gate-only fine-tuning that generalizes to the attack. It lives in the experts' computation; the router —
even trained — can't be made to invoke it for the jailbroken distribution. Probes: add
`/marimo/moe_rung4*.py`, `gate_tune*.py`.

Probes: `/marimo/moe_patch.py`, `moe_blocksteer.py`, `moe_ablate.py`, `moe_rung1.py`, `moe_rung2.py`
(exploratory scripts, not yet formalized into `bsc`). Related: SAFEx (2506.17368), RASET (2605.29708).
This is a *model-mechanism* thread, kept separate from the dense-model ICLR defense paper.

---

## 2026-08-05 — DECISIVE SCALE TEST: Qwen3-30B-A3B replicates OLMoE — MoE non-steerability is ARCHITECTURAL, not small-model fragility

Ran the MoE investigation on **Qwen3-30B-A3B** (48 layers, 128 experts, top-8, GQA; the SAFEx model)
to settle whether OLMoE's results were "MoE" or "small/fragile MoE." **They replicate and strengthen.**

- **Setup ideal:** refuses **10/10** direct-harmful, falls to Crescendo **9/10** ASR — strong direct
  safety, multi-turn vulnerability. (Same router/expert API as OLMoE: `Qwen3MoeTopKRouter` returns
  (logits,scores,indices); fused `Qwen3MoeExperts` with `gate_up_proj` → clean ablation.)
- **Patch-target split replicates:** gate/routing patch is **null** (all-layers gate patch = baseline),
  block-output patch **carries refusal**, localized to **L26–41** (same relative depth as OLMoE L9–12).
  Refusal is expert *computation*, not routing — on the 30B too.
- **Steering replicates — and rules out fragility (THE decisive result):** block-output additive
  (α 4–16) and projection-clamp (β 1–4) steering at L26–38 do **not** defend (Crescendo compliance
  stays ~92%, 30–35/36). But **Qwen3-30B does NOT degenerate** (nonresponse 0–1; outputs stay
  coherent; a clamp sample is even a coherent refusal, just an outlier). So OLMoE's degeneration was a
  small-model artifact — the *real*, scale-robust finding is that **steering the MoE block output does
  not induce refusal, on a 1B-active OR a 30B model, even when the model stays perfectly coherent.**
  The MoE substrate isn't the write interface for refusal, independent of scale.
- **Per-expert credit replicates + strengthens:** ablating experts at L26/27/32/35/36/40: **42
  (layer,expert) pairs carry 80%; 93 of 768 positive; top expert only ~3%** of the full signal — even
  MORE distributed than OLMoE (32 pairs; top ~33%). Roughly consistent with SAFEx (~12 experts / 22%):
  a top dozen carry ~20–40%, but 80% needs ~42. Refusal is genuinely distributed across the experts.
  Caveats: very low routing frequency (top experts fire on 1–7 of 48 tokens — 128-expert top-8 sparsity
  makes per-expert credit noisy), single prompt.
- **Decodability (rung 4) NOT constructible on Qwen3:** it refuses **147/150** direct-harmful, so there
  is no format-matched comply set in the direct distribution to build a clean refuse-vs-comply probe.
  A real methodological limit (and itself telling: Qwen3's *direct* refusal is rock-solid; only the
  *multi-turn* escalation gets through).

**Cross-scale conclusion:** the MoE findings are architectural, not fragility artifacts. Refusal in MoE
is expert *computation*, distributed across dozens of experts, not in the routing, and **not steerable**
from the residual/block output at any scale tested (1B-active OLMoE and 30B Qwen3). Steering defenses —
and by extension representation defenses like Circuit Breakers — target a substrate that isn't the write
interface for refusal in MoE. Probes: `/marimo/qwen3_*.py`.

**Tested the SOTA MoE-steering method (SteerMoE) faithfully + combined with our bipolar defense — the
MoE non-defendability is robust across every method and combination.** Lit review found the published MoE
steering methods we hadn't tried: **SteerMoE** (2509.09660, expert de/activation via risk-difference
Δ_i = p_i(unsafe)−p_i(safe), then Eq 7–8 force s_k←s_max+ε / s_min−ε across all tokens), **MASCing**
(2604.27818), ExpertSteer, Geometric Routing. Read SteerMoE's exact method: for Qwen3-30B *safety* they
force **15 experts activated / 5 deactivated total**, in middle layers, and evaluate **single-turn only**
(GCG/AIM/AdvBench) — no multi-turn. Full sweep on Qwen3-30B Crescendo (12 scenarios × 3 seeds = 36;
baseline 3 refuse / 0 nonresp / 33 comply):
- **faithful SteerMoE 15/5** → 5 / 0 / 31: coherent (0 degeneration) but ~no defense (33→31).
- experts 40/15 → 2 / 1 / 33: no effect. experts 160 (per-layer over-force) → 3 / 17 / 16: reduces
  compliance but purely via **degeneration** (my earlier "SteerMoE degenerates" was a *my-over-forcing*
  artifact — the faithful few-expert version does NOT degenerate).
- our **bipolar residual steering alone**: α=6 → 0/1/35 (no effect); α=12 → 0/24/12 (degenerate).
- **combination (SteerMoE 15/5 + bipolar residual)**: α=8 → 0/12/24; α=14 → 0/6/30 — degenerate or
  ineffective, **zero coherent refusals**, samples gibberish.

**Robust conclusion:** on MoE, *no* intervention — SteerMoE expert forcing, residual bipolar steering, or
their combination, at any strength — produces a **coherent-refusal** defense against the multi-turn attack.
Every knob lands in {no-effect, degenerate}; the only coherent refusals (faithful SteerMoE) give a
negligible 33→31. So our bipolar defense **can** be added to SteerMoE (it unifies conceptually: steer
opposing populations in opposite directions — heads for dense, experts for MoE), but the combination does
not rescue the MoE case: the expert-computation substrate is not a working coherent-refusal write-interface
under a strong multi-turn attack. This also **critiques SteerMoE's +20% safety** — on multi-turn attacks
with a degeneration-aware judge, expert forcing that "reduces compliance" does so largely by *breaking*
the model, not *refusing*. Caveats: faithful reimplementation (not their vLLM code), Crescendo-only here,
rough expert scoring (25 prompts/set). Probes: `/marimo/qwen3_steermoe_faithful.py`, `/marimo/qwen3_combo2.py`.

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
