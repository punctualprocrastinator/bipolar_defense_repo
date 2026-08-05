# Redarc Labs — research + software strategy

Dated 2026-08-05. Scope: how to package the research and the software, and how to extend from
mechanistic interpretability into **AI control** and **security**. Companion to `PLAN_ICLR.md`
(which covers the single next paper) — this covers the lab.

---

## 1. The thesis (what the lab is actually differentiated on)

Do **not** position as "a mechanistic interpretability lab." That space is crowded (Anthropic,
GDM, EleutherAI, Apollo, Transluce, Goodfire, plus dozens of independents), and this project just
learned the hard way that a circuit-discovery finding can be scooped between submission and
follow-up.

Position on the thing that is genuinely scarce and that this project has already demonstrated:

> **Interpretability claims that survive an adversary — and the evidence standards required to
> know whether they do.**

Two words carry the differentiation:

- **Adversarial.** Most interp is done on cooperative inputs. Redarc studies internals under
  active attack (GCG, Crescendo, routing attacks, adaptive attackers), where the mechanism *and
  the method* are being stressed.
- **Falsifiable.** This project has already produced the rarest artifact in the field: an audit
  that **falsified four of its own headline claims** (`CLAIMS_AUDIT.md`), including a script that
  generated a paper figure from `np.random.normal` — plus a mid-session correction where a judge
  bug had inverted a defense conclusion (`FINDINGS.md`). Almost nobody publishes that. It is a
  credibility asset, not an embarrassment.

**The lab's tagline claim:** *interpretability under adversarial conditions demands adversarial
standards of evidence — and we build the tools that enforce them.*

This is defensible because it is not a race for one finding; it is a compounding capability.

---

## 2. Packaging the software: split it into three layers

Right now `bsc` is one package doing three jobs. Separating them is what turns it from "one
project's code" into infrastructure other people cite and use.

### Layer 1 — the harness (general-purpose, most reusable)
Device/dtype resolution with architecture verification, GQA/MHA/MoE-correct head interventions
with identity no-op tests, run manifests with full provenance, seeded trial machinery, paired
statistics (Wilson/bootstrap/McNemar).

Nothing here is bipolar-specific. **Anyone doing head-level intervention work needs it**, and it
already catches bugs the field ships: the Gemma `head_dim` trap (256 vs derived 224 — silently
slices wrong columns), leaked hooks, per-process seed instability, measure-vs-generate mismatch.

→ Release as its own package with its own name and docs. The bipolar study becomes a *user* of it.

### Layer 2 — the evaluation instrument (highest citation potential)
This may be the single most valuable thing produced this session, and it is currently buried
inside `bsc.judge`:

> Keyword refusal classifiers systematically **over-count attack success**, because strong
> interventions degrade models into repetition salad that contains no refusal phrase. Our
> detector was English/whitespace-only, so CJK gibberish (`回答回答回答`, `不存在不存在`) scored as
> *compliance* — which inverted a defense conclusion until it was fixed.

Implication for the field: **any steering/ablation paper reporting ASR from a keyword classifier
without a degeneration filter is likely reporting inflated ASR.** That is a measurement result
affecting many published numbers, not a footnote.

→ Package as a standalone, validated jailbreak-evaluation instrument: keyword + degeneration +
LLM judge, published agreement rates, multilingual degeneration fixtures, versioned so results are
comparable across papers. Evaluation infrastructure (HarmBench, JailbreakBench) gets cited far
more than individual findings. **This is the highest-leverage software play.**

### Layer 3 — the study packages
`bipolar` (refusal/compliance circuits + defense), and later `control`, `moe-safety`, etc. Each is
a thin layer of experiments over Layers 1–2, each reproducible from one command.

**Release discipline:** Layer 1+2 open from day one (adoption = citations = credibility). Layer 3
open at paper submission. Every release carries the manifest/seed guarantees, so a stranger can
regenerate every number — that *is* the brand.

---

## 3. Packaging the research: a ladder, not one paper

| # | Paper | Venue | Status |
|---|---|---|---|
| 1 | Bipolar defense that transfers across attacks + architectures (incl. MoE) | ICLR main | in progress (`PLAN_ICLR.md`) |
| 2 | **The evaluation-instrument paper**: keyword ASR over-counts under intervention; a validated judge | ACL/EMNLP or a benchmarks track | mostly *already done*, needs LLM-judge validation + writeup |
| 3 | **Dense vs sparse safety substrate**: attention-head defenses vs expert-routing attacks on MoE | security venue (USENIX/CCS) or ICML | the novel experiment we scoped |
| 4 | Sign inversion as a vulnerability class: steering along unexamined signs inverts safety | short paper / workshop → fold into 1 or 3 | result exists, needs RMS-corrected DLA |
| 5 | Control protocol: compliance-head probe as a trusted monitor, with adaptive-attacker eval | control/security venue | needs Phase D work |

Paper 2 is underrated: it is cheap (the finding exists), it is field-serving, and it establishes
the lab as the group that gets measurement right. Ship it early — it makes papers 1 and 3 more
credible by association.

---

## 4. Extending into AI control and security

The bridge already exists in this project's own results; it just needs naming.

### → AI control
The **compliance-head probe** (AUC 0.943 on 7B) is not a curiosity — it is a **trusted monitor**,
the core primitive of the control agenda: cheap, internal, runs on every forward pass, and gates
an expensive intervention only when it fires.

Concrete projects:
1. **Monitor-gated defense.** Rather than always-on steering, trigger only when the probe crosses
   τ. Report the ASR/FPR/compute trade against always-on. *Caveat this project already learned:
   a conditional gate creates an off-switch an adaptive attacker can target — so the adaptive
   evaluation is mandatory, not optional.*
2. **Validate the monitor against behavior, not norms.** This repo already found refusal-head norm
   does **not** predict refusal behavior. A monitor must be validated on behavioral outcomes.
3. **Multi-turn monitoring.** Crescendo is the natural control setting: the monitor watches the
   conversation trajectory, not one prompt.

Honest scoping: this is *monitoring an already-validated harm probe*, **not** detecting scheming or
deception. Say so explicitly; over-claiming here is the fastest way to lose credibility with the
control community.

### → Security
The security framing is already latent in three results:
1. **Sign inversion = a vulnerability class.** "A safety intervention amplified along an unexamined
   sign convention degrades safety" is a *defect pattern*, reportable like a CWE. Generalize it:
   audit published steering defenses for sign correctness.
2. **Substrate mismatch = an attack surface.** Defenses that protect a dense pathway fail when the
   safety-critical computation is sparse and routable (the MoE case). That is a security argument
   about *where* defenses must live.
3. **Adaptive attackers as the standard.** Every defense claim gets a white-box adaptive attack.
   This is standard security practice and still rare in interp-flavored safety work.

Positioning: **red-team the defenses, not just the models.** The field is full of proposed
interventions with no adversarial evaluation. A lab that systematically breaks them — and
publishes the failure taxonomy — becomes load-bearing infrastructure fast.

---

## 5. Credibility and funding path (solo → funded lab)

Realistic sequence, cheapest first:

1. **Ship Layer 1+2 software publicly** with docs and tests. Costs nothing, compounds immediately.
2. **Ship paper 2 (evaluation instrument).** Cheap, field-serving, establishes rigor brand.
3. **Land paper 1 at ICLR main** with baselines + cross-model + adaptive attacker.
4. **Publish the audit methodology itself** — a short public writeup of how the harness caught four
   false claims in its own prior work. This is unusually strong evidence of research integrity and
   is exactly what funders and collaborators screen for.
5. **Then** raise: with two papers, adopted tooling, and a public audit record, grant applications
   (Manifund/BlueDot/OpenPhil-adjacent, LTFF) and collaborations become a different conversation
   than they are today. Note the current Manifund draft still has `[FILL IN]` ask amounts and
   should not go out carrying any of the four falsified claims.

Compute reality: the Blackwell (95 GB) is sufficient through ~32B. Beyond that needs either
attribution patching (cheap approximation) or real funding.

---

## 6. Risks, stated plainly

- **Scoop risk.** Discovery findings get scooped (this already happened). Mitigate by leading with
  defenses, evaluation, and adversarial robustness — which require *sustained* capability rather
  than a single lucky result.
- **Solo bandwidth.** Five paper tracks is not a solo agenda. Pick paper 1 + 2, ship, then expand.
- **Over-claiming.** The single greatest asset here is the willingness to publish "this did not
  reproduce." Every over-claim spends that asset. The `[UNVERIFIED]` convention and the audit file
  are the mechanism — keep them.
- **Infra-vs-research trap.** Do not let harness-building crowd out results. Layers 1+2 are done
  enough to release; freeze features and use them.

---

## 7. Next 90 days (concrete)

**Weeks 1–2:** split out Layers 1+2, publish with docs/tests/CI. Draft paper 2. Write the
related-work positioning from `PLAN_ICLR.md` Phase A.
**Weeks 3–6:** paper 1 Phase B (LLM judge, baselines incl. Circuit Breakers/SmoothLLM, HarmBench)
and Phase C (cross-model incl. MoE + third attack family).
**Weeks 7–9:** adaptive attacker (Phase D). Run the dense-vs-sparse MoE substrate experiment
(paper 3's core).
**Weeks 10–12:** write paper 1, submit; publish the audit-methodology note; ship paper 2.

**Immediate next two actions** (both already staged): finish the **OLMoE defense run** (the MoE
empirical anchor), and read **SAFEx (2506.17368) / RASET (2605.29708)** to position the
attention-head analysis as the dense complement to their sparse-expert routing work.
