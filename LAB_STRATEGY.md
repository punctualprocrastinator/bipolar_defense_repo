# Redarc Labs — research + software strategy

Dated 2026-08-05. Scope: how to package the research and the software, and how to extend from
mechanistic interpretability into **AI control** and **security**. Companion to `PLAN_ICLR.md`
(which covers the single next paper) — this covers the lab.

---

## 1. The thesis — already established, needs one addition

**Checked redarclabs.com (2026-08-05): the positioning already exists and is good.** The site
states *"adversarial interpretability"* — interpretability tools designed to work when models
actively resist investigation — with the thesis *"The only conditions that matter for safety are
adversarial ones."* Four directions are live (D1 cross-architecture refusal circuits ← **this
project**, D2 adversarially robust SAEs, D3 pre-commitment probes, D4 emotion deflection vectors),
plus real publications (ICML 2026, TAIS 2026, NeurIPS 2025) and two co-founders who are MARS V
Fellows at Cambridge AI Safety Hub.

So this section is **not** a proposal to reposition. The thesis is right and the credibility base
is further along than a from-scratch plan would assume. One thing is missing from the public
framing, and it is the scarcest asset the lab actually has:

> **Falsifiability as a stated method.** This project produced an audit that **falsified four of
> its own headline claims** (`CLAIMS_AUDIT.md`) — including a script that generated a paper figure
> from `np.random.normal` — plus a mid-session correction where a judge bug had inverted a defense
> conclusion (`FINDINGS.md`). Almost nobody publishes that.

"We study adversarial conditions" is a topic claim; several groups make it. **"We hold adversarial
standards of evidence, and here is the public record of us failing our own claims"** is a method
claim, and it is very hard to copy. Add it to the site as a stated commitment (§5.4).

Refined tagline: *adversarial conditions demand adversarial standards of evidence — we build the
tools that enforce them, and we publish what they falsify.*

---

## 2. Packaging the software: split it into three layers

Right now `bsc` is one package doing three jobs. Separating them is what turns it from "one
project's code" into infrastructure other people cite and use.

**This matters more given four live directions.** D1–D4 all need the same substrate: load a model
without silently wrong head geometry, intervene on activations, seed and record runs, judge
outputs, report rates with CIs. Today only D1 (this project) has that. Promoting Layers 1–2 to
**lab-wide infrastructure** means D2 (adversarially robust SAEs), D3 (pre-commitment probes), and
D4 (deflection vectors) inherit reproducibility for free instead of each re-implementing it — and
every future claim on the site is backed by the same manifest guarantee. That is the single
highest-return engineering decision available right now.

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
**The bridge is already inside the lab — connect D1 to D3.** The site's D3 (*pre-commitment probes
detecting answer encoding before reasoning output, AUC 0.780*) is already a control-shaped result:
it is a probe that reads internal state to catch a model doing something its output does not
reveal. And the site's own stated breakdowns (>98% of configurations show sandbagging; >70% of
backdoors survive fine-tuning) are **the control agenda's motivating claims**, not interp claims.

So the lab is one framing move from being a control lab: **D1 supplies the intervention (steer the
circuit), D3 supplies the detector (read the internal state), and control is the protocol that
combines them** — monitor cheaply, intervene expensively only when the monitor fires. Publishing
D1 and D3 as *one protocol* is far stronger than publishing two probes.

The **compliance-head probe** (AUC 0.943 on 7B) is the D1 half: a **trusted monitor** — cheap,
internal, runs on every forward pass, gates an expensive intervention only when it fires.

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

0. **Audit the website's public numbers — do this first.** The site states `>98%` of
   configurations show sandbagging, `>70%` of backdoors survive fine-tuning, and pre-commitment
   probe `AUC 0.780`. Given that an audit of this project's *own* headline numbers falsified four
   of six (`CLAIMS_AUDIT.md`), every public figure needs the same treatment: trace each to a run
   artifact, or mark it `[UNVERIFIED]`. **These are the lab's most exposed claims** — they are
   public, quotable, and currently unaudited. A funder or reviewer who recomputes one and finds it
   unsupported does far more damage than a missing number. Highest-priority, near-zero cost.
1. **Ship Layer 1+2 software publicly** with docs and tests. Costs nothing, compounds immediately.
2. **Ship paper 2 (evaluation instrument).** Cheap, field-serving, establishes rigor brand.
3. **Land paper 1 at ICLR main** with baselines + cross-model + adaptive attacker.
4. **Publish the audit methodology itself** — a short public writeup of how the harness caught four
   false claims in its own prior work, and add falsifiability as a stated commitment on the site
   (§1). This is unusually strong evidence of research integrity and is exactly what funders and
   collaborators screen for.
5. **Then** raise: with two papers, adopted tooling, and a public audit record, grant applications
   (Manifund/BlueDot/OpenPhil-adjacent, LTFF) and collaborations become a different conversation
   than they are today. Note the current Manifund draft still has `[FILL IN]` ask amounts and
   should not go out carrying any of the four falsified claims.

**Existing credibility is stronger than a from-scratch plan assumes:** three publications (ICML
2026, TAIS 2026, NeurIPS 2025), two co-founders, and MARS V / Cambridge AI Safety Hub affiliation.
The gap is not track record — it is that the *public claims* are not yet backed by the
reproducibility standard the lab is otherwise building.

Compute reality: the Blackwell (95 GB) is sufficient through ~32B. Beyond that needs either
attribution patching (cheap approximation) or real funding.

---

## 6. Risks, stated plainly

- **Scoop risk.** Discovery findings get scooped (this already happened). Mitigate by leading with
  defenses, evaluation, and adversarial robustness — which require *sustained* capability rather
  than a single lucky result.
- **Bandwidth across four directions.** Two co-founders and four live directions (D1–D4) plus five
  paper tracks is over-committed. D1 is the one with a full reproducible pipeline and results
  today — finish it (papers 1+2) before opening more. Shared Layers 1+2 are what make D2–D4
  cheaper later; building them now is leverage, spreading attention now is not.
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
