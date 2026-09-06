# Submission plan — workshop → ICLR main → ICML

Dated 2026-09-06. Ladder: **NeurIPS workshop (UniReps) → ICLR main → ICML on feedback.**
⚠ **Deadlines are NOT verified here — check the actual CFPs before planning around any date.**

## 0. Venue fit (honest)

**UniReps ("Unifying Representations in Neural Models") — good fit, if framed representationally.**
Our strongest asset is a *cross-model representational* result: 8 models differ systematically in whether
refusal is bound to **request framing** or to **content**, with a layer-resolved mechanism (two lenses)
and a validated cheap probe. UniReps cares about when/why different models converge or diverge in
representation — a divergence result with a mechanism is on-topic. **Frame it as representation, not as
a defense paper.**

**ICLR main — needs the causal story, not just the survey.** A measurement + heterogeneity paper is
workshop-strength. For main-track we need the causal chain (sender-intent direction → refusal/compliance
heads) plus the defense with its utility frontier and real baselines.

## 1. Scope split

### Workshop paper (UniReps) — "framing-conditional refusal is a model property"
Content: cross-model survey (8 models) · length-matched design · layer localization (logit + Jacobian
lens) · the Llama-3 dissociation puzzle (2606.09844 finds the opposite-signed interlocutor effect in the
family where ours vanishes) · honest negatives (OLMo saturation, disposition≠behavior).
**Deliberately EXCLUDES** the defense/ASR numbers — keeps one idea.

### ICLR main — "refusal survives as recognition but fails as action under spoofed sender intent"
Adds: MI1–MI4 intent program · the causal path-patch from the intent direction into our bipolar heads ·
defense on validated attacks **with** the ASR-vs-over-refusal frontier · cross-model defense (Mistral) ·
baselines (Ghandeharioun persona-steering; Circuit Breakers) · adaptive attacker.

## 2. P0 — mandatory controls before ANY submission (from the intent lit review)
These are not optional; each maps to a named reviewer objection.
1. **Paraphrase-only null arm** (2605.01048) — length-matching alone may be indistinguishable from
   paraphrase. Report per-sample regression, not aggregate rates.
2. **Format-decorrelation arm** (2603.19426) — agentic scaffolding on a *user* prompt; plain prose on a
   *peer* prompt. Prove we measure sender framing, not format.
3. **Disposition-vs-judged-outcome AUROC, computed WITHIN each framing arm** (2608.09624) — our framing
   is a wrapper, and pooled AUROC hides the anti-ranking failure mode (their AUROC 0.220).
4. **Soften the metric claim** — r=0.948 is agreement with the greedy first token, not behavior
   (2402.14499, >60% divergence). Already corrected in FINDINGS.
5. **Strengthen the steering null** (SteerCheck 2608.24335) — cosine distribution of controls +
   mean-ablation + polarity-reversal, not just matched-norm random.
6. **Cite MULI (2405.18822)** as prior art for the first-token-logit metric.
7. **Position against Ghandeharioun (2406.12094) in paragraph 1**, and run persona-steering as a live
   baseline (same seeds/judge, per CLAUDE.md §2).

## 3. Experiment queue (ordered; smoke each at limit=2 first)
| # | run | for | est |
|---|---|---|---|
| P0-a | paraphrase-null + format-decorrelation arms on the 8-model survey | both | ~40 min |
| P0-b | disposition-vs-outcome AUROC per framing arm (needs sampled gens + judge) | both | ~30 min |
| R1 | re-run α-frontier (ASR vs XSTest vs ordinary-benign over-refusal) — LOST to sandbox drop | ICLR | ~45 min |
| R2 | Mistral defense arm (circuit + validated attacks) — LOST to sandbox drop | ICLR | ~60 min |
| MI1 | intent-conditioned refusal (user / peer_neutral / peer_verified / peer_flagged) | ICLR | ~20 min |
| MI3 | intent spoofing on validated attacks | ICLR | ~30 min |
| MI2 | sender-intent direction: decodability + causal add/subtract + controls | ICLR | ~60 min |
| MI2b | **path-patch intent direction → bipolar heads** (AtP*, 2403.00745) — the unifying result | ICLR | ~45 min |
| B1 | baselines: persona-steering (2406.12094), Circuit Breakers | ICLR | ~45 min |

## 4. Writing plan
- **Figures:** (F1) 8-model survey bar chart with the three groups; (F2) layer×disposition curves (have,
  both lenses); (F3) ASR-vs-over-refusal frontier (needs R1); (F4) causal path-patch schematic (needs MI2b).
- **Related work:** already drafted — MULTIAGENT_DESIGN.md §7 delta table + INTENT_PROGRAM.md review.
- **Limitations section:** we have unusually strong material (Llama null, OLMo correction, Qwen-centricity,
  disposition≠behavior, wrapper threat). Lead with it; it is a credibility asset.
- **Repro appendix:** per-run manifests + seeds already exist.

## 5. Top risks
1. **Q2 is scooped** (2406.12094) — mitigated only by leading with the *spoofing* + *heterogeneity* framing.
2. **Everything behavioral is Qwen-centric** — R2 (Mistral) is the fix and is currently unrun.
3. **Wrapper anti-ranking** could invalidate disposition-based claims — P0-b decides this. Run it early;
   if disposition anti-ranks within arms, the survey becomes a *negative* methodological result (still
   publishable, very different paper).
