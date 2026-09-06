# Bipolar Safety Circuits

**Refusal in safety-tuned LLMs is not only a function of *content* — in some models it is also a
function of *how that content is framed*. This repo measures that, localizes it, exploits it, and
defends against it.**

Target venue: **TMLR** (rolling submission; claims-supported + interest criteria — see
[`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md)). Research log of record: [`FINDINGS.md`](FINDINGS.md).
Working rules: [`CLAUDE.md`](CLAUDE.md). 129 tests pass on CPU.

---

## 1. What we found

### A. Framing-sensitivity is a measurable **model property** (8 models)

Identical harmful content, presented as a *user request* vs a *peer agent's task-to-continue*, with a
length-matched control. Sign test on a validated output-logit refusal-disposition metric, n=30 goals.

| model | request_long > peer | gap | group |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | **29/30** | +23.0 | framing-conditional |
| Mistral-7B-Instruct-v0.3 | **28/30** | +6.6 | framing-conditional |
| Qwen2.5-7B-Instruct | **27/30** | +14.7 | framing-conditional |
| Qwen3-8B | 17/30 | +2.1 | insensitive (chance) |
| Phi-3.5-mini-instruct | 16/30 | +4.2 | insensitive (chance) |
| Meta-Llama-3-8B-Instruct | 14/30 | +0.8 | insensitive (chance) |
| gemma-2-9b-it | 11/30 | −0.2 | insensitive |
| OLMo-2-1124-7B-Instruct | 0/30 | −12.0 | saturated-refusing |

**Qwen2.5 → Qwen3 lost the effect** (27/30 → 17/30): the newer generation of the same family appears to
have closed it.

### B. Mechanism — a **late-layer** effect, confirmed by two lenses

Curves are identical early and diverge only in the last layers. Logit lens: peak gap **+15 at L25/29**.
Jacobian Lens (Anthropic, pretrained Qwen2.5-7B lens): gap ~0 through L9, **peak +11.9 at L26/27**.
Consistent with harm being detected while refusal *expression* is attenuated late.

### C. Defense — steering the refusal/compliance circuit, on **validated** attacks

Qwen2.5-7B, n=100 per family, HarmBench judge, additive sign-correct steering (alpha=8):

| metric | undefended | bipolar | random control |
|---|---|---|---|
| multiturn_crescendo ASR | 70% | **16%** | 74% |
| persona_fiction ASR | 90% | **18%** | 84% |
| **XSTest over-refusal** | **2%** | **27%** | — |

One direction, fit on a *disjoint generic* contrast, transfers across two attack families; the
matched-norm random control is null. **It is not free: +25 points of over-refusal.** Always report the
pair.

### D. Network — propagation decays, and one node can halt it

4-agent chain: cascade **75% → 28% → 19% → 6%**. Steering **agent B alone** drops B's ASR
**28% → 0% (McNemar p=0.002)**, random control null, outputs coherent. Downstream protection is
floor-limited (positions C/D were already low).

---

## 2. Negative results and corrections (read these)

Kept deliberately prominent — they are the credibility of everything above.

| finding | status |
|---|---|
| **M1 does not replicate on Llama-3-8B** (14/30 = chance) | the framing effect is **model-dependent**, not universal |
| **OLMo-2 "reversal" is not a reversal** | it refuses *both* framings identically; saturated-refusing. Caught and corrected before publication |
| **MoE refusal is not defendable** (OLMoE, Qwen3-30B) | routing patches, per-expert steering, residual steering and SteerMoE-style expert forcing all fail or degenerate |
| **Circuit Breakers beats our steering on dense** | CB drove hard adaptive Crescendo **85% → 0%** with no measurable over-refusal on our benign set |
| **The r=0.948 metric validation is narrower than first stated** | it validates against the *greedy first token*, not behavior (2402.14499: >60% divergence, worst in safety-tuned models) |
| **Self-attack propagation was an underpowered null** | fixed by switching to a distinct, less-aligned attacker |
| Keyword judge is unusable for rerouting defenses | 14–35% agreement with HarmBench; HarmBench is the instrument of record |

---

## 3. Known threats to validity (close before submission)

From the intent literature review ([`INTENT_PROGRAM.md`](INTENT_PROGRAM.md)):

1. **Wrapper anti-ranking** (2608.09624) — under wrappers, internal safety scores *anti-ranked* attack
   success (AUROC 0.220). **Our peer framing is a wrapper.** Report disposition-vs-outcome AUROC
   **within each framing arm**, never pooled. *Highest priority: it could invert the paper.*
2. **Length-matching may not suffice** (2605.01048) — add a **paraphrase-only null arm**.
3. **Format confound** (2603.19426) — add a **format-decorrelation arm**.
4. **Random-direction null is necessary but not sufficient** (SteerCheck 2608.24335) — add the cosine
   distribution, a mean-ablation, and a polarity-reversal control.
5. **Positioning:** "refusal depends on the inferred sender" is **scooped** (Ghandeharioun 2406.12094).
   Lead with *spoofing* and *cross-model heterogeneity*; run their persona-steering as a live baseline.
6. **Everything behavioral is Qwen-centric** — the Mistral defense arm is the fix (currently unrun).
7. Cite **MULI (2405.18822)** as prior art for the first-token-logit metric.

---

## 4. Document map

| doc | what it is |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **Research log of record** — dated, append-only, every result and every correction |
| [`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md) | Venue analysis, verified deadlines, scoping |
| [`INTENT_PROGRAM.md`](INTENT_PROGRAM.md) | Multi-agent **intent** program (MI1–MI4) and its literature review |
| [`MULTIAGENT_DESIGN.md`](MULTIAGENT_DESIGN.md) | Multi-agent arm design; §7 is the related-work delta table |
| [`EXPERIMENTS_QUEUE.md`](EXPERIMENTS_QUEUE.md) | Pre-registered runs: config, cost, smoke gate, prediction, falsifier |
| [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) | Audit of the original (legacy) claims |
| [`STATUS.md`](STATUS.md) | In-flight work and artifacts stranded on ephemeral sandboxes |
| [`CLAUDE.md`](CLAUDE.md) | The rules all code here follows |
| `PLAN_ICLR.md`, `LAB_STRATEGY.md`, `CRESCENDO_CIRCUIT_METHOD.md` | earlier planning, partly superseded |

---

## 5. Code

```
src/bsc/
  agents.py        N-agent network primitive (Agent, DefenseSpec, run_chain)
  probes.py        refusal readout: logit-diff, head-mass, layer curves, Jacobian-lens loader
  judge.py         HarmBenchJudge (instrument of record) + keyword fast path + degeneration detector
  hooks.py         head interventions (GQA-correct; identity is a bit-exact no-op)
  circuits.py      circuit maps, head selection, sparsity
  metrics.py       Wilson intervals, paired McNemar, bootstrap
  models.py        device/dtype resolution, head-geometry verification
  generation.py    paired greedy + sampled trial sets, batched trials
  experiments/     peer_vs_request (M1), multiagent_propagation, propagation_chain,
                   discover_circuit, steering_defense, gcg_transfer, crescendo_sweep, ...
sandbox_probes/    verbatim exploratory scripts behind the 2026-08/09 findings (not paper-grade)
```

### Install and run

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,viz]"
pytest -q                                   # 129 tests, CPU-only, no network, no downloads
bsc list                                    # available experiments
bsc peer_vs_request --config configs/peer_vs_request.yaml --set data.limit=2 --dry-run
```

### Reproducibility guarantees

| property | where |
|---|---|
| Seeded; per-trial seeds derived deterministically | `bsc.determinism` |
| Every run writes `manifest.json` (config + hash, git SHA, versions, GPU, dtype, argv) | `bsc.runs`, `bsc.provenance` |
| No magic numbers — typed, YAML-backed config | `bsc.config` |
| Runs never overwrite: `runs/<exp>/<UTC-timestamp>-<confighash>/` | `bsc.runs.RunContext` |
| Rates carry n and a CI; paired comparisons use McNemar | `bsc.metrics` |
| Nothing hardcodes `cuda`; head slicing is GQA-correct; identity is a bit-exact no-op | `bsc.models`, `bsc.hooks` |
| Batched trials (~2–6× faster) without breaking prompt-level pairing | `bsc.generation` |

**Note:** `runs/*/` is gitignored — only `runs/INDEX.md` is tracked. GPU work runs on *ephemeral*
sandboxes, so numbers are committed to `FINDINGS.md` as soon as they exist. Two runs have already been
lost to sandbox drops (see [`STATUS.md`](STATUS.md)).

### Hardware

`bsc.models.resolve_device` picks device and dtype and records why. Blackwell (sm_120) bf16 is the
target for real experiments; pre-Ampere falls back to fp16 and logs it; CPU is fp32 for tests and tiny
smoke tests only. If PyTorch has no compiled kernels for the detected SM, loading raises immediately
with the compiled arch list rather than failing later at the first matmul.

---

## 6. Open work

**Before submission (P0):** paraphrase-null and format-decorrelation arms; per-arm
disposition-vs-outcome AUROC; re-run the alpha-frontier and the Mistral defense arm (both lost to
sandbox drops); strengthened steering nulls; the persona-steering baseline.

**Next program (ICML-scale):** the multi-agent **intent** experiments MI1–MI4 — in particular
**path-patching the sender-intent direction into the bipolar refusal/compliance heads**, which would
unify the mechanism and defense halves of the project.
