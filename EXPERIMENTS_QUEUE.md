# Experiment queue — pre-registered (do not run GPU without an entry here)

Dated 2026-09-04. Purpose: stop wasting GPU hours (this session lost ~30 min to an un-batched L2
run + restarts). Every GPU run must have a row here first: exact config, cost estimate, a **smoke
gate**, and a **pre-registered prediction + falsifier**. Read a result against its pre-registration,
not post-hoc.

## Efficiency rules (learned this session — non-negotiable)

1. **Smoke first, always.** L0 dry-run (CPU) → L1 tiny (`model=Qwen2.5-1.5B`, `data.limit=2`,
   `n_trials=2`) → full. L1 catches every wiring/device bug for ~2 min (it caught the probe device
   bug). Never launch a full run whose code path hasn't smoked.
2. **`generation.batch_trials=true`** on every behavioral run (~2–6× faster; prompt-level stats
   unaffected).
3. **Transcript build is the unbatched bottleneck** (~1 min/transcript on 7B, sequential). Budget
   for it; keep `data.limit` modest until a result is worth scaling; a future perf item is batching
   Phase-1 across goals.
4. **Distinct attacker for any DEFENSE arm.** Self-attack (attacker=receiver) gives ~20% undefended
   ASR on Qwen2.5-7B → underpowered null (FINDINGS 2026-09-04). Mechanism arms (M1) don't need it.
5. **HarmBench-13B is cached on the sandbox now** (first load was the slow part). Keep the sandbox.
6. **Commit artifacts** via FINDINGS + `runs/INDEX.md` immediately (runs/ is gitignored; sandboxes
   are ephemeral — the last one took the adaptive benchmark with it).
7. **Launch detached** (`start_new_session=True`) or the process dies on cell cleanup.

## Pre-registration table

| # | experiment | config (overrides on its yaml) | GPU est | smoke gate | prediction | falsifier |
|---|---|---|---|---|---|---|
| E1 | **Metric validation** — does `refusal_logit_diff` track real refusal? | reuse M1 `readouts.json` (NO new GPU) — correlate per-goal diff vs per-goal refusal | 0 (offline) | n/a | higher diff ⇒ higher refusal rate (ρ>0.3) | no correlation ⇒ metric is not measuring refusal; rebuild it |
| E2 | **Jacobian-lens feasibility** (gated) | `pip install jlens`; inspect API; check hf `neuronpedia/jacobian-lens` for a Qwen2.5-7B lens | ~10 min | import + one `lens.apply` on 1 prompt | a Qwen2.5-7B lens exists / API matches `load_jacobian_lens` | no lens + fit >1 GPU-hr ⇒ **stop**, ship the logit-lens figure (already have it) |
| E3 | **M1 Jacobian rerun** (only if E2 green) | `peer_vs_request` `--set intervention.steering_vectors_path=jlens:<repo>:<file>` `data.limit=30` | ~45 min | L1 with lens on 2 goals | Jacobian layer curve shows the same late-layer peer divergence, more faithfully | curves flat/noisy ⇒ logit-lens figure stands, drop jlens |
| E4 | **C1 propagation chain** | new `propagation_chain` exp on `run_chain`, N=4, distinct attacker, `data.limit=15`, `batch_trials` | ~40 min | L1 chain N=2, 2 goals | ASR grows or holds down-chain; steering one node lowers downstream ASR | if undefended chain ASR ~0 (attacker too weak) ⇒ fix attacker before interpreting |
| E5 | **Distinct-attacker propagation rerun** | `multiagent_propagation` `--set attacker_model=<stronger>` `data.limit=30` `batch_trials` | ~35 min | already smoked | undefended ASR ≫ 20%, bipolar/refusal_only < undefended (p<.05) | still ~null ⇒ report honest null; MoE-style "not defendable at scale" caution |
| E6 | **B1 baselines** | `multiagent_baselines`: prompt-defense + CB checkpoint + AcMAS-style, same benchmark/judge | ~45 min | L1 each defense on 2 goals | our steering ≈ prompt defense; CB strong; AcMAS fails on framing-content | — |

## Recommended order (cheap→expensive, mechanism→defense)

1. **E1** (free, offline) — validates the M1 instrument the whole mechanism story rests on.
2. **E2** (cheap gate) — decides whether jlens is worth any GPU at all; if red, we already have the
   logit-lens figure and skip E3.
3. **E3** (if green) — the faithful layer figure.
4. **E4 / E5** need a **distinct attacker model** chosen first (see open decision below) — do not
   launch until that's picked, or they null out like the self-attack run.
5. **E6** last (most models to load).

## Open decision before E4/E5/E6 (needs a pick, not GPU)

**Which attacker model?** Options: (a) a larger Qwen (Qwen2.5-32B/72B) as a stronger same-family
peer; (b) a deliberately less-aligned instruct model; (c) keep self-attack but strengthen the
compromised-peer system prompt. This choice gates every defense arm — decide before spending GPU.

## Done this session (for reference)
- L1 smoke ✓ · L2 propagation (self-attack null, controls pass) ✓ · M1 + length control ✓
  (framing effect 27/30, p~1e-5) · M1 layer figure (logit lens; late-layer divergence, peak L25) ✓
