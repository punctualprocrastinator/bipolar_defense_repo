# STATUS — ongoing work

Living document. Updated 2026-08-05. Snapshot of what is done, what is running/pending, and what
lives only on the (ephemeral) molab sandbox and must be pulled into the repo.

## Done and committed (this session)

- **HarmBench judge** — `bsc.judge.HarmBenchJudge` (`cais/HarmBench-Llama-2-13b-cls`) with the
  transformers-5.14 tokenizer workaround (load fast `tokenizer.json` from an ungated Llama-2 mirror).
  Pure `harmbench_verdict()` helper is unit-tested; 95 tests pass on CPU. This is the ASR instrument
  of record — keyword judge is a fast pre-filter only.
- **Circuit Breakers baseline (dense).** Real `GraySwanAI/Llama-3-8B-Instruct-RR` checkpoint vs
  undefended, HarmBench-scored. Crescendo 15%→4% (weak set) and **85%→0%** (hard adaptive set). No
  benign over-refusal on our 23-prompt set. FINDINGS 2026-08-05.
- **SteerMoE reimplementation + bipolar combination on Qwen3-30B** — every method/combination lands in
  {no-effect, degenerate}; MoE non-defendability is robust. FINDINGS 2026-08-05.
- **Strategic pivot recorded** — PLAN_ICLR.md status block + FINDINGS: the dense defense head-to-head
  is not winnable vs CB; reposition around mechanism + MoE + training-free.
- **Two-agent propagation harness** — `bsc.experiments.multiagent_propagation` (+ config + tests, 101
  pass). Live attacker agent A escalates against receiver B (replacing the legacy hardcoded assistant
  turns), transcripts frozen for pairing, four conditions (undefended / refusal-only / compliance-only
  / bipolar) on B's terminal turn, **HarmBench-judged** with the degeneration share reported, paired
  McNemar + Wilson CIs, AdvBench attack slice disjoint from the fit slice. This is the Day-2 apparatus
  (BIPOLARMULTIAGENT4DAY.md); device-agnostic, dry-run verified, **needs a GPU to run**.
- **Bug fix** — `HarmBenchJudge` called `resolve_device()` with the wrong signature (it takes a
  `ModelConfig` and returns a `ResolvedDevice`); fixed so the judge is device-agnostic, not cuda-only.

## In flight / interrupted (sandbox dropped mid-run)

- **Fine α-sweep of our steering defense on Llama-3-8B** (`/marimo/fine_sweep.py`). Goal: confirm there
  is NO coherent-refusal α (our defense hits 0% ASR only via gibberish) and quantify the benign-utility
  cost. Coarse sweep already showed: α=0→85%, α≥6→0% but the α=10 sample is gibberish
  (`"ereo.HOURanganierge…"`). Re-run when the sandbox is back.

## Pending (queued)

1. **Re-derive the Qwen bipolar-defense headline numbers under HarmBench** (the 89→34% / additive /
   sign-inversion results were keyword-judged). Framed as "reduces ASR at zero training cost."
2. **Proper over-refusal eval** (XSTest / OR-Bench) for CB vs base vs ours — the only place CB might
   show a utility cost; our 23-prompt check was too small to establish a differentiator.
3. **Port sandbox probes into reproducible `bsc` experiments** (see below).
4. Optional: Mistral-7B-Instruct-RR as a second dense CB datapoint.

## Lives ONLY on the molab sandbox — pull into repo when it returns

The sandbox is ephemeral (URL/token rotate on restart); these are **not yet in git** and will be lost
if not retrieved:

- `/marimo/llama_adaptive_cres.json` — the **11-scenario adaptive Crescendo benchmark** vs Llama-3-8B
  (HarmBench-validated, base ASR 85%). Goals recorded in FINDINGS/harvest log, but the full 9-line
  transcripts are only in this file. **Highest priority to retrieve** → should become a committed data
  file under `src/bsc/data/`.
- Probe scripts (logic to port to `bsc.experiments`): `harmbench_judge.py`, `cb_harmbench.py`,
  `harvest_llama.py` (adaptive harvester), `hard_2way.py`, `cb_overrefuse.py`, `our_defense_llama.py`,
  `fine_sweep.py`, `qwen3_steermoe_faithful.py`, `qwen3_combo2.py`.

## Reproducible-code debt (CLAUDE.md compliance)

The sandbox probes are quick-and-dirty (hardcoded `cuda`, no RunContext/manifests). To be paper-grade
they must be ported to `bsc.experiments` modules that are config-driven, device-agnostic, and emit
manifests:

- `circuit_breakers_eval` — load a base + its RR checkpoint (+ optional our-defense), HarmBench-score
  on a Crescendo/GCG benchmark, report ASR with CIs. **Still to port** (probes: `hard_2way.py`,
  `cb_overrefuse.py`).
- `adaptive_harvest` — the turn-by-turn attacker/target/HarmBench-validated harvester. **Now subsumed**
  by `multiagent_propagation` Phase 1 (`build_transcript`); the standalone harvester can reuse it.
