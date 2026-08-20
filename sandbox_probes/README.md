# Sandbox probes — exact code behind the 2026-08-05 findings

These are the **verbatim** exploratory scripts run on the molab Blackwell sandbox that produced the
`FINDINGS.md` 2026-08-05 entries (Circuit Breakers baseline, HarmBench judge, adaptive benchmark, our
steering-defense degeneration, SteerMoE). They are preserved here so the numbers are traceable
(CLAUDE.md §7) and the code is not lost when the ephemeral sandbox restarts.

**These are NOT paper-grade.** They hardcode `"cuda"`, skip `RunContext`/manifests/seeds-from-config,
and were run by hand. They violate CLAUDE.md §1 and must be **ported to `bsc.experiments`** (config-
driven, device-agnostic, manifest-emitting) before any number goes in the paper. Tracked in `STATUS.md`.

The reusable judge from this work is already ported and tested: `bsc.judge.HarmBenchJudge`.

## Files

| script | produces | FINDINGS entry |
|---|---|---|
| `harmbench_judge.py` | validates the HarmBench classifier judge | "built + validated the HarmBench judge" |
| `cb_harmbench.py` | undefended vs CB, weak set, HarmBench (15%→4%) | "CB *does* defend Crescendo" |
| `harvest_llama.py` | adaptive Crescendo harvester → 11 hard scenarios | "Strengthened the benchmark" |
| `hard_2way.py` | undefended vs CB, hard set (85%→0%) | "CB is even more dominant" |
| `cb_overrefuse.py` | CB benign over-refusal check (none on our set) | "CB does NOT over-refuse" |
| `our_defense_llama.py` | our steering on hard set (0% via **gibberish**) | "our defense hits 0% via degeneration" |
| `fine_sweep.py` | fine α-sweep, coherence + utility breakdown | **interrupted** (sandbox drop) |

## Key models
- base: `NousResearch/Meta-Llama-3-8B-Instruct` (ungated mirror of `meta-llama/Meta-Llama-3-8B-Instruct`)
- Circuit Breakers: `GraySwanAI/Llama-3-8B-Instruct-RR` (+ `Mistral-7B-Instruct-RR`)
- judge: `cais/HarmBench-Llama-2-13b-cls` (tokenizer from a Llama-2 mirror; see `bsc.judge`)
- MoE: `Qwen/Qwen3-30B-A3B`, `allenai/OLMoE-1B-7B-0924-Instruct`

## Data stranded on the sandbox (retrieve when molab returns)
`llama_adaptive_cres.json` — the 11-scenario adaptive benchmark (full 9-line transcripts). Only the
goals are recoverable from logs; the transcripts must be pulled and committed to `src/bsc/data/`.
