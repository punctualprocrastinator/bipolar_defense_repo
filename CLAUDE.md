# Research Programming Standard — Bipolar Safety Circuits

This file is loaded automatically every session. It governs **all code written in this repo**.
The goal is a NeurIPS/ICLR-tier submission, which means every number in the paper must be
regenerable by a stranger from a single command, and every claim must be traceable to an
artifact on disk.

## 0. Project context

- **Research:** bipolar safety circuits — refusal heads vs. compliance heads in safety-tuned
  LLMs, and an inference-time defense against GCG (single-turn) and Crescendo (multi-turn).
- **Target:** NeurIPS/ICLR main conference. ACL SRW submission already made on the GCG portion.
- **New code goes in `bsc/`** (installable package, `src/bsc/`). Legacy scripts in the repo root,
  `bipolar_defense_repo/`, and `crescendo_mech_interp/` are **read-only reference**: port from
  them, cite them in docstrings, do not extend them.
- **Compute:** experiments target an NVIDIA Blackwell GPU. The dev machine is a GTX 1650 (4 GB,
  Turing, CPU-only torch build) — it can run tests and tiny-model smoke tests **only**. Never
  assume CUDA is present; never hardcode `"cuda"`.

## 1. Non-negotiables

Every experiment script MUST:

1. **Be deterministic.** Call `bsc.determinism.set_seed(seed)` before anything stochastic. Seeds
   come from config, never from a literal in the body. Record the seed in the manifest.
2. **Emit a run manifest.** Use `bsc.runs.RunContext`. Every run writes `manifest.json` with the
   resolved config, config hash, git SHA + dirty flag, package versions, GPU/driver, dtype,
   wall-clock, and the exact CLI argv. A result without a manifest does not exist.
3. **Be config-driven.** No magic numbers in function bodies. Layer indices, multipliers,
   thresholds, model names, prompt counts, seeds all live in a dataclass config loaded from YAML
   and overridable from the CLI.
4. **Write artifacts to `runs/<experiment>/<timestamp>-<confighash>/`,** never next to the source.
   Never overwrite a previous run's directory.
5. **Report uncertainty.** Any rate (ASR, FPR, compliance rate) is reported as point estimate +
   bootstrap 95% CI + n, via `bsc.metrics`. A bare percentage with no n is not a result.
6. **Separate compute from plotting.** Experiments write JSON/Parquet. Plot scripts read those
   files and write figures. A plot script never calls a model.
7. **Be device-agnostic.** Get device/dtype from `bsc.models.resolve_device()`. No `.to("cuda")`,
   no `device_map="cuda"`, no `torch.cuda.*` outside `bsc/models.py` and `bsc/provenance.py`.

## 2. Statistical and scientific rules

- **Never report a single greedy generation as an experimental result.** Every behavioral claim
  needs `n_trials` sampled generations at fixed seeds, plus the greedy run reported separately.
  A single sample is an anecdote.
- **Held-out discipline.** Circuits are discovered on a `dev` split and evaluated on a `test`
  split. Never tune a threshold or multiplier on the split you report on. Splits are defined in
  config and recorded in the manifest.
- **Refusal classification is a measurement instrument, not a helper.** Use
  `bsc.judge`. Keyword matching is the fast path but must be validated against an LLM judge on a
  labeled subset, and the agreement rate is reported in the paper. Never silently swap judges
  between conditions being compared.
- **Baselines run in the same process, same seeds, same judge as the treatment.** Comparing a
  number from an old JSON to a freshly computed one is invalid.
- **Ablations are mandatory for causal claims.** "Head X causes Y" needs a patch/ablate result,
  not a correlation between an activation norm and behavior. This repo has already been burned
  by this: refusal-head norm does *not* predict refusal behavior (RESEARCH_OVERVIEW.md §2.3).
- **Record negative and non-reproducing results in `FINDINGS.md`.** Do not delete an experiment
  that contradicted an earlier claim; supersede it with a dated entry and a pointer to the run
  directory.

## 3. Model-internals correctness

- **GQA is a trap.** Qwen2.5-7B has 28 query heads and 4 KV heads. Head slicing on `o_proj`'s
  *input* is indexed by **query** heads and `head_dim = hidden_size // num_attention_heads`, but
  any code touching K/V must use `num_key_value_heads`. Use the helpers in `bsc.hooks`; do not
  recompute head offsets inline.
- **Interventions must be applied identically when measuring and when generating.** A prior
  version of this project measured norms without hooks and generated with them, producing a
  discrepancy of exactly the amplification factor. `bsc.interventions` applies one hook set for
  both; never write a second code path.
- **Position matters.** State whether an intervention applies to the last position only or all
  positions. During generation with a KV cache, `hidden[:, -1, :]` is the only position present
  after the prefill — this makes "last position only" and "all positions" silently identical
  during decode but different during prefill. Make it an explicit config field.
- **Hooks are always removed in a `finally` block or via the `bsc.hooks.applied()` context
  manager.** A leaked hook silently contaminates every subsequent run in the process.

## 4. Code style

- Python 3.11+, type hints on every public function, `from __future__ import annotations`.
- Dataclasses for config and results. No dicts-as-records crossing a function boundary.
- `pathlib.Path` everywhere; no string path concatenation.
- Logging via `bsc.runs` logger, not `print`, in library code. `print` is fine in CLI entry points.
- Docstrings on experiment functions state: what claim this tests, what it writes, and the
  legacy script it was ported from.
- No notebooks in the reproducibility path. Notebooks may explore; results come from the CLI.

## 5. Testing

- `pytest` must pass on CPU with no network and no model download. Anything needing weights is
  marked `@pytest.mark.gpu` or `@pytest.mark.slow` and skipped by default.
- Every intervention has a unit test asserting it is a **no-op when its strength parameter is
  identity** (multiplier 1.0, zero heads ablated) — this catches indexing bugs that would
  otherwise look like real effects.
- Every metric has a test against a hand-computed value.
- Run `pytest -q` before claiming any code works.

## 6. Workflow

- Before running an expensive job: run it with `--limit 2 --dry-run` first.
- After a run completes, append a one-line entry to `runs/INDEX.md` (run dir, experiment, headline
  number, seed).
- Paper numbers are pulled by `bsc.paper.collect` from run directories into `paper/numbers.json`,
  and the LaTeX reads from that. Never type a number into the `.tex` by hand.

## 7. Honesty rules

- If a result does not reproduce, say so in the response and in `FINDINGS.md`. Do not quietly
  re-tune until it does.
- Never write a number into a doc, paper, or grant application that is not present in a run
  artifact. If a number is needed and unavailable, write `[UNVERIFIED]`, not a plausible value.
- Distinguish "not yet run" from "run and negative" in every status table.
