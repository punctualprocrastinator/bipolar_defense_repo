# bsc — Bipolar Safety Circuits

Reproducible research code for the refusal/compliance head circuit work.
The rules this package enforces are in [`../CLAUDE.md`](../CLAUDE.md); the state of the
existing results is in [`../CLAIMS_AUDIT.md`](../CLAIMS_AUDIT.md).

## Why this exists

The legacy scripts (repo root, `bipolar_defense_repo/`, `crescendo_mech_interp/`) produced the
current results but cannot be re-run to reproduce them: no seeds in most paths, six mutually
incompatible refusal classifiers, results written next to source and overwritten each run,
device and dtype hardcoded, and no record of what produced any number. An audit of the headline
claims against the raw data found four unsupported or contradicted numbers. This package is the
replacement for the reproducibility path. The legacy trees stay as read-only reference.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev,viz]"
pytest -q                      # must pass on CPU, no network, no model download
```

On a GPU node, add the attack extras: `uv pip install -e ".[dev,viz,attacks]"`.

## Guarantees

| Property | Where |
|---|---|
| Seeded, with per-trial seeds derived deterministically | `bsc.determinism` |
| Every run writes `manifest.json`: config, config hash, git SHA, package versions, GPU, dtype | `bsc.runs`, `bsc.provenance` |
| No magic numbers — everything is a typed, YAML-backed config field | `bsc.config` |
| Runs never overwrite: `runs/<experiment>/<UTC-timestamp>-<confighash>/` | `bsc.runs.RunContext` |
| Rates carry n and a confidence interval; paired comparisons use McNemar | `bsc.metrics` |
| Device/dtype resolved once, explained in the manifest; nothing hardcodes `cuda` | `bsc.models` |
| Head slicing is GQA-correct and bounds-checked; identity interventions are bit-exact no-ops | `bsc.hooks` |
| One refusal judge, with an explicit `NONRESPONSE` class and a keyword-vs-LLM agreement report | `bsc.judge` |

## Hardware

`bsc.models.resolve_device` picks device and dtype and records why.

- **Blackwell (sm_100/sm_120)** — bf16, the target for real experiments. Verified against
  an RTX PRO 6000 Blackwell Server Edition (95 GB, torch 2.11+cu130, sm_120).
- **Pre-Ampere (sm_75, e.g. GTX 1650)** — no native bf16; falls back to fp16 and logs it.
  4 GB will not hold a 7B model. Smoke tests only.
- **CPU** — fp32. Tests and tiny-model smoke tests only.

If PyTorch has no compiled kernels for the detected SM, loading raises immediately with the
compiled arch list, rather than failing later at the first matmul.

## Layout

```
src/bsc/
  config.py        typed config, YAML + CLI overrides, validated before a model loads
  determinism.py   set_seed, trial_seed
  provenance.py    git / packages / GPU capture, config hashing
  runs.py          RunContext: run dirs, manifests, artifacts, runs/INDEX.md
  models.py        device+dtype resolution, HeadGeometry (GQA), ModelBundle
  hooks.py         Head, HeadEdit, HeadIntervention, applied(), build_bipolar_edits
  circuits.py      circuit map load/save, head selection, sparsity metrics
  generation.py    paired greedy+sampled trial sets
  judge.py         refusal/compliance/nonresponse classification
  metrics.py       Wilson, bootstrap, McNemar, Cohen's d, AUC, threshold sweeps
  experiments/     runnable experiments (one file each)
  cli.py           `bsc <experiment> --config ... --set key=value`
tests/             CPU-only, no network
```

## Running an experiment

```bash
bsc list
bsc sparsity --set circuit.top_k_refusal=10          # CPU, no model needed
bsc --help
```

Every run prints its run directory and appends one line to `runs/INDEX.md`.
