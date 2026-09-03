"""Command-line entry point.

Experiments are run through here, never by executing a module with side effects at import time
(a legacy pattern where ``import eval_defense`` started a multi-hour GPU job).

    bsc list
    bsc sparsity
    bsc sparsity --config configs/sparsity.yaml --set circuit.top_k_refusal=10 --seed 7
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from bsc.config import ExperimentConfig, load_config
from bsc.determinism import set_seed

# name -> (loader, one-line description). Loaders are lazy so `bsc list` does not import torch.
EXPERIMENTS: dict[str, tuple[Callable[[], Any], str]] = {}


def _register(name: str, description: str) -> None:
    def loader() -> Any:
        from importlib import import_module

        return import_module(f"bsc.experiments.{name}").run

    EXPERIMENTS[name] = (loader, description)


_register("sparsity", "Recompute circuit sparsity for all circuit maps (CPU, no model)")
_register("crescendo_sweep", "Crescendo bipolar-defense multiplier sweep (settles P0-4)")
_register("logit_lens_sign", "Logit-lens sign check on refusal heads (tests L25-H1 inversion)")
_register("discover_circuit", "Harm-contrastive circuit discovery for one model")
_register("steering_defense", "Additive (sign-correct) CAA steering defense on Crescendo")
_register("gcg_transfer", "GCG circuit discovery + bipolar defense + cross-attack transfer")
_register("bipolar_steering", "Per-head refusal steering + compliance ablation (true bipolar)")
_register("compliance_calibration", "Steer compliance heads toward refusal (calibrated) vs zero-ablate")
_register("harvest_scenarios", "Generate + validate more Crescendo scenarios to grow the benchmark")
_register("multiagent_propagation", "Two-agent jailbreak propagation; does bipolar steering intercept it (HarmBench)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsc",
        description="Bipolar Safety Circuits experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every run writes runs/<experiment>/<timestamp>-<confighash>/manifest.json",
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="experiment to run, or 'list' to show available experiments",
    )
    parser.add_argument("--config", help="path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set model.name=Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--seed", type=int, help="shorthand for --set seed=N")
    parser.add_argument("--notes", default="", help="free text recorded in the manifest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the config, then exit without running",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.experiment in (None, "list"):
        print("Available experiments:\n")
        for name, (_, description) in sorted(EXPERIMENTS.items()):
            print(f"  {name:<14} {description}")
        print("\nRun one with:  bsc <experiment> [--config FILE] [--set key=value ...]")
        return 0

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment {args.experiment!r}", file=sys.stderr)
        print(f"available: {', '.join(sorted(EXPERIMENTS))}", file=sys.stderr)
        return 2

    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")

    try:
        cfg: ExperimentConfig = load_config(args.config, overrides)
    except (ValueError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    cfg.experiment = args.experiment
    if args.notes:
        cfg.notes = args.notes
    cfg.dry_run = cfg.dry_run or args.dry_run

    if cfg.dry_run:
        import json

        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        print("\n(dry run: nothing executed)", file=sys.stderr)
        return 0

    # Seeded before the experiment sees control, so nothing stochastic can precede it.
    seed_state = set_seed(cfg.seed, strict=cfg.strict_determinism)
    logging.getLogger("bsc.cli").info("seeded with %d", seed_state.seed)

    run_fn = EXPERIMENTS[args.experiment][0]()
    run_fn(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
