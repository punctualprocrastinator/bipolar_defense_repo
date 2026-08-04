"""Runnable experiments.

One experiment per module, each exposing ``run(cfg) -> dict``. Registered in
:mod:`bsc.cli` so they are discoverable with ``bsc list``.
"""

from __future__ import annotations

__all__ = [
    "sparsity",
    "crescendo_sweep",
    "logit_lens_sign",
    "discover_circuit",
    "steering_defense",
    "gcg_transfer",
    "bipolar_steering",
    "compliance_calibration",
]
