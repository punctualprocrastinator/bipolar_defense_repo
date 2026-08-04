"""Bipolar Safety Circuits — refusal/compliance head circuits and inference-time defenses.

Reproducibility contract (see CLAUDE.md, loaded automatically each session):
every experiment is seeded from config, writes a run directory containing a manifest with full
environment provenance, and reports rates with sample sizes and confidence intervals.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "determinism",
    "generation",
    "hooks",
    "judge",
    "metrics",
    "models",
    "provenance",
    "runs",
]
