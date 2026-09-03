"""Pure-logic tests for the two-agent propagation experiment.

Only the condition -> head-edit mapping is tested here; the run itself needs two models plus the
HarmBench judge and is a GPU experiment. The key guarantee is the identity no-op (CLAUDE.md §5):
the ``undefended`` condition must produce an empty edit set, so the baseline is a true baseline.
"""

from __future__ import annotations

import pytest

from bsc.config import ExperimentConfig, InterventionConfig
from bsc.experiments.multiagent_propagation import CONDITIONS, condition_edits
from bsc.hooks import Head

REFUSAL = [Head(16, 6), Head(14, 11)]
COMPLIANCE = [Head(15, 0), Head(30, 14)]


def _cfg(mode: str = "multiplicative") -> ExperimentConfig:
    return ExperimentConfig(
        intervention=InterventionConfig(
            mode=mode, refusal_multiplier=3.0, compliance_scale=0.0, ablate_compliance=True
        )
    )


def test_conditions_are_the_expected_four():
    assert CONDITIONS == ("undefended", "refusal_only", "compliance_only", "bipolar")


def test_undefended_is_identity_noop():
    # The baseline must perturb nothing: empty edit set (CLAUDE.md §5).
    assert condition_edits("undefended", REFUSAL, COMPLIANCE, _cfg()) == {}


def test_refusal_only_touches_only_refusal_heads():
    edits = condition_edits("refusal_only", REFUSAL, COMPLIANCE, _cfg())
    assert set(edits) == set(REFUSAL)
    assert all(e.scale == 3.0 for e in edits.values())


def test_compliance_only_ablates_only_compliance_heads():
    edits = condition_edits("compliance_only", REFUSAL, COMPLIANCE, _cfg())
    assert set(edits) == set(COMPLIANCE)
    assert all(e.scale == 0.0 for e in edits.values())  # compliance_scale=0.0 == zero-ablation


def test_bipolar_touches_both_populations():
    edits = condition_edits("bipolar", REFUSAL, COMPLIANCE, _cfg())
    assert set(edits) == set(REFUSAL) | set(COMPLIANCE)


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        condition_edits("bogus", REFUSAL, COMPLIANCE, _cfg())


# -- additive (sign-correct) mode ------------------------------------------------------------

import torch  # noqa: E402  (kept local to the additive tests)

_VECS = {h: torch.ones(4) for h in REFUSAL + COMPLIANCE}


def test_additive_undefended_still_identity():
    assert condition_edits("undefended", REFUSAL, COMPLIANCE, _cfg("additive_steering"), _VECS) == {}


def test_additive_refusal_only_steers_only_refusal_heads():
    edits = condition_edits("refusal_only", REFUSAL, COMPLIANCE, _cfg("additive_steering"), _VECS)
    assert set(edits) == set(REFUSAL)
    # additive: a vector is attached and alpha is non-zero (not a multiplicative scale).
    assert all(e.vector is not None and e.alpha == 1.0 and e.scale == 1.0 for e in edits.values())


def test_additive_bipolar_steers_both_populations_additively():
    edits = condition_edits("bipolar", REFUSAL, COMPLIANCE, _cfg("additive_steering"), _VECS)
    assert set(edits) == set(REFUSAL) | set(COMPLIANCE)
    assert all(e.vector is not None for e in edits.values())


def test_additive_without_vectors_raises():
    with pytest.raises(ValueError):
        condition_edits("bipolar", REFUSAL, COMPLIANCE, _cfg("additive_steering"), None)
