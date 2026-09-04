"""Tests for the mechanism probe's pure metric. The full readout needs a model (GPU path); the
refusal-logit-diff is the load-bearing pure piece and is hand-computed here."""

from __future__ import annotations

import torch

from bsc.probes import COMPLY_OPENERS, REFUSAL_OPENERS, RefusalReadout, refusal_diff_from_logits


def test_refusal_diff_handcomputed():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # refusal ids {0,1} -> 1+2=3 ; comply ids {3,4} -> 4+5=9 ; diff = -6
    assert refusal_diff_from_logits(logits, [0, 1], [3, 4]) == -6.0


def test_refusal_diff_positive_when_refusal_dominates():
    logits = torch.tensor([10.0, 0.0, 0.0, 1.0])
    assert refusal_diff_from_logits(logits, [0], [3]) == 9.0


def test_refusal_diff_sign_flips_with_framing():
    # Same content, two "framings" that move mass between the opener sets -> sign flips.
    refuse_like = torch.tensor([5.0, 0.0])
    comply_like = torch.tensor([0.0, 5.0])
    assert refusal_diff_from_logits(refuse_like, [0], [1]) > 0
    assert refusal_diff_from_logits(comply_like, [0], [1]) < 0


def test_readout_to_dict_shape():
    r = RefusalReadout(refusal_logit_diff=1.0, refusal_head_mass=0.5,
                       compliance_head_mass=-0.5, layer_refusal_diff=[0.1, 0.2], first_token="I")
    d = r.to_dict()
    assert d["refusal_logit_diff"] == 1.0
    assert d["layer_refusal_diff"] == [0.1, 0.2]
    assert d["first_token"] == "I"


def test_opener_sets_nonempty_and_disjoint_intent():
    assert REFUSAL_OPENERS and COMPLY_OPENERS
    # The two cue sets should not literally share a word (they'd cancel in the diff).
    assert not (set(REFUSAL_OPENERS) & set(COMPLY_OPENERS))
