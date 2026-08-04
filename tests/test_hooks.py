"""Intervention correctness tests.

The identity no-op test (CLAUDE.md §5) is the single most valuable test in this repo. An
off-by-one in head slicing does not crash — it produces a plausible-looking effect that gets
written up as a result. Asserting bit-exact equality under an identity intervention is what
catches it.
"""

from __future__ import annotations

import pytest
import torch

from bsc.hooks import (
    Head,
    HeadEdit,
    HeadIntervention,
    applied,
    build_bipolar_edits,
)


class TestHead:
    def test_parse_paper_notation(self):
        assert Head.parse("L25-H1") == Head(25, 1)
        assert Head.parse("l25-h1") == Head(25, 1)

    def test_str_roundtrip(self):
        assert Head.parse(str(Head(19, 6))) == Head(19, 6)

    def test_from_node(self):
        assert Head.from_node({"layer": 23, "head": 11, "score": 0.74}) == Head(23, 11)

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            Head.parse("layer25head1")

    def test_orderable_and_hashable(self):
        assert Head(1, 0) < Head(2, 0)
        assert len({Head(1, 0), Head(1, 0)}) == 1


class TestIdentityIsNoOp:
    """An identity intervention must not perturb a single bit."""

    def test_multiplier_one_is_exact_noop(self, tiny_bundle, tiny_input):
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()

        edits = {Head(2, h): HeadEdit(scale=1.0) for h in range(8)}
        intervention = HeadIntervention(tiny_bundle, edits, positions="all")
        with applied(intervention), torch.no_grad():
            hooked = tiny_bundle.model(tiny_input)

        assert torch.equal(reference, hooked), "identity intervention changed the output"

    def test_identity_across_all_layers_and_positions(self, tiny_bundle, tiny_input):
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()

        edits = {Head(l, h): HeadEdit() for l in range(4) for h in range(8)}
        for positions in ("last", "all"):
            intervention = HeadIntervention(tiny_bundle, edits, positions=positions)
            with applied(intervention), torch.no_grad():
                hooked = tiny_bundle.model(tiny_input)
            assert torch.equal(reference, hooked), f"positions={positions} was not a no-op"

    def test_empty_intervention_is_noop(self, tiny_bundle, tiny_input):
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()
        with applied(HeadIntervention(tiny_bundle, {})), torch.no_grad():
            hooked = tiny_bundle.model(tiny_input)
        assert torch.equal(reference, hooked)

    def test_zero_alpha_steering_is_noop(self, tiny_bundle, tiny_input):
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()
        edits = {Head(1, 3): HeadEdit(alpha=0.0, vector=torch.randn(4))}
        with applied(HeadIntervention(tiny_bundle, edits)), torch.no_grad():
            hooked = tiny_bundle.model(tiny_input)
        assert torch.equal(reference, hooked)


class TestHeadSlicingIsolation:
    """Editing head i must affect head i and nothing else."""

    def test_edit_affects_only_target_head(self, tiny_bundle, tiny_input):
        geometry = tiny_bundle.geometry
        captured: dict[str, torch.Tensor] = {}

        def capture(module, args):
            captured["x"] = args[0].detach().clone()
            return None

        handle = tiny_bundle.o_proj(2).register_forward_pre_hook(capture)
        with torch.no_grad():
            tiny_bundle.model(tiny_input)
        handle.remove()
        clean = captured["x"]

        target = 3
        edits = {Head(2, target): HeadEdit(scale=5.0)}
        intervention = HeadIntervention(tiny_bundle, edits, positions="all")

        captured.clear()

        def capture_after(module, args):
            captured.setdefault("seen", []).append(args[0].detach().clone())
            return None

        intervention.register()
        handle = tiny_bundle.o_proj(2).register_forward_pre_hook(capture_after)
        with torch.no_grad():
            tiny_bundle.model(tiny_input)
        handle.remove()
        intervention.remove()
        edited = captured["seen"][0]

        for h in range(geometry.num_attention_heads):
            cols = geometry.head_slice(h)
            if h == target:
                assert torch.allclose(edited[..., cols], clean[..., cols] * 5.0, atol=1e-6)
            else:
                assert torch.equal(edited[..., cols], clean[..., cols]), (
                    f"editing head {target} leaked into head {h}"
                )

    def test_zero_ablation_zeroes_exactly_that_head(self, tiny_bundle, tiny_input):
        geometry = tiny_bundle.geometry
        seen: list[torch.Tensor] = []

        edits = {Head(1, 5): HeadEdit(scale=0.0)}
        intervention = HeadIntervention(tiny_bundle, edits, positions="all")
        intervention.register()
        handle = tiny_bundle.o_proj(1).register_forward_pre_hook(
            lambda m, a: seen.append(a[0].detach().clone())
        )
        with torch.no_grad():
            tiny_bundle.model(tiny_input)
        handle.remove()
        intervention.remove()

        cols = geometry.head_slice(5)
        assert torch.all(seen[0][..., cols] == 0.0)
        assert not torch.all(seen[0] == 0.0), "ablation zeroed the whole tensor"

    def test_nonidentity_actually_changes_output(self, tiny_bundle, tiny_input):
        """Guards against a hook that silently fails to apply — which would make every
        intervention look like a null result."""
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()
        edits = {Head(2, 3): HeadEdit(scale=3.0)}
        with applied(HeadIntervention(tiny_bundle, edits, positions="all")), torch.no_grad():
            hooked = tiny_bundle.model(tiny_input)
        assert not torch.equal(reference, hooked)


class TestPositions:
    def test_last_only_leaves_earlier_positions_untouched(self, tiny_bundle, tiny_input):
        seen: list[torch.Tensor] = []
        edits = {Head(0, 2): HeadEdit(scale=0.0)}
        intervention = HeadIntervention(tiny_bundle, edits, positions="last")
        intervention.register()
        handle = tiny_bundle.o_proj(0).register_forward_pre_hook(
            lambda m, a: seen.append(a[0].detach().clone())
        )
        with torch.no_grad():
            tiny_bundle.model(tiny_input)
        handle.remove()
        intervention.remove()

        cols = tiny_bundle.geometry.head_slice(2)
        assert torch.all(seen[0][:, -1, cols] == 0.0)
        assert not torch.all(seen[0][:, :-1, cols] == 0.0)

    def test_single_position_makes_last_and_all_equivalent(self, tiny_bundle):
        """During KV-cached decode only one position exists, so the two modes coincide.
        This is exactly why `positions` is an explicit config field (CLAUDE.md §3.3)."""
        single = torch.randn(1, 1, 32)
        edits = {Head(1, 4): HeadEdit(scale=2.0)}
        outputs = []
        for positions in ("last", "all"):
            with applied(
                HeadIntervention(tiny_bundle, edits, positions=positions)
            ), torch.no_grad():
                outputs.append(tiny_bundle.model(single).clone())
        assert torch.equal(outputs[0], outputs[1])


class TestValidation:
    def test_rejects_out_of_range_layer(self, tiny_bundle):
        with pytest.raises(ValueError, match="layer out of range"):
            HeadIntervention(tiny_bundle, {Head(99, 0): HeadEdit(scale=2.0)})

    def test_rejects_out_of_range_head(self, tiny_bundle):
        with pytest.raises(ValueError, match="head out of range"):
            HeadIntervention(tiny_bundle, {Head(0, 99): HeadEdit(scale=2.0)})

    def test_head_slice_bounds_checked(self, tiny_bundle):
        with pytest.raises(ValueError):
            tiny_bundle.geometry.head_slice(8)

    def test_double_register_rejected(self, tiny_bundle):
        intervention = HeadIntervention(tiny_bundle, {Head(0, 0): HeadEdit(scale=2.0)})
        intervention.register()
        try:
            with pytest.raises(RuntimeError, match="already registered"):
                intervention.register()
        finally:
            intervention.remove()


class TestHookCleanup:
    def test_hooks_removed_on_exception(self, tiny_bundle, tiny_input):
        """A leaked hook contaminates every later run in the process — in a sweep, that means
        the baseline silently stops being a baseline."""
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()

        intervention = HeadIntervention(tiny_bundle, {Head(2, 1): HeadEdit(scale=9.0)})
        with pytest.raises(RuntimeError, match="boom"), applied(intervention):
            raise RuntimeError("boom")

        with torch.no_grad():
            after = tiny_bundle.model(tiny_input)
        assert torch.equal(reference, after), "hook leaked after an exception"

    def test_no_leak_after_normal_exit(self, tiny_bundle, tiny_input):
        with torch.no_grad():
            reference = tiny_bundle.model(tiny_input).clone()
        with applied(HeadIntervention(tiny_bundle, {Head(1, 1): HeadEdit(scale=4.0)})):
            pass
        with torch.no_grad():
            assert torch.equal(reference, tiny_bundle.model(tiny_input))


class TestGQAGeometry:
    """Grouped-query attention is the indexing trap documented in CLAUDE.md §3.1."""

    def test_geometry_reports_gqa(self, tiny_bundle):
        assert tiny_bundle.geometry.is_gqa
        assert tiny_bundle.geometry.kv_group_size == 4

    def test_head_slices_tile_the_hidden_dim(self, tiny_bundle):
        geometry = tiny_bundle.geometry
        covered = set()
        for h in range(geometry.num_attention_heads):
            s = geometry.head_slice(h)
            span = set(range(s.start, s.stop))
            assert not (covered & span), f"head {h} overlaps a previous head"
            covered |= span
        assert covered == set(range(geometry.hidden_size)), "head slices do not tile hidden_size"

    def test_kv_head_mapping(self, tiny_bundle):
        geometry = tiny_bundle.geometry
        # 8 query heads over 2 KV heads: query heads 0-3 -> kv 0, 4-7 -> kv 1.
        assert [geometry.kv_head_for(h) for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_query_head_count_drives_slicing_not_kv_count(self, tiny_bundle):
        """o_proj's input is indexed by query heads. Using num_key_value_heads here would
        slice 4x too wide and silently edit three neighbouring heads."""
        geometry = tiny_bundle.geometry
        assert geometry.head_dim == geometry.hidden_size // geometry.num_attention_heads
        assert geometry.head_dim != geometry.hidden_size // geometry.num_key_value_heads


class TestRecording:
    def test_records_pre_and_post_norms(self, tiny_bundle, tiny_input):
        edits = {Head(2, 1): HeadEdit(scale=3.0)}
        intervention = HeadIntervention(tiny_bundle, edits, positions="all", record=True)
        with applied(intervention), torch.no_grad():
            tiny_bundle.model(tiny_input)

        summary = intervention.record.summary()
        assert "L2-H1" in summary
        entry = summary["L2-H1"]
        # The measured post-norm must be the amplified value actually fed to o_proj. The legacy
        # code measured pre-hook and generated post-hook, disagreeing by exactly this factor.
        assert entry["post_mean"] == pytest.approx(3.0 * entry["pre_mean"], rel=1e-4)


class TestBuildBipolarEdits:
    def test_multiplicative_shape(self):
        refusal = [Head(25, 1), Head(25, 3)]
        compliance = [Head(25, 4)]
        edits = build_bipolar_edits(refusal, compliance, refusal_multiplier=3.0)
        assert edits[Head(25, 1)].scale == 3.0
        assert edits[Head(25, 4)].scale == 0.0
        assert len(edits) == 3

    def test_overlap_rejected(self):
        with pytest.raises(ValueError, match="both refusal and compliance"):
            build_bipolar_edits([Head(25, 1)], [Head(25, 1)])

    def test_identity_parameters_produce_identity_edits(self):
        edits = build_bipolar_edits(
            [Head(25, 1)], [Head(25, 4)], refusal_multiplier=1.0, ablate_compliance=False
        )
        assert all(e.is_identity for e in edits.values())

    def test_additive_requires_vectors(self):
        with pytest.raises(ValueError, match="requires steering_vectors"):
            build_bipolar_edits([Head(25, 1)], [], mode="additive_steering")

    def test_additive_missing_vector_is_an_error(self):
        with pytest.raises(KeyError):
            build_bipolar_edits(
                [Head(25, 1)],
                [],
                mode="additive_steering",
                steering_vectors={"L20-H3": torch.randn(4)},
            )

    def test_partial_compliance_ablation(self):
        edits = build_bipolar_edits([], [Head(19, 6)], compliance_scale=0.5)
        assert edits[Head(19, 6)].scale == 0.5

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown intervention mode"):
            build_bipolar_edits([], [], mode="telepathy")
