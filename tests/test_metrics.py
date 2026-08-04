"""Metric tests against hand-computed values (CLAUDE.md §5)."""

from __future__ import annotations

import math

import pytest

from bsc.metrics import (
    bootstrap_rate,
    cohens_d,
    mcnemar_paired,
    roc_auc,
    sweep_threshold,
    wilson_interval,
)


class TestWilson:
    def test_point_estimate(self):
        est = wilson_interval(33, 100)
        assert est.value == pytest.approx(0.33)
        assert est.n == 100

    def test_interval_contains_estimate(self):
        est = wilson_interval(33, 100)
        assert est.ci_low < est.value < est.ci_high

    def test_zero_successes_stays_in_bounds(self):
        # The published 0% FPR claim. A Wald interval would go negative here; Wilson must not,
        # and must not collapse to a point — 0/100 is consistent with a true rate up to ~3.7%.
        est = wilson_interval(0, 100)
        assert est.value == 0.0
        assert est.ci_low == 0.0
        assert 0.0 < est.ci_high < 0.05

    def test_all_successes_stays_in_bounds(self):
        est = wilson_interval(100, 100)
        assert est.ci_high == pytest.approx(1.0)
        assert est.ci_high <= 1.0
        assert 0.95 < est.ci_low < 1.0

    def test_known_value(self):
        # Hand-computed: n=100, p=0.5, z=1.96 -> center 0.5, half-width 0.0980/1.0384
        est = wilson_interval(50, 100)
        assert est.ci_low == pytest.approx(0.4038, abs=1e-3)
        assert est.ci_high == pytest.approx(0.5962, abs=1e-3)

    def test_wider_interval_at_smaller_n(self):
        wide = wilson_interval(3, 10)
        narrow = wilson_interval(30, 100)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 0)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)


class TestBootstrap:
    def test_matches_mean(self):
        outcomes = [True] * 33 + [False] * 67
        est = bootstrap_rate(outcomes, n_samples=2000, seed=7)
        assert est.value == pytest.approx(0.33)
        assert est.ci_low < 0.33 < est.ci_high

    def test_deterministic_under_seed(self):
        outcomes = [True, False, True, True, False] * 10
        a = bootstrap_rate(outcomes, n_samples=1000, seed=42)
        b = bootstrap_rate(outcomes, n_samples=1000, seed=42)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_different_seeds_differ(self):
        # A balanced 50-sample is a bad probe here: bootstrap means are discrete multiples of
        # 0.02, so two seeds routinely land on identical quantiles. Use an unbalanced sample
        # with a finer grid, where genuinely different resampling must show up.
        outcomes = [True] * 37 + [False] * 163
        a = bootstrap_rate(outcomes, n_samples=1000, seed=1)
        b = bootstrap_rate(outcomes, n_samples=1000, seed=2)
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)
        assert a.value == b.value  # the point estimate is seed-independent

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_rate([])


class TestMcNemar:
    def test_perfect_defense_is_significant(self):
        baseline = [True] * 20
        treatment = [False] * 20
        result = mcnemar_paired(baseline, treatment, n_samples=1000)
        assert result.delta == pytest.approx(-1.0)
        assert result.n_discordant == 20
        assert result.significant()

    def test_no_change_is_not_significant(self):
        outcomes = [True, False] * 10
        result = mcnemar_paired(outcomes, outcomes, n_samples=1000)
        assert result.delta == pytest.approx(0.0)
        assert result.n_discordant == 0
        assert result.p_value == 1.0
        assert not result.significant()

    def test_exact_p_value_hand_computed(self):
        # 5 discordant pairs, all in one direction: two-sided exact p = 2 * (1/2)^5 = 0.0625.
        baseline = [True] * 5 + [False] * 15
        treatment = [False] * 5 + [False] * 15
        result = mcnemar_paired(baseline, treatment, n_samples=500)
        assert result.n_discordant == 5
        assert result.p_value == pytest.approx(0.0625, abs=1e-6)

    def test_symmetric_discordance_not_significant(self):
        baseline = [True] * 5 + [False] * 5
        treatment = [False] * 5 + [True] * 5
        result = mcnemar_paired(baseline, treatment, n_samples=500)
        assert result.n_discordant == 10
        assert result.p_value == pytest.approx(1.0)

    def test_unpaired_lengths_rejected(self):
        with pytest.raises(ValueError):
            mcnemar_paired([True, False], [True])


class TestCohensD:
    def test_hand_computed(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean 3, var 2.5
        b = [3.0, 4.0, 5.0, 6.0, 7.0]  # mean 5, var 2.5
        # pooled sd = sqrt(2.5) -> d = (3-5)/1.5811 = -1.2649
        assert cohens_d(a, b) == pytest.approx(-2 / math.sqrt(2.5), abs=1e-6)

    def test_zero_for_identical_groups(self):
        a = [1.0, 2.0, 3.0, 4.0]
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_zero_variance_rejected(self):
        with pytest.raises(ValueError):
            cohens_d([1.0, 1.0], [1.0, 1.0])

    def test_too_small_rejected(self):
        with pytest.raises(ValueError):
            cohens_d([1.0], [2.0, 3.0])


class TestROCAUC:
    def test_perfect_separation(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [False, False, True, True]
        assert roc_auc(scores, labels) == pytest.approx(1.0)

    def test_inverted_separation(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [False, False, True, True]
        assert roc_auc(scores, labels) == pytest.approx(0.0)

    def test_all_ties_is_chance(self):
        # The tie-handling case: identical scores must give 0.5, not 0.0. bf16-quantised
        # activation norms produce exact ties, so a naive implementation biases AUC downward.
        scores = [0.5] * 6
        labels = [True, True, True, False, False, False]
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_partial_tie(self):
        scores = [1.0, 2.0, 2.0, 3.0]
        labels = [False, False, True, True]
        # pos ranks {2.0 -> avg rank 2.5, 3.0 -> 4}; U = (2.5+4) - 2*3/2 = 3.5; AUC = 3.5/4
        assert roc_auc(scores, labels) == pytest.approx(0.875)

    def test_single_class_rejected(self):
        with pytest.raises(ValueError):
            roc_auc([0.1, 0.2], [True, True])


class TestThresholdSweep:
    def test_finds_separating_threshold(self):
        scores = [1.0, 1.5, 2.0, 8.0, 9.0, 10.0]
        labels = [False, False, False, True, True, True]
        sweep = sweep_threshold(scores, labels)
        tau, f1 = sweep.best_f1()
        assert f1 == pytest.approx(1.0)
        assert 2.0 < tau <= 8.0

    def test_recall_is_one_at_lowest_threshold(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [False, False, True, True]
        sweep = sweep_threshold(scores, labels)
        assert sweep.recall[0] == pytest.approx(1.0)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            sweep_threshold([1.0, 2.0], [True])
