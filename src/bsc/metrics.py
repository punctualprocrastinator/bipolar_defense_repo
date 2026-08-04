"""Statistics with uncertainty.

CLAUDE.md §1.5: a rate reported without ``n`` and a confidence interval is not a result. The
current draft reports "66% -> 33%" on 100 prompts with no interval; at n=100 those have roughly
±9pp of binomial noise each, which reviewers will ask about.

Everything here is pure and seeded — no torch, no model, unit-testable against hand computation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RateEstimate:
    """A proportion with a confidence interval and the sample size that produced it."""

    value: float
    ci_low: float
    ci_high: float
    n: int
    n_success: int
    method: str
    confidence: float = 0.95

    def __str__(self) -> str:
        return (
            f"{self.value:.1%} [{self.ci_low:.1%}, {self.ci_high:.1%}] "
            f"(n={self.n}, {self.method})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_success": self.n_success,
            "method": self.method,
            "confidence": self.confidence,
        }


def wilson_interval(n_success: int, n: int, confidence: float = 0.95) -> RateEstimate:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation for the rates in this project because ASR and FPR
    land near 0 or 1, where the Wald interval produces impossible bounds (an FPR of 0/100 gets
    a symmetric interval straddling zero). Wilson stays inside [0, 1] and behaves at the edges.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= n_success <= n:
        raise ValueError(f"n_success={n_success} outside [0, {n}]")

    z = _z_score(confidence)
    p = n_success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return RateEstimate(
        value=p,
        ci_low=max(0.0, center - half),
        ci_high=min(1.0, center + half),
        n=n,
        n_success=n_success,
        method="wilson",
        confidence=confidence,
    )


def bootstrap_rate(
    outcomes: Sequence[bool | int],
    *,
    n_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> RateEstimate:
    """Percentile bootstrap CI for a rate.

    Use when outcomes are not exchangeable Bernoulli draws — e.g. a per-prompt ASR aggregated
    over several sampled trials per prompt, where resampling *prompts* is the correct unit.
    For plain independent binary outcomes, :func:`wilson_interval` is exact enough and cheaper.
    """
    arr = np.asarray(outcomes, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return RateEstimate(
        value=float(arr.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n=int(arr.size),
        n_success=int(arr.sum()),
        method=f"bootstrap({n_samples})",
        confidence=confidence,
    )


@dataclass(frozen=True)
class ComparisonResult:
    """Paired comparison of two conditions on the same items."""

    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_discordant: int
    test: str

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n": self.n,
            "n_discordant": self.n_discordant,
            "test": self.test,
        }


def mcnemar_paired(
    baseline: Sequence[bool],
    treatment: Sequence[bool],
    *,
    n_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> ComparisonResult:
    """Exact McNemar test for a paired binary comparison, plus a bootstrap CI on the difference.

    This is the correct test for "does the defense reduce ASR", because baseline and defended
    conditions are run on the **same prompts with the same seeds** (CLAUDE.md §2.4). Treating
    them as two independent samples throws away the pairing and loses power.

    The p-value is the exact binomial test on discordant pairs; the CI comes from bootstrapping
    the paired difference, which is what you actually want to quote as the effect size.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"unpaired lengths: {len(baseline)} vs {len(treatment)}")
    if not baseline:
        raise ValueError("empty comparison")

    b = np.asarray(baseline, dtype=bool)
    t = np.asarray(treatment, dtype=bool)

    # b01: baseline success, treatment failure. b10: the reverse.
    b01 = int(np.sum(b & ~t))
    b10 = int(np.sum(~b & t))
    discordant = b01 + b10

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test at p=0.5 on the discordant pairs.
        k = min(b01, b10)
        tail = sum(math.comb(discordant, i) for i in range(k + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)

    rng = np.random.default_rng(seed)
    diff = t.astype(float) - b.astype(float)
    idx = rng.integers(0, diff.size, size=(n_samples, diff.size))
    means = diff[idx].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])

    return ComparisonResult(
        delta=float(diff.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p_value=float(p_value),
        n=int(b.size),
        n_discordant=discordant,
        test="mcnemar_exact+bootstrap_ci",
    )


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Cohen's d with the pooled standard deviation.

    The existing claim of d=2.64 for compliance-head norm as a harm classifier must be
    recomputed with this, since the legacy code's pooling convention was not recorded.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each group needs at least 2 observations")
    # ddof=1: sample standard deviation. Using the population SD inflates d on small samples.
    pooled_var = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (
        a.size + b.size - 2
    )
    if pooled_var == 0:
        raise ValueError("zero pooled variance; d is undefined")
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var))


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUC via the Mann-Whitney U identity, with correct handling of tied scores.

    Ties matter here: activation-norm scores quantised by bf16 produce exact ties, and the
    naive "count pairs where score_pos > score_neg" formula scores those as 0 rather than 0.5,
    biasing AUC downward.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.size != y.size:
        raise ValueError("scores and labels must be the same length")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both positive and negative examples")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # Average ranks within tied groups.
    sorted_scores = s[order]
    start = 0
    for i in range(1, s.size + 1):
        if i == s.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class ThresholdSweep:
    """Operating points for a scalar detector, for choosing tau on dev and reporting on test."""

    thresholds: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    fpr: list[float] = field(default_factory=list)

    def best_f1(self) -> tuple[float, float]:
        """(threshold, f1) at the max-F1 operating point."""
        if not self.f1:
            raise ValueError("empty sweep")
        i = int(np.argmax(self.f1))
        return self.thresholds[i], self.f1[i]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
        }


def sweep_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_points: int = 200,
) -> ThresholdSweep:
    """Sweep a detection threshold over the observed score range.

    Candidate thresholds are drawn from the observed scores rather than a uniform grid, so the
    sweep cannot miss the optimum by landing between two clustered values.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.size != y.size:
        raise ValueError("scores and labels must be the same length")
    if s.size == 0:
        raise ValueError("empty sweep input")

    candidates = np.unique(s)
    if candidates.size > n_points:
        candidates = np.quantile(candidates, np.linspace(0, 1, n_points))

    sweep = ThresholdSweep()
    n_pos = int(y.sum())
    for tau in candidates:
        predicted = s >= tau
        tp = int(np.sum(predicted & y))
        fp = int(np.sum(predicted & ~y))
        tn = int(np.sum(~predicted & ~y))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_pos if n_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.thresholds.append(float(tau))
        sweep.precision.append(precision)
        sweep.recall.append(recall)
        sweep.f1.append(f1)
        sweep.fpr.append(fp / (fp + tn) if fp + tn else 0.0)
    return sweep


def _z_score(confidence: float) -> float:
    """Two-sided normal critical value. Table lookup avoids a scipy dependency."""
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    key = round(confidence, 2)
    if key not in table:
        raise ValueError(f"unsupported confidence {confidence}; supported: {sorted(table)}")
    return table[key]
