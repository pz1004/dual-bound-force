"""Metrics and paired inferential procedures used by every study tier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
from scipy import stats


def normalized_frobenius_error(estimate: np.ndarray, target: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    target = np.asarray(target, dtype=float)
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate and target must be same-shaped matrices")
    if not np.isfinite(estimate).all() or not np.isfinite(target).all():
        raise ValueError("matrices must be finite")
    denominator = max(float(np.linalg.norm(target, ord="fro")), 1e-12)
    return float(np.linalg.norm(estimate - target, ord="fro") / denominator)


def normalized_projection_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("subspace bases must be matrices")
    if reference.shape != estimate.shape or reference.shape[1] == 0:
        raise ValueError("subspace bases must have equal positive rank")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("subspace bases must be finite")
    q_reference, _ = np.linalg.qr(reference, mode="reduced")
    q_estimate, _ = np.linalg.qr(estimate, mode="reduced")
    rank = reference.shape[1]
    return float(
        np.linalg.norm(
            q_reference @ q_reference.T - q_estimate @ q_estimate.T,
            ord="fro",
        )
        / np.sqrt(2.0 * rank)
    )


def principal_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    if not 0 < rank <= matrix.shape[0]:
        raise ValueError("rank must lie in [1, p]")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return eigenvectors[:, np.argsort(eigenvalues)[-int(rank) :]]


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    diagonal = np.maximum(np.diag(covariance), 0.0)
    positive = diagonal > 0.0
    result = np.zeros_like(covariance)
    indices = np.flatnonzero(positive)
    if len(indices):
        denominator = np.sqrt(np.outer(diagonal[indices], diagonal[indices]))
        result[np.ix_(indices, indices)] = np.clip(
            covariance[np.ix_(indices, indices)] / denominator,
            -1.0,
            1.0,
        )
        result[indices, indices] = 1.0
    return result


def rank_auc(nominal_scores: Sequence[float], anomalous_scores: Sequence[float]) -> float:
    """Dependency-free Mann-Whitney rank AUC with half credit for ties."""

    nominal = np.asarray(nominal_scores, dtype=float)
    anomalous = np.asarray(anomalous_scores, dtype=float)
    if nominal.ndim != 1 or anomalous.ndim != 1 or not len(nominal) or not len(anomalous):
        raise ValueError("score arrays must be nonempty vectors")
    if not np.isfinite(nominal).all() or not np.isfinite(anomalous).all():
        raise ValueError("scores must be finite")
    combined = np.concatenate([nominal, anomalous])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    start = 0
    while start < len(combined):
        end = start + 1
        while end < len(combined) and combined[order[end]] == combined[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    anomalous_ranks = ranks[len(nominal) :]
    statistic = float(
        np.sum(anomalous_ranks) - len(anomalous) * (len(anomalous) + 1) / 2.0
    )
    return statistic / (len(nominal) * len(anomalous))


def paired_bootstrap_ci(
    differences: Sequence[float], *, seed: int, resamples: int = 10_000
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired differences must be finite and nonempty")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)):
        raise TypeError("resamples must be an integer")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(resamples), len(values)))
    means = np.mean(values[indices], axis=1)
    probability_nonpositive = float(np.mean(means <= 0.0))
    probability_nonnegative = float(np.mean(means >= 0.0))
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "bootstrap_two_sided_p": float(
            min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        ),
        "n_pairs": int(len(values)),
        "resamples": int(resamples),
    }


def paired_mean_inference(
    differences: Sequence[float],
    *,
    alternative: str = "less",
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 10_000,
) -> dict[str, float | int | str | bool | None | list[float]]:
    """Inference for a paired mean with a directional alternative.

    The paired t test is the inferential procedure because the study's
    estimand is a mean paired difference.  The percentile-bootstrap interval
    is retained as a descriptive sensitivity result; its observation-centered
    tail area is deliberately not used as a hypothesis-test p-value.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired differences must contain at least two finite values")
    if alternative not in {"less", "greater"}:
        raise ValueError("alternative must be 'less' or 'greater'")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    estimate = float(np.mean(values))
    sample_sd = float(np.std(values, ddof=1))
    n_pairs = int(len(values))
    degrees_of_freedom = n_pairs - 1
    standard_error = sample_sd / np.sqrt(n_pairs)
    if standard_error == 0.0:
        t_statistic: float | None = None
        if estimate == 0.0:
            one_sided_p = 0.5
        elif (alternative == "less" and estimate < 0.0) or (
            alternative == "greater" and estimate > 0.0
        ):
            one_sided_p = 0.0
        else:
            one_sided_p = 1.0
        paired_t_95 = [estimate, estimate]
        degenerate = True
    else:
        t_statistic = estimate / standard_error
        one_sided_p = float(
            stats.t.cdf(t_statistic, degrees_of_freedom)
            if alternative == "less"
            else stats.t.sf(t_statistic, degrees_of_freedom)
        )
        critical = float(
            stats.t.ppf(0.5 + confidence / 2.0, degrees_of_freedom)
        )
        paired_t_95 = [
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ]
        degenerate = False

    bootstrap = paired_bootstrap_ci(
        values,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    return {
        "estimate": estimate,
        "sample_standard_deviation": sample_sd,
        "paired_mean_standard_error": float(standard_error),
        # Historical strict bundles used this inaccurate field name.  Retain
        # it as a value-identical compatibility alias while all new reporting
        # and manuscript text use ``paired_mean_standard_error``.
        "monte_carlo_standard_error": float(standard_error),
        "degrees_of_freedom": degrees_of_freedom,
        "t_statistic": t_statistic,
        "one_sided_p": one_sided_p,
        "alternative": alternative,
        "paired_t_95": [float(value) for value in paired_t_95],
        "percentile_bootstrap_95": [
            float(bootstrap["lower_95"]),
            float(bootstrap["upper_95"]),
        ],
        "percentile_bootstrap_resamples": int(bootstrap_resamples),
        "n_pairs": n_pairs,
        "degenerate_standard_error": degenerate,
    }


def exact_one_sided_sign_p(
    differences: Sequence[float], *, alternative: str = "less"
) -> float:
    """Distribution-free one-sided sign probability, excluding exact ties."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired differences must be a nonempty finite vector")
    if alternative not in {"less", "greater"}:
        raise ValueError("alternative must be 'less' or 'greater'")
    nonzero = values[values != 0.0]
    if not len(nonzero):
        return 1.0
    favorable = int(
        np.sum(nonzero < 0.0) if alternative == "less" else np.sum(nonzero > 0.0)
    )
    return float(stats.binom.sf(favorable - 1, len(nonzero), 0.5))


def exact_one_sided_sign_flip_p(
    differences: Sequence[float], *, alternative: str = "less"
) -> float:
    """Exact sign-flip sensitivity test for at most 20 paired differences."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired differences must be a nonempty finite vector")
    if len(values) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 pairs")
    if alternative not in {"less", "greater"}:
        raise ValueError("alternative must be 'less' or 'greater'")
    observed = float(np.mean(values))
    outcomes = np.fromiter(
        (
            np.mean(values * np.asarray(signs, dtype=float))
            for signs in product((-1.0, 1.0), repeat=len(values))
        ),
        dtype=float,
        count=2 ** len(values),
    )
    if alternative == "less":
        return float(np.mean(outcomes <= observed + 1e-15))
    return float(np.mean(outcomes >= observed - 1e-15))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, name in enumerate(ordered):
        running = max(running, (count - index) * float(p_values[name]))
        adjusted[name] = min(1.0, running)
    return adjusted
