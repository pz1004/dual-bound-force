"""Audited internal Noisy Outlier Pursuit comparator.

This module is deliberately not exported by :mod:`dual_bound_force`.  It is a
batch, oracle-assisted structural-recovery ceiling for the secondary study,
not a streaming estimator or a component of Dual-Bound FORCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OutlierPursuitResult:
    low_rank: np.ndarray
    column_sparse: np.ndarray
    noise: np.ndarray
    converged: bool
    iterations: int
    objective: float
    primal_residual: float
    iterate_residual: float
    constraint_violation: float
    working_state_bytes: int
    diagnostics: dict[str, Any]


def _finite_real(name: str, value: Real, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _singular_value_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    retained = np.maximum(values - threshold, 0.0)
    return (left * retained) @ right


def _column_l2_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    factors = np.zeros_like(norms)
    positive = norms > 0.0
    factors[positive] = np.maximum(0.0, 1.0 - threshold / norms[positive])
    return matrix * factors


def _project_frobenius_ball(matrix: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= radius or norm == 0.0:
        return matrix.copy()
    return matrix * (radius / norm)


def noisy_outlier_pursuit(
    matrix: np.ndarray,
    *,
    regularization: float,
    noise_budget: float,
    tolerance: float = 1e-6,
    max_iterations: int = 2_000,
) -> OutlierPursuitResult:
    """Solve the Noisy Outlier Pursuit program with deterministic ADMM.

    Rows are features and columns are observations.  The implementation uses
    singular-value thresholding for the nuclear norm, columnwise Euclidean
    shrinkage for the ``l1,2`` norm, and an explicit projection of the noise
    variable onto the declared Frobenius ball.  A nonconverged result is
    returned with ``converged=False`` so callers can emit a strict failure
    rather than silently consuming a partial solution.
    """

    observed = np.asarray(matrix, dtype=float)
    if observed.ndim != 2 or not observed.size or not np.isfinite(observed).all():
        raise ValueError("matrix must be a nonempty finite two-dimensional array")
    regularization = _finite_real("regularization", regularization, positive=True)
    noise_budget = _finite_real("noise_budget", noise_budget, positive=False)
    tolerance = _finite_real("tolerance", tolerance, positive=True)
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, Integral)
        or int(max_iterations) <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)

    low_rank = np.zeros_like(observed)
    sparse = np.zeros_like(observed)
    noise = np.zeros_like(observed)
    scaled_dual = np.zeros_like(observed)
    observed_norm = max(float(np.linalg.norm(observed, ord="fro")), np.finfo(float).tiny)
    spectral_norm = max(float(np.linalg.norm(observed, ord=2)), np.finfo(float).tiny)
    penalty = 1.25 / spectral_norm
    primal_residual = float("inf")
    iterate_residual = float("inf")
    constraint_violation = float("inf")

    for iteration in range(1, max_iterations + 1):
        previous_low_rank = low_rank
        previous_sparse = sparse
        previous_noise = noise

        low_rank = _singular_value_threshold(
            observed - sparse - noise + scaled_dual,
            1.0 / penalty,
        )
        sparse = _column_l2_threshold(
            observed - low_rank - noise + scaled_dual,
            regularization / penalty,
        )
        noise = _project_frobenius_ball(
            observed - low_rank - sparse + scaled_dual,
            noise_budget,
        )
        residual = observed - low_rank - sparse - noise
        scaled_dual = scaled_dual + residual

        primal_residual = float(np.linalg.norm(residual, ord="fro") / observed_norm)
        iterate_residual = float(
            max(
                np.linalg.norm(low_rank - previous_low_rank, ord="fro"),
                np.linalg.norm(sparse - previous_sparse, ord="fro"),
                np.linalg.norm(noise - previous_noise, ord="fro"),
            )
            / observed_norm
        )
        reconstruction_without_noise = observed - low_rank - sparse
        constraint_violation = float(
            max(0.0, np.linalg.norm(reconstruction_without_noise, ord="fro") - noise_budget)
            / observed_norm
        )
        if (
            primal_residual <= tolerance
            and iterate_residual <= tolerance
            and constraint_violation <= tolerance
        ):
            converged = True
            break

        # Residual balancing changes only the augmented-Lagrangian scale; the
        # scaled dual is adjusted inversely so its unscaled value is preserved.
        if primal_residual > 10.0 * iterate_residual:
            penalty *= 2.0
            scaled_dual /= 2.0
        elif iterate_residual > 10.0 * primal_residual:
            penalty /= 2.0
            scaled_dual *= 2.0
    else:
        iteration = max_iterations
        converged = False

    singular_values = np.linalg.svd(low_rank, compute_uv=False)
    objective = float(
        np.sum(singular_values)
        + regularization * np.sum(np.linalg.norm(sparse, axis=0))
    )
    # Six full matrices are simultaneously resident in the main iteration;
    # SVD workspaces and allocator/RSS overhead are deliberately excluded.
    working_state_bytes = int(6 * observed.nbytes)
    return OutlierPursuitResult(
        low_rank=low_rank,
        column_sparse=sparse,
        noise=noise,
        converged=converged,
        iterations=int(iteration),
        objective=objective,
        primal_residual=primal_residual,
        iterate_residual=iterate_residual,
        constraint_violation=constraint_violation,
        working_state_bytes=working_state_bytes,
        diagnostics={
            "regularization": regularization,
            "noise_budget": noise_budget,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
            "final_penalty": float(penalty),
            "numerical_rank": int(np.sum(singular_values > tolerance * max(singular_values[0] if len(singular_values) else 0.0, 1.0))),
            "qualification": "oracle-assisted batch structural ceiling",
        },
    )


def outlier_support(column_sparse: np.ndarray, *, relative_threshold: float = 1e-6) -> np.ndarray:
    """Return the deterministic nonzero-column support used by the study."""

    matrix = np.asarray(column_sparse, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("column_sparse must be a finite matrix")
    relative_threshold = _finite_real(
        "relative_threshold", relative_threshold, positive=True
    )
    norms = np.linalg.norm(matrix, axis=0)
    scale = float(np.max(norms)) if len(norms) else 0.0
    if scale == 0.0:
        return np.zeros(matrix.shape[1], dtype=bool)
    return norms > relative_threshold * scale
