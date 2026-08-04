"""Auditable numerical forms of the manuscript's conditional bounds."""

from __future__ import annotations

import math
from numbers import Integral, Real
import warnings

import numpy as np


def radial_contraction(
    vector: np.ndarray, *, radius: float, epsilon: float = 1e-10
) -> np.ndarray:
    """Continuous epsilon-regularized projection used by Dual-Bound FORCE.

    For fixed ``radius`` the radial map is nonexpansive in its vector
    argument.  For fixed ``vector`` it is one-Lipschitz in ``radius``.  The
    small positive denominator regularizes the origin without weakening
    either property.
    """

    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("vector must be a finite one-dimensional array")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    norm = float(np.linalg.norm(value))
    factor = min(1.0, float(radius) / (norm + float(epsilon)))
    return factor * value


def marginal_map_rms_bound(
    *,
    location_error: np.ndarray,
    scale_error: np.ndarray,
    centered_second_moments: np.ndarray,
    scale_floor: float,
    marginal_lambda: float,
) -> float:
    """L2 bound for two coordinatewise standardized clipping maps.

    ``centered_second_moments[j]`` is
    ``E[(X_j - mu_0j)^2]``.  Both fitted and ideal scales must be at least
    ``scale_floor``.  This is a deterministic transfer from supplied
    component errors; it is not a statistical quantile-concentration result.
    """

    location = np.asarray(location_error, dtype=float)
    scale = np.asarray(scale_error, dtype=float)
    moments = np.asarray(centered_second_moments, dtype=float)
    if location.ndim != 1 or scale.shape != location.shape or moments.shape != location.shape:
        raise ValueError("component-error arrays must be same-shaped vectors")
    if not (
        np.isfinite(location).all()
        and np.isfinite(scale).all()
        and np.isfinite(moments).all()
    ):
        raise ValueError("component-error arrays must be finite")
    if np.any(scale < 0) or np.any(moments < 0):
        raise ValueError("scale errors and second moments must be nonnegative")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    if not np.isfinite(marginal_lambda) or marginal_lambda < 0:
        raise ValueError("marginal_lambda must be finite and nonnegative")

    local = float(
        np.linalg.norm(location) / scale_floor
        + np.sqrt(np.sum(scale**2 * moments)) / scale_floor**2
    )
    global_cap = float(2.0 * marginal_lambda * np.sqrt(len(location)))
    return min(global_cap, local)


def calibration_transform_stability_bound(
    *,
    marginal_rms_error: float,
    projector_error: float,
    dimension: int,
    marginal_lambda: float,
    parallel_radius_error: float,
    residual_radius_error: float,
) -> float:
    """RMS difference bound between fitted and ideal dual-bound maps."""

    named = (
        marginal_rms_error,
        projector_error,
        parallel_radius_error,
        residual_radius_error,
    )
    if any(not np.isfinite(value) or value < 0 for value in named):
        raise ValueError("calibration component errors must be finite and nonnegative")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if not np.isfinite(marginal_lambda) or marginal_lambda < 0:
        raise ValueError("marginal_lambda must be finite and nonnegative")
    return float(
        2.0 * marginal_rms_error
        + 2.0 * marginal_lambda * np.sqrt(dimension) * projector_error
        + parallel_radius_error
        + residual_radius_error
    )


def calibration_scatter_stability_bound(
    *,
    fitted_radius_squared: float,
    ideal_radius_squared: float,
    transformation_rms_error: float,
) -> float:
    """Operator bound for the calibration-induced transformed-scatter gap."""

    values = (fitted_radius_squared, ideal_radius_squared, transformation_rms_error)
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("radii and transformation error must be finite and nonnegative")
    return float(
        (np.sqrt(fitted_radius_squared) + np.sqrt(ideal_radius_squared))
        * transformation_rms_error
    )


def conditional_operator_error_bound(
    *,
    calibration_bias: float,
    clipping_bias: float,
    sampling_deviation: float,
    contamination_fraction: float,
    parallel_radius: float,
    residual_radius: float,
    fd_tail_loss: float,
    dimension: int,
    marginal_lambda: float,
) -> dict[str, float]:
    """Additive conditional bound used in the manuscript theorem.

    The function deliberately requires the calibration and clipping biases as
    externally justified nonnegative terms.  It therefore cannot be mistaken
    for an unconditional distribution-free guarantee.
    """

    named = {
        "calibration_bias": calibration_bias,
        "clipping_bias": clipping_bias,
        "sampling_deviation": sampling_deviation,
        "fd_tail_loss": fd_tail_loss,
    }
    if any(not np.isfinite(value) or value < 0 for value in named.values()):
        raise ValueError("all supplied error terms must be finite and nonnegative")
    replacement = replacement_contamination_bound(
        contamination_fraction,
        parallel_radius,
        residual_radius,
        dimension=dimension,
        marginal_lambda=marginal_lambda,
    )
    components = {**{key: float(value) for key, value in named.items()}, "replacement_contamination": replacement}
    components["total"] = float(sum(components.values()))
    return components


def influence_radius_squared(parallel_radius: float, residual_radius: float) -> float:
    """Maximum squared norm after orthogonal component clipping."""
    if (
        not np.isfinite(parallel_radius)
        or not np.isfinite(residual_radius)
        or parallel_radius < 0
        or residual_radius < 0
    ):
        raise ValueError("radii must be nonnegative")
    return float(parallel_radius**2 + residual_radius**2)


def combined_influence_radius_squared(
    *,
    dimension: int,
    marginal_lambda: float,
    parallel_radius: float,
    residual_radius: float,
) -> float:
    """Sharp deterministic row-energy envelope for the two clipping stages.

    Coordinate clipping gives ``||y||^2 <= p * lambda_m^2``.  The orthogonal
    component contractions do not increase that norm and separately give
    ``||z||^2 <= tau_parallel^2 + tau_perp^2``.  Both bounds therefore hold
    simultaneously.  This helper audits the manuscript's quantity ``M^2``;
    it does not assert that fitted radial thresholds are dimension-free.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not np.isfinite(marginal_lambda) or marginal_lambda < 0:
        raise ValueError("marginal_lambda must be finite and nonnegative")
    marginal = float(dimension * marginal_lambda**2)
    dual = influence_radius_squared(parallel_radius, residual_radius)
    return min(marginal, dual)


def _retained_rank(
    *, retained_rank: int | None, sketch_rows: int | None
) -> int:
    """Resolve the old, ambiguous ``sketch_rows`` compatibility keyword."""

    if retained_rank is not None and sketch_rows is not None:
        raise ValueError("supply retained_rank only; sketch_rows is deprecated")
    if retained_rank is None:
        if sketch_rows is None:
            raise TypeError("retained_rank is required")
        warnings.warn(
            "sketch_rows is deprecated; pass retained_rank=k, not the physical 2k buffer",
            DeprecationWarning,
            stacklevel=3,
        )
        retained_rank = sketch_rows
    if (
        isinstance(retained_rank, (bool, np.bool_))
        or not isinstance(retained_rank, Integral)
        or retained_rank <= 0
    ):
        raise ValueError("retained_rank must be a positive integer")
    return int(retained_rank)


def explicit_fd_tail_envelope(
    *,
    radius_squared: float,
    retained_rank: int | None = None,
    comparison_rank: int,
    sketch_rows: int | None = None,
) -> float:
    """Data-independent envelope for the normalized FD tail term.

    If each of ``n`` input rows has squared norm at most ``M^2``, then
    ``||A-A_q||_F^2 / (n(k-q)) <= M^2 / (k-q)``.  The result is conservative:
    the data-dependent tail term can be much smaller.
    """

    retained = _retained_rank(retained_rank=retained_rank, sketch_rows=sketch_rows)
    if not np.isfinite(radius_squared) or radius_squared < 0:
        raise ValueError("radius_squared must be finite and nonnegative")
    if (
        isinstance(comparison_rank, (bool, np.bool_))
        or not isinstance(comparison_rank, Integral)
        or not 0 <= comparison_rank < retained
    ):
        raise ValueError("require integer 0 <= comparison_rank < retained_rank")
    return float(radius_squared / (retained - comparison_rank))


def replacement_contamination_bound(
    contamination_fraction: float,
    parallel_radius: float,
    residual_radius: float,
    *,
    dimension: int,
    marginal_lambda: float,
) -> float:
    """Operator-norm change from replacing an epsilon fraction of bounded rows.

    This manuscript-facing helper always uses the simultaneous marginal/radial
    radius ``M^2``.  Use :func:`radial_only_replacement_contamination_bound`
    when explicitly auditing the valid but looser ``L^2`` envelope.
    """
    if not 0.0 <= contamination_fraction <= 1.0:
        raise ValueError("contamination_fraction must lie in [0, 1]")
    radius_squared = combined_influence_radius_squared(
        dimension=dimension,
        marginal_lambda=marginal_lambda,
        parallel_radius=parallel_radius,
        residual_radius=residual_radius,
    )
    # If both outer products lie in [0, M^2 I], their difference lies in
    # [-M^2 I, M^2 I]. This sharpens the triangle-inequality constant 2.
    return float(contamination_fraction * radius_squared)


def radial_only_replacement_contamination_bound(
    contamination_fraction: float,
    parallel_radius: float,
    residual_radius: float,
) -> float:
    """Explicitly looser replacement envelope based only on ``L^2``."""

    if not 0.0 <= contamination_fraction <= 1.0:
        raise ValueError("contamination_fraction must lie in [0, 1]")
    return float(
        contamination_fraction
        * influence_radius_squared(parallel_radius, residual_radius)
    )


def bounded_matrix_bernstein_bound(
    *, radius_squared: float, n: int, dimension: int, delta: float
) -> float:
    """Conservative operator deviation for bounded rank-one observations.

    The expression is used as an auditable sufficient bound, not as an exact
    distribution-specific confidence interval.
    """
    if (
        isinstance(radius_squared, (bool, np.bool_))
        or not isinstance(radius_squared, Real)
        or not np.isfinite(radius_squared)
        or radius_squared < 0
    ):
        raise ValueError("radius_squared must be finite and nonnegative")
    for name, value in (("n", n), ("dimension", dimension)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    log_term = math.log(2.0 * dimension / delta)
    return float(
        radius_squared * math.sqrt(2.0 * log_term / n)
        + (2.0 * radius_squared * log_term) / (3.0 * n)
    )


def fd_tail_bound(
    rows: np.ndarray,
    *,
    retained_rank: int | None = None,
    comparison_rank: int,
    sketch_rows: int | None = None,
) -> float:
    """Classical FD tail term divided by sample size."""
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("rows must be a nonempty finite matrix")
    retained = _retained_rank(retained_rank=retained_rank, sketch_rows=sketch_rows)
    if (
        isinstance(comparison_rank, (bool, np.bool_))
        or not isinstance(comparison_rank, Integral)
        or not 0 <= comparison_rank < retained
    ):
        raise ValueError("require integer 0 <= comparison_rank < retained_rank")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tail = float(np.sum(singular_values[comparison_rank:] ** 2))
    return tail / ((retained - comparison_rank) * len(matrix))


def correlation_normalization_bound(
    scatter_error: float,
    minimum_diagonal: float,
    target_operator_norm: float,
) -> float:
    """Operator envelope for diagonal correlation normalization.

    Unlike a dimension-free shortcut, this expression explicitly carries the
    target scatter operator norm.  The target and estimate diagonals are then
    bounded below by ``d_min`` and ``d_min/2``, respectively.
    """
    if scatter_error < 0 or minimum_diagonal <= 0 or target_operator_norm < 0:
        raise ValueError("errors/norms must be nonnegative and diagonal positive")
    if scatter_error > minimum_diagonal / 2:
        raise ValueError("normalization lemma requires error <= d_min/2")
    return float(
        2.0 * scatter_error / minimum_diagonal
        + (2.0 + math.sqrt(2.0))
        * target_operator_norm
        * scatter_error
        / minimum_diagonal**2
    )


def davis_kahan_bound(operator_error: float, eigengap: float) -> float:
    """Conservative sin-theta envelope, clipped to its natural maximum."""
    if operator_error < 0 or eigengap <= 0:
        raise ValueError("operator_error must be nonnegative and eigengap positive")
    return float(min(1.0, 2.0 * operator_error / eigengap))
