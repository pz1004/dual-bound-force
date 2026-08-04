"""Target-aware adapters around the estimator and separately installed prior-work packages."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from dual_bound_force import DualBoundFORCE
from force import ForceEstimator
from src import DriftMADForce
from sketch_force.research_baselines import (
    ExactMADFrequentDirections,
    FrequentDirectionsCovariance,
    RobustFrequentDirectionsCovariance,
)
from sketch_force import SketchFORCE

from .metrics import covariance_to_correlation, principal_basis


DENSE_METHODS = ("pearson", "force", "mad_force")
BOUNDED_METHODS = (
    "vanilla_fd",
    "rfd",
    "sketch_iqr",
    "sketch_mad",
    "dual_bound",
)


@dataclass
class FitResult:
    method: str
    correlation: np.ndarray
    basis: np.ndarray
    elapsed_seconds: float
    throughput_rows_per_second: float
    state_bytes: int
    calibration_peak_state_bytes: int
    output_bytes: int
    diagnostics: dict[str, Any]
    transformed_target: np.ndarray | None = None
    clean_target_scatter_estimate: np.ndarray | None = None
    transformed_scatter_estimate: np.ndarray | None = None
    transformed_scatter_target: np.ndarray | None = None
    marginal_accepted: np.ndarray | None = None


def _sketch_basis(sketch: np.ndarray, rank: int, epsilon: float = 1e-10) -> np.ndarray:
    if not np.any(sketch):
        return np.zeros((sketch.shape[1], 0), dtype=float)
    _, singular_values, right_vectors = np.linalg.svd(sketch, full_matrices=False)
    used = min(rank, int(np.sum(singular_values > epsilon)))
    return right_vectors[:used].T.copy()


def _state_bytes(estimator: Any) -> int:
    missing = object()
    value = getattr(estimator, "state_bytes", missing)
    if value is not missing:
        if callable(value):
            value = value()
        if isinstance(value, (int, np.integer)):
            return int(value)
    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, np.ndarray):
            return int(item.nbytes)
        if isinstance(item, dict):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(child) for child in item)
        if hasattr(item, "__dict__"):
            return visit(vars(item))
        return 0

    return visit(estimator)


def _sketch_force_target(
    matrix: np.ndarray, *, p: int, k: int, lam: float, trim_mode: str
) -> tuple[SketchFORCE, np.ndarray, np.ndarray, np.ndarray]:
    estimator = SketchFORCE(p, k, lam=lam, trim_mode=trim_mode)
    transformed_rows = np.empty_like(matrix)
    accepted = np.empty(matrix.shape, dtype=bool)
    for index, row in enumerate(matrix):
        estimator.update(row)
        location, scale = estimator.tracker.get_location_scale(trim_mode)
        lower = location - lam * scale
        upper = location + lam * scale
        accepted[index] = (row >= lower) & (row <= upper)
        transformed_rows[index] = np.clip(row, lower, upper) - location
    target = transformed_rows.T @ transformed_rows / len(transformed_rows)
    return estimator, covariance_to_correlation(target), target, accepted


def fit_method(
    matrix: np.ndarray,
    *,
    method: str,
    rank: int,
    k: int,
    calibration_size: int,
    marginal_lambda: float = 3.0,
    parallel_lambda: float = 3.0,
    residual_lambda: float = 3.0,
    epoch_size: int | None = None,
) -> FitResult:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("matrix must be nonempty, finite, and two-dimensional")
    p = values.shape[1]
    if not 0 < rank <= k <= p:
        raise ValueError("require 0 < rank <= k <= p")
    transformed_target = None
    clean_target_scatter_estimate = None
    transformed_scatter_estimate = None
    transformed_scatter_target = None
    marginal_accepted = None
    calibration_peak = 0
    diagnostics: dict[str, Any] = {}

    started = time.perf_counter_ns()
    if method == "pearson":
        covariance = np.cov(values, rowvar=False, bias=True)
        correlation = covariance_to_correlation(covariance)
        clean_target_scatter_estimate = covariance
        estimator = None
        state = int(p * 8)
    elif method == "force":
        estimator = ForceEstimator(lambda_scale=marginal_lambda)
        correlation = estimator.fit(values)
        state = _state_bytes(estimator)
    elif method == "mad_force":
        estimator = DriftMADForce(
            p,
            lam=marginal_lambda,
            use_drift_detection=False,
            tail_mode="none",
        )
        correlation = estimator.fit_stationary(values)
        state = _state_bytes(estimator)
    elif method == "vanilla_fd":
        estimator = FrequentDirectionsCovariance(p, k).fit(values)
        covariance = estimator.get_covariance()
        correlation = covariance_to_correlation(covariance)
        clean_target_scatter_estimate = covariance
        transformed_scatter_estimate = covariance
        transformed_scatter_target = np.cov(values, rowvar=False, bias=True)
        state = estimator.state_bytes
    elif method == "rfd":
        estimator = RobustFrequentDirectionsCovariance(p, k).fit(values)
        covariance = estimator.get_covariance()
        correlation = covariance_to_correlation(covariance)
        clean_target_scatter_estimate = covariance
        transformed_scatter_estimate = covariance
        transformed_scatter_target = np.cov(values, rowvar=False, bias=True)
        state = estimator.state_bytes
        diagnostics["rfd_alpha"] = float(estimator.alpha)
        diagnostics["qualification"] = (
            "RFD robust means rank-deficiency regularization, not outlier resistance"
        )
    elif method in {"sketch_iqr", "sketch_mad"}:
        trim_mode = method.removeprefix("sketch_")
        estimator, transformed_target, transformed_scatter_target, marginal_accepted = _sketch_force_target(
            values, p=p, k=k, lam=marginal_lambda, trim_mode=trim_mode
        )
        correlation = estimator.get_correlation()
        transformed_scatter_estimate = estimator.get_covariance()
        clean_target_scatter_estimate = transformed_scatter_estimate
        state = _state_bytes(estimator)
        diagnostics = {
            "effective_n": int(estimator.n),
            "trim_mode": trim_mode,
        }
    elif method == "exact_mad_fd":
        estimator = ExactMADFrequentDirections(p, k, lam=marginal_lambda).fit(values)
        transformed_scatter_estimate = estimator.get_covariance()
        transformed_scatter_target = estimator.transformed_second_moment
        clean_target_scatter_estimate = transformed_scatter_estimate
        correlation = covariance_to_correlation(transformed_scatter_estimate)
        transformed_target = covariance_to_correlation(
            estimator.transformed_second_moment
        )
        lower = estimator.location - marginal_lambda * estimator.scale
        upper = estimator.location + marginal_lambda * estimator.scale
        marginal_accepted = (values >= lower) & (values <= upper)
        state = estimator.state_bytes
        diagnostics["qualification"] = "offline exact-MAD oracle ablation"
    elif method == "dual_bound":
        estimator = DualBoundFORCE(
            p,
            k,
            calibration_size=calibration_size,
            epoch_size=epoch_size,
            marginal_lambda=marginal_lambda,
            parallel_lambda=parallel_lambda,
            residual_lambda=residual_lambda,
        )
        calibration_peak = estimator.calibration_peak_state_bytes_bound
        estimator.fit(values)
        correlation = estimator.get_correlation()
        transformed_scatter_estimate = estimator._serving().sketch.gram(
            estimator._serving().effective_n
        )
        clean_target_scatter_estimate = estimator.get_scale_scatter()
        state = estimator.state_bytes
        diagnostics = estimator.get_diagnostics()
        # Exact finite-sample target induced by the frozen map on the observed
        # estimation rows.  Its gap from the sketch isolates FD compression;
        # it is distinct from the uncontaminated population target.
        serving = estimator._serving()  # internal study audit, not public API use
        if estimator.phase == "estimating" and epoch_size is None:
            marginal_accepted = np.zeros(values.shape, dtype=bool)
            marginal_accepted[calibration_size:] = (
                (values[calibration_size:] >= serving.location - marginal_lambda * serving.scale)
                & (values[calibration_size:] <= serving.location + marginal_lambda * serving.scale)
            )
            clean_map = np.vstack(
                [
                    estimator._transform_with(
                        row,
                        location=serving.location,
                        scale=serving.scale,
                        basis=serving.basis,
                        parallel_radius=serving.parallel_radius,
                        residual_radius=serving.residual_radius,
                        count_diagnostics=False,
                    )
                    for row in values[calibration_size:]
                ]
            )
            transformed_scatter_target = clean_map.T @ clean_map / len(clean_map)
            transformed_target = covariance_to_correlation(transformed_scatter_target)
    else:
        raise ValueError(f"unknown method {method}")

    elapsed = (time.perf_counter_ns() - started) / 1e9
    basis = principal_basis(correlation, rank)
    output_bytes = int(correlation.nbytes + basis.nbytes)
    return FitResult(
        method=method,
        correlation=np.asarray(correlation, dtype=float),
        basis=basis,
        elapsed_seconds=float(elapsed),
        throughput_rows_per_second=float(len(values) / max(elapsed, 1e-12)),
        state_bytes=int(state),
        calibration_peak_state_bytes=int(calibration_peak),
        output_bytes=output_bytes,
        diagnostics=diagnostics,
        transformed_target=transformed_target,
        clean_target_scatter_estimate=clean_target_scatter_estimate,
        transformed_scatter_estimate=transformed_scatter_estimate,
        transformed_scatter_target=transformed_scatter_target,
        marginal_accepted=marginal_accepted,
    )
