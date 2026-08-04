"""Dual-bound robust streaming correlation sketch."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from ._fd import FrequentDirections


class NotFittedError(RuntimeError):
    """Raised when an estimate is requested before calibration completes."""


@dataclass
class _ServingState:
    sketch: FrequentDirections
    mean: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    parallel_radius: float
    residual_radius: float
    effective_n: int
    epoch_index: int


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _positive(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _robust_upper_radius(values: np.ndarray, multiplier: float, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return float(max(floor, median + multiplier * 1.4826 * mad))


def _correlation_from_gram(gram: np.ndarray, epsilon: float) -> np.ndarray:
    diagonal = np.maximum(np.diag(gram), 0.0)
    # The manuscript's output contract is mathematical positivity, not a
    # numerical-rank test: every strictly positive estimated variance receives
    # a unit diagonal.  ``epsilon`` remains part of the estimator interface for
    # radial regularization and singular-value rank detection.
    positive = diagonal > 0.0
    correlation = np.zeros_like(gram)
    if np.any(positive):
        indices = np.flatnonzero(positive)
        denominator = np.sqrt(np.outer(diagonal[indices], diagonal[indices]))
        block = gram[np.ix_(indices, indices)] / denominator
        correlation[np.ix_(indices, indices)] = np.clip(block, -1.0, 1.0)
        correlation[indices, indices] = 1.0
    return correlation


class DualBoundFORCE:
    """MAD-calibrated, dual-influence-bounded FD correlation estimator.

    Calibration observations set parameters and are excluded from the
    estimation denominator.  With scheduled epochs, the preceding completed
    estimate remains serviceable during the next calibration block.  Epoch
    diagnostics retain only the newest 64 completed epochs so that optional
    scheduling cannot create stream-length-dependent estimator state.
    """

    def __init__(
        self,
        p: int,
        k: int,
        *,
        calibration_size: int = 512,
        epoch_size: int | None = None,
        marginal_lambda: float = 3.0,
        parallel_lambda: float = 3.0,
        residual_lambda: float = 3.0,
        scale_floor: float = 1e-8,
        epsilon: float = 1e-10,
    ):
        self.p = _integer("p", p, minimum=1)
        self.k = _integer("k", k, minimum=1)
        if self.k > self.p:
            raise ValueError("k must not exceed p")
        self.calibration_size = _integer(
            "calibration_size", calibration_size, minimum=5
        )
        if epoch_size is not None:
            epoch_size = _integer("epoch_size", epoch_size, minimum=1)
            if epoch_size <= self.calibration_size:
                raise ValueError("epoch_size must exceed calibration_size")
        self.epoch_size = epoch_size
        self.marginal_lambda = _positive("marginal_lambda", marginal_lambda)
        self.parallel_lambda = _positive("parallel_lambda", parallel_lambda)
        self.residual_lambda = _positive("residual_lambda", residual_lambda)
        self.scale_floor = _positive("scale_floor", scale_floor)
        self.epsilon = _positive("epsilon", epsilon)
        self.reset()

    def reset(self) -> "DualBoundFORCE":
        self.total_seen = 0
        self.epoch_index = 0
        self.epoch_seen = 0
        self.phase = "calibrating"
        self.calibration_count = 0
        self._calibration: np.ndarray | None = np.empty(
            (self.calibration_size, self.p), dtype=float
        )
        self.location: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.parallel_radius: float | None = None
        self.residual_radius: float | None = None
        self._sketch: FrequentDirections | None = None
        self._mean = np.zeros(self.p, dtype=float)
        self.effective_n = 0
        self._last_completed: _ServingState | None = None
        self.marginal_clipped_coordinates = 0
        self.parallel_clipped_rows = 0
        self.residual_clipped_rows = 0
        self.epoch_history: list[dict[str, Any]] = []
        self._epoch_history_limit = 64
        # Conservative simultaneous-array envelope for calibration: the
        # resident raw block, a working copy, standardized/parallel/residual
        # workspaces and one transient C-by-p expression (6Cp); the 2k-by-p
        # preliminary sketch plus p-by-k basis (3kp); the C-by-k projection;
        # and coordinate/norm vectors.  A scheduled recalibration can also
        # retain the preceding serving sketch, basis, mean, location, and
        # scale (an additional 3kp + 3p).  This is an explicit numerical-state
        # bound, not process RSS; stationary initial calibration is cheaper.
        self.calibration_peak_state_bytes_bound = int(
            8
            * (
                6 * self.calibration_size * self.p
                + 6 * self.k * self.p
                + self.calibration_size * self.k
                + 7 * self.p
                + 2 * self.calibration_size
            )
        )
        return self

    def _validate_row(self, row: np.ndarray) -> np.ndarray:
        value = np.asarray(row, dtype=float)
        if value.shape != (self.p,):
            raise ValueError(f"row must have shape ({self.p},)")
        if not np.isfinite(value).all():
            raise ValueError("row contains non-finite values")
        return value

    def _calibrate(self) -> None:
        assert self._calibration is not None
        values = self._calibration.copy()
        location = np.median(values, axis=0)
        raw_mad = np.median(np.abs(values - location), axis=0)
        scale = 1.4826 * raw_mad
        scale = np.maximum(
            scale,
            np.maximum(self.scale_floor, 1e-14 * np.abs(location)),
        )
        standardized = np.clip(
            (values - location) / scale,
            -self.marginal_lambda,
            self.marginal_lambda,
        )
        preliminary = FrequentDirections(self.p, self.k, epsilon=self.epsilon)
        for row in standardized:
            preliminary.update(row)
        basis = preliminary.basis(self.k)
        if basis.shape[1]:
            parallel = standardized @ basis @ basis.T
        else:
            parallel = np.zeros_like(standardized)
        residual = standardized - parallel
        parallel_norms = np.linalg.norm(parallel, axis=1)
        residual_norms = np.linalg.norm(residual, axis=1)

        self.location = location
        self.scale = scale
        self.basis = basis
        self.parallel_radius = _robust_upper_radius(
            parallel_norms, self.parallel_lambda, self.epsilon
        )
        self.residual_radius = _robust_upper_radius(
            residual_norms, self.residual_lambda, self.epsilon
        )
        self._sketch = FrequentDirections(self.p, self.k, epsilon=self.epsilon)
        self._mean.fill(0.0)
        self.effective_n = 0
        self._calibration = None
        self.phase = "estimating"

    def _transform_with(
        self,
        row: np.ndarray,
        *,
        location: np.ndarray,
        scale: np.ndarray,
        basis: np.ndarray,
        parallel_radius: float,
        residual_radius: float,
        count_diagnostics: bool,
    ) -> np.ndarray:
        standardized_raw = (row - location) / scale
        standardized = np.clip(
            standardized_raw, -self.marginal_lambda, self.marginal_lambda
        )
        if count_diagnostics:
            self.marginal_clipped_coordinates += int(
                np.count_nonzero(standardized != standardized_raw)
            )
        if basis.shape[1]:
            parallel = basis @ (basis.T @ standardized)
        else:
            parallel = np.zeros(self.p, dtype=float)
        residual = standardized - parallel
        parallel_norm = float(np.linalg.norm(parallel))
        residual_norm = float(np.linalg.norm(residual))
        # Apply the same continuous epsilon-regularized radial map at every
        # norm.  A former ``norm <= epsilon`` identity shortcut was harmless
        # for the fitted nondegenerate study radii, but made the map
        # discontinuous when a radius attained its numerical floor and thus
        # prevented the global nonexpansiveness result used in the calibration
        # stability lemma.
        parallel_factor = min(
            1.0, parallel_radius / (parallel_norm + self.epsilon)
        )
        residual_factor = min(
            1.0, residual_radius / (residual_norm + self.epsilon)
        )
        if count_diagnostics:
            self.parallel_clipped_rows += int(
                parallel_norm > 0.0 and parallel_factor < 1.0
            )
            self.residual_clipped_rows += int(
                residual_norm > 0.0 and residual_factor < 1.0
            )
        return parallel_factor * parallel + residual_factor * residual

    def _current_transform(self, row: np.ndarray, *, diagnostics: bool) -> np.ndarray:
        if self.phase == "estimating" and self.location is not None:
            assert self.scale is not None and self.basis is not None
            assert self.parallel_radius is not None and self.residual_radius is not None
            return self._transform_with(
                row,
                location=self.location,
                scale=self.scale,
                basis=self.basis,
                parallel_radius=self.parallel_radius,
                residual_radius=self.residual_radius,
                count_diagnostics=diagnostics,
            )
        if self._last_completed is not None:
            state = self._last_completed
            return self._transform_with(
                row,
                location=state.location,
                scale=state.scale,
                basis=state.basis,
                parallel_radius=state.parallel_radius,
                residual_radius=state.residual_radius,
                count_diagnostics=False,
            )
        raise NotFittedError("initial calibration has not completed")

    def transform(self, row: np.ndarray) -> np.ndarray:
        """Transform a query with the parameters of the estimate being served."""
        value = self._validate_row(row)
        if (
            self.phase == "estimating"
            and self.effective_n > 0
            and self.location is not None
        ):
            assert self.scale is not None and self.basis is not None
            assert self.parallel_radius is not None and self.residual_radius is not None
            return self._transform_with(
                value,
                location=self.location,
                scale=self.scale,
                basis=self.basis,
                parallel_radius=self.parallel_radius,
                residual_radius=self.residual_radius,
                count_diagnostics=False,
            )
        if self._last_completed is not None:
            state = self._last_completed
            return self._transform_with(
                value,
                location=state.location,
                scale=state.scale,
                basis=state.basis,
                parallel_radius=state.parallel_radius,
                residual_radius=state.residual_radius,
                count_diagnostics=False,
            )
        raise NotFittedError("no estimation observations are available")

    def _snapshot_current(self) -> None:
        if self._sketch is None or self.effective_n <= 0:
            return
        assert self.location is not None and self.scale is not None
        assert self.basis is not None
        assert self.parallel_radius is not None and self.residual_radius is not None
        self._last_completed = _ServingState(
            sketch=self._sketch,
            mean=self._mean.copy(),
            location=self.location.copy(),
            scale=self.scale.copy(),
            basis=self.basis.copy(),
            parallel_radius=float(self.parallel_radius),
            residual_radius=float(self.residual_radius),
            effective_n=int(self.effective_n),
            epoch_index=int(self.epoch_index),
        )
        self.epoch_history.append(
            {
                "epoch_index": int(self.epoch_index),
                "effective_n": int(self.effective_n),
                "parallel_radius": float(self.parallel_radius),
                "residual_radius": float(self.residual_radius),
            }
        )
        if len(self.epoch_history) > self._epoch_history_limit:
            del self.epoch_history[: -self._epoch_history_limit]

    def _start_next_epoch(self) -> None:
        self._snapshot_current()
        self.epoch_index += 1
        self.epoch_seen = 0
        self.phase = "calibrating"
        self.calibration_count = 0
        self._calibration = np.empty(
            (self.calibration_size, self.p), dtype=float
        )
        self.location = None
        self.scale = None
        self.basis = None
        self.parallel_radius = None
        self.residual_radius = None
        self._sketch = None
        self._mean.fill(0.0)
        self.effective_n = 0

    def update(self, row: np.ndarray) -> "DualBoundFORCE":
        value = self._validate_row(row)
        if self.epoch_size is not None and self.epoch_seen == self.epoch_size:
            self._start_next_epoch()

        self.total_seen += 1
        self.epoch_seen += 1
        if self.phase == "calibrating":
            assert self._calibration is not None
            self._calibration[self.calibration_count] = value
            self.calibration_count += 1
            if self.calibration_count == self.calibration_size:
                self._calibrate()
            return self

        transformed = self._current_transform(value, diagnostics=True)
        if self.effective_n == 0:
            self._last_completed = None
        self.effective_n += 1
        delta = transformed - self._mean
        self._mean += delta / self.effective_n
        assert self._sketch is not None
        # The target is the calibration-centered transformed scatter
        # E[z z^T], not a second re-centered covariance.  Direct insertion is
        # also what makes the replacement term at most epsilon_row M^2, where
        # M^2 is the smaller simultaneous marginal/radial row envelope.
        self._sketch.update(transformed)
        return self

    def fit(self, matrix: np.ndarray) -> "DualBoundFORCE":
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.p:
            raise ValueError(f"matrix must have shape (n, {self.p})")
        if not np.isfinite(values).all():
            raise ValueError("matrix contains non-finite values")
        for row in values:
            self.update(row)
        return self

    def _serving(self) -> _ServingState:
        if self.phase == "estimating" and self._sketch is not None and self.effective_n > 0:
            assert self.location is not None and self.scale is not None
            assert self.basis is not None
            assert self.parallel_radius is not None and self.residual_radius is not None
            return _ServingState(
                self._sketch,
                self._mean,
                self.location,
                self.scale,
                self.basis,
                self.parallel_radius,
                self.residual_radius,
                self.effective_n,
                self.epoch_index,
            )
        if self._last_completed is not None:
            return self._last_completed
        raise NotFittedError("no estimation observations are available")

    def _gram(self) -> tuple[np.ndarray, _ServingState]:
        state = self._serving()
        return state.sketch.gram(state.effective_n), state

    def get_correlation(self) -> np.ndarray:
        """Return correlation of the calibration-centered transformed scatter."""
        gram, _ = self._gram()
        return _correlation_from_gram(gram, self.epsilon)

    def get_scale_scatter(self) -> np.ndarray:
        """Return scale-reconstructed robust scatter, not classical covariance."""
        scatter, state = self._gram()
        return state.scale[:, None] * scatter * state.scale[None, :]

    def get_subspace(self, rank: int | None = None) -> np.ndarray:
        """Return leading directions of the served transformed correlation."""
        state = self._serving()
        requested = self.k if rank is None else _integer("rank", rank, minimum=1)
        if requested > self.k:
            raise ValueError("rank must not exceed sketch rank k")
        gram = state.sketch.gram(state.effective_n)
        diagonal = np.maximum(np.diag(gram), 0.0)
        positive = diagonal > 0.0
        scaled_sketch = state.sketch.B.copy()
        scaled_sketch[:, positive] /= np.sqrt(diagonal[positive])
        scaled_sketch[:, ~positive] = 0.0
        if not np.any(scaled_sketch):
            return np.zeros((self.p, 0), dtype=float)
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_sketch, full_matrices=False
        )
        used = min(requested, int(np.sum(singular_values > self.epsilon)))
        return right_vectors[:used].T.copy()

    def finalize(self) -> np.ndarray:
        return self.get_correlation()

    @property
    def state_bytes(self) -> int:
        arrays = [self._mean]
        if self._calibration is not None:
            arrays.append(self._calibration)
        for value in (self.location, self.scale, self.basis):
            if value is not None:
                arrays.append(value)
        if self._sketch is not None:
            arrays.append(self._sketch.B)
        if self._last_completed is not None:
            arrays.extend(
                [
                    self._last_completed.sketch.B,
                    self._last_completed.mean,
                    self._last_completed.location,
                    self._last_completed.scale,
                    self._last_completed.basis,
                ]
            )
        return int(sum(value.nbytes for value in arrays))

    def get_diagnostics(self) -> dict[str, Any]:
        serving_epoch = None
        serving_n = 0
        try:
            serving = self._serving()
            serving_epoch = int(serving.epoch_index)
            serving_n = int(serving.effective_n)
        except NotFittedError:
            pass
        return {
            "phase": self.phase,
            "ready": serving_epoch is not None,
            "total_seen": int(self.total_seen),
            "epoch_index": int(self.epoch_index),
            "epoch_seen": int(self.epoch_seen),
            "calibration_count": int(self.calibration_count),
            "effective_n": int(self.effective_n),
            "serving_epoch": serving_epoch,
            "serving_n": serving_n,
            "basis_rank": 0 if self.basis is None else int(self.basis.shape[1]),
            "parallel_radius": self.parallel_radius,
            "residual_radius": self.residual_radius,
            "marginal_clipped_coordinates": int(self.marginal_clipped_coordinates),
            "parallel_clipped_rows": int(self.parallel_clipped_rows),
            "residual_clipped_rows": int(self.residual_clipped_rows),
            "state_bytes": self.state_bytes,
            "calibration_peak_state_bytes_bound": int(
                self.calibration_peak_state_bytes_bound
            ),
            "epoch_history": list(self.epoch_history),
        }
