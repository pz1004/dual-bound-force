"""Dual-Bound-owned implementations of standard sketch ablations.

These compact baselines implement the definitions used by this study without
importing any FORCE-family repository.  They are experiment utilities rather
than public estimator exports.  ``RobustFrequentDirectionsCovariance`` follows
the ridge-regularized RFD update; its name does not imply outlier resistance.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _shape(p: int, k: int) -> tuple[int, int]:
    for name, value in (("p", p), ("k", k)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if p <= 0 or k <= 0 or k > p:
        raise ValueError("require p > 0 and 0 < k <= p")
    return int(p), int(k)


def _positive(name: str, value: Real) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _matrix(matrix: np.ndarray, p: int, *, nonempty: bool = False) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != p or (nonempty and not len(values)):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"matrix must be a {qualifier}two-dimensional array with {p} columns")
    if not np.isfinite(values).all():
        raise ValueError("matrix contains non-finite values")
    return values


class FrequentDirectionsCovariance:
    """Fast FD applied to exact Welford covariance-increment rows."""

    def __init__(self, p: int, k: int, *, epsilon: float = 1e-10):
        self.p, self.k = _shape(p, k)
        self.epsilon = _positive("epsilon", epsilon)
        self.B = np.zeros((2 * self.k, self.p), dtype=float)
        self.next_row = 0
        self.n = 0
        self.mean = np.zeros(self.p, dtype=float)

    def _insert(self, row: np.ndarray) -> None:
        self.B[self.next_row] = row
        self.next_row += 1
        if self.next_row < 2 * self.k:
            return
        _, singular_values, right_vectors = np.linalg.svd(self.B, full_matrices=False)
        delta = float(singular_values[self.k] ** 2) if len(singular_values) > self.k else 0.0
        retained = np.sqrt(np.maximum(singular_values[: self.k] ** 2 - delta, 0.0))
        self.B.fill(0.0)
        self.B[: len(retained)] = retained[:, None] * right_vectors[: len(retained)]
        self.next_row = len(retained)

    def update(self, row: np.ndarray) -> "FrequentDirectionsCovariance":
        value = np.asarray(row, dtype=float)
        if value.shape != (self.p,) or not np.isfinite(value).all():
            raise ValueError(f"row must be finite with shape ({self.p},)")
        previous_n = self.n
        delta = value - self.mean
        self.n += 1
        self.mean += delta / self.n
        centered = np.zeros(self.p) if previous_n == 0 else np.sqrt(previous_n / self.n) * delta
        self._insert(centered)
        return self

    def fit(self, matrix: np.ndarray) -> "FrequentDirectionsCovariance":
        for row in _matrix(matrix, self.p):
            self.update(row)
        return self

    def get_covariance(self) -> np.ndarray:
        if self.n == 0:
            return np.zeros((self.p, self.p), dtype=float)
        return self.B.T @ self.B / self.n

    @property
    def state_bytes(self) -> int:
        return int(self.B.nbytes + self.mean.nbytes)


class RobustFrequentDirectionsCovariance(FrequentDirectionsCovariance):
    """Ridge-regularized Robust Frequent Directions (RFD)."""

    def __init__(self, p: int, k: int, *, epsilon: float = 1e-10):
        self.p, self.k = _shape(p, k)
        self.epsilon = _positive("epsilon", epsilon)
        self.m = self.k + 1
        self.B = np.zeros((2 * self.m, self.p), dtype=float)
        self.next_row = 0
        self.n = 0
        self.mean = np.zeros(self.p, dtype=float)
        self.alpha = 0.0

    def _insert(self, row: np.ndarray) -> None:
        self.B[self.next_row] = row
        self.next_row += 1
        if self.next_row < 2 * self.m:
            return
        _, singular_values, right_vectors = np.linalg.svd(self.B, full_matrices=False)
        delta = float(singular_values[self.m - 1] ** 2) if len(singular_values) >= self.m else 0.0
        retained = np.sqrt(np.maximum(singular_values[: self.k] ** 2 - delta, 0.0))
        self.B.fill(0.0)
        self.B[: len(retained)] = retained[:, None] * right_vectors[: len(retained)]
        self.next_row = len(retained)
        self.alpha += 0.5 * delta

    def get_covariance(self) -> np.ndarray:
        if self.n == 0:
            return np.zeros((self.p, self.p), dtype=float)
        covariance = self.B.T @ self.B
        covariance.flat[:: self.p + 1] += self.alpha
        return covariance / self.n

    @property
    def state_bytes(self) -> int:
        return int(self.B.nbytes + self.mean.nbytes + np.dtype(float).itemsize)


class ExactMADFrequentDirections(FrequentDirectionsCovariance):
    """Offline exact empirical-MAD preprocessing followed by FD."""

    def __init__(self, p: int, k: int, *, lam: float = 3.0, epsilon: float = 1e-10):
        super().__init__(p, k, epsilon=epsilon)
        self.lam = _positive("lam", lam)
        self.location = np.zeros(self.p)
        self.scale = np.ones(self.p)
        self.transformed_second_moment = np.zeros((self.p, self.p))

    def fit(self, matrix: np.ndarray) -> "ExactMADFrequentDirections":
        values = _matrix(matrix, self.p, nonempty=True)
        self.location = np.median(values, axis=0)
        raw_mad = np.median(np.abs(values - self.location), axis=0)
        self.scale = np.maximum(
            1.4826 * raw_mad,
            np.maximum(1e-8, 1e-14 * np.abs(self.location)),
        )
        transformed = np.clip(
            values,
            self.location - self.lam * self.scale,
            self.location + self.lam * self.scale,
        ) - self.location
        self.transformed_second_moment = transformed.T @ transformed / len(values)
        self.B.fill(0.0)
        self.next_row = 0
        self.n = 0
        self.mean.fill(0.0)
        for row in transformed:
            self.n += 1
            self._insert(row)
        return self

    @property
    def state_bytes(self) -> int:
        return int(
            self.B.nbytes
            + self.location.nbytes
            + self.scale.nbytes
            + self.transformed_second_moment.nbytes
        )
