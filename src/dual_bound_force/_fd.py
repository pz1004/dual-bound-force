"""Small deterministic Frequent Directions primitive."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


class FrequentDirections:
    """A ``2k x p`` Frequent Directions sketch of supplied rows."""

    def __init__(self, p: int, k: int, *, epsilon: float = 1e-10):
        for name, value in (("p", p), ("k", k)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if p <= 0 or k <= 0 or k > p:
            raise ValueError("require p > 0 and 0 < k <= p")
        if (
            isinstance(epsilon, (bool, np.bool_))
            or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be a finite positive real number")
        self.p = int(p)
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.B = np.zeros((2 * self.k, self.p), dtype=float)
        self.next_row = 0
        self.rows_seen = 0
        self.compressions = 0

    def update(self, row: np.ndarray) -> None:
        value = np.asarray(row, dtype=float)
        if value.shape != (self.p,) or not np.isfinite(value).all():
            raise ValueError("FD row must be finite with shape (p,)")
        self.B[self.next_row] = value
        self.next_row += 1
        self.rows_seen += 1
        if self.next_row == 2 * self.k:
            self._compress()

    def _compress(self) -> None:
        _, singular_values, right_vectors = np.linalg.svd(
            self.B, full_matrices=False
        )
        delta = (
            float(singular_values[self.k] ** 2)
            if len(singular_values) > self.k
            else 0.0
        )
        retained = np.sqrt(
            np.maximum(singular_values[: self.k] ** 2 - delta, 0.0)
        )
        self.B.fill(0.0)
        self.B[: len(retained)] = retained[:, None] * right_vectors[: len(retained)]
        self.next_row = min(self.k, len(retained))
        self.compressions += 1

    def gram(self, denominator: int) -> np.ndarray:
        if denominator <= 0:
            return np.zeros((self.p, self.p), dtype=float)
        return self.B.T @ self.B / int(denominator)

    def basis(self, rank: int | None = None) -> np.ndarray:
        requested = self.k if rank is None else int(rank)
        if requested <= 0:
            raise ValueError("rank must be positive")
        if not np.any(self.B):
            return np.zeros((self.p, 0), dtype=float)
        _, singular_values, right_vectors = np.linalg.svd(
            self.B, full_matrices=False
        )
        positive = int(np.sum(singular_values > self.epsilon))
        used = min(requested, positive)
        return right_vectors[:used].T.copy()

    @property
    def state_bytes(self) -> int:
        return int(self.B.nbytes)
