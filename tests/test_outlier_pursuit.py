import numpy as np
import pytest

from dual_bound_force.experiments.metrics import normalized_projection_error
from dual_bound_force.experiments.outlier_pursuit import (
    noisy_outlier_pursuit,
    outlier_support,
)


def _identifiable_fixture(*, seed: int, noise_scale: float):
    rng = np.random.default_rng(seed)
    p, n, rank = 16, 500, 2
    basis, _ = np.linalg.qr(rng.normal(size=(p, rank)), mode="reduced")
    low_rank = basis @ rng.normal(size=(rank, n))
    support = np.zeros(n, dtype=bool)
    support[rng.choice(n, 10, replace=False)] = True
    sparse = np.zeros((p, n), dtype=float)
    directions = rng.normal(size=(p, int(support.sum())))
    directions -= basis @ (basis.T @ directions)
    sparse[:, support] = 8.0 * directions
    noise = noise_scale * rng.normal(size=(p, n))
    return basis, low_rank, sparse, noise, support


def test_noiseless_outlier_pursuit_recovers_subspace_and_support():
    basis, low_rank, sparse, noise, support = _identifiable_fixture(
        seed=601, noise_scale=0.0
    )
    observed = low_rank + sparse + noise
    gamma_star = 0.02
    result = noisy_outlier_pursuit(
        observed,
        regularization=3.0 / (7.0 * np.sqrt(gamma_star * observed.shape[1])),
        noise_budget=0.0,
    )
    assert result.converged
    recovered_basis = np.linalg.svd(result.low_rank, full_matrices=False)[0][:, :2]
    assert normalized_projection_error(basis, recovered_basis) < 1e-4
    predicted = outlier_support(result.column_sparse, relative_threshold=1e-4)
    assert np.array_equal(predicted, support)
    assert result.primal_residual <= 1e-6
    assert result.constraint_violation <= 1e-6


def test_noisy_outlier_pursuit_respects_oracle_frobenius_budget():
    basis, low_rank, sparse, noise, _ = _identifiable_fixture(
        seed=602, noise_scale=1e-3
    )
    observed = low_rank + sparse + noise
    result = noisy_outlier_pursuit(
        observed,
        regularization=3.0 / (7.0 * np.sqrt(0.02 * observed.shape[1])),
        noise_budget=float(np.linalg.norm(noise, ord="fro")),
    )
    assert result.converged
    assert np.linalg.norm(observed - result.low_rank - result.column_sparse, ord="fro") <= np.linalg.norm(noise, ord="fro") + 1e-4
    recovered_basis = np.linalg.svd(result.low_rank, full_matrices=False)[0][:, :2]
    assert normalized_projection_error(basis, recovered_basis) < 1e-3


def test_outlier_pursuit_is_deterministic_and_strictly_validated():
    _, low_rank, sparse, noise, _ = _identifiable_fixture(
        seed=603, noise_scale=1e-4
    )
    observed = low_rank + sparse + noise
    kwargs = {
        "regularization": 0.1,
        "noise_budget": float(np.linalg.norm(noise, ord="fro")),
        "max_iterations": 100,
    }
    first = noisy_outlier_pursuit(observed, **kwargs)
    second = noisy_outlier_pursuit(observed, **kwargs)
    assert np.array_equal(first.low_rank, second.low_rank)
    assert np.array_equal(first.column_sparse, second.column_sparse)
    assert first.diagnostics == second.diagnostics
    with pytest.raises(ValueError):
        noisy_outlier_pursuit(observed, regularization=-1.0, noise_budget=0.0)
    with pytest.raises(ValueError):
        noisy_outlier_pursuit(observed, regularization=0.1, noise_budget=-1.0)


def test_nonconvergence_is_explicit_and_partial_solution_is_not_hidden():
    rng = np.random.default_rng(604)
    result = noisy_outlier_pursuit(
        rng.normal(size=(10, 30)),
        regularization=0.1,
        noise_budget=0.0,
        tolerance=1e-14,
        max_iterations=1,
    )
    assert not result.converged
    assert result.iterations == 1
