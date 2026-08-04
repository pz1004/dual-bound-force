import copy

import numpy as np
import pytest

from dual_bound_force import DualBoundFORCE, NotFittedError
from dual_bound_force._fd import FrequentDirections


def _independent_fd(rows, k):
    rows = np.asarray(rows, dtype=float)
    sketch = np.zeros((2 * k, rows.shape[1]))
    next_row = 0
    for row in rows:
        sketch[next_row] = row
        next_row += 1
        if next_row == 2 * k:
            _, singular, right = np.linalg.svd(sketch, full_matrices=False)
            delta = singular[k] ** 2 if len(singular) > k else 0.0
            kept = np.sqrt(np.maximum(singular[:k] ** 2 - delta, 0.0))
            sketch.fill(0.0)
            sketch[: len(kept)] = kept[:, None] * right[: len(kept)]
            next_row = min(k, len(kept))
    return sketch


def test_fd_matches_independent_reference_and_thin_edge():
    rows = np.random.default_rng(1).normal(size=(37, 7))
    estimator = FrequentDirections(7, 3)
    for row in rows:
        estimator.update(row)
    assert np.allclose(estimator.B.T @ estimator.B, _independent_fd(rows, 3).T @ _independent_fd(rows, 3))

    thin = FrequentDirections(3, 3)
    for row in rows[:, :3]:
        thin.update(row)
    assert np.isfinite(thin.B).all()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"p": 0, "k": 1}, ValueError),
        ({"p": 2, "k": 3}, ValueError),
        ({"p": 2, "k": 1, "calibration_size": 4}, ValueError),
        ({"p": 2, "k": 1, "epoch_size": 512}, ValueError),
        ({"p": 2, "k": 1, "marginal_lambda": np.inf}, ValueError),
    ],
)
def test_constructor_validation(kwargs, error):
    with pytest.raises(error):
        DualBoundFORCE(**kwargs)


def test_initial_calibration_is_excluded_and_exact_mad_is_floored():
    calibration = np.column_stack(
        [np.ones(8), np.linspace(-1, 1, 8), np.arange(8, dtype=float)]
    )
    estimator = DualBoundFORCE(3, 2, calibration_size=8)
    for row in calibration:
        estimator.update(row)
    diagnostics = estimator.get_diagnostics()
    assert diagnostics["phase"] == "estimating"
    assert diagnostics["effective_n"] == 0
    assert not diagnostics["ready"]
    assert estimator.location[0] == pytest.approx(1.0)
    assert estimator.scale[0] == pytest.approx(1e-8)
    with pytest.raises(NotFittedError):
        estimator.get_correlation()

    estimator.update(np.array([1.0, 0.0, 3.0]))
    assert estimator.get_diagnostics()["effective_n"] == 1


def test_calibration_basis_and_dual_radii_match_independent_construction():
    rng = np.random.default_rng(101)
    calibration = rng.normal(size=(24, 11))
    estimator = DualBoundFORCE(
        11,
        3,
        calibration_size=24,
        parallel_lambda=2.0,
        residual_lambda=4.0,
    ).fit(calibration)
    location = np.median(calibration, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(calibration - location), axis=0),
        np.maximum(1e-8, 1e-14 * np.abs(location)),
    )
    standardized = np.clip((calibration - location) / scale, -3.0, 3.0)
    sketch = _independent_fd(standardized, 3)
    _, singular, right = np.linalg.svd(sketch, full_matrices=False)
    basis = right[: min(3, int(np.sum(singular > 1e-10)))].T
    parallel = standardized @ basis @ basis.T
    residual = standardized - parallel

    def radius(values, multiplier):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return max(1e-10, median + multiplier * 1.4826 * mad)

    assert np.allclose(estimator.location, location)
    assert np.allclose(estimator.scale, scale)
    assert np.allclose(
        estimator.basis @ estimator.basis.T,
        basis @ basis.T,
        atol=1e-10,
    )
    assert estimator.parallel_radius == pytest.approx(
        radius(np.linalg.norm(parallel, axis=1), 2.0)
    )
    assert estimator.residual_radius == pytest.approx(
        radius(np.linalg.norm(residual, axis=1), 4.0)
    )


def test_transform_is_nonmutating_and_components_obey_radii():
    rng = np.random.default_rng(2)
    estimator = DualBoundFORCE(12, 3, calibration_size=20).fit(
        rng.normal(size=(50, 12))
    )
    before = copy.deepcopy(estimator.get_diagnostics())
    query = np.full(12, 1e8)
    transformed = estimator.transform(query)
    after = estimator.get_diagnostics()
    assert before == after
    parallel = estimator.basis @ (estimator.basis.T @ transformed)
    residual = transformed - parallel
    assert np.linalg.norm(parallel) <= estimator.parallel_radius + 1e-8
    assert np.linalg.norm(residual) <= estimator.residual_radius + 1e-8


def test_stationary_estimation_cannot_adapt_its_frozen_boundary():
    rng = np.random.default_rng(2002)
    estimator = DualBoundFORCE(12, 3, calibration_size=20)
    estimator.fit(rng.normal(size=(20, 12)))
    basis = estimator.basis.copy()
    location = estimator.location.copy()
    scale = estimator.scale.copy()
    estimator.update(np.full(12, 1e9))
    assert np.array_equal(estimator.basis, basis)
    assert np.array_equal(estimator.location, location)
    assert np.array_equal(estimator.scale, scale)
    assert np.array_equal(estimator.finalize(), estimator.get_correlation())


def test_estimation_rows_are_inserted_directly_into_scatter_sketch():
    rng = np.random.default_rng(22)
    values = rng.normal(size=(47, 9))
    estimator = DualBoundFORCE(9, 3, calibration_size=12).fit(values)
    transformed = np.vstack([estimator.transform(row) for row in values[12:]])
    independent = _independent_fd(transformed, 3)
    assert estimator.effective_n == len(transformed)
    assert np.allclose(
        estimator._sketch.B.T @ estimator._sketch.B,
        independent.T @ independent,
        atol=1e-10,
    )


def test_zero_variance_correlation_and_positive_diagonal_contract():
    rng = np.random.default_rng(3)
    values = np.column_stack(
        [np.ones(80), rng.normal(size=80), rng.normal(size=80)]
    )
    estimator = DualBoundFORCE(3, 3, calibration_size=20).fit(values)
    correlation = estimator.get_correlation()
    assert np.isfinite(correlation).all()
    assert np.all((correlation >= -1) & (correlation <= 1))
    assert correlation[0, 0] == 0.0
    assert correlation[1, 1] == 1.0
    assert correlation[2, 2] == 1.0


def test_rank_zero_calibration_floors_both_radii():
    estimator = DualBoundFORCE(3, 2, calibration_size=5, epsilon=1e-10)
    estimator.fit(np.zeros((5, 3)))
    assert estimator.basis.shape == (3, 0)
    assert estimator.parallel_radius == estimator.epsilon
    assert estimator.residual_radius == estimator.epsilon
    estimator.update(np.zeros(3))
    assert np.array_equal(estimator.transform(np.zeros(3)), np.zeros(3))


def test_strictly_positive_tiny_variance_receives_unit_diagonal():
    estimator = DualBoundFORCE(2, 2, calibration_size=5, epsilon=1e-10)
    estimator.fit(np.zeros((5, 2)))
    estimator._sketch.B.fill(0.0)
    estimator._sketch.B[0] = np.array([1e-8, 0.0])
    estimator.effective_n = 1
    correlation = estimator.get_correlation()
    assert correlation[0, 0] == 1.0
    assert correlation[1, 1] == 0.0


def test_correlation_and_subspace_paths_use_the_same_positive_variances():
    rng = np.random.default_rng(3030)
    estimator = DualBoundFORCE(7, 4, calibration_size=20).fit(
        rng.normal(size=(100, 7))
    )
    correlation = estimator.get_correlation()
    expected_values, expected_vectors = np.linalg.eigh(
        (correlation + correlation.T) / 2.0
    )
    expected = expected_vectors[:, np.argsort(expected_values)[-3:]]
    actual = estimator.get_subspace(3)
    assert np.linalg.norm(
        expected @ expected.T - actual @ actual.T, ord="fro"
    ) < 1e-10


def test_epoch_transition_serves_previous_then_uses_new_denominator():
    rng = np.random.default_rng(4)
    estimator = DualBoundFORCE(
        5, 2, calibration_size=5, epoch_size=12
    )
    estimator.fit(rng.normal(size=(12, 5)))
    first = estimator.get_correlation().copy()
    assert estimator.get_diagnostics()["effective_n"] == 7

    estimator.update(rng.normal(size=5))
    diagnostics = estimator.get_diagnostics()
    assert diagnostics["phase"] == "calibrating"
    assert diagnostics["serving_epoch"] == 0
    assert np.allclose(estimator.get_correlation(), first)
    query = rng.normal(size=5)
    old_transform = estimator.transform(query)
    estimator.fit(rng.normal(size=(4, 5)))
    assert estimator.get_diagnostics()["phase"] == "estimating"
    # Recalibration is complete, but the old estimate and its preprocessing
    # remain atomically paired until one new estimation row exists.
    assert np.allclose(estimator.transform(query), old_transform)
    estimator.update(rng.normal(size=5))
    assert estimator.get_diagnostics()["effective_n"] == 1
    assert estimator.get_diagnostics()["serving_epoch"] == 1


def test_active_and_calibration_state_scaling():
    active = []
    calibration = []
    for p in (20, 40, 80):
        rng = np.random.default_rng(p)
        estimator = DualBoundFORCE(p, 5, calibration_size=16)
        calibration.append(estimator.state_bytes)
        estimator.fit(rng.normal(size=(17, p)))
        active.append(estimator.state_bytes)
    assert np.allclose(np.array(active) / np.array([20, 40, 80]), active[0] / 20)
    assert np.allclose(np.array(calibration) / np.array([20, 40, 80]), calibration[0] / 20)


def test_calibration_peak_bound_scales_and_exceeds_resident_block():
    values = []
    for p, k, calibration_size in ((20, 4, 16), (40, 8, 24), (80, 16, 32)):
        estimator = DualBoundFORCE(p, k, calibration_size=calibration_size)
        values.append(
            (
                p * (k + calibration_size),
                estimator.calibration_peak_state_bytes_bound,
            )
        )
        assert estimator.calibration_peak_state_bytes_bound > estimator.state_bytes
    predictors = np.asarray([item[0] for item in values], dtype=float)
    bounds = np.asarray([item[1] for item in values], dtype=float)
    assert np.all(np.diff(bounds) > 0)
    assert np.all(bounds / predictors > 8.0)


def test_calibration_peak_bound_includes_retained_scheduled_estimate():
    rng = np.random.default_rng(2080)
    estimator = DualBoundFORCE(
        80, 16, calibration_size=32, epoch_size=40
    ).fit(rng.normal(size=(41, 80)))
    assert estimator.get_diagnostics()["phase"] == "calibrating"
    assert estimator.get_diagnostics()["serving_epoch"] == 0
    assert estimator.calibration_peak_state_bytes_bound > estimator.state_bytes


def test_zero_components_are_not_counted_as_norm_clipped():
    estimator = DualBoundFORCE(3, 3, calibration_size=5).fit(np.zeros((6, 3)))
    diagnostics = estimator.get_diagnostics()
    assert diagnostics["parallel_clipped_rows"] == 0
    assert diagnostics["residual_clipped_rows"] == 0


def test_scheduled_epoch_history_is_bounded():
    estimator = DualBoundFORCE(2, 1, calibration_size=5, epoch_size=6)
    estimator.fit(np.zeros((6 * 70, 2)))
    diagnostics = estimator.get_diagnostics()
    assert len(diagnostics["epoch_history"]) == 64
    assert diagnostics["epoch_history"][0]["epoch_index"] > 0


def test_input_validation_and_scale_scatter():
    estimator = DualBoundFORCE(4, 2, calibration_size=5)
    with pytest.raises(ValueError):
        estimator.update(np.zeros(3))
    with pytest.raises(ValueError):
        estimator.update(np.array([0.0, 1.0, np.nan, 2.0]))
    matrix = np.random.default_rng(5).normal(size=(30, 4))
    estimator.fit(matrix)
    scatter = estimator.get_scale_scatter()
    assert scatter.shape == (4, 4)
    assert np.allclose(scatter, scatter.T)
    with pytest.raises(ValueError):
        estimator.get_subspace(3)
