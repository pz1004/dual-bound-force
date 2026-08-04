import numpy as np
import pytest

from dual_bound_force.theory import (
    bounded_matrix_bernstein_bound,
    calibration_scatter_stability_bound,
    calibration_transform_stability_bound,
    combined_influence_radius_squared,
    conditional_operator_error_bound,
    correlation_normalization_bound,
    davis_kahan_bound,
    explicit_fd_tail_envelope,
    fd_tail_bound,
    influence_radius_squared,
    marginal_map_rms_bound,
    radial_contraction,
    radial_only_replacement_contamination_bound,
    replacement_contamination_bound,
)


def test_dual_radius_and_replacement_bound_are_exact_envelopes():
    assert influence_radius_squared(3.0, 4.0) == 25.0
    assert radial_only_replacement_contamination_bound(0.1, 3.0, 4.0) == 2.5
    assert replacement_contamination_bound(
        0.1,
        3.0,
        4.0,
        dimension=2,
        marginal_lambda=0.1,
    ) == pytest.approx(0.002)
    with pytest.raises(TypeError):
        replacement_contamination_bound(
            0.1, 3.0, 4.0, dimension=2
        )
    with pytest.raises(ValueError):
        replacement_contamination_bound(
            -0.1, 1.0, 1.0, dimension=2, marginal_lambda=1.0
        )


def test_combined_marginal_and_dual_row_envelope_uses_the_sharper_bound():
    assert combined_influence_radius_squared(
        dimension=100,
        marginal_lambda=3.0,
        parallel_radius=4.0,
        residual_radius=1.5,
    ) == pytest.approx(18.25)
    assert combined_influence_radius_squared(
        dimension=2,
        marginal_lambda=1.0,
        parallel_radius=4.0,
        residual_radius=1.5,
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        combined_influence_radius_squared(
            dimension=0,
            marginal_lambda=1.0,
            parallel_radius=1.0,
            residual_radius=1.0,
        )


def test_random_dual_contractions_obey_the_simultaneous_row_envelope():
    rng = np.random.default_rng(92)
    dimension, rank = 17, 4
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")
    marginal_lambda = 1.75
    parallel_radius = 2.4
    residual_radius = 1.3
    envelope = combined_influence_radius_squared(
        dimension=dimension,
        marginal_lambda=marginal_lambda,
        parallel_radius=parallel_radius,
        residual_radius=residual_radius,
    )
    for _ in range(100):
        y = np.clip(
            rng.normal(scale=5.0, size=dimension),
            -marginal_lambda,
            marginal_lambda,
        )
        parallel = basis @ (basis.T @ y)
        residual = y - parallel
        parallel *= min(
            1.0,
            parallel_radius / (np.linalg.norm(parallel) + 1e-10),
        )
        residual *= min(
            1.0,
            residual_radius / (np.linalg.norm(residual) + 1e-10),
        )
        assert np.dot(parallel + residual, parallel + residual) <= envelope + 1e-12


def test_explicit_fd_corollary_dominates_the_data_dependent_tail():
    rng = np.random.default_rng(91)
    rows = rng.normal(size=(80, 9))
    row_norms = np.linalg.norm(rows, axis=1)
    rows /= np.maximum(row_norms[:, None], 1.0)
    radius_squared = float(np.max(np.sum(rows**2, axis=1)))
    observed = fd_tail_bound(rows, retained_rank=5, comparison_rank=2)
    envelope = explicit_fd_tail_envelope(
        radius_squared=radius_squared,
        retained_rank=5,
        comparison_rank=2,
    )
    assert observed <= envelope + 1e-12
    assert envelope == pytest.approx(radius_squared / 3.0)


def test_fd_tail_bound_vanishes_for_exact_low_rank():
    rng = np.random.default_rng(9)
    rows = rng.normal(size=(80, 2)) @ rng.normal(size=(2, 10))
    assert fd_tail_bound(rows, retained_rank=6, comparison_rank=2) < 1e-25


def test_deprecated_fd_sketch_rows_alias_is_explicit_and_equivalent():
    rows = np.eye(4)
    with pytest.deprecated_call(match="retained_rank"):
        old = fd_tail_bound(rows, sketch_rows=3, comparison_rank=1)
    new = fd_tail_bound(rows, retained_rank=3, comparison_rank=1)
    assert old == new
    with pytest.raises(ValueError):
        fd_tail_bound(
            rows, retained_rank=3, sketch_rows=3, comparison_rank=1
        )


def test_sampling_normalization_and_subspace_bounds_validate_assumptions():
    assert bounded_matrix_bernstein_bound(
        radius_squared=4.0, n=1000, dimension=20, delta=0.05
    ) > 0
    assert correlation_normalization_bound(0.1, 1.0, 1.0) == pytest.approx(
        0.2 + (2.0 + np.sqrt(2.0)) * 0.1
    )
    assert davis_kahan_bound(0.1, 1.0) == pytest.approx(0.2)
    assert davis_kahan_bound(1.0, 0.1) == 1.0
    with pytest.raises(ValueError):
        correlation_normalization_bound(0.6, 1.0, 1.0)


def test_conditional_theorem_decomposition_preserves_replacement_constant():
    components = conditional_operator_error_bound(
        calibration_bias=0.1,
        clipping_bias=0.2,
        sampling_deviation=0.3,
        contamination_fraction=0.1,
        parallel_radius=3.0,
        residual_radius=4.0,
        fd_tail_loss=0.4,
        dimension=100,
        marginal_lambda=3.0,
    )
    assert components["replacement_contamination"] == 2.5
    assert components["total"] == pytest.approx(3.5)
    with pytest.raises(ValueError):
        conditional_operator_error_bound(
            calibration_bias=-0.1,
            clipping_bias=0.0,
            sampling_deviation=0.0,
            contamination_fraction=0.0,
            parallel_radius=1.0,
            residual_radius=1.0,
            fd_tail_loss=0.0,
            dimension=2,
            marginal_lambda=1.0,
        )


def test_replacement_and_normalization_bounds_dominate_constructed_errors():
    rng = np.random.default_rng(303)
    n, p, replaced = 200, 8, 20
    clean = rng.normal(size=(n, p))
    clean /= np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1.0)
    observed = clean.copy()
    replacement = rng.normal(size=(replaced, p))
    replacement /= np.maximum(
        np.linalg.norm(replacement, axis=1, keepdims=True), 1.0
    )
    observed[:replaced] = replacement
    actual = np.linalg.norm(
        observed.T @ observed / n - clean.T @ clean / n, ord=2
    )
    assert actual <= replacement_contamination_bound(
        0.1,
        1.0,
        0.0,
        dimension=p,
        marginal_lambda=1.0,
    ) + 1e-12

    target = np.diag(np.linspace(1.0, 2.0, p)) + 0.03 * np.ones((p, p))
    perturbation = np.diag(np.linspace(-0.02, 0.02, p))
    estimate = target + perturbation
    target_scale = np.sqrt(np.diag(target))
    estimate_scale = np.sqrt(np.diag(estimate))
    target_correlation = target / np.outer(target_scale, target_scale)
    estimate_correlation = estimate / np.outer(estimate_scale, estimate_scale)
    eta = np.linalg.norm(perturbation, ord=2)
    bound = correlation_normalization_bound(
        eta, float(np.min(np.diag(target))), float(np.linalg.norm(target, ord=2))
    )
    assert np.linalg.norm(estimate_correlation - target_correlation, ord=2) <= bound


def test_replacement_constant_is_sharp_for_orthogonal_unit_rows():
    clean = np.array([[1.0, 0.0]])
    observed = np.array([[0.0, 1.0]])
    actual = np.linalg.norm(
        observed.T @ observed - clean.T @ clean, ord=2
    )
    assert actual == pytest.approx(
        replacement_contamination_bound(
            1.0,
            1.0,
            0.0,
            dimension=2,
            marginal_lambda=1.0,
        )
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius_squared": np.inf, "n": 10, "dimension": 2, "delta": 0.1},
        {"radius_squared": 1.0, "n": 1.5, "dimension": 2, "delta": 0.1},
        {"radius_squared": 1.0, "n": 10, "dimension": True, "delta": 0.1},
    ],
)
def test_matrix_bernstein_rejects_nonfinite_or_noninteger_inputs(kwargs):
    with pytest.raises(ValueError):
        bounded_matrix_bernstein_bound(**kwargs)


def test_regularized_radial_contraction_is_nonexpansive_and_radius_lipschitz():
    rng = np.random.default_rng(510)
    for radius in (0.0, 1e-10, 0.4, 3.0):
        for _ in range(200):
            left = rng.normal(size=9)
            right = rng.normal(size=9)
            mapped_left = radial_contraction(left, radius=radius, epsilon=1e-10)
            mapped_right = radial_contraction(right, radius=radius, epsilon=1e-10)
            assert np.linalg.norm(mapped_left - mapped_right) <= np.linalg.norm(left - right) + 1e-12

    vector = rng.normal(size=9)
    for first, second in ((0.0, 1e-10), (0.2, 0.7), (2.0, 5.0)):
        difference = np.linalg.norm(
            radial_contraction(vector, radius=first)
            - radial_contraction(vector, radius=second)
        )
        assert difference <= abs(first - second) + 1e-12


def test_marginal_map_bound_dominates_monte_carlo_rms_error():
    rng = np.random.default_rng(511)
    samples = rng.normal(size=(20_000, 6))
    ideal_location = np.linspace(-0.2, 0.2, 6)
    fitted_location = ideal_location + np.linspace(-0.01, 0.015, 6)
    ideal_scale = np.linspace(0.8, 1.3, 6)
    fitted_scale = ideal_scale + np.linspace(0.01, 0.03, 6)
    marginal_lambda = 2.5
    ideal = np.clip(
        (samples - ideal_location) / ideal_scale,
        -marginal_lambda,
        marginal_lambda,
    )
    fitted = np.clip(
        (samples - fitted_location) / fitted_scale,
        -marginal_lambda,
        marginal_lambda,
    )
    actual = float(np.sqrt(np.mean(np.sum((fitted - ideal) ** 2, axis=1))))
    moments = np.mean((samples - ideal_location) ** 2, axis=0)
    bound = marginal_map_rms_bound(
        location_error=np.abs(fitted_location - ideal_location),
        scale_error=np.abs(fitted_scale - ideal_scale),
        centered_second_moments=moments,
        scale_floor=float(min(np.min(ideal_scale), np.min(fitted_scale))),
        marginal_lambda=marginal_lambda,
    )
    assert actual <= bound + 1e-12


def _dual_map(
    rows: np.ndarray,
    *,
    basis: np.ndarray,
    parallel_radius: float,
    residual_radius: float,
) -> np.ndarray:
    projected = rows @ basis @ basis.T if basis.shape[1] else np.zeros_like(rows)
    residual = rows - projected
    return np.vstack(
        [
            radial_contraction(a, radius=parallel_radius)
            + radial_contraction(r, radius=residual_radius)
            for a, r in zip(projected, residual, strict=True)
        ]
    )


def test_calibration_transform_and_scatter_stability_bounds_hold():
    rng = np.random.default_rng(512)
    n, p, rank = 4_000, 8, 3
    rows = np.clip(rng.normal(size=(n, p)), -2.0, 2.0)
    ideal_basis, _ = np.linalg.qr(rng.normal(size=(p, rank)), mode="reduced")
    rotation, _ = np.linalg.qr(np.eye(p) + 0.01 * rng.normal(size=(p, p)))
    fitted_basis = rotation @ ideal_basis
    ideal_projector = ideal_basis @ ideal_basis.T
    fitted_projector = fitted_basis @ fitted_basis.T
    ideal_parallel, ideal_residual = 2.0, 1.2
    fitted_parallel, fitted_residual = 2.1, 1.1
    ideal = _dual_map(
        rows,
        basis=ideal_basis,
        parallel_radius=ideal_parallel,
        residual_radius=ideal_residual,
    )
    fitted = _dual_map(
        rows,
        basis=fitted_basis,
        parallel_radius=fitted_parallel,
        residual_radius=fitted_residual,
    )
    projector_error = float(np.linalg.norm(fitted_projector - ideal_projector, ord=2))
    transform_bound = calibration_transform_stability_bound(
        marginal_rms_error=0.0,
        projector_error=projector_error,
        dimension=p,
        marginal_lambda=2.0,
        parallel_radius_error=abs(fitted_parallel - ideal_parallel),
        residual_radius_error=abs(fitted_residual - ideal_residual),
    )
    actual_rms = float(np.sqrt(np.mean(np.sum((fitted - ideal) ** 2, axis=1))))
    assert actual_rms <= transform_bound + 1e-12

    fitted_m2 = combined_influence_radius_squared(
        dimension=p,
        marginal_lambda=2.0,
        parallel_radius=fitted_parallel,
        residual_radius=fitted_residual,
    )
    ideal_m2 = combined_influence_radius_squared(
        dimension=p,
        marginal_lambda=2.0,
        parallel_radius=ideal_parallel,
        residual_radius=ideal_residual,
    )
    scatter_error = np.linalg.norm(
        fitted.T @ fitted / n - ideal.T @ ideal / n,
        ord=2,
    )
    scatter_bound = calibration_scatter_stability_bound(
        fitted_radius_squared=fitted_m2,
        ideal_radius_squared=ideal_m2,
        transformation_rms_error=transform_bound,
    )
    assert scatter_error <= scatter_bound + 1e-12


def test_calibration_stability_handles_degenerate_projector_and_zero_radius():
    rows = np.array([[1e-12, -1e-12], [0.0, 0.0]])
    mapped = np.vstack(
        [radial_contraction(row, radius=0.0, epsilon=1e-10) for row in rows]
    )
    assert np.array_equal(mapped, np.zeros_like(rows))
    assert calibration_transform_stability_bound(
        marginal_rms_error=0.0,
        projector_error=0.0,
        dimension=2,
        marginal_lambda=3.0,
        parallel_radius_error=0.0,
        residual_radius_error=0.0,
    ) == 0.0
    assert calibration_scatter_stability_bound(
        fitted_radius_squared=0.0,
        ideal_radius_squared=0.0,
        transformation_rms_error=0.0,
    ) == 0.0
