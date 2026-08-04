"""Deterministic paired streams with explicit contamination geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCENARIOS = (
    "clean",
    "casewise_cauchy",
    "cellwise_cauchy",
    "bounded_out_of_subspace",
    "in_subspace_leverage",
    "mixed",
)
SPECTRA = ("strong", "weak")


@dataclass(frozen=True)
class PairedStream:
    clean: np.ndarray
    observed: np.ndarray
    signal_basis: np.ndarray
    clean_estimation_covariance: np.ndarray
    contamination_mask: np.ndarray
    calibration_mask: np.ndarray
    scenario: str
    spectrum: str
    structural_direction: np.ndarray | None


def _orthogonal_direction(rng: np.random.Generator, basis: np.ndarray) -> np.ndarray:
    for _ in range(100):
        direction = rng.normal(size=basis.shape[0])
        direction -= basis @ (basis.T @ direction)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-10:
            return direction / norm
    raise RuntimeError("could not construct an out-of-subspace direction")


def _bounded_vector(
    direction: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    standardized_amplitude: float,
) -> np.ndarray:
    normalized = direction / max(float(np.max(np.abs(direction))), 1e-12)
    return location + standardized_amplitude * scale * normalized


def generate_paired_stream(
    *,
    seed: int,
    n: int,
    p: int,
    rank: int,
    calibration_size: int,
    scenario: str,
    spectrum: str,
    contamination_fraction: float = 0.10,
    cauchy_scale: float = 10.0,
    contaminate_calibration: bool = False,
    front_loaded: bool = False,
    contamination_phase: str | None = None,
) -> PairedStream:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    if spectrum not in SPECTRA:
        raise ValueError(f"unknown spectrum {spectrum}")
    if not 5 <= calibration_size < n:
        raise ValueError("calibration_size must lie in [5, n)")
    if not 0.0 <= contamination_fraction < 0.5:
        raise ValueError("contamination_fraction must lie in [0, 0.5)")
    if not 0 < rank <= min(p, n - calibration_size):
        raise ValueError("rank is incompatible with stream dimensions")

    rng = np.random.default_rng(int(seed) + (0 if spectrum == "strong" else 10_000_000))
    basis, _ = np.linalg.qr(rng.normal(size=(p, rank)), mode="reduced")
    spikes = (
        np.linspace(12.0, 4.0, rank)
        if spectrum == "strong"
        else np.linspace(2.0, 1.2, rank)
    )
    latent = rng.normal(size=(n, rank)) * np.sqrt(spikes)
    # Unit Gaussian idiosyncratic noise yields an explicit eigengap model while
    # leaving the clean reference finite and reproducible.
    clean = latent @ basis.T + rng.normal(size=(n, p))
    observed = clean.copy()
    contamination_mask = np.zeros((n, p), dtype=bool)
    calibration_mask = np.zeros(n, dtype=bool)

    if contamination_phase is None:
        contamination_phase = "both" if contaminate_calibration else "estimation"
    if contamination_phase not in {"calibration", "estimation", "both"}:
        raise ValueError(
            "contamination_phase must be calibration, estimation, or both"
        )
    if contamination_phase == "calibration":
        eligible = np.arange(0, calibration_size)
    elif contamination_phase == "estimation":
        eligible = np.arange(calibration_size, n)
    else:
        eligible = np.arange(0, n)
    eligible_start = int(eligible[0])
    eligible_stop = int(eligible[-1]) + 1
    count = int(round(contamination_fraction * len(eligible)))
    if scenario != "clean" and count == 0:
        count = 1
    if front_loaded:
        rows = eligible[:count]
    else:
        rows = np.sort(rng.choice(eligible, count, replace=False)) if count else np.array([], dtype=int)
    calibration_mask[rows[rows < calibration_size]] = True

    calibration = clean[:calibration_size]
    location = np.median(calibration, axis=0)
    scale = 1.4826 * np.median(np.abs(calibration - location), axis=0)
    scale = np.maximum(scale, np.maximum(1e-8, 1e-14 * np.abs(location)))
    estimation = clean[calibration_size:]
    clean_covariance = np.cov(estimation, rowvar=False, bias=True)
    diagonal = np.maximum(np.diag(clean_covariance), 0.0)
    inverse_std = np.zeros_like(diagonal)
    positive = diagonal > 1e-12
    inverse_std[positive] = 1.0 / np.sqrt(diagonal[positive])
    clean_correlation = (
        inverse_std[:, None] * clean_covariance * inverse_std[None, :]
    )
    _, eigenvectors = np.linalg.eigh(
        (clean_correlation + clean_correlation.T) / 2.0
    )
    # Structural geometry is defined in the same clean correlation space used
    # by the outcome, not in the latent raw-coordinate basis.  Multiplication
    # by the calibration MAD below then makes the standardized attack vector
    # exactly the declared correlation-space direction.
    target_basis = eigenvectors[:, -rank:]
    structural_direction: np.ndarray | None = None

    if scenario == "casewise_cauchy":
        faults = cauchy_scale * np.clip(
            rng.standard_cauchy(size=(len(rows), p)), -1e6, 1e6
        )
        observed[rows] = faults
        contamination_mask[rows] = True
    elif scenario == "cellwise_cauchy":
        # Ten percent of eligible cells, not ten percent of rows.
        mask = np.zeros_like(contamination_mask)
        if front_loaded:
            flat_eligible = np.arange(eligible_start * p, eligible_stop * p)
            chosen = flat_eligible[: int(round(contamination_fraction * len(flat_eligible)))]
            mask.ravel()[chosen] = True
        else:
            mask[eligible] = rng.random(size=(len(eligible), p)) < contamination_fraction
        values = cauchy_scale * np.clip(
            rng.standard_cauchy(size=int(mask.sum())), -1e6, 1e6
        )
        observed[mask] = values
        contamination_mask = mask
        calibration_mask[:calibration_size] = np.any(mask[:calibration_size], axis=1)
    elif scenario in {"bounded_out_of_subspace", "in_subspace_leverage", "mixed"}:
        if scenario == "in_subspace_leverage":
            structural_direction = target_basis[:, -1].copy()
        else:
            structural_direction = _orthogonal_direction(rng, target_basis)
        vector = _bounded_vector(
            structural_direction,
            location=location,
            scale=scale,
            standardized_amplitude=2.5,
        )
        for offset, row in enumerate(rows):
            observed[row] = location + (1.0 if offset % 2 == 0 else -1.0) * (vector - location)
        contamination_mask[rows] = True
        if scenario == "mixed":
            # Half the structural rows are additionally sparse Cauchy faults.
            mixed_rows = rows[::2]
            sparse_mask = rng.random(size=(len(mixed_rows), p)) < 0.10
            faults = cauchy_scale * np.clip(
                rng.standard_cauchy(size=int(sparse_mask.sum())), -1e6, 1e6
            )
            block = observed[mixed_rows].copy()
            block[sparse_mask] = faults
            observed[mixed_rows] = block
            contamination_mask[mixed_rows] |= sparse_mask

    return PairedStream(
        clean=clean,
        observed=observed,
        signal_basis=target_basis,
        clean_estimation_covariance=clean_covariance,
        contamination_mask=contamination_mask,
        calibration_mask=calibration_mask,
        scenario=scenario,
        spectrum=spectrum,
        structural_direction=structural_direction,
    )


def scheduled_epoch_stream(
    *, seed: int, n: int, p: int, rank: int, change_points: tuple[int, int]
) -> dict[str, np.ndarray]:
    if len(change_points) != 2 or not 0 < change_points[0] < change_points[1] < n:
        raise ValueError("two ordered change points are required")
    rng = np.random.default_rng(seed)
    first, _ = np.linalg.qr(rng.normal(size=(p, rank)), mode="reduced")
    second, _ = np.linalg.qr(rng.normal(size=(p, rank)), mode="reduced")
    spikes = np.linspace(10.0, 2.0, rank)
    result = np.empty((n, p), dtype=float)
    bases = []
    for start, stop, basis, scale in (
        (0, change_points[0], first, 1.0),
        (change_points[0], change_points[1], first, 2.0),
        (change_points[1], n, second, 1.0),
    ):
        latent = rng.normal(size=(stop - start, rank)) * np.sqrt(spikes)
        result[start:stop] = scale * (latent @ basis.T + rng.normal(size=(stop - start, p)))
        bases.append(basis)
    return {"matrix": result, "basis_scale_1": first, "basis_scale_2": second}
