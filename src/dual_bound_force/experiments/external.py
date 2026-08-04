"""Checksum-validated PBMC and KDD data access for the standalone study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .reporting import sha256_file


PBMC_PAGE = (
    "https://www.10xgenomics.com/datasets/"
    "fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0"
)
PBMC_ARCHIVE = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
KDD_SOURCE = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"


class ExternalDataUnavailable(RuntimeError):
    """Raised instead of silently substituting generated data."""


@dataclass(frozen=True)
class PreparedPBMC:
    matrix: sparse.csr_matrix
    gene_ids: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray
    provenance: dict[str, Any]
    preprocessing: dict[str, Any]


@dataclass(frozen=True)
class PreparedKDD:
    data: np.ndarray
    labels: np.ndarray
    provenance: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalDataUnavailable(f"unreadable sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalDataUnavailable(f"sidecar {path} is not an object")
    return value


def _require_utc_timestamp(value: Any, *, field: str) -> None:
    if not isinstance(value, str):
        raise ExternalDataUnavailable(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExternalDataUnavailable(
            f"{field} must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ExternalDataUnavailable(f"{field} must include the UTC offset")


def _sha256_strings(values: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_pbmc(data_dir: str | Path) -> PreparedPBMC:
    root = Path(data_dir) / "pbmc68k"
    archive = root / "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
    raw_sidecar = root / "raw.metadata.json"
    matrix_path = root / "log1p-library1e4-top10000.npz"
    feature_path = root / "log1p-library1e4-top10000.features.npz"
    sidecar = root / "log1p-library1e4-top10000.metadata.json"
    required_files = (archive, raw_sidecar, matrix_path, feature_path, sidecar)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ExternalDataUnavailable(
            "validated PBMC cache is incomplete: " + ", ".join(missing)
        )
    raw = _json(raw_sidecar)
    metadata = _json(sidecar)
    required_raw = {
        "archive_sha256",
        "dataset_page_url",
        "source_url",
        "retrieved_at_utc",
        "validated_at_utc",
        "matrix_shape",
    }
    required_prepared = {
        "source_archive_sha256",
        "matrix_sha256",
        "features_sha256",
        "selected_gene_sha256",
        "shape",
        "preprocessing",
        "validated_at_utc",
    }
    if not required_raw <= raw.keys() or not required_prepared <= metadata.keys():
        raise ExternalDataUnavailable("PBMC provenance sidecars are incomplete")
    if raw["dataset_page_url"] != PBMC_PAGE or raw["source_url"] != PBMC_ARCHIVE:
        raise ExternalDataUnavailable("PBMC source identity mismatch")
    if raw["matrix_shape"] != [32_738, 68_579]:
        raise ExternalDataUnavailable("PBMC raw schema mismatch")
    _require_utc_timestamp(raw["retrieved_at_utc"], field="PBMC retrieval time")
    _require_utc_timestamp(raw["validated_at_utc"], field="PBMC raw validation time")
    _require_utc_timestamp(
        metadata["validated_at_utc"], field="PBMC prepared validation time"
    )
    archive_hash = sha256_file(archive)
    if raw["archive_sha256"] != archive_hash:
        raise ExternalDataUnavailable("PBMC archive checksum mismatch")
    if metadata["source_archive_sha256"] != archive_hash:
        raise ExternalDataUnavailable("PBMC prepared source checksum mismatch")
    if metadata["matrix_sha256"] != sha256_file(matrix_path):
        raise ExternalDataUnavailable("PBMC matrix cache checksum mismatch")
    if metadata["features_sha256"] != sha256_file(feature_path):
        raise ExternalDataUnavailable("PBMC feature cache checksum mismatch")
    expected_preprocessing = {
        "orientation": "cells_by_genes",
        "library_size_target": 10_000.0,
        "transform": "log1p",
        "selection": "top_dispersion_nonconstant_genes",
        "top_genes": 10_000,
        "tie_break": "ascending_gene_identifier",
    }
    if metadata["preprocessing"] != expected_preprocessing:
        raise ExternalDataUnavailable("PBMC preprocessing mismatch")
    try:
        matrix = sparse.load_npz(matrix_path).tocsr()
        with np.load(feature_path, allow_pickle=False) as values:
            gene_ids = np.asarray(values["gene_ids"], dtype=str)
            means = np.asarray(values["feature_means"], dtype=float)
            scales = np.asarray(values["feature_scales"], dtype=float)
    except Exception as exc:
        raise ExternalDataUnavailable(f"PBMC cache load failed: {exc}") from exc
    if matrix.shape != (68_579, 10_000) or list(matrix.shape) != metadata["shape"]:
        raise ExternalDataUnavailable(f"PBMC schema mismatch: {matrix.shape}")
    if len(gene_ids) != matrix.shape[1] or _sha256_strings(gene_ids) != metadata["selected_gene_sha256"]:
        raise ExternalDataUnavailable("PBMC selected-gene identity mismatch")
    if (
        means.shape != (matrix.shape[1],)
        or scales.shape != (matrix.shape[1],)
        or not np.isfinite(matrix.data).all()
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ExternalDataUnavailable("PBMC numerical schema is invalid")
    return PreparedPBMC(
        matrix=matrix,
        gene_ids=gene_ids,
        feature_means=means,
        feature_scales=scales,
        provenance={"raw": raw, "prepared": metadata},
        preprocessing=expected_preprocessing,
    )


def load_kdd(data_dir: str | Path, *, cache_seed: int = 0) -> PreparedKDD:
    root = Path(data_dir) / "kdd-real"
    stem = f"kddcup99-sa-percent10-seed{int(cache_seed)}"
    data_path = root / f"{stem}.npz"
    sidecar = root / f"{stem}.metadata.json"
    if not data_path.is_file() or not sidecar.is_file():
        raise ExternalDataUnavailable(f"validated KDD cache is missing for seed {cache_seed}")
    metadata = _json(sidecar)
    required = {
        "continuous_columns",
        "fetch_parameters",
        "label_counts",
        "retrieved_at_utc",
        "seed",
        "sha256",
        "shape",
        "source_url",
    }
    if not required <= metadata.keys():
        raise ExternalDataUnavailable("KDD sidecar is incomplete")
    if metadata["source_url"] != KDD_SOURCE:
        raise ExternalDataUnavailable("KDD source identity mismatch")
    _require_utc_timestamp(metadata["retrieved_at_utc"], field="KDD retrieval time")
    if metadata["seed"] != int(cache_seed):
        raise ExternalDataUnavailable("KDD cache seed mismatch")
    if metadata["sha256"] != sha256_file(data_path):
        raise ExternalDataUnavailable("KDD cache checksum mismatch")
    if len(metadata["continuous_columns"]) != 37 or metadata["shape"] != [100_655, 37]:
        raise ExternalDataUnavailable("KDD sidecar schema mismatch")
    try:
        with np.load(data_path, allow_pickle=False) as values:
            data = np.asarray(values["data"], dtype=float)
            labels = np.asarray(values["labels"], dtype=np.uint8)
    except Exception as exc:
        raise ExternalDataUnavailable(f"KDD cache load failed: {exc}") from exc
    if data.shape != (100_655, 37) or labels.shape != (100_655,):
        raise ExternalDataUnavailable("KDD array schema mismatch")
    unique, counts = np.unique(labels, return_counts=True)
    observed_counts = {int(key): int(value) for key, value in zip(unique, counts)}
    if observed_counts != {0: 97_278, 1: 3_377} or not np.isfinite(data).all():
        raise ExternalDataUnavailable("KDD labels or numerical values are invalid")
    return PreparedKDD(data=data, labels=labels, provenance=metadata)


def centered_scaled_randomized_svd(
    matrix: sparse.csr_matrix,
    *,
    column_scale: np.ndarray,
    rank: int,
    seed: int,
    power_iterations: int = 5,
    oversamples: int = 10,
    convergence_tolerance: float | None = None,
    max_power_iterations: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deterministic randomized SVD of implicitly centered/scaled sparse data."""

    scales = np.asarray(column_scale, dtype=float)
    if scales.shape != (matrix.shape[1],) or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("column_scale must be finite and positive")
    if not 0 < rank <= min(matrix.shape):
        raise ValueError("rank is incompatible with the matrix")
    width = min(rank + oversamples, min(matrix.shape))
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    inverse_scale = 1.0 / scales
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(matrix.shape[1], width))

    def right(value: np.ndarray) -> np.ndarray:
        scaled_value = inverse_scale[:, None] * value
        return np.asarray(matrix @ scaled_value) - np.outer(
            np.ones(matrix.shape[0]), (mean * inverse_scale) @ value
        )

    def transpose(value: np.ndarray) -> np.ndarray:
        raw = np.asarray(matrix.T @ value) - np.outer(mean, np.sum(value, axis=0))
        return inverse_scale[:, None] * raw

    if power_iterations < 0 or oversamples < 0:
        raise ValueError("iteration and oversampling counts must be nonnegative")
    maximum = power_iterations if max_power_iterations is None else int(max_power_iterations)
    if maximum < power_iterations:
        raise ValueError("max_power_iterations must not be below power_iterations")
    if convergence_tolerance is not None and (
        not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0
    ):
        raise ValueError("convergence_tolerance must be finite and positive")

    sample = right(omega)
    previous_basis: np.ndarray | None = None
    change = float("inf")
    converged = False
    basis = np.empty((matrix.shape[1], rank), dtype=float)
    singular_values = np.empty(rank, dtype=float)
    actual_power = 0
    for power in range(maximum + 1):
        left, _ = np.linalg.qr(sample, mode="reduced")
        compressed = transpose(left).T
        _, singular_values_all, right_vectors = np.linalg.svd(
            compressed, full_matrices=False
        )
        basis = right_vectors[:rank].T
        singular_values = singular_values_all[:rank]
        if previous_basis is not None:
            overlap = np.linalg.svd(previous_basis.T @ basis, compute_uv=False)
            change = float(
                np.linalg.norm(
                    np.sqrt(
                        np.maximum(
                            1.0 - np.clip(overlap, 0.0, 1.0) ** 2, 0.0
                        )
                    )
                )
                / np.sqrt(rank)
            )
        actual_power = power
        if (
            convergence_tolerance is not None
            and power >= power_iterations
            and change <= convergence_tolerance
        ):
            converged = True
            break
        if power == maximum:
            converged = convergence_tolerance is None
            break
        previous_basis = basis
        right_basis, _ = np.linalg.qr(transpose(left), mode="reduced")
        sample = right(right_basis)
    return basis, singular_values[:rank], {
        "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(rank))),
        "last_iteration_subspace_change": change,
        "power_iterations": float(actual_power),
        "oversamples": float(oversamples),
        "convergence_tolerance": convergence_tolerance,
        "converged": converged,
    }


def load_or_build_pbmc_reference(
    data_dir: str | Path,
    prepared: PreparedPBMC,
    *,
    rank: int = 20,
    seed: int = 49_001,
    minimum_power_iterations: int = 5,
    maximum_power_iterations: int = 30,
    convergence_tolerance: float = 0.01,
    oversamples: int = 10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or deterministically build a checksummed converged PBMC reference."""

    root = Path(data_dir) / "pbmc68k"
    stem = f"reference-rank{rank}-seed{seed}"
    array_path = root / f"{stem}.npz"
    sidecar_path = root / f"{stem}.metadata.json"
    source = prepared.provenance["prepared"]
    parameters = {
        "rank": rank,
        "seed": seed,
        "minimum_power_iterations": minimum_power_iterations,
        "maximum_power_iterations": maximum_power_iterations,
        "convergence_tolerance": convergence_tolerance,
        "oversamples": oversamples,
        "centering": "implicit_column_mean",
        "scaling": "validated_prepared_feature_scales",
    }
    if array_path.is_file() and sidecar_path.is_file():
        metadata = _json(sidecar_path)
        required = {
            "array_sha256", "source_matrix_sha256", "source_features_sha256",
            "selected_gene_sha256", "parameters", "diagnostics",
            "created_at_utc", "validated_at_utc",
        }
        if not required <= metadata.keys():
            raise ExternalDataUnavailable("PBMC reference sidecar is incomplete")
        _require_utc_timestamp(
            metadata["created_at_utc"], field="PBMC reference creation time"
        )
        _require_utc_timestamp(
            metadata["validated_at_utc"], field="PBMC reference validation time"
        )
        if metadata["array_sha256"] != sha256_file(array_path):
            raise ExternalDataUnavailable("PBMC reference checksum mismatch")
        if (
            metadata["source_matrix_sha256"] != source["matrix_sha256"]
            or metadata["source_features_sha256"] != source["features_sha256"]
            or metadata["selected_gene_sha256"] != source["selected_gene_sha256"]
            or metadata["parameters"] != parameters
        ):
            raise ExternalDataUnavailable("PBMC reference provenance mismatch")
        try:
            with np.load(array_path, allow_pickle=False) as values:
                basis = np.asarray(values["basis"], dtype=float)
        except Exception as exc:
            raise ExternalDataUnavailable(f"PBMC reference load failed: {exc}") from exc
        diagnostics = metadata["diagnostics"]
        if (
            basis.shape != (prepared.matrix.shape[1], rank)
            or not np.isfinite(basis).all()
            or not diagnostics.get("converged")
            or diagnostics["last_iteration_subspace_change"] > convergence_tolerance
            or diagnostics["orthogonality_error"] > 1e-8
        ):
            raise ExternalDataUnavailable("PBMC reference failed convergence/schema checks")
        return basis, metadata

    basis, singular_values, diagnostics = centered_scaled_randomized_svd(
        prepared.matrix,
        column_scale=prepared.feature_scales,
        rank=rank,
        seed=seed,
        power_iterations=minimum_power_iterations,
        max_power_iterations=maximum_power_iterations,
        convergence_tolerance=convergence_tolerance,
        oversamples=oversamples,
    )
    if (
        not diagnostics["converged"]
        or diagnostics["last_iteration_subspace_change"] > convergence_tolerance
        or diagnostics["orthogonality_error"] > 1e-8
    ):
        raise ExternalDataUnavailable(
            "PBMC reference did not satisfy the preregistered convergence check"
        )
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, basis=basis, singular_values=singular_values)
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "array_sha256": sha256_file(array_path),
        "source_matrix_sha256": source["matrix_sha256"],
        "source_features_sha256": source["features_sha256"],
        "selected_gene_sha256": source["selected_gene_sha256"],
        "parameters": parameters,
        "diagnostics": diagnostics,
        "created_at_utc": now,
        "validated_at_utc": now,
    }
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return basis, metadata


def _rayleigh_ritz_certificate(
    matrix: sparse.csr_matrix,
    *,
    column_scale: np.ndarray,
    candidate_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Refine a right subspace and measure its two-sided Ritz residual.

    The input matrix is centered and column-scaled implicitly, exactly as in
    :func:`centered_scaled_randomized_svd`.  The returned residual is
    ``||A.T @ U - V @ diag(s)||_2`` after a Rayleigh--Ritz rotation of the
    candidate right subspace.  Unlike consecutive-iterate movement, this is
    an equation residual for the approximate singular triplets.
    """

    scales = np.asarray(column_scale, dtype=float)
    basis = np.asarray(candidate_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != matrix.shape[1]:
        raise ValueError("candidate_basis has an incompatible shape")
    if scales.shape != (matrix.shape[1],) or np.any(scales <= 0):
        raise ValueError("column_scale must be positive and dimension-compatible")
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    inverse_scale = 1.0 / scales

    scaled_basis = inverse_scale[:, None] * basis
    projected = np.asarray(matrix @ scaled_basis) - np.outer(
        np.ones(matrix.shape[0]), (mean * inverse_scale) @ basis
    )
    left, singular_values, rotation = np.linalg.svd(projected, full_matrices=False)
    refined_basis = basis @ rotation.T
    transpose_projected = inverse_scale[:, None] * (
        np.asarray(matrix.T @ left) - np.outer(mean, np.sum(left, axis=0))
    )
    residual = transpose_projected - refined_basis * singular_values
    residual_spectral = float(np.linalg.norm(residual, ord=2))
    residual_frobenius = float(np.linalg.norm(residual))
    denominator = max(float(singular_values[-2]), np.finfo(float).tiny)
    return refined_basis, singular_values, {
        "ritz_residual_spectral_norm": residual_spectral,
        "ritz_residual_frobenius_norm": residual_frobenius,
        "relative_ritz_residual": residual_spectral / denominator,
        "rayleigh_ritz_orthogonality_error": float(
            np.linalg.norm(refined_basis.T @ refined_basis - np.eye(basis.shape[1]))
        ),
    }


def load_or_build_pbmc_certified_reference(
    data_dir: str | Path,
    prepared: PreparedPBMC,
    *,
    rank: int = 20,
    seeds: tuple[int, int] = (49_001, 49_002),
    minimum_power_iterations: int = 20,
    maximum_power_iterations: int = 160,
    convergence_tolerance: float = 1e-6,
    oversamples: int = 20,
    cross_start_tolerance: float = 1e-4,
    relative_residual_tolerance: float = 1e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or build the independently cross-checked PBMC rank reference.

    Two deterministic block-power starts estimate ``rank + 1`` singular
    vectors.  Certification requires strict iterate convergence for both
    starts, a small cross-start projector distance, a small two-sided Ritz
    residual, and a positive estimated rank/next-rank singular-value gap.
    The earlier loose reference cache is deliberately left untouched.
    """

    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError("certification requires two distinct deterministic seeds")
    if rank >= min(prepared.matrix.shape):
        raise ValueError("rank must leave room for a rank+1 gap diagnostic")
    root = Path(data_dir) / "pbmc68k"
    stem = f"reference-certified-v2-rank{rank}-seeds{seeds[0]}-{seeds[1]}"
    array_path = root / f"{stem}.npz"
    sidecar_path = root / f"{stem}.metadata.json"
    source = prepared.provenance["prepared"]
    parameters = {
        "algorithm": "two_start_block_power_rayleigh_ritz_v2",
        "rank": rank,
        "audit_rank": rank + 1,
        "seeds": list(seeds),
        "minimum_power_iterations": minimum_power_iterations,
        "maximum_power_iterations": maximum_power_iterations,
        "convergence_tolerance": convergence_tolerance,
        "oversamples": oversamples,
        "cross_start_tolerance": cross_start_tolerance,
        "relative_residual_tolerance": relative_residual_tolerance,
        "centering": "implicit_column_mean",
        "scaling": "validated_prepared_feature_scales",
    }

    def validate(
        basis: np.ndarray,
        alternate_basis: np.ndarray,
        singular_values: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        diagnostics = metadata["diagnostics"]
        if (
            basis.shape != (prepared.matrix.shape[1], rank)
            or alternate_basis.shape != basis.shape
            or singular_values.shape != (rank + 1,)
            or not np.isfinite(basis).all()
            or not np.isfinite(alternate_basis).all()
            or not np.isfinite(singular_values).all()
            or any(not item.get("converged") for item in diagnostics["starts"])
            or any(
                item["last_iteration_subspace_change"] > convergence_tolerance
                for item in diagnostics["starts"]
            )
            or diagnostics["cross_start_normalized_subspace_change"]
            > cross_start_tolerance
            or diagnostics["maximum_relative_ritz_residual"]
            > relative_residual_tolerance
            or diagnostics["estimated_rank_gap"] <= 0.0
            or diagnostics["maximum_orthogonality_error"] > 1e-8
        ):
            raise ExternalDataUnavailable(
                "PBMC certified reference failed convergence or residual checks"
            )

    if array_path.is_file() and sidecar_path.is_file():
        metadata = _json(sidecar_path)
        required = {
            "array_sha256",
            "source_matrix_sha256",
            "source_features_sha256",
            "selected_gene_sha256",
            "parameters",
            "diagnostics",
            "created_at_utc",
            "validated_at_utc",
        }
        if not required <= metadata.keys():
            raise ExternalDataUnavailable("PBMC certified-reference sidecar is incomplete")
        _require_utc_timestamp(
            metadata["created_at_utc"], field="PBMC certified-reference creation time"
        )
        _require_utc_timestamp(
            metadata["validated_at_utc"], field="PBMC certified-reference validation time"
        )
        if metadata["array_sha256"] != sha256_file(array_path):
            raise ExternalDataUnavailable("PBMC certified-reference checksum mismatch")
        if (
            metadata["source_matrix_sha256"] != source["matrix_sha256"]
            or metadata["source_features_sha256"] != source["features_sha256"]
            or metadata["selected_gene_sha256"] != source["selected_gene_sha256"]
            or metadata["parameters"] != parameters
        ):
            raise ExternalDataUnavailable("PBMC certified-reference provenance mismatch")
        try:
            with np.load(array_path, allow_pickle=False) as values:
                basis = np.asarray(values["basis"], dtype=float)
                alternate_basis = np.asarray(values["alternate_basis"], dtype=float)
                singular_values = np.asarray(values["singular_values"], dtype=float)
        except Exception as exc:
            raise ExternalDataUnavailable(
                f"PBMC certified-reference load failed: {exc}"
            ) from exc
        validate(basis, alternate_basis, singular_values, metadata)
        return basis, metadata

    start_bases: list[np.ndarray] = []
    start_singular_values: list[np.ndarray] = []
    start_diagnostics: list[dict[str, Any]] = []
    for seed in seeds:
        candidate, _, iterative = centered_scaled_randomized_svd(
            prepared.matrix,
            column_scale=prepared.feature_scales,
            rank=rank + 1,
            seed=seed,
            power_iterations=minimum_power_iterations,
            max_power_iterations=maximum_power_iterations,
            convergence_tolerance=convergence_tolerance,
            oversamples=oversamples,
        )
        refined, singular_values, residual = _rayleigh_ritz_certificate(
            prepared.matrix,
            column_scale=prepared.feature_scales,
            candidate_basis=candidate,
        )
        start_bases.append(refined[:, :rank])
        start_singular_values.append(singular_values)
        start_diagnostics.append({"seed": seed, **iterative, **residual})

    overlap = np.linalg.svd(start_bases[0].T @ start_bases[1], compute_uv=False)
    cross_start_change = float(
        np.linalg.norm(
            np.sqrt(np.maximum(1.0 - np.clip(overlap, 0.0, 1.0) ** 2, 0.0))
        )
        / np.sqrt(rank)
    )
    singular_values = start_singular_values[0]
    diagnostics = {
        "starts": start_diagnostics,
        "cross_start_normalized_subspace_change": cross_start_change,
        "maximum_principal_sine_between_starts": float(
            np.sqrt(max(0.0, 1.0 - float(np.min(overlap)) ** 2))
        ),
        "estimated_rank_singular_value": float(singular_values[rank - 1]),
        "estimated_next_singular_value": float(singular_values[rank]),
        "estimated_rank_gap": float(
            singular_values[rank - 1] - singular_values[rank]
        ),
        "maximum_relative_ritz_residual": float(
            max(item["relative_ritz_residual"] for item in start_diagnostics)
        ),
        "maximum_orthogonality_error": float(
            max(
                item["rayleigh_ritz_orthogonality_error"]
                for item in start_diagnostics
            )
        ),
        "certified": True,
    }
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "source_matrix_sha256": source["matrix_sha256"],
        "source_features_sha256": source["features_sha256"],
        "selected_gene_sha256": source["selected_gene_sha256"],
        "parameters": parameters,
        "diagnostics": diagnostics,
        "created_at_utc": now,
        "validated_at_utc": now,
    }
    validate(
        start_bases[0], start_bases[1], singular_values, metadata
    )
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        array_path,
        basis=start_bases[0],
        alternate_basis=start_bases[1],
        singular_values=singular_values,
    )
    metadata["array_sha256"] = sha256_file(array_path)
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return start_bases[0], metadata
