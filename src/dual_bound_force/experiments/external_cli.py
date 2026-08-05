"""Run checksum-validated PBMC and KDD tiers without synthetic fallback."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import traceback
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from dual_bound_force import DualBoundFORCE
from .baseline_registry import sketch_force_estimator_class
from .native_baselines import (
    FrequentDirectionsCovariance,
    RobustFrequentDirectionsCovariance,
)

from .external import (
    ExternalDataUnavailable,
    load_or_build_pbmc_certified_reference,
    load_kdd,
    load_pbmc,
)
from .baselines import _state_bytes
from .metrics import normalized_projection_error, paired_bootstrap_ci, rank_auc
from .reporting import envelope, sha256_file, write_bundle
from .study import PBMC_SEEDS, load_frozen_configuration


def _parse_seeds(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _basis_from_sketch(estimator: Any, rank: int) -> np.ndarray:
    if isinstance(estimator, DualBoundFORCE):
        return estimator.get_subspace(rank)
    sketch = np.asarray(estimator.B, dtype=float)
    n = int(estimator.n)
    alpha = float(getattr(estimator, "alpha", 0.0))
    diagonal = (np.sum(sketch * sketch, axis=0) + alpha) / max(n, 1)
    positive = diagonal > 1e-12
    inverse = np.zeros_like(diagonal)
    inverse[positive] = 1.0 / np.sqrt(diagonal[positive])
    if alpha == 0.0:
        normalized = sketch * inverse
        _, singular_values, right_vectors = np.linalg.svd(normalized, full_matrices=False)
        used = min(rank, int(np.sum(singular_values > 1e-10)))
        return right_vectors[:used].T

    def multiply(vector: np.ndarray) -> np.ndarray:
        scaled = inverse * vector
        return inverse * (sketch.T @ (sketch @ scaled) + alpha * scaled) / max(n, 1)

    operator = LinearOperator((len(diagonal), len(diagonal)), matvec=multiply, dtype=float)
    _, vectors = eigsh(operator, k=rank, which="LA", tol=1e-6, maxiter=2000)
    return vectors


def _pbmc_row(
    clean: np.ndarray,
    *,
    row_index: int,
    seed: int,
    scenario: str,
    attacked_rows: set[int],
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    structural_location: np.ndarray,
    structural_scale: np.ndarray,
    structural_direction: np.ndarray,
) -> np.ndarray:
    if scenario == "clean":
        return clean
    if scenario == "cellwise":
        if row_index not in attacked_rows:
            return clean
    elif row_index not in attacked_rows:
        return clean
    rng = np.random.default_rng(np.random.SeedSequence([seed, row_index, 0xDBF]))
    if scenario == "casewise":
        return feature_means + 10.0 * feature_scales * np.clip(rng.standard_cauchy(len(clean)), -1e4, 1e4)
    if scenario == "cellwise":
        mask = rng.random(len(clean)) < 0.01
        value = clean.copy()
        value[mask] += 10.0 * feature_scales[mask] * np.clip(
            rng.standard_cauchy(int(mask.sum())), -1e4, 1e4
        )
        return value
    sign = 1.0 if row_index % 2 == 0 else -1.0
    normalized = structural_direction / max(float(np.max(np.abs(structural_direction))), 1e-12)
    return structural_location + sign * 2.5 * structural_scale * normalized


def _fit_pbmc_method(
    *,
    method: str,
    matrix,
    means: np.ndarray,
    scales: np.ndarray,
    reference_basis: np.ndarray,
    scenario: str,
    seed: int,
    k: int,
    calibration_size: int,
    parallel_lambda: float,
    residual_lambda: float,
    baseline_dir: str | None = None,
    basis_output_dir: Path | None = None,
) -> dict[str, Any]:
    import time

    p = matrix.shape[1]
    rank = reference_basis.shape[1]
    rng = np.random.default_rng(seed)
    eligible = np.arange(calibration_size, matrix.shape[0])
    fraction = 0.01 if scenario == "cellwise" else 0.10
    count = max(1, int(round(fraction * len(eligible)))) if scenario != "clean" else 0
    attacked_rows = (
        set(int(value) for value in eligible)
        if scenario == "cellwise"
        else set(int(value) for value in rng.choice(eligible, count, replace=False))
    )
    if scenario == "in_subspace":
        structural = reference_basis[:, 0].copy()
    else:
        structural = rng.normal(size=p)
        structural -= reference_basis @ (reference_basis.T @ structural)
        structural /= max(float(np.linalg.norm(structural)), 1e-12)
    calibration = matrix[:calibration_size].toarray()
    structural_location = np.median(calibration, axis=0)
    structural_scale = np.maximum(
        1.4826 * np.median(np.abs(calibration - structural_location), axis=0),
        np.maximum(1e-8, 1e-14 * np.abs(structural_location)),
    )
    if method == "vanilla_fd":
        estimator = FrequentDirectionsCovariance(p, k)
    elif method == "rfd":
        estimator = RobustFrequentDirectionsCovariance(p, k)
    elif method == "sketch_mad":
        SketchFORCE = sketch_force_estimator_class(baseline_dir)
        estimator = SketchFORCE(p, k, lam=3.0, trim_mode="mad")
    elif method == "dual_bound":
        estimator = DualBoundFORCE(
            p,
            k,
            calibration_size=calibration_size,
            parallel_lambda=parallel_lambda,
            residual_lambda=residual_lambda,
        )
    else:
        raise ValueError(f"unknown PBMC method {method}")
    start_rss = 0
    try:
        import psutil

        process = psutil.Process()
        start_rss = int(process.memory_info().rss)
    except ImportError:
        process = None
    started = time.perf_counter_ns()
    for row_index in range(matrix.shape[0]):
        row = matrix.getrow(row_index).toarray().ravel()
        observed = _pbmc_row(
            row,
            row_index=row_index,
            seed=seed,
            scenario=scenario,
            attacked_rows=attacked_rows,
            feature_means=means,
            feature_scales=scales,
            structural_location=structural_location,
            structural_scale=structural_scale,
            structural_direction=structural,
        )
        estimator.update(observed)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    basis = _basis_from_sketch(estimator, rank)
    if basis.shape != reference_basis.shape:
        raise RuntimeError("PBMC estimator returned a rank-deficient basis")
    end_rss = int(process.memory_info().rss) if process is not None else 0
    try:
        import resource

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (ImportError, AttributeError):
        peak_rss = end_rss
    state = _state_bytes(estimator)
    record = {
        "status": "completed",
        "tier": "pbmc",
        "seed": int(seed),
        "scenario": scenario,
        "method": method,
        "n": int(matrix.shape[0]),
        "p": int(p),
        "rank": int(rank),
        "k": int(k),
        "subspace_error": normalized_projection_error(reference_basis, basis),
        "elapsed_seconds": float(elapsed),
        "throughput_rows_per_second": float(matrix.shape[0] / max(elapsed, 1e-12)),
        "state_bytes": state,
        "calibration_peak_state_bytes": (
            int(estimator.calibration_peak_state_bytes_bound)
            if method == "dual_bound"
            else 0
        ),
        "output_bytes": int(basis.nbytes),
        "incremental_rss_bytes": int(max(0, end_rss - start_rss)),
        "peak_rss_bytes": peak_rss,
        "attacked_rows": len(attacked_rows),
    }
    if basis_output_dir is not None:
        basis_output_dir.mkdir(parents=True, exist_ok=True)
        basis_path = basis_output_dir / (
            f"seed{seed}-{scenario}-{method}.basis.npz"
        )
        np.savez_compressed(basis_path, basis=basis)
        record.update(
            {
                "basis_path": str(basis_path.resolve()),
                "basis_sha256": sha256_file(basis_path),
            }
        )
    return record


def run_pbmc(args: argparse.Namespace, configuration: dict[str, Any]) -> tuple[list[dict], dict, dict]:
    prepared = load_pbmc(args.data_dir)
    seeds = _parse_seeds(args.seeds, (500,) if args.smoke else PBMC_SEEDS)
    if args.smoke:
        matrix = prepared.matrix[:800, :200].tocsr()
        means = prepared.feature_means[:200]
        scales = prepared.feature_scales[:200]
        centered = matrix.toarray() - np.asarray(matrix.mean(axis=0)).ravel()
        standardized = centered / scales
        _, _, right = np.linalg.svd(standardized, full_matrices=False)
        reference_basis = right[:5].T
        reference_diagnostics = {"mode": "dense_smoke_fixture"}
        default_methods = ("vanilla_fd", "sketch_mad", "dual_bound")
        scenarios = ("clean", "bounded_out_of_subspace")
        k = 10
        calibration_size = 64
    else:
        matrix = prepared.matrix
        means = prepared.feature_means
        scales = prepared.feature_scales
        reference_basis, reference_metadata = load_or_build_pbmc_certified_reference(
            args.data_dir, prepared
        )
        reference_diagnostics = {
            "mode": "checksummed_two_start_rayleigh_ritz_certified_cache",
            **reference_metadata,
        }
        default_methods = ("vanilla_fd", "rfd", "sketch_mad", "dual_bound")
        default_scenarios = ("clean", "casewise", "cellwise", "bounded_out_of_subspace", "in_subspace")
        scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip()) if args.scenarios else default_scenarios
        if not scenarios or not set(scenarios) <= set(default_scenarios):
            raise ValueError(f"PBMC scenarios must be drawn from {default_scenarios}")
        k = 50
        calibration_size = 512
    methods = (
        tuple(item.strip() for item in args.methods.split(",") if item.strip())
        if args.methods
        else default_methods
    )
    if not methods or not set(methods) <= set(default_methods):
        raise ValueError(f"PBMC methods must be drawn from {default_methods}")
    records = []
    for seed in seeds:
        for scenario in scenarios:
            for method in methods:
                records.append(
                    _fit_pbmc_method(
                        method=method,
                        matrix=matrix,
                        means=means,
                        scales=scales,
                        reference_basis=reference_basis,
                        scenario=scenario,
                        seed=seed,
                        k=k,
                        calibration_size=calibration_size,
                        parallel_lambda=float(configuration["parallel_lambda"]),
                        residual_lambda=float(configuration["residual_lambda"]),
                        baseline_dir=args.baseline_dir,
                        basis_output_dir=Path(args.output_dir) / "bases",
                    )
                )
    return records, prepared.provenance, {
        **prepared.preprocessing,
        "reference_svd": reference_diagnostics,
        "selected_gene_count": int(matrix.shape[1]),
    }


def run_kdd(args: argparse.Namespace, configuration: dict[str, Any]) -> tuple[list[dict], dict, dict]:
    prepared = load_kdd(args.data_dir, cache_seed=0)
    seeds = _parse_seeds(args.seeds, (500,) if args.smoke else tuple(range(500, 530)))
    records = []
    normals = np.flatnonzero(prepared.labels == 0)
    anomalies = np.flatnonzero(prepared.labels == 1)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        normal_order = rng.permutation(normals)
        anomaly_order = rng.permutation(anomalies)
        train_count = 1200 if args.smoke else 20_000
        query_count = 100 if args.smoke else 500
        training = prepared.data[normal_order[:train_count]]
        nominal = prepared.data[normal_order[train_count : train_count + query_count]].copy()
        anomalous = prepared.data[anomaly_order[:query_count]].copy()
        training_scale = np.maximum(np.std(training, axis=0), 1e-8)
        for query_block in (nominal, anomalous):
            fault_mask = rng.random(size=query_block.shape) < 0.05
            query_block[fault_mask] += 10.0 * np.broadcast_to(
                training_scale, query_block.shape
            )[fault_mask] * np.clip(
                rng.standard_cauchy(int(fault_mask.sum())), -1e4, 1e4
            )
        estimator = DualBoundFORCE(
            training.shape[1],
            min(10, training.shape[1]),
            calibration_size=64 if args.smoke else 512,
            parallel_lambda=float(configuration["parallel_lambda"]),
            residual_lambda=float(configuration["residual_lambda"]),
        ).fit(training)
        serving = estimator._serving()
        transformed_scatter = serving.sketch.gram(serving.effective_n)
        precision = np.linalg.pinv(
            transformed_scatter
            + 1e-3 * np.eye(transformed_scatter.shape[0])
        )

        def scores(block: np.ndarray, aligned: bool) -> np.ndarray:
            if aligned:
                vectors = np.vstack([estimator.transform(row) for row in block])
            else:
                vectors = (block - serving.location) / serving.scale
            return np.einsum("ij,jk,ik->i", vectors, precision, vectors)

        raw_auc = rank_auc(scores(nominal, False), scores(anomalous, False))
        aligned_auc = rank_auc(scores(nominal, True), scores(anomalous, True))
        records.append(
            {
                "status": "completed",
                "tier": "kdd",
                "seed": int(seed),
                "method": "dual_bound",
                "training_rows": int(train_count),
                "query_rows_per_class": int(query_count),
                "raw_auc": raw_auc,
                "aligned_auc": aligned_auc,
                "aligned_minus_raw_auc": float(aligned_auc - raw_auc),
                "state_bytes": estimator.state_bytes,
            }
        )
    return records, prepared.provenance, {
        "continuous_features": 37,
        "split": "seeded disjoint normal training/nominal query and anomaly query",
        "query_faults": "5% feature-scaled Cauchy cells in both classes",
        "score": (
            "regularized transformed-scatter pseudoinverse quadratic form; "
            "raw queries use frozen marginal standardization and aligned queries "
            "use the complete frozen Dual-Bound transformation"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pbmc", "kdd"), required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--methods", help="PBMC method selector")
    parser.add_argument("--scenarios", help="PBMC scenario selector")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--baseline-dir",
        help="directory containing hash-validated external baseline source trees",
    )
    parser.add_argument("--output-dir", default="results/confirmatory")
    parser.add_argument("--configuration", default="preregistration/frozen_configuration.json")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--memory-limit-mb", type=int, default=16384)
    return parser


def _apply_process_limits(args: argparse.Namespace) -> None:
    """Apply the requested cap to this independently launched runner."""
    try:
        import resource

        limit = int(args.memory_limit_mb) * 1024 * 1024
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (limit, limit if hard == resource.RLIM_INFINITY else min(limit, hard)),
        )
    except (ImportError, OSError, ValueError):
        pass
    if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
        raise ValueError("--timeout-seconds must be finite and positive")
    if args.memory_limit_mb <= 0:
        raise ValueError("--memory-limit-mb must be positive")

    def timeout_handler(signum, frame):
        del signum, frame
        raise TimeoutError(
            f"external runner exceeded {args.timeout_seconds:g} seconds"
        )

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(args.timeout_seconds))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    stem = f"dual-bound-force-{args.dataset}-{'smoke' if args.smoke else 'full'}"
    try:
        _apply_process_limits(args)
        configuration = (
            {"parallel_lambda": 3.0, "residual_lambda": 3.0, "qualification": "smoke"}
            if args.smoke
            else load_frozen_configuration(args.configuration)
        )
        if args.dataset == "pbmc":
            records, provenance, preprocessing = run_pbmc(args, configuration)
        else:
            records, provenance, preprocessing = run_kdd(args, configuration)
        seeds = sorted({int(row["seed"]) for row in records})
        results: dict[str, Any] = {"records": len(records)}
        if args.dataset == "kdd":
            results["aligned_query_auc_gain"] = {
                **paired_bootstrap_ci(
                    [float(row["aligned_minus_raw_auc"]) for row in records],
                    seed=77_502,
                    resamples=10_000,
                ),
                "status": "evaluated",
                "difference_definition": "aligned AUC - raw-standardized AUC",
            }
        payload = envelope(
            experiment=f"dual_bound_force_{args.dataset}",
            stage="confirmatory",
            status="completed",
            seeds=seeds,
            parameters={**vars(args), "configuration": configuration},
            results=results,
            provenance=provenance,
            preprocessing=preprocessing,
            evidence_label=("smoke_verification_not_inferential" if args.smoke else "preregistered_confirmatory_evidence"),
        )
    except ExternalDataUnavailable as exc:
        records = [{"status": "skipped_external", "reason": str(exc)}]
        payload = envelope(
            experiment=f"dual_bound_force_{args.dataset}",
            stage="confirmatory",
            status="skipped_external",
            seeds=(),
            parameters=vars(args),
            results={},
            evidence_label="external_data_blocker",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    except (TimeoutError, MemoryError) as exc:
        status = "timeout" if isinstance(exc, TimeoutError) else "oom"
        records = [{"status": status, "error_type": type(exc).__name__, "error": str(exc)}]
        payload = envelope(
            experiment=f"dual_bound_force_{args.dataset}",
            stage="confirmatory",
            status=status,
            seeds=(),
            parameters=vars(args),
            results={},
            evidence_label="measured_resource_failure",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    except Exception as exc:
        records = [{"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}]
        payload = envelope(
            experiment=f"dual_bound_force_{args.dataset}",
            stage="confirmatory",
            status="failed",
            seeds=(),
            parameters=vars(args),
            results={},
            evidence_label="execution_failure",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
        )
        write_bundle(payload, output_dir=args.output_dir, stem=stem, tidy_rows=records)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    paths = write_bundle(payload, output_dir=args.output_dir, stem=stem, tidy_rows=records)
    print(paths["json"])


if __name__ == "__main__":
    main()
