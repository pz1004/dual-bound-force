"""Preregistered development and confirmatory synthetic study."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dual_bound_force import DualBoundFORCE

from .baselines import fit_method
from .metrics import (
    covariance_to_correlation,
    holm_adjust,
    normalized_frobenius_error,
    normalized_projection_error,
    paired_mean_inference,
    principal_basis,
)
from .scenarios import SCENARIOS, SPECTRA, generate_paired_stream, scheduled_epoch_stream


DEVELOPMENT_SEEDS = tuple(range(300, 310))
CONFIRMATORY_SEEDS = tuple(range(500, 530))
PBMC_SEEDS = tuple(range(500, 510))
LAMBDA_GRID = (1.5, 2.0, 3.0, 4.0)
MARGINAL_LAMBDA_GRID = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PRIMARY_DENSE_METHODS = (
    "pearson",
    "force",
    "mad_force",
    "sketch_mad",
    "dual_bound",
)
PRIMARY_SKETCH_METHODS = (
    "vanilla_fd",
    "rfd",
    "sketch_iqr",
    "sketch_mad",
    "dual_bound",
)


@dataclass(frozen=True)
class StudyDimensions:
    n: int
    p: int
    rank: int
    k: int
    calibration_size: int


FULL_DENSE = StudyDimensions(4000, 50, 5, 10, 512)
FULL_SKETCH = StudyDimensions(6000, 200, 10, 20, 512)
SMOKE_DENSE = StudyDimensions(320, 12, 3, 4, 32)
SMOKE_SKETCH = StudyDimensions(420, 20, 3, 5, 40)


def _reference(stream) -> tuple[np.ndarray, np.ndarray]:
    correlation = covariance_to_correlation(stream.clean_estimation_covariance)
    basis = principal_basis(correlation, min(stream.signal_basis.shape[1], correlation.shape[0]))
    return correlation, basis


def _method_record(
    stream,
    *,
    seed: int,
    tier: str,
    method: str,
    dimensions: StudyDimensions,
    parallel_lambda: float,
    residual_lambda: float,
    marginal_lambda: float = 3.0,
    k: int | None = None,
    baseline_dir: str | None = None,
) -> dict[str, Any]:
    used_k = dimensions.k if k is None else int(k)
    reference_correlation, reference_basis = _reference(stream)
    fit = fit_method(
        stream.observed,
        method=method,
        rank=dimensions.rank,
        k=used_k,
        calibration_size=dimensions.calibration_size,
        marginal_lambda=marginal_lambda,
        parallel_lambda=parallel_lambda,
        residual_lambda=residual_lambda,
        baseline_dir=baseline_dir,
    )
    transformed_error = None
    if fit.transformed_target is not None:
        transformed_error = normalized_frobenius_error(
            fit.correlation, fit.transformed_target
        )
    clean_scatter_error = None
    if fit.clean_target_scatter_estimate is not None:
        clean_scatter_error = normalized_frobenius_error(
            fit.clean_target_scatter_estimate,
            stream.clean_estimation_covariance,
        )
    transformed_scatter_error = None
    if (
        fit.transformed_scatter_estimate is not None
        and fit.transformed_scatter_target is not None
    ):
        transformed_scatter_error = normalized_frobenius_error(
            fit.transformed_scatter_estimate,
            fit.transformed_scatter_target,
        )
    retained_contamination_rate = None
    clean_attenuation_rate = None
    if fit.marginal_accepted is not None:
        eligible = np.ones(stream.observed.shape, dtype=bool)
        if method == "dual_bound":
            eligible[: dimensions.calibration_size] = False
        contaminated = stream.contamination_mask & eligible
        clean = (~stream.contamination_mask) & eligible
        if np.any(contaminated):
            retained_contamination_rate = float(
                np.mean(fit.marginal_accepted[contaminated])
            )
        if np.any(clean):
            clean_attenuation_rate = float(
                np.mean(~fit.marginal_accepted[clean])
            )
    return {
        "status": "completed",
        "tier": tier,
        "seed": int(seed),
        "scenario": stream.scenario,
        "spectrum": stream.spectrum,
        "method": method,
        "n": dimensions.n,
        "p": dimensions.p,
        "rank": dimensions.rank,
        "k": used_k,
        "calibration_size": dimensions.calibration_size,
        "marginal_lambda": float(marginal_lambda),
        "parallel_lambda": float(parallel_lambda),
        "residual_lambda": float(residual_lambda),
        "clean_target_correlation_error": normalized_frobenius_error(
            fit.correlation, reference_correlation
        ),
        "subspace_error": normalized_projection_error(reference_basis, fit.basis),
        "transformed_target_compression_error": transformed_error,
        "clean_target_scale_scatter_error": clean_scatter_error,
        "transformed_target_scatter_compression_error": transformed_scatter_error,
        "marginal_retained_contamination_rate": retained_contamination_rate,
        "clean_marginal_attenuation_rate": clean_attenuation_rate,
        "throughput_rows_per_second": fit.throughput_rows_per_second,
        "elapsed_seconds": fit.elapsed_seconds,
        "state_bytes": fit.state_bytes,
        "calibration_peak_state_bytes": fit.calibration_peak_state_bytes,
        "output_bytes": fit.output_bytes,
        "contaminated_cells": int(stream.contamination_mask.sum()),
        "contaminated_calibration_rows": int(stream.calibration_mask.sum()),
        "marginal_clipped_coordinates": int(
            fit.diagnostics.get("marginal_clipped_coordinates", 0)
        ),
        "parallel_clipped_rows": int(fit.diagnostics.get("parallel_clipped_rows", 0)),
        "residual_clipped_rows": int(fit.diagnostics.get("residual_clipped_rows", 0)),
        "effective_n": int(fit.diagnostics.get("effective_n", dimensions.n)),
        "diagnostics": fit.diagnostics,
    }


def run_development(
    *, seeds: Sequence[int], smoke: bool, baseline_dir: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dimensions = SMOKE_DENSE if smoke else FULL_DENSE
    scenarios = ("clean", "casewise_cauchy", "cellwise_cauchy", "bounded_out_of_subspace")
    spectra = ("strong",) if smoke else SPECTRA
    grid = (2.0, 3.0) if smoke else LAMBDA_GRID
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for spectrum in spectra:
            streams = {
                scenario: generate_paired_stream(
                    seed=seed,
                    n=dimensions.n,
                    p=dimensions.p,
                    rank=dimensions.rank,
                    calibration_size=dimensions.calibration_size,
                    scenario=scenario,
                    spectrum=spectrum,
                )
                for scenario in scenarios
            }
            for scenario, stream in streams.items():
                records.append(
                    _method_record(
                        stream,
                        seed=seed,
                        tier="development_reference",
                        method="sketch_mad",
                        dimensions=dimensions,
                        parallel_lambda=3.0,
                        residual_lambda=3.0,
                        baseline_dir=baseline_dir,
                    )
                )
            for parallel_lambda in grid:
                for residual_lambda in grid:
                    for scenario, stream in streams.items():
                        records.append(
                            _method_record(
                                stream,
                                seed=seed,
                                tier="development_grid",
                                method="dual_bound",
                                dimensions=dimensions,
                                parallel_lambda=parallel_lambda,
                                residual_lambda=residual_lambda,
                                baseline_dir=baseline_dir,
                            )
                        )
    selection = select_configuration(records, grid=grid)
    return records, selection


def select_configuration(
    records: Sequence[dict[str, Any]], *, grid: Iterable[float] = LAMBDA_GRID
) -> dict[str, Any]:
    references: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in records:
        if row["method"] == "sketch_mad":
            references[(row["seed"], row["spectrum"], row["scenario"])] = row
    candidates = []
    for parallel_lambda in grid:
        for residual_lambda in grid:
            selected = [
                row
                for row in records
                if row["method"] == "dual_bound"
                and row["parallel_lambda"] == parallel_lambda
                and row["residual_lambda"] == residual_lambda
            ]
            if not selected:
                continue
            ratios: dict[str, list[float]] = defaultdict(list)
            clean_differences = []
            throughput_ratios = []
            for row in selected:
                reference = references[(row["seed"], row["spectrum"], row["scenario"])]
                error_ratio = row["subspace_error"] / max(reference["subspace_error"], 1e-12)
                ratios[row["scenario"]].append(float(error_ratio))
                throughput_ratios.append(
                    row["throughput_rows_per_second"]
                    / max(reference["throughput_rows_per_second"], 1e-12)
                )
                if row["scenario"] == "clean":
                    clean_differences.append(
                        row["subspace_error"] - 1.05 * reference["subspace_error"]
                    )
            minimax = max(
                float(np.mean(values))
                for scenario, values in ratios.items()
                if scenario != "clean"
            )
            clean_noninferior = float(np.mean(clean_differences)) <= 0.0
            throughput_ratio = float(np.median(throughput_ratios))
            throughput_feasible = throughput_ratio >= 0.70
            violation = max(float(np.mean(clean_differences)), 0.0) + max(
                0.70 - throughput_ratio, 0.0
            )
            candidates.append(
                {
                    "parallel_lambda": float(parallel_lambda),
                    "residual_lambda": float(residual_lambda),
                    "minimax_mean_subspace_error_ratio": minimax,
                    "clean_noninferior_point_estimate": clean_noninferior,
                    "throughput_ratio_median": throughput_ratio,
                    "throughput_feasible": throughput_feasible,
                    "feasible": clean_noninferior and throughput_feasible,
                    "constraint_violation_score": float(violation),
                }
            )
    if not candidates:
        raise ValueError("no complete configuration records were provided")
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    ordering = feasible if feasible else candidates
    winner = min(
        ordering,
        key=lambda candidate: (
            0.0 if candidate["feasible"] else candidate["constraint_violation_score"],
            candidate["minimax_mean_subspace_error_ratio"],
            candidate["parallel_lambda"],
            candidate["residual_lambda"],
        ),
    )
    return {
        "selection_rule": (
            "minimize worst contaminated-scenario mean paired subspace-error ratio; "
            "require clean point-estimate noninferiority within 5% and median throughput "
            "at least 70% of MAD Sketch-FORCE"
        ),
        "feasible_configuration_exists": bool(feasible),
        "selected": winner,
        "candidates": candidates,
        "selection_is_development_only": True,
    }


def write_frozen_configuration(
    selection: dict[str, Any], *, path: str | Path, smoke: bool
) -> dict[str, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hash_path = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or hash_path.exists():
        raise FileExistsError(
            "frozen configuration already exists; delete neither artifact after "
            "confirmatory access and use a new preregistration path for a new study"
        )
    selected = selection["selected"]
    payload = {
        "schema_version": "1.0",
        "estimator": "DualBoundFORCE",
        "parallel_lambda": selected["parallel_lambda"],
        "residual_lambda": selected["residual_lambda"],
        "marginal_lambda": 3.0,
        "calibration_size": 512,
        "selection_rule": selection["selection_rule"],
        "feasible_configuration_exists": selection["feasible_configuration_exists"],
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "smoke_only": bool(smoke),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "prohibition": "Do not change after inspecting confirmatory outcomes.",
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    hash_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {"path": str(output.resolve()), "sha256": digest, "hash_file": str(hash_path.resolve())}


def load_frozen_configuration(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    hash_path = source.with_suffix(source.suffix + ".sha256")
    if not hash_path.is_file() or not hash_path.read_text(encoding="ascii").startswith(digest):
        raise ValueError("frozen configuration hash is missing or mismatched")
    if payload.get("smoke_only"):
        raise ValueError("a smoke-only selection cannot authorize confirmatory execution")
    return payload


def run_confirmatory_synthetic(
    *,
    seeds: Sequence[int],
    configuration: dict[str, Any],
    tiers: Sequence[str],
    smoke: bool,
    jobs: int = 1,
    baseline_dir: str | None = None,
    methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    records: list[dict[str, Any]] = []
    parallel_lambda = float(configuration["parallel_lambda"])
    residual_lambda = float(configuration["residual_lambda"])
    scenarios = ("clean", "casewise_cauchy", "cellwise_cauchy", "bounded_out_of_subspace") if smoke else SCENARIOS
    spectra = ("strong",) if smoke else SPECTRA
    tasks: list[dict[str, Any]] = []
    if "dense" in tiers:
        dimensions = SMOKE_DENSE if smoke else FULL_DENSE
        task_methods = ("pearson", "sketch_mad", "dual_bound") if smoke else PRIMARY_DENSE_METHODS
        for seed in seeds:
            for spectrum in spectra:
                for scenario in scenarios:
                    tasks.append(
                        {
                            "seed": seed,
                            "spectrum": spectrum,
                            "scenario": scenario,
                            "tier": "dense",
                            "dimensions": dimensions,
                            "methods": task_methods,
                            "k_values": (dimensions.k,),
                            "parallel_lambda": parallel_lambda,
                            "residual_lambda": residual_lambda,
                            "marginal_lambda": 3.0,
                            "baseline_dir": baseline_dir,
                        }
                    )
    if "sketch" in tiers:
        dimensions = SMOKE_SKETCH if smoke else FULL_SKETCH
        task_methods = ("vanilla_fd", "sketch_mad", "dual_bound") if smoke else PRIMARY_SKETCH_METHODS
        k_values = (dimensions.k,) if smoke else (10, 20, 40)
        for seed in seeds:
            for spectrum in spectra:
                for scenario in scenarios:
                    tasks.append(
                        {
                            "seed": seed,
                            "spectrum": spectrum,
                            "scenario": scenario,
                            "tier": "sketch",
                            "dimensions": dimensions,
                            "methods": task_methods,
                            "k_values": k_values,
                            "parallel_lambda": parallel_lambda,
                            "residual_lambda": residual_lambda,
                            "marginal_lambda": 3.0,
                            "baseline_dir": baseline_dir,
                        }
                    )
    if "oracle" in tiers:
        dimensions = SMOKE_SKETCH if smoke else FULL_SKETCH
        k_values = (dimensions.k,) if smoke else (10, 20, 40)
        for seed in seeds:
            for spectrum in spectra:
                for scenario in scenarios:
                    tasks.append(
                        {
                            "seed": seed,
                            "spectrum": spectrum,
                            "scenario": scenario,
                            "tier": "oracle_ablation",
                            "dimensions": dimensions,
                            "methods": ("exact_mad_fd",),
                            "k_values": k_values,
                            "parallel_lambda": parallel_lambda,
                            "residual_lambda": residual_lambda,
                            "marginal_lambda": 3.0,
                            "baseline_dir": baseline_dir,
                        }
                    )
    if "threshold" in tiers:
        dimensions = SMOKE_SKETCH if smoke else FULL_SKETCH
        grid = (1.5, 3.0) if smoke else MARGINAL_LAMBDA_GRID
        for seed in seeds:
            for spectrum in spectra:
                for scenario in scenarios:
                    for marginal_lambda in grid:
                        tasks.append(
                            {
                                "seed": seed,
                                "spectrum": spectrum,
                                "scenario": scenario,
                                "tier": "threshold_sensitivity",
                                "dimensions": dimensions,
                                "methods": ("sketch_mad", "dual_bound"),
                                "k_values": (dimensions.k,),
                                "parallel_lambda": parallel_lambda,
                                "residual_lambda": residual_lambda,
                                "marginal_lambda": marginal_lambda,
                                "baseline_dir": baseline_dir,
                            }
                        )
    if "retention" in tiers:
        dimensions = SMOKE_SKETCH if smoke else FULL_SKETCH
        for seed in seeds:
            for spectrum in spectra:
                for scenario in scenarios:
                    tasks.append(
                        {
                            "seed": seed,
                            "spectrum": spectrum,
                            "scenario": scenario,
                            "tier": "retention_audit",
                            "dimensions": dimensions,
                            "methods": ("sketch_iqr", "sketch_mad", "dual_bound"),
                            "k_values": (dimensions.k,),
                            "parallel_lambda": parallel_lambda,
                            "residual_lambda": residual_lambda,
                            "marginal_lambda": 3.0,
                            "baseline_dir": baseline_dir,
                        }
                    )
    if methods is not None:
        requested = set(methods)
        known = set(PRIMARY_DENSE_METHODS) | set(PRIMARY_SKETCH_METHODS) | {
            "exact_mad_fd"
        }
        if not requested or not requested <= known:
            raise ValueError(f"methods must be drawn from {sorted(known)}")
        filtered = []
        for task in tasks:
            task["methods"] = tuple(
                method for method in task["methods"] if method in requested
            )
            if task["methods"]:
                filtered.append(task)
        tasks = filtered
    if jobs == 1:
        for task in tasks:
            records.extend(_run_synthetic_block(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for block in executor.map(_run_synthetic_block, tasks):
                records.extend(block)
    return records


def _run_synthetic_block(task: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = task["dimensions"]
    stream = generate_paired_stream(
        seed=task["seed"],
        n=dimensions.n,
        p=dimensions.p,
        rank=dimensions.rank,
        calibration_size=dimensions.calibration_size,
        scenario=task["scenario"],
        spectrum=task["spectrum"],
    )
    rows = []
    for used_k in task["k_values"]:
        for method in task["methods"]:
            rows.append(
                _method_record(
                    stream,
                    seed=task["seed"],
                    tier=task["tier"],
                    method=method,
                    dimensions=dimensions,
                    parallel_lambda=task["parallel_lambda"],
                    residual_lambda=task["residual_lambda"],
                    marginal_lambda=task.get("marginal_lambda", 3.0),
                    k=used_k,
                    baseline_dir=task.get("baseline_dir"),
                )
            )
    return rows


def run_calibration_audit(
    *,
    seeds: Sequence[int],
    configuration: dict[str, Any],
    smoke: bool,
    baseline_dir: str | None = None,
) -> list[dict[str, Any]]:
    dimensions = SMOKE_DENSE if smoke else FULL_DENSE
    orderings = (False, True)
    phases = ("estimation", "calibration")
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for contamination_phase in phases:
            for front_loaded in orderings:
                stream = generate_paired_stream(
                    seed=seed,
                    n=dimensions.n,
                    p=dimensions.p,
                    rank=dimensions.rank,
                    calibration_size=dimensions.calibration_size,
                    scenario="casewise_cauchy",
                    spectrum="strong",
                    contamination_phase=contamination_phase,
                    front_loaded=front_loaded,
                )
                row = _method_record(
                    stream,
                    seed=seed,
                    tier="calibration_audit",
                    method="dual_bound",
                    dimensions=dimensions,
                    parallel_lambda=float(configuration["parallel_lambda"]),
                    residual_lambda=float(configuration["residual_lambda"]),
                    baseline_dir=baseline_dir,
                )
                calibration = stream.observed[: dimensions.calibration_size]
                location = np.median(calibration, axis=0)
                scale = np.maximum(
                    1.4826 * np.median(np.abs(calibration - location), axis=0),
                    np.maximum(1e-8, 1e-14 * np.abs(location)),
                )
                clean_calibration = stream.clean[: dimensions.calibration_size]
                clean_location = np.median(clean_calibration, axis=0)
                clean_scale = np.maximum(
                    1.4826 * np.median(np.abs(clean_calibration - clean_location), axis=0),
                    np.maximum(1e-8, 1e-14 * np.abs(clean_location)),
                )
                observed_calibrator = DualBoundFORCE(
                    dimensions.p,
                    dimensions.k,
                    calibration_size=dimensions.calibration_size,
                    parallel_lambda=float(configuration["parallel_lambda"]),
                    residual_lambda=float(configuration["residual_lambda"]),
                ).fit(calibration)
                clean_calibrator = DualBoundFORCE(
                    dimensions.p,
                    dimensions.k,
                    calibration_size=dimensions.calibration_size,
                    parallel_lambda=float(configuration["parallel_lambda"]),
                    residual_lambda=float(configuration["residual_lambda"]),
                ).fit(clean_calibration)
                row.update(
                    {
                        "contaminate_calibration": contamination_phase == "calibration",
                        "contamination_phase": contamination_phase,
                        "front_loaded": front_loaded,
                        "relative_location_error": float(
                            np.linalg.norm(location - clean_location)
                            / max(np.linalg.norm(clean_location), 1e-12)
                        ),
                        "relative_scale_error": float(
                            np.linalg.norm(scale - clean_scale)
                            / max(np.linalg.norm(clean_scale), 1e-12)
                        ),
                        "calibration_basis_error": normalized_projection_error(
                            clean_calibrator.basis,
                            observed_calibrator.basis,
                        ),
                    }
                )
                records.append(row)
    return records


def run_epoch_audit(
    *,
    seeds: Sequence[int],
    configuration: dict[str, Any],
    smoke: bool,
    baseline_dir: str | None = None,
) -> list[dict[str, Any]]:
    dimensions = SMOKE_DENSE if smoke else StudyDimensions(6000, 50, 5, 10, 512)
    change_points = (dimensions.n // 3, 2 * dimensions.n // 3)
    epoch_size = dimensions.n // 3
    records: list[dict[str, Any]] = []
    for seed in seeds:
        stream = scheduled_epoch_stream(
            seed=seed,
            n=dimensions.n,
            p=dimensions.p,
            rank=dimensions.rank,
            change_points=change_points,
        )
        for scheduled in (False, True):
            fit = fit_method(
                stream["matrix"],
                method="dual_bound",
                rank=dimensions.rank,
                k=dimensions.k,
                calibration_size=dimensions.calibration_size,
                parallel_lambda=float(configuration["parallel_lambda"]),
                residual_lambda=float(configuration["residual_lambda"]),
                epoch_size=epoch_size if scheduled else None,
                baseline_dir=baseline_dir,
            )
            reference_basis = stream["basis_scale_2"]
            records.append(
                {
                    "status": "completed",
                    "tier": "scheduled_epoch",
                    "seed": int(seed),
                    "scheduled": scheduled,
                    "epoch_size": epoch_size if scheduled else None,
                    "subspace_error_final_regime": normalized_projection_error(
                        reference_basis, fit.basis
                    ),
                    "throughput_rows_per_second": fit.throughput_rows_per_second,
                    "state_bytes": fit.state_bytes,
                    "calibration_peak_state_bytes": fit.calibration_peak_state_bytes,
                    "diagnostics": fit.diagnostics,
                }
            )
    return records


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "tier", "scenario", "spectrum", "method", "k",
        "marginal_lambda", "parallel_lambda", "residual_lambda",
    )
    for row in records:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    metric_names = (
        "clean_target_correlation_error",
        "subspace_error",
        "throughput_rows_per_second",
        "state_bytes",
        "calibration_peak_state_bytes",
        "relative_location_error",
        "relative_scale_error",
        "subspace_error_final_regime",
        "marginal_retained_contamination_rate",
        "clean_marginal_attenuation_rate",
    )
    for identity, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        result = {key: value for key, value in zip(keys, identity)}
        result["n_records"] = len(rows)
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                result[f"{metric}_mean"] = float(np.mean(values))
                result[f"{metric}_median"] = float(np.median(values))
                result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(result)
    return output


def _paired_seed_means(
    records: Sequence[dict[str, Any]], *, scenario: str, method: str, k: int | None = None
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in records:
        if row.get("scenario") != scenario or row.get("method") != method:
            continue
        if k is not None and row.get("k") != k:
            continue
        grouped[int(row["seed"])].append(float(row["subspace_error"]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def evaluate_primary_criteria(
    records: Sequence[dict[str, Any]], *, bootstrap_resamples: int = 10_000
) -> dict[str, Any]:
    criteria: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}

    def comparison(
        name: str,
        scenario: str,
        *,
        multiplier: float,
        expected: str,
    ) -> None:
        dual = _paired_seed_means(records, scenario=scenario, method="dual_bound", k=20)
        mad = _paired_seed_means(records, scenario=scenario, method="sketch_mad", k=20)
        seeds = sorted(set(dual) & set(mad))
        if not seeds:
            criteria[name] = {"status": "not_evaluated", "reason": "no paired primary-k records"}
            return
        differences = [dual[seed] - multiplier * mad[seed] for seed in seeds]
        inference = paired_mean_inference(
            differences,
            alternative="less",
            bootstrap_seed=91_000 + len(criteria),
            bootstrap_resamples=bootstrap_resamples,
        )
        inference.update(
            {
                "status": "evaluated",
                "expected_direction": expected,
                "difference_definition": f"DualBound - {multiplier} * MAD_Sketch",
                "historical_percentile_interval_rule_passed": (
                    inference["percentile_bootstrap_95"][1] < 0.0
                ),
                "inference_procedure": (
                    "one-sided paired t test of the mean; Holm adjustment "
                    "is applied across scientific hypotheses"
                ),
            }
        )
        criteria[name] = inference
        raw_p_values[name] = float(inference["one_sided_p"])

    comparison(
        "structural_out_of_subspace_improvement",
        "bounded_out_of_subspace",
        multiplier=1.0,
        expected="upper_95_below_zero",
    )
    comparison(
        "casewise_noninferiority_5pct",
        "casewise_cauchy",
        multiplier=1.05,
        expected="upper_95_below_zero",
    )
    comparison(
        "cellwise_noninferiority_5pct",
        "cellwise_cauchy",
        multiplier=1.05,
        expected="upper_95_below_zero",
    )
    comparison(
        "clean_noninferiority_5pct",
        "clean",
        multiplier=1.05,
        expected="upper_95_below_zero",
    )
    adjusted = holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        criteria[name]["holm_adjusted_one_sided_p"] = value
        criteria[name]["passed"] = bool(
            criteria[name]["estimate"] < 0.0 and value < 0.05
        )

    criteria["throughput_70pct"] = {
        "status": "not_evaluated",
        "reason": "the isolated timing bundle controls the throughput gate",
    }

    state_rows = [row for row in records if row.get("method") == "dual_bound"]
    pairs = sorted({(int(row["p"]), int(row["k"]), int(row["state_bytes"])) for row in state_rows})
    if len(pairs) >= 2:
        predictors = np.asarray([p * k + p for p, k, _ in pairs], dtype=float)
        states = np.asarray([state for _, _, state in pairs], dtype=float)
        slope, intercept = np.polyfit(predictors, states, 1)
        fitted = slope * predictors + intercept
        residual = float(np.linalg.norm(states - fitted))
        total = float(np.linalg.norm(states - np.mean(states)))
        r_squared = 1.0 if total <= 1e-12 else 1.0 - (residual / total) ** 2
        criteria["state_growth_pk_plus_p"] = {
            "status": "evaluated",
            "slope_bytes_per_unit": float(slope),
            "intercept_bytes": float(intercept),
            "r_squared": float(r_squared),
            "passed": bool(slope > 0 and r_squared >= 0.95),
            "qualification": "empirical scaling check; not a complexity proof",
        }
    else:
        criteria["state_growth_pk_plus_p"] = {"status": "not_evaluated"}

    peak_pairs = sorted(
        {
            (
                int(row["p"]),
                int(row["k"]),
                int(row["calibration_size"]),
                int(row["calibration_peak_state_bytes"]),
            )
            for row in state_rows
            if row.get("calibration_peak_state_bytes")
            and row.get("calibration_size") is not None
        }
    )
    if len(peak_pairs) >= 2:
        predictors = np.asarray(
            [p * (k + calibration_size) for p, k, calibration_size, _ in peak_pairs],
            dtype=float,
        )
        states = np.asarray([state for _, _, _, state in peak_pairs], dtype=float)
        slope, intercept = np.polyfit(predictors, states, 1)
        fitted = slope * predictors + intercept
        residual = float(np.linalg.norm(states - fitted))
        total = float(np.linalg.norm(states - np.mean(states)))
        r_squared = 1.0 if total <= 1e-12 else 1.0 - (residual / total) ** 2
        criteria["calibration_peak_growth_p_times_k_plus_c"] = {
            "status": "evaluated",
            "slope_bytes_per_unit": float(slope),
            "intercept_bytes": float(intercept),
            "r_squared": float(r_squared),
            "passed": bool(slope > 0 and r_squared >= 0.95),
            "qualification": "empirical peak-state scaling check; not a complexity proof",
        }
    else:
        criteria["calibration_peak_growth_p_times_k_plus_c"] = {
            "status": "not_evaluated"
        }
    return criteria
