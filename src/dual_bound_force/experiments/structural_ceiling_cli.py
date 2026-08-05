"""Fresh secondary comparison with oracle-assisted Noisy Outlier Pursuit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy import stats

from .baselines import fit_method
from .metrics import covariance_to_correlation, normalized_projection_error, principal_basis
from .outlier_pursuit import noisy_outlier_pursuit, outlier_support
from .reporting import envelope, write_bundle
from .scenarios import generate_paired_stream


FULL_SEEDS = tuple(range(700, 730))
SCENARIOS = ("bounded_out_of_subspace", "in_subspace_leverage", "mixed")
SPECTRA = ("strong", "weak")
METHODS = ("sketch_mad", "dual_bound", "noisy_outlier_pursuit")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _parse_seeds(value: str | None, *, smoke: bool) -> tuple[int, ...]:
    if value is None:
        return (700,) if smoke else FULL_SEEDS
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _dimensions(smoke: bool) -> dict[str, int]:
    if smoke:
        return {"n": 320, "p": 16, "rank": 2, "k": 4, "calibration_size": 64}
    return {"n": 4_000, "p": 50, "rank": 5, "k": 10, "calibration_size": 512}


def _record_base(*, seed: int, spectrum: str, scenario: str, method: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "evidence_label": "post_confirmatory_secondary_structural_ceiling",
        "seed": int(seed),
        "spectrum": spectrum,
        "scenario": scenario,
        "method": method,
        "primary_holm_member": False,
    }


def _run_task(task: tuple[int, str, str, bool, str | None]) -> list[dict[str, Any]]:
    seed, spectrum, scenario, smoke, baseline_dir = task
    dimensions = _dimensions(smoke)
    stream = generate_paired_stream(
        seed=seed,
        n=dimensions["n"],
        p=dimensions["p"],
        rank=dimensions["rank"],
        calibration_size=dimensions["calibration_size"],
        scenario=scenario,
        spectrum=spectrum,
    )
    calibration_size = dimensions["calibration_size"]
    estimation = stream.observed[calibration_size:]
    rows: list[dict[str, Any]] = []

    for method, values in (
        ("sketch_mad", estimation),
        ("dual_bound", stream.observed),
    ):
        base = _record_base(
            seed=seed, spectrum=spectrum, scenario=scenario, method=method
        )
        try:
            result = fit_method(
                values,
                method=method,
                rank=dimensions["rank"],
                k=dimensions["k"],
                calibration_size=calibration_size,
                marginal_lambda=3.0,
                parallel_lambda=4.0,
                residual_lambda=1.5,
                baseline_dir=baseline_dir,
            )
            base.update(
                {
                    "subspace_error": normalized_projection_error(
                        stream.signal_basis, result.basis
                    ),
                    "elapsed_seconds": result.elapsed_seconds,
                    "throughput_rows_per_second": result.throughput_rows_per_second,
                    "working_state_bytes": result.state_bytes,
                    "process_peak_rss_bytes": _rss_bytes(),
                    "effective_estimation_rows": int(len(estimation)),
                    "support_precision": None,
                    "support_recall": None,
                    "iterations": None,
                    "primal_residual": None,
                    "constraint_violation": None,
                    "qualification": (
                        "bounded-state comparator on identical estimation rows"
                    ),
                }
            )
        except Exception as exc:
            base.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        rows.append(base)

    base = _record_base(
        seed=seed,
        spectrum=spectrum,
        scenario=scenario,
        method="noisy_outlier_pursuit",
    )
    try:
        location = np.median(stream.observed[:calibration_size], axis=0)
        clean = (stream.clean[calibration_size:] - location).T
        observed = (estimation - location).T
        left, singular_values, right = np.linalg.svd(clean, full_matrices=False)
        rank = dimensions["rank"]
        clean_rank_r = (left[:, :rank] * singular_values[:rank]) @ right[:rank]
        noise_budget = float(np.linalg.norm(clean - clean_rank_r, ord="fro"))
        gamma_star = 0.10
        regularization = 3.0 / (7.0 * np.sqrt(gamma_star * observed.shape[1]))
        started = time.perf_counter_ns()
        result = noisy_outlier_pursuit(
            observed,
            regularization=regularization,
            noise_budget=noise_budget,
            tolerance=1e-6,
            max_iterations=2_000,
        )
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if not result.converged:
            raise RuntimeError(
                "Noisy Outlier Pursuit did not meet all convergence tolerances"
            )
        predicted = outlier_support(result.column_sparse)
        truth = np.any(stream.contamination_mask[calibration_size:], axis=1)
        true_positive = int(np.count_nonzero(predicted & truth))
        common = {
            "elapsed_seconds": float(elapsed),
            "throughput_rows_per_second": float(
                len(estimation) / max(elapsed, 1e-12)
            ),
            "working_state_bytes": result.working_state_bytes,
            "process_peak_rss_bytes": _rss_bytes(),
            "effective_estimation_rows": int(len(estimation)),
            "support_precision": (
                float(true_positive / np.count_nonzero(predicted))
                if np.any(predicted)
                else 0.0
            ),
            "support_recall": (
                float(true_positive / np.count_nonzero(truth))
                if np.any(truth)
                else 0.0
            ),
            "predicted_outlier_rows": int(np.count_nonzero(predicted)),
            "true_outlier_rows": int(np.count_nonzero(truth)),
            "iterations": result.iterations,
            "objective": result.objective,
            "primal_residual": result.primal_residual,
            "iterate_residual": result.iterate_residual,
            "constraint_violation": result.constraint_violation,
            "regularization": regularization,
            "gamma_star": gamma_star,
            "oracle_noise_budget": noise_budget,
            "oracle_noise_budget_fraction": float(
                noise_budget / max(np.linalg.norm(clean, ord="fro"), 1e-12)
            ),
            "recovered_numerical_rank": result.diagnostics["numerical_rank"],
            "qualification": "oracle-assisted batch structural ceiling",
        }
        if int(result.diagnostics["numerical_rank"]) < dimensions["rank"]:
            base.update(common)
            base.update(
                {
                    "status": "failed",
                    "subspace_error": None,
                    "error_type": "RankDeficientComparatorOutput",
                    "error": (
                        "Noisy Outlier Pursuit returned rank "
                        f"{result.diagnostics['numerical_rank']} below target rank "
                        f"{dimensions['rank']}; no arbitrary completion was scored"
                    ),
                }
            )
        else:
            scatter = result.low_rank @ result.low_rank.T / observed.shape[1]
            basis = principal_basis(
                covariance_to_correlation(scatter), dimensions["rank"]
            )
            base.update(common)
            base["subspace_error"] = normalized_projection_error(
                stream.signal_basis, basis
            )
    except Exception as exc:
        base.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_peak_rss_bytes": _rss_bytes(),
            }
        )
    rows.append(base)
    return rows


def _descriptive_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("descriptive intervals require at least two finite values")
    estimate = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    critical = float(stats.t.ppf(0.975, len(array) - 1))
    return {
        "estimate": estimate,
        "paired_t_95": [
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ],
        "n_pairs": int(len(array)),
        "inferential_status": "descriptive_secondary_not_in_primary_holm_family",
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    means: dict[str, Any] = {}
    for spectrum in SPECTRA:
        for scenario in SCENARIOS:
            for method in METHODS:
                selected = [
                    row
                    for row in completed
                    if row["spectrum"] == spectrum
                    and row["scenario"] == scenario
                    and row["method"] == method
                ]
                key = f"{spectrum}.{scenario}.{method}"
                attempted = [
                    row
                    for row in records
                    if row["spectrum"] == spectrum
                    and row["scenario"] == scenario
                    and row["method"] == method
                ]
                diagnostic = [
                    row
                    for row in attempted
                    if row.get("support_precision") is not None
                ]
                means[key] = {
                    "attempted_n": len(attempted),
                    "completed_n": len(selected),
                    "failed_n": len(attempted) - len(selected),
                    "mean_subspace_error": (
                        float(np.mean([row["subspace_error"] for row in selected]))
                        if selected
                        else None
                    ),
                    "mean_support_precision": (
                        float(np.mean([row["support_precision"] for row in diagnostic]))
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_support_recall": (
                        float(np.mean([row["support_recall"] for row in diagnostic]))
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_recovered_numerical_rank": (
                        float(
                            np.mean(
                                [row["recovered_numerical_rank"] for row in diagnostic]
                            )
                        )
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_oracle_noise_budget_fraction": (
                        float(
                            np.mean(
                                [row["oracle_noise_budget_fraction"] for row in diagnostic]
                            )
                        )
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_iterations": (
                        float(np.mean([row["iterations"] for row in diagnostic]))
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_primal_residual": (
                        float(np.mean([row["primal_residual"] for row in diagnostic]))
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_constraint_violation": (
                        float(
                            np.mean([row["constraint_violation"] for row in diagnostic])
                        )
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                    "mean_predicted_outlier_rows": (
                        float(
                            np.mean([row["predicted_outlier_rows"] for row in diagnostic])
                        )
                        if diagnostic and method == "noisy_outlier_pursuit"
                        else None
                    ),
                }

    contrasts: dict[str, Any] = {}
    index = {
        (row["seed"], row["spectrum"], row["scenario"], row["method"]): row
        for row in completed
    }
    seeds = sorted({int(row["seed"]) for row in completed})
    for spectrum in SPECTRA:
        for scenario in SCENARIOS:
            for comparator in ("sketch_mad", "dual_bound"):
                differences = []
                for seed in seeds:
                    op = index.get((seed, spectrum, scenario, "noisy_outlier_pursuit"))
                    baseline = index.get((seed, spectrum, scenario, comparator))
                    if op is not None and baseline is not None:
                        differences.append(op["subspace_error"] - baseline["subspace_error"])
                key = f"{spectrum}.{scenario}.outlier_pursuit_minus_{comparator}"
                contrasts[key] = (
                    _descriptive_interval(differences)
                    if len(differences) >= 2
                    else {
                        "estimate": None,
                        "paired_t_95": None,
                        "n_pairs": len(differences),
                        "inferential_status": "insufficient_completed_pairs",
                    }
                )
            dual_differences = []
            for seed in seeds:
                dual = index.get((seed, spectrum, scenario, "dual_bound"))
                mad = index.get((seed, spectrum, scenario, "sketch_mad"))
                if dual is not None and mad is not None:
                    dual_differences.append(dual["subspace_error"] - mad["subspace_error"])
            contrasts[f"{spectrum}.{scenario}.dual_bound_minus_sketch_mad"] = (
                _descriptive_interval(dual_differences)
                if len(dual_differences) >= 2
                else {
                    "estimate": None,
                    "paired_t_95": None,
                    "n_pairs": len(dual_differences),
                    "inferential_status": "insufficient_completed_pairs",
                }
            )
    return {
        "record_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(records) - len(completed),
        "means": means,
        "paired_descriptive_contrasts": contrasts,
        "primary_holm_family_unchanged": True,
    }


def run(
    *, seeds: tuple[int, ...], smoke: bool, jobs: int, baseline_dir: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    tasks = [
        (seed, spectrum, scenario, smoke, baseline_dir)
        for seed in seeds
        for spectrum in SPECTRA
        for scenario in SCENARIOS
    ]
    records: list[dict[str, Any]] = []
    if jobs == 1:
        for task in tasks:
            records.extend(_run_task(task))
    else:
        with ProcessPoolExecutor(
            max_workers=jobs,
            max_tasks_per_child=1,
        ) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda row: (row["seed"], row["spectrum"], row["scenario"], row["method"]))
    dimensions = _dimensions(smoke)
    payload = envelope(
        experiment="dual_bound_force_structural_ceiling",
        stage="secondary_smoke" if smoke else "post_confirmatory_secondary",
        status="completed",
        seeds=seeds,
        parameters={
            **dimensions,
            "scenarios": list(SCENARIOS),
            "spectra": list(SPECTRA),
            "methods": list(METHODS),
            "gamma_star": 0.10,
            "regularization_rule": "3/(7*sqrt(gamma_star*n_estimation))",
            "noise_budget": "oracle rank-r residual norm of clean estimation matrix",
            "solver_tolerance": 1e-6,
            "solver_max_iterations": 2_000,
            "smoke": smoke,
            "jobs": jobs,
        },
        results=summarize(records),
        provenance={
            "method": "Noisy Outlier Pursuit",
            "source": "https://doi.org/10.1109/TIT.2011.2173156",
            "qualification": "independent audited implementation from the published convex program",
        },
        preprocessing={
            "outlier_pursuit_center": "coordinate medians of the initial calibration block",
            "evaluation_rows": "identical post-calibration rows for every reported estimate",
            "correlation": "normalized second moment of recovered low-rank component",
        },
        evidence_label=(
            "smoke_verification_not_inferential"
            if smoke
            else "fresh_post_confirmatory_secondary_evidence"
        ),
    )
    return payload, records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", default="results/structural-ceiling")
    parser.add_argument(
        "--baseline-dir",
        help="directory containing hash-validated external baseline source trees",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    seeds: tuple[int, ...] = ()
    stem = f"dual-bound-force-structural-ceiling-{'smoke' if args.smoke else 'full'}"
    try:
        seeds = _parse_seeds(args.seeds, smoke=args.smoke)
        payload, records = run(
            seeds=seeds,
            smoke=args.smoke,
            jobs=args.jobs,
            baseline_dir=args.baseline_dir,
        )
    except Exception as exc:
        payload = envelope(
            experiment="dual_bound_force_structural_ceiling",
            stage="secondary_smoke" if args.smoke else "post_confirmatory_secondary",
            status="failed",
            seeds=seeds,
            parameters=vars(args),
            results={},
            evidence_label="execution_failure",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
        )
        records = [{"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}]
        write_bundle(payload, output_dir=args.output_dir, stem=stem, tidy_rows=records)
        raise
    paths = write_bundle(payload, output_dir=args.output_dir, stem=stem, tidy_rows=records)
    print(paths["json"])


if __name__ == "__main__":
    main()
