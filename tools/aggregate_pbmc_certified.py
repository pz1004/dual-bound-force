#!/usr/bin/env python3
"""Validate and aggregate the certified-reference PBMC primary rerun."""

from __future__ import annotations

import argparse
from itertools import product
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dual_bound_force.experiments.metrics import (
    normalized_projection_error,
    paired_bootstrap_ci,
)
from dual_bound_force.experiments.reporting import envelope, sha256_file, write_bundle


def typed(value: str):
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """One-sided exact paired randomization p-value for a negative mean."""

    observed = float(np.mean(differences))
    outcomes = np.fromiter(
        (
            np.mean(differences * np.asarray(signs, dtype=float))
            for signs in product((-1.0, 1.0), repeat=len(differences))
        ),
        dtype=float,
        count=2 ** len(differences),
    )
    return float(np.mean(outcomes <= observed + 1e-15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate certified-reference PBMC shards."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT / "results/pbmc-certified-rerun",
        help="Directory containing seed*/dual-bound-force-pbmc-full.json shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/confirmatory",
        help="Destination for the strict aggregate bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    shard_paths = sorted(
        input_dir.glob("seed*/dual-bound-force-pbmc-full.json")
    )
    if len(shard_paths) != 10:
        raise RuntimeError(f"expected ten certified PBMC shards, found {len(shard_paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    if any(payload.get("status") != "completed" for payload in payloads):
        raise RuntimeError("every certified PBMC shard must be completed")
    configurations = [payload["parameters"]["configuration"] for payload in payloads]
    if any(value != configurations[0] for value in configurations[1:]):
        raise RuntimeError("certified PBMC estimator configurations differ")
    references = [payload["preprocessing"]["reference_svd"] for payload in payloads]
    if any(value != references[0] for value in references[1:]):
        raise RuntimeError("certified PBMC reference metadata differs between shards")
    provenances = [payload["provenance"] for payload in payloads]
    if any(value != provenances[0] for value in provenances[1:]):
        raise RuntimeError("certified PBMC source provenance differs between shards")

    reference_metadata = references[0]
    if (
        reference_metadata.get("mode")
        != "checksummed_two_start_rayleigh_ritz_certified_cache"
        or not reference_metadata["diagnostics"].get("certified")
    ):
        raise RuntimeError("PBMC rerun did not use the certified reference")
    parameters = reference_metadata["parameters"]
    reference_path = (
        PROJECT
        / "data/pbmc68k"
        / (
            "reference-certified-v2-"
            f"rank{parameters['rank']}-seeds{parameters['seeds'][0]}-"
            f"{parameters['seeds'][1]}.npz"
        )
    )
    if sha256_file(reference_path) != reference_metadata["array_sha256"]:
        raise RuntimeError("certified PBMC reference checksum changed")
    with np.load(reference_path, allow_pickle=False) as values:
        reference_basis = np.asarray(values["basis"], dtype=float)

    records: list[dict] = []
    for path in shard_paths:
        raw_path = path.with_suffix(".raw.csv")
        with raw_path.open(encoding="utf-8", newline="") as stream:
            rows = [
                {key: typed(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
        records.extend(rows)
    expected = {
        (seed, method)
        for seed in range(500, 510)
        for method in ("sketch_mad", "dual_bound")
    }
    observed = {(int(row["seed"]), row["method"]) for row in records}
    if (
        len(records) != len(expected)
        or observed != expected
        or any(row["scenario"] != "bounded_out_of_subspace" for row in records)
    ):
        raise RuntimeError("certified PBMC primary coverage is incomplete")

    maximum_recomputation_difference = 0.0
    for row in records:
        basis_path = Path(str(row["basis_path"]))
        if not basis_path.is_file() or sha256_file(basis_path) != row["basis_sha256"]:
            raise RuntimeError(f"fitted-basis checksum failed for {basis_path}")
        with np.load(basis_path, allow_pickle=False) as values:
            fitted_basis = np.asarray(values["basis"], dtype=float)
        recomputed = normalized_projection_error(reference_basis, fitted_basis)
        maximum_recomputation_difference = max(
            maximum_recomputation_difference,
            abs(recomputed - float(row["subspace_error"])),
        )
        row["recomputed_subspace_error"] = recomputed
    if maximum_recomputation_difference > 1e-12:
        raise RuntimeError("stored PBMC errors do not match retained fitted bases")

    dual = {
        int(row["seed"]): float(row["recomputed_subspace_error"])
        for row in records
        if row["method"] == "dual_bound"
    }
    mad = {
        int(row["seed"]): float(row["recomputed_subspace_error"])
        for row in records
        if row["method"] == "sketch_mad"
    }
    differences = np.asarray(
        [dual[seed] - mad[seed] for seed in range(500, 510)], dtype=float
    )
    bootstrap = paired_bootstrap_ci(differences, seed=77_500, resamples=10_000)
    sample_sd = float(np.std(differences, ddof=1))
    standard_error = sample_sd / math.sqrt(len(differences))
    t_critical = float(stats.t.ppf(0.975, len(differences) - 1))
    t_interval = [
        float(np.mean(differences) - t_critical * standard_error),
        float(np.mean(differences) + t_critical * standard_error),
    ]
    negative = int(np.sum(differences < 0.0))
    nonzero = int(np.sum(differences != 0.0))
    one_sided_sign_p = float(stats.binom.sf(negative - 1, nonzero, 0.5))
    criterion = {
        "status": "evaluated_against_certified_reference",
        "difference_definition": "DualBound - MAD_Sketch",
        "expected_direction": "upper_95_below_zero",
        "estimate": float(np.mean(differences)),
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "paired_t_95": t_interval,
        "percentile_bootstrap_95": [
            float(bootstrap["lower_95"]),
            float(bootstrap["upper_95"]),
        ],
        "percentile_bootstrap_resamples": 10_000,
        "favorable_differences": negative,
        "nonzero_differences": nonzero,
        "exact_one_sided_sign_p": one_sided_sign_p,
        "exact_one_sided_sign_flip_p": exact_sign_flip_p(differences),
        "passed_preregistered_interval_rule": bool(bootstrap["upper_95"] < 0.0),
        "maximum_basis_metric_recomputation_difference": float(
            maximum_recomputation_difference
        ),
    }
    payload = envelope(
        experiment="dual_bound_force_pbmc_certified_reference",
        stage="confirmatory_reference_correction",
        status="completed",
        seeds=range(500, 510),
        parameters={
            "configuration": configurations[0],
            "methods": ["sketch_mad", "dual_bound"],
            "scenario": "bounded_out_of_subspace",
            "amendment": (
                "The numerical PBMC reference gate was strengthened after an "
                "independent statistical audit. Estimator parameters and seeds "
                "were unchanged; the earlier loose-reference bundle is retained."
            ),
        },
        results={
            "records": len(records),
            "primary_criterion": criterion,
            "shards": [str(path.resolve()) for path in shard_paths],
            "retained_basis_files": len(records),
        },
        provenance=provenances[0],
        preprocessing=payloads[0]["preprocessing"],
        evidence_label="confirmatory_pbmc_reference_correction_disclosed_post_run",
    )
    outputs = write_bundle(
        payload,
        output_dir=output_dir,
        stem="dual-bound-force-pbmc-certified-full",
        tidy_rows=sorted(records, key=lambda row: (int(row["seed"]), row["method"])),
    )
    print(json.dumps({"outputs": outputs, "criterion": criterion}, indent=2))


if __name__ == "__main__":
    main()
