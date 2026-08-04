#!/usr/bin/env python3
"""Merge independently isolated PBMC seed shards into strict study bundles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dual_bound_force.experiments.metrics import paired_bootstrap_ci
from dual_bound_force.experiments.reporting import envelope, write_bundle
from dual_bound_force import DualBoundFORCE


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


def _load_records(json_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in json_paths]
    if any(payload["status"] != "completed" for payload in payloads):
        raise RuntimeError("every PBMC shard must have completed")
    records = []
    for path in json_paths:
        raw = path.with_suffix(".raw.csv")
        with raw.open(encoding="utf-8", newline="") as stream:
            records.extend(
                {key: typed(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            )
    return payloads, records


def _validate_shared(payloads: list[dict]) -> tuple[dict, dict]:
    configurations = [payload["parameters"]["configuration"] for payload in payloads]
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise RuntimeError("PBMC shard configurations differ")
    provenances = [payload["provenance"] for payload in payloads]
    if any(provenance != provenances[0] for provenance in provenances[1:]):
        raise RuntimeError("PBMC shard provenance differs")
    preprocessing = [payload["preprocessing"] for payload in payloads]
    if any(value != preprocessing[0] for value in preprocessing[1:]):
        raise RuntimeError("PBMC shard preprocessing/reference metadata differs")
    return configurations[0], provenances[0]


def _apply_final_state_accounting(records: list[dict]) -> None:
    """Apply the final deterministic calibration-workspace envelope.

    PBMC seed shards may have begun before the conservative simultaneous-array
    envelope was tightened.  The estimator's numerical result is unaffected;
    this replaces only a deterministic accounting field from (p, k, C).
    """
    cache: dict[tuple[int, int, int], int] = {}
    for row in records:
        if row.get("method") != "dual_bound":
            continue
        key = (int(row["p"]), int(row["k"]), 512)
        if key not in cache:
            cache[key] = DualBoundFORCE(
                key[0], key[1], calibration_size=key[2]
            ).calibration_peak_state_bytes_bound
        row["calibration_peak_state_bytes"] = cache[key]


def main() -> None:
    original_primary_paths = sorted(
        (PROJECT / "results/pbmc-shards").glob(
            "shard*/dual-bound-force-pbmc-full.json"
        )
    )
    rerun_primary_paths = sorted(
        (PROJECT / "results/pbmc-primary-rerun").glob(
            "seed*/dual-bound-force-pbmc-full.json"
        )
    )
    failed_attempt_paths = [
        path
        for path in original_primary_paths
        if json.loads(path.read_text(encoding="utf-8")).get("status")
        != "completed"
    ]
    primary_paths = [
        path
        for path in original_primary_paths + rerun_primary_paths
        if json.loads(path.read_text(encoding="utf-8")).get("status")
        == "completed"
    ]
    if not primary_paths:
        raise RuntimeError("no completed PBMC primary shards found")
    primary_payloads, primary_records = _load_records(primary_paths)
    _apply_final_state_accounting(primary_records)
    configuration, provenance = _validate_shared(primary_payloads)
    expected = {
        (seed, method)
        for seed in range(500, 510)
        for method in ("sketch_mad", "dual_bound")
    }
    observed = {(row["seed"], row["method"]) for row in primary_records if row.get("status") == "completed"}
    if (
        observed != expected
        or len(primary_records) != len(expected)
        or any(
            row.get("scenario") != "bounded_out_of_subspace"
            for row in primary_records
        )
    ):
        raise RuntimeError("PBMC primary shard coverage is incomplete")
    dual = {row["seed"]: row["subspace_error"] for row in primary_records if row["method"] == "dual_bound"}
    mad = {row["seed"]: row["subspace_error"] for row in primary_records if row["method"] == "sketch_mad"}
    differences = [float(dual[seed]) - float(mad[seed]) for seed in range(500, 510)]
    primary_criterion = {
        **paired_bootstrap_ci(differences, seed=77_500, resamples=10_000),
        "status": "evaluated",
        "difference_definition": "DualBound - MAD_Sketch",
        "expected_direction": "upper_95_below_zero",
    }
    primary_criterion["passed"] = primary_criterion["upper_95"] < 0.0
    primary_payload = envelope(
        experiment="dual_bound_force_pbmc",
        stage="confirmatory",
        status="completed",
        seeds=range(500, 510),
        parameters={
            "configuration": configuration,
            "completed_arm": "primary_bounded_out_of_subspace",
            "completed_methods": ["sketch_mad", "dual_bound"],
            "secondary_arms_status": "separate_bundle",
            "qualification": (
                "This bundle evaluates the preregistered PBMC submission gate. "
                "It does not imply completion of clean, casewise, cellwise, or in-subspace secondary arms."
            ),
            "state_accounting": (
                "final conservative simultaneous-array calibration peak envelope; "
                "deterministically recomputed from p, k, and C without changing outcomes"
            ),
            "resource_measurement_qualification": (
                "PBMC shard elapsed time and RSS are execution diagnostics from a "
                "concurrent bounded batch; the separately isolated timing bundle "
                "controls the throughput criterion"
            ),
        },
        results={
            "records": len(primary_records),
            "primary_criterion": primary_criterion,
            "shards": [str(path.resolve()) for path in primary_paths],
            "failed_attempts_retained": [
                str(path.resolve()) for path in failed_attempt_paths
            ],
        },
        provenance=provenance,
        preprocessing=primary_payloads[0]["preprocessing"],
        evidence_label="preregistered_confirmatory_primary_pbmc_evidence",
    )
    primary_outputs = write_bundle(
        primary_payload,
        output_dir=PROJECT / "results/confirmatory",
        stem="dual-bound-force-pbmc-primary-full",
        tidy_rows=sorted(primary_records, key=lambda row: (row["seed"], row["method"])),
    )
    print(primary_outputs["json"])

    secondary_paths = sorted(
        (PROJECT / "results/pbmc-secondary-isolated").glob(
            "seed*/*/*/dual-bound-force-pbmc-full.json"
        )
    )
    baseline_paths = sorted((PROJECT / "results/pbmc-bounded-baseline-shards").glob("seed*/dual-bound-force-pbmc-full.json"))
    if len(secondary_paths) != 160 or len(baseline_paths) != 10:
        print(
            "PBMC secondary bundle pending: expected 160 atomic secondary "
            "and ten bounded-baseline shards"
        )
        return
    secondary_payloads, secondary_records = _load_records(secondary_paths)
    baseline_payloads, baseline_records = _load_records(baseline_paths)
    _apply_final_state_accounting(secondary_records)
    _apply_final_state_accounting(baseline_records)
    all_payloads = primary_payloads + secondary_payloads + baseline_payloads
    _validate_shared(all_payloads)
    records = primary_records + secondary_records + baseline_records
    methods = ("vanilla_fd", "rfd", "sketch_mad", "dual_bound")
    scenarios = ("clean", "casewise", "cellwise", "bounded_out_of_subspace", "in_subspace")
    expected_full = {
        (seed, scenario, method)
        for seed in range(500, 510)
        for scenario in scenarios
        for method in methods
    }
    observed_full = {
        (row["seed"], row["scenario"], row["method"])
        for row in records
        if row.get("status") == "completed"
    }
    if observed_full != expected_full or len(records) != len(expected_full):
        raise RuntimeError("PBMC complete secondary coverage is inconsistent")
    full_payload = envelope(
        experiment="dual_bound_force_pbmc",
        stage="confirmatory",
        status="completed",
        seeds=range(500, 510),
        parameters={
            "configuration": configuration,
            "methods": list(methods),
            "scenarios": list(scenarios),
            "primary_criterion_scenario": "bounded_out_of_subspace",
            "state_accounting": (
                "final conservative simultaneous-array calibration peak envelope; "
                "deterministically recomputed from p, k, and C without changing outcomes"
            ),
            "resource_measurement_qualification": (
                "PBMC shard elapsed time and RSS are execution diagnostics from a "
                "concurrent bounded batch; the separately isolated timing bundle "
                "controls the throughput criterion"
            ),
        },
        results={
            "records": len(records),
            "primary_criterion": primary_criterion,
            "shards": [str(path.resolve()) for path in primary_paths + secondary_paths + baseline_paths],
            "failed_attempts_retained": [
                str(path.resolve()) for path in failed_attempt_paths
            ],
        },
        provenance=provenance,
        preprocessing=primary_payloads[0]["preprocessing"],
        evidence_label="preregistered_confirmatory_pbmc_evidence",
    )
    full_outputs = write_bundle(
        full_payload,
        output_dir=PROJECT / "results/confirmatory",
        stem="dual-bound-force-pbmc-full",
        tidy_rows=sorted(records, key=lambda row: (row["seed"], row["scenario"], row["method"])),
    )
    print(full_outputs["json"])


if __name__ == "__main__":
    main()
