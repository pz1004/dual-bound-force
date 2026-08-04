#!/usr/bin/env python3
"""Create the post-review statistical analysis without rewriting frozen results."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dual_bound_force.experiments.metrics import (
    exact_one_sided_sign_flip_p,
    exact_one_sided_sign_p,
    holm_adjust,
    normalized_projection_error,
    paired_mean_inference,
)
from dual_bound_force.experiments.reporting import envelope, sha256_file, write_bundle


MAIN = PROJECT / "results/confirmatory/dual-bound-force-confirmatory-full.json"
PBMC = PROJECT / "results/confirmatory/dual-bound-force-pbmc-certified-full.json"
TIMING = PROJECT / "results/confirmatory/dual-bound-force-timing-full.json"
OUTPUT_STEM = "dual-bound-force-statistical-reanalysis"
HYPOTHESES = (
    ("structural_out_of_subspace_improvement", "bounded_out_of_subspace", 1.0),
    ("casewise_noninferiority_5pct", "casewise_cauchy", 1.05),
    ("cellwise_noninferiority_5pct", "cellwise_cauchy", 1.05),
    ("clean_noninferiority_5pct", "clean", 1.05),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard constant {value} in {path}")
        ),
    )


def typed(value: str) -> Any:
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


def read_rows(json_path: Path) -> list[dict[str, Any]]:
    with json_path.with_suffix(".raw.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        return [
            {key: typed(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def synthetic_units(
    rows: list[dict[str, Any]],
    *,
    hypothesis: str,
    scenario: str,
    multiplier: float,
    spectrum: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("tier") != "sketch"
            or row.get("k") != 20
            or row.get("scenario") != scenario
            or row.get("method") not in {"dual_bound", "sketch_mad"}
            or (spectrum is not None and row.get("spectrum") != spectrum)
        ):
            continue
        grouped[(str(row["method"]), int(row["seed"]))].append(
            float(row["subspace_error"])
        )
    seeds = sorted(
        set(seed for method, seed in grouped if method == "dual_bound")
        & set(seed for method, seed in grouped if method == "sketch_mad")
    )
    units = []
    for seed in seeds:
        dual_values = grouped[("dual_bound", seed)]
        mad_values = grouped[("sketch_mad", seed)]
        expected_per_method = 2 if spectrum is None else 1
        if len(dual_values) != expected_per_method or len(mad_values) != expected_per_method:
            raise RuntimeError(
                f"{hypothesis} seed {seed} has incomplete spectrum coverage"
            )
        dual = float(np.mean(dual_values))
        mad = float(np.mean(mad_values))
        units.append(
            {
                "status": "completed",
                "source": "synthetic_confirmatory_tidy_csv",
                "hypothesis": hypothesis,
                "scope": "equal_spectrum_average" if spectrum is None else "spectrum_stratum",
                "spectrum": "strong_and_weak_equal_weight" if spectrum is None else spectrum,
                "seed": seed,
                "multiplier": multiplier,
                "dual_error": dual,
                "mad_error": mad,
                "difference": dual - multiplier * mad,
            }
        )
    if len(units) != 30:
        raise RuntimeError(f"{hypothesis} expected 30 paired seeds, found {len(units)}")
    return units


def verified_pbmc_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reference = payload["preprocessing"]["reference_svd"]
    parameters = reference["parameters"]
    reference_path = PROJECT / "data/pbmc68k" / (
        "reference-certified-v2-"
        f"rank{parameters['rank']}-seeds{parameters['seeds'][0]}-"
        f"{parameters['seeds'][1]}.npz"
    )
    if (
        reference.get("mode")
        != "checksummed_two_start_rayleigh_ritz_certified_cache"
        or not reference["diagnostics"].get("certified")
        or sha256_file(reference_path) != reference["array_sha256"]
    ):
        raise RuntimeError("certified PBMC reference identity check failed")
    with np.load(reference_path, allow_pickle=False) as values:
        reference_basis = np.asarray(values["basis"], dtype=float)
    rows = read_rows(PBMC)
    recomputed: dict[tuple[int, str], float] = {}
    maximum_difference = 0.0
    for row in rows:
        basis_path = Path(str(row["basis_path"]))
        if sha256_file(basis_path) != row["basis_sha256"]:
            raise RuntimeError(f"PBMC fitted-basis checksum failed: {basis_path}")
        with np.load(basis_path, allow_pickle=False) as values:
            basis = np.asarray(values["basis"], dtype=float)
        value = normalized_projection_error(reference_basis, basis)
        maximum_difference = max(
            maximum_difference,
            abs(value - float(row["recomputed_subspace_error"])),
        )
        recomputed[(int(row["seed"]), str(row["method"]))] = value
    if maximum_difference > 1e-12:
        raise RuntimeError("PBMC retained-basis metric reconstruction changed")
    expected = {
        (seed, method)
        for seed in range(500, 510)
        for method in ("dual_bound", "sketch_mad")
    }
    if set(recomputed) != expected:
        raise RuntimeError("certified PBMC coverage is incomplete")
    return [
        {
            "status": "completed",
            "source": "certified_pbmc_retained_bases",
            "hypothesis": "pbmc_structural_improvement",
            "scope": "fixed_dataset_fault_replicate",
            "spectrum": "not_applicable",
            "seed": seed,
            "multiplier": 1.0,
            "dual_error": recomputed[(seed, "dual_bound")],
            "mad_error": recomputed[(seed, "sketch_mad")],
            "difference": (
                recomputed[(seed, "dual_bound")]
                - recomputed[(seed, "sketch_mad")]
            ),
        }
        for seed in range(500, 510)
    ]


def infer(units: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    values = [float(row["difference"]) for row in units]
    result = paired_mean_inference(
        values,
        alternative="less",
        bootstrap_seed=seed,
        bootstrap_resamples=10_000,
    )
    result.update(
        {
            "status": "evaluated",
            "difference_definition": (
                "DualBound - multiplier * MAD_Sketch; negative favors the "
                "declared alternative"
            ),
            "historical_percentile_interval_rule_passed": bool(
                result["percentile_bootstrap_95"][1] < 0.0
            ),
        }
    )
    return result


def main() -> None:
    main_payload = read_json(MAIN)
    pbmc_payload = read_json(PBMC)
    timing_payload = read_json(TIMING)
    if any(
        payload.get("status") != "completed"
        for payload in (main_payload, pbmc_payload, timing_payload)
    ):
        raise RuntimeError("completed main, certified PBMC, and timing bundles are required")
    rows = read_rows(MAIN)
    tidy: list[dict[str, Any]] = []
    hypotheses: dict[str, dict[str, Any]] = {}
    interactions: dict[str, dict[str, Any]] = {}
    for index, (name, scenario, multiplier) in enumerate(HYPOTHESES):
        aggregate = synthetic_units(
            rows,
            hypothesis=name,
            scenario=scenario,
            multiplier=multiplier,
            spectrum=None,
        )
        tidy.extend(aggregate)
        hypotheses[name] = {
            **infer(aggregate, seed=93_000 + index),
            "scenario": scenario,
            "multiplier": multiplier,
            "estimand": "equal-weight mean across strong and weak spectra within seed",
            "margin_qualification": (
                "5% is a prospectively selected engineering tolerance in normalized "
                "projection error, not a biological or clinical minimum-important difference"
                if multiplier == 1.05
                else "superiority against zero paired difference"
            ),
        }
        interactions[name] = {}
        for offset, spectrum in enumerate(("strong", "weak")):
            stratum = synthetic_units(
                rows,
                hypothesis=name,
                scenario=scenario,
                multiplier=multiplier,
                spectrum=spectrum,
            )
            tidy.extend(stratum)
            interactions[name][spectrum] = {
                **infer(stratum, seed=94_000 + 10 * index + offset),
                "scenario": scenario,
                "multiplier": multiplier,
                "estimand": f"mean paired difference within the {spectrum}-eigengap stratum",
                "descriptive_interaction_analysis": True,
            }

    pbmc_units = verified_pbmc_units(pbmc_payload)
    tidy.extend(pbmc_units)
    pbmc_inference = infer(pbmc_units, seed=93_004)
    differences = [float(row["difference"]) for row in pbmc_units]
    pbmc_inference.update(
        {
            "scenario": "bounded_out_of_subspace",
            "multiplier": 1.0,
            "estimand": "mean paired fault-replicate difference on one fixed PBMC matrix",
            "exact_one_sided_sign_p": exact_one_sided_sign_p(differences),
            "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(differences),
            "mean_dual_error": float(np.mean([row["dual_error"] for row in pbmc_units])),
            "mean_mad_error": float(np.mean([row["mad_error"] for row in pbmc_units])),
            "absolute_recovery_qualification": (
                "both normalized projection errors remain near one; the result is a "
                "small conditional relative advantage, not satisfactory recovery"
            ),
        }
    )
    hypotheses["pbmc_structural_improvement"] = pbmc_inference

    adjusted = holm_adjust(
        {name: float(value["one_sided_p"]) for name, value in hypotheses.items()}
    )
    for name, adjusted_p in adjusted.items():
        hypotheses[name]["holm_adjusted_one_sided_p"] = adjusted_p
        hypotheses[name]["passed"] = bool(
            hypotheses[name]["estimate"] < 0.0 and adjusted_p < 0.05
        )
    scientific_family = list(name for name, _, _ in HYPOTHESES) + [
        "pbmc_structural_improvement"
    ]
    old_criteria = main_payload["results"]["primary_criteria"]
    engineering = {
        "throughput_70pct": {
            "status": "evaluated",
            "ratio": float(timing_payload["results"]["throughput_ratio_median"]),
            "threshold": 0.70,
            "threshold_met": bool(
                timing_payload["results"]["throughput_ratio_median"] >= 0.70
            ),
            "qualification": "hardware-qualified paired timing evidence; outside the Holm family",
        },
        "active_state_growth": {
            **old_criteria["state_growth_pk_plus_p"],
            "qualification": "deterministic byte-accounting consistency check; outside the Holm family",
        },
        "calibration_peak_growth": {
            **old_criteria["calibration_peak_growth_p_times_k_plus_c"],
            "qualification": "deterministic peak-envelope consistency check; outside the Holm family",
        },
    }
    payload = envelope(
        experiment="dual_bound_force_statistical_reanalysis",
        stage="post_review_confirmatory_reanalysis",
        status="completed",
        seeds=range(500, 530),
        parameters={
            "alpha": 0.05,
            "scientific_family": scientific_family,
            "primary_test": "one-sided paired t test of the mean",
            "multiplicity": "Holm step-down adjustment across exactly five scientific hypotheses",
            "effect_interval": "two-sided 95% paired t interval",
            "sensitivity_interval": "10,000-resample paired percentile bootstrap",
            "frozen_results_policy": "source bundles are read-only and were not rewritten",
        },
        results={
            "hypotheses": hypotheses,
            "spectrum_interactions": interactions,
            "engineering_evidence": engineering,
            "analysis_units": len(tidy),
            "decision_summary": {
                "supported": [name for name in scientific_family if hypotheses[name]["passed"]],
                "unsupported": [name for name in scientific_family if not hypotheses[name]["passed"]],
                "heterogeneous_gate_count_prohibited": True,
            },
        },
        provenance={
            "input_artifacts": {
                str(path.relative_to(PROJECT)): sha256_file(path)
                for path in (
                    MAIN,
                    MAIN.with_suffix(".raw.csv"),
                    PBMC,
                    PBMC.with_suffix(".raw.csv"),
                    TIMING,
                    TIMING.with_suffix(".raw.csv"),
                )
            }
        },
        preprocessing={
            "synthetic_analysis_unit": "seed after equal weighting of strong and weak spectra",
            "pbmc_analysis_unit": "fault replicate seed on one fixed matrix",
            "pbmc_reference": pbmc_payload["preprocessing"]["reference_svd"],
        },
        evidence_label="authorized_post_review_statistical_correction_no_retuning",
    )
    outputs = write_bundle(
        payload,
        output_dir=PROJECT / "results/confirmatory",
        stem=OUTPUT_STEM,
        tidy_rows=sorted(
            tidy,
            key=lambda row: (
                row["hypothesis"], row["scope"], row["spectrum"], int(row["seed"])
            ),
        ),
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
