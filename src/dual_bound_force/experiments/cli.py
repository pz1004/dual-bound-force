"""Command-line entry point for development and confirmatory study tiers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from .reporting import envelope, write_bundle
from .study import (
    CONFIRMATORY_SEEDS,
    DEVELOPMENT_SEEDS,
    evaluate_primary_criteria,
    load_frozen_configuration,
    run_calibration_audit,
    run_confirmatory_synthetic,
    run_development,
    run_epoch_audit,
    summarize,
    write_frozen_configuration,
)


def _seeds(value: str | None, defaults: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return defaults
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("--seeds cannot be empty")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "confirmatory"), required=True)
    parser.add_argument("--tiers", default="dense,sketch,calibration,epoch")
    parser.add_argument("--seeds")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/confirmatory")
    parser.add_argument("--configuration", default="preregistration/frozen_configuration.json")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--memory-limit-mb", type=int, default=4096)
    return parser


def run(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    tiers = tuple(item.strip() for item in args.tiers.split(",") if item.strip())
    permitted = {
        "dense", "sketch", "oracle", "threshold", "retention",
        "calibration", "epoch"
    }
    if not tiers or not set(tiers) <= permitted:
        raise ValueError(f"--tiers must be drawn from {sorted(permitted)}")
    if args.stage == "development":
        seeds = _seeds(args.seeds, (300, 301) if args.smoke else DEVELOPMENT_SEEDS)
        records, selection = run_development(seeds=seeds, smoke=args.smoke)
        frozen = None
        if not args.smoke:
            frozen = write_frozen_configuration(
                selection, path=args.configuration, smoke=False
            )
        results = {
            "selection": selection,
            "frozen_configuration": frozen,
            "summary": summarize(records),
        }
        parameters = {
            "stage": args.stage,
            "seeds": list(seeds),
            "lambda_grid": [1.5, 2.0, 3.0, 4.0],
            "smoke": args.smoke,
        }
        payload = envelope(
            experiment="dual_bound_force",
            stage=args.stage,
            status="completed",
            seeds=seeds,
            parameters=parameters,
            results=results,
            evidence_label="development_tuning_not_confirmatory_evidence",
        )
        return payload, records

    seeds = _seeds(args.seeds, (500, 501) if args.smoke else CONFIRMATORY_SEEDS)
    if args.smoke:
        configuration = {
            "parallel_lambda": 3.0,
            "residual_lambda": 3.0,
            "marginal_lambda": 3.0,
            "qualification": "smoke configuration; not confirmatory",
        }
    else:
        configuration = load_frozen_configuration(args.configuration)
    records = run_confirmatory_synthetic(
        seeds=seeds,
        configuration=configuration,
        tiers=tiers,
        smoke=args.smoke,
        jobs=args.jobs,
    )
    if "calibration" in tiers:
        records.extend(
            run_calibration_audit(
                seeds=seeds,
                configuration=configuration,
                smoke=args.smoke,
            )
        )
    if "epoch" in tiers:
        records.extend(
            run_epoch_audit(
                seeds=seeds,
                configuration=configuration,
                smoke=args.smoke,
            )
        )
    primary_criteria = (
        evaluate_primary_criteria(
            records,
            bootstrap_resamples=(
                min(args.bootstrap_resamples, 1000)
                if args.smoke
                else args.bootstrap_resamples
            ),
        )
        if set(tiers) & {"dense", "sketch"}
        else {
            "status": "not_evaluated",
            "reason": "separate ablation/sensitivity artifacts do not re-test primary gates",
        }
    )
    results = {
        "summary": summarize(records),
        "primary_criteria": primary_criteria,
        "unimplemented_comparator": {
            "method": "residual_based_online_robust_pca",
            "status": "not_reproducible",
            "reason": (
                "No audited implementation with matching streaming covariance target and "
                "dependency/runtime contract was available; no surrogate was invented."
            ),
        },
    }
    payload = envelope(
        experiment="dual_bound_force",
        stage=args.stage,
        status="completed",
        seeds=seeds,
        parameters={
            "stage": args.stage,
            "tiers": list(tiers),
            "configuration": configuration,
            "smoke": args.smoke,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        results=results,
        evidence_label=(
            "smoke_verification_not_inferential"
            if args.smoke
            else "preregistered_confirmatory_evidence"
            if set(tiers) & {"dense", "sketch"}
            else "prespecified_secondary_confirmatory_evidence"
        ),
    )
    return payload, records


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    stem = f"dual-bound-force-{args.stage}-{'smoke' if args.smoke else 'full'}"
    seeds: tuple[int, ...] = ()
    try:
        payload, records = run(args)
    except Exception as exc:
        defaults = DEVELOPMENT_SEEDS if args.stage == "development" else CONFIRMATORY_SEEDS
        try:
            seeds = _seeds(args.seeds, defaults)
        except Exception:
            seeds = ()
        payload = envelope(
            experiment="dual_bound_force",
            stage=args.stage,
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
