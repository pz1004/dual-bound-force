"""Isolated one-thread timing and memory measurements for bounded-state methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

from dual_bound_force import DualBoundFORCE

from .baselines import _state_bytes
from .baseline_registry import sketch_force_estimator_class
from .reporting import envelope, write_bundle
from .study import load_frozen_configuration


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _worker(args: argparse.Namespace) -> None:
    if args.memory_limit_mb:
        try:
            import resource

            limit = int(args.memory_limit_mb * 1024**2)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            pass
    rng = np.random.default_rng(args.seed)
    basis, _ = np.linalg.qr(rng.normal(size=(args.p, args.rank)), mode="reduced")
    latent = rng.normal(size=(args.n, args.rank)) * np.sqrt(
        np.linspace(10.0, 2.0, args.rank)
    )
    matrix = latent @ basis.T + rng.normal(size=(args.n, args.p))
    def fresh_fit(values: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter_ns()
        if args.method == "sketch_mad":
            SketchFORCE = sketch_force_estimator_class(args.baseline_dir)
            estimator = SketchFORCE(
                args.p, args.k, lam=3.0, trim_mode="mad"
            )
            for row in values:
                estimator.update(row)
            correlation = estimator.get_correlation()
            calibration_peak = 0
        else:
            estimator = DualBoundFORCE(
                args.p,
                args.k,
                calibration_size=args.calibration_size,
                parallel_lambda=args.parallel_lambda,
                residual_lambda=args.residual_lambda,
            )
            estimator.fit(values)
            correlation = estimator.get_correlation()
            calibration_peak = estimator.calibration_peak_state_bytes_bound
        elapsed = (time.perf_counter_ns() - started) / 1e9
        return {
            "elapsed_seconds": elapsed,
            "throughput_rows_per_second": len(values) / max(elapsed, 1e-12),
            "state_bytes": _state_bytes(estimator),
            "calibration_peak_state_bytes": int(calibration_peak),
            "output_bytes": int(correlation.nbytes + args.p * args.rank * 8),
        }

    # Each measured child warms the same method in a throwaway estimator, then
    # times a fresh estimator. Target reconstruction and diagnostic masks are
    # deliberately outside this path.
    warm_rows = max(args.calibration_size + 2, 2 * args.k + 2)
    if warm_rows < len(matrix):
        fresh_fit(matrix[:warm_rows])
    try:
        import psutil

        process = psutil.Process()
        start_rss = int(process.memory_info().rss)
    except ImportError:
        process = None
        start_rss = 0
    result = fresh_fit(matrix)
    end_rss = int(process.memory_info().rss) if process is not None else 0
    print(
        json.dumps(
            {
                "status": "completed",
                "method": args.method,
                "repetition": args.repetition,
                "seed": args.seed,
                "n": args.n,
                "p": args.p,
                "rank": args.rank,
                "k": args.k,
                **result,
                "worker_incremental_rss_bytes": max(0, end_rss - start_rss),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


def _run_child(command: list[str], *, timeout: float) -> dict[str, Any]:
    environment = os.environ.copy()
    for name in THREAD_VARIABLES:
        environment[name] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    peak_rss = 0
    start = time.monotonic()
    ps_process = None
    try:
        import psutil

        ps_process = psutil.Process(process.pid)
    except (ImportError, Exception):
        pass
    while process.poll() is None:
        if time.monotonic() - start > timeout:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            stdout, stderr = process.communicate()
            return {
                "status": "timeout",
                "timeout_seconds": float(timeout),
                "peak_rss_bytes": int(peak_rss),
                "stdout": stdout[-2000:],
                "stderr": stderr[-2000:],
            }
        if ps_process is not None:
            try:
                rss = int(ps_process.memory_info().rss)
                for child in ps_process.children(recursive=True):
                    rss += int(child.memory_info().rss)
                peak_rss = max(peak_rss, rss)
            except Exception:
                pass
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        try:
            record = json.loads(stdout.strip().splitlines()[-1])
            record["peak_rss_bytes"] = int(peak_rss)
            return record
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"worker output parse failure: {exc}",
                "peak_rss_bytes": int(peak_rss),
                "stdout": stdout[-2000:],
                "stderr": stderr[-2000:],
            }
    likely_oom = process.returncode in (-signal.SIGKILL, 137) or "MemoryError" in stderr
    return {
        "status": "oom" if likely_oom else "failed",
        "returncode": int(process.returncode),
        "peak_rss_bytes": int(peak_rss),
        "stdout": stdout[-2000:],
        "stderr": stderr[-2000:],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=("sketch_mad", "dual_bound"))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--parallel-lambda", type=float, default=3.0)
    parser.add_argument("--residual-lambda", type=float, default=3.0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--configuration", default="preregistration/frozen_configuration.json")
    parser.add_argument(
        "--baseline-dir",
        help="directory containing hash-validated external baseline source trees",
    )
    parser.add_argument("--output-dir", default="results/confirmatory")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--memory-limit-mb", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.worker:
        if args.method is None:
            raise ValueError("--method is required for a worker")
        _worker(args)
        return
    if args.runs <= 0 or args.timeout_seconds <= 0 or args.memory_limit_mb <= 0:
        raise ValueError("runs, timeout, and memory limit must be positive")
    configuration = (
        {"parallel_lambda": 3.0, "residual_lambda": 3.0, "qualification": "smoke"}
        if args.smoke
        else load_frozen_configuration(args.configuration)
    )
    runs = 2 if args.smoke else args.runs
    dimensions = {
        "n": 400 if args.smoke else args.n,
        "p": 20 if args.smoke else args.p,
        "rank": 3 if args.smoke else args.rank,
        "k": 5 if args.smoke else args.k,
        "calibration_size": 40 if args.smoke else args.calibration_size,
    }
    records = []
    for method in ("sketch_mad", "dual_bound"):
        for repetition in range(runs):
            command = [
                sys.executable,
                "-m",
                "dual_bound_force.experiments.timing_cli",
                "--worker",
                "--method",
                method,
                "--repetition",
                str(repetition),
                "--seed",
                str(800 + repetition),
                "--n",
                str(dimensions["n"]),
                "--p",
                str(dimensions["p"]),
                "--rank",
                str(dimensions["rank"]),
                "--k",
                str(dimensions["k"]),
                "--calibration-size",
                str(dimensions["calibration_size"]),
                "--parallel-lambda",
                str(configuration["parallel_lambda"]),
                "--residual-lambda",
                str(configuration["residual_lambda"]),
                "--memory-limit-mb",
                str(args.memory_limit_mb),
            ]
            if args.baseline_dir:
                command.extend(["--baseline-dir", args.baseline_dir])
            record = _run_child(command, timeout=args.timeout_seconds)
            record.setdefault("method", method)
            record.setdefault("repetition", repetition)
            record["worker_threads_forced_to_one"] = True
            records.append(record)
    completed = [row for row in records if row["status"] == "completed"]
    ratios = []
    for repetition in range(runs):
        by_method = {
            row["method"]: row
            for row in completed
            if row["repetition"] == repetition
        }
        if set(by_method) == {"sketch_mad", "dual_bound"}:
            ratios.append(
                by_method["dual_bound"]["throughput_rows_per_second"]
                / max(by_method["sketch_mad"]["throughput_rows_per_second"], 1e-12)
            )
    summary = {
        "requested_runs_per_method": runs,
        "completed_records": len(completed),
        "throughput_ratio_median": float(np.median(ratios)) if ratios else None,
        "throughput_criterion_70pct_passed": bool(ratios and np.median(ratios) >= 0.70),
        "statuses": {status: sum(row["status"] == status for row in records) for status in {row["status"] for row in records}},
    }
    payload = envelope(
        experiment="dual_bound_force_timing",
        stage="confirmatory",
        status="completed",
        seeds=range(800, 800 + runs),
        parameters={**dimensions, "runs": runs, "configuration": configuration, "one_thread": True},
        results=summary,
        evidence_label=("smoke_verification_not_inferential" if args.smoke else "preregistered_resource_evidence"),
    )
    payload["environment"]["worker_thread_environment"] = {
        name: "1" for name in THREAD_VARIABLES
    }
    paths = write_bundle(
        payload,
        output_dir=args.output_dir,
        stem=f"dual-bound-force-timing-{'smoke' if args.smoke else 'full'}",
        tidy_rows=records,
    )
    print(paths["json"])


if __name__ == "__main__":
    main()
