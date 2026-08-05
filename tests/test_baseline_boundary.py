from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dual_bound_force.experiments.baseline_registry import (
    ExternalBaselineUnavailable,
    baseline_manifest,
    sketch_force_estimator_class,
    validate_baseline_source,
)
from dual_bound_force.experiments.baselines import fit_method


PROJECT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT / "src")
    return environment


def test_public_native_methods_run_without_prior_packages():
    matrix = np.random.default_rng(77).normal(size=(80, 12))
    for method in ("vanilla_fd", "rfd", "exact_mad_fd", "dual_bound"):
        result = fit_method(
            matrix,
            method=method,
            rank=2,
            k=4,
            calibration_size=16,
        )
        assert result.correlation.shape == (12, 12)
        assert np.isfinite(result.correlation).all()


def test_missing_or_wrong_external_sources_fail_explicitly(tmp_path):
    with pytest.raises(ExternalBaselineUnavailable, match="supply --baseline-dir"):
        sketch_force_estimator_class()
    source = tmp_path / "sketch-force/src/sketch_force/sketch_force.py"
    source.parent.mkdir(parents=True)
    source.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(ExternalBaselineUnavailable, match="hash mismatch|missing"):
        validate_baseline_source("sketch_force", tmp_path)


def test_every_public_cli_help_is_dependency_free():
    modules = ("cli", "audit_cli", "external_cli", "timing_cli", "structural_ceiling_cli")
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", f"dual_bound_force.experiments.{module}", "--help"],
            cwd=PROJECT,
            env=_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout


def test_native_method_subset_completes_and_missing_baseline_emits_failed_bundle(tmp_path):
    native_dir = tmp_path / "native"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dual_bound_force.experiments.cli",
            "--stage",
            "confirmatory",
            "--tiers",
            "dense",
            "--methods",
            "pearson,dual_bound",
            "--seeds",
            "500",
            "--smoke",
            "--output-dir",
            str(native_dir),
        ],
        cwd=PROJECT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(next(native_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["results"]["primary_criteria"]["status"] == "not_evaluated"

    failed_dir = tmp_path / "failed"
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dual_bound_force.experiments.cli",
            "--stage",
            "confirmatory",
            "--tiers",
            "dense",
            "--seeds",
            "500",
            "--smoke",
            "--output-dir",
            str(failed_dir),
        ],
        cwd=PROJECT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    failure_payload = json.loads(
        next(failed_dir.glob("*.json")).read_text(encoding="utf-8")
    )
    assert failure_payload["status"] == "failed"
    assert failure_payload["error"]["type"] == "ExternalBaselineUnavailable"
    assert list(failed_dir.glob("*.raw.csv"))


def test_baseline_manifest_records_nonidentical_public_commits():
    manifest = baseline_manifest()
    assert set(manifest["projects"]) == {"force", "mad_force", "sketch_force"}
    assert all(
        not record["public_commit_matches_generating_source"]
        for record in manifest["projects"].values()
    )
    path = PROJECT / "src/dual_bound_force/experiments/baseline_sources.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
