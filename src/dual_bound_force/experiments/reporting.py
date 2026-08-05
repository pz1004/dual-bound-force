"""Strict machine-readable artifact and environment utilities."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np


ALLOWED_STATUSES = {
    "completed",
    "skipped_external",
    "not_reproducible",
    "failed",
    "timeout",
    "oom",
}
DEPENDENCIES = ("numpy", "scipy", "numba", "scikit-learn", "psutil")
PROJECT = Path(__file__).resolve().parents[3]
IMPLEMENTATION_MANIFEST = PROJECT / "provenance/CURRENT_IMPLEMENTATION_MANIFEST.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return strict_value(value.item())
    if isinstance(value, np.ndarray):
        return strict_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("artifact contains NaN or Infinity")
    return value


def environment_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for dependency in DEPENDENCIES:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "dependency_versions": versions,
    }


def implementation_provenance() -> dict[str, Any]:
    """Return the exact implementation fingerprint attached to new bundles."""

    if not IMPLEMENTATION_MANIFEST.is_file():
        return {"status": "manifest_missing"}
    payload = json.loads(
        IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    return {
        "status": "recorded",
        "path": str(IMPLEMENTATION_MANIFEST.relative_to(PROJECT)),
        "manifest_sha256": sha256_file(IMPLEMENTATION_MANIFEST),
        "source_tree_sha256": payload["scientific"]["source_tree_sha256"],
        "package_version": payload["package_version"],
    }


def envelope(
    *,
    experiment: str,
    stage: str,
    status: str,
    seeds: Iterable[int],
    parameters: dict[str, Any],
    results: Any,
    provenance: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    evidence_label: str,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported status {status}")
    payload = {
        "schema_version": "1.0",
        "experiment": experiment,
        "stage": stage,
        "status": status,
        "evidence_label": evidence_label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": list(sys.argv),
        "seeds": [int(seed) for seed in seeds],
        "parameters": parameters,
        "environment": environment_metadata(),
        "provenance": {
            **(provenance or {}),
            "implementation": implementation_provenance(),
        },
        "preprocessing": preprocessing or {},
        "results": results,
    }
    if error is not None:
        payload["error"] = error
    return strict_value(payload)


def write_bundle(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    stem: str,
    tidy_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    summary_path = root / f"{stem}.summary.csv"
    raw_path = root / f"{stem}.raw.csv"
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    strict = strict_value(payload)
    json_tmp.write_text(
        json.dumps(strict, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_rows = _flatten(strict)
    with summary_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "value"))
        writer.writerows(summary_rows)
    rows = [] if tidy_rows is None else [strict_value(row) for row in tidy_rows]
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with raw_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    # Re-parse before atomic replacement as the final serialization gate.
    json.loads(json_tmp.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    json_tmp.replace(json_path)
    summary_tmp.replace(summary_path)
    raw_tmp.replace(raw_path)
    return {
        "json": str(json_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "raw_csv": str(raw_path.resolve()),
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _flatten(value: Any, path: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{path}.{key}" if path else str(key)
            rows.extend(_flatten(value[key], child))
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            rows.extend(_flatten(child_value, f"{path}[{index}]"))
    else:
        rendered = "" if value is None else str(value)
        rows.append((path, rendered))
    return rows
