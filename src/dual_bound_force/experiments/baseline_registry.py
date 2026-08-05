"""Lazy, source-validated access to prior FORCE-family estimators."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import importlib.util
from importlib.resources import files
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


class ExternalBaselineUnavailable(ImportError):
    """Raised when an exact recorded baseline source cannot be loaded."""


@lru_cache(maxsize=1)
def baseline_manifest() -> dict[str, Any]:
    resource = files("dual_bound_force.experiments").joinpath("baseline_sources.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_baseline_source(project: str, baseline_dir: str | Path) -> Path:
    manifest = baseline_manifest()["projects"]
    if project not in manifest:
        raise ValueError(f"unknown baseline project {project!r}")
    record = manifest[project]
    root = (Path(baseline_dir).expanduser().resolve() / record["directory"]).resolve()
    failures: list[str] = []
    for relative, expected in record["required_files"].items():
        source = root / relative
        if not source.is_file():
            failures.append(f"missing {relative}")
        else:
            actual = _sha256(source)
            if actual != expected:
                failures.append(f"hash mismatch {relative}: {actual}")
    if failures:
        details = "; ".join(failures)
        raise ExternalBaselineUnavailable(
            f"{project} does not match the frozen generating source under {root}: {details}"
        )
    return root


def _internal(module: str, attribute: str) -> Any | None:
    try:
        return getattr(importlib.import_module(module), attribute)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("dual_bound_force._snapshots"):
            return None
        raise


def _module_from_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExternalBaselineUnavailable(f"cannot construct an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_package(name: str, search_root: Path, expected_root: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "")).resolve()
        try:
            origin.relative_to(expected_root)
        except ValueError as exc:
            raise ExternalBaselineUnavailable(
                f"module {name!r} is already loaded from unvalidated source {origin}"
            ) from exc
        return existing
    sys.path.insert(0, str(search_root))
    try:
        module = importlib.import_module(name)
    finally:
        try:
            sys.path.remove(str(search_root))
        except ValueError:
            pass
    origin = Path(getattr(module, "__file__", "")).resolve()
    try:
        origin.relative_to(expected_root)
    except ValueError as exc:
        raise ExternalBaselineUnavailable(
            f"module {name!r} resolved outside validated source {expected_root}: {origin}"
        ) from exc
    return module


def _require_dir(baseline_dir: str | Path | None, project: str) -> Path:
    if baseline_dir is None:
        raise ExternalBaselineUnavailable(
            f"{project} requires the exact external sources recorded in baseline_sources.json; "
            "supply --baseline-dir. The public upstream commit is not source-identical to the "
            "frozen generating revision."
        )
    return validate_baseline_source(project, baseline_dir)


def force_estimator_class(baseline_dir: str | Path | None = None) -> type:
    if baseline_dir is None:
        internal = _internal("dual_bound_force._snapshots.force.core", "ForceEstimator")
        if internal is not None:
            return internal
    root = _require_dir(baseline_dir, "force")
    module = _module_from_file("_dual_bound_force_external_force_core", root / "src/force/core.py")
    return module.ForceEstimator


def mad_force_estimator_class(baseline_dir: str | Path | None = None) -> type:
    if baseline_dir is None:
        internal = _internal(
            "dual_bound_force._snapshots.mad_force.drift_mad_force", "DriftMADForce"
        )
        if internal is not None:
            return internal
    root = _require_dir(baseline_dir, "mad_force")
    module = _import_package("src", root, root / "src")
    return module.DriftMADForce


def sketch_force_estimator_class(baseline_dir: str | Path | None = None) -> type:
    if baseline_dir is None:
        internal = _internal(
            "dual_bound_force._snapshots.sketch_force.sketch_force", "SketchFORCE"
        )
        if internal is not None:
            return internal
    root = _require_dir(baseline_dir, "sketch_force")
    module = _import_package("sketch_force", root / "src", root / "src/sketch_force")
    return module.SketchFORCE
