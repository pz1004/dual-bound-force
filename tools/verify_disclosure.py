#!/usr/bin/env python3
"""Verify the public source tree and the paper-result disclosure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from disclosure_spec import (
    PAPER_ASSET_DEPENDENCIES,
    PAPER_RESULT_FILES,
    REPOSITORY_URL,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def is_ephemeral(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return any(
        part in {"__pycache__", ".pytest_cache", ".git", "build", "dist"}
        or part.endswith(".egg-info")
        for part in relative.parts
    )


def public_files() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not is_ephemeral(path)
        and not path.relative_to(ROOT).as_posix().startswith(
            "generated-paper-assets/"
        )
    }


def verify_results() -> None:
    expected = set(PAPER_RESULT_FILES)
    observed = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "results/paper").rglob("*")
        if path.is_file() and not is_ephemeral(path)
    }
    if observed != expected:
        raise RuntimeError(
            "paper-result set mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    for relative in PAPER_RESULT_FILES:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        if "/home/" in text or "\\Users\\" in text:
            raise RuntimeError(f"machine-local path remains in {relative}")
        if path.suffix == ".json":
            strict_json(path)
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or next(reader, None) is None:
                    raise RuntimeError(f"empty CSV result {relative}")

    for asset, inputs in PAPER_ASSET_DEPENDENCIES.items():
        if not inputs:
            raise RuntimeError(f"asset has no declared input: {asset}")
        unknown = set(inputs) - expected
        if unknown:
            raise RuntimeError(f"unknown inputs for {asset}: {sorted(unknown)}")


def verify_boundaries(files: set[str]) -> None:
    forbidden = {"manuscript", "output", "data", "literature", "supplement"}
    leaked = sorted(path for path in files if Path(path).parts[0] in forbidden)
    if leaked:
        raise RuntimeError(f"excluded research material leaked into disclosure: {leaked}")
    snapshots = sorted(path for path in files if "_snapshots" in Path(path).parts)
    if snapshots:
        raise RuntimeError(f"prior-study source snapshots leaked into disclosure: {snapshots}")
    if "DISCLOSURE_MANIFEST.json" in files:
        raise RuntimeError("repository-level disclosure manifest must not be tracked")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text:
        raise RuntimeError("repository license is not MIT")
    for relative in ("README.md", "pyproject.toml"):
        if REPOSITORY_URL not in (ROOT / relative).read_text(encoding="utf-8"):
            raise RuntimeError(f"repository URL missing from {relative}")


def verify_current_fingerprints() -> None:
    current = strict_json(ROOT / "provenance/CURRENT_IMPLEMENTATION_MANIFEST.json")
    if current.get("schema_version") != "2.0":
        raise RuntimeError("unexpected current implementation-manifest schema")
    for scope in ("scientific", "distribution"):
        for record in current[scope]["files"]:
            source = ROOT / record["path"]
            if not source.is_file() or sha256(source) != record["sha256"]:
                raise RuntimeError(
                    f"current {scope} implementation hash mismatch for "
                    f"{record['path']}"
                )


def main() -> None:
    files = public_files()
    verify_results()
    verify_boundaries(files)
    verify_current_fingerprints()
    print(
        json.dumps(
            {
                "status": "verified",
                "file_count": len(files),
                "source_file_count": sum(path.endswith(".py") for path in files),
                "paper_result_file_count": len(PAPER_RESULT_FILES),
                "paper_asset_count": len(PAPER_ASSET_DEPENDENCIES),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
