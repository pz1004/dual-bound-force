#!/usr/bin/env python3
"""Verify the minimal public-disclosure staging tree."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    manifest = strict_json(ROOT / "DISCLOSURE_MANIFEST.json")
    records = manifest["files"]
    recorded_paths = {record["path"] for record in records}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not is_ephemeral(path)
        and path.name != "DISCLOSURE_MANIFEST.json"
        and not path.relative_to(ROOT).as_posix().startswith("generated-paper-assets/")
    }
    if actual_paths != recorded_paths:
        missing = sorted(recorded_paths - actual_paths)
        unexpected = sorted(actual_paths - recorded_paths)
        raise RuntimeError(f"manifest mismatch: missing={missing}, unexpected={unexpected}")

    for record in records:
        path = ROOT / record["path"]
        observed = sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"hash mismatch for {record['path']}")
        if record["category"] == "paper_result":
            text = path.read_text(encoding="utf-8")
            if "/home/" in text or "\\Users\\" in text:
                raise RuntimeError(f"machine-local path remains in {record['path']}")
            if path.suffix == ".json":
                strict_json(path)
            elif path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if not reader.fieldnames or next(reader, None) is None:
                        raise RuntimeError(f"empty CSV result {record['path']}")

    dependencies = manifest["paper_asset_dependencies"]
    for asset, inputs in dependencies.items():
        if not inputs:
            raise RuntimeError(f"asset has no declared input: {asset}")
        for item in inputs:
            if item not in recorded_paths:
                raise RuntimeError(f"undeclared input {item} for {asset}")

    forbidden = {"manuscript", "output", "data", "literature", "supplement"}
    leaked = sorted(path for path in actual_paths if Path(path).parts[0] in forbidden)
    if leaked:
        raise RuntimeError(f"excluded research material leaked into disclosure: {leaked}")
    snapshots = sorted(
        path for path in actual_paths if "_snapshots" in Path(path).parts
    )
    if snapshots:
        raise RuntimeError(f"prior-study source snapshots leaked into disclosure: {snapshots}")
    if manifest.get("license") != "MIT":
        raise RuntimeError("disclosure license is not MIT")
    if manifest.get("repository") != "https://github.com/pz1004/dual-bound-force":
        raise RuntimeError("unexpected disclosure repository URL")
    print(
        json.dumps(
            {
                "status": "verified",
                "file_count": len(records),
                "source_file_count": sum(r["category"] == "source" for r in records),
                "paper_result_file_count": sum(
                    r["category"] == "paper_result" for r in records
                ),
                "paper_asset_count": len(dependencies),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
