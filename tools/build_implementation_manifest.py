#!/usr/bin/env python3
"""Build current scientific and distribution fingerprints for the public tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance/CURRENT_IMPLEMENTATION_MANIFEST.json"
HISTORICAL_SHA256 = "d69c9597fbd51fa20a870f22ceeec0e55fc7b300eb30c93fe34ffede2bed51d1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(paths: list[Path]) -> tuple[list[dict], str]:
    output = []
    accumulator = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        output.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
        accumulator.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return output, accumulator.hexdigest()


def main() -> None:
    scientific_paths = sorted((ROOT / "src/dual_bound_force").rglob("*.py"))
    scientific_paths.append(ROOT / "src/dual_bound_force/experiments/baseline_sources.json")
    scientific_paths.extend(
        [
            ROOT / "requirements.lock",
            ROOT / "preregistration/frozen_configuration.json",
            ROOT / "preregistration/frozen_configuration.json.sha256",
        ]
    )
    distribution_paths = [
        ROOT / "pyproject.toml",
        ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "tools/build_implementation_manifest.py",
        ROOT / "tools/verify_disclosure.py",
        ROOT / "tools/generate_paper_assets.py",
    ]
    scientific_records, scientific_hash = records(scientific_paths)
    distribution_records, distribution_hash = records(distribution_paths)
    payload = {
        "schema_version": "2.0",
        "package_version": "0.1.0",
        "historical_result_manifest": {
            "sha256": HISTORICAL_SHA256,
            "qualification": "immutable private scientific provenance for frozen paper results",
        },
        "scientific": {
            "source_tree_sha256": scientific_hash,
            "files": scientific_records,
            "scope": "public current scientific code, frozen configuration, and runtime lock",
        },
        "distribution": {
            "source_tree_sha256": distribution_hash,
            "files": distribution_records,
            "scope": "public packaging, MIT licensing, documentation, and verification tools",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
