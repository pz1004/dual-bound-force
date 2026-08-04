"""Validate strict study artifacts and a frozen preregistration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .study import load_frozen_configuration


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--configuration")
    args = parser.parse_args(argv)
    if args.configuration:
        load_frozen_configuration(args.configuration)
    for raw in args.paths:
        path = Path(raw)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if payload.get("status") not in {
                "completed", "skipped_external", "not_reproducible", "failed", "timeout", "oom"
            }:
                raise ValueError(f"invalid status in {path}")
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                list(csv.reader(stream))
        else:
            raise ValueError(f"unsupported artifact {path}")
        print(path.resolve())


if __name__ == "__main__":
    main()
