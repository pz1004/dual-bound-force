# Dual-Bound FORCE — Code and Paper-Result Disclosure

Repository: https://github.com/pz1004/dual-bound-force

This directory is the minimal disclosure staging tree for the Dual-Bound FORCE
study. It intentionally contains only executable research source, portable tests,
the frozen configuration, and the exact result inputs used by the final paper's
tables and figures.

It does **not** contain manuscript sources or PDFs, rendered tables/figures, raw
external datasets or caches, exploratory/development/smoke results, PBMC basis
arrays, review memoranda, or verification-replay bundles.

## Layout

- `src/dual_bound_force/`: estimator, theory, baselines, and experiment runners.
- `results/paper/`: only the JSON/CSV bundles mapped to final paper assets.
- `tools/generate_paper_assets.py`: regenerates all disclosed paper tables/figures.
- `tools/verify_disclosure.py`: validates hashes, strict JSON/CSV, path portability,
  and the allowlist.
- `DISCLOSURE_MANIFEST.json`: file hashes, original-result hashes, path-normalization
  records, and the asset-to-result dependency map.

## Verification

```bash
python3 tools/verify_disclosure.py
python3 tools/generate_paper_assets.py --output-dir generated-paper-assets
python3 -m pytest -q
```

Machine-local absolute paths in result metadata were replaced by repository-relative
paths. No numerical field, status, seed, parameter, checksum, or statistical decision
was changed. Both original and disclosed hashes are recorded in the manifest.

## Prior-study boundary

FORCE, MAD-FORCE, and Sketch-FORCE are separately distributed prior projects. Their
source code is not included here. The optional comparison runners require those
packages to be installed separately; the frozen paper-result inputs remain fully
available for table and figure regeneration without those packages.

Dual-Bound FORCE is released under the MIT License. The GitHub repository is the
intended code disclosure; no DOI or separate archival version is planned.
