#!/usr/bin/env python3
"""Regenerate every final-paper table and figure from disclosed result inputs.

This is the portable disclosure entry point.  It deliberately generates assets
into a user-selected directory and does not read manuscript sources, cached data,
or any result bundle outside the allowlist recorded in DISCLOSURE_MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paper_assets as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/paper"

CONFIRMATORY = RESULTS / "confirmatory.json"
ORACLE = RESULTS / "oracle.json"
THRESHOLD = RESULTS / "threshold.json"
RETENTION = RESULTS / "retention.json"
STATISTICAL = RESULTS / "statistical_reanalysis.json"
PBMC_CERTIFIED = RESULTS / "pbmc_certified.json"
PBMC_SECONDARY = RESULTS / "pbmc_secondary.json"
TIMING = RESULTS / "timing.json"
STRUCTURAL = RESULTS / "structural_ceiling.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def require_completed(path: Path) -> dict[str, Any]:
    payload = strict_json(path)
    if payload.get("status") != "completed":
        raise RuntimeError(f"paper input is not completed: {path}")
    return payload


def make_paired_contrast_figure(
    statistical: dict[str, Any], output: Path
) -> None:
    hypotheses = statistical["results"]["hypotheses"]
    strata = statistical["results"]["spectrum_interactions"]
    ordered = (
        ("structural_out_of_subspace_improvement", "Bounded structural"),
        ("casewise_noninferiority_5pct", "Casewise NI"),
        ("cellwise_noninferiority_5pct", "Cellwise NI"),
        ("clean_noninferiority_5pct", "Clean NI"),
    )
    series = (
        ("Equal-spectrum", "aggregate", "#1565c0", "o", -0.18),
        ("Strong gap", "strong", "#ef6c00", "s", 0.0),
        ("Weak gap", "weak", "#2e7d32", "^", 0.18),
    )
    fig, axis = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    positions = list(range(len(ordered)))
    for label, stratum, color, marker, offset in series:
        estimates: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        y: list[float] = []
        for index, (hypothesis, _) in enumerate(ordered):
            record = (
                hypotheses[hypothesis]
                if stratum == "aggregate"
                else strata[hypothesis][stratum]
            )
            if int(record["n_pairs"]) != 30:
                raise RuntimeError("paired figure requires 30 seed-level pairs")
            estimate = float(record["estimate"])
            lo, hi = (float(value) for value in record["paired_t_95"])
            estimates.append(estimate)
            lower.append(estimate - lo)
            upper.append(hi - estimate)
            y.append(index + offset)
        axis.errorbar(
            estimates,
            y,
            xerr=[lower, upper],
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=5.5,
            elinewidth=1.4,
            capsize=2.5,
            linestyle="none",
            label=label,
        )
    axis.axvline(0.0, color="#424242", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions, [label for _, label in ordered])
    axis.invert_yaxis()
    axis.set_xlabel("Paired contrast (negative favors the declared alternative)")
    axis.set_title("Confirmatory paired-mean contrasts with 95% paired-t intervals")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_structural_tables(structural: dict[str, Any], output: Path) -> None:
    results = structural["results"]
    means = results["means"]
    contrasts = results["paired_descriptive_contrasts"]
    spectrum_labels = {"strong": "Strong", "weak": "Weak"}
    scenario_labels = {
        "bounded_out_of_subspace": "Bounded out-of-subspace",
        "in_subspace_leverage": "In-subspace leverage",
        "mixed": "Mixed structural/Cauchy",
    }
    primary = [
        r"\begin{table*}[t]",
        r"\caption{Fresh Secondary Structural Comparison at Feasible Dimension}",
        r"\label{tab:structural-ceiling}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrp{0.24\textwidth}}",
        r"\toprule",
        r"Spectrum & Scenario & MAD-SF & Dual-Bound & Dual$-$MAD [95\%] & Noisy Outlier Pursuit \\",
        r"\midrule",
    ]
    diagnostic = [
        r"\begin{table*}[t]",
        r"\caption{Noisy Outlier Pursuit Diagnostic Outcomes}",
        r"\label{tab:outlier-pursuit-diagnostics}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Spectrum & Scenario & $\eta/\|X_{\rm clean}\|_F$ & Rank & Precision & Recall & Iterations & Primal residual \\",
        r"\midrule",
    ]
    for spectrum in ("strong", "weak"):
        for scenario in (
            "bounded_out_of_subspace",
            "in_subspace_leverage",
            "mixed",
        ):
            prefix = f"{spectrum}.{scenario}"
            mad = means[f"{prefix}.sketch_mad"]["mean_subspace_error"]
            dual = means[f"{prefix}.dual_bound"]["mean_subspace_error"]
            contrast = contrasts[f"{prefix}.dual_bound_minus_sketch_mad"]
            lo, hi = contrast["paired_t_95"]
            outlier = means[f"{prefix}.noisy_outlier_pursuit"]
            primary.append(
                f"{spectrum_labels[spectrum]} & {scenario_labels[scenario]} & "
                f"{mad:.3f} & {dual:.3f} & {contrast['estimate']:+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}] & rank 0; 30/30 unscored \\\\"
            )
            diagnostic.append(
                f"{spectrum_labels[spectrum]} & {scenario_labels[scenario]} & "
                f"{outlier['mean_oracle_noise_budget_fraction']:.3f} & "
                f"{outlier['mean_recovered_numerical_rank']:.0f} & "
                f"{outlier['mean_support_precision']:.3f} & "
                f"{outlier['mean_support_recall']:.3f} & "
                f"{outlier['mean_iterations']:.1f} & "
                f"{outlier['mean_primal_residual']:.2e} \\\\"
            )
    primary.extend(
        [
            r"\bottomrule",
            r"\multicolumn{6}{p{0.96\textwidth}}{\scriptsize Errors are means over 30 fresh seeds. Intervals are descriptive paired-$t$ intervals and are outside the primary Holm family. The batch program converged numerically but returned no rank-$r$ subspace; no arbitrary basis completion was scored.}\\",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    diagnostic.extend(
        [
            r"\bottomrule",
            r"\multicolumn{8}{p{0.96\textwidth}}{\scriptsize The fixed oracle noise budget and canonical penalty were not retuned. Rank-zero outputs were retained as failed comparator outcomes. Precision and recall describe the column-support estimate even when a subspace was unavailable.}\\",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    (output / "table_structural_ceiling.tex").write_text(
        "\n".join(primary) + "\n", encoding="utf-8"
    )
    (output / "table_outlier_pursuit_diagnostics.tex").write_text(
        "\n".join(diagnostic) + "\n", encoding="utf-8"
    )


def apply_final_editorial_wording(output: Path) -> None:
    eigengap = output / "table_eigengap_interactions.tex"
    text = eigengap.read_text(encoding="utf-8")
    text = text.replace(
        "Eigengap-Stratified Criterion Contrasts",
        "Spectrum-Stratified Descriptive Contrasts",
    ).replace(
        "Negative values support the declared alternative. NI contrasts use Dual$-1.05\\,$MAD.",
        "Negative values favor the declared alternative. NI contrasts use Dual$-1.05\\,$MAD. The strata describe heterogeneity; no formal interaction test was performed.",
    )
    eigengap.write_text(text, encoding="utf-8")

    external = output / "table_external.tex"
    text = external.read_text(encoding="utf-8")
    text = text.replace("External-Data Evidence", "Certified PBMC Structural-Fault Evidence")
    text = text.replace(r"{lp{0.53\columnwidth}}", r"{lp{0.60\columnwidth}}")
    text = text.replace("Tier & Paired outcome", "Comparison & Paired outcome")
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith("KDD aligned query")
    ) + "\n"
    external.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "generated-paper-assets",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    confirmatory = require_completed(CONFIRMATORY)
    oracle = require_completed(ORACLE)
    threshold = require_completed(THRESHOLD)
    retention = require_completed(RETENTION)
    statistical = require_completed(STATISTICAL)
    pbmc_certified = require_completed(PBMC_CERTIFIED)
    pbmc_secondary = require_completed(PBMC_SECONDARY)
    timing = require_completed(TIMING)
    structural = require_completed(STRUCTURAL)
    del confirmatory, oracle, threshold, retention, pbmc_secondary, timing

    base.ARTICLE = output
    base.PROJECT = ROOT
    base.CONFIRMATORY = CONFIRMATORY
    base.ORACLE = ORACLE
    base.THRESHOLD = THRESHOLD
    base.RETENTION = RETENTION
    base.STATISTICAL = STATISTICAL
    base.PBMC_CERTIFIED = PBMC_CERTIFIED
    base.PBMC_FULL = PBMC_SECONDARY

    rows = base.read_rows(CONFIRMATORY)
    oracle_rows = base.read_rows(ORACLE)
    threshold_rows = base.read_rows(THRESHOLD)
    retention_rows = base.read_rows(RETENTION)
    certified_rows = base.read_rows(PBMC_CERTIFIED)
    secondary_rows = base.read_rows(PBMC_SECONDARY)
    timing_rows = base.read_rows(TIMING)
    criteria = statistical["results"]["hypotheses"]
    strata = statistical["results"]["spectrum_interactions"]

    base.write_primary_table(criteria)
    base.write_eigengap_table(strata)
    base.write_synthetic_table(rows + oracle_rows)
    base.write_dense_table(rows)
    base.write_target_decomposition_table(rows + oracle_rows)
    base.write_external_table(pbmc_certified, certified_rows, None, [])
    base.write_pbmc_table(secondary_rows)
    base.write_resource_table(rows, timing_rows)
    base.write_retention_table(retention_rows)
    base.make_figures(rows, criteria, strata)
    base.make_threshold_figure(threshold_rows)
    make_paired_contrast_figure(statistical, output / "figure_synthetic_frontier.pdf")
    make_structural_tables(structural, output)
    apply_final_editorial_wording(output)

    asset_dependencies = strict_json(ROOT / "DISCLOSURE_MANIFEST.json")[
        "paper_asset_dependencies"
    ]
    produced = sorted(
        path for path in output.iterdir() if path.suffix in {".tex", ".pdf"}
    )
    manifest = {
        "schema_version": 1,
        "generator": "tools/generate_paper_assets.py",
        "inputs": {
            path.name: sha256(path)
            for path in (
                CONFIRMATORY,
                CONFIRMATORY.with_suffix(".raw.csv"),
                ORACLE,
                ORACLE.with_suffix(".raw.csv"),
                THRESHOLD,
                THRESHOLD.with_suffix(".raw.csv"),
                RETENTION,
                RETENTION.with_suffix(".raw.csv"),
                STATISTICAL,
                PBMC_CERTIFIED,
                PBMC_CERTIFIED.with_suffix(".raw.csv"),
                PBMC_SECONDARY,
                PBMC_SECONDARY.with_suffix(".raw.csv"),
                TIMING,
                TIMING.with_suffix(".raw.csv"),
                STRUCTURAL,
            )
        },
        "outputs": {path.name: sha256(path) for path in produced},
        "paper_asset_dependencies": asset_dependencies,
    }
    (output / "ASSET_BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
