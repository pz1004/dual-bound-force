#!/usr/bin/env python3
"""Generate manuscript numbers, tables, and figures from strict study artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_DPI = 600

PROJECT = Path(__file__).resolve().parents[1]
ARTICLE = PROJECT / "manuscript/article"
DEVELOPMENT = PROJECT / "results/development/dual-bound-force-development-full.json"
CONFIRMATORY = PROJECT / "results/confirmatory/dual-bound-force-confirmatory-full.json"
ORACLE = PROJECT / "results/oracle/dual-bound-force-confirmatory-full.json"
THRESHOLD = PROJECT / "results/threshold/dual-bound-force-confirmatory-full.json"
RETENTION = PROJECT / "results/retention/dual-bound-force-confirmatory-full.json"
STATISTICAL = PROJECT / "results/confirmatory/dual-bound-force-statistical-reanalysis.json"
PBMC_CERTIFIED = PROJECT / "results/confirmatory/dual-bound-force-pbmc-certified-full.json"
PBMC_FULL = PROJECT / "results/confirmatory/dual-bound-force-pbmc-full.json"


def read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def typed(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_rows(json_path: Path) -> list[dict[str, Any]]:
    path = json_path.with_suffix(".raw.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {key: typed(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def bootstrap_mean(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.mean(array)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def optional_external(name: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = PROJECT / f"results/confirmatory/dual-bound-force-{name}-full.json"
    if name == "pbmc" and not path.is_file():
        path = PROJECT / "results/confirmatory/dual-bound-force-pbmc-primary-full.json"
    if not path.is_file():
        return None, []
    return read_json(path), read_rows(path)


def write_primary_table(criteria: dict[str, Any]) -> None:
    labels = {
        "structural_out_of_subspace_improvement": "Out-of-subspace superiority",
        "casewise_noninferiority_5pct": "Casewise noninferiority (5\\%)",
        "cellwise_noninferiority_5pct": "Cellwise noninferiority (5\\%)",
        "clean_noninferiority_5pct": "Clean noninferiority (5\\%)",
        "pbmc_structural_improvement": "PBMC structural superiority",
    }
    estimands = {
        "structural_out_of_subspace_improvement": r"mean Dual$-$MAD",
        "casewise_noninferiority_5pct": r"mean Dual$-1.05\,$MAD",
        "cellwise_noninferiority_5pct": r"mean Dual$-1.05\,$MAD",
        "clean_noninferiority_5pct": r"mean Dual$-1.05\,$MAD",
        "pbmc_structural_improvement": r"mean Dual$-$MAD",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Scientific Hypotheses}",
        r"\label{tab:criteria}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.24\textwidth}p{0.17\textwidth}p{0.09\textwidth}p{0.27\textwidth}c}",
        r"\toprule",
        r"Hypothesis & Estimand & Estimate & Paired-$t$ 95\% interval; one-sided Holm $p$ & Supported \\",
        r"\midrule",
    ]
    for key in labels:
        item = criteria.get(key, {"status": "not_evaluated"})
        if item.get("status") != "evaluated":
            estimate = "Not evaluated"
            interval = escape(item.get("reason", "required records unavailable"))
            passed = "--"
        else:
            estimate = fmt(item["estimate"], 4)
            lower, upper = item["paired_t_95"]
            interval = f"[{fmt(lower,4)}, {fmt(upper,4)}]"
            adjusted_p = float(item["holm_adjusted_one_sided_p"])
            rendered_p = "<0.001" if adjusted_p < 0.001 else f"={fmt(adjusted_p,3)}"
            interval += f"; $p_{{\\mathrm{{Holm}}}}{rendered_p}$"
            passed = "Yes" if item["passed"] else "No"
        lines.append(
            f"{labels[key]} & {estimands[key]} & {estimate} & {interval} & {passed}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ARTICLE / "table_primary_criteria.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eigengap_table(interactions: dict[str, Any]) -> None:
    labels = {
        "structural_out_of_subspace_improvement": "Out-of-subspace",
        "casewise_noninferiority_5pct": "Casewise NI",
        "cellwise_noninferiority_5pct": "Cellwise NI",
        "clean_noninferiority_5pct": "Clean NI",
    }
    lines = [
        r"\begin{table}[t]",
        r"\caption{Eigengap-Stratified Criterion Contrasts}",
        r"\label{tab:eigengap}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Criterion & Strong gap & Weak gap \\",
        r"\midrule",
    ]
    for key, label in labels.items():
        rendered = []
        for spectrum in ("strong", "weak"):
            item = interactions[key][spectrum]
            lower, upper = item["paired_t_95"]
            rendered.append(
                f"{fmt(item['estimate'],4)} [{fmt(lower,4)}, {fmt(upper,4)}]"
            )
        lines.append(f"{label} & {rendered[0]} & {rendered[1]}" + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{3}{p{0.92\columnwidth}}{\scriptsize Negative values support the declared alternative. NI contrasts use Dual$-1.05\,$MAD.}\\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (ARTICLE / "table_eigengap_interactions.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_synthetic_table(rows: list[dict[str, Any]]) -> None:
    methods = (
        "vanilla_fd", "rfd", "sketch_iqr", "sketch_mad",
        "exact_mad_fd", "dual_bound",
    )
    scenarios = (
        "clean", "casewise_cauchy", "cellwise_cauchy",
        "bounded_out_of_subspace", "in_subspace_leverage", "mixed",
    )
    labels = {
        "clean": "Clean",
        "casewise_cauchy": "Casewise Cauchy",
        "cellwise_cauchy": "Cellwise Cauchy",
        "bounded_out_of_subspace": "Bounded out-of-subspace",
        "in_subspace_leverage": "In-subspace leverage",
        "mixed": "Mixed",
    }
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("tier") in {"sketch", "oracle_ablation"} and row.get("k") == 20:
            grouped[(row["scenario"], row["method"])].append(float(row["subspace_error"]))
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Primary-$k$ Bounded-State Synthetic Subspace Error}",
        r"\label{tab:synthetic}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Scenario & FD & RFD & IQR-SF & MAD-SF & Exact-MAD & Dual-Bound \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        values = [np.mean(grouped.get((scenario, method), [np.nan])) for method in methods]
        lines.append(f"{labels[scenario]} & " + " & ".join(fmt(value, 3) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ARTICLE / "table_synthetic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dense_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    methods = ("pearson", "force", "mad_force", "sketch_mad", "dual_bound")
    scenarios = (
        "clean", "casewise_cauchy", "cellwise_cauchy",
        "bounded_out_of_subspace", "in_subspace_leverage", "mixed",
    )
    labels = {
        "clean": "Clean", "casewise_cauchy": "Casewise Cauchy",
        "cellwise_cauchy": "Cellwise Cauchy",
        "bounded_out_of_subspace": "Bounded out-of-subspace",
        "in_subspace_leverage": "In-subspace leverage", "mixed": "Mixed",
    }
    values: dict[str, dict[str, float]] = {}
    lines = [
        r"\begin{table*}[t]", r"\caption{Dense-Tier Clean-Reference Subspace Error ($p=50$)}",
        r"\label{tab:dense}", r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{lrrrrr}",
        r"\toprule", r"Scenario & Pearson & FORCE & MAD-FORCE & MAD-SF & Dual-Bound \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        item: dict[str, float] = {}
        for method in methods:
            selected = [
                float(row["subspace_error"]) for row in rows
                if row.get("tier") == "dense"
                and row.get("scenario") == scenario
                and row.get("method") == method
            ]
            item[method] = float(np.mean(selected)) if selected else float("nan")
        values[scenario] = item
        lines.append(
            f"{labels[scenario]} & "
            + " & ".join(fmt(item[method], 3) for method in methods)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ARTICLE / "table_dense.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


def write_target_decomposition_table(rows: list[dict[str, Any]]) -> None:
    methods = ("sketch_mad", "exact_mad_fd", "dual_bound")
    scenarios = (
        "clean", "casewise_cauchy", "cellwise_cauchy",
        "bounded_out_of_subspace",
    )
    labels = {
        "clean": "Clean", "casewise_cauchy": "Casewise",
        "cellwise_cauchy": "Cellwise",
        "bounded_out_of_subspace": "Bounded residual",
    }
    lines = [
        r"\begin{table*}[t]", r"\caption{Clean-Target Versus Transformed-Target Scatter Error at $k=20$}",
        r"\label{tab:targets}", r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{MAD-SF} & \multicolumn{2}{c}{Exact-MAD oracle} & \multicolumn{2}{c}{Dual-Bound} \\",
        r"Scenario & Clean & Transform & Clean & Transform & Clean & Transform \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        rendered = []
        for method in methods:
            selected = [
                row for row in rows
                if row.get("scenario") == scenario
                and row.get("method") == method
                and row.get("k") == 20
                and row.get("tier") in {"sketch", "oracle_ablation"}
            ]
            clean = [
                float(row["clean_target_scale_scatter_error"]) for row in selected
                if row.get("clean_target_scale_scatter_error") is not None
            ]
            transformed = [
                float(row["transformed_target_scatter_compression_error"])
                for row in selected
                if row.get("transformed_target_scatter_compression_error") is not None
            ]
            rendered.extend([
                fmt(float(np.mean(clean)) if clean else None, 3),
                fmt(float(np.mean(transformed)) if transformed else None, 3),
            ])
        lines.append(f"{labels[scenario]} & " + " & ".join(rendered) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ARTICLE / "table_target_decomposition.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_external_table(
    pbmc_payload: dict[str, Any] | None,
    pbmc_rows: list[dict[str, Any]],
    kdd_payload: dict[str, Any] | None,
    kdd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if pbmc_payload is None:
        pbmc_text = "Not executed"
        pbmc_pass = None
    elif pbmc_payload["status"] != "completed":
        pbmc_text = escape(pbmc_payload["status"])
        pbmc_pass = None
    else:
        recorded = pbmc_payload.get("results", {}).get("primary_criterion")
        if isinstance(recorded, dict) and str(recorded.get("status", "")).startswith("evaluated"):
            lower, upper = recorded["paired_t_95"]
            pbmc_text = (
                f"$\\Delta={fmt(recorded['estimate'],4)}$ "
                f"[{fmt(lower,4)}, {fmt(upper,4)}]"
            )
            pbmc_pass = bool(upper < 0.0)
            recorded = True
        else:
            recorded = False
        dual = {row["seed"]: row["subspace_error"] for row in pbmc_rows if row.get("scenario") == "bounded_out_of_subspace" and row.get("method") == "dual_bound"}
        mad = {row["seed"]: row["subspace_error"] for row in pbmc_rows if row.get("scenario") == "bounded_out_of_subspace" and row.get("method") == "sketch_mad"}
        seeds = sorted(set(dual) & set(mad))
        if not recorded and seeds:
            difference = [dual[seed] - mad[seed] for seed in seeds]
            estimate, lower, upper = bootstrap_mean(difference, 77_001)
            pbmc_text = f"$\\Delta={fmt(estimate,4)}$ [{fmt(lower,4)}, {fmt(upper,4)}]"
            pbmc_pass = upper < 0
        elif not recorded:
            pbmc_text = "Paired structural records unavailable"
            pbmc_pass = None
    if kdd_payload is None:
        kdd_text = "Not executed"
    elif kdd_payload["status"] != "completed":
        kdd_text = escape(kdd_payload["status"])
    else:
        recorded = kdd_payload.get("results", {}).get("aligned_query_auc_gain")
        if isinstance(recorded, dict) and recorded.get("status") == "evaluated":
            estimate = float(recorded["estimate"])
            lower = float(recorded["lower_95"])
            upper = float(recorded["upper_95"])
        else:
            differences = [float(row["aligned_minus_raw_auc"]) for row in kdd_rows if row.get("status") == "completed"]
            estimate, lower, upper = bootstrap_mean(differences, 77_502)
        kdd_text = f"$\\Delta\\mathrm{{AUC}}={fmt(estimate,4)}$ [{fmt(lower,4)}, {fmt(upper,4)}]"
    dual_absolute = [
        float(row.get("recomputed_subspace_error", row["subspace_error"]))
        for row in pbmc_rows if row.get("method") == "dual_bound"
    ]
    mad_absolute = [
        float(row.get("recomputed_subspace_error", row["subspace_error"]))
        for row in pbmc_rows if row.get("method") == "sketch_mad"
    ]
    pbmc_absolute = (
        f"Dual/MAD abs. {fmt(float(np.mean(dual_absolute)),3)}/"
        f"{fmt(float(np.mean(mad_absolute)),3)}"
        if dual_absolute and mad_absolute else "absolute errors unavailable"
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{External-Data Evidence}",
        r"\label{tab:external}",
        r"\centering\small",
        r"\begin{tabular}{lp{0.53\columnwidth}}",
        r"\toprule",
        r"Tier & Paired outcome \\",
        r"\midrule",
        f"PBMC bounded structural & {pbmc_text}; {pbmc_absolute}" + r" \\",
        f"KDD aligned query & {kdd_text}" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (ARTICLE / "table_external.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"pbmc_text": pbmc_text, "pbmc_pass": pbmc_pass, "kdd_text": kdd_text}


def write_pbmc_table(rows: list[dict[str, Any]]) -> None:
    scenarios = (
        "clean", "casewise", "cellwise", "bounded_out_of_subspace", "in_subspace"
    )
    labels = {
        "clean": "Clean", "casewise": "Casewise", "cellwise": "Cellwise",
        "bounded_out_of_subspace": "Bounded residual", "in_subspace": "In-subspace",
    }
    methods = ("vanilla_fd", "rfd", "sketch_mad", "dual_bound")
    lines = [
        r"\begin{table}[t]", r"\caption{Exploratory PBMC Secondary Errors Under the Superseded Reference}",
        r"\label{tab:pbmc}", r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}",
        r"\toprule", r"Scenario & FD & RFD & MAD-SF & Dual \\", r"\midrule",
    ]
    for scenario in scenarios:
        values = []
        for method in methods:
            selected = [
                float(row["subspace_error"]) for row in rows
                if row.get("scenario") == scenario and row.get("method") == method
            ]
            values.append(float(np.mean(selected)) if selected else None)
        lines.append(
            f"{labels[scenario]} & " + " & ".join(fmt(value, 3) for value in values) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{5}{p{0.94\columnwidth}}{\scriptsize Descriptive only; primary inference uses the separately certified reference and retained bases.}\\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (ARTICLE / "table_pbmc.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resource_table(rows: list[dict[str, Any]], timing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ("sketch_mad", "dual_bound")
    lines = [
        r"\begin{table*}[t]",
        # IEEE Access's full-width float caption is shifted one page margin
        # left by the XeTeX/Tectonic compatibility path on this page.  Keep
        # the corrective offset in the generator so regenerated assets retain
        # the visually verified placement.
        r"\caption{\hspace*{0.5in}Synthetic Resource Measures (20-Run Medians)}",
        r"\label{tab:resources}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Active state (KiB) & Calibration peak (MiB) & Output (KiB) & Peak RSS (MiB) & Rows/s \\",
        r"\midrule",
    ]
    medians = {}
    for method in methods:
        selected = [row for row in rows if row.get("tier") == "sketch" and row.get("k") == 20 and row.get("method") == method]
        timed = [
            float(row["throughput_rows_per_second"]) for row in timing_rows
            if row.get("status") == "completed" and row.get("method") == method
        ]
        state = float(np.median([row["state_bytes"] for row in selected])) / 1024 if selected else None
        speed = float(np.median(timed)) if timed else None
        timed_rows = [
            row for row in timing_rows
            if row.get("status") == "completed" and row.get("method") == method
        ]
        calibration_peak = (
            float(np.median([row["calibration_peak_state_bytes"] for row in timed_rows])) / 2**20
            if timed_rows else None
        )
        output = (
            float(np.median([row["output_bytes"] for row in timed_rows])) / 1024
            if timed_rows else None
        )
        peak_rss = (
            float(np.median([row["peak_rss_bytes"] for row in timed_rows])) / 2**20
            if timed_rows else None
        )
        incremental_rss = (
            float(np.median([row["worker_incremental_rss_bytes"] for row in timed_rows])) / 2**20
            if timed_rows else None
        )
        medians[method] = {
            "state_kib": state,
            "calibration_peak_mib": calibration_peak,
            "output_kib": output,
            "peak_rss_mib": peak_rss,
            "incremental_rss_mib": incremental_rss,
            "throughput": speed,
        }
        method_label = {
            "sketch_mad": "MAD Sketch",
            "dual_bound": "Dual-Bound",
        }[method]
        lines.append(
            f"{method_label} & {fmt(state,1)} & {fmt(calibration_peak,2)} & "
            f"{fmt(output,1)} & {fmt(peak_rss,1)} & {fmt(speed,0)}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ARTICLE / "table_resources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    timing_completed = [row for row in timing_rows if row.get("status") == "completed"]
    ratio = None
    if timing_completed:
        by_rep: dict[int, dict[str, float]] = defaultdict(dict)
        for row in timing_completed:
            by_rep[int(row["repetition"])][row["method"]] = float(row["throughput_rows_per_second"])
        ratios = [values["dual_bound"] / values["sketch_mad"] for values in by_rep.values() if set(values) >= {"dual_bound", "sketch_mad"}]
        ratio = float(np.median(ratios)) if ratios else None
    return {"medians": medians, "isolated_ratio": ratio}


def write_retention_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    scenarios = (
        "casewise_cauchy", "cellwise_cauchy", "bounded_out_of_subspace",
        "in_subspace_leverage", "mixed",
    )
    labels = {
        "casewise_cauchy": "Casewise",
        "cellwise_cauchy": "Cellwise",
        "bounded_out_of_subspace": "Bounded residual",
        "in_subspace_leverage": "In-subspace",
        "mixed": "Mixed",
    }
    values: dict[str, dict[str, float]] = {}
    lines = [
        r"\begin{table}[t]", r"\caption{Marginal Retention and Clean Attenuation}",
        r"\label{tab:retention}", r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}",
        r"\toprule", r"& \multicolumn{2}{c}{Contamination retained} & \multicolumn{2}{c}{Clean clipped} \\",
        r"Scenario & MAD-SF & Dual & MAD-SF & Dual \\", r"\midrule",
    ]
    for scenario in scenarios:
        item: dict[str, float] = {}
        for method in ("sketch_mad", "dual_bound"):
            selected = [row for row in rows if row.get("scenario") == scenario and row.get("method") == method]
            item[f"{method}_retained"] = float(np.mean([
                row["marginal_retained_contamination_rate"] for row in selected
                if row.get("marginal_retained_contamination_rate") is not None
            ]))
            item[f"{method}_clean"] = float(np.mean([
                row["clean_marginal_attenuation_rate"] for row in selected
                if row.get("clean_marginal_attenuation_rate") is not None
            ]))
        values[scenario] = item
        lines.append(
            f"{labels[scenario]} & {fmt(item['sketch_mad_retained'],3)} & "
            f"{fmt(item['dual_bound_retained'],3)} & "
            f"{fmt(item['sketch_mad_clean'],3)} & {fmt(item['dual_bound_clean'],3)}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ARTICLE / "table_retention.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


def make_figures(
    rows: list[dict[str, Any]],
    criteria: dict[str, Any],
    interactions: dict[str, Any],
) -> None:
    methods = ("vanilla_fd", "sketch_mad", "dual_bound")
    method_labels = ("Vanilla FD", "MAD Sketch", "Dual-Bound")
    scenarios = ("clean", "casewise_cauchy", "cellwise_cauchy", "bounded_out_of_subspace")
    scenario_labels = ("Clean", "Casewise", "Cellwise", "Bounded structural")
    colors = ("#7f7f7f", "#2878B5", "#D95319")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    hypothesis_names = (
        "clean_noninferiority_5pct",
        "casewise_noninferiority_5pct",
        "cellwise_noninferiority_5pct",
        "structural_out_of_subspace_improvement",
    )
    y = np.arange(len(hypothesis_names))
    for label, marker, offset, source in (
        ("Equal-spectrum", "o", 0.00, criteria),
        ("Strong gap", "s", 0.18, {key: interactions[key]["strong"] for key in hypothesis_names}),
        ("Weak gap", "^", -0.18, {key: interactions[key]["weak"] for key in hypothesis_names}),
    ):
        estimates, lower, upper = [], [], []
        for name in hypothesis_names:
            item = source[name]
            if int(item["n_pairs"]) != 30:
                raise RuntimeError("primary contrast figure requires 30 paired seed units")
            lo, hi = item["paired_t_95"]
            estimates.append(float(item["estimate"]))
            lower.append(float(item["estimate"]) - float(lo))
            upper.append(float(hi) - float(item["estimate"]))
        axes[0].errorbar(
            estimates,
            y + offset,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            capsize=2,
            label=label,
        )
    axes[0].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_yticks(y, scenario_labels)
    axes[0].set_xlabel("Criterion contrast (negative supports alternative)")
    axes[0].set_title("Paired seed-level inference")
    axes[0].legend(frameon=False, fontsize=8)

    for method_index, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means, states = [], []
        for scenario in scenarios:
            selected = [float(row["subspace_error"]) for row in rows if row.get("tier") == "sketch" and row.get("k") == 20 and row.get("method") == method and row.get("scenario") == scenario]
            means.append(float(np.mean(selected)))
            state_values = [float(row["state_bytes"]) / 1024 for row in rows if row.get("tier") == "sketch" and row.get("k") == 20 and row.get("method") == method and row.get("scenario") == scenario]
            states.append(float(np.median(state_values)))
        axes[1].plot(means, states, "o-", color=color, label=label)
    axes[1].set_xlabel("Normalized subspace error")
    axes[1].set_ylabel("Explicit state (KiB)")
    axes[1].set_title("Descriptive accuracy-resource points")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        ARTICLE / "figure_synthetic_frontier.png",
        dpi=FIGURE_DPI,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    audit = [row for row in rows if row.get("tier") == "calibration_audit"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    labels, scale_values, basis_values, subspace_values = [], [], [], []
    for contaminate in (False, True):
        for front in (False, True):
            selected = [row for row in audit if row.get("contaminate_calibration") == contaminate and row.get("front_loaded") == front]
            labels.append(("Cal" if contaminate else "Est") + ("-front" if front else "-random"))
            scale_values.append(float(np.mean([row["relative_scale_error"] for row in selected])))
            basis_values.append(float(np.mean([row["calibration_basis_error"] for row in selected])))
            subspace_values.append(float(np.mean([row["subspace_error"] for row in selected])))
    positions = np.arange(len(labels))
    ax.bar(positions - 0.25, scale_values, 0.25, label="Scale error", color="#E5A84B")
    ax.bar(positions, basis_values, 0.25, label="Calibration-basis error", color="#59A14F")
    ax.bar(positions + 0.25, subspace_values, 0.25, label="Output subspace error", color="#5B8FF9")
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel("Relative / normalized error")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        ARTICLE / "figure_calibration_audit.png",
        dpi=FIGURE_DPI,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def make_threshold_figure(rows: list[dict[str, Any]]) -> dict[str, float]:
    scenarios = (
        "clean", "casewise_cauchy", "cellwise_cauchy",
        "bounded_out_of_subspace",
    )
    labels = ("Clean", "Casewise", "Cellwise", "Bounded structural")
    methods = (("sketch_mad", "MAD Sketch", "#2878B5"),
               ("dual_bound", "Dual-Bound", "#D95319"))
    grid = sorted({float(row["marginal_lambda"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharex=True, sharey=True)
    optima: dict[str, float] = {}
    for axis, scenario, label in zip(axes.ravel(), scenarios, labels):
        for method, method_label, color in methods:
            means = []
            for value in grid:
                selected = [
                    float(row["subspace_error"]) for row in rows
                    if row.get("scenario") == scenario
                    and row.get("method") == method
                    and float(row.get("marginal_lambda")) == value
                ]
                means.append(float(np.mean(selected)))
            axis.plot(grid, means, "o-", color=color, label=method_label)
            if method == "dual_bound":
                optima[scenario] = grid[int(np.argmin(means))]
        axis.set_title(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Marginal multiplier")
    axes[1, 1].set_xlabel("Marginal multiplier")
    axes[0, 0].set_ylabel("Subspace error")
    axes[1, 0].set_ylabel("Subspace error")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        ARTICLE / "figure_threshold_sensitivity.png",
        dpi=FIGURE_DPI,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return optima


def main() -> None:
    development = read_json(DEVELOPMENT)
    confirmatory = read_json(CONFIRMATORY)
    statistical = read_json(STATISTICAL)
    if any(
        payload["status"] != "completed"
        for payload in (development, confirmatory, statistical)
    ):
        raise RuntimeError(
            "completed development, confirmatory, and statistical-reanalysis artifacts are required"
        )
    rows = read_rows(CONFIRMATORY)
    oracle_payload = read_json(ORACLE) if ORACLE.is_file() else None
    oracle_rows = read_rows(ORACLE) if oracle_payload is not None else []
    threshold_payload = read_json(THRESHOLD) if THRESHOLD.is_file() else None
    threshold_rows = read_rows(THRESHOLD) if threshold_payload is not None else []
    retention_payload = read_json(RETENTION) if RETENTION.is_file() else None
    retention_rows = read_rows(RETENTION) if retention_payload is not None else []
    pbmc_payload = read_json(PBMC_CERTIFIED)
    pbmc_rows = read_rows(PBMC_CERTIFIED)
    pbmc_secondary_payload = read_json(PBMC_FULL) if PBMC_FULL.is_file() else None
    pbmc_secondary_rows = read_rows(PBMC_FULL) if pbmc_secondary_payload is not None else []
    kdd_payload, kdd_rows = optional_external("kdd")
    timing_payload, timing_rows = optional_external("timing")
    selection = development["results"]["selection"]
    frozen = read_json(PROJECT / "preregistration/frozen_configuration.json")
    frozen_hash = (PROJECT / "preregistration/frozen_configuration.json.sha256").read_text().split()[0]

    criteria = statistical["results"]["hypotheses"]
    interactions = statistical["results"]["spectrum_interactions"]
    write_primary_table(criteria)
    write_eigengap_table(interactions)
    write_synthetic_table(rows + oracle_rows)
    dense = write_dense_table(rows)
    write_target_decomposition_table(rows + oracle_rows)
    external = write_external_table(pbmc_payload, pbmc_rows, kdd_payload, kdd_rows)
    write_pbmc_table(pbmc_secondary_rows)
    resources = write_resource_table(rows, timing_rows)
    retention = write_retention_table(retention_rows) if retention_rows else {}
    make_figures(rows, criteria, interactions)
    threshold_optima = make_threshold_figure(threshold_rows) if threshold_rows else {}

    structural = criteria["structural_out_of_subspace_improvement"]
    casewise = criteria["casewise_noninferiority_5pct"]
    cellwise = criteria["cellwise_noninferiority_5pct"]
    clean = criteria["clean_noninferiority_5pct"]
    pbmc = criteria["pbmc_structural_improvement"]
    dual_primary = [
        row for row in rows
        if row.get("tier") == "sketch"
        and row.get("method") == "dual_bound"
        and row.get("k") == 20
    ]
    clean_residual_clip = [
        float(row["residual_clipped_rows"]) / max(float(row["effective_n"]), 1.0)
        for row in dual_primary if row.get("scenario") == "clean"
    ]
    structural_residual_clip = [
        float(row["residual_clipped_rows"]) / max(float(row["effective_n"]), 1.0)
        for row in dual_primary
        if row.get("scenario") == "bounded_out_of_subspace"
    ]
    structural_lower, structural_upper = structural["paired_t_95"]
    clean_strong = interactions["clean_noninferiority_5pct"]["strong"]
    clean_weak = interactions["clean_noninferiority_5pct"]["weak"]
    primary_sentence = (
        f"Across 30 paired synthetic seeds, structural superiority was not supported "
        f"($\\Delta={fmt(structural.get('estimate'),4)}$, paired-$t$ 95\\% interval "
        f"[{fmt(structural_lower,4)}, {fmt(structural_upper,4)}]). "
        f"Casewise and cellwise noninferiority were supported. Clean noninferiority "
        f"held only for the prospectively specified equal-spectrum average and failed "
        f"in the strong-eigengap stratum. The certified-PBMC comparison showed a "
        f"small relative advantage ($\\Delta={fmt(pbmc.get('estimate'),4)}$), while "
        "both absolute errors remained near one."
    )
    primary_narrative = (
        f"The bounded out-of-subspace paired difference was {fmt(structural.get('estimate'),4)} "
        f"with paired-$t$ interval [{fmt(structural_lower,4)}, {fmt(structural_upper,4)}], "
        "so the declared superiority hypothesis failed. One-sided paired mean tests "
        "with Holm adjustment supported the casewise and cellwise noninferiority "
        f"hypotheses. Aggregate clean noninferiority was supported, but its strong-gap "
        f"contrast was {fmt(clean_strong['estimate'],4)} and its weak-gap contrast was "
        f"{fmt(clean_weak['estimate'],4)}. "
        f"Dual-Bound residual-clipping fractions were "
        f"{fmt(float(np.mean(clean_residual_clip)) if clean_residual_clip else None,3)} on clean rows and "
        f"{fmt(float(np.mean(structural_residual_clip)) if structural_residual_clip else None,3)} under the bounded structural attack; "
        "the attack therefore did not cross the frozen residual boundary more often than clean variation. "
        "Engineering checks are reported separately and are not counted as scientific hypotheses."
    )
    compression = [float(row["transformed_target_scatter_compression_error"]) for row in rows if row.get("method") == "dual_bound" and row.get("k") == 20 and row.get("transformed_target_scatter_compression_error") is not None]
    oracle_compression = [
        float(row["transformed_target_scatter_compression_error"])
        for row in oracle_rows
        if row.get("k") == 20
        and row.get("transformed_target_scatter_compression_error") is not None
    ]
    target_narrative = (
        f"The median Dual-Bound observed-transformed-target compression error at $k=20$ was {fmt(float(np.median(compression)) if compression else None,3)}. "
        f"The offline exact-MAD FD oracle median was {fmt(float(np.median(oracle_compression)) if oracle_compression else None,3)}. "
        "Their separation from clean-target error confirms that robust transformation and FD loss are empirically distinct mediators; the oracle is an ablation, not a streaming competitor."
    )
    dense_narrative = (
        f"At feasible $p=50$, MAD-FORCE had mean clean-reference subspace error "
        f"{fmt(dense['casewise_cauchy']['mad_force'],3)} under casewise and "
        f"{fmt(dense['cellwise_cauchy']['mad_force'],3)} under cellwise Cauchy faults, "
        f"compared with {fmt(dense['casewise_cauchy']['dual_bound'],3)} and "
        f"{fmt(dense['cellwise_cauchy']['dual_bound'],3)} for Dual-Bound. "
        f"Under bounded out-of-subspace replacement, FORCE and MAD-FORCE were "
        f"{fmt(dense['bounded_out_of_subspace']['force'],3)} and "
        f"{fmt(dense['bounded_out_of_subspace']['mad_force'],3)}, showing that "
        "marginally invisible structure remains a shared limitation. These are shared clean-reference outcomes, not claims that pairwise and sketch estimators have identical transformed targets."
    )
    audit = [row for row in rows if row.get("tier") == "calibration_audit"]
    cal_clean = [row["relative_scale_error"] for row in audit if not row.get("contaminate_calibration")]
    cal_bad = [row["relative_scale_error"] for row in audit if row.get("contaminate_calibration")]
    basis_clean = [row["calibration_basis_error"] for row in audit if not row.get("contaminate_calibration")]
    basis_bad = [row["calibration_basis_error"] for row in audit if row.get("contaminate_calibration")]
    clipping_clean = [
        float(row["marginal_clipped_coordinates"]) /
        max(float(row["effective_n"]) * float(row["p"]), 1.0)
        for row in audit if not row.get("contaminate_calibration")
    ]
    clipping_bad = [
        float(row["marginal_clipped_coordinates"]) /
        max(float(row["effective_n"]) * float(row["p"]), 1.0)
        for row in audit if row.get("contaminate_calibration")
    ]
    calibration_narrative = (
        f"Mean relative scale error was {fmt(float(np.mean(cal_clean)) if cal_clean else None,3)} when faults were restricted to estimation and "
        f"{fmt(float(np.mean(cal_bad)) if cal_bad else None,3)} when calibration was contaminated. "
        f"The corresponding calibration-basis errors were "
        f"{fmt(float(np.mean(basis_clean)) if basis_clean else None,3)} and "
        f"{fmt(float(np.mean(basis_bad)) if basis_bad else None,3)}, while the marginal clipping fractions were "
        f"{fmt(float(np.mean(clipping_clean)) if clipping_clean else None,3)} and "
        f"{fmt(float(np.mean(clipping_bad)) if clipping_bad else None,3)}. "
        "This is direct boundary evidence for the conditional theorem."
    )
    scheduled = [row for row in rows if row.get("tier") == "scheduled_epoch"]
    stationary_epoch_error = [
        float(row["subspace_error_final_regime"]) for row in scheduled
        if row.get("scheduled") is False
    ]
    scheduled_epoch_error = [
        float(row["subspace_error_final_regime"]) for row in scheduled
        if row.get("scheduled") is True
    ]
    scheduled_narrative = (
        f"Mean final-regime subspace error was "
        f"{fmt(float(np.mean(stationary_epoch_error)) if stationary_epoch_error else None,3)} "
        f"for the stationary estimator and "
        f"{fmt(float(np.mean(scheduled_epoch_error)) if scheduled_epoch_error else None,3)} "
        "for the prescheduled policy. The comparison is descriptive because the "
        "change times were supplied to the schedule."
    )
    threshold_narrative = (
        "The separately labeled marginal-threshold arm was not used for retuning. "
        + (
            "Dual-Bound's lowest mean subspace errors occurred at marginal multipliers "
            f"{fmt(threshold_optima.get('clean'),2)} (clean), "
            f"{fmt(threshold_optima.get('casewise_cauchy'),2)} (casewise), "
            f"{fmt(threshold_optima.get('cellwise_cauchy'),2)} (cellwise), and "
            f"{fmt(threshold_optima.get('bounded_out_of_subspace'),2)} (bounded structural)."
            if threshold_optima else "The full threshold artifact was unavailable."
        )
    )
    bounded_retention = retention.get("bounded_out_of_subspace", {})
    retention_narrative = (
        "Marginal retention is a mechanism diagnostic, not a breakdown estimate. "
        + (
            f"For bounded out-of-subspace rows, the retained contaminated-coordinate rates were "
            f"{fmt(bounded_retention.get('sketch_mad_retained'),3)} for MAD Sketch-FORCE and "
            f"{fmt(bounded_retention.get('dual_bound_retained'),3)} for Dual-Bound FORCE. "
            "The dual norm bound can still attenuate a row after marginal retention."
            if bounded_retention else "The full retention audit was unavailable."
        )
    )
    reference_diagnostics = pbmc_payload["preprocessing"]["reference_svd"]["diagnostics"]
    reference_iterations = "/".join(
        str(int(item["power_iterations"])) for item in reference_diagnostics["starts"]
    )
    external_narrative = (
        f"PBMC structural evidence: {external['pbmc_text']}. KDD aligned-query evidence: {external['kdd_text']}. "
        f"The certified PBMC reference used two deterministic starts ({reference_iterations} power iterations), "
        f"cross-start distance {fmt(reference_diagnostics['cross_start_normalized_subspace_change'],6)}, "
        f"maximum relative Ritz residual {fmt(reference_diagnostics['maximum_relative_ritz_residual'],6)}, "
        f"and estimated rank-20/21 gap {fmt(reference_diagnostics['estimated_rank_gap'],3)}. "
        f"The exact one-sided sign and sign-flip probabilities were "
        f"{fmt(pbmc['exact_one_sided_sign_p'],6)} and "
        f"{fmt(pbmc['exact_one_sided_sign_flip_p'],6)}. "
        f"Absolute Dual/MAD errors were {fmt(pbmc['mean_dual_error'],4)}/"
        f"{fmt(pbmc['mean_mad_error'],4)}, so the PBMC result is relative rather than satisfactory recovery. "
        "Unavailable or failed external tiers are retained as statuses rather than replaced."
    )
    ratio = resources["isolated_ratio"]
    resource_narrative = (
        f"The isolated median Dual-Bound/MAD-Sketch throughput ratio was {fmt(ratio,3)}. "
        f"Dual-Bound's median calibration-state envelope and process peak were "
        f"{fmt(resources['medians'].get('dual_bound', {}).get('calibration_peak_mib'),2)} MiB and "
        f"{fmt(resources['medians'].get('dual_bound', {}).get('peak_rss_mib'),1)} MiB. "
        "Explicit numerical state, output storage, and operating-system RSS are reported separately because allocator and runtime overhead are not estimator state."
    )
    conclusion = (
        "The prospectively specified structural superiority hypothesis failed, "
        "whereas casewise and cellwise noninferiority were supported under the "
        "declared generators. Clean noninferiority was spectrum-dependent, and "
        "the certified PBMC advantage was small relative to uniformly high absolute "
        "errors. No estimator parameter was retuned after confirmation."
    )
    macros = {
        "PrimaryOutcomeSentence": primary_sentence,
        "SelectedParallelLambda": fmt(float(frozen["parallel_lambda"]), 1),
        "SelectedResidualLambda": fmt(float(frozen["residual_lambda"]), 1),
        "FrozenConfigurationShortHash": frozen_hash[:12],
        "DevelopmentFeasibilitySentence": (
            "met both development constraints."
            if selection["feasible_configuration_exists"]
            else "did not meet all development constraints; the minimum-violation fallback was frozen."
        ),
        "PrimaryCriteriaNarrative": primary_narrative,
        "TargetDecompositionNarrative": target_narrative,
        "DenseTierNarrative": dense_narrative,
        "CalibrationNarrative": calibration_narrative,
        "ScheduledNarrative": scheduled_narrative,
        "ThresholdNarrative": threshold_narrative,
        "RetentionNarrative": retention_narrative,
        "ExternalNarrative": external_narrative,
        "ResourceNarrative": resource_narrative,
        "ConclusionSentence": conclusion,
    }
    (ARTICLE / "generated_results.tex").write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "development": str(DEVELOPMENT),
        "confirmatory": str(CONFIRMATORY),
        "statistical_reanalysis": str(STATISTICAL),
        "oracle_ablation": None if oracle_payload is None else str(ORACLE),
        "threshold_sensitivity": None if threshold_payload is None else str(THRESHOLD),
        "retention_audit": None if retention_payload is None else str(RETENTION),
        "pbmc_primary_certified": str(PBMC_CERTIFIED),
        "pbmc_secondary_superseded_reference": (
            None if pbmc_secondary_payload is None else str(PBMC_FULL)
        ),
        "kdd": None if kdd_payload is None else str(PROJECT / "results/confirmatory/dual-bound-force-kdd-full.json"),
        "timing": None if timing_payload is None else str(PROJECT / "results/confirmatory/dual-bound-force-timing-full.json"),
        "supported_scientific_hypotheses": statistical["results"]["decision_summary"]["supported"],
        "unsupported_scientific_hypotheses": statistical["results"]["decision_summary"]["unsupported"],
        "engineering_evidence_separate": True,
        "synthetic_inference_figure": {
            "analysis_unit": "paired seed after equal weighting of spectra for the aggregate",
            "aggregate_pairs_per_hypothesis": 30,
            "strong_gap_pairs_per_hypothesis": 30,
            "weak_gap_pairs_per_hypothesis": 30,
            "interval": "two-sided 95% paired-t interval",
        },
        "generated_files": [
            "generated_results.tex", "table_primary_criteria.tex", "table_synthetic.tex",
            "table_eigengap_interactions.tex",
            "table_dense.tex",
            "table_target_decomposition.tex",
            "table_external.tex", "table_pbmc.tex", "table_resources.tex",
            "table_retention.tex",
            "figure_synthetic_frontier.png",
            "figure_calibration_audit.png",
            "figure_threshold_sensitivity.png",
        ],
    }
    (PROJECT / "manuscript/ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(PROJECT / "manuscript/ASSET_MANIFEST.json")


if __name__ == "__main__":
    main()
