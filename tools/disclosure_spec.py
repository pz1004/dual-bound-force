"""Public result set and paper-asset dependencies.

Keep this information in executable source so verification and asset generation
do not depend on a repository-level disclosure manifest.
"""

from __future__ import annotations


REPOSITORY_URL = "https://github.com/pz1004/dual-bound-force"

PAPER_RESULT_FILES = (
    "results/paper/confirmatory.json",
    "results/paper/confirmatory.raw.csv",
    "results/paper/oracle.json",
    "results/paper/oracle.raw.csv",
    "results/paper/pbmc_certified.json",
    "results/paper/pbmc_certified.raw.csv",
    "results/paper/pbmc_secondary.json",
    "results/paper/pbmc_secondary.raw.csv",
    "results/paper/retention.json",
    "results/paper/retention.raw.csv",
    "results/paper/statistical_reanalysis.json",
    "results/paper/structural_ceiling.json",
    "results/paper/threshold.json",
    "results/paper/threshold.raw.csv",
    "results/paper/timing.json",
    "results/paper/timing.raw.csv",
)

PAPER_ASSET_DEPENDENCIES = {
    "figure_calibration_audit.png": (
        "results/paper/confirmatory.json",
        "results/paper/confirmatory.raw.csv",
    ),
    "figure_synthetic_frontier.png": (
        "results/paper/statistical_reanalysis.json",
    ),
    "figure_threshold_sensitivity.png": (
        "results/paper/threshold.json",
        "results/paper/threshold.raw.csv",
    ),
    "table_dense.tex": (
        "results/paper/confirmatory.json",
        "results/paper/confirmatory.raw.csv",
    ),
    "table_eigengap_interactions.tex": (
        "results/paper/statistical_reanalysis.json",
    ),
    "table_external.tex": (
        "results/paper/pbmc_certified.json",
        "results/paper/pbmc_certified.raw.csv",
    ),
    "table_outlier_pursuit_diagnostics.tex": (
        "results/paper/structural_ceiling.json",
    ),
    "table_pbmc.tex": (
        "results/paper/pbmc_secondary.json",
        "results/paper/pbmc_secondary.raw.csv",
    ),
    "table_primary_criteria.tex": (
        "results/paper/statistical_reanalysis.json",
    ),
    "table_resources.tex": (
        "results/paper/confirmatory.json",
        "results/paper/confirmatory.raw.csv",
        "results/paper/timing.json",
        "results/paper/timing.raw.csv",
    ),
    "table_retention.tex": (
        "results/paper/retention.json",
        "results/paper/retention.raw.csv",
    ),
    "table_structural_ceiling.tex": (
        "results/paper/structural_ceiling.json",
    ),
    "table_synthetic.tex": (
        "results/paper/confirmatory.json",
        "results/paper/confirmatory.raw.csv",
        "results/paper/oracle.json",
        "results/paper/oracle.raw.csv",
    ),
    "table_target_decomposition.tex": (
        "results/paper/confirmatory.json",
        "results/paper/confirmatory.raw.csv",
        "results/paper/oracle.json",
        "results/paper/oracle.raw.csv",
    ),
}
