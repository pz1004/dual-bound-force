"""Reproducible experiments for the standalone Dual-Bound FORCE study.

The experiment package is deliberately separate from the estimator API.  Its
default settings are the frozen confirmatory settings in ``PREREGISTRATION.md``.
"""

from .metrics import (
    exact_one_sided_sign_flip_p,
    exact_one_sided_sign_p,
    holm_adjust,
    normalized_frobenius_error,
    normalized_projection_error,
    paired_bootstrap_ci,
    paired_mean_inference,
    rank_auc,
)

__all__ = [
    "exact_one_sided_sign_flip_p",
    "exact_one_sided_sign_p",
    "holm_adjust",
    "normalized_frobenius_error",
    "normalized_projection_error",
    "paired_bootstrap_ci",
    "paired_mean_inference",
    "rank_auc",
]
