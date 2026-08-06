# Dual-Bound FORCE

Dual-Bound FORCE estimates correlations and low-rank principal subspaces from
contaminated, high-dimensional data streams. It controls each row before adding
that row to a compact Frequent Directions sketch, so memory does not grow with
the length of the stream.

This repository provides the estimator, experiment code, tests, fixed study
configuration, and the result files that generated the paper's tables and
figures. It does not redistribute FORCE, MAD-FORCE, or Sketch-FORCE, which are
available through their own repositories. We release Dual-Bound FORCE under the
MIT License at <https://github.com/pz1004/dual-bound-force>.

## Algorithm summary

Dual-Bound FORCE processes a stream in five stages:

1. **Calibrate each feature.** The estimator reserves the first
   `calibration_size` rows and computes an exact empirical median and a
   normal-consistent median absolute deviation for every feature. A positive
   scale floor handles constant and nearly constant features.
2. **Learn a reference subspace.** It standardizes the calibration rows, clips
   each coordinate to `[-marginal_lambda, marginal_lambda]`, and builds a
   preliminary Frequent Directions sketch. The leading right-singular vectors
   form the reference basis for the estimation phase.
3. **Set two influence limits.** It splits each standardized calibration row
   into a component parallel to the reference basis and an orthogonal residual.
   It then uses robust summaries of their norms to set separate parallel and
   residual radii.
4. **Transform each new row.** It applies the fixed location and scale, clips
   extreme coordinates, and contracts the parallel and residual components
   independently when they exceed their radii. This step controls both
   modeled-direction leverage and out-of-subspace energy.
5. **Update the correlation sketch.** It inserts the bounded row into a
   `2k`-by-`p` Frequent Directions buffer. It excludes calibration rows from the
   estimation denominator and normalizes the resulting scatter to obtain a
   correlation matrix and its leading subspace.

For a fixed sketch rank `k`, the active estimation state grows in proportion to
`p*k + p`. Calibration temporarily stores data proportional to
`p*(calibration_size + k)`. A dense correlation or scatter output still needs a
`p`-by-`p` array. Treat `get_scale_scatter()` as a scale-reconstructed robust
scatter, not as an unconditional estimate of classical covariance.

You can also set `epoch_size` to recalibrate at known intervals. This scheduled
mode does not detect change points.

## Environments

- Use Python 3.11 or newer. We verified the reported release with Python 3.12.3.
- The core package uses NumPy, SciPy, and Numba.
- The test suite also uses pytest, scikit-learn, and psutil.
- Paper-asset generation uses Matplotlib.
- The estimator runs on a central processing unit and does not require a
  graphics processor.
- The resource-capped experiment runners use the POSIX `resource` interface, so
  run those tools on Linux or macOS.

See `requirements.lock` for the exact package versions used during verification.
Timing will vary with the processor and numerical library, and the experiment
runners record that environment in their output.

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/pz1004/dual-bound-force.git
cd dual-bound-force
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

Add the testing and paper-asset dependencies when you need them:

```bash
python -m pip install -e ".[test,paper]"
```

To recreate the verified environment exactly, install the pinned packages first:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

On Windows, activate the environment with `.venv\Scripts\activate`. The estimator
API works across platforms, but the resource-capped experiment runners require a
POSIX-like system.

## Usage

### Estimate a correlation matrix and subspace

The example below uses the parameter values fixed for the paper. Pass a finite
two-dimensional array with `p` columns. The estimator uses the first 512 rows for
calibration and the remaining rows for estimation.

```python
import numpy as np

from dual_bound_force import DualBoundFORCE

rng = np.random.default_rng(0)
stream = rng.standard_normal((2_000, 50))

estimator = DualBoundFORCE(
    p=50,
    k=10,
    calibration_size=512,
    marginal_lambda=3.0,
    parallel_lambda=4.0,
    residual_lambda=1.5,
)
estimator.fit(stream)

correlation = estimator.get_correlation()
subspace = estimator.get_subspace(rank=5)
robust_scatter = estimator.get_scale_scatter()
transformed_query = estimator.transform(rng.standard_normal(50))
diagnostics = estimator.get_diagnostics()

print(correlation.shape, subspace.shape)
print(diagnostics["effective_n"], diagnostics["state_bytes"])
```

For a row-at-a-time stream, call `update()`:

```python
estimator = DualBoundFORCE(p=50, k=10, calibration_size=512)
for row in stream:
    estimator.update(row)

correlation = estimator.finalize()
```

You may call `transform()` as soon as calibration finishes; it does not change
the estimator state. The correlation, scatter, subspace, and `finalize()` methods
also need at least one estimation row. The estimator returns zero for undefined
zero-variance correlations, including their diagonal entries.

### Run a quick native experiment

This smoke run uses only Dual-Bound FORCE and does not need an external baseline
repository:

```bash
dual-bound-force-study \
  --stage confirmatory \
  --tiers sketch \
  --methods dual_bound \
  --smoke \
  --output-dir results/smoke
```

Run `dual-bound-force-study --help` to see the available tiers, seeds, resource
limits, and output options. Each runner writes machine-readable JSON and CSV
files.

Experiments that select FORCE, MAD-FORCE, or Sketch-FORCE need a separate
`--baseline-dir` containing those three source trees. The runner checks the
required files against `src/dual_bound_force/experiments/baseline_sources.json`
and stops when a source is missing or does not match. It never substitutes a
different implementation silently. Native Frequent Directions,
ridge-regularized Frequent Directions, exact-median-absolute-deviation Frequent
Directions, and Dual-Bound FORCE do not need external baseline repositories.

### Recreate the paper tables and figures

The `results/paper/` directory contains the exact JSON and CSV inputs used in the
paper. You can regenerate the disclosed tables and 600-dpi PNG figures without
rerunning an experiment:

```bash
python tools/verify_disclosure.py
python tools/generate_paper_assets.py --output-dir generated-paper-assets
```

Run the tests with:

```bash
python -m pytest -q
```

The verification tool checks the disclosed result set, strict JSON and CSV
syntax, portable paths, source fingerprints, licensing, and repository
boundaries. We normalized machine-local paths without changing any numerical
result, seed, parameter, status, checksum, or statistical decision.

## Repository layout

- `src/dual_bound_force/` contains the estimator, theory utilities, native
  baselines, and experiment runners.
- `results/paper/` contains only the result bundles used by paper tables and
  figures.
- `preregistration/` contains the fixed experimental configuration and checksum.
- `tools/generate_paper_assets.py` rebuilds the disclosed tables and figures.
- `tools/verify_disclosure.py` checks the public repository and result files.
- `tests/` contains portable estimator, theory, baseline, and solver tests.

We do not plan to assign a permanent code DOI or maintain a separate versioned
archive.
