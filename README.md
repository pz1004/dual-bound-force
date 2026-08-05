# Dual-Bound FORCE

Dual-Bound FORCE is a robust streaming correlation and principal-subspace
estimator for contaminated, high-dimensional data. The source code is released
under the MIT License at <https://github.com/pz1004/dual-bound-force>.

This disclosure contains the executable estimator, experiment runners, portable
tests, frozen experimental configuration, and only the result files used to
generate the paper's tables and figures. It excludes manuscript files, external
datasets, caches, exploratory results, and source code from the separately
published FORCE, MAD-FORCE, and Sketch-FORCE projects.

## Algorithm summary

Dual-Bound FORCE combines robust marginal calibration, two complementary row
influence bounds, and a Frequent Directions sketch:

1. **Calibrate the marginals.** The first `calibration_size` observations are
   stored. For each coordinate, the estimator computes the exact empirical
   median and the normal-consistent median absolute deviation. A positive scale
   floor protects constant or nearly constant coordinates.
2. **Construct a calibration subspace.** Calibration rows are standardized,
   clipped coordinatewise to `[-marginal_lambda, marginal_lambda]`, and passed
   through a preliminary Frequent Directions sketch. Its leading right-singular
   vectors define a basis that remains fixed during the estimation phase.
3. **Calibrate component radii.** Each standardized calibration row is decomposed
   into a component parallel to the basis and an orthogonal residual. Robust
   upper radii are computed from the empirical medians and median absolute
   deviations of the two component norms.
4. **Bound each estimation row.** A new row is standardized and marginally
   clipped using the fixed calibration map. Its parallel and residual components
   are then contracted independently when their norms exceed the corresponding
   calibrated radii. Thus, large modeled-direction leverage and large
   out-of-subspace residuals are controlled separately.
5. **Update the sketch.** The bounded row is inserted into a `2k`-by-`p`
   Frequent Directions buffer. Calibration rows are excluded from the estimation
   denominator. The primary outputs are the normalized correlation estimate and
   its leading subspace. `get_scale_scatter()` returns a scale-reconstructed
   robust scatter and should not be interpreted as an unconditional classical
   covariance estimate.

With fixed sketch rank `k`, active estimation state is proportional to `p*k + p`.
Calibration temporarily requires storage proportional to
`p*(calibration_size + k)`, and requesting a dense correlation or scatter matrix
creates a `p`-by-`p` output. Optional scheduled epochs recalibrate at known
intervals; they do not perform change-point detection.

## Environments

- **Python:** 3.11 or newer; the reported verification used Python 3.12.3.
- **Core dependencies:** NumPy 2.0 or newer, SciPy 1.12 or newer, and Numba 0.65
  or newer.
- **Testing dependencies:** pytest, scikit-learn, and psutil.
- **Paper-asset dependency:** Matplotlib.
- **Hardware:** CPU execution; no graphics processor is required. Resource-capped
  experiment runners use the POSIX `resource` interface and are therefore best
  run on Linux or macOS.

The exact package versions used for verification are recorded in
`requirements.lock`. Numerical timing depends on the processor and linear
algebra library, so timing experiments explicitly record their environment.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/pz1004/dual-bound-force.git
cd dual-bound-force
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the package for ordinary use:

```bash
python -m pip install -e .
```

Install the testing and paper-asset dependencies when needed:

```bash
python -m pip install -e ".[test,paper]"
```

For the exact verified environment, install the pinned dependencies before the
package:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of
the POSIX activation command. The estimator API is portable, although the
resource-capped experiment runners noted above require a POSIX-like system.

## Usage

### Python API

The following example uses the fixed parameter values reported in the paper.
The input must be a finite two-dimensional array whose columns match `p`. At
least one observation after calibration is required before correlation, scatter,
or subspace estimates are available.

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

For a row-at-a-time stream, call `update()` instead of `fit()`:

```python
estimator = DualBoundFORCE(p=50, k=10, calibration_size=512)
for row in stream:
    estimator.update(row)

correlation = estimator.finalize()
```

`transform()` becomes available as soon as initial calibration completes and
does not modify estimator state. `get_correlation()`, `get_scale_scatter()`,
`get_subspace()`, and `finalize()` additionally require at least one estimation
row. Undefined zero-variance correlations, including their diagonal entries,
are returned as zero.

### Run a native smoke experiment

This command exercises only Dual-Bound FORCE and does not require the prior
FORCE-family repositories:

```bash
dual-bound-force-study \
  --stage confirmatory \
  --tiers sketch \
  --methods dual_bound \
  --smoke \
  --output-dir results/smoke
```

Use `dual-bound-force-study --help` to inspect all tier, seed, resource, and
output options. Every experiment writes machine-readable JSON and CSV output.

Comparisons using FORCE, MAD-FORCE, or Sketch-FORCE require a separate
`--baseline-dir` containing those three source trees. Required files are checked
against `src/dual_bound_force/experiments/baseline_sources.json`; missing or
mismatched sources cause an explicit failure rather than a silent substitution.
Native Frequent Directions, ridge-regularized Frequent Directions, exact-median-
absolute-deviation Frequent Directions, and Dual-Bound FORCE do not require
external baseline repositories.

### Regenerate the disclosed paper assets

The files under `results/paper/` are the exact JSON and CSV inputs used by the
paper's tables and figures. Regenerate the disclosed assets without rerunning
the experiments as follows:

```bash
python tools/verify_disclosure.py
python tools/generate_paper_assets.py --output-dir generated-paper-assets
```

Run the portable test suite with:

```bash
python -m pytest -q
```

`DISCLOSURE_MANIFEST.json` records the disclosure allowlist, file hashes,
path-normalization records, and the mapping from each generated paper asset to
its result inputs. Machine-local paths were normalized without changing any
numerical result, seed, parameter, status, checksum, or statistical decision.

## Repository layout

- `src/dual_bound_force/`: estimator, theory utilities, native baselines, and
  experiment runners.
- `results/paper/`: only the result bundles used by paper tables and figures.
- `preregistration/`: frozen experimental configuration and checksum.
- `tools/generate_paper_assets.py`: table and figure regeneration.
- `tools/verify_disclosure.py`: disclosure integrity validation.
- `DISCLOSURE_MANIFEST.json`: hashes and result-to-asset dependency records.

No permanent code DOI or separately versioned archive is planned.
