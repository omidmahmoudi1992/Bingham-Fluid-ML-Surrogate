# Bingham Dam-Break Surrogate — Reconstruction Package

## Provenance and honesty note

The applicant's original executed Python notebooks were not preserved in the public repository; the uploaded notebooks contain workflow descriptions and the repository contains output CSV/PNG files. This package is therefore a **new reconstruction prepared in October 2026** from newly exported COMSOL data. It must not be described as the original code.

## Data

The reconstruction uses eight COMSOL 6.3 wide-format exports from `Bingham_DamBreak_2D_LevelSet.mph` (one file per rheological case). Each file contains 17,231 mesh nodes and `ls.Vf1` at 151 output times from 0 to 1.5 s.

The eight specified combinations are:

| case | tau_y (Pa) | mu_p (Pa·s) |
|---:|---:|---:|
| 1 | 0 | 0.05 |
| 2 | 5 | 0.10 |
| 3 | 10 | 0.20 |
| 4 | 20 | 0.30 |
| 5 | 0 | 0.30 |
| 6 | 20 | 0.05 |
| 7 | 5 | 0.30 |
| 8 | 10 | 0.05 |

`src/convert_comsol_wide.py` verifies identical coordinates/times across cases and creates a reproducible 320,000-row uniform sample (`x, y, time, tau_y, mu_p, Vf, case_no, node_id`).

## Main reconstruction run

Reported main metrics use a 160,000-row reproducible subset of the sampled tidy dataset and a Random Forest with 150 trees and `min_samples_leaf=10`:

| protocol | test n | R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| random rows | 32,000 | 0.971 | 0.0516 | 0.0181 |
| leave-one-case-out | 160,000 | 0.961 | 0.0604 | 0.0223 |
| future time | 47,834 | 0.529 | 0.1619 | 0.0687 |

The random-row result measures interpolation between nearby sampled space–time points and is optimistic. The leave-one-case-out result tests complete unseen rheological combinations. The future-time result shows that a pointwise Random Forest without temporal state does **not** extrapolate reliably to later times.

At the diffuse interface (`0.05 < Vf < 0.95`), errors are larger: RMSE 0.117 (random rows), 0.135 (leave-one-case-out) and 0.264 (future time).

## Reproduce

All commands below are run from inside this `reconstruction_2026/` folder. The sampled dataset distributed here is gzip-compressed; pandas reads it directly.

Convert raw COMSOL exports:

```bash
python src/convert_comsol_wide.py \
  --raw-dir /path/to/raw_cases \
  --out-npz outputs_reconstruction/bingham_cases.npz \
  --tidy-csv data/bingham_tidy_sample_320k.csv.gz \
  --samples-per-case 40000 --seed 42
```

Train and evaluate the reported main configuration:

```bash
python src/train_evaluate.py \
  --data data/bingham_tidy_sample_320k.csv.gz \
  --out outputs_reconstruction \
  --n-estimators 150 --min-samples-leaf 10 --max-rows 160000
```

Supplementary interface, reduced legacy-subset and front-position diagnostics:

```bash
python src/analyze_reconstruction.py \
  --tidy-csv data/bingham_tidy_sample_320k.csv.gz \
  --outputs outputs_reconstruction \
  --npz outputs_reconstruction/bingham_cases.npz
```

## Interpretation limits

- The archived outputs report a higher random-sample score (R² ≈ 0.996), but their exact sampling and model configuration are unavailable. A reduced comparison restricted to cases 1–4 and `t <= 1.0 s` gives R² = 0.988 and RMSE = 0.0361 under a different, explicitly documented configuration; it is not an exact reproduction claim.
- Impurity feature importance is dominated by `y`, `x` and `time`. The small pointwise importance of `tau_y` and `mu_p` does not mean rheology has no physical effect: the near-bottom COMSOL front is systematically slower for the higher-resistance cases at intermediate times (for example at 0.30 s, x ≈ 0.785 m for case 1 versus 0.710 m for case 4).
- This is an offline pointwise surrogate, not a live digital twin and not a time-stepping reduced-order model.
