# Machine Learning Surrogate Model for Bingham Fluid CFD Simulation

## Project Overview
A Python-based machine learning workflow for predicting CFD simulation outputs of Bingham fluid dam-break flow.

The workflow combines:
- COMSOL CFD simulations
- Python data processing
- Machine Learning regression
- Model validation
- Sensitivity analysis

## ML Problem Definition

Input features:
- x coordinate
- y coordinate
- time
- yield stress (tau_y)
- plastic viscosity (mu_p)

Target:
- Volume Fraction (Vf)

## Machine Learning Model

Model:
- Random Forest Regression

Evaluation:
- R2 score
- RMSE
- MAE

## Analysis Performed

- COMSOL data extraction
- Dataset preparation
- Surrogate model training
- COMSOL vs ML comparison
- Error analysis
- Feature importance
- Rheological sensitivity analysis

## Technologies

Python, NumPy, Pandas, Scikit-learn, Matplotlib, Jupyter Notebook
## Reproducibility note (2026)

The notebooks previously uploaded to this repository contain workflow descriptions rather than the original executed code cells. A transparent, runnable reconstruction from newly exported COMSOL data for all eight rheological cases is provided in [`reconstruction_2026/`](reconstruction_2026/). It reports random-row, leave-one-case-out and future-time evaluation separately; the future-time result is intentionally retained as a limitation of the pointwise Random Forest surrogate.

## Orange Data Mining Visual Workflow

A pre-wired, portable **Orange Data Mining (`.ows`)** visual workflow reproducing all three validation protocols, error diagnostics, and unseen-fluid 2D contour predictions is available in [`orange_workflow/`](orange_workflow/).

