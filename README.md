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
