# Orange Data Mining Workflow — Bingham Fluid ML Surrogate

This folder provides a ready-to-open **Orange Data Mining (`.ows`)** visual workflow that reproduces the repository's machine learning surrogate modeling, scientific validation protocols, error diagnostics, and 2D dam-break contour predictions without writing code.

## Contents

| File | Size | Description |
|---|---:|---|
| `Bingham_Surrogate_Workflow.ows` | ~32 KB | Pre-wired 25-widget Orange 3 workflow (portable relative paths) |
| `bingham_orange_ready_40k.csv` | ~1.9 MB | Reproducible 40,000-row sample across all 8 COMSOL rheological cases (5,000 uniform rows/case) with Orange `#` header role annotations |
| `unseen_rheology_mesh_t05.csv` | ~195 KB | 6,000 COMSOL mesh nodes at `t = 0.50 s` for an unseen Bingham fluid (`tau_y = 12 Pa`, `mu_p = 0.15 Pa·s`) |
| `prepare_orange_data.py` | ~3 KB | Script to regenerate the CSVs (use `--full` for all 320,000 rows) |
| `build_workflow.py` | ~18 KB | Script that programmatically builds and validates the `.ows` file |

## Quick Start (How to Run in Orange)

1. Keep `Bingham_Surrogate_Workflow.ows`, `bingham_orange_ready_40k.csv`, and `unseen_rheology_mesh_t05.csv` **in the same folder** on your computer.
2. Open **Orange Data Mining** and click **`File -> Open...`** (or press `Ctrl + O`), then select `Bingham_Surrogate_Workflow.ows`.
3. Orange automatically loads both CSV files from the workflow directory, assigns `x, y, time, tau_y, mu_p` as **Features**, `Vf` as **Target**, and `case_no, node_id` as **Metas**, and executes all widgets in ~20–40 seconds.

## Workflow Architecture (4 Zones)

1. **Data & Feature Ranking (Left)**
   - **`1. Bingham CFD Dataset (8 Cases)`** -> **`Data Info`**, **`CFD Data Table`**, **`Select Columns`**, and **`Data Sampler`** (set to `50%` = 20,000 rows by default for fast interactive response; increase to `100%` anytime).
   - **`Feature Importance (Rank)`**: Ranks `y`, `x`, `time`, `tau_y`, and `mu_p` using both **Random Forest** impurity importance and **Univariate Regression**.

2. **All 3 Scientific Validation Protocols (Center)**
   - **`Eval 1: Random 80/20 (Interpolation)`**: Compares **Random Forest** vs. **Gradient Boosting** under random row splitting (`R² ≈ 0.94` vs `0.83` on the fast 20k subset).
   - **`Eval 2: Leave-One-Case-Out (8 Cases)`**: Uses `Cross validation by feature (case_no)` to hold out each complete `(tau_y, mu_p)` rheological case in turn (`R² ≈ 0.935`).
   - **`Eval 3: Future-Time (t > 1.05 s)`**: Uses `Select Rows (time <= 1.05)` + `Test on test data` to demonstrate temporal extrapolation limits (`R² ≈ 0.51`).

3. **Out-of-Sample Case 4 Diagnostics (Top/Middle Right)**
   - Holds out **`Case_4` (`tau_y = 20 Pa, mu_p = 0.30 Pa·s`)**, trains **`Trained Surrogate (7 Cases)`**, and feeds **`Case 4 Validation Predictions`** into:
     - **`Actual vs Predicted Vf`** (`Scatter Plot` with regression line)
     - **`Formula (abs_error)`**: Computes `abs_error = abs(Vf - RF_Surrogate)`
     - **`Error Distribution`** (`Distributions` histogram)
     - **`2D Interface Error Map (Case 4)`** (`Scatter Plot`: `X = x`, `Y = y`, `Color = abs_error`)
     - **`Mean Error by Time`** (`Group by` on `time`) -> **`Error vs Time Curve`** (`Scatter Plot`: `X = time`, `Y = abs_error - Mean`)

4. **Surrogate Deployment on an Unseen Fluid (Bottom Right)**
   - Connects **`Trained Surrogate (7 Cases)`** and **`5. Unseen Fluid Mesh @ t=0.5s`** to **`Predict Unseen Fluid @ t=0.5s`** and **`2D Contour: Unseen Fluid (t=0.5s)`**, rendering a 6,000-node 2D dam-break volume-fraction contour without running a new COMSOL simulation.
