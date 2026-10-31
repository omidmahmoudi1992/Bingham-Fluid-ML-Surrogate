"""Supplementary diagnostics for the October 2026 Bingham reconstruction.

Computes:
1. interface-band metrics from saved prediction files;
2. a legacy-subset random split (first four diagonal cases, t <= 1.0 s) for an
   apples-to-apples comparison with the archived output files;
3. near-bottom COMSOL front positions from the stacked NPZ archive.

This is new reconstruction analysis, not the applicant's original code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["x", "y", "time", "tau_y", "mu_p"]


def calc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def interface_metrics(outputs: Path) -> pd.DataFrame:
    rows = []
    files = {
        "random_rows": outputs / "predictions_random_rows.csv",
        "leave_one_case_out": outputs / "predictions_leave_one_case_out.csv",
        "future_time": outputs / "predictions_future_time.csv",
    }
    for protocol, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        mask = (df["COMSOL_Vf"] > 0.05) & (df["COMSOL_Vf"] < 0.95)
        rows.append(
            {
                "protocol": protocol,
                "subset": "interface_0.05<Vf<0.95",
                **calc(df.loc[mask, "COMSOL_Vf"].to_numpy(), df.loc[mask, "ML_Vf"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def legacy_subset(tidy_path: Path, out: Path, n_estimators: int, min_samples_leaf: int) -> pd.DataFrame:
    df = pd.read_csv(tidy_path)
    legacy = df[(df["case_no"].isin([1, 2, 3, 4])) & (df["time"] <= 1.000001)].copy()
    # The reported supplementary comparison uses a fixed 60k subsample so it is
    # reproducible on a small machine. It is not claimed to reproduce the
    # unknown original training configuration.
    if len(legacy) > 60000:
        legacy = legacy.sample(n=60000, random_state=7)
    train, test = train_test_split(legacy, test_size=0.20, random_state=42)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train[FEATURES], train["Vf"])
    pred = np.clip(model.predict(test[FEATURES]), 0.0, 1.0)

    pred_frame = test[FEATURES + ["Vf", "case_no", "node_id"]].rename(columns={"Vf": "COMSOL_Vf"}).copy()
    pred_frame["ML_Vf"] = pred
    pred_frame["abs_error"] = np.abs(pred_frame["COMSOL_Vf"] - pred_frame["ML_Vf"])
    pred_frame.to_csv(out / "predictions_legacy_subset_reduced.csv", index=False)

    mask = (test["Vf"] > 0.05) & (test["Vf"] < 0.95)
    rows = [
        {"protocol": "legacy_subset_random_rows_reduced", "subset": "all", **calc(test["Vf"].to_numpy(), pred)},
        {
            "protocol": "legacy_subset_random_rows_reduced",
            "subset": "interface_0.05<Vf<0.95",
            **calc(test.loc[mask, "Vf"].to_numpy(), pred[mask.to_numpy()]),
        },
    ]
    pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).to_csv(
        out / "feature_importance_legacy_subset_reduced.csv", index=False
    )
    return pd.DataFrame(rows)


def front_positions(npz_path: Path, out: Path) -> pd.DataFrame:
    archive = np.load(npz_path)
    x, y, times, params, vf = archive["x"], archive["y"], archive["times"], archive["case_params"], archive["Vf"]
    near_bottom = y <= 0.012
    rows = []
    for case_idx, (tau_y, mu_p) in enumerate(params):
        for time_idx, time in enumerate(times):
            values = vf[case_idx, near_bottom, time_idx]
            coords = x[near_bottom]
            occupied = coords[values >= 0.5]
            front = float(np.quantile(occupied, 0.995)) if len(occupied) else np.nan
            rows.append(
                {
                    "case_no": case_idx + 1,
                    "tau_y": float(tau_y),
                    "mu_p": float(mu_p),
                    "time": float(time),
                    "front_x_near_bottom_m": front,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "comsol_front_position_near_bottom.csv", index=False)
    selected = result[np.isclose(result["time"], 1.0) | np.isclose(result["time"], 1.5)].copy()
    selected.to_csv(out / "comsol_front_position_selected_times.csv", index=False)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tidy-csv", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True, help="Directory produced by train_evaluate.py")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--legacy-n-estimators", type=int, default=120)
    parser.add_argument("--legacy-min-samples-leaf", type=int, default=5)
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    interface = interface_metrics(args.outputs)
    interface.to_csv(args.outputs / "metrics_interface_band.csv", index=False)
    print("\nInterface-band metrics")
    print(interface.to_string(index=False))

    legacy = legacy_subset(args.tidy_csv, args.outputs, args.legacy_n_estimators, args.legacy_min_samples_leaf)
    legacy.to_csv(args.outputs / "metrics_legacy_subset_reduced.csv", index=False)
    print("\nReduced legacy-subset comparison (cases 1-4, t <= 1.0 s, 60k sample)")
    print(legacy.to_string(index=False))

    front = front_positions(args.npz, args.outputs)
    print("\nNear-bottom COMSOL front positions at selected times")
    print(front.to_string(index=False))


if __name__ == "__main__":
    main()
