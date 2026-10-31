"""Train and honestly evaluate a Random Forest surrogate for Bingham dam-break data.

Status
------
New reconstruction code prepared in October 2026. It is NOT the applicant's
original (unpreserved) training code. Its purpose is to rebuild a transparent,
runnable analysis from newly exported COMSOL data.

Input
-----
A tidy CSV with columns: x, y, time, tau_y, mu_p, Vf

Evaluation protocols
--------------------
1. random_rows       : random 80/20 row split (interpolation only; optimistic)
2. leave_one_case_out: hold out each complete (tau_y, mu_p) case in turn
3. future_time       : train on early times, test on later times

Outputs are written to the chosen output directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["x", "y", "time", "tau_y", "mu_p"]
TARGET = "Vf"
REQUIRED = FEATURES + [TARGET]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def make_model(n_estimators: int, min_samples_leaf: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, args) -> tuple[np.ndarray, RandomForestRegressor]:
    model = make_model(args.n_estimators, args.min_samples_leaf)
    model.fit(train[FEATURES], train[TARGET])
    pred = np.clip(model.predict(test[FEATURES]), 0.0, 1.0)
    return pred, model


def save_predictions(frame: pd.DataFrame, pred: np.ndarray, protocol: str, out: Path) -> pd.DataFrame:
    result = frame[FEATURES + [TARGET, "case_id"]].copy()
    result = result.rename(columns={TARGET: "COMSOL_Vf"})
    result["ML_Vf"] = pred
    result["abs_error"] = np.abs(result["COMSOL_Vf"] - result["ML_Vf"])
    result.to_csv(out / f"predictions_{protocol}.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Tidy CSV: x,y,time,tau_y,mu_p,Vf")
    parser.add_argument("--out", type=Path, default=Path("outputs_reconstruction"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--time-fraction", type=float, default=0.70, help="Fraction of unique times used for training in future_time split")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional reproducible subsample for a quick smoke test")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[REQUIRED].dropna().copy()
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42).sort_index()

    if df[TARGET].min() < -0.05 or df[TARGET].max() > 1.05:
        raise ValueError("Vf is far outside the expected [0, 1] range; check the exported expression and units.")
    df[TARGET] = df[TARGET].clip(0.0, 1.0)
    df["case_id"] = (
        "tau_y=" + df["tau_y"].round(8).astype(str) + ", mu_p=" + df["mu_p"].round(8).astype(str)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, float | str]] = []

    # 1) Random row split (reported only as interpolation performance).
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    pred, model = fit_predict(train, test, args)
    pred_frame = save_predictions(test, pred, "random_rows", args.out)
    summary.append({"protocol": "random_rows", **metrics(test[TARGET].to_numpy(), pred)})

    pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).to_csv(
        args.out / "feature_importance_random_rows.csv", index=False
    )

    # Simple diagnostic plots for the random-row model.
    plt.figure(figsize=(5, 5))
    plt.scatter(pred_frame["COMSOL_Vf"], pred_frame["ML_Vf"], s=2, alpha=0.35)
    plt.plot([0, 1], [0, 1], linewidth=1)
    plt.xlabel("COMSOL Vf")
    plt.ylabel("Random Forest Vf")
    plt.title("Random-row split (interpolation only)")
    plt.tight_layout()
    plt.savefig(args.out / "actual_vs_prediction_random_rows.png", dpi=200)
    plt.close()

    err_time = (
        pred_frame.assign(error=pred_frame["COMSOL_Vf"] - pred_frame["ML_Vf"])
        .groupby("time")["error"]
        .apply(lambda e: float(np.sqrt(np.mean(np.square(e)))))
        .rename("RMSE")
        .reset_index()
    )
    err_time.to_csv(args.out / "rmse_vs_time_random_rows.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(err_time["time"], err_time["RMSE"], marker="o", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("RMSE")
    plt.title("Random-row split: RMSE versus time")
    plt.tight_layout()
    plt.savefig(args.out / "rmse_vs_time_random_rows.png", dpi=200)
    plt.close()

    # 2) Leave-one-case-out.
    cases = sorted(df["case_id"].unique())
    if len(cases) >= 3:
        all_true, all_pred, frames, per_case = [], [], [], []
        for case in cases:
            test_c = df[df["case_id"] == case]
            train_c = df[df["case_id"] != case]
            pred_c, _ = fit_predict(train_c, test_c, args)
            all_true.append(test_c[TARGET].to_numpy())
            all_pred.append(pred_c)
            frames.append(save_predictions(test_c, pred_c, f"loco_{case.replace('=', '').replace(',', '').replace(' ', '_')}", args.out))
            per_case.append({"held_out_case": case, **metrics(test_c[TARGET].to_numpy(), pred_c)})
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        summary.append({"protocol": "leave_one_case_out", **metrics(y_true, y_pred)})
        pd.DataFrame(per_case).to_csv(args.out / "metrics_leave_one_case_out_by_case.csv", index=False)
        pd.concat(frames, ignore_index=True).to_csv(args.out / "predictions_leave_one_case_out.csv", index=False)
    else:
        print("Leave-one-case-out skipped: fewer than three rheological cases.")

    # 3) Future-time split.
    times = np.array(sorted(df["time"].unique()))
    if len(times) >= 5:
        cut_index = max(1, min(len(times) - 1, int(round(args.time_fraction * len(times)))))
        cut = times[cut_index - 1]
        train_t = df[df["time"] <= cut]
        test_t = df[df["time"] > cut]
        pred_t, _ = fit_predict(train_t, test_t, args)
        save_predictions(test_t, pred_t, "future_time", args.out)
        summary.append({"protocol": "future_time", **metrics(test_t[TARGET].to_numpy(), pred_t)})
    else:
        print("Future-time split skipped: fewer than five unique time values.")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out / "metrics_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\nImportant: random_rows measures interpolation between nearby sampled points and is optimistic.")
    print("Use leave_one_case_out and future_time to discuss generalisation in an interview.")


if __name__ == "__main__":
    main()
