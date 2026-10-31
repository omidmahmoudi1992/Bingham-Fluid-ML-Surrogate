"""Prepare Orange Data Mining ready CSV datasets from the Bingham CFD tidy sample.

Generates two lightweight CSV files in `orange_workflow/` with Orange's native
single-row `#` type/role header annotations:

1. `bingham_orange_ready_40k.csv` (40,000 rows; 5,000 uniform rows per case):
   - Features (Numeric) : x, y, time, tau_y, mu_p
   - Target (Numeric)   : Vf
   - Metas              : case_no (Categorical: Case_1..Case_8), node_id (Numeric)

2. `unseen_rheology_mesh_t05.csv` (6,000 COMSOL mesh nodes at t = 0.50 s):
   - Unseen rheological combination (tau_y = 12.0 Pa, mu_p = 0.15 Pa*s) with no
     ground-truth Vf required, demonstrating instant 2D CFD contour prediction
     in Orange without running a new COMSOL simulation.

Pass `--full` to export all 320,000 rows for the training dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "reconstruction_2026" / "data" / "bingham_tidy_sample_320k.csv.gz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bingham_orange_ready_40k.csv"
DEFAULT_QUERY_OUTPUT = Path(__file__).resolve().parent / "unseen_rheology_mesh_t05.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-output", type=Path, default=DEFAULT_QUERY_OUTPUT)
    parser.add_argument("--rows-per-case", type=int, default=5000)
    parser.add_argument("--query-nodes", type=int, default=6000)
    parser.add_argument("--full", action="store_true", help="Export all 320,000 rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    raw_df["Vf"] = raw_df["Vf"].clip(0.0, 1.0)

    # 1) Build dense 2D mesh query at t = 0.50 s for an unseen rheology (tau_y=12, mu_p=0.15)
    nodes = raw_df[["node_id", "x", "y"]].drop_duplicates("node_id").sort_values("node_id")
    if args.query_nodes and len(nodes) > args.query_nodes:
        nodes = nodes.sample(n=args.query_nodes, random_state=args.seed).sort_values("node_id")

    query_df = pd.DataFrame(
        {
            "C#x": nodes["x"].round(6),
            "C#y": nodes["y"].round(6),
            "C#time": 0.50,
            "C#tau_y": 12.0,
            "C#mu_p": 0.15,
            "mC#node_id": nodes["node_id"].astype(int),
        }
    )
    args.query_output.parent.mkdir(parents=True, exist_ok=True)
    query_df.to_csv(args.query_output, index=False)
    print(
        f"Wrote {len(query_df):,} mesh query rows to {args.query_output} "
        f"({args.query_output.stat().st_size / 1e3:.1f} KB)"
    )

    # 2) Build main 8-case training & validation dataset
    df = raw_df
    if not args.full:
        parts = []
        for case_no, group in df.groupby("case_no", sort=True):
            n = min(args.rows_per_case, len(group))
            parts.append(group.sample(n=n, random_state=args.seed + int(case_no)))
        df = pd.concat(parts, ignore_index=True)

    df = df.sort_values(["case_no", "time", "node_id"]).reset_index(drop=True)

    out_df = pd.DataFrame(
        {
            "C#x": df["x"].round(6),
            "C#y": df["y"].round(6),
            "C#time": df["time"].round(4),
            "C#tau_y": df["tau_y"].round(4),
            "C#mu_p": df["mu_p"].round(4),
            "cC#Vf": df["Vf"].round(6),
            "mD#case_no": "Case_" + df["case_no"].astype(int).astype(str),
            "mC#node_id": df["node_id"].astype(int),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(
        f"Wrote {len(out_df):,} training rows to {args.output} "
        f"({args.output.stat().st_size / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
