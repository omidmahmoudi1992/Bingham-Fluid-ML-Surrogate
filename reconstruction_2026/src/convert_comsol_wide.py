"""Convert COMSOL wide-format Level Set exports into analysis-ready arrays.

Expected raw layout (COMSOL 6.3 Data export):
    8 metadata lines starting with '%'
    1 header line starting with '% x,y,ls.Vf1 (1) @ t=...'
    one row per mesh node: x, y, Vf(t0), Vf(t1), ...

The script verifies that every case uses the same coordinates and output
times, writes a compressed stacked NPZ archive, and optionally writes a
reproducible uniformly sampled tidy CSV with columns:
    x, y, time, tau_y, mu_p, Vf, case_no, node_id

This is reconstruction code prepared in October 2026; it is not presented as
the applicant's original, unpreserved training code.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Order is the COMSOL Parametric Sweep "Specified combinations" order.
CASES = [
    (1, 0.0, 0.05),
    (2, 5.0, 0.10),
    (3, 10.0, 0.20),
    (4, 20.0, 0.30),
    (5, 0.0, 0.30),
    (6, 20.0, 0.05),
    (7, 5.0, 0.30),
    (8, 10.0, 0.05),
]


def read_comsol_wide(path: Path) -> tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        meta_lines = []
        header_line = None
        for line in handle:
            if not line.startswith("%"):
                break
            if line.startswith("% x,"):
                header_line = line.rstrip("\n")
            else:
                meta_lines.append(line.rstrip("\n"))
        if header_line is None:
            raise ValueError(f"No COMSOL '% x,...' header found in {path}")

    metadata: dict[str, str] = {}
    for line in meta_lines:
        text = line.lstrip("% ").strip()
        if "," in text:
            key, value = text.split(",", 1)
            metadata[key.strip()] = value.strip().strip('"')

    times = np.array(
        [float(x) for x in re.findall(r"@\s*t=([0-9]*\.?[0-9]+)", header_line)],
        dtype=np.float64,
    )
    if len(times) == 0:
        raise ValueError(f"No output times parsed from {path}")

    numeric = np.loadtxt(path, delimiter=",", skiprows=9, dtype=np.float32)
    expected_cols = 2 + len(times)
    if numeric.ndim != 2 or numeric.shape[1] != expected_cols:
        raise ValueError(
            f"{path}: expected a 2D matrix with {expected_cols} columns, got {numeric.shape}"
        )

    coords = numeric[:, :2].astype(np.float64)
    values = numeric[:, 2:].astype(np.float32)
    return metadata, times, coords, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--tidy-csv", type=Path, default=None)
    parser.add_argument("--qc-json", type=Path, default=None)
    parser.add_argument("--samples-per-case", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reference_coords: np.ndarray | None = None
    reference_times: np.ndarray | None = None
    stacks: list[np.ndarray] = []
    qc: list[dict] = []
    tidy_frames: list[pd.DataFrame] = []

    for case_no, tau_y, mu_p in CASES:
        pattern = f"case{case_no:02d}_*.csv"
        matches = sorted(args.raw_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one file for {pattern}, found {matches}")
        path = matches[0]
        print(f"Reading case {case_no}: {path.name} ...", flush=True)
        metadata, times, coords, values = read_comsol_wide(path)

        if reference_coords is None:
            reference_coords, reference_times = coords, times
        else:
            if coords.shape != reference_coords.shape or not np.allclose(coords, reference_coords, rtol=0, atol=1e-10):
                raise ValueError(f"{path.name}: coordinates differ from case 01")
            if times.shape != reference_times.shape or not np.allclose(times, reference_times, rtol=0, atol=1e-12):
                raise ValueError(f"{path.name}: output times differ from case 01")

        if int(metadata.get("Nodes", -1)) != coords.shape[0]:
            raise ValueError(f"{path.name}: metadata node count does not match data rows")
        if int(metadata.get("Expressions", -1)) != len(times):
            raise ValueError(f"{path.name}: metadata expression count does not match parsed times")

        stacks.append(values)
        qc.append(
            {
                "case_no": case_no,
                "file": path.name,
                "tau_y_Pa": tau_y,
                "mu_p_Pa_s": mu_p,
                "model": metadata.get("Model"),
                "version": metadata.get("Version"),
                "date": metadata.get("Date"),
                "nodes": int(coords.shape[0]),
                "times": int(len(times)),
                "time_min_s": float(times.min()),
                "time_max_s": float(times.max()),
                "x_min_m": float(coords[:, 0].min()),
                "x_max_m": float(coords[:, 0].max()),
                "y_min_m": float(coords[:, 1].min()),
                "y_max_m": float(coords[:, 1].max()),
                "unique_x": int(len(np.unique(coords[:, 0]))),
                "unique_y": int(len(np.unique(coords[:, 1]))),
                "Vf_min": float(np.nanmin(values)),
                "Vf_max": float(np.nanmax(values)),
                "Vf_nan_count": int(np.isnan(values).sum()),
                "Vf_fraction_between_0.05_0.95": float(np.mean((values > 0.05) & (values < 0.95))),
            }
        )

        if args.tidy_csv:
            total = values.size
            n = min(args.samples_per_case, total)
            flat_idx = rng.choice(total, size=n, replace=False)
            node_id, time_id = np.unravel_index(flat_idx, values.shape)
            tidy_frames.append(
                pd.DataFrame(
                    {
                        "x": coords[node_id, 0],
                        "y": coords[node_id, 1],
                        "time": times[time_id],
                        "tau_y": tau_y,
                        "mu_p": mu_p,
                        "Vf": values[node_id, time_id].astype(np.float64),
                        "case_no": case_no,
                        "node_id": node_id,
                    }
                )
            )

    assert reference_coords is not None and reference_times is not None
    stacked = np.stack(stacks, axis=0)  # case, node, time
    case_params = np.array([(tau, mu) for _, tau, mu in CASES], dtype=np.float64)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        x=reference_coords[:, 0],
        y=reference_coords[:, 1],
        times=reference_times,
        case_params=case_params,
        Vf=stacked,
    )
    print(f"Wrote {args.out_npz} with Vf shape {stacked.shape}")

    if args.tidy_csv:
        tidy = pd.concat(tidy_frames, ignore_index=True).sample(frac=1.0, random_state=args.seed)
        args.tidy_csv.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(args.tidy_csv, index=False)
        print(f"Wrote sampled tidy CSV: {args.tidy_csv} ({len(tidy)} rows)")

    qc_path = args.qc_json or (args.out_npz.with_suffix(".qc.json"))
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Wrote QC summary: {qc_path}")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
