"""Inspect a raw COMSOL export before the tidy-dataset parser is finalised.

This helper only prints the file header, separator guess and first rows. It does
not assume a particular COMSOL version or column layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Raw COMSOL .txt/.csv export")
    parser.add_argument("--lines", type=int, default=35, help="Number of lines to show")
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"File: {args.file}")
    print(f"Total lines: {len(text)}")
    print("\n--- first lines ---")
    for i, line in enumerate(text[: args.lines], 1):
        print(f"{i:4d}: {line}")

    candidates = {",": 0, ";": 0, "\t": 0, " ": 0}
    for line in text[: min(len(text), 200)]:
        if line.lstrip().startswith("%"):
            continue
        for sep in candidates:
            candidates[sep] += line.count(sep)
    print("\nSeparator counts in first non-comment lines:", candidates)
    print(
        "\nSend this output (or the sample file itself) before exporting all cases, "
        "so the converter can be matched to the exact COMSOL layout."
    )


if __name__ == "__main__":
    main()
