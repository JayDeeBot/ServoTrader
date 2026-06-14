"""
export_tb_csvs.py

Reads TensorBoard event files directly from disk and exports every scalar
tag as an individual CSV — identical format to TensorBoard's manual download.

No running TensorBoard server required. Run this at any point during or
after training to get fresh CSVs.

Output format (matches TensorBoard manual export exactly)
---------------------------------------------------------
    Wall time, Step, Value
    1700000000.0, 2048, 0.05
    ...

Usage
-----
    # Export latest run (default)
    python3 scripts/ export_tb_csvs.py

    # Export a specific run directory
    python3 scripts/export_tb_csvs.py --logdir /home/jarred/git/ServoTrader/logs/tb/events.out.tfevents.1780042695.jarred-GF62-7RD.77554.0
    
    # Change output directory
    python3 scripts/ export_tb_csvs.py --outdir /tmp/tb_export

    # Export only specific tags (comma-separated, supports wildcards)
    python3 scripts/ export_tb_csvs.py --tags "rollout/*,train/entropy"

Author: Jarred Deluca
Project: ServoTrader
License: MIT
"""

import os
import glob
import argparse
import csv
from pathlib import Path
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------------

DEFAULT_TB_DIR  = "/home/jarred/git/ServoTrader/logs/tb"
DEFAULT_OUT_DIR = "/home/jarred/git/ServoTrader/logs/tb_exports"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _find_latest_run(tb_dir: str) -> str:
    """
    Returns the most recently modified subdirectory of tb_dir.
    If tb_dir itself contains event files, returns tb_dir directly.
    """
    # Check if the root contains event files directly
    if glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")):
        return tb_dir

    # Otherwise find the most recently modified subdirectory
    subdirs = [
        d for d in Path(tb_dir).iterdir()
        if d.is_dir() and list(d.glob("events.out.tfevents.*"))
    ]
    if not subdirs:
        raise FileNotFoundError(
            f"No TensorBoard event files found under: {tb_dir}\n"
            "Make sure training has started and tb_log_dir is correct."
        )
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return str(latest)


def _tag_to_filename(tag: str) -> str:
    """Convert a TensorBoard tag like 'rollout/win_rate' to 'win_rate.csv'."""
    # Take the last component of the tag path
    name = tag.split("/")[-1]
    # Replace any remaining unsafe characters
    name = name.replace(" ", "_").replace(":", "_")
    return f"{name}.csv"


def _matches_filter(tag: str, filters: list) -> bool:
    """Return True if tag matches any of the filter patterns (supports * wildcard)."""
    if not filters:
        return True
    import fnmatch
    return any(fnmatch.fnmatch(tag, pat) for pat in filters)


# ---------------------------------------------------------------------------
#  Main export function
# ---------------------------------------------------------------------------

def export_scalars(
    logdir:   str,
    out_dir:  str,
    tags:     list = None,
    verbose:  bool = True,
) -> dict:
    """
    Load TensorBoard event files from `logdir` and write one CSV per scalar tag.

    Parameters
    ----------
    logdir  : path to a directory containing events.out.tfevents.* files
    out_dir : directory to write CSVs into (created if needed)
    tags    : list of tag filter strings (None = export all)
    verbose : print progress

    Returns
    -------
    dict mapping tag → output CSV path
    """
    SEP = "=" * 60

    if verbose:
        print(f"\n{SEP}")
        print(f"  TENSORBOARD CSV EXPORTER")
        print(f"  Reading: {logdir}")
        print(f"  Output : {out_dir}")
        print(SEP)

    # Load event accumulator
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})   # 0 = load all
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if not available_tags:
        raise RuntimeError(
            f"No scalar tags found in: {logdir}\n"
            "The run may not have written any scalar summaries yet."
        )

    if verbose:
        print(f"\n  Found {len(available_tags)} scalar tags:")
        for t in sorted(available_tags):
            events = ea.Scalars(t)
            print(f"    {t:<40}  ({len(events):,} points)")

    # Apply tag filter
    selected = [t for t in available_tags if _matches_filter(t, tags or [])]
    if verbose:
        print(f"\n  Exporting {len(selected)} tags…")

    os.makedirs(out_dir, exist_ok=True)
    exported = {}

    for tag in sorted(selected):
        events   = ea.Scalars(tag)
        filename = _tag_to_filename(tag)
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Wall time", "Step", "Value"])
            for e in events:
                writer.writerow([e.wall_time, e.step, e.value])

        exported[tag] = out_path
        if verbose:
            print(f"  ✅  {tag:<40} → {filename}  ({len(events):,} rows)")

    if verbose:
        print(f"\n{SEP}")
        print(f"  Done — {len(exported)} CSVs written to:")
        print(f"  {out_dir}")
        print(SEP + "\n")

    return exported


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar tags to CSV files."
    )
    parser.add_argument(
        "--logdir", default=None,
        help=f"Path to TensorBoard log directory or run subdirectory. "
             f"Default: auto-detect latest run under {DEFAULT_TB_DIR}"
    )
    parser.add_argument(
        "--outdir", default=None,
        help=f"Output directory for CSVs. "
             f"Default: {DEFAULT_OUT_DIR}/<run_name>"
    )
    parser.add_argument(
        "--tags", default=None,
        help="Comma-separated list of tag patterns to export (supports * wildcard). "
             "Default: export all tags. "
             "Example: --tags 'rollout/*,train/entropy,train/approx_kl'"
    )
    parser.add_argument(
        "--latest", action="store_true", default=True,
        help="Auto-detect and use the most recent run (default: True)"
    )
    args = parser.parse_args()

    # Resolve logdir
    if args.logdir:
        logdir = args.logdir
    else:
        logdir = _find_latest_run(DEFAULT_TB_DIR)

    # Resolve output dir — use run name as subdirectory
    if args.outdir:
        out_dir = args.outdir
    else:
        run_name = Path(logdir).name
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir  = os.path.join(DEFAULT_OUT_DIR, f"{run_name}_{ts}")

    # Parse tag filters
    tag_filters = [t.strip() for t in args.tags.split(",")] if args.tags else None

    export_scalars(logdir=logdir, out_dir=out_dir, tags=tag_filters)


if __name__ == "__main__":
    main()