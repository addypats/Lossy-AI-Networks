#!/usr/bin/env python3
"""Summarize sampled shard-impact JSONL logs.

Groups by collective name and global_step, then reports means and p95 for the
core shard-damage metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize shard-impact JSONL logs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing rank*.jsonl shard-impact logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shard_impact_summary.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_records(input_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(input_dir.glob("rank*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> pd.DataFrame:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record.get("fn_name", "unknown"),
            int(record.get("global_step", 0)),
        )
        grouped[key].append(record)

    rows = []
    for (fn_name, global_step), items in sorted(grouped.items()):
        affected = np.array([float(item.get("affected_elem_fraction", 0.0)) for item in items], dtype=float)
        zeroed = np.array([float(item.get("newly_zeroed_fraction", 0.0)) for item in items], dtype=float)
        norm_ratio = np.array([float(item.get("norm_ratio", 0.0)) for item in items], dtype=float)
        rel_error = np.array([float(item.get("relative_error", 0.0)) for item in items], dtype=float)
        packet_drop = np.array([float(item.get("packet_drop_fraction", 0.0)) for item in items], dtype=float)

        rows.append(
            {
                "fn_name": fn_name,
                "global_step": global_step,
                "sample_count": len(items),
                "mean_affected_elem_fraction": float(affected.mean()) if len(affected) else 0.0,
                "mean_newly_zeroed_fraction": float(zeroed.mean()) if len(zeroed) else 0.0,
                "mean_norm_ratio": float(norm_ratio.mean()) if len(norm_ratio) else 0.0,
                "mean_relative_error": float(rel_error.mean()) if len(rel_error) else 0.0,
                "p95_relative_error": float(np.percentile(rel_error, 95)) if len(rel_error) else 0.0,
                "mean_packet_drop_fraction": float(packet_drop.mean()) if len(packet_drop) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    records = load_records(args.input_dir)
    if not records:
        raise RuntimeError(f"No JSONL records found in {args.input_dir}")

    summary = summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
