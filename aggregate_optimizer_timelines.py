#!/usr/bin/env python3
"""Build representative optimizer timelines from rank-level audit files.

This script scans optimizer audit folders, groups them by configuration
(batch size + loss rate), selects one file per rank, and computes per-step
means for:
- m_post_norm
- v_post_norm
- actual_delta_norm

It supports either JSONL or CSV rank files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


METRIC_KEYS = ("m_post_norm", "v_post_norm", "actual_delta_norm")
TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")
RANK_RE = re.compile(r"rank(\d+)\.(jsonl|csv)$")


@dataclass
class FolderInfo:
    path: Path
    batch_size: Optional[int]
    loss_rate: float
    timestamp: str
    rank_files: Dict[int, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate rank audit files into one timeline per configuration."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("optimizer_audit_logs"),
        help="Root directory containing optimizer audit folders.",
    )
    parser.add_argument(
        "--file-type",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Rank file type to load from each folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimizer_audit_logs") / "aggregated_timelines",
        help="Directory where aggregated timelines will be written.",
    )
    parser.add_argument(
        "--strict-ranks",
        action="store_true",
        help="Fail if a configuration does not resolve to exactly 16 ranks.",
    )
    return parser.parse_args()


def parse_batch_size(folder_name: str) -> Optional[int]:
    match = re.search(r"_bs(\d+)_", folder_name)
    return int(match.group(1)) if match else None


def parse_loss_rate(folder_name: str) -> float:
    if "loss-rate0" in folder_name:
        return 0.0

    marker = "loss-rate_high_persistence_low_intensity_1_"
    idx = folder_name.find(marker)
    if idx == -1:
        raise ValueError(f"Could not parse loss rate from folder: {folder_name}")

    tail = folder_name[idx + len(marker) :]
    first_token = tail.split("_", 1)[0]
    if re.fullmatch(r"\d{8}-\d{6}", first_token):
        return 1.0

    return float(first_token)


def parse_timestamp(folder_name: str) -> str:
    match = TIMESTAMP_RE.search(folder_name)
    return match.group(1) if match else "00000000-000000"


def collect_rank_files(folder_path: Path, file_type: str) -> Dict[int, Path]:
    rank_files: Dict[int, Path] = {}
    for child in folder_path.iterdir():
        if not child.is_file():
            continue
        match = RANK_RE.match(child.name)
        if not match or match.group(2) != file_type:
            continue
        rank = int(match.group(1))
        rank_files[rank] = child
    return rank_files


def discover_folders(root: Path, file_type: str) -> list[FolderInfo]:
    infos: list[FolderInfo] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        rank_files = collect_rank_files(folder, file_type)
        if not rank_files:
            continue
        info = FolderInfo(
            path=folder,
            batch_size=parse_batch_size(folder.name),
            loss_rate=parse_loss_rate(folder.name),
            timestamp=parse_timestamp(folder.name),
            rank_files=rank_files,
        )
        infos.append(info)
    return infos


def infer_batch_size(info: FolderInfo) -> int:
    if info.batch_size is not None:
        return info.batch_size
    # Historical loss-rate0 folders may omit batch size in the folder name.
    # Infer from rank coverage: full 16-rank folder is bs32, partial shard is bs8.
    return 32 if len(info.rank_files) >= 16 else 8


def choose_rank_files(folders: Iterable[FolderInfo]) -> Dict[int, Path]:
    chosen: Dict[int, Tuple[str, Path]] = {}
    for info in folders:
        for rank, file_path in info.rank_files.items():
            prev = chosen.get(rank)
            if prev is None or info.timestamp >= prev[0]:
                chosen[rank] = (info.timestamp, file_path)
    return {rank: item[1] for rank, item in chosen.items()}


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_csv_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def aggregate_files(rank_files: Dict[int, Path], file_type: str) -> Dict[int, dict]:
    stats = defaultdict(
        lambda: {
            "m_post_norm_sum": 0.0,
            "m_post_norm_count": 0,
            "v_post_norm_sum": 0.0,
            "v_post_norm_count": 0,
            "actual_delta_norm_sum": 0.0,
            "actual_delta_norm_count": 0,
            "rows_seen": 0,
        }
    )

    record_iter = iter_jsonl_records if file_type == "jsonl" else iter_csv_records

    for rank in sorted(rank_files):
        path = rank_files[rank]
        for record in record_iter(path):
            step_value = record.get("global_step")
            try:
                step = int(step_value)
            except (TypeError, ValueError):
                continue

            bucket = stats[step]
            bucket["rows_seen"] += 1

            for metric in METRIC_KEYS:
                val = to_float(record.get(metric))
                if val is None:
                    continue
                bucket[f"{metric}_sum"] += val
                bucket[f"{metric}_count"] += 1

    return stats


def format_loss_for_filename(loss_rate: float) -> str:
    if loss_rate.is_integer():
        return str(int(loss_rate))
    return str(loss_rate).replace(".", "p")


def write_output(output_path: Path, stats: Dict[int, dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "global_step",
                "mean_m_post_norm",
                "mean_v_post_norm",
                "mean_actual_delta_norm",
                "m_post_norm_count",
                "v_post_norm_count",
                "actual_delta_norm_count",
                "rows_seen",
            ]
        )

        for step in sorted(stats):
            bucket = stats[step]

            def mean(metric: str) -> Optional[float]:
                count = bucket[f"{metric}_count"]
                if count == 0:
                    return None
                return bucket[f"{metric}_sum"] / count

            writer.writerow(
                [
                    step,
                    mean("m_post_norm"),
                    mean("v_post_norm"),
                    mean("actual_delta_norm"),
                    bucket["m_post_norm_count"],
                    bucket["v_post_norm_count"],
                    bucket["actual_delta_norm_count"],
                    bucket["rows_seen"],
                ]
            )


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Root path not found: {args.root}")

    folders = discover_folders(args.root, args.file_type)
    if not folders:
        raise RuntimeError(
            f"No rank*.{args.file_type} files found under: {args.root}"
        )

    grouped: Dict[Tuple[int, float], list[FolderInfo]] = defaultdict(list)
    for info in folders:
        bs = infer_batch_size(info)
        grouped[(bs, info.loss_rate)].append(info)

    for (bs, loss_rate), info_list in sorted(grouped.items()):
        rank_files = choose_rank_files(info_list)
        found_ranks = sorted(rank_files)
        if len(found_ranks) != 16:
            message = (
                f"Config bs={bs}, loss={loss_rate}: found {len(found_ranks)} ranks "
                f"({found_ranks})"
            )
            if args.strict_ranks:
                raise RuntimeError(message)
            print(f"WARN: {message}")

        stats = aggregate_files(rank_files, args.file_type)

        loss_label = format_loss_for_filename(loss_rate)
        output_name = f"timeline_bs{bs}_loss{loss_label}.csv"
        output_path = args.output_dir / output_name
        write_output(output_path, stats)
        print(
            f"Wrote {output_path} | bs={bs} loss={loss_rate} "
            f"ranks={len(rank_files)} steps={len(stats)}"
        )


if __name__ == "__main__":
    main()
