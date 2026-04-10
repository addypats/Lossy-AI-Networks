#!/usr/bin/env python3
"""Visualize optimizer-audit timeline signals for all bs/loss configurations.

Creates three plots (zoomed to a global_step range):
1) Inertia: mean_actual_delta_norm
2) Deflation: mean_v_post_norm
3) Amends Spike: mean_effective_lr_proxy (fallback: actual/manual delta ratio)
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

import matplotlib.pyplot as plt
import pandas as pd


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
        description="Generate optimizer-audit timeline plots from rank files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("optimizer_audit_logs"),
        help="Root directory that contains audit run folders.",
    )
    parser.add_argument(
        "--file-type",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Input rank file type to parse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimizer_audit_logs") / "plots",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--step-min",
        type=int,
        default=80,
        help="Lower bound for global_step zoom.",
    )
    parser.add_argument(
        "--step-max",
        type=int,
        default=130,
        help="Upper bound for global_step zoom.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
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
        rank_files[int(match.group(1))] = child
    return rank_files


def discover_folders(root: Path, file_type: str) -> list[FolderInfo]:
    infos: list[FolderInfo] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        rank_files = collect_rank_files(folder, file_type)
        if not rank_files:
            continue
        infos.append(
            FolderInfo(
                path=folder,
                batch_size=parse_batch_size(folder.name),
                loss_rate=parse_loss_rate(folder.name),
                timestamp=parse_timestamp(folder.name),
                rank_files=rank_files,
            )
        )
    return infos


def infer_batch_size(info: FolderInfo) -> int:
    if info.batch_size is not None:
        return info.batch_size
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
            if line:
                yield json.loads(line)


def iter_csv_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def aggregate_config(rank_files: Dict[int, Path], file_type: str) -> pd.DataFrame:
    stats = defaultdict(
        lambda: {
            "sum_actual": 0.0,
            "cnt_actual": 0,
            "sum_v_post": 0.0,
            "cnt_v_post": 0,
            "sum_eff_lr": 0.0,
            "cnt_eff_lr": 0,
        }
    )

    record_iter = iter_jsonl_records if file_type == "jsonl" else iter_csv_records

    for rank in sorted(rank_files):
        path = rank_files[rank]
        for record in record_iter(path):
            step_raw = record.get("global_step")
            try:
                step = int(step_raw)
            except (TypeError, ValueError):
                continue

            bucket = stats[step]

            actual_delta = to_float(record.get("actual_delta_norm"))
            v_post = to_float(record.get("v_post_norm"))

            eff_lr = to_float(record.get("effective_lr_proxy"))
            if eff_lr is None:
                manual_delta = to_float(record.get("manual_delta_norm"))
                eff_lr = safe_ratio(actual_delta, manual_delta)

            if actual_delta is not None:
                bucket["sum_actual"] += actual_delta
                bucket["cnt_actual"] += 1

            if v_post is not None:
                bucket["sum_v_post"] += v_post
                bucket["cnt_v_post"] += 1

            if eff_lr is not None:
                bucket["sum_eff_lr"] += eff_lr
                bucket["cnt_eff_lr"] += 1

    rows = []
    for step in sorted(stats):
        bucket = stats[step]

        def mean(sum_key: str, cnt_key: str) -> Optional[float]:
            count = bucket[cnt_key]
            return None if count == 0 else bucket[sum_key] / count

        rows.append(
            {
                "global_step": step,
                "mean_actual_delta_norm": mean("sum_actual", "cnt_actual"),
                "mean_v_post_norm": mean("sum_v_post", "cnt_v_post"),
                "mean_effective_lr_proxy": mean("sum_eff_lr", "cnt_eff_lr"),
            }
        )

    return pd.DataFrame(rows)


def build_config_tables(root: Path, file_type: str, strict_ranks: bool) -> Dict[str, pd.DataFrame]:
    folders = discover_folders(root, file_type)
    if not folders:
        raise RuntimeError(f"No rank*.{file_type} files found under {root}")

    grouped: Dict[Tuple[int, float], list[FolderInfo]] = defaultdict(list)
    for info in folders:
        grouped[(infer_batch_size(info), info.loss_rate)].append(info)

    config_tables: Dict[str, pd.DataFrame] = {}
    for (bs, loss), info_list in sorted(grouped.items()):
        rank_files = choose_rank_files(info_list)
        found_ranks = sorted(rank_files)
        if len(found_ranks) != 16:
            message = (
                f"Config bs={bs}, loss={loss}: found {len(found_ranks)} ranks ({found_ranks})"
            )
            if strict_ranks:
                raise RuntimeError(message)
            print(f"WARN: {message}")

        label = f"bs{bs}_loss{loss:g}"
        table = aggregate_config(rank_files, file_type)
        config_tables[label] = table
        print(f"Loaded {label}: ranks={len(rank_files)} steps={len(table)}")

    return config_tables


def make_plot(
    config_tables: Dict[str, pd.DataFrame],
    column: str,
    title: str,
    y_label: str,
    step_min: int,
    step_max: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 7))

    for label, table in config_tables.items():
        if table.empty:
            continue
        sliced = table[(table["global_step"] >= step_min) & (table["global_step"] <= step_max)]
        if sliced.empty:
            continue
        plt.plot(sliced["global_step"], sliced[column], linewidth=1.7, label=label)

    plt.title(title)
    plt.xlabel("global_step")
    plt.ylabel(y_label)
    plt.xlim(step_min, step_max)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved {output_path}")


def filter_tables_for_batch(config_tables: Dict[str, pd.DataFrame], batch_size: int) -> Dict[str, pd.DataFrame]:
    prefix = f"bs{batch_size}_"
    filtered: Dict[str, pd.DataFrame] = {}
    for label, table in config_tables.items():
        if not label.startswith(prefix):
            continue
        # Keep legends compact inside each batch-size figure.
        loss_label = label.replace(prefix, "")
        filtered[loss_label] = table
    return filtered


def main() -> None:
    args = parse_args()
    config_tables = build_config_tables(args.root, args.file_type, args.strict_ranks)

    batch_sizes = sorted(
        {
            int(label.split("_", 1)[0].replace("bs", ""))
            for label in config_tables
            if label.startswith("bs") and "_" in label
        }
    )

    saved_images: list[Path] = []

    for batch_size in batch_sizes:
        tables_for_bs = filter_tables_for_batch(config_tables, batch_size)
        if not tables_for_bs:
            continue

        bs_output_dir = args.output_dir / f"bs{batch_size}"
        inertia_path = bs_output_dir / "inertia_mean_actual_delta_norm.png"
        deflation_path = bs_output_dir / "deflation_mean_v_post_norm.png"
        amends_path = bs_output_dir / "amends_mean_effective_lr_proxy.png"

        make_plot(
            config_tables=tables_for_bs,
            column="mean_actual_delta_norm",
            title=f"Inertia Plot (Weight Update Persistence) - bs{batch_size}",
            y_label="mean_actual_delta_norm",
            step_min=args.step_min,
            step_max=args.step_max,
            output_path=inertia_path,
        )
        make_plot(
            config_tables=tables_for_bs,
            column="mean_v_post_norm",
            title=f"Deflation Plot (Denominator Decay) - bs{batch_size}",
            y_label="mean_v_post_norm",
            step_min=args.step_min,
            step_max=args.step_max,
            output_path=deflation_path,
        )
        make_plot(
            config_tables=tables_for_bs,
            column="mean_effective_lr_proxy",
            title=f"Amends Spike (Effective LR Proxy) - bs{batch_size}",
            y_label="mean_effective_lr_proxy",
            step_min=args.step_min,
            step_max=args.step_max,
            output_path=amends_path,
        )

        saved_images.extend([inertia_path, deflation_path, amends_path])

    if args.show and saved_images:
        for image in saved_images:
            img = plt.imread(image)
            plt.figure(figsize=(12, 7))
            plt.imshow(img)
            plt.axis("off")
            plt.title(image.stem)
        plt.show()


if __name__ == "__main__":
    main()
