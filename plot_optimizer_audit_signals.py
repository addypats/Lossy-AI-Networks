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
        "--full-run",
        action="store_true",
        help="Plot the full available run duration (ignores step-min/step-max slicing).",
    )
    parser.add_argument(
        "--filename-suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filenames (for example: _full_run).",
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
    parser.add_argument(
        "--include-keyword",
        type=str,
        default="",
        help="Only include run folders whose names contain this keyword (case-insensitive).",
    )
    parser.add_argument(
        "--force-batch-size",
        type=int,
        default=None,
        help="Force this batch size when a folder name does not include bs<number>.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional dataset folder name (for example: squad, piqa). If set, plots are written under <output-dir>/<dataset-name>/.",
    )
    parser.add_argument(
        "--also-include-loss0",
        action="store_true",
        help="Include loss-rate0 folders in addition to include-keyword matches.",
    )
    parser.add_argument(
        "--loss0-dataset-keyword",
        type=str,
        default="",
        help="When --also-include-loss0 is set, require this dataset keyword in loss-rate0 folder names.",
    )
    parser.add_argument(
        "--only-batch-size",
        type=int,
        default=None,
        help="If set, skip folders whose explicit bs token does not match this value.",
    )
    return parser.parse_args()


def parse_batch_size(folder_name: str) -> Optional[int]:
    match = re.search(r"_bs(\d+)_", folder_name)
    return int(match.group(1)) if match else None


def parse_loss_rate(folder_name: str) -> float:
    if "loss-rate0" in folder_name:
        return 0.0

    # Handles folder names like:
    # - ...loss-rate_high_persistence_low_intensity_1_0.2_<timestamp>...
    # - ...loss-rate_high_frequency_low_intensity_0.5_<timestamp>...
    # - ...loss-rate_high_persistence_low_frequency_1_<timestamp>...
    match = re.search(r"loss-rate_.*_(\d+(?:\.\d+)?)_\d{8}-\d{6}", folder_name)
    if not match:
        raise ValueError(f"Could not parse loss rate from folder: {folder_name}")
    return float(match.group(1))


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


def discover_folders(
    root: Path,
    file_type: str,
    include_keyword: str = "",
    also_include_loss0: bool = False,
    loss0_dataset_keyword: str = "",
    only_batch_size: Optional[int] = None,
) -> list[FolderInfo]:
    infos: list[FolderInfo] = []
    keyword = include_keyword.lower().strip()
    loss0_keyword = loss0_dataset_keyword.lower().strip()
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        name_l = folder.name.lower()
        is_loss0 = "loss-rate0" in name_l

        matches_profile_keyword = (not keyword) or (keyword in name_l)
        include_folder = matches_profile_keyword
        if also_include_loss0 and is_loss0:
            include_folder = True
            if loss0_keyword and loss0_keyword not in name_l:
                include_folder = False

        if not include_folder:
            continue
        parsed_bs = parse_batch_size(folder.name)
        if only_batch_size is not None and parsed_bs is not None and parsed_bs != int(only_batch_size):
            continue

        rank_files = collect_rank_files(folder, file_type)
        if not rank_files:
            continue
        try:
            loss_rate = parse_loss_rate(folder.name)
        except ValueError:
            # Skip folders that do not match supported naming schemes.
            continue
        infos.append(
            FolderInfo(
                path=folder,
                batch_size=parsed_bs,
                loss_rate=loss_rate,
                timestamp=parse_timestamp(folder.name),
                rank_files=rank_files,
            )
        )
    return infos


def infer_batch_size(info: FolderInfo, force_batch_size: Optional[int] = None) -> int:
    if info.batch_size is not None:
        return info.batch_size
    if force_batch_size is not None:
        return int(force_batch_size)
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
            "sum_m_post": 0.0,
            "cnt_m_post": 0,
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

            m_post = to_float(record.get("m_post_norm"))
            actual_delta = to_float(record.get("actual_delta_norm"))
            v_post = to_float(record.get("v_post_norm"))

            eff_lr = to_float(record.get("effective_lr_proxy"))
            if eff_lr is None:
                manual_delta = to_float(record.get("manual_delta_norm"))
                eff_lr = safe_ratio(actual_delta, manual_delta)

            if m_post is not None:
                bucket["sum_m_post"] += m_post
                bucket["cnt_m_post"] += 1

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
                "mean_m_post_norm": mean("sum_m_post", "cnt_m_post"),
                "mean_actual_delta_norm": mean("sum_actual", "cnt_actual"),
                "mean_v_post_norm": mean("sum_v_post", "cnt_v_post"),
                "mean_effective_lr_proxy": mean("sum_eff_lr", "cnt_eff_lr"),
            }
        )

    return pd.DataFrame(rows)


def build_config_tables(
    root: Path,
    file_type: str,
    strict_ranks: bool,
    include_keyword: str = "",
    force_batch_size: Optional[int] = None,
    also_include_loss0: bool = False,
    loss0_dataset_keyword: str = "",
    only_batch_size: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    folders = discover_folders(
        root,
        file_type,
        include_keyword=include_keyword,
        also_include_loss0=also_include_loss0,
        loss0_dataset_keyword=loss0_dataset_keyword,
        only_batch_size=only_batch_size,
    )
    if not folders:
        raise RuntimeError(f"No rank*.{file_type} files found under {root}")

    grouped: Dict[Tuple[int, float], list[FolderInfo]] = defaultdict(list)
    for info in folders:
        grouped[(infer_batch_size(info, force_batch_size=force_batch_size), info.loss_rate)].append(info)

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
    full_run: bool = False,
) -> None:
    plt.figure(figsize=(12, 7))

    for label, table in config_tables.items():
        if table.empty:
            continue
        if full_run:
            sliced = table.dropna(subset=["global_step", column])
        else:
            sliced = table[(table["global_step"] >= step_min) & (table["global_step"] <= step_max)]
        if sliced.empty:
            continue
        plt.plot(sliced["global_step"], sliced[column], linewidth=1.7, label=label)

    plt.title(title)
    plt.xlabel("global_step")
    plt.ylabel(y_label)
    if not full_run:
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
    dataset_name = args.dataset_name.strip()
    effective_output_dir = args.output_dir / dataset_name if dataset_name else args.output_dir

    config_tables = build_config_tables(
        args.root,
        args.file_type,
        args.strict_ranks,
        include_keyword=args.include_keyword,
        force_batch_size=args.force_batch_size,
        also_include_loss0=args.also_include_loss0,
        loss0_dataset_keyword=args.loss0_dataset_keyword,
        only_batch_size=args.only_batch_size,
    )

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

        bs_output_dir = effective_output_dir / f"bs{batch_size}"
        suffix = args.filename_suffix
        inertia_path = bs_output_dir / f"inertia_mean_actual_delta_norm{suffix}.png"
        deflation_path = bs_output_dir / f"deflation_mean_v_post_norm{suffix}.png"
        amends_path = bs_output_dir / f"amends_mean_effective_lr_proxy{suffix}.png"

        make_plot(
            config_tables=tables_for_bs,
            column="mean_actual_delta_norm",
            title=f"Inertia Plot (Weight Update Persistence) - bs{batch_size}",
            y_label="mean_actual_delta_norm",
            step_min=args.step_min,
            step_max=args.step_max,
            output_path=inertia_path,
            full_run=args.full_run,
        )
        make_plot(
            config_tables=tables_for_bs,
            column="mean_v_post_norm",
            title=f"Deflation Plot (Denominator Decay) - bs{batch_size}",
            y_label="mean_v_post_norm",
            step_min=args.step_min,
            step_max=args.step_max,
            output_path=deflation_path,
            full_run=args.full_run,
        )
        make_plot(
            config_tables=tables_for_bs,
            column="mean_effective_lr_proxy",
            title=f"Amends Spike (Effective LR Proxy) - bs{batch_size}",
            y_label="mean_effective_lr_proxy",
            step_min=args.step_min,
            step_max=args.step_max,
            output_path=amends_path,
            full_run=args.full_run,
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
