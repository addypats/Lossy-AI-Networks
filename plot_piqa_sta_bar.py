#!/usr/bin/env python3
"""Create a dataset STA bar chart for selected loss-profile runs.

STA is computed here as the final observed global_step (max global_step)
from the selected optimizer-audit rank logs for each loss-rate configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import matplotlib.pyplot as plt


TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")
RANK_RE = re.compile(r"rank(\d+)\.(jsonl|csv)$")
TARGET_LOSS_RATES = [0.0, 0.2, 0.5, 0.7, 1.0]


@dataclass
class FolderInfo:
    path: Path
    loss_rate: float
    timestamp: str
    rank_files: Dict[int, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build STA bar chart for selected dataset/profile runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("optimizer_audit_logs"),
        help="Optimizer audit root directory.",
    )
    parser.add_argument(
        "--file-type",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Rank file type to parse.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("optimizer_audit_logs") / "plots_by_dataset" / "piqa",
        help="Base output folder for chart outputs.",
    )
    parser.add_argument(
        "--chart-title",
        type=str,
        default="All-Gather loss rates",
        help="Chart title and output subfolder/filename stem.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to prefer when filtering runs.",
    )
    parser.add_argument(
        "--dataset",
        choices=("piqa", "squad"),
        default="piqa",
        help="Dataset run family to select.",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "high_persistence_low_intensity",
            "high_frequency_low_intensity",
            "high_persistence_low_frequency",
        ),
        default="high_persistence_low_intensity",
        help="PIQA loss profile to chart.",
    )
    parser.add_argument(
        "--non-converge-threshold",
        type=int,
        default=600,
        help="If sta_steps is greater than this threshold, mark as non-converged.",
    )
    parser.add_argument(
        "--non-converge-plot-value",
        type=int,
        default=800,
        help="Bar height to use for non-converged runs.",
    )
    return parser.parse_args()


def parse_loss_rate(folder_name: str, profile: str) -> float:
    if "loss-rate0" in folder_name:
        return 0.0

    marker_with_intensity = f"loss-rate_{profile}_1_"
    idx = folder_name.find(marker_with_intensity)
    if idx != -1:
        tail = folder_name[idx + len(marker_with_intensity) :]
        token = tail.split("_", 1)[0]
        if re.fullmatch(r"\d{8}-\d{6}", token):
            return 1.0
        return float(token)

    marker = f"loss-rate_{profile}_"
    idx = folder_name.find(marker)
    if idx == -1:
        raise ValueError(f"Not a {profile} folder")
    tail = folder_name[idx + len(marker) :]
    token = tail.split("_", 1)[0]
    if re.fullmatch(r"\d{8}-\d{6}", token):
        return 1.0
    return float(token)


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


def discover_dataset_folders(
    root: Path,
    file_type: str,
    batch_size: int,
    profile: str,
    dataset: str,
) -> list[FolderInfo]:
    infos: list[FolderInfo] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if dataset not in name:
            continue
        if profile not in name and "loss-rate0" not in name:
            continue
        # Keep only the requested batch size family (loss-rate0 folders may omit bs token).
        if "loss-rate0" not in name and f"_bs{batch_size}_" not in name:
            continue

        rank_files = collect_rank_files(folder, file_type)
        if not rank_files:
            continue

        try:
            loss_rate = parse_loss_rate(folder.name, profile)
        except ValueError:
            continue

        infos.append(
            FolderInfo(
                path=folder,
                loss_rate=loss_rate,
                timestamp=parse_timestamp(folder.name),
                rank_files=rank_files,
            )
        )
    return infos


def choose_rank_files(folders: Iterable[FolderInfo]) -> Dict[int, Path]:
    chosen: Dict[int, Tuple[str, Path]] = {}
    for info in folders:
        for rank, path in info.rank_files.items():
            prev = chosen.get(rank)
            if prev is None or info.timestamp >= prev[0]:
                chosen[rank] = (info.timestamp, path)
    return {rank: item[1] for rank, item in chosen.items()}


def iter_global_steps(path: Path, file_type: str) -> Iterator[int]:
    if file_type == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    yield int(record.get("global_step"))
                except Exception:
                    continue
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    yield int(row.get("global_step"))
                except Exception:
                    continue


def final_step_for_config(rank_files: Dict[int, Path], file_type: str) -> Optional[int]:
    max_step: Optional[int] = None
    for _, path in sorted(rank_files.items()):
        for step in iter_global_steps(path, file_type):
            if max_step is None or step > max_step:
                max_step = step
    return max_step


def plot_sta(
    loss_rates: list[float],
    steps: list[int],
    output_path: Path,
    title: str,
    non_converged: list[bool],
) -> None:
    plt.figure(figsize=(10, 6))
    labels = [f"{lr:g}" for lr in loss_rates]
    bars = plt.bar(labels, steps, width=0.65)

    ymax = max(steps) if steps else 0
    text_y = ymax * 0.95 if ymax > 0 else 0.0
    for i, is_non_converged in enumerate(non_converged):
        if not is_non_converged:
            continue
        bars[i].set_color("#d62728")
        plt.text(
            i,
            text_y,
            "Did not converge on it",
            color="red",
            fontsize=10,
            rotation=90,
            ha="center",
            va="top",
        )

    plt.title(title)
    plt.xlabel("loss_rate")
    plt.ylabel("STA (steps to accuracy)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_sta_table(
    loss_rates: list[float],
    raw_steps: list[int],
    plotted_steps: list[int],
    non_converged: list[bool],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss_rate", "sta_steps_raw", "sta_steps_plotted", "did_not_converge"])
        for loss_rate, raw, plotted, non_conv in zip(loss_rates, raw_steps, plotted_steps, non_converged):
            writer.writerow([loss_rate, raw, plotted, int(non_conv)])


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Root directory not found: {args.root}")

    folders = discover_dataset_folders(
        args.root,
        args.file_type,
        args.batch_size,
        args.profile,
        args.dataset,
    )
    if not folders:
        raise RuntimeError(
            f"No matching {args.dataset.upper()} runs found for profile: {args.profile}"
        )

    grouped: Dict[float, list[FolderInfo]] = {}
    for loss_rate in TARGET_LOSS_RATES:
        grouped[loss_rate] = [f for f in folders if abs(f.loss_rate - loss_rate) < 1e-9]

    sta_by_loss: Dict[float, int] = {}
    for loss_rate in TARGET_LOSS_RATES:
        info_list = grouped[loss_rate]
        if not info_list:
            raise RuntimeError(
                f"Missing {args.dataset.upper()} run for loss rate {loss_rate:g}"
            )
        rank_files = choose_rank_files(info_list)
        final_step = final_step_for_config(rank_files, args.file_type)
        if final_step is None:
            raise RuntimeError(f"No global_step data found for loss rate {loss_rate:g}")
        sta_by_loss[loss_rate] = final_step

    chart_folder = args.output_root / args.chart_title
    chart_path = chart_folder / f"{args.chart_title}.png"
    csv_path = chart_folder / f"{args.chart_title}.csv"

    ordered_loss = TARGET_LOSS_RATES
    raw_steps = [sta_by_loss[lr] for lr in ordered_loss]
    non_converged = [step > args.non_converge_threshold for step in raw_steps]
    plotted_steps = [args.non_converge_plot_value if nc else step for step, nc in zip(raw_steps, non_converged)]

    plot_sta(ordered_loss, plotted_steps, chart_path, args.chart_title, non_converged)
    write_sta_table(ordered_loss, raw_steps, plotted_steps, non_converged, csv_path)

    print(f"Saved chart: {chart_path}")
    print(f"Saved data:  {csv_path}")
    for lr, raw, plotted, nc in zip(ordered_loss, raw_steps, plotted_steps, non_converged):
        suffix = " (Did not converge on it)" if nc else ""
        print(f"loss={lr:g} -> sta_steps_raw={raw} sta_steps_plotted={plotted}{suffix}")


if __name__ == "__main__":
    main()
