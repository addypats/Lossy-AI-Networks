#!/usr/bin/env python3
"""
extract_loss_lengths.py

Extract per-episode loss lengths from `bad_*.json` files into a CSV.

Adapts to your schema:
  - Duration from `wall_time_sec` (preferred), else (end - start)
  - Start/End from `start_time_wall` / `end_time_wall`
  - EpisodeID from `episode_index` (preferred), else from filename

Output columns:
  Nodes,LossType,LossRate,Seed,EpisodeID,Duration_sec,Start,End

Usage:
  python extract_loss_lengths.py \
    --root /home/ubuntu/Lossy-AI-Networks/State_Times \
    --out  /home/ubuntu/Lossy-AI-Networks/analysis/loss_lengths.csv \
    --stats /home/ubuntu/Lossy-AI-Networks/analysis/loss_length_stats.csv  # optional
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd


# Robust mapping from folder tokens to rates
LOSS_RATE_PATTERNS = [
    (re.compile(r"(point[\-_]?1|0p?1|\b0\.1\b|\b0_1\b)"), 0.1),
    (re.compile(r"(half|0p?5|\b0\.5\b|\b0_5\b)"), 0.5),
    (re.compile(r"(1percent|1_percent|_1percent|_1_percent|one[_\-]?pre?cent|\b1\b|^1$)"), 1.0),
]


def parse_nodes(segment: str) -> Optional[int]:
    m = re.match(r"nodes=(\d+)", segment)
    return int(m.group(1)) if m else None


def parse_seed(segment: str) -> Optional[int]:
    m = re.match(r"seed=(\d+)", segment)
    return int(m.group(1)) if m else None


def parse_loss(segment: str) -> Tuple[Optional[str], Optional[float]]:
    """
    segment example: 'loss=short_half_percent', 'loss=one_precent', 'loss=long_point1_percent'
    """
    m = re.match(r"loss=([A-Za-z0-9_\-\.]+)", segment)
    if not m:
        return None, None
    token = m.group(1).lower()

    loss_type = "Short" if "short" in token else "Long"

    rate = None
    for pat, val in LOSS_RATE_PATTERNS:
        if pat.search(token):
            rate = val
            break
    # final fallbacks
    if rate is None:
        if "0.1" in token or "0_1" in token:
            rate = 0.1
        elif "0.5" in token or "0_5" in token:
            rate = 0.5
        elif token.endswith("1") or token.endswith("1p") or token.endswith("1pc"):
            rate = 1.0
    return loss_type, rate


def coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def get_duration_start_end(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (duration_sec, start, end) using wall-clock keys first,
    with safe fallbacks.
    """
    # Preferred: wall-clock keys in your schema
    if "wall_time_sec" in payload:
        dur = coerce_float(payload["wall_time_sec"])
    else:
        dur = None

    start = None
    end = None

    # Prefer wall-clock start/end first
    start_keys = ("start_time_wall", "start_time_sec", "start_sec", "start_time")
    end_keys   = ("end_time_wall",   "end_time_sec",   "end_sec",   "end_time")

    start = next((coerce_float(payload[k]) for k in start_keys if k in payload), None)
    end   = next((coerce_float(payload[k]) for k in end_keys   if k in payload), None)

    # If duration missing but start/end present, compute it
    if dur is None and start is not None and end is not None:
        dur = max(0.0, end - start)

    return dur, start, end


def episode_id_from(payload: Dict[str, Any], path: Path) -> str:
    if "episode_index" in payload:
        try:
            return str(int(payload["episode_index"]))
        except Exception:
            pass
    # fallback to filename stem (strip 'bad_' if present)
    stem = path.stem  # e.g., 'bad_1977'
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else stem


def collect_lengths(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("bad_*.json"):
        parts = p.parts
        nodes = None
        loss_type = None
        loss_rate = None
        seed = None

        for seg in parts:
            if nodes is None:
                nodes = parse_nodes(seg) or nodes
            if loss_type is None or loss_rate is None:
                lt, lr = parse_loss(seg)
                loss_type = lt or loss_type
                loss_rate = lr if lr is not None else loss_rate
            if seed is None:
                seed = parse_seed(seg) or seed

        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue

        episode_id = episode_id_from(payload, p)
        dur, start, end = get_duration_start_end(payload)

        rows.append({
            "Nodes": nodes,
            "LossType": loss_type,
            "LossRate": loss_rate,
            "Seed": seed,
            "EpisodeID": episode_id,
            "Duration_sec": dur,
            "Start": start,
            "End": end,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Nodes"] = pd.to_numeric(df["Nodes"], errors="coerce").astype("Int64")
        df["LossRate"] = pd.to_numeric(df["LossRate"], errors="coerce")
        df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce").astype("Int64")
        # Sort for sanity
        df = df.sort_values(["Nodes", "LossType", "LossRate", "Seed", "EpisodeID"], na_position="last")
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    def agg(grp: pd.DataFrame) -> pd.Series:
        x = grp["Duration_sec"].dropna().astype(float)
        n = len(x)
        if n == 0:
            return pd.Series({
                "n": 0, "mean": math.nan, "median": math.nan, "p75": math.nan, "p90": math.nan,
                "p95": math.nan, "p99": math.nan, "std": math.nan, "cv": math.nan,
                "max": math.nan, "sum": math.nan, "top10_share": math.nan
            })
        x_sorted = x.sort_values().values
        q = lambda v: float(pd.Series(x_sorted).quantile(v))
        std = float(x.std(ddof=1)) if n > 1 else 0.0
        mean = float(x.mean())
        cv = std / mean if mean != 0 else math.nan
        k = max(1, int(0.1 * n))  # top 10%
        top10 = x_sorted[-k:]
        top10_share = float(top10.sum() / x.sum()) if x.sum() > 0 else math.nan
        return pd.Series({
            "n": n,
            "mean": mean,
            "median": float(x.median()),
            "p75": q(0.75),
            "p90": q(0.90),
            "p95": q(0.95),
            "p99": q(0.99),
            "std": std,
            "cv": cv,
            "max": float(x.max()),
            "sum": float(x.sum()),
            "top10_share": top10_share
        })

    stats = (
        df.groupby(["Nodes", "LossType", "LossRate"], dropna=False)
          .apply(agg)
          .reset_index()
          .sort_values(["Nodes", "LossType", "LossRate"])
          .reset_index(drop=True)
    )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory containing run folders")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path for per-episode lengths")
    ap.add_argument("--stats", type=str, default=None, help="Optional output CSV path for grouped statistics")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    df = collect_lengths(root)
    if df.empty:
        print("No bad_*.json files found or no durations parsed.")
        return

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote per-episode lengths to: {out_path} ({len(df)} rows)")

    if args.stats:
        stats = compute_stats(df)
        stats_path = Path(args.stats).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(stats_path, index=False)
        print(f"Wrote grouped stats to: {stats_path}")


if __name__ == "__main__":
    main()

