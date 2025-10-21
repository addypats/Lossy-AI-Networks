#!/usr/bin/env python3
"""
Compute 'perseverance of loss' metrics from GE episode logs.

Input: a folder containing per-episode *.json files produced by your logger
       (e.g., good_0.json, bad_1.json, ...). Works recursively if requested.

Outputs (JSON + CSV):
- bad_episode_count
- good_episode_count
- total_runtime_sec
- bad_time_sec, good_time_sec, bad_time_fraction
- bad_episode_rate_per_min, bad_episode_rate_per_hour
- mean/median bad duration (sec)
- mean/median good interval (sec)
- p75/p90/p95 for both bad and good durations
- stickiness_index = mean_bad_duration / mean_good_duration
- notes about ordering used (timestamp vs episode_index)

Usage:
  python compute_perseverance.py /path/to/run \
      --recursive \
      --out-dir out_metrics/PIQA/10_nodes/short_1pct \
      --label "TinyLLaMA_PIQA_nodes=10_short_1pct"
"""

import os, json, csv, glob, math, argparse
from statistics import mean, median

# ----------------------- utils -----------------------

def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if p <= 0: return sorted_vals[0]
    if p >= 100: return sorted_vals[-1]
    k = (len(sorted_vals)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def safe_mean(vals):
    return mean(vals) if vals else float("nan")

def safe_median(vals):
    return median(vals) if vals else float("nan")

# ----------------------- loading -----------------------

def load_episodes(folder: str, recursive: bool):
    """
    Return a list of episode dicts with at least:
      - 'state' in {'good','bad'}
      - 'wall_time_sec' (float)
    We also try to read 'start_time_wall', 'end_time_wall', 'episode_index' if available.
    """
    pattern = os.path.join(folder, "**", "*.json") if recursive else os.path.join(folder, "*.json")
    paths = sorted(glob.glob(pattern, recursive=recursive))
    eps = []
    for p in paths:
        try:
            with open(p, "r") as f:
                ep = json.load(f)
            if isinstance(ep, dict) and ep.get("state") in ("good", "bad"):
                # Normalize / ensure numeric
                wt = float(ep.get("wall_time_sec", 0.0))
                ep["wall_time_sec"] = wt
                # Optional fields
                if "start_time_wall" in ep:
                    ep["start_time_wall"] = float(ep["start_time_wall"])
                if "end_time_wall" in ep and ep["end_time_wall"] is not None:
                    ep["end_time_wall"] = float(ep["end_time_wall"])
                eps.append(ep)
        except Exception:
            # ignore non-episode jsons or parse errors
            pass
    if not eps:
        return [], "no_files"

    # Sort preference: start_time_wall -> episode_index -> fallback natural
    if all("start_time_wall" in e for e in eps):
        eps.sort(key=lambda e: e["start_time_wall"])
        order_note = "ordered_by_start_time_wall"
    elif all("episode_index" in e for e in eps):
        eps.sort(key=lambda e: int(e["episode_index"]))
        order_note = "ordered_by_episode_index"
    else:
        # best effort: keep as listed
        order_note = "no_explicit_order_keys_found"

    return eps, order_note

# ----------------------- metrics -----------------------

def compute_perseverance_metrics(episodes):
    total_time = sum(e["wall_time_sec"] for e in episodes)
    bad_eps = [e for e in episodes if e["state"] == "bad"]
    good_eps = [e for e in episodes if e["state"] == "good"]

    bad_durs = [e["wall_time_sec"] for e in bad_eps]
    good_durs = [e["wall_time_sec"] for e in good_eps]

    bad_time = sum(bad_durs)
    good_time = sum(good_durs)
    bad_frac = (bad_time / total_time) if total_time > 0 else float("nan")

    bad_count = len(bad_eps)
    good_count = len(good_eps)

    # Episode rate (how often we enter bad) normalized by time
    bad_rate_per_min  = (bad_count / (total_time / 60.0)) if total_time > 0 else float("nan")
    bad_rate_per_hour = (bad_count / (total_time / 3600.0)) if total_time > 0 else float("nan")

    # Extent summaries for bad (episodes) and good (intervals between bad)
    bad_mean   = safe_mean(bad_durs)
    bad_median = safe_median(bad_durs)
    bad_p75 = percentile(sorted(bad_durs), 75) if bad_durs else float("nan")
    bad_p90 = percentile(sorted(bad_durs), 90) if bad_durs else float("nan")
    bad_p95 = percentile(sorted(bad_durs), 95) if bad_durs else float("nan")

    good_mean   = safe_mean(good_durs)
    good_median = safe_median(good_durs)
    good_p75 = percentile(sorted(good_durs), 75) if good_durs else float("nan")
    good_p90 = percentile(sorted(good_durs), 90) if good_durs else float("nan")
    good_p95 = percentile(sorted(good_durs), 95) if good_durs else float("nan")

    # Stickiness: how long we stay bad vs how long we stay good
    stickiness = (bad_mean / good_mean) if (not math.isnan(bad_mean) and not math.isnan(good_mean) and good_mean > 0) else float("nan")

    return {
        "bad_episode_count": bad_count,
        "good_episode_count": good_count,

        "total_runtime_sec": total_time,
        "bad_time_sec": bad_time,
        "good_time_sec": good_time,
        "bad_time_fraction": bad_frac,

        "bad_episode_rate_per_min": bad_rate_per_min,
        "bad_episode_rate_per_hour": bad_rate_per_hour,

        "bad_mean_sec": bad_mean,
        "bad_median_sec": bad_median,
        "bad_p75_sec": bad_p75,
        "bad_p90_sec": bad_p90,
        "bad_p95_sec": bad_p95,

        "good_mean_sec": good_mean,
        "good_median_sec": good_median,
        "good_p75_sec": good_p75,
        "good_p90_sec": good_p90,
        "good_p95_sec": good_p95,

        "stickiness_index": stickiness,
    }

# ----------------------- save -----------------------

def save_json(obj, path):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_csv_row(dct, path):
    ensure_parent(path)
    # write single-row CSV with stable column order
    keys = list(dct.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerow([dct[k] for k in keys])

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Compute perseverance-of-loss metrics from episode logs.")
    ap.add_argument("folder", help="Folder containing episode JSONs (good_*.json, bad_*.json).")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders recursively.")
    ap.add_argument("--out-dir", default=None, help="Directory to write metrics files. Default: alongside input folder.")
    ap.add_argument("--label", default=None, help="Optional label to include in outputs (e.g., TinyLLaMA_PIQA_nodes=10_short1pct).")
    args = ap.parse_args()

    episodes, order_note = load_episodes(args.folder, args.recursive)
    if not episodes:
        print("No episode JSONs found. Check the folder and try --recursive if needed.")
        return

    metrics = compute_perseverance_metrics(episodes)

    # Attach provenance
    meta = {
        "source_folder": os.path.abspath(args.folder),
        "ordered": order_note,
        "episode_files_count": len(episodes),
        "label": args.label,
    }
    result = {"meta": meta, "metrics": metrics}

    # Decide output paths
    base_name = args.label if args.label else os.path.basename(os.path.abspath(args.folder)).rstrip("/\\")
    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.folder))
    json_path = os.path.join(out_dir, f"{base_name}_perseverance.json")
    csv_path  = os.path.join(out_dir, f"{base_name}_perseverance.csv")

    save_json(result, json_path)

    # Flatten for CSV (meta.* prefixed)
    flat = {f"meta.{k}": v for k,v in meta.items()}
    flat.update(metrics)
    save_csv_row(flat, csv_path)

    print(f"[ok] wrote:\n  {json_path}\n  {csv_path}")
    print(f"Quick summary → bad_episodes={metrics['bad_episode_count']}, "
          f"bad_rate/min={metrics['bad_episode_rate_per_min']:.3f}, "
          f"mean_bad={metrics['bad_mean_sec']:.4f}s, mean_good={metrics['good_mean_sec']:.4f}s, "
          f"stickiness={metrics['stickiness_index']:.3f}, bad_frac={metrics['bad_time_fraction']:.3f}")

if __name__ == "__main__":
    main()

