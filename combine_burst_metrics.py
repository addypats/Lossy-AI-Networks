#!/usr/bin/env python3
import os, json, csv, argparse, glob, math, re

# ---------- I/O helpers ----------
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            pass
        return
    header = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def find_perseverance_jsons(root):
    root = os.path.abspath(os.path.expanduser(root))
    rec = sorted(glob.glob(os.path.join(root, "**", "*_perseverance.json"), recursive=True))
    if rec:
        return rec
    flat = sorted(glob.glob(os.path.join(root, "*_perseverance.json")))
    return flat

# ---------- robust numeric ----------
def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

# ---------- parse label: node_count, loss_pattern_root (short/long), loss_rate ----------
LABEL_RE = re.compile(
    r"""
    _n(?P<node>\d+)_            # _n8_
    (?P<pattern>short|long)     # short or long
    (?P<rate>[^_]+)?            # trailing rate token like 0p1 / 0p5 / 1 / 1p0 (optional)
    """,
    re.VERBOSE | re.IGNORECASE
)

def parse_label_bits(label: str):
    node_count = math.nan
    pattern_root = ""
    rate_str = ""
    rate_float = math.nan

    m = LABEL_RE.search(label)
    if m:
        try:
            node_count = int(m.group("node"))
        except Exception:
            pass
        pattern_root = (m.group("pattern") or "").lower()
        rate_str = (m.group("rate") or "").strip().lower()

        # normalize "0p1" -> 0.1, "0p5" -> 0.5, "1" -> 1.0, "1p0" -> 1.0
        if rate_str:
            # strip non-alphanum except p and .
            token = re.sub(r"[^0-9p.]", "", rate_str)
            if "p" in token:
                token = token.replace("p", ".")
            try:
                rate_float = float(token)
            except Exception:
                rate_float = math.nan

    return node_count, pattern_root, rate_str, rate_float

# ---------- Burst Loss Rate from per-episode JSONs ----------
PACKETS_KEYS   = ("packets", "packets_sent", "pkts_sent", "sent")
DROPPED_KEYS   = ("dropped", "packets_lost", "pkts_lost", "lost")

def extract_numeric(dct, keys):
    for k in keys:
        if k in dct:
            try:
                return float(dct[k])
            except Exception:
                pass
    return None

def compute_burst_loss_rate_from_episodes(source_folder):
    """
    Walk source_folder recursively, sum dropped/packets for state=='bad' episodes.
    Returns float('nan') if nothing usable is found.
    """
    if not source_folder or not os.path.exists(source_folder):
        return float("nan")

    total_sent_burst = 0.0
    total_lost_burst = 0.0
    for path in glob.glob(os.path.join(source_folder, "**", "*.json"), recursive=True):
        try:
            ep = read_json(path)
        except Exception:
            continue
        if not isinstance(ep, dict):
            continue
        state = str(ep.get("state", "")).lower()
        if state != "bad":
            continue

        sent = extract_numeric(ep, PACKETS_KEYS)
        lost = extract_numeric(ep, DROPPED_KEYS)
        if sent is None or lost is None:
            continue

        total_sent_burst += sent
        total_lost_burst += lost

    if total_sent_burst <= 0:
        return float("nan")
    return total_lost_burst / total_sent_burst

# ---------- bucketing ----------
def compute_terciles(vals):
    xs = sorted([v for v in vals if not math.isnan(v)])
    if not xs:
        return (float("nan"), float("nan"))
    n = len(xs)
    idx = lambda p: max(0, min(n-1, int(round(p*(n-1)))))
    return xs[idx(1/3)], xs[idx(2/3)]

def bucket_value(x, low_hi, med_hi):
    if math.isnan(x):
        return "NA"
    if x <= low_hi: return "Low"
    if x <= med_hi: return "Medium"
    return "High"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Combine 5 burst metrics (numeric + buckets) and produce tidy bucket-counts for plotting."
    )
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Root directories to scan for *_perseverance.json (e.g., Perseverance_logs/)")
    ap.add_argument("--out", default="combined/combined_metrics_v2",
                    help="Output base path (no extension). Writes combined row-per-run CSV/JSON + tidy bucket counts CSV")
    args = ap.parse_args()

    # Gather all perseverance JSONs
    files = []
    for r in args.roots:
        files += find_perseverance_jsons(r)
    if not files:
        print("No *_perseverance.json found under:", args.roots)
        return

    # Collect row-per-run data
    rows = []
    for fp in files:
        try:
            data = read_json(fp)
        except Exception:
            continue

        meta    = data.get("meta", {})
        metrics = data.get("metrics", {})
        label   = meta.get("label") or os.path.splitext(os.path.basename(fp))[0].replace("_perseverance", "")
        src     = meta.get("source_folder", "")

        node_count, pattern_root, rate_str, rate_float = parse_label_bits(label)

        # 4 time-structure metrics
        bad_time_fraction = to_float(metrics.get("bad_time_fraction"))   # Existence
        bad_mean_sec      = to_float(metrics.get("bad_mean_sec"))        # Perseverance
        bad_episode_count = to_float(metrics.get("bad_episode_count"))   # Extent (frequency)
        good_mean_sec     = to_float(metrics.get("good_mean_sec"))       # Extent (recovery)

        # 5th: Burst Loss Rate (intensity) from bad episode JSONs in source_folder
        burst_loss_rate = compute_burst_loss_rate_from_episodes(src)

        rows.append({
            "label": label,
            "source_file": fp,
            "source_folder": src,
            "node_count": node_count,
            "loss_pattern_root": pattern_root,   # "short" / "long"
            "loss_rate_token": rate_str,         # e.g. "0p1"
            "loss_rate": rate_float,             # e.g. 0.1

            # numeric metrics
            "bad_time_fraction": bad_time_fraction,
            "bad_mean_sec": bad_mean_sec,
            "bad_episode_count": bad_episode_count,
            "good_mean_sec": good_mean_sec,
            "burst_loss_rate": burst_loss_rate,
        })

    # Compute tercile cut-points per metric and assign buckets
    metric_keys = ["bad_time_fraction","bad_mean_sec","bad_episode_count","good_mean_sec","burst_loss_rate"]
    cuts = {}
    for k in metric_keys:
        vals = [r[k] for r in rows]
        cuts[k] = compute_terciles(vals)

    for r in rows:
        r["existence_bucket"]       = bucket_value(r["bad_time_fraction"], *cuts["bad_time_fraction"])
        r["perseverance_bucket"]    = bucket_value(r["bad_mean_sec"], *cuts["bad_mean_sec"])
        r["episode_count_bucket"]   = bucket_value(r["bad_episode_count"], *cuts["bad_episode_count"])
        r["gap_duration_bucket"]    = bucket_value(r["good_mean_sec"], *cuts["good_mean_sec"])
        r["burst_loss_rate_bucket"] = bucket_value(r["burst_loss_rate"], *cuts["burst_loss_rate"])

    # Save the combined (row-per-run) outputs with numeric + buckets + parsed fields
    base = args.out
    write_csv(rows, base + ".csv")
    write_json({"cuts": cuts, "rows": rows}, base + ".json")

    # ---- Build tidy bucket-counts table for your bar charts ----
    # Long/Short split × bucket (Low/Med/High) × node_count, for EACH METRIC.
    # We'll produce a single tidy CSV with columns:
    # metric, loss_pattern_root, bucket, node_count, count
    tidy = []
    # map metrics -> bucket column names
    metric_to_bucket = {
        "bad_time_fraction": "existence_bucket",
        "bad_mean_sec": "perseverance_bucket",
        "bad_episode_count": "episode_count_bucket",
        "good_mean_sec": "gap_duration_bucket",
        "burst_loss_rate": "burst_loss_rate_bucket",
    }
    for metric, bucket_col in metric_to_bucket.items():
        # build counts per (loss_pattern_root, bucket, node_count)
        counts = {}
        for r in rows:
            pat = r.get("loss_pattern_root") or "unknown"
            bkt = r.get(bucket_col) or "NA"
            n   = r.get("node_count")
            if n != n:   # NaN check
                continue
            key = (pat, bkt, int(n))
            counts[key] = counts.get(key, 0) + 1

        # emit rows
        for (pat, bkt, n), c in sorted(counts.items()):
            tidy.append({
                "metric": metric,
                "loss_pattern_root": pat,
                "bucket": bkt,             # Low / Medium / High / NA
                "node_count": n,
                "count": c,
            })

    write_csv(tidy, base + "_bucket_counts_tidy.csv")

    print(f"[ok] wrote:\n  {base}.csv\n  {base}.json\n  {base}_bucket_counts_tidy.csv")
    print("Cut points (low_hi, med_hi):")
    for k,(a,b) in cuts.items():
        try:
            print(f"  {k}: {a:.6f}, {b:.6f}")
        except Exception:
            print(f"  {k}: {a}, {b}")

if __name__ == "__main__":
    main()

