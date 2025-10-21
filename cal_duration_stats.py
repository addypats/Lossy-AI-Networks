#!/usr/bin/env python3
import os, json, csv, glob, statistics as stats

def load_durations(path):
    ext = os.path.splitext(path)[1].lower()
    vals = []
    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        vals = data if isinstance(data, list) else data.get("wall_time_sec", [])
    elif ext == ".csv":
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            col_idx = 0
            if header and "wall_time_sec" in header:
                col_idx = header.index("wall_time_sec")
            else:
                # rewind if first row wasn't a header
                try:
                    float(header[0])
                    vals.append(float(header[0]))
                except Exception:
                    pass
            for row in reader:
                if not row:
                    continue
                try:
                    vals.append(float(row[col_idx]))
                except Exception:
                    continue
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return vals

def compute_stats(durations):
    if not durations:
        return {}
    return {
        "count": len(durations),
        "total_sec": sum(durations),
        "mean_sec": stats.mean(durations),
        "median_sec": stats.median(durations),
        "stdev_sec": stats.stdev(durations) if len(durations) > 1 else 0.0,
        "min_sec": min(durations),
        "max_sec": max(durations),
    }

def save_stats(base, stats_dict):
    out_json = base + "_stats.json"
    out_csv  = base + "_stats.csv"

    with open(out_json, "w") as f:
        json.dump(stats_dict, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(stats_dict.keys())
        w.writerow(stats_dict.values())

    print(f"Saved → {out_json}, {out_csv}")

def main(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.*")))
    if not files:
        print("No files found in", folder)
        return

    for path in files:
        if path.endswith(("_stats.json", "_stats.csv")):
            continue  # skip already-generated stats files
        try:
            durations = load_durations(path)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue

        if not durations:
            print(f"[skip] {path}: no durations")
            continue

        s = compute_stats(durations)
        base = os.path.splitext(path)[0]  # drop extension
        save_stats(base, s)

if __name__ == "__main__":
    folder = "Time_Extent_Logs/TinyLLaMa/PIQA/10_Nodes"
    main(folder)

