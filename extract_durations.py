#!/usr/bin/env python3
import os, glob, json, argparse, csv

def collect_durations(folder, state="bad", recursive=True):
    """
    Return a list of wall_time_sec for all episodes with state == {state}.
    """
    pattern = os.path.join(folder, "**", "*.json") if recursive else os.path.join(folder, "*.json")
    durations = []
    for path in glob.glob(pattern, recursive=recursive):
        try:
            with open(path, "r") as f:
                ep = json.load(f)
            if ep.get("state") == state:
                t = float(ep.get("wall_time_sec", 0.0))
                durations.append(t)
        except Exception:
            # ignore unreadable files or non-episode JSONs
            pass
    return durations

def save_json(durations, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(durations, f, indent=2)

def save_txt(durations, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for t in durations:
            f.write(f"{t}\n")

def save_csv(durations, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_time_sec"])
        for t in durations:
            w.writerow([t])

def main():
    p = argparse.ArgumentParser(description="Extract episode durations by state from GE logs.")
    p.add_argument("folder", help="Path to a run folder (or parent). Use r'...' if your path has backslashes.")
    p.add_argument("--state", choices=["bad","good"], default="bad", help="Which state to extract (default: bad).")
    p.add_argument("--recursive", action="store_true", help="Search subfolders recursively (useful with rank=*/).")
    p.add_argument("--out-base", default=None,
                   help="Base path (without extension) for outputs. Default: <folder>/<state>_durations_seconds")
    p.add_argument("--formats", nargs="+", choices=["json","txt","csv"], default=["json","txt"],
                   help="Output formats to write (default: json txt).")
    args = p.parse_args()

    durations = collect_durations(args.folder, state=args.state, recursive=args.recursive)
    if not durations:
        print(f"No episode JSONs found for state='{args.state}' under: {args.folder}")
        return

    # choose default output base name
    if args.out_base is None:
        out_base = os.path.join(args.folder, f"{args.state}_durations_seconds")
    else:
        out_base = args.out_base

    # write in requested formats
    if "json" in args.formats:
        save_json(durations, out_base + ".json")
    if "txt" in args.formats:
        save_txt(durations, out_base + ".txt")
    if "csv" in args.formats:
        save_csv(durations, out_base + ".csv")

    total = sum(durations)
    print(f"Extracted {len(durations)} {args.state.upper()} episodes.")
    print(f"Saved: " + ", ".join(f"{out_base}.{ext}" for ext in args.formats))
    print(f"Quick check → total {args.state} time: {total:.6f} s; mean: {total/len(durations):.6f} s")

if __name__ == "__main__":
    main()

