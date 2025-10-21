import json, glob, os

def total_state_times(folder):
    bad_time = good_time = 0.0
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path) as f:
            ep = json.load(f)
        t = float(ep.get("wall_time_sec", 0.0))
        if ep.get("state") == "bad":
            bad_time += t
        elif ep.get("state") == "good":
            good_time += t
    return bad_time, good_time

# Example: replace with the path to your run folder
folder = r"State_Times/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T_piqa/nodes=2/loss=short_point1_percent/seed=10/"
bad_sec, good_sec = total_state_times(folder)

print("Looking in:", folder)
# print("Found JSON files:", glob.glob(os.path.join(folder, "*.json")))

print(f"Total BAD state time:  {bad_sec:.3f} seconds")
print(f"Total GOOD state time: {good_sec:.3f} seconds")
print(f"Total runtime:         {bad_sec + good_sec:.3f} seconds")

