# deterministic_loss.py
"""
Deterministic Burst-Loss Engine (CSV-driven, no recomputation)
==============================================================

This class reads ONE ROW from runs_profiles.csv and uses the provided metrics:
  - T_steps, N, L_overall, lrg, lrb
  - piB, B, Eb, rho
to build a deterministic BAD/GOOD timeline with exactly N episodes summing to B BAD steps.
It applies packet-level masking identical to your original code and logs per-episode files.

Identity recap (provided by params_gen.py; NOT recomputed here):
  L = π_B * l_rb + (1 - π_B) * l_rg
  B = round(π_B * T)
  E[b] = B / N
  ρ = N / T = π_B / E[b]

If you change tactics (e.g., vary ρ instead of E[b]), do it in params_gen.py.
This file simply *uses* what the CSV provides.
"""

import math, csv, os, json, time
from time import perf_counter
from typing import List, Optional
import torch

MAX_PAYLOAD_BYTES = 1450
GOOD, BAD = 0, 1

def get_num_packets(data: torch.Tensor) -> int:
    data_size   = data.numel()
    float_size  = data.element_size()
    num_bytes   = data_size * float_size
    return math.ceil(num_bytes / MAX_PAYLOAD_BYTES)

def _build_equal_lengths(total: int, parts: int) -> List[int]:
    base = total // parts
    rem  = total - base * parts
    arr  = [base] * parts
    for i in range(rem):
        arr[i] += 1
    return arr

def _skew_longer(lengths: List[int], frac: float) -> List[int]:
    if frac <= 0.0: return lengths
    n = len(lengths); k = max(1, int(n * frac))
    for i in range(k):
        j = n - 1 - i
        give = max(0, lengths[j] - 1)
        take = min(give, lengths[i])
        if take > 0:
            lengths[i] += take
            lengths[j] -= take
    return lengths

def _timeline_from_bursts(T_steps: int, burst_lengths: List[int], gap_mode: str = "even") -> List[int]:
    B = sum(burst_lengths)
    G = T_steps - B
    assert G >= 0, "Total BAD steps exceed T_steps."
    N = len(burst_lengths)

    # even gap distribution (others can be added later)
    base_gap = G // (N + 1)
    rem_gap  = G - base_gap * (N + 1)
    gaps     = [base_gap] * (N + 1)
    for i in range(rem_gap):
        gaps[i] += 1

    state = []
    state.extend([GOOD] * gaps[0])
    for i in range(N):
        state.extend([BAD] * burst_lengths[i])
        state.extend([GOOD] * gaps[i + 1])
    if len(state) < T_steps:
        state.extend([GOOD] * (T_steps - len(state)))
    return state[:T_steps]

class DeterministicBurstLossyNetwork:
    """
    CSV-driven, GE-free lossy network:
      • Builds deterministic BAD/GOOD schedule with exactly N episodes and B BAD steps,
      • Drops packets at per-BAD intensity lrb with exact overall-loss accumulator,
      • Mirrors your packet-masking semantics,
      • Logs per-episode JSON/TXT + run-level CSV summary.

    NOTE: This class TRUSTS metrics from CSV (no recomputation). It can assert consistency.
    """

    def __init__(
        self,
        *,
        csv_path: str,
        run_id: Optional[str] = None,
        row_index: Optional[int] = None,
        log_dir: str = "det_logs",
        strict_validate: bool = True,   # only asserts; never recomputes
    ):
        row = self._load_row(csv_path, run_id, row_index)

        # Core params (as provided by params_gen.py)
        self.run_id    = row.get("run_id", "run")
        self.T_steps   = int(row["T_steps"])
        self.N         = int(row["N"])
        self.L_overall = float(row["L_overall"])
        self.lrg       = float(row.get("lrg", 0.0))
        self.lrb       = float(row["lrb"])
        self.piB       = float(row["piB"])
        self.B         = int(row["B"])
        self.Eb        = float(row["Eb"])
        self.rho       = float(row["rho"])
        self.seed      = int(row.get("seed", 0))
        self.skew_frac = float(row.get("skew_frac", 0.0))
        self.gap_mode  = str(row.get("gap_mode", "even"))

        if strict_validate:
            # Pure assertions; no recomputation or overriding
            assert 0.0 <= self.lrg <= 1.0
            assert 0.0 < self.lrb <= 1.0
            assert 0.0 <= self.L_overall <= 1.0
            # Identities (tolerate small rounding error from B = round(πB*T))
            assert abs(self.piB - (self.L_overall - self.lrg) / (self.lrb - self.lrg)) < 1e-6, "piB mismatch"
            assert abs(self.Eb - (self.B / self.N)) < 1e-9, "Eb mismatch"
            assert abs(self.rho - (self.N / self.T_steps)) < 1e-12, "rho mismatch"
            assert abs(self.piB - (self.B / self.T_steps)) < 1e-3, "piB vs B/T mismatch (rounding tolerated)"

        # Build burst lengths from B and N (sum must equal B)
        bl = _build_equal_lengths(self.B, self.N)
        bl = _skew_longer(bl, self.skew_frac)
        assert sum(bl) == self.B
        self.state = _timeline_from_bursts(self.T_steps, bl, self.gap_mode)

        # Logging folders (optional metadata like model/task can be added to CSV if desired)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.filename_prefix = ""

        # Accumulators
        self.step_idx            = 0
        self.total_steps         = 0
        self.total_packets       = 0
        self.total_dropped       = 0
        self.total_time_sec      = 0.0
        self.target_drops_accum  = 0.0
        self.dropped_so_far      = 0

        # Episode bookkeeping
        self.episode_index = 0
        self.good_ordinal  = 0
        self.bad_ordinal   = 0
        self.episodes      = []
        self._open_episode(self.state[0] if self.T_steps > 0 else GOOD)

        # Optional: announce the parameters you provided (no recomputation)
        print(f"[{self.run_id}] T={self.T_steps}, N={self.N}, L={self.L_overall}, "
              f"lrg={self.lrg}, lrb={self.lrb}, piB={self.piB:.6f}, B={self.B}, Eb={self.Eb:.3f}, rho={self.rho:.6f}")

    def set_seed(self, seed: int):
        """Override the RNG seed used for packet-drop mask generation."""
        self.seed = int(seed)
        return self


    @classmethod
    def from_params(cls,
        *,
        run_id: str,
        T_steps: int,
        N: int,
        L_overall: float,
        lrg: float,
        lrb: float,
        piB: float,
        B: int,
        Eb: float,
        rho: float,
        seed: int,
        skew_frac: float,
        gap_mode: str,
        log_dir: str = "det_logs",
        strict_validate: bool = True,
    ):
        """
        Alternate constructor that takes parameters directly (no CSV access).
        Mirrors the columns in your CSV and uses them as-is.
        """
        # Build a row-dict identical to what _load_row would return
        row = {
            "run_id": run_id,
            "T_steps": int(T_steps),
            "N": int(N),
            "L_overall": float(L_overall),
            "lrg": float(lrg),
            "lrb": float(lrb),
            "piB": float(piB),
            "B": int(B),
            "Eb": float(Eb),
            "rho": float(rho),
            "seed": int(seed),
            "skew_frac": float(skew_frac),
            "gap_mode": str(gap_mode),
        }

        # Instantiate an object without triggering CSV loading
        self = cls.__new__(cls)  # bypass __init__
        # ---- paste the same body you run after loading a row ----
        # Core params
        self.run_id    = row.get("run_id", "run")
        self.T_steps   = int(row["T_steps"])
        self.N         = int(row["N"])
        self.L_overall = float(row["L_overall"])
        self.lrg       = float(row.get("lrg", 0.0))
        self.lrb       = float(row["lrb"])
        self.piB       = float(row["piB"])
        self.B         = int(row["B"])
        self.Eb        = float(row["Eb"])
        self.rho       = float(row["rho"])
        self.seed      = int(row.get("seed", 0))
        self.skew_frac = float(row.get("skew_frac", 0.0))
        self.gap_mode  = str(row.get("gap_mode", "even"))

        if strict_validate:
            assert 0.0 <= self.lrg <= 1.0
            assert 0.0 <  self.lrb <= 1.0
            assert 0.0 <= self.L_overall <= 1.0
            # identities (allowing small rounding error on B = round(piB*T))
            assert abs(self.Eb - (self.B / self.N)) < 1e-9, "Eb mismatch"
            assert abs(self.rho - (self.N / self.T_steps)) < 1e-12, "rho mismatch"
            assert abs(self.piB - (self.B / self.T_steps)) < 1e-3, "piB vs B/T mismatch"

        # Build burst lengths & timeline (same helpers you already have)
        bl = _build_equal_lengths(self.B, self.N)
        bl = _skew_longer(bl, self.skew_frac)
        assert sum(bl) == self.B
        self.state = _timeline_from_bursts(self.T_steps, bl, self.gap_mode)

        # Logging setup (unchanged from your class)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.filename_prefix = ""

        # Accumulators & episode bookkeeping (unchanged)
        self.step_idx = 0
        self.total_steps = 0
        self.total_packets = 0
        self.total_dropped = 0
        self.total_time_sec = 0.0
        self.target_drops_accum = 0.0
        self.dropped_so_far = 0

        self.episode_index = 0
        self.good_ordinal  = 0
        self.bad_ordinal   = 0
        self.episodes      = []
        self._open_episode(self.state[0] if self.T_steps > 0 else GOOD)

        print(f"[{self.run_id}] (from_params) T={self.T_steps}, N={self.N}, L={self.L_overall}, "
              f"lrg={self.lrg}, lrb={self.lrb}, piB={self.piB:.6f}, B={self.B}, Eb={self.Eb:.3f}, rho={self.rho:.6f}")

        return self

    # ---- Episode logging helpers (same style as your GE logger) ----
    # def _episode_template(self, state: int, start_wall_ts: float):
    #     return {
    #         "state": "bad" if state == BAD else "good",
    #         "episode_index": self.episode_index,
    #         "state_ordinal": None,
    #         "start_time_wall": start_wall_ts,
    #         "end_time_wall": None,
    #         "wall_time_sec": 0.0,
    #         "steps": 0,
    #         "packets": 0,
    #         "dropped": 0,
    #         "running_total_steps": None,
    #         "running_total_packets": None,
    #         "running_total_dropped": None,
    #         "running_total_time_sec": None,
    #         "lrb": self.lrb,
    #         "lrg": self.lrg,
    #     }
    
    def _episode_template(self, state: int, start_wall_ts: float):
        return {
            "state": "bad" if state == BAD else "good",
            "episode_index": self.episode_index,
            "state_ordinal": None,
            "start_time_wall": start_wall_ts,
            "end_time_wall": None,
            "wall_time_sec": 0.0,

            # IMPORTANT: keep this as "lossy_calls" (not optimizer steps)
            "lossy_calls": 0,

            # New: optimizer-step-based accounting
            "optimizer_steps": 0,
            "global_step_start": None,
            "global_step_end": None,

            "packets": 0,
            "dropped": 0,

            # Running totals (split by meaning)
            "running_total_lossy_calls": None,
            "running_total_optimizer_steps": None,
            "running_total_packets": None,
            "running_total_dropped": None,
            "running_total_time_sec": None,

            "lrb": self.lrb,
            "lrg": self.lrg,
        }


    # def _write_episode_files(self, ep: dict):
    #     base = f"{self.filename_prefix}{ep['state']}_{ep['episode_index']}"
    #     json_path = os.path.join(self.log_dir, base + ".json")
    #     txt_path  = os.path.join(self.log_dir, base + ".txt")
    #     with open(json_path, "w") as f:
    #         json.dump(ep, f, indent=2)
    #     start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep['start_time_wall']))
    #     end_str   = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep['end_time_wall'])) if ep['end_time_wall'] else '-'
    #     lines = [
    #         f"Episode: {ep['state'].upper()} #{ep['state_ordinal']}  (index {ep['episode_index']})",
    #         f"Start:   {start_str}",
    #         f"End:     {end_str}",
    #         f"Duration (wall): {ep['wall_time_sec']:.6f} s",
    #         f"Steps:           {ep['steps']}",
    #         f"Packets sent:    {ep['packets']}",
    #         f"Packets dropped: {ep['dropped']}",
    #         f"Running totals → steps={ep['running_total_steps']}, "
    #         f"packets={ep['running_total_packets']}, dropped={ep['running_total_dropped']}, "
    #         f"wall_time={ep['running_total_time_sec']:.6f} s",
    #         f"Params: lrb={ep['lrb']}, lrg={ep['lrg']}",
    #         ""
    #     ]
    #     with open(txt_path, "w") as f:
    #         f.write("\n".join(lines))
    
    # def _write_episode_files(self, ep: dict):
        # --- be defensive: FSDP can trigger close/write in odd moments ---
        state = ep.get("state", "?")
        episode_index = ep.get("episode_index", "?")
        base = f"{self.filename_prefix}{state}_{episode_index}"

        json_path = os.path.join(self.log_dir, base + ".json")
        txt_path  = os.path.join(self.log_dir, base + ".txt")

        # JSON (write whatever we have)
        with open(json_path, "w") as f:
            json.dump(ep, f, indent=2)

        # Human-readable TXT (use defaults if missing)
        start_wall = ep.get("start_time_wall", None)
        end_wall   = ep.get("end_time_wall", None)

        start_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_wall))
            if start_wall is not None
            else "-"
        )
        end_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_wall))
            if end_wall
            else "-"
        )

        wall_time_sec = float(ep.get("wall_time_sec", 0.0) or 0.0)
        steps   = int(ep.get("steps", 0) or 0)
        packets = int(ep.get("packets", 0) or 0)
        dropped = int(ep.get("dropped", 0) or 0)

        running_total_steps   = ep.get("running_total_steps", 0)
        running_total_packets = ep.get("running_total_packets", 0)
        running_total_dropped = ep.get("running_total_dropped", 0)
        running_total_time    = float(ep.get("running_total_time_sec", 0.0) or 0.0)

        state_ordinal = ep.get("state_ordinal", "?")
        lrb = ep.get("lrb", None)
        lrg = ep.get("lrg", None)

        lines = [
            f"Episode: {str(state).upper()} #{state_ordinal}  (index {episode_index})",
            f"Start:   {start_str}",
            f"End:     {end_str}",
            f"Duration (wall): {wall_time_sec:.6f} s",
            f"Steps:           {steps}",
            f"Packets sent:    {packets}",
            f"Packets dropped: {dropped}",
            f"Running totals → steps={running_total_steps}, "
            f"packets={running_total_packets}, dropped={running_total_dropped}, "
            f"wall_time={running_total_time:.6f} s",
            f"Params: lrb={lrb}, lrg={lrg}",
            ""
        ]

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    def _write_episode_files(self, ep: dict):
        # Defensive defaults
        state = ep.get("state", "?")
        episode_index = ep.get("episode_index", "?")
        base = f"{self.filename_prefix}{state}_{episode_index}"

        json_path = os.path.join(self.log_dir, base + ".json")
        txt_path  = os.path.join(self.log_dir, base + ".txt")

        # Write JSON
        with open(json_path, "w") as f:
            json.dump(ep, f, indent=2)

        # Format timestamps
        start_wall = ep.get("start_time_wall")
        end_wall   = ep.get("end_time_wall")

        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_wall)) if start_wall else "-"
        end_str   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_wall)) if end_wall else "-"

        wall_time_sec = float(ep.get("wall_time_sec") or 0.0)

        lossy_calls     = int(ep.get("lossy_calls") or 0)
        optimizer_steps = int(ep.get("optimizer_steps") or 0)
        gs_start        = ep.get("global_step_start", None)
        gs_end          = ep.get("global_step_end", None)

        packets = int(ep.get("packets") or 0)
        dropped = int(ep.get("dropped") or 0)

        rt_lossy_calls    = ep.get("running_total_lossy_calls", None)
        rt_opt_steps      = ep.get("running_total_optimizer_steps", None)
        rt_packets        = ep.get("running_total_packets", None)
        rt_dropped        = ep.get("running_total_dropped", None)
        rt_time           = ep.get("running_total_time_sec", None)

        state_ordinal = ep.get("state_ordinal", "?")
        lrb = ep.get("lrb", None)
        lrg = ep.get("lrg", None)

        lines = [
            f"Episode: {str(state).upper()} #{state_ordinal}  (index {episode_index})",
            f"Start:   {start_str}",
            f"End:     {end_str}",
            f"Duration (wall): {wall_time_sec:.6f} s",
            f"Lossy calls (collectives): {lossy_calls}",
            f"Optimizer steps:          {optimizer_steps}",
            f"Global step range:        {gs_start} → {gs_end}",
            f"Packets sent:             {packets}",
            f"Packets dropped:          {dropped}",
            f"Running totals → lossy_calls={rt_lossy_calls}, opt_steps={rt_opt_steps}, "
            f"packets={rt_packets}, dropped={rt_dropped}, wall_time={rt_time}",
            f"Params: lrb={lrb}, lrg={lrg}",
            ""
        ]

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))


    # def _open_episode(self, state: int):
    #     now_wall = time.time()
    #     self._last_mono_ts = perf_counter()
    #     self._last_wall_ts = now_wall
    #     self._episode = self._episode_template(state, now_wall)
    #     if state == GOOD:
    #         self._episode["state_ordinal"] = self.good_ordinal; self.good_ordinal += 1
    #     else:
    #         self._episode["state_ordinal"] = self.bad_ordinal;  self.bad_ordinal  += 1

    def _open_episode(self, state: int):
        now_wall = time.time()
        self._last_mono_ts = perf_counter()
        self._last_wall_ts = now_wall
        self._episode = self._episode_template(state, now_wall)

        # ordinals
        if state == GOOD:
            self._episode["state_ordinal"] = self.good_ordinal; self.good_ordinal += 1
        else:
            self._episode["state_ordinal"] = self.bad_ordinal;  self.bad_ordinal  += 1

        # set global step start immediately
        try:
            gs = int(os.environ.get("LOSSY_GLOBAL_STEP", "0"))
        except Exception:
            gs = 0
        self._episode["global_step_start"] = gs
        self._episode["global_step_end"] = gs

        # reset per-episode step tracking so the first send() in this episode counts correctly
        self._last_seen_global_step = gs


    def _close_episode(self):
        now_wall = time.time()
        now_mono = perf_counter()
        dt = max(0.0, now_mono - self._last_mono_ts)
        self._episode["wall_time_sec"] += dt
        self.total_time_sec            += dt
        self._last_mono_ts = now_mono
        self._last_wall_ts = now_wall

        self._episode["end_time_wall"]         = now_wall
        # self._episode["running_total_steps"]   = self.total_steps
        # self._episode["running_total_packets"] = self.total_packets
        # self._episode["running_total_dropped"] = self.total_dropped
        # self._episode["running_total_time_sec"]= self.total_time_sec
        
        # Ensure global_step_end is set
        if self._episode.get("global_step_end", None) is None:
            try:
                self._episode["global_step_end"] = int(os.environ.get("LOSSY_GLOBAL_STEP", "0"))
            except Exception:
                self._episode["global_step_end"] = self._episode.get("global_step_start", 0)

        self._episode["running_total_lossy_calls"]      = getattr(self, "total_lossy_calls", None)
        self._episode["running_total_optimizer_steps"]  = getattr(self, "total_optimizer_steps", None)
        self._episode["running_total_packets"]          = self.total_packets
        self._episode["running_total_dropped"]          = self.total_dropped
        self._episode["running_total_time_sec"]         = self.total_time_sec
        
        # normalize missing counters (can happen if episode was opened in a minimal dict path)
        self._episode.setdefault("steps", 0)
        self._episode.setdefault("packets", 0)
        self._episode.setdefault("dropped", 0)
        self._episode.setdefault("wall_time_sec", 0.0)

        ep_copy = dict(self._episode)
        self.episodes.append(ep_copy)
        self._write_episode_files(ep_copy)
        self.episode_index += 1

    def _roll_episode_if_needed(self, next_state: int):
        curr_state = BAD if (self._episode["state"] == "bad") else GOOD
        if next_state != curr_state:
            self._close_episode()
            self._open_episode(next_state)

    # ---- Masking (packet-level; identical semantics to your code) ----
    def _packet_mask_from_k(self, num_packets: int, k_drop: int) -> torch.Tensor:
        if k_drop <= 0:
            return torch.ones(num_packets, dtype=torch.bool)
        if k_drop >= num_packets:
            return torch.zeros(num_packets, dtype=torch.bool)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed * 2_147_483_647 + self.step_idx + 1)
        scores = torch.rand(num_packets, generator=gen)
        drop_idx = torch.argsort(scores)[:k_drop]
        mask = torch.ones(num_packets, dtype=torch.bool)
        mask[drop_idx] = False
        return mask

    # def send(self, data: torch.Tensor) -> torch.Tensor:
    #     if self.step_idx >= self.T_steps:
    #         num_packets = get_num_packets(data)
    #         return torch.ones(num_packets, dtype=torch.bool)

    #     now_mono = perf_counter()
    #     dt = max(0.0, now_mono - getattr(self, "_last_mono_ts", now_mono))
    #     self._episode["wall_time_sec"] += dt
    #     self.total_time_sec += dt
    #     self._last_mono_ts = now_mono
    #     self._last_wall_ts = time.time()

    #     curr_state = self.state[self.step_idx]
    #     num_packets = get_num_packets(data)

    #     if curr_state == GOOD:
    #         k_drop = 0
    #     else:
    #         # Accumulate target drops and emit integer part now — guarantees exact overall loss by end
    #         self.target_drops_accum += self.lrb * num_packets
    #         k_drop = int(math.floor(self.target_drops_accum - self.dropped_so_far))
    #         k_drop = max(0, min(k_drop, num_packets))

    #     mask = self._packet_mask_from_k(num_packets, k_drop)

    #     dropped = int((~mask).sum().item())
    #     self._episode["steps"]   += 1
    #     self._episode["packets"] += num_packets
    #     self._episode["dropped"] += dropped

    #     self.total_steps   += 1
    #     self.total_packets += num_packets
    #     self.total_dropped += dropped
    #     self.dropped_so_far+= dropped

    #     self.step_idx += 1
    #     next_state = self.state[self.step_idx] if self.step_idx < self.T_steps else curr_state
    #     self._roll_episode_if_needed(next_state)
    #     return mask
    
    def send(self, data: torch.Tensor) -> torch.Tensor:
        """
        STEP-BASED deterministic schedule for FSDP:
        - Use HF global_step (LOSSY_GLOBAL_STEP) to choose GOOD/BAD state.
        - Do NOT advance the schedule per collective call.
        - Keep per-episode counts:
            * lossy_calls = how many collectives invoked send()
            * optimizer_steps = how many distinct global_steps occurred in this episode
        """

        # ---- Lazy-init extra counters (keeps patch minimal) ----
        if not hasattr(self, "total_lossy_calls"):
            self.total_lossy_calls = 0
        if not hasattr(self, "total_optimizer_steps"):
            self.total_optimizer_steps = 0
        if not hasattr(self, "_last_seen_global_step"):
            self._last_seen_global_step = None

        # ---- Read optimizer step from HF Trainer ----
        try:
            gs = int(os.environ.get("LOSSY_GLOBAL_STEP", "0"))
        except Exception:
            gs = 0

        # Clamp schedule index to [0, T_steps-1]
        if self.T_steps <= 0:
            idx = 0
        else:
            idx = min(max(gs, 0), self.T_steps - 1)

        curr_state = self.state[idx] if self.T_steps > 0 else GOOD

        # ---- Roll episode if the GOOD/BAD state changed (based on optimizer step) ----
        curr_ep_state = BAD if (self._episode["state"] == "bad") else GOOD
        if curr_state != curr_ep_state:
            self._close_episode()
            self._open_episode(curr_state)

        # ---- Track optimizer-step transitions (global_step changes) ----
        if self._last_seen_global_step is None:
            self._last_seen_global_step = gs
            self._episode["global_step_start"] = gs

        if gs != self._last_seen_global_step:
            # new optimizer step
            self.total_optimizer_steps += 1
            self._episode["optimizer_steps"] += 1
            self._episode["global_step_end"] = gs
            self._last_seen_global_step = gs

        # ---- Update wall-clock accounting ----
        now_mono = perf_counter()
        dt = max(0.0, now_mono - getattr(self, "_last_mono_ts", now_mono))
        self._episode["wall_time_sec"] += dt
        self.total_time_sec += dt
        self._last_mono_ts = now_mono
        self._last_wall_ts = time.time()

        # ---- Packetization + masking ----
        num_packets = get_num_packets(data)

        # If beyond schedule horizon, treat as GOOD (no drops)
        if gs >= self.T_steps:
            k_drop = 0
        elif curr_state == GOOD:
            k_drop = 0
        else:
            # Accumulate target drops (BAD intensity only)
            self.target_drops_accum += self.lrb * num_packets
            k_drop = int(math.floor(self.target_drops_accum - self.dropped_so_far))
            k_drop = max(0, min(k_drop, num_packets))

        # Track "lossy call count" (collective-call count)
        self.total_lossy_calls += 1
        self._episode["lossy_calls"] += 1

        mask = self._packet_mask_from_k(num_packets, k_drop)

        dropped = int((~mask).sum().item())
        self._episode["packets"] += num_packets
        self._episode["dropped"] += dropped

        self.total_packets += num_packets
        self.total_dropped += dropped
        self.dropped_so_far += dropped

        return mask


    # def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:
    #     if packets_mask.all():
    #         return data
    #     num_packets = len(packets_mask)
    #     number_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1
    #     flat = data.flatten()
    #     indices = torch.arange(num_packets * number_per_packet, device=data.device)
    #     indices = indices[indices < flat.numel()]
    #     mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]
    #     flat[~mask] = 0.0
    #     return flat.view_as(data)
    
    def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:
        # packets_mask: True = keep, False = drop
        if packets_mask.all():
            return data

        flat = data.view(-1)

        # how many tensor elements correspond to one "packet"
        elems_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1

        # get dropped packet indices (do this on CPU to avoid GPU temp allocations)
        if packets_mask.is_cuda:
            drop_idx = (~packets_mask).nonzero(as_tuple=False).view(-1).cpu()
        else:
            drop_idx = (~packets_mask).nonzero(as_tuple=False).view(-1)

        # zero the corresponding ranges in-place (bounded; no huge temp tensors)
        for p in drop_idx.tolist():
            start = p * elems_per_packet
            if start >= flat.numel():
                break
            end = min(start + elems_per_packet, flat.numel())
            flat[start:end] = 0

        return data


    def finalize(self):
        self._close_episode()
        # Summary CSV per run
        csv_path = os.path.join(self.log_dir, f"{self.run_id}_episodes_summary.csv")
        # fieldnames = [
        #     "episode_index","state","state_ordinal",
        #     "start_time_wall","end_time_wall","wall_time_sec",
        #     "steps","packets","dropped",
        #     "running_total_steps","running_total_packets","running_total_dropped","running_total_time_sec",
        #     "lrb","lrg"
        # ]
        fieldnames = [
            "episode_index","state","state_ordinal",
            "start_time_wall","end_time_wall","wall_time_sec",
            "lossy_calls","optimizer_steps","global_step_start","global_step_end",
            "packets","dropped",
            "running_total_lossy_calls","running_total_optimizer_steps",
            "running_total_packets","running_total_dropped","running_total_time_sec",
            "lrb","lrg"
        ]

        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for ep in self.episodes:
                w.writerow({k: ep.get(k) for k in fieldnames})

    # ---- CSV loader (no computations) ----
    def _load_row(self, path: str, run_id: Optional[str], row_index: Optional[int]):
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        if run_id is not None:
            for r in rows:
                if r.get("run_id", "") == run_id:
                    return r
            raise ValueError(f"run_id={run_id} not found in {path}")
        if row_index is not None:
            if row_index < 0 or row_index >= len(rows):
                raise ValueError(f"row_index {row_index} out of range 0..{len(rows)-1}")
            return rows[row_index]
        return rows[0]

