# from typing import List
# import torch
# import math

# MAX_PAYLOAD_BYTES = 1450

# class LossyNetwork:
#     def __init__(self, loss_rate: float = 0.001):
#         self.loss_rate = loss_rate

#     def set_seed(self, seed: int):
#         self.seed = seed

#     def send(self, data: torch.Tensor) -> torch.Tensor:
#         data_size = data.numel()
#         float_size = data.element_size()
#         num_bytes = data_size * float_size
#         num_packets = math.ceil(num_bytes / MAX_PAYLOAD_BYTES)
#         packets_mask = torch.rand(num_packets) > self.loss_rate
#         return packets_mask
    
#     def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:

#         if packets_mask.all():
#             return data
#         num_packets = len(packets_mask)
#         number_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1

#         flat = data.flatten()
#         indices = torch.arange(num_packets * number_per_packet, device=data.device)
#         indices = indices[indices < flat.numel()]
#         mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]
#         flat[~mask] = 0.0
#         return flat.view_as(data)
    
#         # received_data = data.clone()
#         # received_data = received_data.view(-1)
#         # for i in range(num_packets):
#         #     if not packets_mask[i]:
#         #         start = i * number_per_packet
#         #         end = min(start + number_per_packet, data.numel())
#         #         received_data[start:end] = 0.0
#         # received_data = received_data.view(data.shape)
#         # return received_data



import torch
import math

MAX_PAYLOAD_BYTES = 1450

# def get_num_packets(data: torch.Tensor):
#     data_size = data.numel()
#     float_size = data.element_size()
#     num_bytes = data_size * float_size
#     num_packets = math.ceil(num_bytes / MAX_PAYLOAD_BYTES)
#     return num_packets
# class LossyNetwork:
#     """
#     Simulates a lossy network by randomly dropping packets based on a specified loss rate.
#     You can inherit from this class to create custom network simulations without changing the training code.
#     """
#     def __init__(self, args):
#         self.loss_rate = float(args.loss_rate)

#     def set_seed(self, seed: int):
#         self.seed = seed
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     def send(self, data: torch.Tensor) -> torch.Tensor:
#         num_packets = get_num_packets(data)
#         packets_mask = torch.rand(num_packets) > self.loss_rate
#         return packets_mask
    
#     def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:

#         if packets_mask.all(): # when no packets are lost
#             return data
        
#         num_packets = len(packets_mask)
#         number_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1

#         flat = data.flatten()
#         indices = torch.arange(num_packets * number_per_packet, device=data.device)
#         indices = indices[indices < flat.numel()]
#         mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]
#         flat[~mask] = 0.0
#         return flat.view_as(data)

def get_num_packets(data: torch.Tensor):
    data_size = data.numel()
    float_size = data.element_size()
    num_bytes = data_size * float_size
    num_packets = math.ceil(num_bytes / MAX_PAYLOAD_BYTES)
    return num_packets

class LossyNetwork:
    """
    Simulates a lossy network by randomly dropping packets based on a specified loss rate.
    You can inherit from this class to create custom network simulations without changing the training code.
    """
    def __init__(self, args):
        self.loss_rate = float(args.loss_rate)

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def send(self, data: torch.Tensor) -> torch.Tensor:
        num_packets = get_num_packets(data)
        packets_mask = torch.rand(num_packets) > self.loss_rate
        return packets_mask
    
    def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply packet loss mask to `data` in-place, zeroing elements belonging
        to dropped packets. Works for CPU and CUDA tensors.
        """
        if packets_mask.all():  # when no packets are lost
            return data
        
        num_packets = len(packets_mask)
        number_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1

        flat = data.flatten()

        # Make sure mask is on the same device as data
        packets_mask = packets_mask.to(flat.device)

        indices = torch.arange(num_packets * number_per_packet, device=flat.device)
        indices = indices[indices < flat.numel()]
        mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]

        # Zero out elements belonging to dropped packets
        flat[~mask] = 0.0
        return flat.view_as(data)


# class GillbertElliotLossyNetwork(LossyNetwork):
  #   """
  #   Simulates a Gilbert-Elliott lossy network.
  #   """
  #   def __init__(self, p_gb, p_bg, good_loss_rate=0.0, bad_loss_rate=1.0, args=None):
    #     super().__init__(args)
    #     self.p_gb = p_gb
    #     self.p_bg = p_bg
    #     self.good_loss_rate = good_loss_rate
    #     self.bad_loss_rate = bad_loss_rate
    #     if self.good_loss_rate > self.bad_loss_rate:
      #       raise ValueError("Good loss rate must be less than or equal to bad loss rate.")
    #     self.state = 'good'

  #   def take_step(self, n_steps=1):
    #     transition_probabilities = torch.rand(n_steps)
    #     if self.state == 'good':
      #       if torch.rand(1).item() < self.p_gb:
        #         self.state = 'bad'
      #   else:
        #     if torch.rand(1).item() < self.p_bg:
          #       self.state = 'good'
      #   return self.state
            
  #   def send(self, data: torch.Tensor) -> torch.Tensor:
    #     self.take_step()
    #     num_packets = get_num_packets(data)
    #     if self.state == 'good':
      #       packets_mask = torch.rand(num_packets) > self.good_loss_rate
    #     else:
      #       packets_mask = torch.rand(num_packets) > self.bad_loss_rate
    #     return packets_mask


    # def send_alternative(self, data:torch.Tensor):
    #     num_packets = get_num_packets(data)
    #     step_per_packet = [self.take_step() for _ in range(num_packets)]
    #     packets_mask = torch.tensor([
    #         torch.rand(1).item() > (self.good_loss_rate if step == 'good' else self.bad_loss_rate)
    #         for step in step_per_packet
    #     ], device=data.device)
    #     return packets_mask


import os, json, csv, time
from time import perf_counter
import torch
from datetime import datetime
from types import SimpleNamespace

class GillbertElliotLossyNetwork(LossyNetwork):
    """
    Gilbert–Elliott lossy channel with full episode logging.
    - Per episode (contiguous run in 'good' or 'bad'):
        * state, episode_index, state_ordinal
        * steps, wall_time_sec, start/end timestamps
        * packets sent, packets dropped
        * running totals at episode end
    - Persists both JSON (machine-readable) and TXT (human-readable) per episode.
    - Writes a run-level CSV summary on finalize().
    """

    def __init__(
        self,
        p_gb, p_bg,
        good_loss_rate=0.0, bad_loss_rate=1.0,
        args=None,
        log_dir="ge_logs",
        filename_prefix="",         # e.g. "rank0_"
        start_in_state="good",       # "good" or "bad"
        # --- new optional metadata for auto-folder naming ---
        auto_runs_root="State_Times_Burst_Summary",
        model_name=None,                # e.g., "TinyLLaMA"
        task_name=None,                 # e.g., "PIQA"
        nodes=None,                     # e.g., 8
        loss_label=None,                # e.g., "0.005"
        seed=None,
        rank=None,
        extra_tag=None,                 # e.g., "adamw"
    ):
        shim = args if (args is not None and hasattr(args, "loss_rate")) else SimpleNamespace(loss_rate=0.0)
        super().__init__(shim)
        # Params
        self.p_gb = float(p_gb)              # P(good->bad) per step
        self.p_bg = float(p_bg)              # P(bad->good) per step
        self.good_loss_rate = float(good_loss_rate)
        self.bad_loss_rate  = float(bad_loss_rate)
        self.model_name = model_name
        self.task_name = task_name
        self.nodes = nodes
        # self.loss_label = args.ge_config
        self.seed = seed
        self.loss_label = loss_label

        if not (0.0 <= self.good_loss_rate <= self.bad_loss_rate <= 1.0):
            raise ValueError("Require 0 <= good_loss_rate <= bad_loss_rate <= 1.")
        if not (0.0 <= self.p_gb <= 1.0 and 0.0 <= self.p_bg <= 1.0):
            raise ValueError("p_gb and p_bg must be in [0,1].")
        if start_in_state not in ("good", "bad"):
            raise ValueError("start_in_state must be 'good' or 'bad'.")

        # # Files
        # self.log_dir = log_dir
        # self.filename_prefix = filename_prefix
        # os.makedirs(self.log_dir, exist_ok=True)

        # Files
        if log_dir == "ge_logs":
            # print("############################## -- 1 -- #################################")
            # ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Build pieces safely, skipping Nones
            parts = []
            # if extra_tag:   parts.append(str(extra_tag))
            if self.model_name:  parts.append(str(model_name))
            if self.task_name:   parts.append(str(task_name))
            # join as "{ts}_{extra_tag}_{model}_{task}" for the first level
            first = "_".join(parts)
            # print("############################## -- 2 -- #################################")

            subdirs = []
            if self.nodes is not None:      subdirs.append(f"nodes={nodes}")
            if self.loss_label is not None: subdirs.append(f"loss={loss_label}")
            if self.seed is not None:       subdirs.append(f"seed={seed}")
            # if rank is not None:       subdirs.append(f"rank={rank}")

            full_path = os.path.join(auto_runs_root, first, *subdirs)
            # print(full_path)
            # print("\n\n############################## -- 3 -- #################################")
            os.makedirs(full_path, exist_ok=True)
            self.log_dir = full_path
            # print(self.log_dir)
            # print("\n\n############################## -- 4 -- #################################")
        else:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)

        self.filename_prefix = filename_prefix
        # os.makedirs(self.log_dir, exist_ok=True)


        # Markov state and episode counters
        self.state = start_in_state
        self.episode_index = 0
        self.good_ordinal = 0
        self.bad_ordinal  = 0

        # Global totals
        self.total_steps = 0
        self.total_packets = 0
        self.total_dropped = 0
        self.total_time_sec = 0.0

        # Timing anchors (monotonic for deltas, wall for stamps)
        now_wall = time.time()
        now_mono = perf_counter()
        self._last_wall_ts = now_wall
        self._last_mono_ts = now_mono

        # Open first episode
        self._episode = self._new_episode(self.state, now_wall)
        if self.state == "good":
            self._episode["state_ordinal"] = self.good_ordinal
            self.good_ordinal += 1
        else:
            self._episode["state_ordinal"] = self.bad_ordinal
            self.bad_ordinal += 1

        # Keep a copy of closed episodes in memory (optional)
        self.episodes = []

    # ---------- internals ----------
    def _new_episode(self, state, start_wall_ts):
        return {
            "state": state,                         # "good" | "bad"
            "episode_index": self.episode_index,    # 0,1,2,...
            "state_ordinal": None,                  # which #good or #bad (0-based)
            "start_time_wall": start_wall_ts,       # epoch seconds
            "end_time_wall": None,
            "wall_time_sec": 0.0,                   # accumulated while episode is open
            "steps": 0,
            "packets": 0,
            "dropped": 0,

            # Running totals AT EPISODE END (filled on close)
            "running_total_steps": None,
            "running_total_packets": None,
            "running_total_dropped": None,
            "running_total_time_sec": None,

            # Params snapshot for provenance
            "p_gb": self.p_gb,
            "p_bg": self.p_bg,
            "good_loss_rate": self.good_loss_rate,
            "bad_loss_rate": self.bad_loss_rate,
        }

    def _write_episode_files(self, ep):
        """Write both JSON and human-readable TXT for a closed episode."""
        base = f"{self.filename_prefix}{ep['state']}_{ep['episode_index']}"
        json_path = os.path.join(self.log_dir, base + ".json")
        txt_path  = os.path.join(self.log_dir, base + ".txt")

        # JSON
        with open(json_path, "w") as f:
            json.dump(ep, f, indent=2)

        # TXT (readable summary)
        start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep['start_time_wall']))
        end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep['end_time_wall'])) if ep['end_time_wall'] else '-'
        lines = [
            f"Episode: {ep['state'].upper()} #{ep['state_ordinal']}  (index {ep['episode_index']})",
            f"Start:   {start_str}",
            f"End:     {end_str}",
            f"Duration (wall): {ep['wall_time_sec']:.6f} s",
            f"Steps:           {ep['steps']}",
            f"Packets sent:    {ep['packets']}",
            f"Packets dropped: {ep['dropped']}",
            f"Running totals → steps={ep['running_total_steps']}, "
            f"packets={ep['running_total_packets']}, dropped={ep['running_total_dropped']}, "
            f"wall_time={ep['running_total_time_sec']:.6f} s",
            f"Params: p_gb={ep['p_gb']}, p_bg={ep['p_bg']}, "
            f"good_loss_rate={ep['good_loss_rate']}, bad_loss_rate={ep['bad_loss_rate']}",
            ""
        ]
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    def _close_episode(self, close_wall_ts):
        """Finalize current episode, persist to disk, bump indices."""
        self._episode["end_time_wall"] = close_wall_ts
        # Fill running totals snapshot
        self._episode["running_total_steps"]    = self.total_steps
        self._episode["running_total_packets"]  = self.total_packets
        self._episode["running_total_dropped"]  = self.total_dropped
        self._episode["running_total_time_sec"] = self.total_time_sec

        ep_copy = dict(self._episode)  # freeze
        self.episodes.append(ep_copy)
        self._write_episode_files(ep_copy)

        self.episode_index += 1

    def _open_new_episode(self, new_state, start_wall_ts):
        self._episode = self._new_episode(new_state, start_wall_ts)
        if new_state == "good":
            self._episode["state_ordinal"] = self.good_ordinal
            self.good_ordinal += 1
        else:
            self._episode["state_ordinal"] = self.bad_ordinal
            self.bad_ordinal += 1

    def _maybe_transition(self, now_wall):
        # Transition BEFORE counting this step; elapsed time belongs to the episode that just ended.
        if self.state == 'good':
            if torch.rand(1).item() < self.p_gb:
                self._close_episode(close_wall_ts=now_wall)
                self.state = 'bad'
                self._open_new_episode('bad', start_wall_ts=now_wall)
        else:  # 'bad'
            if torch.rand(1).item() < self.p_bg:
                self._close_episode(close_wall_ts=now_wall)
                self.state = 'good'
                self._open_new_episode('good', start_wall_ts=now_wall)

    def _write_summary_csv(self):
        """Run-level CSV over all finalized episodes (call after finalizes))."""
        csv_path = os.path.join(self.log_dir, f"{self.filename_prefix}episodes_summary.csv")
        fieldnames = [
            "episode_index","state","state_ordinal",
            "start_time_wall","end_time_wall","wall_time_sec",
            "steps","packets","dropped",
            "running_total_steps","running_total_packets","running_total_dropped","running_total_time_sec",
            "p_gb","p_bg","good_loss_rate","bad_loss_rate"
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for ep in self.episodes:
                w.writerow({k: ep.get(k) for k in fieldnames})

    # ---------- public API ----------
    def take_step(self, n_steps=1):
        """
        Advance the GE chain n steps.
        Define 'one step' at the cadence you want (e.g., one comm/all-reduce round).
        """
        for _ in range(n_steps):
            # elapsed time since last step -> accrue to current episode
            now_wall = time.time()
            now_mono = perf_counter()
            dt = max(0.0, now_mono - self._last_mono_ts)
            self._episode["wall_time_sec"] += dt
            self.total_time_sec += dt
            self._last_mono_ts = now_mono
            self._last_wall_ts = now_wall

            # possible state transition
            self._maybe_transition(now_wall)

            # count this step
            self._episode["steps"] += 1
            self.total_steps += 1

        return self.state

    def send(self, data: torch.Tensor) -> torch.Tensor:
        """
        Advance one step and apply per-packet Bernoulli drops conditioned on state.
        Returns a 1-D bool mask (True=kept, False=dropped) of length get_num_packets(data).
        """
        self.take_step(1)

        num_packets = get_num_packets(data)
        if num_packets == 0:
            return torch.ones(0, dtype=torch.bool)

        loss_rate = self.good_loss_rate if self.state == 'good' else self.bad_loss_rate
        packets_mask = torch.rand(num_packets) > loss_rate

        kept = int(packets_mask.sum().item())
        dropped = int(num_packets - kept)

        # update episode + totals
        self._episode["packets"] += num_packets
        self._episode["dropped"] += dropped
        self.total_packets += num_packets
        self.total_dropped += dropped

        return packets_mask

    def finalize(self):
        """Close the open episode, persist it, and write a CSV summary."""
        now_wall = time.time()
        now_mono = perf_counter()
        dt = max(0.0, now_mono - self._last_mono_ts)
        self._episode["wall_time_sec"] += dt
        self.total_time_sec += dt
        self._last_mono_ts = now_mono
        self._last_wall_ts = now_wall

        self._close_episode(close_wall_ts=now_wall)
        self._write_summary_csv()

    # ---------- convenience ----------
    @property
    def packet_loss_rate_overall(self) -> float:
        return 0.0 if self.total_packets == 0 else self.total_dropped / self.total_packets

    @property
    def bad_fraction_steps(self) -> float:
        bad_steps = sum(ep["steps"] for ep in self.episodes if ep["state"] == "bad")
        return 0.0 if self.total_steps == 0 else bad_steps / self.total_steps

