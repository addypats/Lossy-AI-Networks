# lossy_patch_sanity_check.py

import os
import time
import math
import csv
from collections import defaultdict

import torch
import torch.distributed as dist

from comms import LossyNetwork, MAX_PAYLOAD_BYTES

FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

# ---- Log root + run id -------------------------------------------------

_LOG_ROOT = os.environ.get("SANITY_CHECK_LOGS", ".")
_RUN_ID = os.environ.get("RUN_ID", "default_run")

os.makedirs(_LOG_ROOT, exist_ok=True)
_PACKET_LOG_PATH = os.path.join(_LOG_ROOT, f"{_RUN_ID}_packet_log.csv")

# ---- Per-layer-instance logging state -----------------------------------

# We treat each (global_elems, dtype, occurrence_index) as a distinct "layer instance".
# occurrence_index is the k-th time we see an AG/RS of that shape.

_LAYER_INSTANCE_INFO = {}
_NEXT_LAYER_ID = 0

# Per-shape occurrence counters for AG and RS
_AG_OCC_COUNT = defaultdict(int)   # (global_elems, dtype) -> count
_RS_OCC_COUNT = defaultdict(int)   # (global_elems, dtype) -> count


def _is_payload_tensor(t: torch.Tensor) -> bool:
    """Only touch non-empty floating tensors."""
    return (
        isinstance(t, torch.Tensor)
        and t.numel() > 0
        and t.dtype in FLOAT_DTYPES
    )


def _layer_shape_str(global_elems: int, dtype: torch.dtype) -> str:
    """Human-readable layer shape string for the CSV."""
    return f"global_elems={global_elems}, dtype={str(dtype)}"


def _write_layer_row(entry: dict) -> None:
    """
    Append a single row for this layer instance to the CSV:

    Layer,LayerShape,
    ag_total_packets,ag_dropped_packets,ag_received_packets,
    rs_total_packets,rs_dropped_packets,rs_received_packets

    Only rank 0 writes.
    """
    try:
        r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        r = 0

    if r != 0:
        return

    file_exists = os.path.exists(_PACKET_LOG_PATH)

    with open(_PACKET_LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Layer",
                    "LayerShape",
                    "ag_total_packets",
                    "ag_dropped_packets",
                    "ag_received_packets",
                    "rs_total_packets",
                    "rs_dropped_packets",
                    "rs_received_packets",
                ]
            )

        layer_shape = _layer_shape_str(entry["global_elems"], entry["dtype"])
        writer.writerow(
            [
                entry["layer_id"],
                layer_shape,
                entry.get("ag_total_packets", 0),
                entry.get("ag_dropped_packets", 0),
                entry.get("ag_received_packets", 0),
                entry.get("rs_total_packets", 0),
                entry.get("rs_dropped_packets", 0),
                entry.get("rs_received_packets", 0),
            ]
        )


def _record_packets_instance(fn_name: str, t: torch.Tensor, stats: dict) -> None:
    """
    Associate this call (AG or RS) with a "layer instance" using:
      shape_key = (global_elems, dtype)
      occurrence index (k-th AG/RS of that shape)

    stats: {
        "num_packets": int,
        "dropped_packets": int,
        "received_packets": int,
    }

    Once both AG and RS stats are known for that instance, log a row.
    """
    global _NEXT_LAYER_ID

    if stats is None:
        return

    world_size = dist.get_world_size()

    # For FSDP:
    #  - all_gather_into_tensor input: shard => global_elems = shard * world_size
    #  - reduce_scatter_tensor input: full grads => global_elems = input numel
    if fn_name == "all_gather_into_tensor":
        global_elems = t.numel() * world_size
    elif fn_name == "reduce_scatter_tensor":
        global_elems = t.numel()
    else:
        return

    shape_key = (global_elems, t.dtype)

    # Determine occurrence index per shape & collective type
    if fn_name == "all_gather_into_tensor":
        occ = _AG_OCC_COUNT[shape_key]
        _AG_OCC_COUNT[shape_key] += 1
    elif fn_name == "reduce_scatter_tensor":
        occ = _RS_OCC_COUNT[shape_key]
        _RS_OCC_COUNT[shape_key] += 1
    else:
        return

    # Layer instance key is (shape, occurrence index)
    instance_key = (shape_key, occ)

    if instance_key not in _LAYER_INSTANCE_INFO:
        _LAYER_INSTANCE_INFO[instance_key] = {
            "layer_id": _NEXT_LAYER_ID,
            "global_elems": global_elems,
            "dtype": t.dtype,
            "ag_total_packets": None,
            "ag_dropped_packets": None,
            "ag_received_packets": None,
            "rs_total_packets": None,
            "rs_dropped_packets": None,
            "rs_received_packets": None,
            "written": False,
        }
        _NEXT_LAYER_ID += 1

    entry = _LAYER_INSTANCE_INFO[instance_key]

    if fn_name == "all_gather_into_tensor":
        entry["ag_total_packets"] = int(stats["num_packets"])
        entry["ag_dropped_packets"] = int(stats["dropped_packets"])
        entry["ag_received_packets"] = int(stats["received_packets"])
    elif fn_name == "reduce_scatter_tensor":
        entry["rs_total_packets"] = int(stats["num_packets"])
        entry["rs_dropped_packets"] = int(stats["dropped_packets"])
        entry["rs_received_packets"] = int(stats["received_packets"])

    # Don't double-write
    if entry.get("written", False):
        return

    # Once both AG and RS stats are present, write the CSV row and mark written
    if (
        entry["ag_total_packets"] is not None
        and entry["rs_total_packets"] is not None
    ):
        _write_layer_row(entry)
        entry["written"] = True


def _apply_packet_loss_(
    tensor: torch.Tensor,
    loss: LossyNetwork,
    *,
    tag: str = "",
) -> dict:
    """
    Model C: Fractional packet loss per collective.

    For THIS all-gather / reduce-scatter call:
      - Compute num_packets for the tensor.
      - Drop approximately (loss.loss_rate * num_packets) packets,
        chosen uniformly at random without replacement.
      - Return stats:
          {
              "num_packets": int,
              "dropped_packets": int,
              "received_packets": int,
          }
    """
    if not _is_payload_tensor(tensor):
        return {"num_packets": 0, "dropped_packets": 0, "received_packets": 0}

    flat = tensor.view(-1)
    elems_per_packet = max(1, MAX_PAYLOAD_BYTES // flat.element_size())
    num_packets = math.ceil(flat.numel() / elems_per_packet)

    if num_packets == 0:
        return {"num_packets": 0, "dropped_packets": 0, "received_packets": 0}

    device = flat.device if flat.is_cuda else torch.device("cpu")

    # Clamp loss_rate to [0, 1]
    loss_fraction = float(loss.loss_rate)
    loss_fraction = max(0.0, min(1.0, loss_fraction))

    # Determine how many packets to drop (fixed fraction of total)
    drop_count = int(round(loss_fraction * num_packets))
    drop_count = max(0, min(drop_count, num_packets))

    if drop_count == 0:
        # No packets dropped for this collective
        dropped_packets = 0
        received_packets = num_packets
        return {
            "num_packets": num_packets,
            "dropped_packets": dropped_packets,
            "received_packets": received_packets,
        }

    # Sample exactly drop_count packets to drop (without replacement)
    with torch.random.fork_rng(devices=[device] if flat.is_cuda else []):
        if hasattr(loss, "seed"):
            torch.manual_seed(loss.seed)
        drop_indices = torch.randperm(num_packets, device=device)[:drop_count]

    pkt_mask = torch.ones(num_packets, dtype=torch.bool, device=device)
    pkt_mask[drop_indices] = False  # False => drop

    # Expand packet mask to element mask
    elem_mask = pkt_mask.repeat_interleave(elems_per_packet)[: flat.numel()]
    flat.mul_(elem_mask.to(dtype=flat.dtype))

    dropped_packets = int((~pkt_mask).sum().item())
    received_packets = num_packets - dropped_packets

    return {
        "num_packets": num_packets,
        "dropped_packets": dropped_packets,
        "received_packets": received_packets,
    }


def install_lossy_collectives(
    loss: LossyNetwork,
    enable_allgather: bool = True,
    enable_rs: bool = True,
    enable_allreduce: bool = True,
    min_numel: int = 0,
):
    """
    Monkey-patch torch.distributed collectives to inject loss AND
    log per-layer-instance packet counts.

    Call on every rank AFTER init_process_group,
    BEFORE constructing FSDP/HF objects.

    min_numel: if >0, skip tensors smaller than this (avoid control traffic).
    """

    def _wrap(fn_name):
        original = getattr(dist, fn_name)

        def wrapped(*args, **kwargs):
            t0 = time.time()
            try:
                if fn_name == "all_gather_into_tensor" and enable_allgather:
                    # all_gather_into_tensor(output, input, group=...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            stats = _apply_packet_loss_(
                                t,
                                loss,
                                tag="all_gather_into_tensor.input",
                            )
                            _record_packets_instance(fn_name, t, stats)

                elif fn_name == "reduce_scatter_tensor" and enable_rs:
                    # reduce_scatter_tensor(output, input, group=...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            stats = _apply_packet_loss_(
                                t,
                                loss,
                                tag="reduce_scatter_tensor.input",
                            )
                            _record_packets_instance(fn_name, t, stats)

                elif fn_name == "all_reduce" and enable_allreduce:
                    # all_reduce(tensor, group=...)
                    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                        t = args[0]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            _ = _apply_packet_loss_(
                                t,
                                loss,
                                tag="all_reduce.tensor",
                            )

                return original(*args, **kwargs)
            finally:
                dt = (time.time() - t0) * 1000.0
                try:
                    r = int(
                        os.environ.get(
                            "RANK", os.environ.get("LOCAL_RANK", "0")
                        )
                    )
                except Exception:
                    r = 0
                if r == 0:
                    print(f"[INTROSPECT] {fn_name} dt={dt:.2f} ms")

        return wrapped

    for name in ("all_gather_into_tensor", "reduce_scatter_tensor", "all_reduce"):
        if hasattr(dist, name):
            setattr(dist, name, _wrap(name))

