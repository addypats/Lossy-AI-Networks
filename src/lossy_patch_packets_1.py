# lossy_patch.py

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

# --- Per-layer-instance packet logging state ---
#
# We treat each (global_elems, dtype, occurrence_index) as a distinct "layer instance".
# occurrence_index is the k-th time we see an AG or RS of that shape.
#
# For each instance_key we store:
#   { layer_id, global_elems, dtype, ag_packets, rs_packets, written }
#

_LAYER_INSTANCE_INFO = {}
_NEXT_LAYER_ID = 0

# Per-shape occurrence counters for AG and RS
_AG_OCC_COUNT = defaultdict(int)   # (global_elems, dtype) -> count
_RS_OCC_COUNT = defaultdict(int)   # (global_elems, dtype) -> count

# CSV path (override with env if desired)
_PACKET_LOG_PATH = os.environ.get("PACKET_LOG_PATH", "fsdp_packet_log.csv")


def _is_payload_tensor(t: torch.Tensor) -> bool:
    """Only touch non-empty floating tensors with float dtypes."""
    return (
        isinstance(t, torch.Tensor)
        and t.numel() > 0
        and t.dtype in FLOAT_DTYPES
    )


def _num_packets_for_tensor(t: torch.Tensor) -> int:
    """
    Compute logical packet count for this tensor based on MAX_PAYLOAD_BYTES.
    This is your "network packet" abstraction.
    """
    flat = t.view(-1)
    elems_per_packet = max(1, MAX_PAYLOAD_BYTES // flat.element_size())
    num_packets = math.ceil(flat.numel() / elems_per_packet)
    return num_packets


def _layer_shape_str(global_elems: int, dtype: torch.dtype) -> str:
    """
    Human-readable layer shape string for the CSV.
    We only know total element count and dtype here (not full tensor dims).
    """
    return f"global_elems={global_elems}, dtype={str(dtype)}"


def _write_layer_row(entry: dict) -> None:
    """
    Append a single row for this layer instance to the CSV:
    Layer, LayerShape, all_gather_packets, reduce_scatter_packets

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
                ["Layer", "LayerShape", "all_gather_packets", "reduce_scatter_packets"]
            )

        layer_shape = _layer_shape_str(entry["global_elems"], entry["dtype"])
        writer.writerow(
            [
                entry["layer_id"],
                layer_shape,
                entry.get("ag_packets", None),
                entry.get("rs_packets", None),
            ]
        )


def _record_packets_instance(fn_name: str, t: torch.Tensor, num_packets: int) -> None:
    """
    Associate this call (AG or RS) with a "layer instance" using:
      shape_key = (global_elems, dtype)
      occurrence index (k-th AG/RS of that shape)

    Then record packet counts. Once both AG and RS are known for that instance,
    log a row.
    """
    global _NEXT_LAYER_ID

    world_size = dist.get_world_size()

    # Compute global_elems for FSDP:
    # - all_gather_into_tensor input: shard => global_elems = shard * world_size
    # - reduce_scatter_tensor input: full grads => global_elems = input numel
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
            "ag_packets": None,
            "rs_packets": None,
            "written": False,
        }
        _NEXT_LAYER_ID += 1

    entry = _LAYER_INSTANCE_INFO[instance_key]

    if fn_name == "all_gather_into_tensor":
        # Only set if not already set (first time we see this instance's AG)
        if entry["ag_packets"] is None:
            entry["ag_packets"] = num_packets
    elif fn_name == "reduce_scatter_tensor":
        if entry["rs_packets"] is None:
            entry["rs_packets"] = num_packets

    # Don't double-write
    if entry.get("written", False):
        return

    # Once both are present, write the CSV row and mark written
    if entry["ag_packets"] is not None and entry["rs_packets"] is not None:
        _write_layer_row(entry)
        entry["written"] = True


def _apply_packet_loss_(
    tensor: torch.Tensor,
    loss: LossyNetwork,
    *,
    tag: str = "",
) -> int:
    """
    Apply Bernoulli packet loss in-place and return the number of logical packets.
    """
    if not _is_payload_tensor(tensor):
        return 0

    flat = tensor.view(-1)
    elems_per_packet = max(1, MAX_PAYLOAD_BYTES // flat.element_size())
    num_packets = math.ceil(flat.numel() / elems_per_packet)

    device = flat.device if flat.is_cuda else torch.device("cpu")

    # Reproducible RNG (optional)
    with torch.random.fork_rng(devices=[device] if flat.is_cuda else []):
        if hasattr(loss, "seed"):
            torch.manual_seed(loss.seed)
        pkt_mask = (
            torch.rand(num_packets, device=device) > loss.loss_rate
        )

    elem_mask = pkt_mask.repeat_interleave(elems_per_packet)[: flat.numel()]
    flat.mul_(elem_mask.to(dtype=flat.dtype))

    # Optional debug:
    # if int(os.environ.get("RANK", "0")) == 0:
    #     print(
    #         f"[LOSS] {tag} dtype={tensor.dtype} numel={tensor.numel()} "
    #         f"num_packets={num_packets} drop_rate~={loss.loss_rate:.3f}"
    #     )

    return num_packets


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
                            num_packets = _apply_packet_loss_(
                                t,
                                loss,
                                tag="all_gather_into_tensor.input",
                            )
                            if num_packets > 0:
                                _record_packets_instance(fn_name, t, num_packets)

                elif fn_name == "reduce_scatter_tensor" and enable_rs:
                    # reduce_scatter_tensor(output, input, group=...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            num_packets = _apply_packet_loss_(
                                t,
                                loss,
                                tag="reduce_scatter_tensor.input",
                            )
                            if num_packets > 0:
                                _record_packets_instance(fn_name, t, num_packets)

                elif fn_name == "all_reduce" and enable_allreduce:
                    # all_reduce(tensor, group=...)
                    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                        t = args[0]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            _apply_packet_loss_(
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

