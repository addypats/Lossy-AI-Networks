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

_HIT_ONCE = {"all_gather_into_tensor": False,
             "reduce_scatter_tensor": False,
             "all_reduce": False}



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
        # if not file_exists:
        #     writer.writerow(
        #         [
        #             "Layer",
        #             "LayerShape",
        #             "ag_total_packets",
        #             "ag_dropped_packets",
        #             "ag_received_packets",
        #             "rs_total_packets",
        #             "rs_dropped_packets",
        #             "rs_received_packets",
        #         ]
        #     )
        
        
        # Additional logging - Tensor (gradient) size
        
        if not file_exists:
            writer.writerow(
                [
                    "Layer",
                    "LayerShape",
                    "global_elems",
                    "elem_size_bytes",
                    "global_bytes",
                    "ag_tensor_elems",
                    "ag_tensor_bytes",
                    "ag_total_packets",
                    "ag_dropped_packets",
                    "ag_received_packets",
                    "rs_tensor_elems",
                    "rs_tensor_bytes",
                    "rs_total_packets",
                    "rs_dropped_packets",
                    "rs_received_packets",
                ]
            )


        # layer_shape = _layer_shape_str(entry["global_elems"], entry["dtype"])
        # writer.writerow(
        #     [
        #         entry["layer_id"],
        #         layer_shape,
        #         entry.get("ag_total_packets", 0),
        #         entry.get("ag_dropped_packets", 0),
        #         entry.get("ag_received_packets", 0),
        #         entry.get("rs_total_packets", 0),
        #         entry.get("rs_dropped_packets", 0),
        #         entry.get("rs_received_packets", 0),
        #     ]
        # )
        
        
        # Additional logging - Tensor (gradient) size
        
        layer_shape = _layer_shape_str(entry["global_elems"], entry["dtype"])
        writer.writerow(
            [
                entry["layer_id"],
                layer_shape,
                entry["global_elems"],
                entry["elem_size_bytes"],
                entry["global_bytes"],
                entry.get("ag_tensor_elems", 0),
                entry.get("ag_tensor_bytes", 0),
                entry.get("ag_total_packets", 0),
                entry.get("ag_dropped_packets", 0),
                entry.get("ag_received_packets", 0),
                entry.get("rs_tensor_elems", 0),
                entry.get("rs_tensor_bytes", 0),
                entry.get("rs_total_packets", 0),
                entry.get("rs_dropped_packets", 0),
                entry.get("rs_received_packets", 0),
            ]
        )



def _record_packets_instance(fn_name: str, t: torch.Tensor, stats: dict, *, enable_allgather: bool) -> None:
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

    # if instance_key not in _LAYER_INSTANCE_INFO:
    #     _LAYER_INSTANCE_INFO[instance_key] = {
    #         "layer_id": _NEXT_LAYER_ID,
    #         "global_elems": global_elems,
    #         "dtype": t.dtype,
    #         "ag_total_packets": None,
    #         "ag_dropped_packets": None,
    #         "ag_received_packets": None,
    #         "rs_total_packets": None,
    #         "rs_dropped_packets": None,
    #         "rs_received_packets": None,
    #         "written": False,
    #     }
    #     _NEXT_LAYER_ID += 1
    
    
    # Additional logging - Tensor (gradient) size
    
    if instance_key not in _LAYER_INSTANCE_INFO:
        elem_size_bytes = t.element_size()
        _LAYER_INSTANCE_INFO[instance_key] = {
            "layer_id": _NEXT_LAYER_ID,
            "global_elems": int(global_elems),
            "dtype": t.dtype,
            "elem_size_bytes": int(elem_size_bytes),
            "global_bytes": int(global_elems * elem_size_bytes),

            # AG-specific tensor size on this rank
            "ag_tensor_elems": None,
            "ag_tensor_bytes": None,

            # RS-specific tensor size on this rank
            "rs_tensor_elems": None,
            "rs_tensor_bytes": None,

            # packet stats (existing)
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
        # Additional logging - Tensor (gradient) size
        entry["ag_tensor_elems"]      = int(t.numel())
        entry["ag_tensor_bytes"]      = int(t.numel() * t.element_size())
    elif fn_name == "reduce_scatter_tensor":
        entry["rs_total_packets"] = int(stats["num_packets"])
        entry["rs_dropped_packets"] = int(stats["dropped_packets"])
        entry["rs_received_packets"] = int(stats["received_packets"])
        # Additional logging - Tensor (gradient) size
        entry["rs_tensor_elems"]      = int(t.numel())
        entry["rs_tensor_bytes"]      = int(t.numel() * t.element_size())

    # Don't double-write
    if entry.get("written", False):
        return

    # # Once both AG and RS stats are present, write the CSV row and mark written
    # if (
    #     entry["ag_total_packets"] is not None
    #     and entry["rs_total_packets"] is not None
    # ):
    #     _write_layer_row(entry)
    #     entry["written"] = True
    
    # Write once we have *something meaningful*.
    # - If you run RS-only, you'll still get rows.
    # - If you run AG+RS, it will still only write once (after both appear).
    have_ag = (entry["ag_total_packets"] is not None)
    have_rs = (entry["rs_total_packets"] is not None)

    if have_rs and (not enable_allgather or have_ag):
        _write_layer_row(entry)
        entry["written"] = True



def _apply_packet_loss_(
    tensor: torch.Tensor,
    loss: LossyNetwork,
    *,
    tag: str = "",
) -> dict:
    """
    Generic packet-loss application using the LossyNetwork-style interface.

    - loss.send(tensor)   -> packet mask (True = kept, False = dropped)
    - loss.receive(..)    -> applies that mask (zeroing dropped packets)

    Works for:
      - Bernoulli LossyNetwork (uses loss_rate)
      - GillbertElliotLossyNetwork (uses GE state machine)
      - Any subclass that implements send/receive.
    """
    if not _is_payload_tensor(tensor):
        return {"num_packets": 0, "dropped_packets": 0, "received_packets": 0}

    flat = tensor.view(-1)

    # Use the loss model's packet generator
    pkt_mask = loss.send(flat)
    if pkt_mask is None:
        # fall back: no loss applied
        return {"num_packets": 0, "dropped_packets": 0, "received_packets": 0}

    # Apply loss (zero out dropped packets) in-place
    loss.receive(flat, pkt_mask)

    num_packets = int(len(pkt_mask))
    dropped_packets = int((~pkt_mask).sum().item())
    received_packets = num_packets - dropped_packets

    return {
        "num_packets": num_packets,
        "dropped_packets": dropped_packets,
        "received_packets": received_packets,
    }


# def install_lossy_collectives(
#     loss: LossyNetwork,
#     enable_allgather: bool = True,
#     enable_rs: bool = True,
#     enable_allreduce: bool = True,
#     min_numel: int = 0,
# ):

# """
#     Monkey-patch torch.distributed collectives to inject loss AND
#     log per-layer-instance packet counts.

#     Call on every rank AFTER init_process_group,
#     BEFORE constructing FSDP/HF objects.

#     min_numel: if >0, skip tensors smaller than this (avoid control traffic).
#     """

# for rank aware strategy with input gpu getting loss for dist training on multiple instances

# def install_lossy_collectives(
#     loss: LossyNetwork,
#     enable_allgather: bool = True,
#     enable_rs: bool = True,
#     enable_allreduce: bool = True,
#     min_numel: int = 0,
#     num_nodes: int = 1,
# ):
#     """
#     Monkey-patch torch.distributed collectives to inject loss AND
#     log per-layer-instance packet counts.

#     2A-style rank-aware behavior:
#     - Divide world_size into `num_nodes` blocks.
#     - For each node, designate the *first* rank in that block as the
#       logical "input rank".
#     - Only that input rank applies loss before the collective.
#     """    

#     def _wrap(fn_name):
#         original = getattr(dist, fn_name)

#         # def wrapped(*args, **kwargs):
#         #     t0 = time.time()
#         #     try:
#         #         if fn_name == "all_gather_into_tensor" and enable_allgather:
#         #             # all_gather_into_tensor(output, input, group=...)
#         #             if len(args) >= 2 and isinstance(args[1], torch.Tensor):
#         #                 t = args[1]
#         #                 if _is_payload_tensor(t) and t.numel() >= min_numel:
#         #                     stats = _apply_packet_loss_(
#         #                         t,
#         #                         loss,
#         #                         tag="all_gather_into_tensor.input",
#         #                     )
#         #                     _record_packets_instance(fn_name, t, stats)

#         #         elif fn_name == "reduce_scatter_tensor" and enable_rs:
#         #             # reduce_scatter_tensor(output, input, group=...)
#         #             if len(args) >= 2 and isinstance(args[1], torch.Tensor):
#         #                 t = args[1]
#         #                 if _is_payload_tensor(t) and t.numel() >= min_numel:
#         #                     stats = _apply_packet_loss_(
#         #                         t,
#         #                         loss,
#         #                         tag="reduce_scatter_tensor.input",
#         #                     )
#         #                     _record_packets_instance(fn_name, t, stats)

#         #         elif fn_name == "all_reduce" and enable_allreduce:
#         #             # all_reduce(tensor, group=...)
#         #             if len(args) >= 1 and isinstance(args[0], torch.Tensor):
#         #                 t = args[0]
#         #                 if _is_payload_tensor(t) and t.numel() >= min_numel:
#         #                     _ = _apply_packet_loss_(
#         #                         t,
#         #                         loss,
#         #                         tag="all_reduce.tensor",
#         #                     )

#         #         return original(*args, **kwargs)
#         #     finally:
#         #         dt = (time.time() - t0) * 1000.0
#         #         try:
#         #             r = int(
#         #                 os.environ.get(
#         #                     "RANK", os.environ.get("LOCAL_RANK", "0")
#         #                 )
#         #             )
#         #         except Exception:
#         #             r = 0
#         #         if r == 0:
#         #             print(f"[INTROSPECT] {fn_name} dt={dt:.2f} ms")
        
#         # for rank aware strategy with input gpu getting loss for dist training on multiple instances

#         def wrapped(*args, **kwargs):
#             t0 = time.time()
#             try:
#                 # ---- 2A: determine node_id and input rank for this rank ----
#                 try:
#                     if dist.is_available() and dist.is_initialized():
#                         rank = dist.get_rank()
#                         world_size = dist.get_world_size()
#                     else:
#                         rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
#                         world_size = int(os.environ.get("WORLD_SIZE", "1"))
#                 except Exception:
#                     rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
#                     world_size = int(os.environ.get("WORLD_SIZE", "1"))
                
#                 ###############################################################################    
#                 # Bump per-process lossy call counter (useful for debugging)
#                 try:
#                     c = int(os.environ.get("LOSSY_CALL_COUNTER", "0"))
#                 except Exception:
#                     c = 0
#                 os.environ["LOSSY_CALL_COUNTER"] = str(c + 1)
                
#                 ###############################################################################


#                 gpus_per_node = max(1, world_size // max(1, num_nodes))
#                 node_id = rank // gpus_per_node
#                 input_rank = node_id * gpus_per_node
                
#                 ###############################################################################
#                 # One-time topology print per rank
#                 if not hasattr(wrapped, "_printed_topology"):
#                     import socket
#                     host = socket.gethostname()
#                     print(
#                         f"[TOPO] host={host} rank={rank}/{world_size} "
#                         f"num_nodes={num_nodes} gpus_per_node={gpus_per_node} "
#                         f"node_id={node_id} input_rank={input_rank} "
#                         f"inject={(rank==input_rank)}",
#                         flush=True
#                     )
#                     wrapped._printed_topology = True
                    
#                 ###############################################################################
                
#                 # ------------------------------------------------------------

#                 # ---- SAFE LOSS INJECTION: never kill the rank on error ----
#                 try:
#                     if fn_name == "all_gather_into_tensor" and enable_allgather:
#                         # all_gather_into_tensor(output, input, group=...)
#                         if len(args) >= 2 and isinstance(args[1], torch.Tensor):
#                             t = args[1]
#                             if (
#                                 _is_payload_tensor(t)
#                                 and t.numel() >= min_numel
#                                 and rank == input_rank   # 2A: boundary rank per node
#                             ):
#                                 stats = _apply_packet_loss_(
#                                     t,
#                                     loss,
#                                     tag="all_gather_into_tensor.input",
#                                 )
#                                 _record_packets_instance(fn_name, t, stats)

#                     elif fn_name == "reduce_scatter_tensor" and enable_rs:
#                         # reduce_scatter_tensor(output, input, group=...)
#                         if len(args) >= 2 and isinstance(args[1], torch.Tensor):
#                             t = args[1]
#                             if (
#                                 _is_payload_tensor(t)
#                                 and t.numel() >= min_numel
#                                 and rank == input_rank   # 2A
#                             ):
#                                 stats = _apply_packet_loss_(
#                                     t,
#                                     loss,
#                                     tag="reduce_scatter_tensor.input",
#                                 )
#                                 _record_packets_instance(fn_name, t, stats)

#                     elif fn_name == "all_reduce" and enable_allreduce:
#                         # all_reduce(tensor, group=...)
#                         if len(args) >= 1 and isinstance(args[0], torch.Tensor):
#                             t = args[0]
#                             if (
#                                 _is_payload_tensor(t)
#                                 and t.numel() >= min_numel
#                                 and rank == input_rank   # 2A
#                             ):
#                                 _ = _apply_packet_loss_(
#                                     t,
#                                     loss,
#                                     tag="all_reduce.tensor",
#                                 )

#                 except Exception as e:
#                     # IMPORTANT: log and fall back to *no-loss* for this call
#                     # instead of crashing this rank
#                     import traceback
#                     print(
#                         f"[LOSSY][ERROR] rank={rank} fn={fn_name}: "
#                         f"{type(e).__name__}: {e}",
#                         flush=True,
#                     )
#                     traceback.print_exc()

#                 # Always call the real collective on all ranks
#                 return original(*args, **kwargs)

#             finally:
#                 dt = (time.time() - t0) * 1000.0
#                 try:
#                     r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
#                 except Exception:
#                     r = 0
#                 # if r == 0:
#                     # print(f"[INTROSPECT] {fn_name} dt={dt:.2f} ms")

#         return wrapped
        

#     for name in ("all_gather_into_tensor", "reduce_scatter_tensor", "all_reduce"):
#         if hasattr(dist, name):
#             setattr(dist, name, _wrap(name))


def install_lossy_collectives(
    loss: LossyNetwork,
    enable_allgather: bool = True,
    enable_rs: bool = True,
    enable_allreduce: bool = True,
    min_numel: int = 0,
    num_nodes: int = 1,
):
    """
    Monkey-patch torch.distributed collectives to inject loss + (optional) log stats.

    Rank-aware behavior (logical nodes):
      - Partition world_size into `num_nodes` blocks of size gpus_per_node.
      - Designate the *first* rank in each block as the "boundary" (input_rank).
      - Only boundary ranks inject loss (emulates loss only on inter-node edges).

    IMPORTANT:
      - If num_nodes <= 1, we disable injection (no "inter-node" network).
      - LOSSY_GLOBAL_STEP is provided by your HF callback.
      - We also increment LOSSY_CALL_COUNTER per collective call for stable per-call seeding/debug.
    """

    import socket

    def _get_rank_world():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        # fallback for early debug
        r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        w = int(os.environ.get("WORLD_SIZE", "1"))
        return r, w

    def _logical_topology(rank: int, world_size: int, num_nodes: int):
        # If user says 1 node, treat as single node (no boundary injection)
        if num_nodes is None or num_nodes <= 1:
            gpn = world_size
            node_id = 0
            input_rank = 0
            return gpn, node_id, input_rank

        # Assume num_nodes divides world_size in your experiments (8 and {1,2,4,8})
        # If it doesn't, last "node" may be smaller; we still compute a sane mapping.
        gpn = max(1, world_size // num_nodes)
        if gpn * num_nodes != world_size:
            # uneven partition: fall back to ceil partitioning
            gpn = (world_size + num_nodes - 1) // num_nodes

        node_id = rank // gpn
        if node_id >= num_nodes:
            node_id = num_nodes - 1

        input_rank = node_id * gpn
        if input_rank >= world_size:
            input_rank = max(0, world_size - 1)
        return gpn, node_id, input_rank

    def _bump_call_counter():
        try:
            c = int(os.environ.get("LOSSY_CALL_COUNTER", "0"))
        except Exception:
            c = 0
        os.environ["LOSSY_CALL_COUNTER"] = str(c + 1)
        return c + 1

    def _should_touch_tensor(t: torch.Tensor) -> bool:
        return _is_payload_tensor(t) and (t.numel() >= int(min_numel))

    def _wrap(fn_name: str):
        original = getattr(dist, fn_name)

        def wrapped(*args, **kwargs):
            t0 = time.time()
            rank, world_size = _get_rank_world()
            gpn, node_id, input_rank = _logical_topology(rank, world_size, num_nodes)
            
            # One-time "wrapper hit" print (rank 0 only)
            if rank == 0 and not _HIT_ONCE.get(fn_name, False):
                _HIT_ONCE[fn_name] = True
                print(f"[HIT] wrapper={fn_name} first_seen call_counter={os.environ.get('LOSSY_CALL_COUNTER')} "
                    f"enable_ag={enable_allgather} enable_rs={enable_rs} enable_ar={enable_allreduce}",
                    flush=True)


            # One-time topology print per rank (helps validate 8x1 / 4x2 / 2x4 emulation)
            if not hasattr(wrapped, "_printed_topo"):
                host = socket.gethostname()
                print(
                    f"[TOPO] host={host} rank={rank}/{world_size} "
                    f"num_nodes={num_nodes} gpus_per_node={gpn} node_id={node_id} "
                    f"input_rank={input_rank} inject={(rank == input_rank)}",
                    flush=True
                )
                wrapped._printed_topo = True

            # Always bump call counter (even if we don't inject) for debugging correlation
            _bump_call_counter()

            # If user wants "no inter-node loss", skip injection
            if num_nodes is None or num_nodes <= 1:
                return original(*args, **kwargs)

            # Only boundary ranks inject (your intended model)
            inject_here = (rank == input_rank)

            try:
                if fn_name == "all_gather_into_tensor" and enable_allgather and inject_here:
                    # all_gather_into_tensor(output, input, group=...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _should_touch_tensor(t):
                            stats = _apply_packet_loss_(t, loss, tag="ag.input")
                            _record_packets_instance(fn_name, t, stats, enable_allgather=enable_allgather)

                elif fn_name == "reduce_scatter_tensor" and enable_rs and inject_here:
                    # reduce_scatter_tensor(output, input, group=...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _should_touch_tensor(t):
                            stats = _apply_packet_loss_(t, loss, tag="rs.input")
                            _record_packets_instance(fn_name, t, stats, enable_allgather=enable_allgather)

                elif fn_name == "all_reduce" and enable_allreduce and inject_here:
                    # all_reduce(tensor, group=...)
                    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                        t = args[0]
                        if _should_touch_tensor(t):
                            _ = _apply_packet_loss_(t, loss, tag="ar.tensor")

            except Exception as e:
                # Never crash training due to the lossy wrapper.
                import traceback
                print(
                    f"[LOSSY][ERROR] rank={rank} fn={fn_name}: {type(e).__name__}: {e}",
                    flush=True
                )
                traceback.print_exc()

            # Always call the real collective on all ranks
            out = original(*args, **kwargs)

            # Optional timing print (kept off by default)
            # dt = (time.time() - t0) * 1000.0
            # if rank == 0:
            #     print(f"[INTROSPECT] {fn_name} dt={dt:.2f} ms", flush=True)

            return out

        return wrapped

    # Patch the collectives you use
    for name in ("all_gather_into_tensor", "reduce_scatter_tensor", "all_reduce"):
        if hasattr(dist, name):
            setattr(dist, name, _wrap(name))

