# lossy_patch.py
import os, time, math, torch
import torch.distributed as dist
from comms import LossyNetwork, MAX_PAYLOAD_BYTES

FLOAT_DTYPES = (
    torch.float16, torch.bfloat16, torch.float32, torch.float64
)

def _is_payload_tensor(t: torch.Tensor) -> bool:
    # Only touch floating tensors; skip ints/bools used by control paths.
    return isinstance(t, torch.Tensor) and t.numel() > 0 and t.dtype in FLOAT_DTYPES

def _apply_packet_loss_(tensor: torch.Tensor, loss: LossyNetwork, *, tag: str = "") -> None:
    if not _is_payload_tensor(tensor):
        return

    flat = tensor.view(-1)
    elems_per_packet = max(1, MAX_PAYLOAD_BYTES // flat.element_size())
    num_packets = math.ceil(flat.numel() / elems_per_packet)

    device = flat.device if flat.is_cuda else torch.device("cpu")
    with torch.random.fork_rng(devices=[device] if flat.is_cuda else []):
        if hasattr(loss, "seed"):
            torch.manual_seed(loss.seed)
        pkt_mask = (torch.rand(num_packets, device=device) > loss.loss_rate)

    elem_mask = pkt_mask.repeat_interleave(elems_per_packet)[: flat.numel()]

    # Optional lightweight sanity log (safe on floats only)
    # norm_before = float(torch.linalg.vector_norm(flat).item())

    flat.mul_(elem_mask.to(dtype=flat.dtype))

    # norm_after = float(torch.linalg.vector_norm(flat).item())
    # if int(os.environ.get("RANK", "0")) == 0:
    #     print(f"[LOSS] {tag} dtype={tensor.dtype} numel={tensor.numel()} "
    #           f"drop_rate~={loss.loss_rate:.3f}")

def install_lossy_collectives(loss: LossyNetwork,
                              enable_allgather=True,
                              enable_rs=True,
                              enable_allreduce=True,
                              min_numel: int = 0):
    """
    Monkey-patch torch.distributed collectives to inject loss.
    Call on every rank BEFORE constructing FSDP/HF objects.
    min_numel: if >0, skip tensors smaller than this (avoid control traffic).
    """
    def _wrap(fn_name):
        original = getattr(dist, fn_name)

        def wrapped(*args, **kwargs):
            t0 = time.time()
            try:
                if fn_name == "all_gather_into_tensor" and enable_allgather:
                    # (output, input, ...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            _apply_packet_loss_(t, loss, tag="all_gather_into_tensor.input")

                elif fn_name == "reduce_scatter_tensor" and enable_rs:
                    # (output, input, ...)
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        t = args[1]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            _apply_packet_loss_(t, loss, tag="reduce_scatter_tensor.input")

                elif fn_name == "all_reduce" and enable_allreduce:
                    # (tensor, ...)
                    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                        t = args[0]
                        if _is_payload_tensor(t) and t.numel() >= min_numel:
                            _apply_packet_loss_(t, loss, tag="all_reduce.tensor")

                return original(*args, **kwargs)
            finally:
                dt = (time.time() - t0) * 1000.0
                try:
                    r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                except Exception:
                    r = 0
                if r == 0:
                    print(f"[INTROSPECT] {fn_name} dt={dt:.2f} ms")
        return wrapped

    for name in ("all_gather_into_tensor", "reduce_scatter_tensor", "all_reduce"):
        if hasattr(dist, name):
            setattr(dist, name, _wrap(name))

