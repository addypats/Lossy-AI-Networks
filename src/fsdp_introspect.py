# fsdp_introspect.py
# Usage in your script (before Trainer()):
#   import fsdp_introspect
#   fsdp_introspect.enable(model, optimizer, log_dir="logs/fsdp_probe")
#   ...
#   trainer.train()
#   fsdp_introspect.finalize()  # optional: writes sorted files once more at the end

import os, csv, json, time, functools, threading
from collections import deque

import torch
import torch.distributed as dist

# Optional internals (available on recent PyTorch)
try:
    from torch.distributed.fsdp._flat_param import FlatParamHandle  # PyTorch 2.2+
except Exception:
    FlatParamHandle = None
try:
    from torch.distributed.fsdp.flat_param import FlatParameter
except Exception:
    FlatParameter = None

# ---------------------------
# Config & small utilities
# ---------------------------
_EVENT_ORDER = {
    "init": 0,
    "all_gather": 1,
    "reduce_scatter": 2,
    "optim_step_pre": 3,
    "optim_step_post": 4,
}

def _rk():
    try: return dist.get_rank()
    except Exception: return -1

def _ws():
    try: return dist.get_world_size()
    except Exception: return 1

def _bs(t): return t.element_size() * t.numel()
def _fmt_shape(t): return tuple(t.shape) if isinstance(t, torch.Tensor) else ()

def _flatparam_like(p):
    if FlatParameter is not None and isinstance(p, FlatParameter):
        return True
    return hasattr(p, "_is_sharded") or hasattr(p, "_fsdp_flattened")

# ---------------------------
# Structured log buffer
# ---------------------------
class _StructuredLogger:
    def __init__(self, log_dir=None, flush_every=64):
        self.rank = _rk()
        self.log_dir = log_dir or os.environ.get("FSDP_INTROSPECT_DIR", "fsdp_probe_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Filenames (per-rank)
        self.jsonl_path = os.path.join(self.log_dir, f"fsdp_introspect_rank{self.rank}.jsonl")
        self.csv_path   = os.path.join(self.log_dir, f"fsdp_introspect_rank{self.rank}.csv")

        # In-memory buffer
        self.buf = deque()
        self.lock = threading.Lock()
        self.event_id = 0
        self.flush_every = flush_every

        # CSV header
        self.csv_header = [
            "event_id","ts_ms","rank","world_size","event","global_step",
            "numel","bytes","dtype","shape","duration_ms"
        ]
        # Initialize CSV file with header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_header)

        # Prepare JSONL file (touch)
        if not os.path.exists(self.jsonl_path):
            open(self.jsonl_path, "a").close()

        # Accessor for global_step (optional)
        self._global_step_fn = None

    def set_global_step_accessor(self, fn):
        self._global_step_fn = fn

    def _global_step(self):
        if self._global_step_fn is None:
            return None
        try:
            return int(self._global_step_fn())
        except Exception:
            return None

    def log(self, event, *, numel=None, bytes_=None, dtype=None, shape=None, duration_ms=None):
        with self.lock:
            self.event_id += 1
            rec = {
                "event_id": self.event_id,
                "ts_ms": int(time.time() * 1000),
                "rank": self.rank,
                "world_size": _ws(),
                "event": str(event),
                "global_step": self._global_step(),
                "numel": int(numel) if numel is not None else None,
                "bytes": int(bytes_) if bytes_ is not None else None,
                "dtype": str(dtype) if dtype is not None else None,
                "shape": list(shape) if shape is not None else None,
                "duration_ms": float(duration_ms) if duration_ms is not None else None,
            }
            self.buf.append(rec)
            if len(self.buf) >= self.flush_every:
                self._flush_locked()

    def _flush_locked(self):
        if not self.buf:
            return
        # copy out
        batch = list(self.buf)
        self.buf.clear()

        # Append to JSONL
        with open(self.jsonl_path, "a") as jf:
            for r in batch:
                jf.write(json.dumps(r) + "\n")

        # Append to CSV
        with open(self.csv_path, "a", newline="") as cf:
            writer = csv.writer(cf)
            for r in batch:
                writer.writerow([
                    r["event_id"], r["ts_ms"], r["rank"], r["world_size"], r["event"],
                    r["global_step"], r["numel"], r["bytes"], r["dtype"], r["shape"], r["duration_ms"]
                ])

    def flush(self):
        with self.lock:
            self._flush_locked()

    # Sorting writer (per-rank)
    def write_sorted(self):
        # Load current JSONL
        if not os.path.exists(self.jsonl_path):
            return
        records = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

        def key_fn(r):
            gs = (r.get("global_step"),) if r.get("global_step") is not None else (float("inf"),)
            eo = (_EVENT_ORDER.get(r.get("event"), 999),)
            ei = (r.get("event_id", 0),)
            return gs + eo + ei

        records.sort(key=key_fn)

        # Write sorted files
        base = os.path.splitext(self.jsonl_path)[0]
        jsonl_sorted = base + "_sorted.jsonl"
        csv_sorted   = base + "_sorted.csv"

        with open(jsonl_sorted, "w") as jf:
            for r in records:
                jf.write(json.dumps(r) + "\n")

        with open(csv_sorted, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(self.csv_header)
            for r in records:
                writer.writerow([
                    r.get("event_id"), r.get("ts_ms"), r.get("rank"), r.get("world_size"),
                    r.get("event"), r.get("global_step"), r.get("numel"), r.get("bytes"),
                    r.get("dtype"), r.get("shape"), r.get("duration_ms"),
                ])

_logger = _StructuredLogger()  # created now; reconfigured in enable()

# ---------------------------
# Helpers to compute sizes
# ---------------------------
def _sum_optim_state_bytes(optim):
    total = 0
    for pg in optim.param_groups if optim else []:
        for p in pg.get("params", []):
            st = optim.state.get(p, {})
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += _bs(v)
    return total

# ---------------------------
# Hooks/Wrappers
# ---------------------------
def _wrap_fsdp_collectives():
    if FlatParamHandle is None:
        return

    if hasattr(FlatParamHandle, "_all_gather_flat_param"):
        orig_ag = FlatParamHandle._all_gather_flat_param
        @functools.wraps(orig_ag)
        def ag_wrapper(self, *args, **kwargs):
            t0 = time.time()
            out = orig_ag(self, *args, **kwargs)
            dt = (time.time() - t0) * 1000.0
            # Best-effort payload size
            numel = int(getattr(self, "_unpadded_unsharded_flat_param_numel", -1))
            _logger.log(
                "all_gather",
                numel=numel,
                bytes_=None if numel < 0 else numel * torch.finfo(torch.float32).bits // 8,  # approx
                dtype="flatparam",
                shape=None,
                duration_ms=dt,
            )
            return out
        FlatParamHandle._all_gather_flat_param = ag_wrapper

    if hasattr(FlatParamHandle, "_reduce_scatter_grads"):
        orig_rs = FlatParamHandle._reduce_scatter_grads
        @functools.wraps(orig_rs)
        def rs_wrapper(self, *args, **kwargs):
            t0 = time.time()
            out = orig_rs(self, *args, **kwargs)
            dt = (time.time() - t0) * 1000.0
            numel = int(getattr(self, "_unpadded_unsharded_flat_param_numel", -1))
            _logger.log(
                "reduce_scatter",
                numel=numel,
                bytes_=None,
                dtype="flatgrad",
                shape=None,
                duration_ms=dt,
            )
            return out
        FlatParamHandle._reduce_scatter_grads = rs_wrapper

def _wrap_dist_collectives_verbose():
    if not dist.is_available():
        return

    def wrap(name):
        if not hasattr(dist, name):
            return
        original = getattr(dist, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            # meta
            numel = None; bytes_ = None; dtype = None; shape = None
            try:
                if name == "all_gather_into_tensor":
                    out, inp = args[0], args[1]
                    numel = int(inp.numel())
                    bytes_ = _bs(inp)
                    dtype  = str(inp.dtype)
                    shape  = list(inp.shape)
                elif name == "reduce_scatter_tensor":
                    out, inp = args[0], args[1]
                    # inp is a flattened concat; still OK to log
                    numel = int(inp.numel())
                    bytes_ = _bs(inp)
                    dtype  = str(inp.dtype)
                    shape  = list(inp.shape)
                elif name in ("all_reduce", "broadcast") and args and isinstance(args[0], torch.Tensor):
                    t = args[0]
                    numel = int(t.numel()); bytes_=_bs(t); dtype=str(t.dtype); shape=list(t.shape)
            except Exception:
                pass

            res = original(*args, **kwargs)
            dt = (time.time() - t0) * 1000.0
            _logger.log(name, numel=numel, bytes_=bytes_, dtype=dtype, shape=shape, duration_ms=dt)
            return res

        setattr(dist, name, wrapper)
    def _wrap_any_reduce_scatter():
        if not dist.is_available():
            return
        # also catch post-backward reduce-scatter variant
        for fn in dir(dist):
            if "reduce_scatter" in fn and callable(getattr(dist, fn)):
                orig = getattr(dist, fn)
                def wrapper(*args, **kwargs):
                    t0 = time.time()
                    out = orig(*args, **kwargs)
                    dt = (time.time() - t0) * 1000
                    print(f"[INTROSPECT] rank={_rk()} {fn} called, {dt:.2f} ms")
                    _logger.log(fn, duration_ms=dt)
                    return out
                setattr(dist, fn, wrapper)
    _wrap_any_reduce_scatter()


def _one_time_shard_summary(model, optimizer=None):
    # per-rank shard summary
    p_local_bytes, p_local_cnt = 0, 0
    for p in model.parameters():
        p_local_cnt += 1
        p_local_bytes += _bs(p)

    _logger.log(
        "init",
        numel=p_local_cnt,
        bytes_=p_local_bytes,
        dtype="params_local",
        shape=None,
        duration_ms=0.0,
    )

    # grads (likely None at init)
    g_bytes, g_cnt = 0, 0
    for p in model.parameters():
        if p.grad is not None:
            g_cnt += 1
            g_bytes += _bs(p.grad)
    _logger.log(
        "init",
        numel=g_cnt,
        bytes_=g_bytes,
        dtype="grads_local",
        shape=None,
        duration_ms=0.0,
    )

    # optimizer states (local)
    if optimizer is not None:
        _logger.log(
            "init",
            numel=None,
            bytes_=_sum_optim_state_bytes(optimizer),
            dtype="optim_state_local",
            shape=None,
            duration_ms=0.0,
        )

def _wrap_optimizer(optimizer):
    if optimizer is None:
        return
    orig_step = optimizer.step
    @functools.wraps(orig_step)
    def step_wrapper(*args, **kwargs):
        _logger.log("optim_step_pre", bytes_=_sum_optim_state_bytes(optimizer))
        out = orig_step(*args, **kwargs)
        _logger.log("optim_step_post", bytes_=_sum_optim_state_bytes(optimizer))
        return out
    optimizer.step = step_wrapper

# ---------------------------
# Public API
# ---------------------------
def enable(model=None, optimizer=None, log_dir=None, flush_every=64, global_step_fn=None):
    """
    Enable structured logging of FSDP comms/events.
    Args:
      model, optimizer: if available.
      log_dir: directory for per-rank logs (default: env FSDP_INTROSPECT_DIR or ./fsdp_probe_logs)
      flush_every: buffer flush cadence to disk
      global_step_fn: optional callable returning an int (e.g., lambda: trainer.state.global_step)
    """
    # Reconfigure logger
    global _logger
    _logger = _StructuredLogger(log_dir=log_dir, flush_every=flush_every)
    if global_step_fn is not None:
        _logger.set_global_step_accessor(global_step_fn)

    # Banner
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        print(f"[FSDP-INTROSPECT] rank={_rk()} world_size={_ws()} backend={backend} log_dir={_logger.log_dir}")
    else:
        print(f"[FSDP-INTROSPECT] dist not initialized yet; rank={_rk()} log_dir={_logger.log_dir}")

    _wrap_fsdp_collectives()
    _wrap_dist_collectives_verbose()
    _wrap_optimizer(optimizer)

    if model is not None:
        _one_time_shard_summary(model, optimizer)
        _logger.flush()

def attach_trainer(trainer):
    """
    Optional: attach HF Trainer to get accurate global_step in logs.
    Call right after you construct Trainer.
    """
    try:
        _logger.set_global_step_accessor(lambda: trainer.state.global_step)
    except Exception:
        pass

def flush():
    _logger.flush()

def finalize(sort_and_write=True):
    """
    Flush buffers and (optionally) write sorted per-rank files:
      * fsdp_introspect_rank{R}_sorted.jsonl
      * fsdp_introspect_rank{R}_sorted.csv
    """
    _logger.flush()
    if sort_and_write:
        _logger.write_sorted()

