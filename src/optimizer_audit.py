import csv
import json
import math
import os
import threading
import time
import types
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.distributed as dist


def _rk() -> int:
    try:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def _ws() -> int:
    try:
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    except Exception:
        return 1


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)
    except Exception:
        return default


def _tensor_sample(tensor: Optional[torch.Tensor], sample_size: int) -> Optional[list]:
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    count = min(sample_size, flat.numel())
    return flat[:count].float().cpu().tolist()


def _tensor_norm(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None:
        return None
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def _safe_clone(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone()


def _group_meta(group: dict) -> dict:
    keys = (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "momentum",
        "dampening",
        "nesterov",
        "maximize",
        "foreach",
        "capturable",
        "fused",
    )
    meta = {}
    for key in keys:
        if key in group:
            value = group.get(key)
            if isinstance(value, tuple):
                meta[key] = list(value)
            else:
                meta[key] = value
    return meta


class OptimizerAuditLogger:
    def __init__(
        self,
        *,
        log_root: Optional[str],
        run_id: str,
        optimizer_name: str,
        sample_size: int,
        global_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.rank = _rk()
        self.world_size = _ws()
        self.run_id = run_id
        self.optimizer_name = optimizer_name
        self.sample_size = sample_size
        self._global_step_fn = global_step_fn
        self._lock = threading.Lock()

        root = log_root or os.environ.get("SANITY_CHECK_LOGS", ".")
        self.log_dir = os.path.join(root, f"{run_id}_opt_audit")
        os.makedirs(self.log_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.log_dir, f"rank{self.rank}.jsonl")
        self.csv_path = os.path.join(self.log_dir, f"rank{self.rank}.csv")
        self.manifest_path = os.path.join(self.log_dir, "manifest.json")

        self.csv_header = [
            "ts_ms",
            "rank",
            "world_size",
            "run_id",
            "global_step",
            "step_in_optimizer",
            "param_name",
            "group_idx",
            "param_idx",
            "shape",
            "numel",
            "dtype",
            "device",
            "optimizer_name",
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "momentum",
            "grad_norm",
            "grad_sample",
            "param_pre_norm",
            "param_pre_sample",
            "param_post_norm",
            "param_post_sample",
            "actual_delta_norm",
            "actual_delta_sample",
            "m_pre_norm",
            "m_pre_sample",
            "v_pre_norm",
            "v_pre_sample",
            "m_post_norm",
            "m_post_sample",
            "v_post_norm",
            "v_post_sample",
            "manual_m_norm",
            "manual_m_sample",
            "manual_v_norm",
            "manual_v_sample",
            "manual_delta_norm",
            "manual_delta_sample",
            "residual_m_norm",
            "residual_v_norm",
            "residual_delta_norm",
            "step_intensity_ratio",
            "effective_lr_proxy",
            "state_keys_before",
            "state_keys_after",
            "audit_mode",
        ]

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(self.csv_header)

        self._write_manifest()

    def _global_step(self) -> Optional[int]:
        if self._global_step_fn is None:
            return None
        try:
            return _to_int(self._global_step_fn(), default=None)  # type: ignore[arg-type]
        except Exception:
            return None

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "optimizer_name": self.optimizer_name,
            "sample_size": self.sample_size,
            "created_ms": int(time.time() * 1000),
            "schema": self.csv_header,
        }
        with open(self.manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

    def write_records(self, records: Iterable[dict]) -> None:
        records = list(records)
        if not records:
            return

        with self._lock:
            with open(self.jsonl_path, "a") as jsonl_handle:
                for record in records:
                    jsonl_handle.write(json.dumps(record) + "\n")

            with open(self.csv_path, "a", newline="") as csv_handle:
                writer = csv.writer(csv_handle)
                for record in records:
                    writer.writerow([self._csv_value(record.get(field)) for field in self.csv_header])

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        return value


def _snapshot_param(
    *,
    param: torch.nn.Parameter,
    param_name: str,
    group_idx: int,
    param_idx: int,
    optimizer: torch.optim.Optimizer,
    sample_size: int,
) -> Optional[dict]:
    grad = param.grad
    if grad is None:
        return None

    state = optimizer.state.get(param, {})
    state_keys_before = sorted(list(state.keys()))
    param_pre = _safe_clone(param.data)
    grad_clone = _safe_clone(grad)
    m_pre = _safe_clone(state.get("exp_avg"))
    v_pre = _safe_clone(state.get("exp_avg_sq"))
    step_pre = _to_int(state.get("step", 0), default=0)

    return {
        "param": param,
        "param_name": param_name,
        "group_idx": group_idx,
        "param_idx": param_idx,
        "state_keys_before": state_keys_before,
        "param_pre": param_pre,
        "grad": grad_clone,
        "m_pre": m_pre,
        "v_pre": v_pre,
        "step_pre": step_pre,
        "sample_size": sample_size,
        "group_meta": _group_meta(optimizer.param_groups[group_idx]),
    }


def _manual_adamw_update(
    *,
    param_pre: torch.Tensor,
    grad: torch.Tensor,
    m_pre: torch.Tensor,
    v_pre: torch.Tensor,
    lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    step_t: int,
) -> dict:
    beta1, beta2 = betas
    grad = grad.float()
    param_pre = param_pre.float()
    m_pre = m_pre.float()
    v_pre = v_pre.float()

    m_t = beta1 * m_pre + (1.0 - beta1) * grad
    v_t = beta2 * v_pre + (1.0 - beta2) * (grad * grad)

    bias_correction1 = 1.0 - beta1 ** step_t
    bias_correction2 = 1.0 - beta2 ** step_t
    m_hat = m_t / bias_correction1
    v_hat = v_t / bias_correction2
    denom = torch.sqrt(v_hat) + eps
    adam_step = m_hat / denom

    manual_delta = (-lr * adam_step) - (lr * weight_decay * param_pre)
    return {
        "m_t": m_t,
        "v_t": v_t,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "manual_delta": manual_delta,
        "step_intensity_ratio": float((torch.mean(torch.abs(m_hat) / denom)).item()),
        "effective_lr_proxy": float((lr / torch.mean(denom)).item()),
    }


def _finalize_record(pre: dict, optimizer: torch.optim.Optimizer) -> dict:
    param = pre["param"]
    state_after = optimizer.state.get(param, {})
    state_keys_after = sorted(list(state_after.keys()))
    param_post = param.data.detach().clone()
    actual_delta = param_post - pre["param_pre"]

    group_meta = pre["group_meta"]
    lr = _to_float(group_meta.get("lr", 0.0), default=0.0)
    betas = group_meta.get("betas", (0.9, 0.999))
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        betas = (0.9, 0.999)
    betas = (float(betas[0]), float(betas[1]))
    eps = _to_float(group_meta.get("eps", 1e-8), default=1e-8)
    weight_decay = _to_float(group_meta.get("weight_decay", 0.0), default=0.0)
    momentum = group_meta.get("momentum")

    has_adam_state = "exp_avg" in state_after and "exp_avg_sq" in state_after
    if has_adam_state:
        step_t = _to_int(state_after.get("step", pre["step_pre"] + 1), default=pre["step_pre"] + 1)
        m_post = state_after["exp_avg"].detach().clone()
        v_post = state_after["exp_avg_sq"].detach().clone()
        manual = _manual_adamw_update(
            param_pre=pre["param_pre"],
            grad=pre["grad"],
            m_pre=pre["m_pre"] if pre["m_pre"] is not None else torch.zeros_like(pre["param_pre"]),
            v_pre=pre["v_pre"] if pre["v_pre"] is not None else torch.zeros_like(pre["param_pre"]),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            step_t=step_t,
        )
        manual_m = manual["m_t"]
        manual_v = manual["v_t"]
        manual_delta = manual["manual_delta"]
        residual_m_norm = _tensor_norm(m_post - manual_m)
        residual_v_norm = _tensor_norm(v_post - manual_v)
        residual_delta_norm = _tensor_norm(actual_delta - manual_delta)
        step_intensity_ratio = manual["step_intensity_ratio"]
        effective_lr_proxy = manual["effective_lr_proxy"]
        audit_mode = "adamw"
    else:
        step_t = _to_int(state_after.get("step", pre["step_pre"] + 1), default=pre["step_pre"] + 1)
        m_post = state_after.get("momentum_buffer")
        v_post = state_after.get("square_avg")
        manual_m = None
        manual_v = None
        manual_delta = None
        residual_m_norm = None
        residual_v_norm = None
        residual_delta_norm = None
        step_intensity_ratio = None
        effective_lr_proxy = None
        audit_mode = "generic"

    record = {
        "ts_ms": int(time.time() * 1000),
        "rank": _rk(),
        "world_size": _ws(),
        "run_id": os.environ.get("RUN_ID", "default_run"),
        "global_step": _to_int(os.environ.get("LOSSY_GLOBAL_STEP", -1), default=-1),
        "step_in_optimizer": step_t,
        "param_name": pre["param_name"],
        "group_idx": pre["group_idx"],
        "param_idx": pre["param_idx"],
        "shape": list(pre["param_pre"].shape),
        "numel": int(pre["param_pre"].numel()),
        "dtype": str(pre["param_pre"].dtype),
        "device": str(pre["param_pre"].device),
        "optimizer_name": type(optimizer).__name__,
        "lr": lr,
        "betas": list(betas),
        "eps": eps,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "grad_norm": _tensor_norm(pre["grad"]),
        "grad_sample": _tensor_sample(pre["grad"], pre["sample_size"]),
        "param_pre_norm": _tensor_norm(pre["param_pre"]),
        "param_pre_sample": _tensor_sample(pre["param_pre"], pre["sample_size"]),
        "param_post_norm": _tensor_norm(param_post),
        "param_post_sample": _tensor_sample(param_post, pre["sample_size"]),
        "actual_delta_norm": _tensor_norm(actual_delta),
        "actual_delta_sample": _tensor_sample(actual_delta, pre["sample_size"]),
        "m_pre_norm": _tensor_norm(pre["m_pre"]),
        "m_pre_sample": _tensor_sample(pre["m_pre"], pre["sample_size"]),
        "v_pre_norm": _tensor_norm(pre["v_pre"]),
        "v_pre_sample": _tensor_sample(pre["v_pre"], pre["sample_size"]),
        "m_post_norm": _tensor_norm(m_post),
        "m_post_sample": _tensor_sample(m_post, pre["sample_size"]),
        "v_post_norm": _tensor_norm(v_post),
        "v_post_sample": _tensor_sample(v_post, pre["sample_size"]),
        "manual_m_norm": _tensor_norm(manual_m),
        "manual_m_sample": _tensor_sample(manual_m, pre["sample_size"]),
        "manual_v_norm": _tensor_norm(manual_v),
        "manual_v_sample": _tensor_sample(manual_v, pre["sample_size"]),
        "manual_delta_norm": _tensor_norm(manual_delta),
        "manual_delta_sample": _tensor_sample(manual_delta, pre["sample_size"]),
        "residual_m_norm": residual_m_norm,
        "residual_v_norm": residual_v_norm,
        "residual_delta_norm": residual_delta_norm,
        "step_intensity_ratio": step_intensity_ratio,
        "effective_lr_proxy": effective_lr_proxy,
        "state_keys_before": pre["state_keys_before"],
        "state_keys_after": state_keys_after,
        "audit_mode": audit_mode,
    }

    return record


def enable(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_id: str,
    log_root: Optional[str] = None,
    sample_size: int = 16,
    global_step_fn: Optional[Callable[[], int]] = None,
) -> OptimizerAuditLogger:
    logger = OptimizerAuditLogger(
        log_root=log_root,
        run_id=run_id,
        optimizer_name=type(optimizer).__name__,
        sample_size=sample_size,
        global_step_fn=global_step_fn,
    )

    named_params = {id(param): name for name, param in model.named_parameters()}
    original_step = optimizer.step
    original_step_fn = original_step.__func__ if hasattr(original_step, "__func__") else None

    if getattr(original_step, "__optimizer_audit_wrapped__", False):
        return logger

    def step_wrapper(self, *args, **kwargs):
        pre_snapshots = []
        with torch.no_grad():
            for group_idx, group in enumerate(self.param_groups):
                for param_idx, param in enumerate(group.get("params", [])):
                    if not isinstance(param, torch.nn.Parameter):
                        continue
                    snapshot = _snapshot_param(
                        param=param,
                        param_name=named_params.get(id(param), f"param_{group_idx}_{param_idx}"),
                        group_idx=group_idx,
                        param_idx=param_idx,
                        optimizer=self,
                        sample_size=sample_size,
                    )
                    if snapshot is not None:
                        pre_snapshots.append(snapshot)

        if original_step_fn is not None:
            result = original_step_fn(self, *args, **kwargs)
        else:
            result = original_step(*args, **kwargs)

        records = []
        with torch.no_grad():
            for snapshot in pre_snapshots:
                records.append(_finalize_record(snapshot, self))

        logger.write_records(records)
        return result

    step_wrapper.__optimizer_audit_wrapped__ = True
    optimizer.step = types.MethodType(step_wrapper, optimizer)
    return logger