# lossy_grad_hooks.py

from typing import Callable
import torch
from torch import nn

from comms import LossyNetwork


def make_bernoulli_grad_lossy(lossy: LossyNetwork) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Adapt your LossyNetwork (Bernoulli) to operate on a single gradient tensor.

    Conceptually:
      - Treat grad as the payload.
      - Call lossy.send(...) to pick which packets survive.
      - Call lossy.receive(...) to zero out the dropped packets.
    """
    def _fn(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        if not grad.is_floating_point() or grad.numel() == 0:
            return grad

        # Work on a detached clone to avoid in-place weirdness on autograd's view
        with torch.no_grad():
            g = grad.detach().clone()
            pkt_mask = lossy.send(g)          # CPU mask
            g = lossy.receive(g, pkt_mask)    # handles device move internally now
        return g

    return _fn


def attach_lossy_grad_hooks(
    model: nn.Module,
    lossy: LossyNetwork,
    *,
    include_bias: bool = False,
    verbose: bool = True,
) -> None:
    """
    Attach gradient hooks to parameters of `model` so that every gradient
    passes through a Bernoulli-style packet loss model BEFORE FSDP sees it.

    Call this on the *FSDP-wrapped* model (so we'll do it from a TrainerCallback).
    """
    grad_fn = make_bernoulli_grad_lossy(lossy)

    num_hooked = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (not include_bias) and name.endswith(".bias"):
            continue

        def _make_hook(param_name: str):
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                # Optional: you can add debug prints here if needed
                # print(f"[grad_hooks] applying lossy grad to {param_name}, norm={grad.norm().item():.4f}")
                return grad_fn(grad)
            return _hook

        p.register_hook(_make_hook(name))
        num_hooked += 1

    if verbose:
        print(f"[grad_hooks] Attached lossy gradient hooks to {num_hooked} parameters (include_bias={include_bias}).")
