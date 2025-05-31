import torch
import math

MAX_PAYLOAD_BYTES = 1450

class LossyNetwork:
    """
    Simulates a lossy network by randomly dropping packets based on a specified loss rate.
    You can inherit from this class to create custom network simulations without changing the training code.
    """
    def __init__(self, args):
        self.loss_rate = float(args.loss_rate)

    def set_seed(self, seed: int):
        self.seed = seed

    def send(self, data: torch.Tensor) -> torch.Tensor:
        data_size = data.numel()
        float_size = data.element_size()
        num_bytes = data_size * float_size
        num_packets = math.ceil(num_bytes / MAX_PAYLOAD_BYTES)
        packets_mask = torch.rand(num_packets) > self.loss_rate
        return packets_mask
    
    def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:

        if packets_mask.all(): # when no packets are lost
            return data
        
        num_packets = len(packets_mask)
        number_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1

        flat = data.flatten()
        indices = torch.arange(num_packets * number_per_packet, device=data.device)
        indices = indices[indices < flat.numel()]
        mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]
        flat[~mask] = 0.0
        return flat.view_as(data)