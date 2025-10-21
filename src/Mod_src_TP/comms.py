from typing import List
import torch
import math

MAX_PAYLOAD_BYTES = 1450

class LossyNetwork:
    def __init__(self, loss_rate: float = 0.001):
        self.loss_rate = loss_rate

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

        if packets_mask.all():
            return data
        num_packets = len(packets_mask)
        number_per_packet = MAX_PAYLOAD_BYTES // data.element_size() + 1

        flat = data.flatten()
        indices = torch.arange(num_packets * number_per_packet, device=data.device)
        indices = indices[indices < flat.numel()]
        mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]
        flat[~mask] = 0.0
        return flat.view_as(data)
    
        # received_data = data.clone()
        # received_data = received_data.view(-1)
        # for i in range(num_packets):
        #     if not packets_mask[i]:
        #         start = i * number_per_packet
        #         end = min(start + number_per_packet, data.numel())
        #         received_data[start:end] = 0.0
        # received_data = received_data.view(data.shape)
        # return received_data