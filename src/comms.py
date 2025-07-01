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


class GillbertElliotLossyNetwork(LossyNetwork):
    """
    Simulates a Gilbert-Elliott lossy network.
    """
    def __init__(self, p_gb, p_bg, good_loss_rate=0.0, bad_loss_rate=1.0, args=None):
        super().__init__(args)
        self.p_gb = p_gb
        self.p_bg = p_bg
        self.good_loss_rate = good_loss_rate
        self.bad_loss_rate = bad_loss_rate
        if self.good_loss_rate > self.bad_loss_rate:
            raise ValueError("Good loss rate must be less than or equal to bad loss rate.")
        self.state = 'good'

    def take_step(self, n_steps=1):
        transition_probabilities = torch.rand(n_steps)
        if self.state == 'good':
            if torch.rand(1).item() < self.p_gb:
                self.state = 'bad'
        else:
            if torch.rand(1).item() < self.p_bg:
                self.state = 'good'
        return self.state
            
    def send(self, data: torch.Tensor) -> torch.Tensor:
        self.take_step()
        num_packets = get_num_packets(data)
        if self.state == 'good':
            packets_mask = torch.rand(num_packets) > self.good_loss_rate
        else:
            packets_mask = torch.rand(num_packets) > self.bad_loss_rate
        return packets_mask


    # def send_alternative(self, data:torch.Tensor):
    #     num_packets = get_num_packets(data)
    #     step_per_packet = [self.take_step() for _ in range(num_packets)]
    #     packets_mask = torch.tensor([
    #         torch.rand(1).item() > (self.good_loss_rate if step == 'good' else self.bad_loss_rate)
    #         for step in step_per_packet
    #     ], device=data.device)
    #     return packets_mask