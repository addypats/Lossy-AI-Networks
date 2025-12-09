import os
import socket
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)

    print(f"[hostname={hostname}] rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)

    dist.barrier()
    if rank == 0:
        print("All ranks reached barrier. Multi-node works!", flush=True)

if __name__ == "__main__":
    main()

