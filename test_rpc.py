# test_rpc_gloo.py
import os
import torch.distributed.rpc as rpc

def main():
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    print(f"[Rank {rank}/{world_size}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")

    # Use Gloo (CPU) instead of TensorPipe (CUDA)
    rpc_backend_options = rpc.ProcessGroupRpcBackendOptions()
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options,
    )

    if rank == 0:
        print("[Rank 0] Sending RPC to rank1 …")
        fut = rpc.rpc_async("worker1", torch.distributed.rpc.rpc_sync, args=(None,))
        fut.wait()
        print("[Rank 0] RPC to rank1 succeeded.")
    rpc.shutdown()
    print(f"[Rank {rank}] Completed shutdown.")

if __name__ == "__main__":
    main()

