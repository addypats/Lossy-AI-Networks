# cd ~/Lossy-AI-Networks
# source 3.10tp-env/bin/activate

export MASTER_ADDR=172.31.26.125   # Node 0 private IP
export MASTER_PORT=29500
export NNODES=2
export NPROC_PER_NODE=4

export NCCL_SOCKET_IFNAME=ens5   # same as above
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  dist_hello.py
