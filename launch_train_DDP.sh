#!/bin/bash
# 省略--master_addr --master_port 将自动设置为 --nnodes 1 --node_rank 0 --master_addr='127.0.0.1' --master_port='23333'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nnodes=1 --node_rank=0 --nproc_per_node=4  --use_env  train_DDP_launch.py