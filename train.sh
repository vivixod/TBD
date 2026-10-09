#!/bin/sh
# sleep 6h
PARTITION=Segmentation

GPU_ID=1,2
# GPU_ID=4,5,6,7
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

# 在 CUDA_VISIBLE_DEVICES 行之前添加
export NCCL_P2P_DISABLE=1      # 禁用 GPU 间 P2P 通信，防止 NCCL 扫描并尝试连接有问题的 0 号卡[reference:0]
export NCCL_IB_DISABLE=1       # 禁用 InfiniBand，强制使用 Socket 通信，减少对 NVML 的依赖[reference:1]
export NCCL_IGNORE_DISABLED_P2P=1  # 忽略 P2P 禁用时的警告信息[reference:2]
export NCCL_CUMEM_ENABLE=0     # 禁用新的 CUDA 内存管理 API，回退到更稳定的传统路径[reference:3]

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=${FREE_PORT} \
        train.py \
        --datapath /vcl/2025zhutaotao/BTCV \
        --logpath trn_btcv_dualpath_shot4\
        --benchmark btcv2 \
        --backbone resnet50 \
        --shot 4 \
        --fold 0 \
        --condition mask \
        --num_query 50 \
        --epochs 300 \
        --lr 1e-4 \
        --bsz 1