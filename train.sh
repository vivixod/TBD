#!/bin/sh
# sleep 6h
PARTITION=Segmentation

GPU_ID=2
# GPU_ID=4,5,6,7
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=${FREE_PORT} \
        train.py \
        --datapath /vcl/2025zhutaotao/BTCV \
        --logpath trn_btcv_shot8_nbr\
        --benchmark btcv2 \
        --backbone resnet50 \
        --shot 8 \
        --neighbor-prob 1.0 \
        --fold 0 \
        --condition mask \
        --num_query 50 \
        --epochs 100 \
        --lr 1e-4 \
        --bsz 1