#!/bin/sh
# 联合训练脚本：BTCV2 + AMOS2022 MR
# 用法：bash train_joint.sh
PARTITION=Segmentation

GPU_ID=6
# GPU_ID=4,5,6,7
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

# 子数据集列表：name:datapath（逗号分隔），新增数据集只需在这里追加
JOINT_SUBDATASETS="btcv2:/vcl/2025zhutaotao/BTCV,amos:/vcl/2025zhutaotao/AMOS2022_MR"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=${FREE_PORT} \
        train.py \
        --benchmark joint \
        --joint-subdatasets "${JOINT_SUBDATASETS}" \
        --datapath /vcl/2025zhutaotao/BTCV \
        --logpath trn_joint_btcv2_amos_shot4 \
        --backbone resnet50 \
        --shot 4 \
        --fold 0 \
        --condition mask \
        --num_query 50 \
        --epochs 300 \
        --lr 1e-4 \
        --bsz 1