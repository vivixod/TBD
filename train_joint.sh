#!/bin/sh
# 4 数据集联合训练：BTCV2 + AMOS2022_MR + ADAM2020 + BraTS2019
# episode 模式：每 epoch 固定 n_episodes，先按权重选数据集，再从该数据集随机抽样本
PARTITION=Segmentation

GPU_ID=0,1,6
# GPU_ID=4,5,6,7
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

# 子数据集列表：name:datapath（逗号分隔），新增数据集只需在这里追加
JOINT_SUBDATASETS="btcv2:/vcl/2025zhutaotao/BTCV,amos:/vcl/2025zhutaotao/AMOS2022_MR,adam:/vcl/2025zhutaotao/ADAM2020,brats:/vcl/2025zhutaotao/BraTS2019"

# 采样权重（长度=数据集数，逗号分隔；留空则均匀）
JOINT_WEIGHTS="1,1,1,1"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node=3 \
        --master_port=${FREE_PORT} \
        train.py \
        --benchmark joint \
        --joint-subdatasets "${JOINT_SUBDATASETS}" \
        --joint-weights "${JOINT_WEIGHTS}" \
        --n-episodes 4000 \
        --datapath /vcl/2025zhutaotao/BTCV \
        --logpath trn_joint4_shot8 \
        --backbone resnet50 \
        --shot 8 \
        --fold 0 \
        --condition mask \
        --num_query 50 \
        --epochs 300 \
        --lr 1e-4 \
        --bsz 1
