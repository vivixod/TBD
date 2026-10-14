#!/bin/sh
# 4 数据集联合训练：BTCV2 + AMOS2022_MR + ADAM2020 + BraTS2019
# episode 模式：每 epoch 固定 n_episodes，先按权重选数据集，再从该数据集随机抽样本
PARTITION=Segmentation

GPU_ID=0,1,2
# GPU_ID=4,5,6,7
# auto: 支持原生 BF16 时加速冻结的 SAM 编码器；可用环境变量改为 fp32 做对照。
SAM_ENCODER_PRECISION=${SAM_ENCODER_PRECISION:-auto}
# 0=原采样；0.5=一半先选器官再选 query；1=全部类别均衡。
CLASS_BALANCE_PROB=${CLASS_BALANCE_PROB:-0.5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-trn_joint4_shot4_cb${CLASS_BALANCE_PROB}}
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

# 子数据集列表：name:datapath（逗号分隔），新增数据集只需在这里追加
JOINT_SUBDATASETS="btcv2:/data/2025Zhutaotao/BTCV,amos:/data/2025Zhutaotao/AMOS2022_MR,adam:/data/2025Zhutaotao/ADAM2020,brats:/data/2025Zhutaotao/BraTS2019"

# 采样权重（长度=数据集数，逗号分隔；留空则均匀）
# 非均匀：adam 小体量难任务高权重、brats 次之（学 MASS 数据集级加权思路）
JOINT_WEIGHTS="1,1,2,1.5"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node=3 \
        --master_port=${FREE_PORT} \
        train.py \
        --benchmark joint \
        --joint-subdatasets "${JOINT_SUBDATASETS}" \
        --joint-weights "${JOINT_WEIGHTS}" \
        --n-episodes 8000 \
        --neighbor-prob 1.0 \
        --class-balance-prob "${CLASS_BALANCE_PROB}" \
        --datapath /data/2025Zhutaotao/BTCV \
        --logpath "${EXPERIMENT_NAME}" \
        --backbone resnet50 \
        --shot 4 \
        --fold 0 \
        --condition mask \
        --num_query 50 \
        --epochs 300 \
        --sam-encoder-precision "${SAM_ENCODER_PRECISION}" \
        --lr 1e-4 \
        --bsz 1
