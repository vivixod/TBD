#!/bin/sh
# 4 数据集联合训练：BTCV2 + AMOS2022_MR + ADAM2020 + BraTS2019
# episode 模式：每 epoch 固定 n_episodes，先按权重选数据集，再从该数据集随机抽样本
PARTITION=Segmentation

GPU_ID=${GPU_ID:-1,2}
NPROC_PER_NODE=$(printf '%s' "$GPU_ID" | awk -F, '{print NF}')
# GPU_ID=4,5,6,7
# auto: 支持原生 BF16 时加速冻结的 SAM 编码器；可用环境变量改为 fp32 做对照。
SAM_ENCODER_PRECISION=${SAM_ENCODER_PRECISION:-auto}
# 0=原采样；0.5=一半先选器官再选 query；1=全部类别均衡。
CLASS_BALANCE_PROB=${CLASS_BALANCE_PROB:-0.5}
# Current experiment: all datasets use max pooling. Launch with
# SUPPORT_MASK_POOLING=max to override an older exported adam-only setting.
# nearest and adam-only remain available for reproducing previous ablations.
SUPPORT_MASK_POOLING=${SUPPORT_MASK_POOLING:-max}
case "$SUPPORT_MASK_POOLING" in
    max|nearest|adam-only)
        :
        ;;
    *)
        echo "SUPPORT_MASK_POOLING must be max, nearest, or adam-only" >&2
        exit 2
        ;;
esac
SUPPORT_FUSION=${SUPPORT_FUSION:-mean}
FUSION_QUALITY_WEIGHT=${FUSION_QUALITY_WEIGHT:-0.1}
FUSION_TARGET_TEMPERATURE=${FUSION_TARGET_TEMPERATURE:-0.2}
FUSION_WARMUP_EPOCHS=${FUSION_WARMUP_EPOCHS:-2}
FUSION_RAMP_EPOCHS=${FUSION_RAMP_EPOCHS:-3}
case "$SUPPORT_FUSION" in
    mean|quality) ;;
    *) echo "SUPPORT_FUSION must be mean or quality" >&2; exit 2 ;;
esac
# Independent ablations; both disabled preserves the max-pooling baseline.
BBOX_SUPPORT_ATTN_MASK=${BBOX_SUPPORT_ATTN_MASK:-0}
ICL_SUPPORT_ATTN_MASK=${ICL_SUPPORT_ATTN_MASK:-0}
case "$BBOX_SUPPORT_ATTN_MASK:$ICL_SUPPORT_ATTN_MASK" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "BBOX_SUPPORT_ATTN_MASK and ICL_SUPPORT_ATTN_MASK must be 0 or 1" >&2; exit 2 ;;
esac
DEFAULT_EXPERIMENT_NAME=trn_joint4_shot4_cb${CLASS_BALANCE_PROB}_pool${SUPPORT_MASK_POOLING}
if [ "$BBOX_SUPPORT_ATTN_MASK:$ICL_SUPPORT_ATTN_MASK" != "0:0" ]; then
    DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_bboxm${BBOX_SUPPORT_ATTN_MASK}_iclm${ICL_SUPPORT_ATTN_MASK}
fi
if [ "$SUPPORT_FUSION" = "quality" ]; then
    DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_fusionquality_qw${FUSION_QUALITY_WEIGHT}_qt${FUSION_TARGET_TEMPERATURE}_fw${FUSION_WARMUP_EPOCHS}_fr${FUSION_RAMP_EPOCHS}
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

# 子数据集列表：name:datapath（逗号分隔），新增数据集只需在这里追加
JOINT_SUBDATASETS="btcv2:/data/2025Zhutaotao/BTCV,amos:/data/2025Zhutaotao/AMOS2022_MR,adam:/data/2025Zhutaotao/ADAM2020,brats:/data/2025Zhutaotao/BraTS2019"

# 采样权重（长度=数据集数，逗号分隔；留空则均匀）
# 非均匀：adam 小体量难任务高权重、brats 次之（学 MASS 数据集级加权思路）
JOINT_WEIGHTS="1,1,2,1.5"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node="${NPROC_PER_NODE}" \
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
        --support-mask-pooling "${SUPPORT_MASK_POOLING}" \
        --bbox-support-attn-mask "${BBOX_SUPPORT_ATTN_MASK}" \
        --icl-support-attn-mask "${ICL_SUPPORT_ATTN_MASK}" \
        --support-fusion "${SUPPORT_FUSION}" \
        --fusion-quality-weight "${FUSION_QUALITY_WEIGHT}" \
        --fusion-target-temperature "${FUSION_TARGET_TEMPERATURE}" \
        --fusion-warmup-epochs "${FUSION_WARMUP_EPOCHS}" \
        --fusion-ramp-epochs "${FUSION_RAMP_EPOCHS}" \
        --lr 1e-4 \
        --bsz 1
