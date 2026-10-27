#!/bin/sh
# 4 数据集联合训练：BTCV2 + AMOS2022_MR + PAPILA + BraTS2019
# episode 模式：每 epoch 固定 n_episodes，先按权重选数据集，再从该数据集随机抽样本
PARTITION=Segmentation
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

GPU_ID=${GPU_ID:-1,2}
EPOCHS=${EPOCHS:-100}
NPROC_PER_NODE=$(printf '%s' "$GPU_ID" | awk -F, '{print NF}')
# GPU_ID=4,5,6,7
# auto: 支持原生 BF16 时加速冻结的 SAM 编码器；可用环境变量改为 fp32 做对照。
SAM_ENCODER_PRECISION=${SAM_ENCODER_PRECISION:-auto}
SUPPORT_SAMPLING=${SUPPORT_SAMPLING:-similarity}
SUPPORT_RETRIEVAL_DIR=${SUPPORT_RETRIEVAL_DIR:-"${SCRIPT_DIR}/cache_support_retrieval"}
SUPPORT_RETRIEVAL_DEVICE=${SUPPORT_RETRIEVAL_DEVICE:-cuda:0}
case "$SUPPORT_SAMPLING" in
    random) NEIGHBOR_PROB=${NEIGHBOR_PROB:-1.0} ;;
    cross-patient-random|similarity) NEIGHBOR_PROB=0.0 ;;
    *) echo "SUPPORT_SAMPLING must be random, cross-patient-random, or similarity" >&2; exit 2 ;;
esac
# finetune=原设置；finetune-frozen-bn=仅固定 BN；frozen=固定 layer3 权重和 BN。
LAYER3_MODE=${LAYER3_MODE:-finetune}
case "$LAYER3_MODE" in
    finetune|finetune-frozen-bn|frozen) ;;
    *) echo "LAYER3_MODE must be finetune, finetune-frozen-bn, or frozen" >&2; exit 2 ;;
esac
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
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-trn_joint4_papila}
DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_PREFIX}_shot4_cb${CLASS_BALANCE_PROB}_pool${SUPPORT_MASK_POOLING}
if [ "$BBOX_SUPPORT_ATTN_MASK:$ICL_SUPPORT_ATTN_MASK" != "0:0" ]; then
    DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_bboxm${BBOX_SUPPORT_ATTN_MASK}_iclm${ICL_SUPPORT_ATTN_MASK}
fi
if [ "$SUPPORT_FUSION" = "quality" ]; then
    DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_fusionquality_qw${FUSION_QUALITY_WEIGHT}_qt${FUSION_TARGET_TEMPERATURE}_fw${FUSION_WARMUP_EPOCHS}_fr${FUSION_RAMP_EPOCHS}
fi
if [ "$SUPPORT_SAMPLING" != "random" ]; then
    DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_support${SUPPORT_SAMPLING}
fi
if [ "$LAYER3_MODE" != "finetune" ]; then
    DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_l3${LAYER3_MODE}
fi
DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME}_ep${EPOCHS}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}

# 子数据集列表：name:datapath（逗号分隔），新增数据集只需在这里追加
JOINT_SUBDATASETS=${JOINT_SUBDATASETS:-"btcv2:/data/2025Zhutaotao/BTCV,amos:/data/2025Zhutaotao/AMOS2022_MR,papila:/data/2025Zhutaotao/PAPILA,brats:/data/2025Zhutaotao/BraTS2019"}

# 采样权重（长度=数据集数，逗号分隔；留空则均匀）
# PAPILA 使用权重 1，与 BTCV / AMOS 相同；BraTS 保留权重 1.5。
JOINT_WEIGHTS=${JOINT_WEIGHTS-"1,1,1,1.5"}

if [ "$SUPPORT_SAMPLING" = "similarity" ]; then
    # One process prepares/reuses both split caches before DDP starts. cuda:0 is
    # the first GPU in GPU_ID (physical GPU 1 by default), not physical GPU 0.
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 prepare_support_retrieval.py \
        --joint-subdatasets "${JOINT_SUBDATASETS}" \
        --output-dir "${SUPPORT_RETRIEVAL_DIR}" \
        --device "${SUPPORT_RETRIEVAL_DEVICE}" --max-shot 4 || exit $?
fi
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port=${FREE_PORT} \
        train.py \
        --benchmark joint \
        --joint-subdatasets "${JOINT_SUBDATASETS}" \
        --joint-weights "${JOINT_WEIGHTS}" \
        --n-episodes 8000 \
        --neighbor-prob "${NEIGHBOR_PROB}" \
        --support-sampling "${SUPPORT_SAMPLING}" \
        --support-retrieval-dir "${SUPPORT_RETRIEVAL_DIR}" \
        --class-balance-prob "${CLASS_BALANCE_PROB}" \
        --datapath /data/2025Zhutaotao/BTCV \
        --logpath "${EXPERIMENT_NAME}" \
        --backbone resnet50 \
        --layer3-mode "${LAYER3_MODE}" \
        --shot 4 \
        --fold 0 \
        --condition mask \
        --num_query 50 \
        --epochs "${EPOCHS}" \
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
