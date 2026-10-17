#!/bin/sh
# BTCV + AMOS MR + PAPILA + BraTS. Reuse the existing training options.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_ROOT=$(dirname -- "$SCRIPT_DIR")
export JOINT_SUBDATASETS=${JOINT_SUBDATASETS:-"btcv2:${DATA_ROOT}/BTCV,amos:${DATA_ROOT}/AMOS2022_MR,papila:${DATA_ROOT}/PAPILA,brats:${DATA_ROOT}/BraTS2019"}
# PAPILA starts with weight 1; the ADAM-specific weight 2 is not carried over.
export JOINT_WEIGHTS=${JOINT_WEIGHTS-"1,1,1,1.5"}
export EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-trn_joint4_papila}
cd "$SCRIPT_DIR" || exit 1
exec sh "$SCRIPT_DIR/train_joint.sh"
