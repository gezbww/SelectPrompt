#!/bin/bash
set -euo pipefail


DATA=${DATA:-/path/to/datasets}
TRAINER=${TRAINER:-}
DATASET=$1
CFG=${CFG:-vit_b16}
SHOTS=$2
RATE=$3
TYPE=$4
CLASS=$5

REG_E_LIST=${REG_E_LIST:-"0.001"}
SEED_LIST=${SEED_LIST:-"1 2 3"}
LR_LIST=${LR_LIST:-"0.0005"}

for REG_E in ${REG_E_LIST}; do
  for LR in ${LR_LIST}; do
    for SEED in ${SEED_LIST}; do
      DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${TYPE}_${RATE}/lr${LR}/seed${SEED}_regE${REG_E}
      echo "Run this job and save the output to ${DIR}"
      python train.py \
        --root "${DATA}" \
        --seed "${SEED}" \
        --trainer "${TRAINER}" \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --output-dir "${DIR}" \
        DATASET.NUM_SHOTS "${SHOTS}" \
        DATASET.NOISE_RATE "${RATE}" \
        DATASET.NOISE_TYPE "${TYPE}" \
        DATASET.num_class "${CLASS}" \
        DATASET.REG_E "${REG_E}" \
        OPTIM.LR "${LR}"
    done
  done
done

