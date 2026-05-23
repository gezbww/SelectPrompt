 #!/usr/bin/env bash
 set -euo pipefail



 DATA=/
 CFG=vit_b16
 SEED_LIST="1"
 REG_E_LIST="0.005"
 LR_LIST="0.0005"

 DATASET=""
 SHOTS=16
 NUM_CLASSES=class_num

 NOISE_TYPES=("sym" "asym")
 NOISE_RATES=("0.125" "0.25" "0.375" "0.5" "0.625" "0.75")

 for NOISE_TYPE in "${NOISE_TYPES[@]}"; do
   for NOISE_RATE in "${NOISE_RATES[@]}"; do
     echo "============================================================"
     echo "Running LightNLPrompt on ${DATASET}"
     echo "Noise type: ${NOISE_TYPE}"
     echo "Noise rate: ${NOISE_RATE}"
     echo "Shots: ${SHOTS}"
     echo "Classes: ${NUM_CLASSES}"
     echo "CFG: ${CFG}"
     echo "============================================================"

     DATA="${DATA}" \
     CFG="${CFG}" \
     SEED_LIST="${SEED_LIST}" \
     REG_E_LIST="${REG_E_LIST}" \
     LR_LIST="${LR_LIST}" \
     bash main.sh \
       "${DATASET}" "${SHOTS}" "${NOISE_RATE}" "${NOISE_TYPE}" "${NUM_CLASSES}"
   done
 done



