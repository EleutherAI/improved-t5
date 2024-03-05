#!/bin/bash 
MODEL_PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

echo "Evaluating ${MODEL}"
echo "BBH"
bash eval-bbh.sh ${MODEL_PATH} ${MODEL} ${LM_EVAL} ${EXTRA}
echo "MMLU"
bash eval-mmlu.sh ${MODEL_PATH} ${MODEL} ${LM_EVAL} ${EXTRA}
echo "Held In"
bash eval-held_in.sh ${MODEL_PATH} ${MODEL} ${LM_EVAL} ${EXTRA}
echo "CoT"
bash eval-cot.sh ${MODEL_PATH} ${MODEL} ${LM_EVAL} ${EXTRA}
