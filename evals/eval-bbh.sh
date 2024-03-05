#!/bin/bash 
MODEL_PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

for TASK in bbh_zeroshot bbh_fewshot
do
    ${LM_EVAL} \
        --model hf \
        --model_args "pretrained=${MODEL_PATH}${MODEL}${EXTRA}" \
        --tasks ${TASK} \
        --batch_size 8 \
        --output "output/${TASK}/${MODEL}" \
        --log_samples
done
